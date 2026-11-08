# -*- coding: utf-8 -*-
"""
设计4 v2：HGNN vs DeepSets / SetTransformer（向量化 + 类别权重 + 折内集成）
=============================================================================
在 v1 基础上的三点改进（均为标准且公平的做法，四臂同等享受）：
  1) 图前向向量化（整 batch 一次聚合），使多种子集成可行；
  2) 类别加权交叉熵（权重 = N/(3·n_c)），缓解 54/89/110 不均衡，提升 BalAcc / macro-F1；
  3) 折内集成：每折训练 ENS=2 个不同随机种子模型，softmax 平均后再评估。
协议：外层 StratifiedKFold(5) × seeds=[42,123,456]（15 次评估）；每次用训练折内 20% 早停选择。
数据：干净口径 hg_clean_run/exp_4dchain_v2_dataset.pkl
输出：hg_event_run/_hgnn_vs_set_v2.json
"""
import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, recall_score

BASE = os.environ.get('EXPERIMENT_BASE', os.getcwd())
OUT = os.environ.get('EVENT_OUT', f'{BASE}/hg_event_run')
DS = os.environ.get('DS4_PATH', f'{BASE}/hg_clean_run/exp_4dchain_v2_dataset.pkl')
SEEDS = [42, 123, 456]
NAMES = ['S-only(MLP)', 'DeepSets(实体集合)', 'SetTransformer(实体+天)', 'HGNN(四维超图)']
# 超参与输出可由环境变量覆盖（用于 F：主模型稳定化）
D = int(os.environ.get('HG_D', 32))
EPOCHS = int(os.environ.get('HG_EP', 120))
LR = float(os.environ.get('HG_LR', 5e-3))
WD = float(os.environ.get('HG_WD', 1e-3))
DROP = float(os.environ.get('HG_DROP', 0.3))
ENS = int(os.environ.get('HG_ENS', 2))          # 折内集成成员数
OUTFILE = os.environ.get('HG_OUT', '_hgnn_vs_set_v2.json')
BATCH = 64
MAXD = 5
DIM2I = {'hist': 0, 'evidence': 1, 'treat': 2, 'severity': 3}
P2D = {'sym': 'hist', 'diag': 'hist', 'lab': 'evidence', 'img': 'evidence',
       'med': 'treat', 'score': 'severity', 'sev': 'severity'}


def pack(ds):
    ent2id = ds['ent2id']
    rec = {}
    for pos, pid in enumerate(ds['pids']):
        first_d, he = {}, {}
        for d, es in ds['day_ent_data'].get(pid, []):
            if d >= MAXD:
                continue
            for e in es:
                if e not in ent2id:
                    continue
                i = ent2id[e]
                first_d.setdefault(i, d)
                he.setdefault((d, DIM2I[P2D[e[0]]]), set()).add(i)
        tok_e = np.array(sorted(first_d), dtype=np.int64)
        if tok_e.size == 0:
            tok_e = np.array([0], dtype=np.int64); first_d = {0: 0}
            he[(0, 0)] = {0}                       # 兜底：至少一条超边，避免空图
        tok_d = np.array([first_d[int(i)] for i in tok_e], dtype=np.int64)
        loc = {int(i): k for k, i in enumerate(tok_e)}
        he_ptr, he_ent, he_dim, he_day = [0], [], [], []
        for (d, dm), s in sorted(he.items()):
            for i in s:
                he_ent.append(loc[i])
            he_ptr.append(len(he_ent)); he_dim.append(dm); he_day.append(d)
        rec[pos] = dict(uent=tok_e.copy(), tok_e=tok_e, tok_d=tok_d,
                        he_ptr=np.array(he_ptr, np.int64), he_ent=np.array(he_ent, np.int64),
                        he_dim=np.array(he_dim, np.int64), he_day=np.array(he_day, np.int64))
    return rec, len(ent2id)


class SEnc(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, D), nn.ReLU(), nn.Dropout(DROP), nn.Linear(D, D), nn.ReLU())

    def forward(self, s):
        return self.net(s)


class SOnly(nn.Module):
    def __init__(self, V, n_s):
        super().__init__(); self.s = SEnc(n_s); self.cls = nn.Linear(D, 3)

    def forward(self, b):
        return self.cls(self.s(b['s']))


class DeepSetsPerm(nn.Module):
    def __init__(self, V, n_s):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.phi = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
        self.rho = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(), nn.Dropout(DROP))
        self.s = SEnc(n_s); self.cls = nn.Linear(D, 3)

    def forward(self, b):
        h = self.phi(self.emb(b['uent_pad'])) * b['uent_mask'].unsqueeze(-1)
        z = h.sum(1)
        return self.cls(self.rho(torch.cat([z, self.s(b['s'])], -1)))


class SetTransformerPerm(nn.Module):
    def __init__(self, V, n_s):
        super().__init__()
        self.emb = nn.Embedding(V, D); self.day = nn.Embedding(MAXD, D)
        layer = nn.TransformerEncoderLayer(D, 4, 2 * D, dropout=DROP, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 1)
        self.rho = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(), nn.Dropout(DROP))
        self.s = SEnc(n_s); self.cls = nn.Linear(D, 3)

    def forward(self, b):
        x = self.emb(b['tok_e_pad']) + self.day(b['tok_d_pad'])
        x = self.enc(x, src_key_padding_mask=(b['tok_mask'] < 0.5))
        z = (x * b['tok_mask'].unsqueeze(-1)).sum(1) / b['tok_mask'].sum(1, keepdim=True).clamp_min(1)
        return self.cls(self.rho(torch.cat([z, self.s(b['s'])], -1)))


class HGNNPerm(nn.Module):
    """整 batch 一次完成 实体 → (day,dim)超边 → 实体 聚合"""
    def __init__(self, V, n_s):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.dim = nn.Embedding(4, D)
        self.daye = nn.Embedding(MAXD, D)
        self.e2he = nn.Linear(D, 1)
        self.q = nn.Linear(D, D); self.k = nn.Linear(D, D)
        self.w_self = nn.Linear(D, D)
        self.rho = nn.Sequential(nn.Linear(3 * D, D), nn.ReLU(), nn.Dropout(DROP))
        self.s = SEnc(n_s); self.cls = nn.Linear(D, 3)

    def forward(self, b):
        x = self.emb(b['ent'])
        ptr, hE = b['he_ptr'], b['he_ent']
        gidx = torch.repeat_interleave(torch.arange(len(ptr) - 1), ptr[1:] - ptr[:-1])
        he_x = x[hE] + self.dim(b['he_dim'][gidx]) + self.daye(b['he_day'][gidx])
        a = torch.sigmoid(self.e2he(he_x)).squeeze(-1)
        agg = torch.zeros(len(ptr) - 1, D).index_add(0, gidx, a.unsqueeze(-1) * he_x)
        q = self.q(x[hE]); k = self.k(agg[gidx])
        score = (q * k).sum(-1) / D ** 0.5
        Ne = x.size(0)
        mx = torch.full((Ne,), -1e9).index_reduce_(0, hE, score, 'amax', include_self=True)
        ex = torch.exp(score - mx[hE])
        den = torch.zeros(Ne).index_add_(0, hE, ex)
        beta = ex / den[hE].clamp_min(1e-9)
        upd = torch.zeros_like(x).index_add(0, hE, beta.unsqueeze(-1) * agg[gidx])
        x2 = F.relu(self.w_self(x) + upd)
        B = b['s'].size(0)
        cnt = torch.zeros(B).index_add_(0, b['pat'], torch.ones(Ne))
        z = (torch.zeros(B, D).index_add_(0, b['pat'], x2)) / cnt.unsqueeze(-1).clamp_min(1)
        nh = torch.zeros(B).index_add_(0, b['he_pat'], torch.ones(len(ptr) - 1))
        z_he = (torch.zeros(B, D).index_add_(0, b['he_pat'], agg)) / nh.unsqueeze(-1).clamp_min(1)
        return self.cls(self.rho(torch.cat([z, z_he, self.s(b['s'])], -1)))


def build_net(name, V, n_s):
    return {'S-only(MLP)': SOnly, 'DeepSets(实体集合)': DeepSetsPerm,
            'SetTransformer(实体+天)': SetTransformerPerm, 'HGNN(四维超图)': HGNNPerm}[name](V, n_s)


def make_batch(rec, idx, St):
    def pad(seqs):
        L = max(len(x) for x in seqs)
        out = np.zeros((len(seqs), L), np.int64); msk = np.zeros((len(seqs), L), np.float32)
        for i, x in enumerate(seqs):
            out[i, :len(x)] = x; msk[i, :len(x)] = 1.0
        return torch.from_numpy(out), torch.from_numpy(msk)

    ue, um = pad([rec[i]['uent'] for i in idx])
    te, tm = pad([rec[i]['tok_e'] for i in idx])
    td, _ = pad([rec[i]['tok_d'] for i in idx])
    ents, he_ent, he_ptr, he_dim, he_day, pat, he_pat = [], [], [0], [], [], [], []
    e_off = h_off = 0
    for bi, i in enumerate(idx):
        r = rec[i]; n = len(r['tok_e'])
        ents.append(r['tok_e'])
        he_ent.append(r['he_ent'] + e_off)
        he_ptr.extend((r['he_ptr'][1:] + h_off).tolist())
        he_dim.append(r['he_dim']); he_day.append(r['he_day'])
        pat.append(np.full(n, bi))
        he_pat.append(np.full(len(r['he_ptr']) - 1, bi))
        e_off += n; h_off += int(r['he_ptr'][-1])
    return dict(uent_pad=ue, uent_mask=um, tok_e_pad=te, tok_d_pad=td, tok_mask=tm, s=St[idx],
                ent=torch.from_numpy(np.concatenate(ents)),
                he_ent=torch.from_numpy(np.concatenate(he_ent)),
                he_ptr=torch.tensor(he_ptr, dtype=torch.long),
                he_dim=torch.from_numpy(np.concatenate(he_dim)),
                he_day=torch.from_numpy(np.concatenate(he_day)),
                pat=torch.from_numpy(np.concatenate(pat)),
                he_pat=torch.from_numpy(np.concatenate(he_pat)))


def safe_auc(y, prob):
    try:
        return roc_auc_score(y, prob, multi_class='ovr', average='macro', labels=[0, 1, 2])
    except ValueError:
        return 0.5


def class_w(y):
    c = np.bincount(y, minlength=3).astype(float)
    return torch.tensor(len(y) / (3 * np.maximum(c, 1)), dtype=torch.float32)


def infer(net, rec, idx, St):
    with torch.no_grad():
        return net(make_batch(rec, idx, St))


def train_one(name, rec, y, St, tr_in, va, seed, V):
    torch.manual_seed(seed)
    net = build_net(name, V, St.size(1))
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    w = class_w(y[tr_in])
    best, best_va = None, -1.0
    for ep in range(EPOCHS):
        net.train()
        perm = np.random.RandomState(seed * 1000 + ep).permutation(tr_in)
        for b0 in range(0, len(perm), BATCH):
            bidx = perm[b0:b0 + BATCH]
            opt.zero_grad()
            F.cross_entropy(net(make_batch(rec, bidx, St)), torch.tensor(y[bidx]), weight=w).backward()
            opt.step()
        net.eval()
        a = safe_auc(y[va], torch.softmax(infer(net, rec, va, St), -1).numpy())
        if a > best_va:
            best_va = a; best = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best); net.eval()
    return net


def main():
    ds = pickle.load(open(DS, 'rb'))
    y = np.array([m['label3'] for m in ds['meta']])
    S = ds['X_struct'].astype(np.float64)
    rec, V = pack(ds)
    res = {n: dict(auc=[], bal_acc=[], macro_f1=[], recall=[]) for n in NAMES}

    for seed in SEEDS:
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for fi, (tr_idx, te_idx) in enumerate(skf.split(S, y)):
            mu = np.nanmean(S[tr_idx], 0); mu[np.isnan(mu)] = 0.0
            sd = np.nanstd(S[tr_idx], 0); sd[np.isnan(sd) | (sd < 1e-6)] = 1.0
            St = torch.tensor((np.where(np.isnan(S), mu, S) - mu) / sd, dtype=torch.float32)
            for name in NAMES:
                P = np.zeros((len(te_idx), 3))
                for m in range(ENS):
                    sd_m = seed + 7 * m
                    tr_in, va = train_test_split(tr_idx, test_size=0.2, stratify=y[tr_idx], random_state=sd_m)
                    net = train_one(name, rec, y, St, tr_in, va, sd_m, V)
                    P += torch.softmax(infer(net, rec, te_idx, St), -1).numpy()
                P /= ENS
                pred = P.argmax(1)
                res[name]['auc'].append(safe_auc(y[te_idx], P))
                res[name]['bal_acc'].append(balanced_accuracy_score(y[te_idx], pred))
                res[name]['macro_f1'].append(f1_score(y[te_idx], pred, average='macro'))
                res[name]['recall'].append(recall_score(y[te_idx], pred, average=None, labels=[0, 1, 2]))
            print('[seed=%d fold=%d] done' % (seed, fi), flush=True)

    out = {}
    for n in NAMES:
        r = res[n]
        out[n] = dict(auc=round(float(np.mean(r['auc'])), 4), auc_sd=round(float(np.std(r['auc'])), 4),
                      bal_acc=round(float(np.mean(r['bal_acc'])), 4),
                      macro_f1=round(float(np.mean(r['macro_f1'])), 4),
                      recall=[round(float(np.mean([x[i] for x in r['recall']])), 4) for i in range(3)])
        print('%-26s AUC=%.4f(±%.3f) BalAcc=%.4f mf1=%.4f' %
              (n, out[n]['auc'], out[n]['auc_sd'], out[n]['bal_acc'], out[n]['macro_f1']), flush=True)
    # 配对检验
    print('\n== 配对检验（vs HGNN）==')
    from scipy import stats
    ref = np.array(res['HGNN(四维超图)']['auc'])
    for n in NAMES:
        if n.startswith('HGNN'):
            continue
        d = ref - np.array(res[n]['auc'])
        print('%-26s ΔAUC=%+.4f 95%%CI=[%+.4f,%+.4f] p=%.4f 胜/负=%d/%d' %
              (n, d.mean(), d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d)),
               d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(len(d)), stats.wilcoxon(ref, res[n]['auc']).pvalue,
               int((d > 0).sum()), int((d < 0).sum())), flush=True)
    out['_config'] = dict(D=D, EPOCHS=EPOCHS, LR=LR, WD=WD, DROP=DROP, ENS=ENS)
    json.dump(out, open(f'{OUT}/{OUTFILE}', 'w'), ensure_ascii=False, indent=2)
    print('saved', f'{OUT}/{OUTFILE}')


if __name__ == '__main__':
    main()
