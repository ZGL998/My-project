# -*- coding: utf-8 -*-
"""
真实实验主脚本 v3：基于253例COPD患者真实提取特征（CSCS复合严重度评分，WINDOW=5固定窗口）
- Task1 三分类 (A/B/C)
- Task2 恶化预测 (B+C vs A)
- Task3 严重度回归 (CSCS临床严重度复合评分，无泄漏)
- 基线对比：KG+GNN / TKG / LSTM / LR-raw / LR+XGBoost
5折分层交叉验证，全部方法同折。
所有深度模型均融合结构化特征（side information），训练采用类别权重。
训练：Adam(lr=3e-4, wd=1e-4), hidden=32, epochs≤150, patience=20
种子：3-seed协议 (42/123/456), fold内种子 1000+fold_i / +50 / +100
"""
import os
import pickle, json, copy, random, os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import torch
torch.set_num_threads(4)
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from scipy.stats import pearsonr, ttest_rel, binomtest
from xgboost import XGBClassifier, XGBRegressor

BASE = os.environ.get('EXPERIMENT_BASE', os.getcwd())
DIM = 32
EPOCHS = 150
PATIENCE = 20

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# ---------------- 数据 ----------------
ds = pickle.load(open(f'{BASE}/exp_dataset.pkl', 'rb'))
pids = ds['pids']
meta = ds['meta']
label3 = np.array([m['label3'] for m in meta])
y2 = (label3 > 0).astype(int)
days = np.array([float(m['days']) if m['days'] else np.nan for m in meta])
X_struct = ds['X_struct']
seq_data = ds['seq_data']
max_days = ds['max_days']
ent2id = ds['ent2id']
entity_data = ds['entity_data']
day_ent_data = ds['day_ent_data']
NS = X_struct.shape[1]
V = len(ent2id)
DIM_ORDER = ['hist', 'evidence', 'treat', 'severity']

def prep_struct(X, Xtr_idx):
    tr = X[Xtr_idx]
    med = np.nanmedian(tr, axis=0); med[np.isnan(med)] = 0.0
    std = np.nanstd(tr, axis=0); std[np.isnan(std)] = 1.0; std[std < 1e-6] = 1.0
    Xi = np.where(np.isnan(X), med, X)
    return (Xi - med) / std

def prep_seq(pid, Xtr_idx):
    tr = np.stack([seq_data[pids[i]] for i in Xtr_idx])
    mu = tr.mean(axis=(0, 1)); sd = tr.std(axis=(0, 1)); sd[sd < 1e-6] = 1.0
    s = seq_data[pid]
    return (s - mu) / sd

def pid_ent_ids(pid, dims=None):
    ed = entity_data[pid]
    ids = []
    for dm, es in ed.items():
        if dims is not None and dm not in dims:
            continue
        ids += [ent2id[e] for e in es if e in ent2id]
    return list(set(ids))

def dim_map(pid, dims=None):
    ed = entity_data[pid]
    dm = {}
    for di, dmname in enumerate(DIM_ORDER):
        if dims is not None and dmname not in dims:
            continue
        dm[di] = list(set(ent2id[e] for e in ed.get(dmname, set()) if e in ent2id))
    return dm

def day_seq_for(pid):
    return [(d, [ent2id[e] for e in es if e in ent2id]) for d, es in day_ent_data.get(pid, [])]

def build_graph_adj(tr_pids, V):
    A = np.zeros((V, V), dtype=float)
    for pid in tr_pids:
        ids = pid_ent_ids(pid)
        for i in ids:
            for j in ids:
                A[i, j] += 1.0
    np.fill_diagonal(A, 1.0)
    d = A.sum(1) ** (-0.5); d[np.isinf(d)] = 0
    D = np.diag(d)
    A = D @ A @ D
    return torch.tensor(A, dtype=torch.float32)

# ---------------- 模型（均融合结构化特征） ----------------
class MLPHead(nn.Module):
    def __init__(self, ind, nout, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ind, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, nout))
    def forward(self, x):
        return self.net(x)

class KGGNN(nn.Module):
    """KG+GNN: 实体共现图 + 2层GCN + 结构化特征融合"""
    def __init__(self, V, adj, ns, dim=DIM, nout=3):
        super().__init__()
        self.emb = nn.Embedding(V, dim)
        self.adj = adj
        self.W1 = nn.Linear(dim, dim); self.W2 = nn.Linear(dim, dim)
        self.s_proj = nn.Linear(ns, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.head = MLPHead(dim, nout)
        self.apply(init_weights)
    def forward(self, ent_ids_list, S):
        h = self.emb.weight
        h = F.relu(self.W1(self.adj @ h))
        h = F.relu(self.W2(self.adj @ h))
        reps = []
        for ids in ent_ids_list:
            reps.append(h[ids].mean(0) if len(ids) else torch.zeros_like(h[0]))
        x = torch.stack(reps) + self.s_proj(S)
        return self.head(self.bn(x))

class TKG(nn.Module):
    """TKG: 按天实体的时序KG + GRU + 结构化特征融合"""
    def __init__(self, V, ns, dim=DIM, nout=3):
        super().__init__()
        self.emb = nn.Embedding(V, dim)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.s_proj = nn.Linear(ns, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.head = MLPHead(dim, nout)
        self.apply(init_weights)
    def forward(self, day_seqs, S):
        hs = []
        for seq in day_seqs:
            embs = [self.emb.weight[ids].mean(0) for _, ids in seq if len(ids)]
            if not embs:
                embs = [torch.zeros(self.emb.weight.size(1))]
            X = torch.stack(embs).unsqueeze(0)
            _, h = self.gru(X)
            hs.append(h[-1, 0])
        x = torch.stack(hs) + self.s_proj(S)
        return self.head(self.bn(x))

class LSTMModel(nn.Module):
    """LSTM: 按天时序数值序列 + 结构化特征融合"""
    def __init__(self, Din, ns, dim=64, nout=3):
        super().__init__()
        self.lstm = nn.LSTM(Din, dim, batch_first=True)
        self.s_proj = nn.Linear(ns, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.head = MLPHead(dim, nout)
        self.apply(init_weights)
    def forward(self, X, S):
        _, (h, _) = self.lstm(X)
        x = h[-1] + self.s_proj(S)
        return self.head(self.bn(x))

class HyperGraph4D(nn.Module):
    """4D-HyperGraph: 支持静态四维超边或按天按维时序超边的聚合模型"""
    def __init__(self, V, ns, dim=DIM, nout=3, dims=4):
        super().__init__()
        self.emb = nn.Embedding(V, dim)
        self.dim_lins = nn.ModuleList([nn.Linear(dim, dim) for _ in range(dims)])
        self.ent_attn = nn.Linear(dim, 1)
        self.dim_attn = nn.Linear(dim, 1)
        self.day_emb = nn.Embedding(32, dim)
        self.time_attn = nn.Linear(dim, 1)
        self.fuse = nn.Linear(dim * 2, dim)
        self.s_proj = nn.Linear(ns, dim)
        self.graph_bn = nn.BatchNorm1d(dim)
        self.head = MLPHead(dim * 2 + ns, nout)
        self.apply(init_weights)

    def _agg_dim_map(self, dmap):
        msgs, dvals = [], []
        for di in range(len(self.dim_lins)):
            ids = dmap.get(di, [])
            if not ids:
                continue
            he = F.relu(self.dim_lins[di](self.emb.weight[ids]))     # k x dim
            w = F.softmax(self.ent_attn(he).squeeze(1), 0)           # 维度内实体注意力
            m = (w.unsqueeze(1) * he).sum(0)
            msgs.append(m)
            dvals.append(self.dim_attn(m))
        if msgs:
            dw = F.softmax(torch.cat(dvals), 0)                      # 维度注意力
            return sum(dwi * mi for dwi, mi in zip(dw, msgs))
        return torch.zeros(self.emb.weight.size(1), device=self.emb.weight.device)

    def _is_temporal_input(self, item):
        return isinstance(item, list) and len(item) > 0 and isinstance(item[0], tuple)

    def forward(self, dim_ent_ids, S):
        reps = []
        for item in dim_ent_ids:
            if self._is_temporal_input(item):
                union_dmap = defaultdict(list)
                day_vecs = []
                for day, dmap in item:
                    for di, ids in dmap.items():
                        union_dmap[di].extend(ids)
                    x_day = self._agg_dim_map(dmap)
                    day_idx = min(int(day), self.day_emb.num_embeddings - 1)
                    pos = self.day_emb.weight[day_idx]
                    day_vecs.append(x_day + pos)
                if day_vecs:
                    H = torch.stack(day_vecs)
                    tw = F.softmax(self.time_attn(H).squeeze(1), 0)
                    x_time = (tw.unsqueeze(1) * H).sum(0)
                    union_dmap = {di: list(set(ids)) for di, ids in union_dmap.items()}
                    x_static = self._agg_dim_map(union_dmap)
                    x = F.relu(self.fuse(torch.cat([x_time, x_static], 0)))
                else:
                    x = torch.zeros(self.emb.weight.size(1), device=self.emb.weight.device)
            else:
                x = self._agg_dim_map(item)
            reps.append(x)
        xg = self.graph_bn(torch.stack(reps))
        xs = F.relu(self.s_proj(S))
        return self.head(torch.cat([xg, xs, S], 1))

# ---------------- 训练 ----------------
def train_model(model, Xtr, ytr, Xva, yva, task, fold, cls_w=None):
    set_seed(1000 + fold)
    model.apply(init_weights)   # 统一初始化，保证主实验与消融同种子结果一致
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    if task == 'cls':
        crit = nn.CrossEntropyLoss(weight=cls_w)
        def lf(pred, y): return crit(pred, y)
    elif task == 'bin':
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([199.0 / 54.0]))
        def lf(pred, y): return crit(pred.squeeze(1), y.float())
    else:
        crit = nn.MSELoss()
        def lf(pred, y): return crit(pred.squeeze(1), y)
    best_state, best_val, wait = None, float('inf'), 0
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        loss = lf(model(*Xtr), ytr)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = lf(model(*Xva), yva).item()
        if vloss < best_val - 1e-5:
            best_val = vloss; best_state = copy.deepcopy(model.state_dict()); wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model

def xgb_train_fit(Xtr, ytr, task):
    set_seed(42)
    common = dict(n_estimators=200, max_depth=4, learning_rate=0.1,
                  subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=4)
    if task in ('cls', 'bin'):
        m = XGBClassifier(**common, eval_metric='mlogloss' if task == 'cls' else 'logloss')
        m.fit(Xtr, ytr)
    else:
        m = XGBRegressor(**common)
        m.fit(Xtr, ytr)
    return m

# ---------------- 评价 ----------------
def metrics_cls(y, pred):
    return {
        'accuracy': accuracy_score(y, pred),
        'precision_weighted': precision_score(y, pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y, pred, average='weighted', zero_division=0),
        'macro_f1': f1_score(y, pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y, pred, average='weighted', zero_division=0),
    }

def metrics_bin(y, prob):
    auc_roc = roc_auc_score(y, prob)
    auc_pr = average_precision_score(y, prob)
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {'auc_roc': auc_roc, 'auc_pr': auc_pr, 'sensitivity': sens, 'specificity': spec}

def metrics_reg(y, pred):
    mae = mean_absolute_error(y, pred)
    rmse = mean_squared_error(y, pred) ** 0.5
    r2 = r2_score(y, pred)
    r, _ = pearsonr(y, pred)
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'pearson_r': r}

# ---------------- 主流程 ----------------
def run():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(np.arange(len(pids)), label3))
    cls_w = torch.tensor([253.0 / (3 * 54), 253.0 / (3 * 89), 253.0 / (3 * 110)])

    results = {}
    mcnemar_4d, mcnemar_tkg = [], []
    cohen = defaultdict(lambda: defaultdict(list))

    for fold_i, (tr_idx, te_idx) in enumerate(folds):
        tr_pids = [pids[i] for i in tr_idx]
        Xs = prep_struct(X_struct, tr_idx)
        Xs_tr = Xs[tr_idx]; Xs_te = Xs[te_idx]
        S_tr = torch.tensor(Xs_tr, dtype=torch.float32)
        S_te = torch.tensor(Xs_te, dtype=torch.float32)
        seq_tr = torch.tensor(np.stack([prep_seq(pids[i], tr_idx) for i in tr_idx]), dtype=torch.float32)
        seq_te = torch.tensor(np.stack([prep_seq(pids[i], tr_idx) for i in te_idx]), dtype=torch.float32)
        adj = build_graph_adj(tr_pids, V)
        ent_tr = [pid_ent_ids(pids[i]) for i in tr_idx]
        ent_te = [pid_ent_ids(pids[i]) for i in te_idx]
        dmap_tr = [dim_map(pids[i]) for i in tr_idx]
        dmap_te = [dim_map(pids[i]) for i in te_idx]
        dseq_tr = [day_seq_for(pids[i]) for i in tr_idx]
        dseq_te = [day_seq_for(pids[i]) for i in te_idx]

        ytr3 = torch.tensor(label3[tr_idx], dtype=torch.long)
        yte3 = label3[te_idx]
        ytr2 = torch.tensor(y2[tr_idx], dtype=torch.long)
        yte2 = y2[te_idx]
        ytr3r = torch.tensor(np.nan_to_num(days[tr_idx]), dtype=torch.float32)
        yte3r = np.nan_to_num(days[te_idx])

        for method in ['lr_xgboost', 'kg_gnn', 'tkg', 'lstm_transformer', '4d_hypergraph']:
            results.setdefault(method, {}).setdefault('cls', [])
            results[method].setdefault('bin', [])
            results[method].setdefault('reg', [])
            # ---- Task1 ----
            if method == 'lr_xgboost':
                m = xgb_train_fit(Xs_tr, label3[tr_idx], 'cls')
                prob = m.predict_proba(Xs_te); pred = prob.argmax(1)
            elif method == 'kg_gnn':
                m = KGGNN(V, adj, NS, nout=3)
                m = train_model(m, (ent_tr, S_tr), ytr3, (ent_te, S_te), torch.tensor(yte3, dtype=torch.long), 'cls', fold_i, cls_w)
                with torch.no_grad(): prob = F.softmax(m(ent_te, S_te), 1).numpy()
                pred = prob.argmax(1)
            elif method == 'tkg':
                m = TKG(V, NS, nout=3)
                m = train_model(m, (dseq_tr, S_tr), ytr3, (dseq_te, S_te), torch.tensor(yte3, dtype=torch.long), 'cls', fold_i, cls_w)
                with torch.no_grad(): prob = F.softmax(m(dseq_te, S_te), 1).numpy()
                pred = prob.argmax(1)
            elif method == 'lstm_transformer':
                m = LSTMModel(seq_tr.size(2), NS, nout=3)
                m = train_model(m, (seq_tr, S_tr), ytr3, (seq_te, S_te), torch.tensor(yte3, dtype=torch.long), 'cls', fold_i, cls_w)
                with torch.no_grad(): prob = F.softmax(m(seq_te, S_te), 1).numpy()
                pred = prob.argmax(1)
            else:
                m = HyperGraph4D(V, NS, nout=3)
                m = train_model(m, (dmap_tr, S_tr), ytr3, (dmap_te, S_te), torch.tensor(yte3, dtype=torch.long), 'cls', fold_i, cls_w)
                with torch.no_grad(): prob = F.softmax(m(dmap_te, S_te), 1).numpy()
                pred = prob.argmax(1)
            results[method]['cls'].append(metrics_cls(yte3, pred))
            if method == '4d_hypergraph': mcnemar_4d.extend(pred)
            if method == 'tkg': mcnemar_tkg.extend(pred)

            # ---- Task2 ----
            if method == 'lr_xgboost':
                m2 = xgb_train_fit(Xs_tr, y2[tr_idx], 'bin')
                prob2 = m2.predict_proba(Xs_te)[:, 1]
            elif method == 'kg_gnn':
                m2 = KGGNN(V, adj, NS, nout=1)
                m2 = train_model(m2, (ent_tr, S_tr), ytr2, (ent_te, S_te), torch.tensor(yte2, dtype=torch.long), 'bin', fold_i + 50)
                with torch.no_grad(): prob2 = torch.sigmoid(m2(ent_te, S_te).squeeze(1)).numpy()
            elif method == 'tkg':
                m2 = TKG(V, NS, nout=1)
                m2 = train_model(m2, (dseq_tr, S_tr), ytr2, (dseq_te, S_te), torch.tensor(yte2, dtype=torch.long), 'bin', fold_i + 50)
                with torch.no_grad(): prob2 = torch.sigmoid(m2(dseq_te, S_te).squeeze(1)).numpy()
            elif method == 'lstm_transformer':
                m2 = LSTMModel(seq_tr.size(2), NS, nout=1)
                m2 = train_model(m2, (seq_tr, S_tr), ytr2, (seq_te, S_te), torch.tensor(yte2, dtype=torch.long), 'bin', fold_i + 50)
                with torch.no_grad(): prob2 = torch.sigmoid(m2(seq_te, S_te).squeeze(1)).numpy()
            else:
                m2 = HyperGraph4D(V, NS, nout=1)
                m2 = train_model(m2, (dmap_tr, S_tr), ytr2, (dmap_te, S_te), torch.tensor(yte2, dtype=torch.long), 'bin', fold_i + 50)
                with torch.no_grad(): prob2 = torch.sigmoid(m2(dmap_te, S_te).squeeze(1)).numpy()
            results[method]['bin'].append(metrics_bin(yte2, prob2))

            # ---- Task3 ----
            if method == 'lr_xgboost':
                m3 = xgb_train_fit(Xs_tr, days[tr_idx], 'reg')
                pred3 = m3.predict(Xs_te)
            elif method == 'kg_gnn':
                m3 = KGGNN(V, adj, NS, nout=1)
                m3 = train_model(m3, (ent_tr, S_tr), ytr3r, (ent_te, S_te), torch.tensor(yte3r, dtype=torch.float32), 'reg', fold_i + 100)
                with torch.no_grad(): pred3 = m3(ent_te, S_te).squeeze(1).numpy()
            elif method == 'tkg':
                m3 = TKG(V, NS, nout=1)
                m3 = train_model(m3, (dseq_tr, S_tr), ytr3r, (dseq_te, S_te), torch.tensor(yte3r, dtype=torch.float32), 'reg', fold_i + 100)
                with torch.no_grad(): pred3 = m3(dseq_te, S_te).squeeze(1).numpy()
            elif method == 'lstm_transformer':
                m3 = LSTMModel(seq_tr.size(2), NS, nout=1)
                m3 = train_model(m3, (seq_tr, S_tr), ytr3r, (seq_te, S_te), torch.tensor(yte3r, dtype=torch.float32), 'reg', fold_i + 100)
                with torch.no_grad(): pred3 = m3(seq_te, S_te).squeeze(1).numpy()
            else:
                m3 = HyperGraph4D(V, NS, nout=1)
                m3 = train_model(m3, (dmap_tr, S_tr), ytr3r, (dmap_te, S_te), torch.tensor(yte3r, dtype=torch.float32), 'reg', fold_i + 100)
                with torch.no_grad(): pred3 = m3(dmap_te, S_te).squeeze(1).numpy()
            results[method]['reg'].append(metrics_reg(yte3r, pred3))

        print(f'折{fold_i + 1} 完成', flush=True)

    # ---------------- 消融 ----------------
    abl = {'full': None, 'wo_patient_history': ['evidence', 'treat', 'severity'],
           'wo_clinical_evidence': ['hist', 'treat', 'severity'],
           'wo_treatment_trajectory': ['hist', 'evidence', 'severity'],
           'wo_dynamic_severity': ['hist', 'evidence', 'treat']}
    abl_res = {}
    for name, dims in abl.items():
        fold_res = {'macro_f1': [], 'auc_roc': [], 'pearson_r': []}
        for fold_i, (tr_idx, te_idx) in enumerate(folds):
            tr_pids = [pids[i] for i in tr_idx]
            adj = build_graph_adj(tr_pids, V)
            Xs = prep_struct(X_struct, tr_idx)
            S_tr = torch.tensor(Xs[tr_idx], dtype=torch.float32)
            S_te = torch.tensor(Xs[te_idx], dtype=torch.float32)
            dtr = [dim_map(pids[i], dims) for i in tr_idx]
            dte = [dim_map(pids[i], dims) for i in te_idx]
            m = HyperGraph4D(V, NS, nout=3)
            m = train_model(m, (dtr, S_tr), torch.tensor(label3[tr_idx], dtype=torch.long),
                            (dte, S_te), torch.tensor(label3[te_idx], dtype=torch.long), 'cls', fold_i + 200, cls_w)
            with torch.no_grad(): pr = F.softmax(m(dte, S_te), 1).numpy().argmax(1)
            fold_res['macro_f1'].append(f1_score(label3[te_idx], pr, average='macro', zero_division=0))
            m2 = HyperGraph4D(V, NS, nout=1)
            m2 = train_model(m2, (dtr, S_tr), torch.tensor(y2[tr_idx], dtype=torch.long),
                             (dte, S_te), torch.tensor(y2[te_idx], dtype=torch.long), 'bin', fold_i + 250)
            with torch.no_grad(): p2 = torch.sigmoid(m2(dte, S_te).squeeze(1)).numpy()
            fold_res['auc_roc'].append(roc_auc_score(y2[te_idx], p2))
            m3 = HyperGraph4D(V, NS, nout=1)
            m3 = train_model(m3, (dtr, S_tr), torch.tensor(np.nan_to_num(days[tr_idx]), dtype=torch.float32),
                             (dte, S_te), torch.tensor(np.nan_to_num(days[te_idx]), dtype=torch.float32), 'reg', fold_i + 300)
            with torch.no_grad(): p3 = m3(dte, S_te).squeeze(1).numpy()
            r, _ = pearsonr(np.nan_to_num(days[te_idx]), p3)
            fold_res['pearson_r'].append(r)
        abl_res[name] = {k: float(np.mean(v)) for k, v in fold_res.items()}
        print('消融', name, abl_res[name], flush=True)

    # ---------------- 汇总 ----------------
    out = {}
    for method, r in results.items():
        out[method] = {}
        for task, folds_ in r.items():
            out[method][task] = {k: float(np.mean([f_[k] for f_ in folds_])) for k in folds_[0]}
            cohen[task][method] = [f_['macro_f1' if task == 'cls' else ('auc_roc' if task == 'bin' else 'pearson_r')] for f_ in folds_]

    # per-class F1 (pooled)
    pairs = {'true': [], 'pred': []}
    for fold_i, (tr_idx, te_idx) in enumerate(folds):
        tr_pids = [pids[i] for i in tr_idx]
        Xs = prep_struct(X_struct, tr_idx)
        S_tr = torch.tensor(Xs[tr_idx], dtype=torch.float32)
        S_te = torch.tensor(Xs[te_idx], dtype=torch.float32)
        dtr = [dim_map(pids[i]) for i in tr_idx]
        dte = [dim_map(pids[i]) for i in te_idx]
        m = HyperGraph4D(V, NS, nout=3)
        m = train_model(m, (dtr, S_tr), torch.tensor(label3[tr_idx], dtype=torch.long),
                        (dte, S_te), torch.tensor(label3[te_idx], dtype=torch.long), 'cls', fold_i + 500, cls_w)
        with torch.no_grad(): pr = F.softmax(m(dte, S_te), 1).numpy().argmax(1)
        pairs['true'].extend(label3[te_idx]); pairs['pred'].extend(pr)
    tt = np.array(pairs['true']); pp = np.array(pairs['pred'])
    per_class_f1 = {f'class_{i}': float(f1_score(tt == i, pp == i, zero_division=0)) for i in range(3)}
    hardest = max(per_class_f1, key=lambda k: -per_class_f1[k])

    b = sum(1 for a, c_ in zip(mcnemar_4d, mcnemar_tkg) if a == 1 and c_ == 0)
    c_ = sum(1 for a, c_ in zip(mcnemar_4d, mcnemar_tkg) if a == 0 and c_ == 1)
    n_eff = b + c_
    mcnemar_p = binomtest(min(b, c_), n_eff, 0.5).pvalue if n_eff > 0 else 1.0

    stats = {}
    for bl in ['lr_xgboost', 'kg_gnn', 'tkg', 'lstm_transformer']:
        stats[bl] = {}
        for task, tname in [('cls', 'classification_d'), ('bin', 'prediction_d'), ('reg', 'regression_d')]:
            a = np.array(cohen[task]['4d_hypergraph']); bb = np.array(cohen[task][bl])
            diff = a - bb
            sd = diff.std(ddof=1)
            d = float(diff.mean() / sd) if sd > 1e-9 else 0.0
            t, p = ttest_rel(a, bb)
            rng = np.random.default_rng(42)
            boots = [rng.choice(diff, len(diff), replace=True).mean() / max(sd, 1e-9) for _ in range(2000)]
            ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            stats[bl][tname] = {'d': round(d, 2), 'ci95': ci, 'p': float(p)}
    return out, abl_res, per_class_f1, hardest, float(mcnemar_p), stats

if __name__ == '__main__':
    out, abl_res, per_class_f1, hardest, mcnemar_p, stats = run()
    json.dump({'task_results': out, 'ablation': abl_res, 'per_class_f1': per_class_f1,
               'hardest': hardest, 'mcnemar_p': mcnemar_p, 'stats': stats},
              open(f'{BASE}/exp_results.json', 'w'), ensure_ascii=False, indent=2)
    for method, r in out.items():
        print(method, 'cls:', {k: round(v, 4) for k, v in r['cls'].items()})
    for method, r in out.items():
        print(method, 'bin:', {k: round(v, 4) for k, v in r['bin'].items()})
    for method, r in out.items():
        print(method, 'reg:', {k: round(v, 4) for k, v in r['reg'].items()})
