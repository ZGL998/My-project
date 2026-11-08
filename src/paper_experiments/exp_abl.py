
# -*- coding: utf-8 -*-
"""
import os
4D-HyperGraph 消融重跑（v3/CSCS 口径）。

与 exp_4dchain.py run4(do_ablation=True) 的消融块完全一致，保证可复现：
- 数据集: exp_4dchain_v2_dataset.pkl（meta['days'] = CSCS 复合评分 = 0.30×GOLD + 0.30×加重负担 + 0.20×合并症 + 0.20×呼衰）
- 模型: exp_run.HyperGraph4D（维度线性层 + 实体/维度注意力 + 日位置嵌入 + 时间注意力池化 + 静态/时序融合 + MLP 头）
- 时序超边: day_dim_seq_for 按天按维聚合（固定 5 天窗口 WINDOW=5）；移除某维度 = 从超边中剔除该维度全部实体
- 种子/折: StratifiedKFold(random_state=42)；fold 训练种子 1000+fold_i(分类) / +50(二分类) / +100(回归)
- Full = dims=None 重新训练（与主实验 4D 行同构，应复现 §6.3-6.5 的 4D-HyperGraph 行）

用法: python exp_abl.py   # 输出 exp_ablation.json
"""
import pickle, copy, random, os, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import torch
torch.set_num_threads(4)
import torch.nn.functional as F
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import pearsonr

BASE = os.environ.get('EXPERIMENT_BASE', os.getcwd())
import exp_run as ER  # 复用 HyperGraph4D / train_model / EPOCHS=150 / PATIENCE=20 / DIM=32

# ---- v3 数据集（meta['days'] = CSCS，替代 v2 住院天数标签）----
ds4 = pickle.load(open(f'{BASE}/exp_4dchain_v2_dataset.pkl', 'rb'))
pids = ds4['pids']
meta = ds4['meta']
label3 = np.array([m['label3'] for m in meta])
y2 = (label3 > 0).astype(int)
days = np.array([float(m['days']) if m['days'] else np.nan for m in meta])
X_struct = ds4['X_struct']
ent2id = ds4['ent2id']
entity_data = ds4['entity_data']
day_ent_data = ds4['day_ent_data']
NS = X_struct.shape[1]
DIM_ORDER = ['hist', 'evidence', 'treat', 'severity']
cls_w = torch.tensor([253.0 / (3 * 54), 253.0 / (3 * 89), 253.0 / (3 * 110)])

def prep_struct(X, Xtr_idx):
    tr = X[Xtr_idx]
    med = np.nanmedian(tr, axis=0); med[np.isnan(med)] = 0.0
    std = np.nanstd(tr, axis=0); std[np.isnan(std)] = 1.0; std[std < 1e-6] = 1.0
    Xi = np.where(np.isnan(X), med, X)
    return (Xi - med) / std

def dim_map(pid, dims=None):
    ed = entity_data[pid]
    dm = {}
    for di, dmname in enumerate(DIM_ORDER):
        if dims is not None and dmname not in dims:
            continue
        dm[di] = list(set(ent2id[e] for e in ed.get(dmname, set()) if e in ent2id))
    return dm

def day_dim_seq_for(pid, dims=None):
    """按天按维聚合实体 → [(day, {dim_idx: [ent_ids]}), ...]；dims 过滤即移除该维度全部实体。"""
    dim_by_prefix = {
        'sym': 'hist', 'diag': 'hist',
        'lab': 'evidence', 'img': 'evidence',
        'med': 'treat',
        'score': 'severity', 'sev': 'severity',
    }
    out = []
    for d, es in day_ent_data.get(pid, []):
        dmap = defaultdict(list)
        for e in es:
            if e not in ent2id:
                continue
            dmname = dim_by_prefix.get(e[0])
            if dmname is None:
                continue
            if dims is not None and dmname not in dims:
                continue
            di = DIM_ORDER.index(dmname)
            dmap[di].append(ent2id[e])
        if dmap:
            out.append((d, {di: list(set(ids)) for di, ids in dmap.items()}))
    if out:
        return out
    dm = dim_map(pid, dims)
    return [(0, dm)] if dm else [(0, {})]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(np.arange(len(pids)), label3))

variants = {
    'full': None,
    'wo_patient_history': ['evidence', 'treat', 'severity'],
    'wo_clinical_evidence': ['hist', 'treat', 'severity'],
    'wo_treatment_trajectory': ['hist', 'evidence', 'severity'],
    'wo_dynamic_severity': ['hist', 'evidence', 'treat'],
}

out = {}
for name, dims in variants.items():
    fold_res = {'macro_f1': [], 'auc_roc': [], 'pearson_r': []}
    for fold_i, (tr_idx, te_idx) in enumerate(folds):
        Xs = prep_struct(X_struct, tr_idx)
        S_tr = torch.tensor(Xs[tr_idx], dtype=torch.float32)
        S_te = torch.tensor(Xs[te_idx], dtype=torch.float32)
        dtr = [day_dim_seq_for(pids[i], dims) for i in tr_idx]
        dte = [day_dim_seq_for(pids[i], dims) for i in te_idx]
        # ---- Task1 三分类 ----
        m = ER.HyperGraph4D(len(ent2id), NS, nout=3)
        m = ER.train_model(m, (dtr, S_tr), torch.tensor(label3[tr_idx], dtype=torch.long),
                           (dte, S_te), torch.tensor(label3[te_idx], dtype=torch.long), 'cls', fold_i, cls_w)
        with torch.no_grad(): pr = F.softmax(m(dte, S_te), 1).numpy().argmax(1)
        fold_res['macro_f1'].append(f1_score(label3[te_idx], pr, average='macro', zero_division=0))
        # ---- Task2 恶化二分类 ----
        m2 = ER.HyperGraph4D(len(ent2id), NS, nout=1)
        m2 = ER.train_model(m2, (dtr, S_tr), torch.tensor(y2[tr_idx], dtype=torch.long),
                            (dte, S_te), torch.tensor(y2[te_idx], dtype=torch.long), 'bin', fold_i + 50)
        with torch.no_grad(): p2 = torch.sigmoid(m2(dte, S_te).squeeze(1)).numpy()
        fold_res['auc_roc'].append(roc_auc_score(y2[te_idx], p2))
        # ---- Task3 CSCS 回归（z-score 标准化，预测反标准化）----
        reg_mu = float(np.nanmean(days[tr_idx]))
        reg_sd = float(np.nanstd(days[tr_idx]))
        reg_sd = reg_sd if reg_sd > 1e-6 else 1.0
        m3 = ER.HyperGraph4D(len(ent2id), NS, nout=1)
        m3 = ER.train_model(m3, (dtr, S_tr),
                            torch.tensor((np.nan_to_num(days[tr_idx]) - reg_mu) / reg_sd, dtype=torch.float32),
                            (dte, S_te),
                            torch.tensor((np.nan_to_num(days[te_idx]) - reg_mu) / reg_sd, dtype=torch.float32),
                            'reg', fold_i + 100)
        with torch.no_grad(): p3 = m3(dte, S_te).squeeze(1).numpy() * reg_sd + reg_mu
        r, _ = pearsonr(np.nan_to_num(days[te_idx]), p3)
        fold_res['pearson_r'].append(r)
    out[name] = {k: float(np.mean(v)) for k, v in fold_res.items()}
    print(name, out[name], flush=True)

json.dump(out, open(f'{BASE}/exp_ablation.json', 'w'), ensure_ascii=False, indent=2)
print('已写入 exp_ablation.json')
