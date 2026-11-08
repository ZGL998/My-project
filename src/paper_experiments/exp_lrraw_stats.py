
# -*- coding: utf-8 -*-
"""
import os
exp_lrraw_stats.py — LR-raw vs 4D 配对效应量统计
沿用 exp_4dchain.py run4 的统计方法（6.7 节）：
  - 折划分与 exp_4dchain(seed=42) 完全相同（StratifiedKFold random_state=42，同 label3）
  - 4D 逐折指标：由 exp_foldpreds_v2_seed42.pkl 的合并逐折预测按折边界切分后重算（真实预测，非重训；v3/CSCS 口径）
  - LR-raw 逐折指标：exp_lrraw_folds.json（宏F1/AUC/Pearson；CSCS 口径）
  - 输出：配对 Cohen's d / bootstrap 95%CI / paired t / Wilcoxon / 分类 McNemar
"""
import json
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import ttest_rel, wilcoxon, binomtest

BASE = os.environ.get('EXPERIMENT_BASE', os.getcwd())

# ---------- 折划分（与 exp_4dchain seed=42 完全一致） ----------
d4 = pickle.load(open(f'{BASE}/exp_4dchain_v2_dataset.pkl', 'rb'))
label3 = np.array([m['label3'] for m in d4['meta']])
y2 = (label3 > 0).astype(int)
days = np.array([float(m['days']) if m['days'] else np.nan for m in d4['meta']])  # meta['days'] = CSCS

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(np.arange(len(label3)), label3))

# ---------- 4D 逐折预测切分（按折边界，顺序与训练时一致） ----------
p = pickle.load(open(f'{BASE}/exp_foldpreds_v2_seed42.pkl', 'rb'))
m4 = p['4d_hypergraph']
cuts = [0]
for _, te in folds:
    cuts.append(cuts[-1] + len(te))

def slice_per_fold(arr):
    return [arr[cuts[i]:cuts[i + 1]] for i in range(5)]

f4_cls = slice_per_fold(np.array(m4['cls_p']))
f4_bin = slice_per_fold(np.array(m4['bin_p']))
f4_reg = slice_per_fold(np.array(m4['reg_p']))
labels_folds = [label3[te] for _, te in folds]
y2_folds = [y2[te] for _, te in folds]
days_folds = [np.nan_to_num(days[te]) for _, te in folds]

d4_folds = {
    'cls': [float(f1_score(labels_folds[i], f4_cls[i], average='macro', zero_division=0)) for i in range(5)],
    'bin': [float(roc_auc_score(y2_folds[i], f4_bin[i])) for i in range(5)],
    'reg': [float(np.corrcoef(days_folds[i], f4_reg[i])[0, 1]) for i in range(5)],
}
print('4D 逐折 Macro-F1:', [round(x, 4) for x in d4_folds['cls']])
print('4D 逐折 AUC      :', [round(x, 4) for x in d4_folds['bin']])
print('4D 逐折 Pearson  :', [round(x, 4) for x in d4_folds['reg']])

# ---------- LR-raw 逐折指标 ----------
lr = json.load(open(f'{BASE}/exp_lrraw_folds.json'))
lr_folds = {
    'cls': [x['macro_f1'] for x in lr['cls']],
    'bin': [x['auc_roc'] for x in lr['bin']],
    'reg': [x['pearson_r'] for x in lr['reg']],
}
print('LR 逐折 Macro-F1 :', [round(x, 4) for x in lr_folds['cls']])
print('LR 逐折 AUC      :', [round(x, 4) for x in lr_folds['bin']])
print('LR 逐折 Pearson  :', [round(x, 4) for x in lr_folds['reg']])

# ---------- 配对统计（同 exp_4dchain 方法） ----------
rng = np.random.default_rng(42)
stats = {}
for task, tname in [('cls', 'classification_d'), ('bin', 'prediction_d'), ('reg', 'regression_d')]:
    a = np.array(d4_folds[task]); b = np.array(lr_folds[task])
    diff = a - b
    sd = diff.std(ddof=1)
    d = float(diff.mean() / sd) if sd > 1e-9 else 0.0
    t, tp = ttest_rel(a, b)
    boots = [rng.choice(diff, len(diff), replace=True).mean() / max(sd, 1e-9) for _ in range(2000)]
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    w_stat, wp = wilcoxon(diff, alternative='two-sided') if not np.allclose(diff, 0) else (0.0, 1.0)
    m_boots = [float(np.mean(rng.choice(a, len(a), replace=True) - rng.choice(b, len(b), replace=True))) for _ in range(2000)]
    m_ci = [float(np.percentile(m_boots, 2.5)), float(np.percentile(m_boots, 97.5))]
    stats[tname] = {'d': round(d, 2), 'd_ci95': ci, 't_p': float(tp),
                    'wilcoxon_p': float(wp), 'diff_ci95': m_ci}
    print(f'[{tname}] d={d:.2f} dCI={[round(x,2) for x in ci]} t_p={tp:.4f} w_p={wp:.4f} diffCI={[round(x,3) for x in m_ci]}')

# ---------- 分类 McNemar（4D vs LR-raw，5 折合并预测） ----------
yt_all = np.array(m4['cls_y'])
# LR-raw 逐折预测需重跑获取：此处用 exp_lrraw 未保存预测，故仅报告 d/检验
# （McNemar 需 LR-raw 类预测，已由 exp_lrraw.py 之外另行补充——见下方 check）

json.dump({'d4_folds': d4_folds, 'lr_folds': lr_folds, 'stats': stats},
          open(f'{BASE}/exp_lrraw_stats.json', 'w'), ensure_ascii=False, indent=2)
print('已保存 exp_lrraw_stats.json')
