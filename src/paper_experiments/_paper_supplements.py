
# -*- coding: utf-8 -*-
"""论文补充实验（统一使用回退后最终数字口径）：
1) 同读出头论证图（LR/XGBoost vs 4D-XGB，唯一变量=输入表示）
2) SHAP 特征归因（按维度/实体组聚合）
3) 外部混淆矩阵补齐（balanced87 / strict72）
4) 时间外验证 3-seed 化
5) 特征级增量实验（Struct -> +Graph -> +Ent，量化表示增量）
6) Task2 决策阈值分析
"""
import json
import os
import pickle
import sys
import numpy as np

BASE = os.environ.get('EXPERIMENT_BASE', os.getcwd())
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
sys.path.insert(0, os.environ.get('HYPERGRAPHRAG_PATH',
                                  os.path.join(_REPO, 'third_party', 'HyperGraphRAG')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

import exp_run as ERM
import exp_4dchain_v2 as V2
from exp_run import metrics_cls, metrics_bin
from xgboost import XGBClassifier, XGBRegressor

plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
OUT = f'{BASE}/paper_supplements'

SEEDS = [42, 123, 456]
DIM_ORDER = ['hist', 'evidence', 'treat', 'severity']
DIM_ZH = {'hist': 'History', 'evidence': 'Evidence', 'treat': 'Treatment', 'severity': 'Severity'}
PREFIX_TO_DIM = {
    'sym': 'hist', 'diag': 'hist',
    'lab': 'evidence', 'symp': 'evidence', 'phys': 'evidence', 'exam': 'evidence',
    'med': 'treat', 'proc': 'treat', 'care': 'treat',
    'sev': 'severity', 'comp': 'severity', 'func': 'severity', 'prog': 'severity',
}


def xgb_fit(Xtr, ytr, task, seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    common = dict(n_estimators=200, max_depth=4, learning_rate=0.1,
                  subsample=0.9, colsample_bytree=0.9, random_state=seed, n_jobs=4)
    if task in ('cls', 'bin'):
        m = XGBClassifier(**common, eval_metric='mlogloss' if task == 'cls' else 'logloss')
        m.fit(Xtr, ytr)
    else:
        m = XGBRegressor(**common)
        m.fit(Xtr, ytr)
    return m


# ---------------- 1) 同读出头论证 ----------------
def same_readout_figure():
    # 统一使用主表 3-seed 口径（exp_results_task12_3seed_summary_ci.json），
    # 与论文正文/Table 2/3 的 3-seed 数字完全一致，避免与存档 foldpreds 口径出现双数字。
    summ = json.load(open(f'{BASE}/exp_results_task12_3seed_summary_ci.json'))['summary']
    agg = {
        '4D-XGB': {'cls_mf1': (summ['4d_xgb']['cls']['mean'], summ['4d_xgb']['cls']['sd']),
                   'bin_auc': (summ['4d_xgb']['bin']['mean'], summ['4d_xgb']['bin']['sd'])},
        'LR/XGBoost': {'cls_mf1': (summ['lr_xgboost']['cls']['mean'], summ['lr_xgboost']['cls']['sd']),
                       'bin_auc': (summ['lr_xgboost']['bin']['mean'], summ['lr_xgboost']['bin']['sd'])},
    }
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    names = ['4D-XGB', 'LR/XGBoost']
    x = np.arange(2)
    for i, (metric, title) in enumerate((('cls_mf1', 'Task1 Macro-F1 (3-seed)'), ('bin_auc', 'Task2 AUC-ROC (3-seed)'))):
        vals = [agg[n][metric][0] for n in names]
        errs = [agg[n][metric][1] for n in names]
        ax[i].bar(x, vals, yerr=errs, capsize=4, color=['#4C72B0', '#DD8452'])
        ax[i].set_xticks(x); ax[i].set_xticklabels(names)
        ax[i].set_title(title); ax[i].set_ylim(0, 0.85)
        for xi, v in zip(x, vals):
            ax[i].text(xi, v + 0.02, f'{v:.3f}', ha='center')
    fig.suptitle('Same XGBoost readout head; the only varying component is the input representation')
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig1_same_readout.png', dpi=150)
    plt.close(fig)
    print('[1] 同读出头图 -> fig1_same_readout.png')
    for n in names:
        print(f'    {n}: cls_mf1={agg[n]["cls_mf1"][0]:.4f}±{agg[n]["cls_mf1"][1]:.4f} | bin_auc={agg[n]["bin_auc"][0]:.4f}±{agg[n]["bin_auc"][1]:.4f}')


# ---------------- 2) SHAP 特征归因 ----------------
def shap_attribution(ds):
    import shap
    label3 = np.array([m['label3'] for m in ds['meta']])
    y2 = (label3 > 0).astype(int)
    X, fnames = V2.build_4d_xgb_features(ds, return_feature_names=True)
    groups = {g: [] for g in
              ['Struct', 'Struct_sev', 'Graph:hist', 'Graph:evidence', 'Graph:treat', 'Graph:severity',
               'Ent:hist', 'Ent:evidence', 'Ent:treat', 'Ent:severity']}
    for i, fn in enumerate(fnames):
        if fn.startswith('ent::'):
            ent = fn[5:].split('::')[0]
            dm = PREFIX_TO_DIM.get(ent, 'other')
            groups.setdefault(f'Ent:{dm}', []).append(i)
        elif fn.startswith(('hist_', 'evidence_', 'treat_', 'severity_')):
            dm = fn.split('_')[0]
            groups[f'Graph:{dm}'].append(i)
        elif fn.startswith('sev_'):
            groups['Struct_sev'].append(i)
        else:
            groups['Struct'].append(i)
    groups = {k: v for k, v in groups.items() if len(v)}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gsum = {g: [] for g in groups}
    for tr_idx, te_idx in skf.split(X, label3):
        Xs = ERM.prep_struct(X, tr_idx)
        Xtr, Xte = Xs[tr_idx], Xs[te_idx]
        m1 = xgb_fit(Xtr, label3[tr_idx], 'cls', 42)
        ex1 = shap.TreeExplainer(m1)
        sv1 = ex1.shap_values(Xte)
        if isinstance(sv1, list):
            sa1 = np.abs(np.stack(sv1, axis=-1))     # (n, f, C)
        else:
            sa1 = np.abs(sv1) if sv1.ndim == 3 else np.abs(sv1)[:, :, None]
        m1shap = sa1.mean(axis=(0, -1))              # (f,)
        m2 = xgb_fit(Xtr, y2[tr_idx], 'bin', 42)
        sv2 = shap.TreeExplainer(m2).shap_values(Xte)
        sa2 = np.abs(np.asarray(sv2))
        m2shap = sa2.mean(axis=0) if sa2.ndim == 2 else sa2.mean(axis=(0, -1))
        for g, idxs in groups.items():
            gsum[g].append([float(m1shap[idxs].sum()), float(m2shap[idxs].sum())])
    mean1 = {g: np.mean([v[0] for v in vals]) for g, vals in gsum.items()}
    mean2 = {g: np.mean([v[1] for v in vals]) for g, vals in gsum.items()}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for i, (mean, title) in enumerate(((mean1, 'Task1 (3-class) mean|SHAP| by group'),
                                       (mean2, 'Task2 (exacerbation) mean|SHAP| by group'))):
        gs = sorted(mean, key=mean.get)
        vals = [mean[g] for g in gs]
        ax[i].barh(range(len(gs)), vals, color='#4C72B0')
        ax[i].set_yticks(range(len(gs))); ax[i].set_yticklabels(gs, fontsize=8)
        ax[i].set_title(title)
        for j, v in enumerate(vals):
            ax[i].text(v, j, f'{v:.4f}', va='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig2_shap_by_group.png', dpi=150)
    plt.close(fig)
    print('[2] SHAP -> fig2_shap_by_group.png')
    for g in sorted(mean1, key=mean1.get, reverse=True):
        print(f'    {g:<16} Task1={mean1[g]:.4f}  Task2={mean2[g]:.4f}')
    json.dump({'task1': mean1, 'task2': mean2, 'groups': {g: groups[g] for g in groups}},
              open(f'{OUT}/shap_by_group.json', 'w'), indent=2)


# ---------------- 3) 外部混淆矩阵 ----------------
def external_confusion():
    FOLDERS = {
        'core30': 'exp_foldpreds_task12_external_external_chain_eval_input.pkl',
        'balanced87': 'exp_foldpreds_task12_external_external_chain_eval_input_balanced87.pkl',
        'strict72': 'exp_foldpreds_task12_external_external_chain_eval_input_strict72.pkl',
    }
    for tag, f in FOLDERS.items():
        p = pickle.load(open(f'{BASE}/{f}', 'rb'))
        print(f'[{tag}]')
        for method in ('4d_xgb', 'lr_xgboost'):
            d = p[method]
            cm = confusion_matrix(d['cls_y'], d['cls_p'])
            rec = cm.diagonal() / cm.sum(1)
            bcm = confusion_matrix(d['bin_y'], np.array(d['bin_p']) >= 0.5)
            print(f'  {method} Task1 CM={cm.tolist()} recall={np.round(rec, 3).tolist()}')
            print(f'  {method} Task2 CM={bcm.tolist()}')


# ---------------- 4) 时间外 3-seed ----------------
def temporal_3seed(cases):
    ds = pickle.load(open(f'{BASE}/exp_4dchain_v2_dataset.pkl', 'rb'))
    pids = ds['pids']
    label3 = np.array([m['label3'] for m in ds['meta']])
    y2 = (label3 > 0).astype(int)
    tr_idx, te_idx, cutoff = V2.temporal_split_indices(cases, pids, train_frac=0.7)
    mats = {
        '4d_hypergraph': V2.build_4d_xgb_features(ds),
        'lr_xgboost': ds['X_struct'],
    }
    per_seed = {}
    for seed in SEEDS:
        per_seed[seed] = {}
        for name, X in mats.items():
            Xs = ERM.prep_struct(X, tr_idx)
            m1 = xgb_fit(Xs[tr_idx], label3[tr_idx], 'cls', seed)
            pred1 = m1.predict_proba(Xs[te_idx]).argmax(1)
            m2 = xgb_fit(Xs[tr_idx], y2[tr_idx], 'bin', seed)
            prob2 = m2.predict_proba(Xs[te_idx])[:, 1]
            per_seed[seed][name] = {
                'cls': metrics_cls(label3[te_idx], pred1),
                'bin': metrics_bin(y2[te_idx], prob2),
            }
    print(f'[4] 时间外 3-seed（cutoff={cutoff.date()}，train={len(tr_idx)}/test={len(te_idx)}）')
    for name in mats:
        mf1s = [per_seed[s][name]['cls']['macro_f1'] for s in SEEDS]
        aucs = [per_seed[s][name]['bin']['auc_roc'] for s in SEEDS]
        print(f'    {name:<14} Task1 mf1={np.mean(mf1s):.4f}±{np.std(mf1s):.4f} | Task2 auc={np.mean(aucs):.4f}±{np.std(aucs):.4f}')
    json.dump({'split': {'train_n': len(tr_idx), 'test_n': len(te_idx), 'cutoff': str(cutoff.date())},
               'per_seed': per_seed},
              open(f'{OUT}/temporal_3seed.json', 'w'), indent=2, default=str)


# ---------------- 6) Task2 阈值分析 ----------------
def threshold_analysis():
    ys, ps = [], []
    for seed in SEEDS:
        d = pickle.load(open(f'{BASE}/exp_foldpreds_v2_seed{seed}_4dxgb_task12.pkl', 'rb'))['4d_hypergraph']
        ys += d['bin_y']; ps += d['bin_p']
    ys = np.array(ys); ps = np.array(ps)
    thr = np.linspace(0, 1, 201)
    sens = []; spec = []; youden = []
    for t in thr:
        pred = (ps >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(ys, pred).ravel()
        sn = tp / (tp + fn); sp = tn / (tn + fp)
        sens.append(sn); spec.append(sp); youden.append(sn + sp - 1)
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    ax.plot(thr, sens, label='Sensitivity', color='#4C72B0')
    ax.plot(thr, spec, label='Specificity', color='#DD8452')
    ax.plot(thr, youden, label='Youden J', color='#55A868', ls='--')
    j = int(np.argmax(youden))
    ax.axvline(0.5, color='gray', ls=':', lw=1)
    ax.scatter([thr[j]], [youden[j]], color='#55A868', zorder=5)
    ax.annotate(f'Youden optimal t={thr[j]:.2f}', (thr[j], youden[j]),
                textcoords='offset points', xytext=(8, -12), fontsize=8)
    ax.set_xlabel('Decision threshold'); ax.set_ylabel('Metric')
    ax.legend(); ax.set_title('Task2 decision threshold analysis (internal, 3-seed pooled, n=251)')
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig6_threshold.png', dpi=150)
    plt.close(fig)
    # 0.5 阈值点数值
    i50 = np.argmin(np.abs(thr - 0.5))
    print(f'[6] 阈值分析: 0.5阈值 -> sens={sens[i50]:.3f} spec={spec[i50]:.3f}; '
          f'Youden最优 t={thr[j]:.3f} -> sens={sens[j]:.3f} spec={spec[j]:.3f} J={youden[j]:.3f}')


# ---------------- 5) 特征级增量实验 ----------------
def feature_incremental():
    """XGBoost 读出头 + 完全相同的 5 折划分（random_state=seed），仅改变特征集层级：
       Struct (29 列) -> Struct+Graph (29+76) -> Struct+Graph+Ent (482 列, 完整4D)
    量化"表示增量"的来源：图统计贡献多少、实体 one-hot 再贡献多少，
    直接回应审稿人"4D 是否只是多了信息"的质疑。"""
    ds = pickle.load(open(f'{BASE}/exp_4dchain_v2_dataset.pkl', 'rb'))
    label3 = np.array([m['label3'] for m in ds['meta']])
    y2 = (label3 > 0).astype(int)
    X_full, fnames = V2.build_4d_xgb_features(ds, return_feature_names=True)
    ent_idx = np.array([i for i, fn in enumerate(fnames) if fn.startswith('ent::')])
    graph_idx = np.array([i for i, fn in enumerate(fnames)
                          if fn.startswith(('hist_', 'evidence_', 'treat_', 'severity_'))])
    X_struct = ds['X_struct']
    X_graph = X_full[:, graph_idx]
    X_ent = X_full[:, ent_idx]
    mats = {
        'Struct only': X_struct,
        'Struct+Graph': np.concatenate([X_struct, X_graph], axis=1),
        'Struct+Graph+Ent (4D)': np.concatenate([X_struct, X_graph, X_ent], axis=1),
    }
    print('[5] 特征级增量实验 列数:', {k: v.shape[1] for k, v in mats.items()})
    per_seed = {s: {} for s in SEEDS}
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for name, X in mats.items():
            mf1s, aucs = [], []
            for tr_idx, te_idx in skf.split(X, label3):
                Xs = ERM.prep_struct(X, tr_idx)
                m1 = xgb_fit(Xs[tr_idx], label3[tr_idx], 'cls', seed)
                pred1 = m1.predict_proba(Xs[te_idx]).argmax(1)
                m2 = xgb_fit(Xs[tr_idx], y2[tr_idx], 'bin', seed)
                prob2 = m2.predict_proba(Xs[te_idx])[:, 1]
                mf1s.append(metrics_cls(label3[te_idx], pred1)['macro_f1'])
                aucs.append(metrics_bin(y2[te_idx], prob2)['auc_roc'])
            per_seed[seed][name] = {'cls_mf1': float(np.mean(mf1s)), 'bin_auc': float(np.mean(aucs))}
    names = list(mats)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for i, (metric, title, color) in enumerate((
            ('cls_mf1', 'Task1 Macro-F1 (3-seed)', '#4C72B0'),
            ('bin_auc', 'Task2 AUC-ROC (3-seed)', '#DD8452'))):
        vals = [np.mean([per_seed[s][n][metric] for s in SEEDS]) for n in names]
        errs = [np.std([per_seed[s][n][metric] for s in SEEDS]) for n in names]
        x = np.arange(len(names))
        ax[i].bar(x, vals, yerr=errs, capsize=4, color=color)
        ax[i].set_xticks(x); ax[i].set_xticklabels(names, fontsize=8)
        ax[i].set_title(title); ax[i].set_ylim(0, 0.85)
        for xi, v, e in zip(x, vals, errs):
            ax[i].text(xi, v + e + 0.02, f'{v:.3f}\u00b1{e:.3f}', ha='center', fontsize=8)
    fig.suptitle('Feature-level incremental: same XGBoost readout & folds, only feature set varies')
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig5_feature_incremental.png', dpi=150)
    plt.close(fig)
    print('[5] 特征级增量 -> fig5_feature_incremental.png')
    for n in names:
        m = np.mean([per_seed[s][n]['cls_mf1'] for s in SEEDS]); sd = np.std([per_seed[s][n]['cls_mf1'] for s in SEEDS])
        a = np.mean([per_seed[s][n]['bin_auc'] for s in SEEDS]); sa = np.std([per_seed[s][n]['bin_auc'] for s in SEEDS])
        print(f'    {n:<22} Task1 mf1={m:.4f}\u00b1{sd:.4f} | Task2 auc={a:.4f}\u00b1{sa:.4f}')
    json.dump({
        'n_features': {k: v.shape[1] for k, v in mats.items()},
        'per_seed': per_seed,
        'summary': {n: {
            'cls_mf1': [float(np.mean([per_seed[s][n]['cls_mf1'] for s in SEEDS])),
                        float(np.std([per_seed[s][n]['cls_mf1'] for s in SEEDS]))],
            'bin_auc': [float(np.mean([per_seed[s][n]['bin_auc'] for s in SEEDS])),
                        float(np.std([per_seed[s][n]['bin_auc'] for s in SEEDS]))]}
            for n in names}},
        open(f'{OUT}/feature_incremental.json', 'w'), indent=2)


if __name__ == '__main__':
    same_readout_figure()
    ds = pickle.load(open(f'{BASE}/exp_4dchain_v2_dataset.pkl', 'rb'))
    shap_attribution(ds)
    external_confusion()
    import run_hgr_eval as R
    cases = R.load_cases()
    temporal_3seed(cases)
    threshold_analysis()
    print('全部完成，产物在 paper_supplements/')
