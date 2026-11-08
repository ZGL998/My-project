# -*- coding: utf-8 -*-
"""
Task 2 / Task 3 支撑数字（干净口径）
====================================
Task 2（横断面二分类 A vs B/C）
  - 表示：S / S+GRAPH / S+GRAPH+ENT
  - 指标：AUC-ROC、AUC-PR、sensitivity、specificity（逐折平均 × 3 seeds）
Task 3（CSCS 严重度回归）——论文 §3.5 / Table 3b / Figure 3 的实现
  - 目标：CSCS（存于 meta['days']）
  - 两种输入：
      circular : 含 CSCS 分量（GOLD/FEV1/mMRC/SpO2/PaCO2 列 + score/sev 实体）→ 定义泄漏，仅作对照
      fixed    : 剔除上述分量（列 24-28 去掉 + score/sev 实体去掉）→ **论文报告的 component-blind 口径**
  - 指标：MAE、RMSE、R²、Pearson r（5 折 × 3 seeds 均值）
输出：$EVENT_OUT/_task_fix_probe.json
"""
import json
import os
import pickle
from itertools import combinations

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, \
    mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from xgboost import XGBClassifier, XGBRegressor

BASE = os.environ.get('EXPERIMENT_BASE', os.getcwd())
OUT = os.environ.get('EVENT_OUT', f'{BASE}/hg_event_run')
DS = os.environ.get('DS4_PATH', f'{BASE}/exp_4dchain_v2_dataset.pkl')
SEEDS = [42, 123, 456]
DIMS = ['hist', 'evidence', 'treat', 'severity']
P2D = {'sym': 'hist', 'diag': 'hist', 'lab': 'evidence', 'img': 'evidence',
       'med': 'treat', 'score': 'severity', 'sev': 'severity'}
MAXD = 5
SEV_COLS = list(range(24, 29))


def prep(X, tr):
    m = np.nanmedian(X[tr], 0); m[np.isnan(m)] = 0.0
    s = np.nanstd(X[tr], 0); s[np.isnan(s) | (s < 1e-6)] = 1.0
    return (np.where(np.isnan(X), m, X) - m) / s


def build_blocks(ds, drop_types=frozenset()):
    pids = ds['pids']
    ent_use = [e for e in ds['ent2id'] if e[0] not in drop_types]
    vid = {e: i for i, e in enumerate(ent_use)}
    G, E = [], []
    for pid in pids:
        by_day = {d: {dm: set() for dm in DIMS} for d in range(MAXD)}
        ent = np.zeros(len(vid), float)
        for d, es in ds['day_ent_data'].get(pid, []):
            if d >= MAXD:
                continue
            for e in es:
                if e in vid:
                    ent[vid[e]] = 1.0
                dm = P2D.get(e[0])
                if dm:
                    by_day[d][dm].add(e)
        f = []
        for dm in DIMS:
            cnt = np.array([len(by_day[d][dm]) for d in range(MAXD)], float)
            act = np.where(cnt > 0)[0]
            f += [float(len(set().union(*[by_day[d][dm] for d in range(MAXD)]))), float(act.size)]
            f += [float(act[0]), float(act[-1]), float(cnt[act].mean())] if act.size else [0.0, 0.0, 0.0]
            f += cnt.tolist()
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = DIMS[i], DIMS[j]
                co = [1.0 if by_day[d][a] and by_day[d][b] else 0.0 for d in range(MAXD)]
                f += [sum(co)] + co
        G.append(np.array(f, float)); E.append(ent)
    return np.array(ds['X_struct'], float), np.stack(G), np.stack(E)


def task2(ds):
    y3 = np.array([m['label3'] for m in ds['meta']])
    y = (y3 > 0).astype(int)
    S, G, E = build_blocks(ds)
    arms = {'S': S, 'S+GRAPH': np.concatenate([S, G], 1), 'S+GRAPH+ENT': np.concatenate([S, G, E], 1)}
    out = {}
    for k, X in arms.items():
        au, ap, se, sp = [], [], [], []
        for seed in SEEDS:
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                Xs = prep(X, tr)
                m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9,
                                  colsample_bytree=0.9, random_state=seed, n_jobs=4, eval_metric='logloss')
                m.fit(Xs[tr], y[tr])
                p = m.predict_proba(Xs[te])[:, 1]; pr = (p >= 0.5).astype(int)
                tn, fp, fn, tp = confusion_matrix(y[te], pr, labels=[0, 1]).ravel()
                au.append(roc_auc_score(y[te], p)); ap.append(average_precision_score(y[te], p))
                se.append(tp / max(tp + fn, 1)); sp.append(tn / max(tn + fp, 1))
        out[k] = dict(auc=round(float(np.mean(au)), 4), auc_sd=round(float(np.std(au)), 4),
                      auprc=round(float(np.mean(ap)), 4), sens=round(float(np.mean(se)), 4),
                      spec=round(float(np.mean(sp)), 4))
        print('Task2 %-13s AUC=%.4f(±%.3f) AUPRC=%.4f Sens=%.4f Spec=%.4f' %
              (k, out[k]['auc'], out[k]['auc_sd'], out[k]['auprc'], out[k]['sens'], out[k]['spec']), flush=True)
    return out


def task3(ds):
    tgt = np.array([m['days'] if m['days'] is not None else np.nan for m in ds['meta']], float)
    ok = ~np.isnan(tgt)
    out = {}
    for tag, drop, cols in [('circular', frozenset(), None), ('fixed', frozenset({'score', 'sev'}), SEV_COLS)]:
        S, G, E = build_blocks(ds, drop)
        if cols is not None:
            S = np.delete(S, cols, axis=1)
        X = np.concatenate([S, G, E], 1)
        mae, rmse, r2, pr = [], [], [], []
        for seed in SEEDS:
            for tr, te in KFold(5, shuffle=True, random_state=seed).split(X[ok]):
                itr = np.where(ok)[0][tr]; ite = np.where(ok)[0][te]
                Xs = prep(X, itr)
                m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                                 colsample_bytree=0.9, random_state=seed, n_jobs=4)
                m.fit(Xs[itr], tgt[itr])
                p = m.predict(Xs[ite])
                mae.append(mean_absolute_error(tgt[ite], p))
                rmse.append(np.sqrt(mean_squared_error(tgt[ite], p)))
                r2.append(r2_score(tgt[ite], p))
                if np.std(p) > 1e-9:
                    pr.append(pearsonr(tgt[ite], p)[0])
        out[tag] = dict(mae=round(float(np.mean(mae)), 4), rmse=round(float(np.mean(rmse)), 4),
                        r2=round(float(np.mean(r2)), 4), pearson=round(float(np.mean(pr)), 4),
                        n_features=int(X.shape[1]))
        print('Task3 %-9s MAE=%.4f RMSE=%.4f R2=%.4f r=%.4f (feat %d)' %
              (tag, out[tag]['mae'], out[tag]['rmse'], out[tag]['r2'], out[tag]['pearson'],
               out[tag]['n_features']), flush=True)
    return out


if __name__ == '__main__':
    ds = pickle.load(open(DS, 'rb'))
    res = {'task2': task2(ds), 'task3': task3(ds), 'cscs_range': [float(np.nanmin([m['days'] for m in ds['meta']])),
                                                                  float(np.nanmax([m['days'] for m in ds['meta']]))]}
    json.dump(res, open(f'{OUT}/_task_fix_probe.json', 'w'), ensure_ascii=False, indent=2)
    print('saved _task_fix_probe.json')
