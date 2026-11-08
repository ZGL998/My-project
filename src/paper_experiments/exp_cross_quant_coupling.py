# -*- coding: utf-8 -*-
"""
跨维耦合的定量化重构 + 配对检验（10 seeds）
============================================
动机：在超边族消融中，「跨维耦合」（CROSS）是证据最弱的一族 —— 因为它被构造成
      很粗的 0/1（某天两个维度是否**同时**有记录）。本脚本把它换成**定量耦合统计**，
      在与旧构造完全相同的构建器 / 折 / 读出头下做配对检验。

定量耦合（每个维度对 (i,j) 用 7 个统计，6 对 → 42 维）：
  n_codays   共激活天数
  Jaccard    共激活天数 / 至少一方激活的天数
  phi        逐日激活 0/1 序列的二分 φ 相关系数
  mass       共激活质量 Σ_d |E_i(d)|·|E_j(d)|
  density    mass / (|E_i|·|E_j|)，实体对密度
  corr       两维逐日计数序列的相关系数
  leadlag    corr(C_i[:-1], C_j[1:]) − corr(C_j[:-1], C_i[1:])，领先-滞后非对称性

对照臂：
  S(29) / S+CROSS_bin(65, 旧 0/1 耦合) / S+CROSS_q(71, 定量耦合)
  / S+GRAPH_old(105, 旧全结构) / S+GRAPH_q(111, CARD+TEMP+定量耦合)
  / S+GRAPH_q+ENT(234)

输出：$HG_OUT_DIR/_cross_quant.json、_cross_quant_paired.json
"""
import json
import os
import pickle
from itertools import combinations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from exp_hyperedge_family_ablation import (BASE, DS, SEEDS, DIMS, P2D, MAXD,
                                           build_blocks, prep, run_arm, boot_ci)

OUT = os.environ.get('HG_OUT_DIR', f'{BASE}/probes')


def _corr(a, b):
    """皮尔逊相关；任一方为常量时返回 0。"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def build_cross_quant(ds):
    """每对维度 7 个定量耦合统计 → (n, 42)。"""
    Q = []
    for pid in ds['pids']:
        by_day = {d: {dm: set() for dm in DIMS} for d in range(MAXD)}
        for d, es in ds['day_ent_data'].get(pid, []):
            if d >= MAXD:
                continue
            for e in es:
                dm = P2D.get(e[0])
                if dm:
                    by_day[d][dm].add(e)
        tot = {dm: set().union(*[by_day[d][dm] for d in range(MAXD)]) for dm in DIMS}
        f = []
        for i, j in combinations(range(4), 2):
            a, b = DIMS[i], DIMS[j]
            ca = np.array([len(by_day[d][a]) for d in range(MAXD)], float)
            cb = np.array([len(by_day[d][b]) for d in range(MAXD)], float)
            ni = int((ca > 0).sum()); nj = int((cb > 0).sum())
            n11 = int(((ca > 0) & (cb > 0)).sum())
            n10 = ni - n11; n01 = nj - n11; n00 = MAXD - n11 - n10 - n01
            den = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
            phi = ((n11 * n00 - n10 * n01) / den) if den > 1e-9 else 0.0
            jac = n11 / max(1, (ni + nj - n11))
            mass = float((ca * cb).sum())
            dens = mass / max(1.0, float(len(tot[a]) * len(tot[b])))
            f += [n11, jac, phi, mass, dens,
                  _corr(ca, cb),
                  _corr(ca[:4], cb[1:]) - _corr(cb[:4], ca[1:])]
        Q.append(np.array(f, float))
    return np.stack(Q)


def fold_auc(X, y3):
    """逐折 T1 AUC-ovr 序列（供配对 bootstrap）。"""
    out = []
    for seed in SEEDS:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y3):
            Xs = prep(X, tr)
            clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9,
                                colsample_bytree=0.9, random_state=seed, n_jobs=4, eval_metric='mlogloss')
            clf.fit(Xs[tr], y3[tr])
            out.append(roc_auc_score(y3[te], clf.predict_proba(Xs[te]),
                                     multi_class='ovr', average='macro', labels=[0, 1, 2]))
    return np.array(out)


def main():
    ds = pickle.load(open(DS, 'rb'))
    y3 = np.array([m['label3'] for m in ds['meta']])
    y2 = (y3 > 0).astype(int)
    S, G, E, IDX = build_blocks(ds)
    Gq = build_cross_quant(ds)
    print('loaded pids=%d V=%d' % (len(ds['pids']), len(ds['ent2id'])), flush=True)
    print('G_old=%s  G_q(cross=42)=%s' % (G.shape, Gq.shape), flush=True)

    # 旧结构块去掉「二进制跨维耦合」后的其余 40 列（CARD+TEMP），保持自然列序
    card_temp = sorted(IDX['CARD'] + IDX['TEMP'])
    arms = {
        'S': S,
        'S+CROSS_bin': np.concatenate([S, G[:, IDX['CROSS']]], 1),
        'S+CROSS_q': np.concatenate([S, Gq], 1),
        'S+GRAPH_old': np.concatenate([S, G], 1),
        'S+GRAPH_q': np.concatenate([S, G[:, card_temp], Gq], 1),
        'S+GRAPH_q+ENT': np.concatenate([S, G[:, card_temp], Gq, E], 1),
    }
    print('arms:', {k: v.shape for k, v in arms.items()}, flush=True)

    res, m = {}, {}
    for k, X in arms.items():
        r, mf1 = run_arm(X, y3, y2)
        r['n_feat'] = int(X.shape[1]); res[k] = r; m[k] = mf1
        print('ARM %-16s T1auc=%.4f(±%.3f) T1mf1=%.4f T2auc=%.4f (d=%d)' %
              (k, r['t1_auc'], r['t1_auc_sd'], r['t1_mf1'], r['t2_auc'], r['n_feat']), flush=True)

    diffs = {
        'S+CROSS_q_minus_S+CROSS_old': boot_ci(m['S+CROSS_q'], m['S+GRAPH_old']),
        'S+GRAPH_q_minus_S+GRAPH_old': boot_ci(m['S+GRAPH_q'], m['S+GRAPH_old']),
        'S+GRAPH_q+ENT_minus_S+CROSS_old': boot_ci(m['S+GRAPH_q+ENT'], m['S+GRAPH_old']),
        'S+GRAPH_q+ENT_minus_S': boot_ci(m['S+GRAPH_q+ENT'], m['S']),
    }
    for k, v in diffs.items():
        print('DIFF %-34s T1macroF1 95%%CI [%.4f, %.4f]' % (k, v[0], v[1]), flush=True)

    paired = {
        'cross_q_minus_bin_f1': boot_ci(m['S+CROSS_q'], m['S+CROSS_bin']),
        'cross_q_minus_bin_auc': boot_ci(fold_auc(arms['S+CROSS_q'], y3),
                                         fold_auc(arms['S+CROSS_bin'], y3)),
    }
    print('PAIRED q-bin f1  [%.4f, %.4f]' % tuple(paired['cross_q_minus_bin_f1']), flush=True)
    print('PAIRED q-bin auc [%.4f, %.4f]' % tuple(paired['cross_q_minus_bin_auc']), flush=True)

    names = {'S': 'S', 'S+GRAPH_old': 'S+CROSS_old', 'S+CROSS_q': 'S+CROSS_q',
             'S+GRAPH_q': 'S+GRAPH_q', 'S+GRAPH_q+ENT': 'S+GRAPH_q+ENT'}
    out = dict(arms={alias: res[k] for k, alias in names.items()},
               diffs_t1_macrof1=diffs, seeds=SEEDS)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f'{OUT}/_cross_quant.json', 'w'), ensure_ascii=False, indent=2)
    json.dump(paired, open(f'{OUT}/_cross_quant_paired.json', 'w'), ensure_ascii=False, indent=2)
    print('saved', f'{OUT}/_cross_quant.json', f'{OUT}/_cross_quant_paired.json')


if __name__ == '__main__':
    main()
