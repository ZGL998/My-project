# -*- coding: utf-8 -*-
"""
超边族消融 + 结构的模型无关分组检验（10 seeds）
================================================
问题：论文的结构增益，是否真的来自「四维时序超图」的**结构本身**，
      而不是「特征更多」或「实体词表更大」？

做法（同构建器 / 同读出头 / 同折，唯一变量 = 结构块的哪一族）：
  - 读出头固定为 XGBoost；折为 StratifiedKFold(5, shuffle=True)，种子取自 SEEDS（10 个）
  - 把 4D 结构块按**超边族**拆开做增量消融
  - 再对结构量做**模型无关**的 Kruskal-Wallis 分组检验（A/B/C 三组）

结构块 76 维（由「维度超边」的逐日/跨天统计构成）：
  CARD   8 维 = 4 维 × 2（实体并集规模、活跃天数）
  TEMP  32 维 = 4 维 × 8（首活跃日、末活跃日、活跃日均规模、逐日命中数 C0..C4）
                → **时序超边**（跨天聚合）
  CROSS 36 维 = 6 个维度对 × 6（共激活天数 + 逐日共激活 0/1）
                → **跨维耦合超边**

消融臂：
  S(29) / S+CARD(37) / S+TEMP(61) / S+CROSS(65)
  / S+GRAPH(105) / S+ENT(152) / S+GRAPH+ENT(228)

输出：$HG_OUT_DIR/_hyperedge_ablation.json

用法：
  EXPERIMENT_BASE=/path/to/exp DS4_PATH=/path/to/exp_4dchain_v2_dataset.pkl \
  HG_OUT_DIR=/path/to/out python exp_hyperedge_family_ablation.py
"""
import json
import os
import pickle
from itertools import combinations

import numpy as np
from scipy.stats import kruskal
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

BASE = os.environ.get('EXPERIMENT_BASE', os.getcwd())
OUT = os.environ.get('HG_OUT_DIR', f'{BASE}/probes')
DS = os.environ.get('DS4_PATH', f'{BASE}/exp_4dchain_v2_dataset.pkl')
SEEDS = [42, 123, 456, 7, 13, 99, 202, 404, 777, 1234]   # 10-seed 协议
NBOOT = 2000
DIMS = ['hist', 'evidence', 'treat', 'severity']
P2D = {'sym': 'hist', 'diag': 'hist', 'lab': 'evidence', 'img': 'evidence',
       'med': 'treat', 'score': 'severity', 'sev': 'severity'}
MAXD = 5


def prep(X, tr):
    """训练折内标准化（median/std），避免测试折信息泄漏。"""
    m = np.nanmedian(X[tr], 0); m[np.isnan(m)] = 0.0
    s = np.nanstd(X[tr], 0); s[np.isnan(s) | (s < 1e-6)] = 1.0
    return (np.where(np.isnan(X), m, X) - m) / s


def build_blocks(ds):
    """
    返回 (S, G, E, IDX)：
      S   : 结构化特征 X_struct（29 维）
      G   : 4D 结构块（76 维） = CARD(8) + TEMP(32) + CROSS(36)
      E   : 实体存在向量（词表大小维）
      IDX : {'CARD'/'TEMP'/'CROSS'} → G 的列索引
    """
    pids = ds['pids']
    ent_use = list(ds['ent2id'])
    vid = {e: i for i, e in enumerate(ent_use)}
    G, E = [], []
    IDX_CARD, IDX_TEMP, IDX_CROSS = [], [], []
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
    # 族索引：每维 10 列 = [union, n_act] + [first, last, mean, C0..C4]
    pos = 0
    for _ in DIMS:
        IDX_CARD += [pos, pos + 1]
        IDX_TEMP += [pos + 2, pos + 3, pos + 4, pos + 5, pos + 6, pos + 7, pos + 8, pos + 9]
        pos += 10
    for _ in combinations(range(4), 2):
        IDX_CROSS += list(range(pos, pos + 6)); pos += 6
    assert pos == 76, pos
    return (np.array(ds['X_struct'], float), np.stack(G), np.stack(E),
            {'CARD': IDX_CARD, 'TEMP': IDX_TEMP, 'CROSS': IDX_CROSS})


def run_arm(X, y3, y2, seeds=SEEDS):
    """逐折算指标后取均值/标准差（T1: AUC-ovr + macro-F1；T2: AUC）。"""
    auc, mf1, auc2 = [], [], []
    for seed in seeds:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y3):
            Xs = prep(X, tr)
            m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9,
                              colsample_bytree=0.9, random_state=seed, n_jobs=4, eval_metric='mlogloss')
            m.fit(Xs[tr], y3[tr])
            prob = m.predict_proba(Xs[te])
            auc.append(roc_auc_score(y3[te], prob, multi_class='ovr', average='macro', labels=[0, 1, 2]))
            mf1.append(f1_score(y3[te], m.predict(Xs[te]), average='macro'))
            # Task2：A/B/C>0 的二分类
            mb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9,
                               colsample_bytree=0.9, random_state=seed, n_jobs=4, eval_metric='logloss')
            mb.fit(Xs[tr], y2[tr])
            auc2.append(roc_auc_score(y2[te], mb.predict_proba(Xs[te])[:, 1]))
    return (dict(t1_auc=round(float(np.mean(auc)), 4), t1_auc_sd=round(float(np.std(auc)), 4),
                 t1_mf1=round(float(np.mean(mf1)), 4), t1_mf1_sd=round(float(np.std(mf1)), 4),
                 t2_auc=round(float(np.mean(auc2)), 4), t2_auc_sd=round(float(np.std(auc2)), 4)),
            np.array(mf1))


def boot_ci(a, b, n=NBOOT, seed=20260927):
    """配对 bootstrap 95%CI（对逐折差值重采样）。"""
    d = np.asarray(a) - np.asarray(b)
    rs = np.random.RandomState(seed)
    bs = [d[rs.randint(0, len(d), len(d))].mean() for _ in range(n)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def structure_by_class(ds, y3):
    """模型无关：结构量在 A/B/C 三组的分布（Kruskal-Wallis）。"""
    per = {k: [] for k in ['card_mean', 'card_max', 'n_hyperedge', 'span_mean', 'cross_rate']}
    for pid in ds['pids']:
        ed = ds['entity_data'].get(pid, {})
        day_dim = {d: {dm: set() for dm in DIMS} for d in range(MAXD)}
        for d, es in ds['day_ent_data'].get(pid, []):
            if d >= MAXD:
                continue
            for e in es:
                dm = P2D.get(e[0])
                if dm:
                    day_dim[d][dm].add(e)
        # 维度超边（跨天聚合）的规模：每维实体并集
        dim_union = [len(set().union(*[day_dim[d][dm] for d in range(MAXD)])) for dm in DIMS]
        # 超边数：非空 (天, 维) 超边个数
        n_he = sum(1 for d in range(MAXD) for dm in DIMS if day_dim[d][dm])
        # 时序跨度：各活跃维 (末-首+1) 之和 / 4
        spans = []
        for dm in DIMS:
            act = [d for d in range(MAXD) if day_dim[d][dm]]
            if act:
                spans.append(act[-1] - act[0] + 1)
        # 跨维共激活率：6 对维度对的共激活天数占比均值
        cross = np.mean([sum(1 for d in range(MAXD) if day_dim[d][a] and day_dim[d][b]) / MAXD
                         for a, b in combinations(DIMS, 2)])
        per['card_mean'].append(float(np.mean(dim_union)))
        per['card_max'].append(float(np.max(dim_union)))
        per['n_hyperedge'].append(float(n_he))
        per['span_mean'].append(float(sum(spans) / len(DIMS)))
        per['cross_rate'].append(round(float(cross), 4))
    out = {}
    for k, v in per.items():
        v = np.asarray(v, float)
        g = [v[y3 == c] for c in (0, 1, 2)]
        out[k] = dict(A=round(float(g[0].mean()), 4), B=round(float(g[1].mean()), 4),
                      C=round(float(g[2].mean()), 4), kruskal_p=round(float(kruskal(*g).pvalue), 6))
    return out


def main():
    ds = pickle.load(open(DS, 'rb'))
    y3 = np.array([m['label3'] for m in ds['meta']])
    y2 = (y3 > 0).astype(int)
    S, G, E, IDX = build_blocks(ds)
    print('loaded %s pids=%d V=%d' % (DS, len(ds['pids']), len(ds['ent2id'])), flush=True)
    print('GRAPH 维数=%d | CARD=%d TEMP=%d CROSS=%d' %
          (G.shape[1], len(IDX['CARD']), len(IDX['TEMP']), len(IDX['CROSS'])), flush=True)

    arms = {
        'S': S,
        'S+CARD': np.concatenate([S, G[:, IDX['CARD']]], 1),
        'S+TEMP': np.concatenate([S, G[:, IDX['TEMP']]], 1),
        'S+CROSS': np.concatenate([S, G[:, IDX['CROSS']]], 1),
        'S+GRAPH': np.concatenate([S, G], 1),
        'S+ENT': np.concatenate([S, E], 1),
        'S+GRAPH+ENT': np.concatenate([S, G, E], 1),
    }
    print('arms:', {k: v.shape for k, v in arms.items()}, flush=True)

    res, mf1 = {}, {}
    for k, X in arms.items():
        r, m = run_arm(X, y3, y2)
        r['n_feat'] = int(X.shape[1]); res[k] = r; mf1[k] = m
        print('ARM %-14s T1auc=%.4f(±%.3f) T1mf1=%.4f T2auc=%.4f  (d=%d)' %
              (k, r['t1_auc'], r['t1_auc_sd'], r['t1_mf1'], r['t2_auc'], r['n_feat']), flush=True)

    full = 'S+GRAPH+ENT'
    diffs = {}
    for k in ['S', 'S+ENT', 'S+CARD', 'S+TEMP', 'S+CROSS']:
        ci = boot_ci(mf1[full], mf1[k])
        diffs['%s_minus_%s' % (full, k)] = ci
        print('DIFF %s - %-14s T1macroF1 95%%CI [%.4f, %.4f]' % (full, k, ci[0], ci[1]), flush=True)

    struct = structure_by_class(ds, y3)
    for k, v in struct.items():
        print('STRUCT %-12s A=%.4f B=%.4f C=%.4f  p=%.6f' % (k, v['A'], v['B'], v['C'], v['kruskal_p']),
              flush=True)

    out = dict(ablation=res, diff_ci_t1_macrof1=diffs, structure_by_class=struct, seeds=SEEDS)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f'{OUT}/_hyperedge_ablation.json', 'w'), ensure_ascii=False, indent=2)
    print('saved', f'{OUT}/_hyperedge_ablation.json')


if __name__ == '__main__':
    main()
