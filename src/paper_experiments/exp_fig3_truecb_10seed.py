# -*- coding: utf-8 -*-
"""
corrected component-blind · 10 种子 Task3 回归 OOF 导出
========================================================
用途：生成 Figure 3 需要的 253 个 (真实 CSCS, 预测 CSCS) 点对。

协议（corrected component-blind，177 维）：
  - X_struct 去掉第 24-28 列（sev_FEV1_pct / sev_GOLD / sev_mMRC / sev_SpO2_min / sev_PaCO2_hi）→ 24 维
  - 实体 one-hot 去掉全部 score/sev 实体 → 105 维
  - 图统计去掉 28 个 severity 衍生统计量 → 48 维
  - 合计 24 + 105 + 48 = 177 维

评估：KFold(5, shuffle=True, random_state=seed) × 10 种子 [42,123,456,7,13,99,202,404,777,1234]
      每名患者在其处于测试折时得到一次 OOF 预测；10 种子按患者求平均得到池化预测。

产出：
  - fig3_truecb_oof_10seed.npz  （逐种子 OOF：seeds / actual / oof(10,253)）
  - fig3_scatter_points.csv     （253 行，actual_cscs / predicted_cscs）
"""
import os
import pickle
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

BASE = '/home/pumc/呼吸疾病/超图实验'
DS_PATH = f'{BASE}/hg_fix_run/exp_4dchain_v2_dataset_comorb_fixed.pkl'  # 合并症修复后 CSCS 目标
SEEDS = [42, 123, 456, 7, 13, 99, 202, 404, 777, 1234]
DIMS = ['hist', 'evidence', 'treat', 'severity']
P2D = {'sym': 'hist', 'diag': 'hist', 'lab': 'evidence', 'img': 'evidence',
       'med': 'treat', 'score': 'severity', 'sev': 'severity'}
MAXD = 5
SEV_COLS = list(range(24, 29))   # X_struct 中被扣的 5 个 severity 结构化列


def prep(X, tr):
    """训练折内 median/std 标准化，避免测试折信息泄漏。"""
    m = np.nanmedian(X[tr], 0); m[np.isnan(m)] = 0.0
    s = np.nanstd(X[tr], 0); s[np.isnan(s) | (s < 1e-6)] = 1.0
    return (np.where(np.isnan(X), m, X) - m) / s


def build_blocks(ds, drop_types=frozenset()):
    """与 exp_task_fix_probe.build_blocks 完全一致：返回 (S, G, E)。

    注意：G 仍会包含 severity 图统计（by_day 未按 drop_types 过滤），
    因此 corrected 口径需在外部额外剔除 severity 图统计列（见 sev_graph_indices）。
    """
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


def sev_graph_indices():
    """G 中 28 个 severity 衍生图统计的列索引。

    G 布局（76 列）：每维 10 列 [union, n_active, first, last, mean, cnt_d0..d4]
    → severity 维 = 30..39；交叉对按 combinations(DIMS,2) 顺序：
    (hist,evidence)=40..45 (hist,treat)=46..51 (hist,severity)=52..57
    (evidence,treat)=58..63 (evidence,severity)=64..69 (treat,severity)=70..75
    """
    idx = list(range(30, 40))   # severity 单维 10 列
    idx += list(range(52, 58))  # hist×severity
    idx += list(range(64, 70))  # evidence×severity
    idx += list(range(70, 76))  # treat×severity
    return idx


def _r2(a, b):
    return float(1.0 - np.sum((a - b) ** 2) / np.sum((a - np.mean(a)) ** 2))


def pooled_metrics(y, pred):
    return dict(
        r2=round(_r2(y, pred), 4),
        r=round(float(np.corrcoef(y, pred)[0, 1]), 4),
        mae=round(float(np.mean(np.abs(y - pred))), 4),
        rmse=round(float(np.sqrt(np.mean((y - pred) ** 2))), 4),
    )


def run_oof(X, y, seeds):
    oof = np.zeros((len(seeds), len(y)), float)
    for si, seed in enumerate(seeds):
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
            Xs = prep(X, tr)
            m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9, random_state=seed, n_jobs=4)
            m.fit(Xs[tr], y[tr])
            oof[si, te] = m.predict(Xs[te])
        print(f'  seed {seed} done', flush=True)
    return oof


def main():
    ds = pickle.load(open(DS_PATH, 'rb'))
    S, G, E = build_blocks(ds, frozenset({'score', 'sev'}))
    S = np.delete(S, SEV_COLS, axis=1)          # 24
    G = np.delete(G, sev_graph_indices(), axis=1)  # 48
    X = np.concatenate([S, G, E], 1)            # 177
    y = np.array([m['days'] for m in ds['meta']], float)

    print(f'X={X.shape} (S{len(S[0])}+G{G.shape[1]}+E{E.shape[1]})  y={y.shape}  CSCS mean={y.mean():.4f}', flush=True)
    assert X.shape[1] == 177, X.shape

    print('10 种子 OOF 中...', flush=True)
    oof = run_oof(X, y, SEEDS)

    pred10 = oof.mean(axis=0)
    pred3 = oof[:3].mean(axis=0)

    print('\n=== 3 种子池化（对照现有 fig3_predicted_actual_trueCB.csv）===')
    print(pooled_metrics(y, pred3))
    print('\n=== 10 种子池化（Figure 3 用）===')
    print(pooled_metrics(y, pred10))

    slope, intercept = np.polyfit(y, pred10, 1)
    print(f'\n线性拟合: slope={slope:.4f} intercept={intercept:.4f}')

    # 逐种子 OOF 存档
    np.savez(f'{BASE}/fig3_truecb_oof_10seed.npz',
             seeds=np.array(SEEDS), actual=y, oof=oof)
    # 池化 CSV
    pd.DataFrame({'actual_cscs': y, 'predicted_cscs': pred10}).to_csv(
        f'{BASE}/fig3_scatter_points.csv', index=False)
    print(f'\nsaved fig3_truecb_oof_10seed.npz + fig3_scatter_points.csv ({len(y)} rows)')


if __name__ == '__main__':
    main()
