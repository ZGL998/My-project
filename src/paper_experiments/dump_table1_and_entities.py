# -*- coding: utf-8 -*-
"""导出 Table 1（253 例队列特征）+ 122 实体分布（干净口径）"""
import pickle
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

DS = '/home/pumc/呼吸疾病/超图实验/hg_clean_run/exp_4dchain_v2_dataset.pkl'
ds = pickle.load(open(DS, 'rb'))
meta = ds['meta']
X = np.array(ds['X_struct'], float)
y = np.array([m['label3'] for m in meta])
pids = ds['pids']
GL = {0: 'A 稳定', 1: 'B 急性加重', 2: 'C 合并感染'}

COLS = (['age', 'sex', 'smoking', 'comorb_cnt']
        + ['CRP', 'PCT', 'WBC', 'NEUT', 'HGB', 'ALB', 'PLT']
        + ['体温', '脉搏', '呼吸', '收缩压']
        + ['抗生素', '激素', '支气管扩张剂', '祛痰药', '抗凝药',
           '利尿剂', '血管活性药', '氧疗/通气', '营养支持']
        + ['FEV1_pct', 'GOLD', 'mMRC', 'SpO2_min', 'PaCO2_hi'])
CAT = set(range(15, 24)) | {1, 2, 28}   # 二值/计数列 + sex/smoking/PaCO2_hi
print('列数 =', X.shape[1], '| 名称数 =', len(COLS))

print('\n===== 分组 =====')
for g in (0, 1, 2):
    print('  %-12s n=%d' % (GL[g], int((y == g).sum())))
print('  合计 n=%d' % len(y))

print('\n===== 队列特征（by group） =====')
print('| 变量 | 可用 n (pct) | A 稳定 (n=%d) | B 急性加重 (n=%d) | C 合并感染 (n=%d) | p |'
      % tuple(int((y == g).sum()) for g in (0, 1, 2)))
print('|---|---|---|---|---|---|')


def fmt(v):
    return '%.1f ± %.1f' % (np.mean(v), np.std(v, ddof=1))


for ci, name in enumerate(COLS):
    col = X[:, ci]
    ok = ~np.isnan(col)
    avail = '%d (%.1f%%)' % (ok.sum(), 100 * ok.mean())
    if ci in CAT:
        cells = []
        for g in (0, 1, 2):
            c = col[(y == g) & ok]
            cells.append('%d (%.1f%%)' % ((c > 0).sum(), 100 * (c > 0).mean()) if len(c) else '—')
        try:
            tab = [[(col[(y == g) & ok] > 0).sum() for g in (0, 1, 2)],
                   [(col[(y == g) & ok] <= 0).sum() for g in (0, 1, 2)]]
            p = stats.chi2_contingency(tab)[1]
        except Exception:
            p = float('nan')
    else:
        cells = []
        for g in (0, 1, 2):
            c = col[(y == g) & ok]
            cells.append(fmt(c) if len(c) > 1 else '—')
        grp = [col[(y == g) & ok] for g in (0, 1, 2)]
        try:
            p = stats.kruskal(*[g for g in grp if len(g) > 1])[1]
        except Exception:
            p = float('nan')
    print('| %s | %s | %s | %s | %s | %s |' % (name, avail, cells[0], cells[1], cells[2],
                                               ('%.3f' % p) if p == p else '—'))

# 结局变量
print('\n===== 结局变量 =====')
cscs = np.array([m['days'] for m in meta], float)
hd = np.array([m.get('hospital_days', np.nan) for m in meta], float)
for nm, v in (('CSCS', cscs), ('住院天数', hd)):
    print('| %s | %.4f ± %.4f (min %.2f, max %.2f) |' % (nm, np.nanmean(v), np.nanstd(v, ddof=1),
                                                          np.nanmin(v), np.nanmax(v)))
print('\nCSCS 分组:')
for g in (0, 1, 2):
    v = cscs[y == g]
    print('  %-12s %.4f ± %.4f' % (GL[g], v.mean(), v.std(ddof=1)))

# ---------------- 实体分布 ----------------
print('\n===== 实体分布（干净口径，词表 >=2，diag 已剔除） =====')
ent2id = ds['ent2id']
print('实体词表大小 =', len(ent2id))

by_type, by_dim = Counter(), Counter()
per_pat = []
DIM_OF = {}
for pid, ed in ds['entity_data'].items():
    ids = set()
    for dm, es in ed.items():
        for e in es:
            if e in ent2id:
                by_type[e[0]] += 1
                by_dim[dm] += 1
                DIM_OF.setdefault(dm, Counter())[e[0]] += 1
                ids.add(e)
    per_pat.append(sum(1 for _, es in ed.items() for e in es if e in ent2id))

vocab_by_type = Counter(e[0] for e in ent2id)
print('\n按实体类型（词表数 / 出现次数）:')
for t in sorted(vocab_by_type, key=lambda x: -vocab_by_type[x]):
    print('  %-6s 词表 %3d | 出现 %5d' % (t, vocab_by_type[t], by_type[t]))
print('  合计   词表 %3d | 出现 %5d' % (sum(vocab_by_type.values()), sum(by_type.values())))

print('\n按四维:')
for dm in ('hist', 'evidence', 'treat', 'severity'):
    sub = DIM_OF.get(dm, Counter())
    print('  %-9s 出现 %5d | 构成 %s' % (dm, by_dim[dm],
                                        ', '.join('%s=%d' % (k, v) for k, v in sub.most_common())))

print('\n每例实体数: mean %.1f, median %.1f, min %d, max %d'
      % (np.mean(per_pat), np.median(per_pat), min(per_pat), max(per_pat)))

print('\n词表 Top-25（按出现次数）:')
occ = Counter()
for pid, ed in ds['entity_data'].items():
    for es in ed.values():
        for e in es:
            if e in ent2id:
                occ[e] += 1
for e, c in occ.most_common(25):
    print('  %-28s %4d' % ('%s:%s' % (e[0], e[1]), c))
