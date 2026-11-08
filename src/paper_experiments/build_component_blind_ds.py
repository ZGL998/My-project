# -*- coding: utf-8 -*-
"""
构造 component-blind 数据集（用于神经模型 Task 3 非循环评估）
==========================================================
与 exp_task_fix_probe.py 的 'fixed' 口径一致：
  - X_struct 删除第 24-28 列（FEV1%pred / GOLD / mMRC / SpO2_min / PaCO2_hi）
  - entity_data / day_ent_data 删除 ('score', *) 与 ('sev', *) 实体
输出：$CB_DST/exp_4dchain_v2_dataset.pkl（并软链 exp_dataset.pkl / exp_features.pkl）
"""
import os
import pickle
import shutil

BASE = os.environ.get('EXPERIMENT_BASE', os.getcwd())
SRC = os.environ.get('CB_SRC', f'{BASE}/hg_clean_run')
DST = os.environ.get('CB_DST', f'{BASE}/hg_cb_run')
DROP_TYPES = {'score', 'sev'}
SEV_COLS = list(range(24, 29))

os.makedirs(DST, exist_ok=True)
ds = pickle.load(open(f'{SRC}/exp_4dchain_v2_dataset.pkl', 'rb'))

import numpy as np
X = np.array(ds['X_struct'], float)
print('原 X_struct:', X.shape)
ds['X_struct'] = np.delete(X, SEV_COLS, axis=1)
print('component-blind X_struct:', ds['X_struct'].shape)

n_ent_drop = 0
new_ent = {}
for pid, ed in ds['entity_data'].items():
    d2 = {}
    for dm, es in ed.items():
        keep = {e for e in es if e[0] not in DROP_TYPES}
        n_ent_drop += len(es) - len(keep)
        d2[dm] = keep
    new_ent[pid] = d2
ds['entity_data'] = new_ent

n_day_drop = 0
new_day = {}
for pid, seq in ds['day_ent_data'].items():
    out = []
    for d, es in seq:
        keep = [e for e in es if e[0] not in DROP_TYPES]
        n_day_drop += len(es) - len(keep)
        out.append((d, keep))
    new_day[pid] = out
ds['day_ent_data'] = new_day

# seq_data 第 5 列为 SpO₂（见 build_raw_seq 注释），component-blind 下必须清零
n_seq = 0
new_seq = {}
for pid, s in ds['seq_data'].items():
    s2 = np.array(s, float).copy()
    if s2.ndim == 2 and s2.shape[1] > 5:
        n_seq += int((s2[:, 5] > 0).sum())
        s2[:, 5] = 0.0
    new_seq[pid] = s2
ds['seq_data'] = new_seq
print('seq_data SpO2 列清零: %d 个非零单元' % n_seq)

# 重算实体词表（去掉不再出现的实体）
from collections import Counter
cnt = Counter()
for pid, ed in ds['entity_data'].items():
    for es in ed.values():
        cnt.update(es)
for _, es in ds['day_ent_data'].items():
    pass
for pid, seq in ds['day_ent_data'].items():
    for _, es in seq:
        cnt.update(es)
ent2id = {e: i for i, (e, c) in enumerate(cnt.most_common()) if c >= 2}
ds['ent2id'] = ent2id
print('实体剔除: entity_data %d 个, day_ent_data %d 个' % (n_ent_drop, n_day_drop))
print('过滤后实体词表: %d' % len(ent2id))

pickle.dump(ds, open(f'{DST}/exp_4dchain_v2_dataset.pkl', 'wb'))
for f in ('exp_dataset.pkl', 'exp_features.pkl'):
    src, dst = f'{SRC}/{f}', f'{DST}/{f}'
    if os.path.exists(src) and not os.path.exists(dst):
        os.symlink(src, dst)
print('saved', f'{DST}/exp_4dchain_v2_dataset.pkl')
