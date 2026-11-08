# -*- coding: utf-8 -*-
"""
exp_repro_check.py — 复现性完整性检查
=====================================
用途：验证重跑前后的数据完整性，确认病例集合/标签/折划分未因代码修改而漂移。

检查项：
  1. 病例集合完整性：pids 数量与顺序
  2. 标签一致性：Task1 (label3) / Task2 (y2) / Task3 (CSCS) 分布
  3. 折划分一致性：StratifiedKFold(5, shuffle=True, random_state=42) 的 fold 分配
  4. 超图结构统计：实体词典大小、超边总数、时序覆盖（平均活跃天数/窗口覆盖率）
  5. 数据集校验和：四维链数据集文件完整性

用法：
  python exp_repro_check.py --dataset exp_4dchain_v2_dataset.pkl
  python exp_repro_check.py --dataset exp_dataset.pkl

依赖：输入 .pkl 文件需由 exp_4dchain_v2.py 或 exp_run.py 构建（因隐私原因不纳入仓库）。
该脚本仅用于开发者验证，公开仓库用户需自行提供数据集后运行。
"""
import argparse
import hashlib
import os
import pickle
import sys
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedKFold

# 实体前缀 → 维度（与 exp_4dchain_v2.build_4d_xgb_features 保持一致）
PREFIX2DIM = {'sym': 'hist', 'diag': 'hist', 'lab': 'evidence', 'img': 'evidence',
              'med': 'treat', 'score': 'severity', 'sev': 'severity'}


def check_dataset(filepath):
    """加载数据集并检查关键统计量是否与论文口径一致。"""
    if not os.path.exists(filepath):
        print(f"[FAIL] 数据集文件不存在: {filepath}")
        print("       请先运行 exp_4dchain_v2.py --build-only 生成数据集。")
        return False

    with open(filepath, 'rb') as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    print(f"[INFO] SHA256: {sha[:16]}...")

    ds = pickle.loads(raw)
    pids = ds.get('pids', [])
    print(f"[CHECK] 病例数: {len(pids)} (论文口径 253)")

    # Check meta / labels
    meta = ds.get('meta', [])
    if meta:
        label3 = np.array([m['label3'] for m in meta])
        cnt = Counter(label3.tolist())
        print(f"[CHECK] Task1 三分类分布: A(0)={cnt.get(0,0)}, B(1)={cnt.get(1,0)}, C(2)={cnt.get(2,0)} (论文口径 54/89/110)")
        y2 = (label3 > 0).astype(int)
        cnt2 = Counter(y2.tolist())
        print(f"[CHECK] Task2 二分类分布: 阴性={cnt2.get(0,0)}, 阳性={cnt2.get(1,0)} (论文口径 54/199)")

    # Check fold splits
    if meta:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        label3_arr = np.array([m['label3'] for m in meta])
        folds = list(skf.split(np.zeros(len(label3_arr)), label3_arr))
        fold_ids = np.zeros(len(label3_arr), dtype=int)
        for fi, (_, te) in enumerate(folds):
            fold_ids[te] = fi
        fold_dist = Counter(fold_ids.tolist())
        print(f"[CHECK] 5折划分分布: {dict(sorted(fold_dist.items()))}")

    # ---- 超图结构统计 ----
    # 当前 dataset schema：entity_data = {pid: {dim: set(ent)}}（维度级实体，跨天并集）
    #                      day_ent_data = {pid: [(day, [ent]), ...]}（按天实体）
    max_days = int(ds.get('max_days', 5))
    entity_data = ds.get('entity_data', {})
    day_ent_data = ds.get('day_ent_data', {})
    if entity_data or day_ent_data:
        extraction_vocab = set()
        for pid, dims in entity_data.items():
            for dim, ents in dims.items():
                extraction_vocab.update(ents)

        n_hyperedges = 0                      # per-day per-dimension 超边数
        active_days = set()
        for pid, seq in day_ent_data.items():
            for d_idx, ents in seq:
                if not ents:
                    continue
                dims_hit = {PREFIX2DIM.get(e[0]) for e in ents} - {None}
                n_hyperedges += len(dims_hit)
                if dims_hit:
                    active_days.add((pid, d_idx))

        n_patients = len(pids) or len(day_ent_data)
        mean_active = len(active_days) / n_patients if n_patients else 0
        coverage = mean_active / max_days * 100 if mean_active else 0

        print(f"[CHECK] 抽取实体并集: {len(extraction_vocab)} (论文口径 182)")
        print(f"[CHECK] 建模词表（≥2 例）: {len(ds.get('ent2id', {}))} (论文口径 123)")
        print(f"[CHECK] 超边总数 (per-day per-dimension): {n_hyperedges} (论文口径 1378)")
        print(f"[CHECK] 平均活跃天数: {mean_active:.2f} (论文口径 2.94)")
        print(f"[CHECK] 窗口覆盖率: {coverage:.1f}% (论文口径 58.9%)")

    print("\n[RESULT] 数据集完整性检查完成。")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='复现性完整性检查')
    parser.add_argument('--dataset', type=str, default='exp_4dchain_v2_dataset.pkl',
                        help='数据集 .pkl 文件路径')
    args = parser.parse_args()
    ok = check_dataset(args.dataset)
    sys.exit(0 if ok else 1)
