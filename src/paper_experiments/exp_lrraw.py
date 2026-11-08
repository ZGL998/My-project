
# -*- coding: utf-8 -*-
"""
import os
exp_lrraw.py — LR-raw 基线（不经 LLM 的规则抽取结构化特征）

与四维链实验严格对齐：
  - 同 251 例（规则特征经 SHA1 源文件名映射到 CASE_xxx 病例编号，覆盖 251/251）
  - 同 seed=42 的 StratifiedKFold 5 折分层划分
  - 同 exp_run.xgb_train_fit 训练配置（random_state=42）
  - 特征模式与 4dchain X_struct 同构（24 列：age/sex/smoking/合并症数/7检验/4生命体征/9用药类）
    —— 区别仅在抽取来源：LR-raw=规则正则抽取，LR/XGBoost=四维链 LLM 抽取
  - Task3 回归目标：CSCS 复合评分（v3 起；数据集读 exp_4dchain_v2_dataset.pkl，其 meta['days'] 即 CSCS）

用法：
  python3 exp_lrraw.py
"""
import hashlib
import json
import pickle
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

BASE = os.environ.get('EXPERIMENT_BASE', os.getcwd())
sys.path.insert(0, BASE)

import exp_run as ER  # noqa: E402


def case_id(pid):
    stem = str(pid).rsplit('.', 1)[0]
    return 'CASE_' + hashlib.sha1(stem.encode('utf-8')).hexdigest()[:12].upper()


def main():
    ds = pickle.load(open(f'{BASE}/exp_dataset.pkl', 'rb'))
    d4 = pickle.load(open(f'{BASE}/exp_4dchain_v2_dataset.pkl', 'rb'))

    # 注意：exp_dataset.pkl 中 X_struct/meta 按字典插入序追加，与排序后的 pids 不对齐（历史 bug），
    # 因此必须用 meta['pid'] 建立"行 ↔ 病例"映射，不能用 ds['pids'][i] 索引行。
    row_by_rulepid = {m['pid']: k for k, m in enumerate(ds['meta'])}
    rulepid_by_case = {case_id(p): p for p in ds['pids']}
    rows = [row_by_rulepid[rulepid_by_case[pid]] for pid in d4['pids']]
    X = ds['X_struct'][rows]
    assert X.shape[0] == len(d4['pids']) == 251

    meta = d4['meta']
    label3 = np.array([m['label3'] for m in meta])
    y2 = (label3 > 0).astype(int)
    days = np.nan_to_num(np.array([float(m['days']) if m['days'] else np.nan for m in meta]))

    # 对齐自检：修正后规则侧 label3 应与 4dchain 完全一致
    mism = sum(1 for r, pid in zip(rows, d4['pids']) if ds['meta'][r]['label3'] != d4['meta'][d4['pids'].index(pid)]['label3'])
    print(f'对齐自检 label 不一致数: {mism} / 251', flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {'cls': [], 'bin': [], 'reg': []}
    fold_preds = {'cls_y': [], 'cls_p': [], 'bin_y': [], 'bin_p': [], 'reg_y': [], 'reg_p': []}
    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X, label3)):
        Xs = ER.prep_struct(X, tr_idx)
        Xs_tr, Xs_te = Xs[tr_idx], Xs[te_idx]
        # ---- Task1 三分类 ----
        m = ER.xgb_train_fit(Xs_tr, label3[tr_idx], 'cls')
        prob = m.predict_proba(Xs_te); pred = prob.argmax(1)
        results['cls'].append(ER.metrics_cls(label3[te_idx], pred))
        fold_preds['cls_y'].extend([int(x) for x in label3[te_idx]])
        fold_preds['cls_p'].extend([int(x) for x in pred])
        # ---- Task2 恶化二分类 ----
        m2 = ER.xgb_train_fit(Xs_tr, y2[tr_idx], 'bin')
        prob2 = m2.predict_proba(Xs_te)[:, 1]
        results['bin'].append(ER.metrics_bin(y2[te_idx], prob2))
        fold_preds['bin_y'].extend([int(x) for x in y2[te_idx]])
        fold_preds['bin_p'].extend([float(x) for x in prob2])
        # ---- Task3 CSCS 回归 ----
        m3 = ER.xgb_train_fit(Xs_tr, days[tr_idx], 'reg')
        pred3 = m3.predict(Xs_te)
        results['reg'].append(ER.metrics_reg(days[te_idx], pred3))
        fold_preds['reg_y'].extend([float(x) for x in days[te_idx]])
        fold_preds['reg_p'].extend([float(x) for x in pred3])
        print(f'折{fold_i + 1} 完成', flush=True)

    out = {k: {m_: float(np.mean([f_[m_] for f_ in v])) for m_ in v[0]} for k, v in results.items()}
    json.dump(out, open(f'{BASE}/exp_results_lrraw.json', 'w'), ensure_ascii=False, indent=2)
    fold_out = {
        'cls': [{k: v[k] for k in ('macro_f1',)} for v in results['cls']],
        'bin': [{k: v[k] for k in ('auc_roc',)} for v in results['bin']],
        'reg': [{k: v[k] for k in ('pearson_r',)} for v in results['reg']],
    }
    json.dump(fold_out, open(f'{BASE}/exp_lrraw_folds.json', 'w'), ensure_ascii=False, indent=2)
    pickle.dump(fold_preds, open(f'{BASE}/exp_lrraw_foldpreds.pkl', 'wb'))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
