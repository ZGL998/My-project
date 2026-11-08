# -*- coding: utf-8 -*-
"""
动作③（component-blind 逐特征扣留清单）+ 动作④（逐种子 McNemar 三分类）
======================================================================
数据源：
  - 数据集：hg_fix_run/exp_4dchain_v2_dataset.pkl（253 例，LABFIX）
  - 28 个 severity 衍生图统计：hg_fix_run/probes/_task3_sev_graph_leak.json（泄漏审计产物）
  - 逐种子预测：hg_fix_run/exp_foldpreds_v2_seed{seed}_4dxgb_task12.pkl（4D-XGB）
                hg_fix_run/exp_foldpreds_task12_baselines_seed{seed}.pkl（lr_xgboost / lrraw）
输出：
  - withholding_list.json
  - mcnemar_per_seed.json
"""
import json
import pickle
import numpy as np
from scipy.stats import binomtest

BASE = '/home/pumc/呼吸疾病/超图实验'
FIX = f'{BASE}/hg_fix_run'
SEEDS = [42, 123, 456, 7, 13, 99, 202, 404, 777, 1234]

# ----------------------------------------------------------------------
# 动作③  逐特征扣留清单
# ----------------------------------------------------------------------
ds = pickle.load(open(f'{FIX}/exp_4dchain_v2_dataset.pkl', 'rb'))

# 1) 被扣的 severity 结构化列（X_struct 第 24-28 列，即 sev_* 5 列）
#    struct_feature_names 定义为 [age, ..., med_nutrition] + [f'sev_{k}' for k in sev_cols]
sev_cols = ds.get('sev_cols', ['FEV1_pct', 'GOLD', 'mMRC', 'SpO2_min', 'PaCO2_hi'])
withheld_columns = [f'sev_{k}' for k in sev_cols]

# 2) 被扣的 severity 实体 = 全部 score/sev 类型实体（component-blind 物理删除 score/sev）
sev_entities = sorted([e for e in ds['ent2id'] if e[0] in ('score', 'sev')])
withheld_severity_entities = ['::'.join(e) for e in sev_entities]

# 3) 28 个 severity 衍生图统计（泄漏审计识别，直接取自审计产物）
leak = json.load(open(f'{FIX}/probes/_task3_sev_graph_leak.json', 'r', encoding='utf-8'))
severity_derived_stats = list(leak['severity_graph_features_kept'])

withholding = {
    'withheld_columns': withheld_columns,
    'withheld_severity_entities': withheld_severity_entities,
    'severity_derived_stats': severity_derived_stats,
    '_meta': {
        'n_withheld_columns': len(withheld_columns),
        'n_withheld_severity_entities': len(withheld_severity_entities),
        'n_severity_derived_stats': len(severity_derived_stats),
        'severity_entity_type_breakdown': {
            'score': sum(1 for e in sev_entities if e[0] == 'score'),
            'sev': sum(1 for e in sev_entities if e[0] == 'sev'),
        },
        'note': 'severity 实体按 typed-tuple (type, name) 计为 18 个（score 12 + sev 6）；'
                '其中 spo2_low 在 score 与 sev 各出现一次，去重后唯一名称为 17 个。'
                'component-blind 扣留全部 score/sev 实体。',
        'source_dataset': 'hg_fix_run/exp_4dchain_v2_dataset.pkl',
        'source_28_stats': 'hg_fix_run/probes/_task3_sev_graph_leak.json',
    },
}
json.dump(withholding, open(f'{BASE}/withholding_list.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)

print('=== 动作③ withholding_list.json ===')
print('被扣 severity 结构化列数:', len(withheld_columns))
for c in withheld_columns:
    print('  ', c)
print('被扣 severity 实体数:', len(withheld_severity_entities))
for e in withheld_severity_entities:
    print('  ', e)
print('severity 衍生图统计量数:', len(severity_derived_stats), '(应为 28)')
for s in severity_derived_stats:
    print('  ', s)

# ----------------------------------------------------------------------
# 动作④  逐种子 McNemar（三分类，4D-XGB vs lr_xgboost / lrraw）
# ----------------------------------------------------------------------
def mcnemar_row(y, p4, pb):
    y = np.asarray(y); p4 = np.asarray(p4); pb = np.asarray(pb)
    b = int(np.sum((p4 == y) & (pb != y)))   # 4D 对且基线错
    c = int(np.sum((p4 != y) & (pb == y)))   # 4D 错且基线对
    n = b + c
    p = float(binomtest(min(b, c), n, 0.5).pvalue) if n > 0 else 1.0
    return {'b': b, 'c': c, 'n_discordant': n, 'p_exact': p}


mcnemar = {'task': 'Task1 三分类', 'metric': 'predicted class == true class',
           'note': 'p_exact = 精确 McNemar（二项检验 binomtest(min(b,c), b+c, 0.5) 双侧），'
                   '与 exp_4dchain.run4 中 ER.binomtest 一致；b=4D 对且基线错，c=4D 错且基线对。',
           'seeds': SEEDS,
           'vs_lr_xgboost': {}, 'vs_lrraw': {}}

for seed in SEEDS:
    d4 = pickle.load(open(f'{FIX}/exp_foldpreds_v2_seed{seed}_4dxgb_task12.pkl', 'rb'))['4d_hypergraph']
    base = pickle.load(open(f'{FIX}/exp_foldpreds_task12_baselines_seed{seed}.pkl', 'rb'))
    y = np.array(d4['cls_y']); p4 = np.array(d4['cls_p'])
    mcnemar['vs_lr_xgboost'][str(seed)] = mcnemar_row(y, p4, np.array(base['lr_xgboost']['cls_p']))
    mcnemar['vs_lrraw'][str(seed)] = mcnemar_row(y, p4, np.array(base['lrraw']['cls_p']))

json.dump(mcnemar, open(f'{BASE}/mcnemar_per_seed.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)

print()
print('=== 动作④ mcnemar_per_seed.json ===')
print('%-5s | %-26s | %-26s' % ('seed', 'vs lr_xgboost (b/c/p)', 'vs lrraw (b/c/p)'))
for seed in SEEDS:
    a = mcnemar['vs_lr_xgboost'][str(seed)]
    b = mcnemar['vs_lrraw'][str(seed)]
    print('%-5d | %2d/%-2d p=%.4f           | %2d/%-2d p=%.4f'
          % (seed, a['b'], a['c'], a['p_exact'], b['b'], b['c'], b['p_exact']))
print('ALLDONE')
