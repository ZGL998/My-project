# -*- coding: utf-8 -*-
"""从 4D-HyperGraph 数据集导出四维实体词典。

用法：
    python export_entity_dictionary.py <dataset.pkl> [entity_dictionary.json]

说明：
    - 抽取期实体并集 = `entity_data` 各维在 253 例患者上的并集（应为 182）
    - 建模期词汇     = `ent2id` 的键（出现 >= 2 患者，应为 123）
    - 实体格式为 `(前缀, 名称)` 二元组，前缀 -> 维度映射见 PREFIX2DIM。
"""
import pickle
import json
import sys
from collections import defaultdict

# 实体前缀 -> 维度（论文 §3.3 四维定义）
PREFIX2DIM = {
    'sym': 'hist',       # 症状
    'diag': 'hist',      # 诊断
    'lab': 'evidence',   # 检验
    'img': 'evidence',   # 影像
    'med': 'treat',      # 用药
    'score': 'severity', # 严重度评分
    'sev': 'severity',   # 严重度体征
}


def main():
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else 'hg_fix_run/exp_4dchain_v2_dataset.pkl'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'entity_dictionary.json'

    with open(pkl_path, 'rb') as f:
        ds = pickle.load(f)

    # 1) 抽取期实体（182）：entity_data 各维并集
    extraction_by_dim = defaultdict(set)
    for pid, dims in ds['entity_data'].items():
        for dim, ents in dims.items():
            extraction_by_dim[dim].update(ents)

    # 2) 建模期实体（123）：ent2id 键，按前缀映射维度
    modeling_by_dim = defaultdict(set)
    for (prefix, name) in ds['ent2id'].keys():
        modeling_by_dim[PREFIX2DIM.get(prefix, 'other')].add((prefix, name))

    def fmt(ents):
        return sorted(f"{p}::{n}" for p, n in ents)

    result = {
        "source_dataset": pkl_path,
        "extraction_stage": {d: fmt(es) for d, es in sorted(extraction_by_dim.items())},
        "modeling_vocab":   {d: fmt(es) for d, es in sorted(modeling_by_dim.items())},
        "counts": {
            "extraction_total": sum(len(v) for v in extraction_by_dim.values()),
            "extraction_by_dim": {d: len(es) for d, es in sorted(extraction_by_dim.items())},
            "modeling_total": sum(len(v) for v in modeling_by_dim.values()),
            "modeling_by_dim": {d: len(es) for d, es in sorted(modeling_by_dim.items())},
        },
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"抽取期实体总数: {result['counts']['extraction_total']} (应为 182)")
    print(f"建模期词汇总数: {result['counts']['modeling_total']} (应为 123)")
    print(f"抽取期按维度: {result['counts']['extraction_by_dim']}")
    print(f"建模期按维度: {result['counts']['modeling_by_dim']}")
    print(f"已保存 -> {out_path}")


if __name__ == '__main__':
    main()
