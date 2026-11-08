#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 core30 四维相似链输出转为 exp_4dchain_v2.py 外部验证所需格式。"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


DEFAULT_ROOT = Path(os.environ.get('EXPERIMENT_BASE', os.getcwd())) / "core30_4d_reextract_prep"

FILE_TO_DIM = {
    "history_background_similarity.json": "history_chain",
    "clinical_evidence_chain_similarity.json": "evidence_chain",
    "treatment_trajectory_similarity.json": "care_chain",
    "clinical_severity_similarity.json": "severity_chain",
}

GROUP_TO_DIAG = {
    "慢性阻塞性肺病": "慢性阻塞性肺疾病",
    "慢性阻塞性肺病伴有急性加重": "慢性阻塞性肺疾病急性加重",
    "慢性阻塞性肺病伴有急性下呼吸道感染": "慢性阻塞性肺疾病伴急性下呼吸道感染",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_str(v):
    if v is None:
        return ""
    text = str(v).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def parse_args():
    parser = argparse.ArgumentParser(description="把四维相似链输出转为 exp_4dchain_v2.py 外部验证所需格式。")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--src-subdir", default="merged_core30_outputs")
    parser.add_argument("--out-subdir", default="external_chain_eval_input")
    parser.add_argument("--dataset-tag", default="core30")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    src_root = root / args.src_subdir
    out_root = root / args.out_subdir
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": args.dataset_tag,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_root": str(src_root.resolve()),
        "output_root": str(out_root.resolve()),
        "files": [],
    }

    for filename, dim in FILE_TO_DIM.items():
        src_path = src_root / filename
        payload = read_json(src_path)
        rows = payload.get("病例相似链数据", [])
        data = []
        for row in rows:
            ident = row.get("病例标识", {})
            groups = row.get("字段分组明细", [])
            source_group = normalize_str(row.get("来源诊断目录"))
            diag = GROUP_TO_DIAG.get(source_group) or normalize_str(ident.get("主诊断"))
            case_id = normalize_str(ident.get("病例编号"))
            meta = {
                "f": f"{case_id}.json",
                "id": case_id,
                "vid": case_id,
                "name": None,
                "diag": diag,
                "case_id": case_id,
            }
            formatted_groups = []
            for g in groups:
                formatted_groups.append({
                    "g": g.get("字段分组"),
                    "r": g.get("相似度匹配核心依据"),
                    "d": g.get("提取结果") or {},
                    "s": g.get("相似链条"),
                })
            data.append({
                "meta": meta,
                "score": 1.0,
                "groups": formatted_groups,
                "chain": row.get("合并相似链条", ""),
                "logic_score": 1.0,
            })

        out_payload = {
            "dim": dim,
            "cnt": len(data),
            "threshold": None,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data": data,
        }
        out_path = out_root / filename
        out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["files"].append({
            "filename": filename,
            "dimension": dim,
            "count": len(data),
            "path": str(out_path.resolve()),
        })

    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_root))
    print(str((out_root / "manifest.json")))


if __name__ == "__main__":
    main()
