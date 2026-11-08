#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 core30 三类 4D 重抽取结果。"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


DEFAULT_ROOT = Path(os.environ.get('EXPERIMENT_BASE', os.getcwd())) / "core30_4d_reextract_prep"

DIMENSION_FILES = {
    "病史背景": "history_background_similarity.json",
    "临床证据链": "clinical_evidence_chain_similarity.json",
    "诊疗轨迹": "treatment_trajectory_similarity.json",
    "临床严重度": "clinical_severity_similarity.json",
}

GROUPS = [
    "慢性阻塞性肺病",
    "慢性阻塞性肺病伴有急性加重",
    "慢性阻塞性肺病伴有急性下呼吸道感染",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args():
    parser = argparse.ArgumentParser(description="汇总外部子集三类 4D 重抽取结果。")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--source-subdir", default="four_d_outputs")
    parser.add_argument("--out-subdir", default="merged_core30_outputs")
    parser.add_argument("--dataset-tag", default="core30")
    parser.add_argument("--summary-title", default="core30 四维链总汇")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    source_dir = root / args.source_subdir
    out_dir = root / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_manifest = {
        "dataset": args.dataset_tag,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_root": str(source_dir.resolve()),
        "output_root": str(out_dir.resolve()),
        "dimensions": [],
    }

    for dim_name, file_name in DIMENSION_FILES.items():
        merged_rows = []
        source_summary = []
        for group in GROUPS:
            src = source_dir / group / file_name
            payload = read_json(src)
            rows = payload.get("病例相似链数据", [])
            for row in rows:
                row["来源诊断目录"] = group
            merged_rows.extend(rows)
            source_summary.append({
                "group": group,
                "source_file": str(src.resolve()),
                "case_count": payload.get("病例数量", len(rows)),
                "skipped_files": payload.get("跳过文件数", 0),
            })

        out_payload = {
            "相似度维度": dim_name,
            "dataset": args.dataset_tag,
            "来源目录": source_summary,
            "病例数量": len(merged_rows),
            "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "病例相似链数据": merged_rows,
        }
        out_path = out_dir / file_name
        out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        merged_manifest["dimensions"].append({
            "dimension": dim_name,
            "file_name": file_name,
            "path": str(out_path.resolve()),
            "case_count": len(merged_rows),
        })

    total_cases = 0
    if merged_manifest["dimensions"]:
        total_cases = max(x["case_count"] for x in merged_manifest["dimensions"])
    merged_manifest["total_cases"] = total_cases

    (out_dir / "manifest.json").write_text(
        json.dumps(merged_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "summary.txt").write_text(
        "\n".join(
            [
                args.summary_title,
                f"生成时间\t{merged_manifest['generated_at']}",
                f"总病例数\t{total_cases}",
                "",
            ]
            + [
                f"{x['dimension']}\t{x['file_name']}\t{x['case_count']}"
                for x in merged_manifest["dimensions"]
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(str(out_dir))
    print(str((out_dir / "manifest.json")))


if __name__ == "__main__":
    main()
