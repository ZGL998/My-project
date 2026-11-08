#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基于标签支持强度，从北京医院外部候选池构建 strict72 敏感性分析子集。

策略：
1. A_stable / B_ae 各取 completeness_score 最高的 24 例。
2. C_infection 仅保留“主诊断感染 + 至少 2 项支持证据 + 无明显矛盾语句”的 24 例。
3. 该子集用于外部验证敏感性分析，目标是提高标签可信度，而非作为全新 untouched 测试集。
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd


WORK = Path(os.environ.get('EXPERIMENT_BASE', os.getcwd()))
SRC_JSON = WORK / "external_raw_copd_manifest_beijing.json"
OUT_JSON = WORK / "external_raw_copd_subset_strict72.json"
OUT_CSV = WORK / "external_raw_copd_subset_strict72.csv"


def completeness_score(item):
    return (
        3.0 * int(item.get("has_admission_note", False)) +
        2.0 * int(item.get("has_discharge_note", False)) +
        1.0 * int(item.get("order_count", 0) > 0) +
        min(int(item.get("lab_detail_count", 0)), 300) / 300.0
    )


def infection_support(item):
    text = "\n".join([
        str(item.get("main_diag_text", "") or ""),
        str(item.get("dis_diag_name", "") or ""),
        str(item.get("admission_preview", "") or ""),
        str(item.get("discharge_preview", "") or ""),
    ])
    flags = {
        "infection_word": bool(re.search(r"感染|肺炎|下呼吸道", text)),
        "antibiotic_word": bool(re.search(r"美罗培南|哌拉西林|头孢|莫西沙星|左氧氟沙星|抗感染", text)),
        "fever_word": bool(re.search(r"发热|39\.?0?°?C|38\.\d", text)),
        "lab_inflam_word": bool(re.search(r"CRP|PCT|白细胞|中性粒细胞", text)),
        "imaging_word": bool(re.search(r"CT[^\n]{0,20}(感染|炎症|斑片|浸润)|胸部CT[^\n]{0,30}(感染|炎症|斑片|浸润)", text)),
        "contradiction_word": bool(re.search(r"未见肺实质感染|不考虑感染|未见明显感染", text)),
    }
    flags["support_score"] = sum(
        int(flags[k]) for k in ["infection_word", "antibiotic_word", "fever_word", "lab_inflam_word", "imaging_word"]
    )
    return flags


def main():
    items = json.loads(SRC_JSON.read_text(encoding="utf-8"))
    stable = [x for x in items if x["label"] == "A_stable"]
    ae = [x for x in items if x["label"] == "B_ae"]
    inf = [x for x in items if x["label"] == "C_infection"]

    stable = sorted(stable, key=lambda x: (completeness_score(x), x.get("lab_detail_count", 0), x.get("order_count", 0)), reverse=True)[:24]
    ae = sorted(ae, key=lambda x: (completeness_score(x), x.get("lab_detail_count", 0), x.get("order_count", 0)), reverse=True)[:24]

    strict_inf = []
    for item in inf:
        flags = infection_support(item)
        row = dict(item)
        row["infection_support"] = flags
        if flags["support_score"] >= 2 and not flags["contradiction_word"]:
            strict_inf.append(row)
    strict_inf = sorted(
        strict_inf,
        key=lambda x: (
            x["infection_support"]["support_score"],
            completeness_score(x),
            x.get("lab_detail_count", 0),
            x.get("order_count", 0),
        ),
        reverse=True,
    )[:24]

    chosen = stable + ae + strict_inf
    pd.DataFrame(chosen).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(chosen, ensure_ascii=False, indent=2), encoding="utf-8")

    print("strict72:", len(chosen), dict(Counter(x["label"] for x in chosen)))
    print("saved:", OUT_JSON)
    print("saved:", OUT_CSV)
    print("infection_retained:", len(strict_inf))
    for x in strict_inf[:5]:
        print("top_infection:", x["case_key"], x["infection_support"]["support_score"], x["main_diag_text"])


if __name__ == "__main__":
    main()
