# -*- coding: utf-8 -*-
"""从中文原始 CSV 中筛选 COPD 三类外部候选病例，并输出 manifest。

目标：
1. 不依赖目录名打标签，而是统一按“出院主诊断”映射为
   A_stable / B_ae / C_infection。
2. 合并北京医院的三个原始目录：
   - 慢性阻塞性肺疾病
   - 慢性阻塞性肺疾病急性加重
   - 肺部感染
3. 为后续 4D 重新抽取或规则侧外部验证准备病例清单。
"""
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd

BASE = Path(os.environ.get('RAW_DATA_ROOT', '/path/to/raw_data'))
OUT = Path(os.environ.get('EXPERIMENT_BASE', os.getcwd()))

BJ_DIRS = [
    BASE / "慢性阻塞性肺疾病",
    BASE / "慢性阻塞性肺疾病急性加重",
    BASE / "肺部感染",
]


def read_csv_auto(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"无法读取: {path}")


def map_label(text: str):
    s = str(text or "")
    if ("慢性阻塞" not in s) and ("慢阻肺" not in s) and ("COPD" not in s):
        return None
    if ("感染" in s) or ("肺炎" in s) or ("下呼吸道" in s):
        return "C_infection"
    if ("急性加重" in s) or ("AECOPD" in s):
        return "B_ae"
    return "A_stable"


def add_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pid"] = out["pid"].astype(str)
    out["visit_id"] = out["visit_id"].astype(str)
    out["case_key"] = out["pid"] + "||" + out["visit_id"]
    return out


def load_main_diagnosis_candidates():
    rows = []
    for d in BJ_DIRS:
        diag_file = d / ("病案首页诊断记录-北京.csv" if d.name != "肺部感染" else "首页诊断-北京.csv")
        df = add_key(read_csv_auto(diag_file))
        df = df[df["inpatient_diagnosis_type"].astype(str).str.contains("出院诊断", na=False)]
        df = df[df["main_diagnosis_yes_no"].astype(str).isin(["1", "1.0"])].copy()
        main_txt = df["diag_name"].fillna(df["institute_dis_name"]).astype(str)
        df["label"] = main_txt.map(map_label)
        df["main_diag_text"] = main_txt
        df = df[df["label"].notna()].copy()
        df["source_dir"] = d.name
        rows.append(df[["case_key", "pid", "visit_id", "organization_name", "label", "main_diag_text", "source_dir"]])
    cand = pd.concat(rows, ignore_index=True)
    cand = cand.sort_values(["case_key", "source_dir"]).drop_duplicates("case_key", keep="first")
    return cand


def index_table(path: Path) -> pd.DataFrame:
    return add_key(read_csv_auto(path))


def summarize_note(row: pd.Series, max_len=400):
    if row is None:
        return ""
    parts = []
    for k in ("chief_complaint", "current_disease", "physical_exam", "treat_process_desc", "dis_diag_name"):
        v = str(row.get(k, "") or "").strip()
        if v and v.lower() != "nan":
            parts.append(f"{k}: {v}")
    return "\n".join(parts)[:max_len]


def build_manifest():
    cand = load_main_diagnosis_candidates()

    adm = pd.concat(
        [
            index_table(BASE / "慢性阻塞性肺疾病" / "入院记录-北京.csv"),
            index_table(BASE / "慢性阻塞性肺疾病急性加重" / "入院记录-北京.csv"),
            index_table(BASE / "肺部感染" / "入院记录-北京.csv"),
        ],
        ignore_index=True,
    ).drop_duplicates("case_key", keep="first")
    dis = pd.concat(
        [
            index_table(BASE / "慢性阻塞性肺疾病" / "出院记录-北京.csv"),
            index_table(BASE / "慢性阻塞性肺疾病急性加重" / "出院记录-北京.csv"),
            index_table(BASE / "肺部感染" / "出院记录-北京.csv"),
        ],
        ignore_index=True,
    ).drop_duplicates("case_key", keep="first")
    lab = pd.concat(
        [
            index_table(BASE / "慢性阻塞性肺疾病" / "检验详细记录-北京.csv"),
            index_table(BASE / "慢性阻塞性肺疾病急性加重" / "检验详细记录-北京.csv"),
            index_table(BASE / "肺部感染" / "检验详细记录-北京.csv"),
        ],
        ignore_index=True,
    )
    orders = pd.concat(
        [
            index_table(BASE / "慢性阻塞性肺疾病" / "住院医嘱-北京.csv"),
            index_table(BASE / "慢性阻塞性肺疾病急性加重" / "住院医嘱-北京.csv"),
            index_table(BASE / "肺部感染" / "住院医嘱-北京.csv"),
        ],
        ignore_index=True,
    )

    adm_by = {k: r for k, r in adm.set_index("case_key").iterrows()}
    dis_by = {k: r for k, r in dis.set_index("case_key").iterrows()}
    lab_cnt = lab.groupby("case_key").size().to_dict()
    order_cnt = orders.groupby("case_key").size().to_dict()

    items = []
    for _, r in cand.iterrows():
        key = r["case_key"]
        ar = adm_by.get(key)
        dr = dis_by.get(key)
        items.append(
            {
                "case_key": key,
                "pid": r["pid"],
                "visit_id": r["visit_id"],
                "organization_name": r["organization_name"],
                "source_dir": r["source_dir"],
                "label": r["label"],
                "main_diag_text": r["main_diag_text"],
                "admission_dt": "" if ar is None else str(ar.get("admission_dt", "") or ""),
                "discharge_dt": "" if dr is None else str(dr.get("discharge_dt", "") or ""),
                "dis_diag_name": "" if dr is None else str(dr.get("dis_diag_name", "") or ""),
                "has_admission_note": ar is not None,
                "has_discharge_note": dr is not None,
                "lab_detail_count": int(lab_cnt.get(key, 0)),
                "order_count": int(order_cnt.get(key, 0)),
                "admission_preview": summarize_note(ar),
                "discharge_preview": summarize_note(dr),
            }
        )
    return items


def completeness_score(item):
    """按外部验证可用性打分：入院/出院文本优先，其次医嘱，再看检验密度。"""
    return (
        3.0 * int(item["has_admission_note"]) +
        2.0 * int(item["has_discharge_note"]) +
        1.0 * int(item["order_count"] > 0) +
        min(item["lab_detail_count"], 300) / 300.0
    )


def select_balanced_subset(items, per_class):
    by_label = {"A_stable": [], "B_ae": [], "C_infection": []}
    for x in items:
        by_label[x["label"]].append(x)
    chosen = []
    for lb in ("A_stable", "B_ae", "C_infection"):
        ranked = sorted(
            by_label[lb],
            key=lambda x: (completeness_score(x), x["lab_detail_count"], x["order_count"]),
            reverse=True,
        )
        chosen.extend(ranked[:per_class])
    return chosen


def main():
    items = build_manifest()
    out_json = OUT / "external_raw_copd_manifest_beijing.json"
    out_csv = OUT / "external_raw_copd_manifest_beijing.csv"
    balanced87_json = OUT / "external_raw_copd_subset_balanced87.json"
    balanced87_csv = OUT / "external_raw_copd_subset_balanced87.csv"
    core30_json = OUT / "external_raw_copd_subset_core30.json"
    core30_csv = OUT / "external_raw_copd_subset_core30.csv"

    pd.DataFrame(items).to_csv(out_csv, index=False, encoding="utf-8-sig")
    json.dump(items, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    balanced87 = select_balanced_subset(items, per_class=29)
    core30 = select_balanced_subset(items, per_class=10)
    pd.DataFrame(balanced87).to_csv(balanced87_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(core30).to_csv(core30_csv, index=False, encoding="utf-8-sig")
    json.dump(balanced87, open(balanced87_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(core30, open(core30_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    c = Counter(x["label"] for x in items)
    print("外部原始 COPD 候选病例:", len(items), dict(c))
    print("有入院记录:", sum(int(x["has_admission_note"]) for x in items))
    print("有出院记录:", sum(int(x["has_discharge_note"]) for x in items))
    print("检验详情>0:", sum(int(x["lab_detail_count"] > 0) for x in items))
    print("医嘱>0:", sum(int(x["order_count"] > 0) for x in items))
    print("保存:", out_json)
    print("保存:", out_csv)
    print("balanced87:", Counter(x["label"] for x in balanced87), "| 有出院记录:", sum(int(x["has_discharge_note"]) for x in balanced87),
          "| 有医嘱:", sum(int(x["order_count"] > 0) for x in balanced87))
    print("保存:", balanced87_json)
    print("保存:", balanced87_csv)
    print("core30:", Counter(x["label"] for x in core30), "| 有出院记录:", sum(int(x["has_discharge_note"]) for x in core30),
          "| 有医嘱:", sum(int(x["order_count"] > 0) for x in core30))
    print("保存:", core30_json)
    print("保存:", core30_csv)


if __name__ == "__main__":
    main()
