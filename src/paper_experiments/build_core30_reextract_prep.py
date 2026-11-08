#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""组装 core30 原始病例包，并生成 4D 重抽取准备目录。

输入：
- $EXPERIMENT_BASE/external_raw_copd_subset_core30.json
- 北京医院三类原始 CSV

输出：
- core30_4d_reextract_prep/input_cases/{三类}/
- core30_4d_reextract_prep/raw_texts/{三类}/
- core30_4d_reextract_prep/four_d_outputs/{三类}/
- core30_4d_reextract_prep/run_core30_4d_reextract.sh
- core30_4d_reextract_prep/core30_reextract_manifest.json

说明：
1. 输入病例 JSON 尽量贴近训练集既有 schema，便于直接喂给 build_similarity_chains.py。
2. 原始文本摘要文件保留 narrative + 检查/检验/医嘱的紧凑展开，便于人工抽查。
3. 这里做“准备”而不直接跑 4D 重抽取；如需执行，可运行生成的 shell 脚本。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


BASE = Path(os.environ.get('RAW_DATA_ROOT', '/path/to/raw_data'))
WORK = Path(os.environ.get('EXPERIMENT_BASE', os.getcwd()))
DEFAULT_MANIFEST_JSON = WORK / "external_raw_copd_subset_core30.json"
DEFAULT_OUT_ROOT = WORK / "core30_4d_reextract_prep"

TRAIN_ROOT = Path(os.environ.get('TRAIN_ROOT', '/path/to/training_set'))
TRAIN_SCRIPT = TRAIN_ROOT / "build_similarity_chains.py"
RULE_JSON = TRAIN_ROOT / "出院主诊断_COPD三类" / "四维相似链处理规则.json"

LABEL_TO_FOLDER = {
    "A_stable": "慢性阻塞性肺病",
    "B_ae": "慢性阻塞性肺病伴有急性加重",
    "C_infection": "慢性阻塞性肺病伴有急性下呼吸道感染",
}

SOURCE_INFO = {
    "慢性阻塞性肺疾病": {
        "basic": "患者就诊基本信息-北京.csv",
        "home": "病案首页基本信息-北京.csv",
        "diag": "病案首页诊断记录-北京.csv",
        "surg": "病案首页手术信息-北京.csv",
        "admission": "入院记录-北京.csv",
        "discharge": "出院记录-北京.csv",
        "exam": "检查记录-北京.csv",
        "lab": "检验详细记录-北京.csv",
        "orders": "住院医嘱-北京.csv",
    },
    "慢性阻塞性肺疾病急性加重": {
        "basic": "患者就诊基本信息-北京.csv",
        "home": "病案首页基本信息-北京.csv",
        "diag": "病案首页诊断记录-北京.csv",
        "surg": "病案首页手术信息-北京.csv",
        "admission": "入院记录-北京.csv",
        "discharge": "出院记录-北京.csv",
        "exam": "检查记录-北京.csv",
        "lab": "检验详细记录-北京.csv",
        "orders": "住院医嘱-北京.csv",
    },
    "肺部感染": {
        "basic": "患者基本信息-北京.csv",
        "home": "患者病案首页-北京.csv",
        "diag": "首页诊断-北京.csv",
        "surg": "首页手术-北京.csv",
        "admission": "入院记录-北京.csv",
        "discharge": "出院记录-北京.csv",
        "exam": "检查记录-北京.csv",
        "lab": "检验详细记录-北京.csv",
        "orders": "住院医嘱-北京.csv",
    },
}

TABLE_KEYS = ["basic", "home", "diag", "surg", "admission", "discharge", "exam", "lab", "orders"]


def read_csv_auto(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    raise RuntimeError(f"无法读取 CSV: {path}")


def normalize_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def clean_multiline(text) -> str:
    text = normalize_str(text)
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def add_case_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pid"] = out["pid"].map(normalize_str)
    out["visit_id"] = out["visit_id"].map(normalize_str)
    out["case_key"] = out["pid"] + "||" + out["visit_id"]
    return out


def compact_dict(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            inner = compact_dict(v)
            if inner:
                out[k] = inner
        elif isinstance(v, list):
            inner = [x for x in v if x not in ("", None, [], {})]
            if inner:
                out[k] = inner
        elif v not in ("", None):
            out[k] = v
    return out


def unique_preserve(seq):
    seen = set()
    out = []
    for item in seq:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, (dict, list)) else str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def pick_first(records, *keys) -> str:
    for rec in records:
        for key in keys:
            value = normalize_str(rec.get(key))
            if value:
                return value
    return ""


def build_case_stem(item: dict, dataset_tag: str) -> str:
    pid = re.sub(r"[^A-Za-z0-9]+", "_", normalize_str(item["pid"])) or "PID"
    vid = re.sub(r"[^A-Za-z0-9]+", "_", normalize_str(item["visit_id"])) or "VID"
    prefix = re.sub(r"[^A-Za-z0-9]+", "_", dataset_tag).upper()
    return f"zy@BJ{prefix}_{pid}_{vid}"


def build_identifiers(item: dict, stem: str, dataset_tag: str) -> dict:
    return {
        "关联号": f"{dataset_tag}::{item['case_key']}",
        "唯一号": stem,
        "住院号/门诊号": normalize_str(item["visit_id"]),
        "姓名": "",
        "年龄": "",
        "性别": "",
        "出生日期": "",
        "入院时间": normalize_str(item.get("admission_dt")),
        "出院时间": normalize_str(item.get("discharge_dt")),
    }


def group_records(df: pd.DataFrame | None) -> dict[str, list[dict]]:
    if df is None or df.empty:
        return {}
    grouped = defaultdict(list)
    for rec in df.to_dict("records"):
        grouped[normalize_str(rec["case_key"])].append(rec)
    return dict(grouped)


def build_tables_for_source(source_dir: str, case_keys: set[str]) -> dict[str, pd.DataFrame]:
    cfg = SOURCE_INFO[source_dir]
    tables = {}
    base_dir = BASE / source_dir
    for table_key in TABLE_KEYS:
        path = base_dir / cfg[table_key]
        df = add_case_key(read_csv_auto(path))
        df = df[df["case_key"].isin(case_keys)].copy()
        tables[table_key] = df
    return tables


def compose_emr_record(item, ids, admission_rows, discharge_rows, home_rows):
    sex = pick_first(admission_rows, "sex", "patient_sex") or pick_first(home_rows, "sex")
    age = pick_first(admission_rows, "patient_age") or pick_first(home_rows, "patient_age", "年龄（岁）")
    birth = pick_first(home_rows, "birth_dt", "birthday")
    in_diag = pick_first(admission_rows, "admi_diag_name")
    out_diag = pick_first(discharge_rows, "dis_diag_name") or normalize_str(item.get("dis_diag_name"))
    treat_process = clean_multiline(pick_first(discharge_rows, "treat_process_desc"))
    discharge_order = clean_multiline(pick_first(discharge_rows, "discharge_order"))
    discharge_note_parts = [
        f"主诉：{clean_multiline(pick_first(discharge_rows, 'chief_complaint'))}",
        f"诊疗经过：{treat_process}",
        f"出院情况：{clean_multiline(pick_first(discharge_rows, 'current_situation', 'discharge_symptom_sign', 'dis_condition'))}",
        f"出院诊断：{out_diag}",
        f"出院医嘱：{discharge_order}",
    ]
    discharge_note = "\n".join([x for x in discharge_note_parts if x.split("：", 1)[1].strip()])
    return compact_dict({
        **ids,
        "年龄": age,
        "性别": sex,
        "出生日期": birth,
        "入院诊断": in_diag,
        "现病史": clean_multiline(pick_first(admission_rows, "current_disease")),
        "既往史": clean_multiline(pick_first(admission_rows, "disease_history")),
        "个人史": clean_multiline(pick_first(admission_rows, "personal_history")),
        "过敏史": clean_multiline(pick_first(admission_rows, "allergy_history")),
        "家族史": clean_multiline(pick_first(admission_rows, "family_history")),
        "婚育史": clean_multiline(pick_first(admission_rows, "marriage_history")),
        "主诉": clean_multiline(pick_first(admission_rows, "chief_complaint")),
        "体格检查": clean_multiline(pick_first(admission_rows, "physical_exam")),
        "出院诊断": out_diag,
        "出院记录": discharge_note,
    })


def compose_home_record(item, ids, home_rows, diag_rows, admission_rows):
    home = home_rows[0] if home_rows else {}
    sex = normalize_str(home.get("sex")) or pick_first(admission_rows, "sex", "patient_sex")
    age = normalize_str(home.get("patient_age")) or pick_first(admission_rows, "patient_age")
    birth = normalize_str(home.get("birth_dt")) or normalize_str(home.get("birthday"))
    main_diags = []
    other_diags = []
    for rec in diag_rows:
        diag_name = normalize_str(rec.get("diag_name")) or normalize_str(rec.get("institute_dis_name"))
        if not diag_name:
            continue
        if normalize_str(rec.get("main_diagnosis_yes_no")) in {"1", "1.0"}:
            main_diags.append(diag_name)
        else:
            other_diags.append(diag_name)
    main_diag = main_diags[0] if main_diags else normalize_str(item.get("main_diag_text"))
    out = {
        **ids,
        "年龄": age,
        "年龄（岁）": age,
        "性别": sex,
        "出生日期": birth,
        "病案号": normalize_str(home.get("case_no")) or normalize_str(home.get("card_no")),
        "入院科别": normalize_str(home.get("admi_dept_name")) or normalize_str(home.get("dept_name")),
        "出院科别": normalize_str(home.get("dis_dept_name")) or normalize_str(home.get("discharge_department")),
        "入院时间.1": normalize_str(home.get("admission_dt")) or normalize_str(item.get("admission_dt")),
        "出院时间.1": normalize_str(home.get("discharge_dt")) or normalize_str(item.get("discharge_dt")),
        "实际住院（天）": normalize_str(home.get("actual_inpat_days")),
        "出院主要诊断名称": main_diag,
        "出院主要诊断编码": normalize_str(next((r.get("diag_code") for r in diag_rows if normalize_str(r.get("main_diagnosis_yes_no")) in {"1", "1.0"}), "")),
        "离院方式": normalize_str(home.get("discharge_class_name")),
        "入院诊断名称": pick_first(admission_rows, "admi_diag_name"),
        "门（急）诊诊断名称": pick_first(admission_rows, "admi_diag_name"),
    }
    for idx, diag_name in enumerate(other_diags[:40], start=1):
        out[f"出院其他诊断名称{idx}"] = diag_name
    return compact_dict(out)


def compose_lab_rows(ids, lab_rows):
    out = []
    for rec in lab_rows:
        item_name = normalize_str(rec.get("lab_type_name"))
        subitem = normalize_str(rec.get("lab_item_subitem_name"))
        value = normalize_str(rec.get("lab_result_value")) or normalize_str(rec.get("lab_result_value_qualitative"))
        out.append(compact_dict({
            **ids,
            "报告日期": normalize_str(rec.get("create_date")) or normalize_str(rec.get("update_date")),
            "检验项目": item_name,
            "检验指标名": subitem,
            "英文名称": normalize_str(rec.get("lab_item_code")),
            "检验指标值": value,
            "单位": normalize_str(rec.get("lab_unit")),
            "参考范围": normalize_str(rec.get("value_range")),
            "样本类型": normalize_str(rec.get("test_sample_no")),
            "接收日期": normalize_str(rec.get("report_id")),
            "检查时间": normalize_str(rec.get("create_date")),
            "检验编码": normalize_str(rec.get("lab_item_subitem_code")),
            "_source_file": "检验详细记录-北京.csv",
        }))
    return unique_preserve(out)


def compose_order_rows(ids, order_rows):
    out = []
    for rec in order_rows:
        name = normalize_str(rec.get("order_detail_name")) or normalize_str(rec.get("drug_name"))
        out.append(compact_dict({
            **ids,
            "开始时间": normalize_str(rec.get("drug_start_dt")) or normalize_str(rec.get("order_dt")) or normalize_str(rec.get("exe_start_dt")),
            "结束时间": normalize_str(rec.get("drug_end_dt")) or normalize_str(rec.get("stop_order_dt")) or normalize_str(rec.get("exe_stop_dt")),
            "医嘱组号": normalize_str(rec.get("order_no")),
            "医嘱名称": name,
            "频次": normalize_str(rec.get("drug_fre")),
            "用法": normalize_str(rec.get("drug_way_name")),
            "剂量": normalize_str(rec.get("drug_per_dose")) or normalize_str(rec.get("drug_total_dose")),
            "单位": normalize_str(rec.get("drug_unit")) or normalize_str(rec.get("dispen_drug_unit")),
            "_source_file": "住院医嘱-北京.csv",
        }))
    return unique_preserve(out)


def compose_exam_rows(ids, exam_rows):
    out = []
    for rec in exam_rows:
        out.append(compact_dict({
            **ids,
            "检查号": normalize_str(rec.get("report_id")) or normalize_str(rec.get("apply_id")),
            "检查设备类型": normalize_str(rec.get("examination_device")) or normalize_str(rec.get("exam_type_name")),
            "检查项目": normalize_str(rec.get("hospital_exam_item_name")) or normalize_str(rec.get("report_title")),
            "检查部位": normalize_str(rec.get("exam_site")),
            "检查时间": normalize_str(rec.get("exam_dt")) or normalize_str(rec.get("report_dt")),
            "影像所见": clean_multiline(normalize_str(rec.get("exam_desc"))),
            "检查结论": clean_multiline(normalize_str(rec.get("exam_conclusion"))),
            "送检科室名称": normalize_str(rec.get("exam_dept_code")) or normalize_str(rec.get("apply_dept_code")),
            "_source_file": "检查记录-北京.csv",
        }))
    return unique_preserve(out)


def compose_surgery_rows(ids, surg_rows):
    out = []
    for rec in surg_rows:
        name = normalize_str(rec.get("openration_name")) or normalize_str(rec.get("operation_code"))
        if not name:
            continue
        out.append(compact_dict({
            **ids,
            "手术名称": name,
            "手术日期": normalize_str(rec.get("openration_date")) or normalize_str(rec.get("standard_operation_date")),
            "手术级别": normalize_str(rec.get("opera_grade_code")),
            "麻醉方式": normalize_str(rec.get("anes_way_name")),
            "术前诊断": normalize_str(rec.get("before_operation_dis_name")),
            "术后诊断": normalize_str(rec.get("after_operation_dis_name")),
            "手术医师": normalize_str(rec.get("operation_doctor_name")),
            "_source_file": "病案首页手术信息-北京.csv",
        }))
    return unique_preserve(out)


def build_case_json(item, grouped_tables, dataset_tag: str):
    stem = build_case_stem(item, dataset_tag)
    ids = build_identifiers(item, stem, dataset_tag)
    admission_rows = grouped_tables["admission"].get(item["case_key"], [])
    discharge_rows = grouped_tables["discharge"].get(item["case_key"], [])
    basic_rows = grouped_tables["basic"].get(item["case_key"], [])
    home_rows = grouped_tables["home"].get(item["case_key"], [])
    diag_rows = grouped_tables["diag"].get(item["case_key"], [])
    surg_rows = grouped_tables["surg"].get(item["case_key"], [])
    exam_rows = grouped_tables["exam"].get(item["case_key"], [])
    lab_rows = grouped_tables["lab"].get(item["case_key"], [])
    order_rows = grouped_tables["orders"].get(item["case_key"], [])

    base_info = compact_dict({
        **ids,
        "性别": pick_first(basic_rows, "sex") or pick_first(home_rows, "sex"),
        "年龄": pick_first(home_rows, "patient_age") or pick_first(basic_rows, "patient_age"),
        "出生日期": pick_first(home_rows, "birth_dt", "birthday"),
        "就诊科室名": pick_first(home_rows, "admi_dept_name", "dept_name"),
    })

    case_json = {
        "基本数据": base_info,
        "电子病历": [compose_emr_record(item, ids, admission_rows, discharge_rows, home_rows)],
        "病案首页": [compose_home_record(item, ids, home_rows, diag_rows, admission_rows)],
        "护理记录": [],
        "病程记录": [],
        "实验室检查": compose_lab_rows(ids, lab_rows),
        "影像学检查": compose_exam_rows(ids, exam_rows),
        "手术记录": compose_surgery_rows(ids, surg_rows),
        "医嘱": compose_order_rows(ids, order_rows),
    }
    return stem, case_json


def top_n_lines(rows, limit):
    if len(rows) <= limit:
        return rows
    return rows[:limit] + [f"...（其余 {len(rows) - limit} 条省略，详见同名 JSON）"]


def build_raw_text(item, stem, case_json, out_root: Path, dataset_title: str):
    emr = (case_json.get("电子病历") or [{}])[0]
    home = (case_json.get("病案首页") or [{}])[0]
    exams = case_json.get("影像学检查") or []
    labs = case_json.get("实验室检查") or []
    orders = case_json.get("医嘱") or []
    surgeries = case_json.get("手术记录") or []

    diag_lines = []
    main_diag = normalize_str(home.get("出院主要诊断名称"))
    if main_diag:
        diag_lines.append(f"主要诊断：{main_diag}")
    for i in range(1, 41):
        other = normalize_str(home.get(f"出院其他诊断名称{i}"))
        if other:
            diag_lines.append(f"其他诊断{i}：{other}")

    exam_lines = [
        f"- {normalize_str(r.get('检查时间'))} | {normalize_str(r.get('检查项目'))} | "
        f"{normalize_str(r.get('检查部位'))} | 结论：{normalize_str(r.get('检查结论'))}"
        for r in exams
    ]
    lab_lines = [
        f"- {normalize_str(r.get('报告日期'))} | {normalize_str(r.get('检验项目'))} | "
        f"{normalize_str(r.get('检验指标名'))} = {normalize_str(r.get('检验指标值'))}"
        f"{normalize_str(r.get('单位'))} | 参考范围：{normalize_str(r.get('参考范围'))}"
        for r in labs
    ]
    order_lines = [
        f"- {normalize_str(r.get('开始时间'))} ~ {normalize_str(r.get('结束时间'))} | "
        f"{normalize_str(r.get('医嘱名称'))} | {normalize_str(r.get('频次'))} | "
        f"{normalize_str(r.get('用法'))} | {normalize_str(r.get('剂量'))}{normalize_str(r.get('单位'))}"
        for r in orders
    ]
    surg_lines = [
        f"- {normalize_str(r.get('手术日期'))} | {normalize_str(r.get('手术名称'))} | "
        f"术前：{normalize_str(r.get('术前诊断'))} | 术后：{normalize_str(r.get('术后诊断'))}"
        for r in surgeries
    ]

    sections = [
        f"# {dataset_title} 原始病例组装文本",
        f"病例文件名：{stem}.json",
        f"case_key：{item['case_key']}",
        f"标签：{item['label']} -> {LABEL_TO_FOLDER[item['label']]}",
        f"来源目录：{item['source_dir']}",
        f"机构：{item['organization_name']}",
        f"主诊断：{item['main_diag_text']}",
        f"入院日期：{normalize_str(item.get('admission_dt'))}",
        f"出院日期：{normalize_str(item.get('discharge_dt'))}",
        "",
        "## 主诉",
        clean_multiline(emr.get("主诉")),
        "",
        "## 现病史",
        clean_multiline(emr.get("现病史")),
        "",
        "## 既往史",
        clean_multiline(emr.get("既往史")),
        "",
        "## 个人史",
        clean_multiline(emr.get("个人史")),
        "",
        "## 家族史",
        clean_multiline(emr.get("家族史")),
        "",
        "## 体格检查",
        clean_multiline(emr.get("体格检查")),
        "",
        "## 出院记录",
        clean_multiline(emr.get("出院记录")),
        "",
        "## 病案首页诊断",
        "\n".join(diag_lines),
        "",
        f"## 检查记录（共 {len(exam_lines)} 条）",
        "\n".join(top_n_lines(exam_lines, 80)),
        "",
        f"## 检验详细记录（共 {len(lab_lines)} 条）",
        "\n".join(top_n_lines(lab_lines, 200)),
        "",
        f"## 住院医嘱（共 {len(order_lines)} 条）",
        "\n".join(top_n_lines(order_lines, 200)),
        "",
        f"## 手术记录（共 {len(surg_lines)} 条）",
        "\n".join(surg_lines) if surg_lines else "无",
        "",
        "## 对应 JSON",
        str((out_root / "input_cases" / LABEL_TO_FOLDER[item["label"]] / f"{stem}.json").resolve()),
    ]
    return "\n".join(sections).strip() + "\n"


def ensure_dirs(out_root: Path):
    for base in ("input_cases", "raw_texts", "four_d_outputs"):
        for folder in LABEL_TO_FOLDER.values():
            (out_root / base / folder).mkdir(parents=True, exist_ok=True)


def write_run_script(out_root: Path, run_name: str, dataset_tag: str):
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'PYTHON_BIN="${{PYTHON_BIN:-python3}}"',
        f'SCRIPT="{TRAIN_SCRIPT}"',
        f'RULES="{RULE_JSON}"',
        "",
    ]
    for folder in LABEL_TO_FOLDER.values():
        input_dir = (out_root / "input_cases" / folder).resolve()
        output_dir = (out_root / "four_d_outputs" / folder).resolve()
        lines.extend([
            f'echo "[{dataset_tag}] 处理 {folder}"',
            '"$PYTHON_BIN" "$SCRIPT" \\',
            f'  --rules "$RULES" \\',
            f'  --input-dir "{input_dir}" \\',
            f'  --output-dir "{output_dir}"',
            "",
        ])
    path = out_root / run_name
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)
    return path


def write_readme(out_root: Path, run_script_path: Path, summary_rows: list[dict], dataset_title: str):
    lines = [
        f"# {dataset_title} 4D 重抽取准备",
        "",
        "已生成三类输入病例 JSON、可读原文摘要和输出目录。",
        "",
        "## 目录",
        f"- 输入病例：`{(out_root / 'input_cases').resolve()}`",
        f"- 原文摘要：`{(out_root / 'raw_texts').resolve()}`",
        f"- 4D 输出目录：`{(out_root / 'four_d_outputs').resolve()}`",
        f"- 执行脚本：`{run_script_path.resolve()}`",
        "",
        "## 分组统计",
    ]
    for row in summary_rows:
        lines.append(
            f"- {row['label_folder']}：{row['n']} 例，检验中位数 {row['lab_median']}，医嘱中位数 {row['order_median']}"
        )
    lines += [
        "",
        "## 运行方式",
        f"直接执行：`bash {run_script_path.resolve()}`",
        "",
        "说明：这里只做准备，不自动跑重抽取。",
    ]
    (out_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def median_int(values):
    if not values:
        return 0
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return int(vals[mid])
    return int(round((vals[mid - 1] + vals[mid]) / 2.0))


def parse_args():
    parser = argparse.ArgumentParser(description="组装外部 COPD 子集原始病例包，并生成 4D 重抽取准备目录。")
    parser.add_argument("--manifest-json", default=str(DEFAULT_MANIFEST_JSON))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--dataset-tag", default="core30")
    parser.add_argument("--dataset-title", default="Core30")
    parser.add_argument("--run-script-name", default="run_core30_4d_reextract.sh")
    parser.add_argument("--manifest-name", default="core30_reextract_manifest.json")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_json = Path(args.manifest_json)
    out_root = Path(args.out_root)
    ensure_dirs(out_root)
    items = json.loads(manifest_json.read_text(encoding="utf-8"))
    case_keys = {normalize_str(x["case_key"]) for x in items}

    tables_by_source = {}
    grouped_by_source = {}
    for source_dir in sorted({x["source_dir"] for x in items}):
        tables = build_tables_for_source(source_dir, case_keys)
        tables_by_source[source_dir] = tables
        grouped_by_source[source_dir] = {k: group_records(v) for k, v in tables.items()}

    manifest_rows = []
    label_stats = defaultdict(lambda: {"labs": [], "orders": [], "count": 0})
    for item in items:
        source_dir = item["source_dir"]
        grouped_tables = grouped_by_source[source_dir]
        stem, case_json = build_case_json(item, grouped_tables, args.dataset_tag)
        folder = LABEL_TO_FOLDER[item["label"]]

        json_path = out_root / "input_cases" / folder / f"{stem}.json"
        txt_path = out_root / "raw_texts" / folder / f"{stem}.txt"
        json_path.write_text(json.dumps(case_json, ensure_ascii=False, indent=2), encoding="utf-8")
        txt_path.write_text(build_raw_text(item, stem, case_json, out_root, args.dataset_title), encoding="utf-8")

        lab_count = len(case_json.get("实验室检查") or [])
        order_count = len(case_json.get("医嘱") or [])
        exam_count = len(case_json.get("影像学检查") or [])
        surg_count = len(case_json.get("手术记录") or [])
        manifest_rows.append({
            "case_key": item["case_key"],
            "label": item["label"],
            "label_folder": folder,
            "source_dir": source_dir,
            "file_stem": stem,
            "json_path": str(json_path.resolve()),
            "text_path": str(txt_path.resolve()),
            "lab_count": lab_count,
            "order_count": order_count,
            "exam_count": exam_count,
            "surgery_count": surg_count,
            "has_admission_note": bool((case_json.get("电子病历") or [{}])[0].get("现病史")),
            "has_discharge_note": bool((case_json.get("电子病历") or [{}])[0].get("出院记录")),
        })
        label_stats[folder]["count"] += 1
        label_stats[folder]["labs"].append(lab_count)
        label_stats[folder]["orders"].append(order_count)

    run_script = write_run_script(out_root, args.run_script_name, args.dataset_tag)

    summary_rows = []
    for folder in LABEL_TO_FOLDER.values():
        stat = label_stats[folder]
        summary_rows.append({
            "label_folder": folder,
            "n": stat["count"],
            "lab_median": median_int(stat["labs"]),
            "order_median": median_int(stat["orders"]),
        })
    write_readme(out_root, run_script, summary_rows, args.dataset_title)

    manifest = {
        "dataset": args.dataset_tag,
        "description": f"北京医院外部 COPD {args.dataset_tag} 原始病例组装与 4D 重抽取准备",
        "source_manifest": str(manifest_json.resolve()),
        "input_cases_root": str((out_root / "input_cases").resolve()),
        "raw_texts_root": str((out_root / "raw_texts").resolve()),
        "output_root": str((out_root / "four_d_outputs").resolve()),
        "run_script": str(run_script.resolve()),
        "rules_file": str(RULE_JSON.resolve()),
        "counts_by_label": dict(Counter(x["label_folder"] for x in manifest_rows)),
        "cases": manifest_rows,
        "summary": summary_rows,
    }
    (out_root / args.manifest_name).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[done] 输出目录: {out_root}")
    print(f"[done] 病例数: {len(manifest_rows)}")
    for row in summary_rows:
        print(f"[done] {row['label_folder']}: {row['n']} 例")


if __name__ == "__main__":
    main()
