# -*- coding: utf-8 -*-
"""
exp_4dchain_v2.py — 修复版四维链数据集（severity 补强 + 时序密度提升）
============================================================
修复点（诚实归因，模型/训练/统计完全复用 exp_4dchain.run4）：
  1. severity 实体补强：从原始病历文本规则抽取 FEV1%pred→GOLD、mMRC、SpO₂ 最低值、
     血气异常标记、6MWT 距离，补入 ents['severity']（原 LLM 抽取的疾病分期字段覆盖率仅 9/30）
  2. 时序密度提升：用原始病历的护理记录（带时间戳的体温/脉搏/呼吸/血压/SpO₂）+
     实验室检查（报告日期 100% 覆盖）+ 医嘱（开始/结束时间）按天重建 seq_data
     （原实现只用 LLM 汇总字段重建，序列非零率仅 6.9%）
  3. X_struct 增强：新增 5 个规则严重度列（FEV1_pct/GOLD/mMRC/SpO2_min/PaCO2_hi），
     所有方法（含 XGBoost 基线）与 LR-raw 均可见，保证公平（不通过削基线求胜）

公平性说明：
  - 五方法（lr_xgboost/kg_gnn/tkg/lstm_transformer/4d_hypergraph）用同一 v2 数据集
  - LR-raw 复用 v2 的规则严重度列重训（见 exp_lrraw.py --v2 分支）
  - 折划分、fold 训练种子、统计方法全部不变

用法：
  python3 exp_4dchain_v2.py            # 构建并跑 seed=42 主实验
  python3 exp_4dchain_v2.py --build-only  # 仅构建并自检
"""
import hashlib
import json
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

plt.rcParams['font.family'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.environ.get('EXPERIMENT_BASE', os.getcwd())
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
sys.path.insert(0, os.environ.get('HYPERGRAPHRAG_PATH',
                                  os.path.join(_REPO, 'third_party', 'HyperGraphRAG')))

import run_hgr_eval as R  # noqa: E402
import exp_run as ERM  # noqa: E402
import exp_4dchain as E4  # noqa: E402
from exp_4dchain import (group_by, parse_dt, first_num, parse_gold,
                         infer_gold_from_basis, LAB_PAT, MED_CATS, MED_CLASS_KW,
                         SYM_KW, severity_grade_to_num)
from exp_run import (KGGNN, TKG, LSTMModel, HyperGraph4D,
                     train_model, xgb_train_fit, metrics_cls, metrics_bin, metrics_reg)

BASE_TR = R.BASE_TR
GROUPS = R.GROUPS


def case_id(pid):
    stem = str(pid).rsplit('.', 1)[0]
    return 'CASE_' + hashlib.sha1(stem.encode('utf-8')).hexdigest()[:12].upper()


def load_raw_cases():
    """case_id(stem) -> 原始病历 dict（三个组全部 json）
    2 个坏 JSON（zy@ZY010002487376 / zy@ZY020002487003）恰为无四维链输出的病例，跳过。"""
    raw = {}
    n_bad = 0
    for group in GROUPS:
        for f in (BASE_TR / group).glob('*.json'):
            try:
                raw[case_id(f.stem)] = json.load(open(f, encoding='utf-8'))
            except Exception:
                n_bad += 1
    print('原始病历载入:', len(raw), '例 | 跳过坏JSON:', n_bad)
    return raw


# ---------------- 原文规则抽取：severity 信号 ----------------
FEV1_PCT_RE = re.compile(r'FEV1[^。；;\n]{0,40}?占预计值\s*(\d{1,3}(?:\.\d+)?)\s*%')
FEV1_PCT_RE2 = re.compile(r'FEV1[^。；;\n]{0,25}?(\d{1,3}(?:\.\d+)?)\s*%')
MMRC_RE = re.compile(r'mMRC(?:评分)?\s*(\d+)\s*分?')
PAO2_RE = re.compile(r'PaO2|动脉血氧分压|氧分压', re.I)
PACO2_RE = re.compile(r'PaCO2|动脉血二氧化碳分压|二氧化碳分压', re.I)
SPO2_NAME = 'SpO₂'
WINDOW = 5  # 泄漏修复：时序输入固定为入院前 5 天窗口（标签=总住院天数，窗口截断消除长度泄漏）


def text_of(raw):
    buf = []

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)
        elif isinstance(o, str):
            buf.append(o)
    walk(raw)
    return '\n'.join(buf)


# CSCS（Clinical Severity Composite Score）：Task 3 的回归目标
CSCS_WEIGHTS = {'gold': 0.30, 'exacerbation': 0.30, 'comorbidity': 0.20, 'resp_failure': 0.20}
EXAC_PAT = re.compile(r'过去一年.{0,15}?急性加重\s*(\d+)\s*次')
EXAC_PAT2 = re.compile(r'每年.{0,8}(?:因)?(?:急性加重|AECOPD|发作).{0,8}?(\d+)\s*次')
EXAC_PAT3 = re.compile(r'平均.*?(?:每年|/年).*?(\d+)[-~]\s*(\d+)\s*次')
RESP_FAIL_RE = re.compile(r'呼吸衰竭|呼衰|II型呼吸衰竭|II型呼衰|Ⅰ型呼吸衰竭|Ⅰ型呼衰|呼吸功能不全')
HOSP_PAST_RE = re.compile(r'过去一年.{0,10}(?:住院|入院)\s*(\d+)\s*次')
BARTHEL_RE = re.compile(r'Barthel|巴氏指数|巴氏评分', re.I)


def compute_exacerbation_count(hist_text):
    """从病史文本估计过去1年急性加重次数。"""
    m = EXAC_PAT.search(hist_text)
    if m:
        return int(m.group(1))
    m = EXAC_PAT2.search(hist_text)
    if m:
        return int(m.group(1))
    m = EXAC_PAT3.search(hist_text)
    if m:
        return int(m.group(2))
    # fallback: 计数"急性加重"出现次数作为粗略下界
    return len(re.findall(r'急性加重', hist_text))


def check_respiratory_failure(hist_text, raw_text):
    """检查是否存在呼吸衰竭。"""
    combined = (hist_text or '') + ' ' + (raw_text or '')
    return 1 if RESP_FAIL_RE.search(combined) else 0


def compute_hospitalization_past_year(hist_text):
    """从病史文本提取过去1年住院次数。"""
    m = HOSP_PAST_RE.search(hist_text)
    if m:
        return int(m.group(1))
    return 0


def compute_cscs(gold, exac_count, comorb_count, resp_failure):
    """CSCS（Clinical Severity Composite Score，Task 3 回归目标）。
    CSCS = 0.30 × GOLD_stage_score + 0.30 × exacerbation_burden
         + 0.20 × comorbidity_burden + 0.20 × respiratory_failure_score
    输出: (cscs_float, components_dict)
    """
    gold_score = gold / 4.0 if (gold is not None and 1 <= gold <= 4) else 0.0
    exac_burden = min(exac_count / 3.0, 1.0)
    comorb_burden = min(comorb_count / 10.0, 1.0) if comorb_count else 0.0
    resp_fail_score = float(resp_failure) if resp_failure else 0.0
    cscs = (CSCS_WEIGHTS['gold'] * gold_score +
            CSCS_WEIGHTS['exacerbation'] * exac_burden +
            CSCS_WEIGHTS['comorbidity'] * comorb_burden +
            CSCS_WEIGHTS['resp_failure'] * resp_fail_score)
    return cscs, {
        'gold_stage': gold, 'gold_score': gold_score,
        'exacerbation_count': exac_count, 'exacerbation_burden': exac_burden,
        'comorbidity_count': comorb_count, 'comorbidity_burden': comorb_burden,
        'respiratory_failure': resp_failure, 'resp_failure_score': resp_fail_score,
    }


def extract_severity(raw):
    """从原始病历返回 dict: fev1_pct/gold/mmrc/spo2_min/paco2_hi/pao2_low/sixmwt
     + exacerbation_count/respiratory_failure/hospitalization_past_year/barthel"""
    out = {'fev1_pct': None, 'gold': None, 'mmrc': None,
           'spo2_min': None, 'paco2_hi': None, 'pao2_low': None, 'sixmwt': None,
           'exacerbation_count': 0, 'respiratory_failure': 0,
           'hospitalization_past_year': 0, 'barthel': None}
    txt = text_of(raw)
    # FEV1%pred → GOLD
    m = FEV1_PCT_RE.search(txt)
    if m:
        out['fev1_pct'] = float(m.group(1))
    else:
        m = FEV1_PCT_RE2.search(txt)
        if m and 'FEV1/FVC' not in txt[max(0, m.start() - 10):m.start()]:
            out['fev1_pct'] = float(m.group(1))
    if out['fev1_pct'] is not None:
        p = out['fev1_pct']
        out['gold'] = 1 if p >= 80 else (2 if p >= 50 else (3 if p >= 30 else 4))
    # mMRC
    m = MMRC_RE.search(txt)
    if m:
        out['mmrc'] = min(4, int(m.group(1)))
    # 血气（定性标记，数值常缺失）
    if PAO2_RE.search(txt):
        out['pao2_low'] = True
    if PACO2_RE.search(txt):
        out['paco2_hi'] = True
    # 呼吸衰竭（从全文）
    if RESP_FAIL_RE.search(txt):
        out['respiratory_failure'] = 1
    # 护理记录 SpO₂ 最低值
    spo2_vals = []
    for n in raw.get('护理记录', []) or []:
        if str(n.get('项目名称', '')) == SPO2_NAME:
            v = n.get('项目值(字符)') or n.get('项目值(数值)')
            f = first_num(v)
            if f is not None and 0 < f <= 100:
                spo2_vals.append(f)
    if spo2_vals:
        out['spo2_min'] = min(spo2_vals)
    # 6MWT
    m = re.search(r'6MWT[^。；;\n]{0,15}?(\d{2,3})', txt)
    if m:
        out['sixmwt'] = float(m.group(1))
    # Barthel
    m = BARTHEL_RE.search(txt)
    if m:
        n = re.search(r'(\d{2,3})', txt[m.start():m.start() + 30])
        if n:
            bart = int(n.group(1))
            if 0 <= bart <= 100:
                out['barthel'] = bart
    return out


# ---------------- 原始病历按天序列（护理记录 + 实验室 + 医嘱） ----------------
def build_raw_seq(raw, admit, max_days):
    """返回 (seq: (max_days,24) ndarray, day_ents: [(day, [ents])])"""
    seq = np.zeros((max_days, 24), dtype=float)
    day_ents = defaultdict(set)
    # 8 生命体征: 0体温 1脉搏 2呼吸 3收缩压 4舒张压 5SpO2(新增) 6心率 7—(保留)
    vital_daily = {}
    for n in raw.get('护理记录', []) or []:
        proj = str(n.get('项目名称', '') or '')
        if proj not in ('体温', '脉搏', '呼吸', '血压', 'SpO₂', '心率'):
            continue
        t = parse_dt(n.get('记录时间'))
        if t is None or admit is None:
            continue
        di = (t - admit).days
        if di < 0 or di >= max_days:
            continue  # 泄漏修复：窗口外记录丢弃（不压入末天）
        vital_daily.setdefault(di, {})
        if proj == '体温':
            f = first_num(n.get('项目值(数值)') or n.get('项目值(字符)'))
            if f is not None:
                vital_daily[di][0] = max(vital_daily[di].get(0, float('-inf')), f)
        elif proj == '脉搏':
            f = first_num(n.get('项目值(数值)') or n.get('项目值(字符)'))
            if f is not None:
                vital_daily[di][1] = max(vital_daily[di].get(1, float('-inf')), f)
        elif proj == '呼吸':
            f = first_num(n.get('项目值(数值)') or n.get('项目值(字符)'))
            if f is not None:
                vital_daily[di][2] = max(vital_daily[di].get(2, float('-inf')), f)
        elif proj == '血压':
            s = str(n.get('项目值(字符)') or n.get('项目值(数值)') or '')
            m = re.search(r'(\d{2,3})\s*/\s*(\d{2,3})', s)
            if m:
                vital_daily[di][3] = max(vital_daily[di].get(3, float('-inf')), float(m.group(1)))
                vital_daily[di][4] = max(vital_daily[di].get(4, float('-inf')), float(m.group(2)))
        elif proj == 'SpO₂':
            f = first_num(n.get('项目值(字符)') or n.get('项目值(数值)'))
            if f is not None and 0 < f <= 100:
                vital_daily[di][5] = max(vital_daily[di].get(5, float('-inf')), f)
        elif proj == '心率':
            f = first_num(n.get('项目值(数值)') or n.get('项目值(字符)'))
            if f is not None:
                vital_daily[di][6] = max(vital_daily[di].get(6, float('-inf')), f)
    for di, kv in vital_daily.items():
        for idx, v in kv.items():
            seq[di, idx] = v
        for idx, kw in ((5, 'SpO2'), (3, 'BP_high')):
            pass
        if 5 in kv and kv[5] < 90:
            day_ents[di].add(('sev', 'spo2_low'))
        if 3 in kv and kv[3] >= 140:
            day_ents[di].add(('sev', 'bp_high'))

    # 7 检验（LAB_PAT）按报告日期
    lab_daily = {}
    for it in raw.get('实验室检查', []) or []:
        name = str(it.get('检验指标名', '') or '') + str(it.get('检验项目', '') or '')
        for key, pat in LAB_PAT.items():
            if re.search(pat, name):
                v = first_num(it.get('检验指标值'))
                if v is not None:
                    di = 0
                    rdate = parse_dt(it.get('报告日期') or it.get('检查时间') or it.get('接收日期'))
                    if rdate is not None and admit is not None:
                        di = (rdate - admit).days
                    if di < 0 or di >= max_days:
                        continue  # 泄漏修复：窗口外检验丢弃（不压入末天）
                    lab_daily.setdefault(di, {})
                    lab_daily[di][key] = max(lab_daily[di].get(key, float('-inf')), v)
    for di, kv in lab_daily.items():
        for i, key in enumerate(LAB_PAT):
            if key in kv:
                seq[di, 8 + i] = kv[key]
                if key == 'CRP' and kv[key] > 10:
                    day_ents[di].add(('lab', 'CRP_high'))
                if key == 'PCT' and kv[key] > 0.5:
                    day_ents[di].add(('lab', 'PCT_high'))
                if key == 'WBC' and kv[key] > 10:
                    day_ents[di].add(('lab', 'WBC_high'))
                if key == 'NEUT' and kv[key] > 75:
                    day_ents[di].add(('lab', 'NEUT_high'))

    # 9 用药类按开始时间
    for o in raw.get('医嘱', []) or []:
        if not isinstance(o, dict):
            continue
        name = str(o.get('医嘱名称', '') or '')
        cat = None
        for c_ in MED_CATS:
            if any(k in name for k in MED_CLASS_KW[c_]):
                cat = c_
                break
        if cat is None:
            continue
        t = parse_dt(o.get('开始时间'))
        di = 0
        if t is not None and admit is not None:
            di = (t - admit).days
        if di < 0 or di >= max_days:
            continue  # 泄漏修复：窗口外医嘱丢弃（不压入末天）
        seq[di, 15 + MED_CATS.index(cat)] += 1
        day_ents[di].add(('med', cat))
    return seq, [(d, sorted(es)) for d, es in sorted(day_ents.items())]


# ---------------- v2 数据集构建 ----------------
def build_v2(cases, raw):
    pids = sorted(cases)
    DIM_ORDER = ['hist', 'evidence', 'treat', 'severity']
    X_struct = []
    seq_data = {}
    entity_data = {}
    day_ent_data = {}
    meta = []
    ent_counter = Counter()

    days_all = []
    for pid in pids:
        ident = cases[pid]['ident']
        a = parse_dt(ident.get('入院时间', ''))
        d = parse_dt(ident.get('出院时间', ''))
        days_all.append(max(0, (d - a).days) if (a and d) else 8.0)
    # 泄漏修复（v3）：固定前 W 天窗口，消除"序列长度=住院天数"标签泄漏
    # （v2 中序列非零天数与住院天数 Pearson=0.976，LSTM 直接"数天数"作弊）
    max_days = WINDOW
    print('v3 泄漏修复：固定前 %d 天窗口（原 v2 为 95 分位 %d 天）' % (WINDOW, int(np.ceil(np.percentile(days_all, 95)))))

    sev_cols = ['FEV1_pct', 'GOLD', 'mMRC', 'SpO2_min', 'PaCO2_hi']
    sev_missing = {k: 0 for k in sev_cols}
    n_raw_miss = 0

    for pid in pids:
        case = cases[pid]
        ident = case['ident']
        label3 = {'A_stable': 0, 'B_ae': 1, 'C_infection': 2}[case['label']]
        days = max(0, (parse_dt(ident.get('出院时间', '')) - parse_dt(ident.get('入院时间', ''))).days) \
            if (parse_dt(ident.get('出院时间', '')) and parse_dt(ident.get('入院时间', ''))) else None

        r = raw.get(pid)
        if r is None:
            n_raw_miss += 1
            r = {}

        # ---- 人口学 ----
        demo = group_by(case, 'history_chain', '患者人口学基础信息')
        age = first_num(demo.get('年龄'))
        sex = 1 if str(demo.get('性别', '')) == '男' else 0

        # ---- 病史文本（主诉/现病史/个人史）----
        hist = group_by(case, 'history_chain', '全维度病史信息')
        chief = str(hist.get('主诉', '') or '')
        present = str(hist.get('现病史', '') or '')
        personal = str(hist.get('个人史', '') or '')
        past_hist = str(hist.get('既往史', '') or '')
        smoking = 1 if ('烟' in personal or '烟' in present) else 0
        # 全文拼接（供 CSCS 急性加重计数与呼吸衰竭判断）
        full_hist_text = chief + ' ' + present + ' ' + past_hist

        # ---- 合并症（其他诊断 + 并发症，去重）----
        diag = group_by(case, 'evidence_chain', '诊断信息')
        other_dx = list(diag.get('其他诊断', []) or [])
        comp = group_by(case, 'severity_chain', '并发症信息')
        comp_list = list(comp.get('并发症列表', []) or [])
        comorb = sorted(set(str(x) for x in other_dx + comp_list if str(x).strip() and str(x) != '无'))

        # ---- LLM 四维链检验/影像（供 hist/evidence 实体，与 v1 一致）----
        labs_info = group_by(case, 'evidence_chain', '检查检验信息')
        lab_items = list(labs_info.get('检验项目列表', []) or [])
        lab_max = {}
        for key, pat in LAB_PAT.items():
            vals = []
            for it in lab_items:
                name = str(it.get('检验项目', '') or '') + str(it.get('检验指标名', '') or '')
                if re.search(pat, name):
                    v = first_num(it.get('检验指标值'))
                    if v is not None:
                        vals.append(v)
            lab_max[key] = max(vals) if vals else None
        img_items = list(labs_info.get('影像检查列表', []) or [])

        # ---- 实体（四维）----
        ents = defaultdict(set)
        symp = group_by(case, 'evidence_chain', '症状体征信息')
        sym_text = json.dumps(symp, ensure_ascii=False)
        for kw in SYM_KW:
            if kw in sym_text:
                ents['hist'].add(('sym', kw))
        for dx in comorb:
            ents['hist'].add(('diag', dx))
        for key, v in lab_max.items():
            if v is not None:
                if key == 'CRP' and v > 10:
                    ents['evidence'].add(('lab', 'CRP_high'))
                if key == 'PCT' and v > 0.5:
                    ents['evidence'].add(('lab', 'PCT_high'))
                if key == 'WBC' and v > 10:
                    ents['evidence'].add(('lab', 'WBC_high'))
                if key == 'NEUT' and v > 75:
                    ents['evidence'].add(('lab', 'NEUT_high'))
                if key == 'HGB' and v < 110:
                    ents['evidence'].add(('lab', 'HGB_low'))
                if key == 'ALB' and v < 35:
                    ents['evidence'].add(('lab', 'ALB_low'))
        for it in img_items:
            if isinstance(it, dict):
                ents['evidence'].add(('img', str(it.get('检查项目', '') or '影像')[:20]))

        # ---- 治疗轨迹（LLM 四维链用药）----
        meds_info = group_by(case, 'care_chain', '药物治疗信息')
        med_orders = list(meds_info.get('用药医嘱列表', []) or [])
        med_cat_count = Counter()
        for o in med_orders:
            if not isinstance(o, dict):
                continue
            name = str(o.get('医嘱名称', '') or '')
            for c_ in MED_CATS:
                if any(k in name for k in MED_CLASS_KW[c_]):
                    med_cat_count[c_] += 1
                    break
        for cat, n in med_cat_count.items():
            if n > 0:
                ents['treat'].add(('med', cat))

        # ---- severity 补强（规则从原文抽取）----
        sv = extract_severity(r)
        # 从病史文本补充急性加重计数（以 LLM 文本为主，规则为后备）
        exac_count = compute_exacerbation_count(full_hist_text)
        raw_text = text_of(r) if r else ''
        resp_fail = sv['respiratory_failure'] or check_respiratory_failure(full_hist_text, raw_text)
        hosp_past = compute_hospitalization_past_year(full_hist_text)
        if sv['gold'] is not None:
            ents['severity'].add(('score', f'gold_{sv["gold"]}'))
        if sv['mmrc'] is not None:
            ents['severity'].add(('score', f'mmrc_{sv["mmrc"]}'))
        if sv['spo2_min'] is not None:
            ents['severity'].add(('score', 'spo2_low' if sv['spo2_min'] < 90 else 'spo2_ok'))
        if sv['paco2_hi']:
            ents['severity'].add(('sev', 'paco2_high'))
        if sv['pao2_low']:
            ents['severity'].add(('sev', 'pao2_rec'))
        if sv['sixmwt'] is not None:
            ents['severity'].add(('sev', 'sixmwt_lt300' if sv['sixmwt'] < 300 else 'sixmwt_ok'))
        # 新增 Dim 4 字段（Step 1.4 更新）：呼吸衰竭 / Barthel / 急性加重严重度 / 过去一年住院
        if resp_fail:
            ents['severity'].add(('sev', 'resp_failure'))
        if sv.get('barthel') is not None:
            ents['severity'].add(('score', f'barthel_{"low" if sv["barthel"] < 60 else "ok"}'))
        if exac_count >= 2:
            ents['severity'].add(('score', 'exacerbation_frequent'))
        elif exac_count >= 1:
            ents['severity'].add(('score', 'exacerbation_once'))
        if hosp_past >= 2:
            ents['severity'].add(('sev', 'hosp_past_frequent'))
        elif hosp_past >= 1:
            ents['severity'].add(('sev', 'hosp_past_once'))
        # LLM 侧 GOLD 兜底（原文无 FEV1 时）
        sev_llm = group_by(case, 'severity_chain', '疾病分期与严重程度')
        gold_llm = parse_gold(sev_llm.get('疾病分期'))
        if gold_llm is None:
            gold_llm = infer_gold_from_basis(' '.join(str(x) for x in (sev_llm.get('分期依据') or []) if str(x).strip()))
        if gold_llm is not None and sv['gold'] is None:
            ents['severity'].add(('score', f'gold_{gold_llm}'))
            sv['gold'] = gold_llm
            if sv['fev1_pct'] is None:
                sv['fev1_pct'] = {1: 85.0, 2: 65.0, 3: 40.0, 4: 20.0}[gold_llm]
        if len(comp_list) >= 3:
            ents['severity'].add(('score', 'comorb_heavy'))

        entity_data[pid] = ents

        # ---- 时序序列（原始病历重建）----
        admit = parse_dt(ident.get('入院时间', ''))
        seq, day_ents = build_raw_seq(r, admit, max_days)
        # 单点 LLM 生命体征兜底（原文无护理记录时）
        if not np.any(seq[:, :8]):
            symp_v = group_by(case, 'evidence_chain', '症状体征信息')
            vt = symp_v.get('生命体征', {}) if isinstance(symp_v.get('生命体征'), dict) else {}
            for vk, idx in [('体温', 0), ('脉搏', 1), ('呼吸', 2)]:
                sub = vt.get(vk)
                if isinstance(sub, dict) and first_num(sub.get('数值')) is not None:
                    seq[0, idx] = first_num(sub['数值'])
            if lab_max.get('CRP') is not None:
                seq[0, 8] = lab_max['CRP']
        seq_data[pid] = seq
        # day_ent_data = 原始序列实体 + LLM 实体（hist/severity 归 day0）
        day_e_map = {d: set(es) for d, es in day_ents}
        for dm, es in ents.items():
            for e in es:
                day_e_map.setdefault(0, set())
                if dm in ('hist', 'severity'):
                    day_e_map[0].add(e)
        day_ent_data[pid] = [(d, sorted(es)) for d, es in sorted(day_e_map.items())]
        all_case_ents = set(x for es in ents.values() for x in es)
        all_case_ents.update(x for _, es in day_ent_data[pid] for x in es)
        for e in all_case_ents:
            ent_counter[e] += 1

        # ---- X_struct（24 列 LLM 侧 + 5 列规则严重度）----
        row = [age if age else np.nan, sex, smoking, len(comorb)]
        for key in LAB_PAT:
            row.append(lab_max.get(key) if lab_max.get(key) is not None else np.nan)
        row.append(first_num(symp.get('生命体征', {}).get('体温', {}).get('数值')) if isinstance(symp.get('生命体征'), dict) else np.nan)
        row.append(first_num(symp.get('生命体征', {}).get('脉搏', {}).get('数值')) if isinstance(symp.get('生命体征'), dict) else np.nan)
        row.append(first_num(symp.get('生命体征', {}).get('呼吸', {}).get('数值')) if isinstance(symp.get('生命体征'), dict) else np.nan)
        bp_s = None
        if isinstance(symp.get('生命体征'), dict) and isinstance(symp['生命体征'].get('血压'), dict):
            m = re.search(r'(\d+)\s*/\s*(\d+)', str(symp['生命体征']['血压'].get('数值', '')))
            if m:
                bp_s = float(m.group(1))
        row.append(bp_s if bp_s else np.nan)
        for cat in MED_CATS:
            row.append(med_cat_count.get(cat, 0))
        # +5 规则严重度列
        row.append(sv['fev1_pct'] if sv['fev1_pct'] is not None else np.nan)
        row.append(sv['gold'] if sv['gold'] is not None else np.nan)
        row.append(sv['mmrc'] if sv['mmrc'] is not None else np.nan)
        row.append(sv['spo2_min'] if sv['spo2_min'] is not None else np.nan)
        row.append(1.0 if sv['paco2_hi'] else 0.0)
        X_struct.append(row)
        # CSCS 复合评分（Task 3 回归目标，替代原住院天数标签）
        cscs, cscs_comp = compute_cscs(sv['gold'], exac_count, len(comorb), resp_fail)
        meta.append({'pid': pid, 'group': case['group'], 'label3': label3,
                     'days': cscs,  # Task 3 回归目标 = CSCS（替代住院天数）
                     'hospital_days': days,  # 原始住院天数（参考用）
                     'cscs_components': cscs_comp})
        for k, v in sev_missing.items():
            if v == 0:
                pass
        for k in sev_cols:
            idx = 24 + sev_cols.index(k)
            if row[idx] is None or (isinstance(row[idx], float) and np.isnan(row[idx])):
                sev_missing[k] += 1

    # ---- 实体词典（频次 >=2）----
    ent2id = {}
    for e, c in ent_counter.most_common():
        if c >= 2:
            ent2id[e] = len(ent2id)
    print('v2 实体总数:', len(ent_counter), '| 过滤后(>=2):', len(ent2id))

    X_struct = np.array(X_struct, dtype=float)
    print('v2 X_struct 形状:', X_struct.shape, '| 规则严重度列缺失:',
          {k: v for k, v in sev_missing.items()})
    print('v2 原始病历未命中例数:', n_raw_miss)
    return {
        'pids': pids, 'X_struct': X_struct, 'ent2id': ent2id,
        'entity_data': entity_data, 'seq_data': seq_data, 'max_days': max_days,
        'day_ent_data': day_ent_data, 'meta': meta,
        'sev_cols': sev_cols,
    }


def inspect(ds4):
    print('X_struct shape:', ds4['X_struct'].shape)
    labels = np.array([m['label3'] for m in ds4['meta']])
    days = np.array([m['days'] if m['days'] else np.nan for m in ds4['meta']])
    hdays = np.array([m.get('hospital_days') if m.get('hospital_days') else np.nan for m in ds4['meta']])
    seq_ok = sum(1 for pid in ds4['pids'] if ds4['seq_data'][pid] is not None)
    tot = len(ds4['pids']) * ds4['max_days'] * 24
    nz_cells = sum(int((ds4['seq_data'][pid] != 0).sum()) for pid in ds4['pids'])
    print('CSCS(回归目标): mean=%.4f std=%.4f min=%.4f max=%.4f' % (np.nanmean(days), np.nanstd(days), np.nanmin(days), np.nanmax(days)))
    print('住院天数(参考): mean=%.1f std=%.1f' % (np.nanmean(hdays), np.nanstd(hdays)))
    for g in [0, 1, 2]:
        dd = days[labels == g]
        hd = hdays[labels == g]
        print(f'  组{g} n={(labels==g).sum()} CSCS=%.4f±%.4f | days=%.1f±%.1f' % (np.nanmean(dd), np.nanstd(dd), np.nanmean(hd), np.nanstd(hd)))
    print('序列非零单元率: %.2f%% (%d/%d)' % (100 * nz_cells / tot, nz_cells, tot))
    for dm in ['hist', 'evidence', 'treat', 'severity']:
        c = sum(1 for pid in ds4['pids'] for e in ds4['entity_data'][pid].get(dm, set()) if e in ds4['ent2id'])
        print(f'  维度{dm} 有效实体关联数:', c)


def run_v2(ds4, seed=42, do_ablation=True):
    """完全复用 exp_4dchain.run4（模型/折/训练/统计），仅替换数据表示层。"""
    out, abl_res, per_class_f1, hardest, mcnemar_vs, stats, fold_preds = \
        E4.run4(ds4, seed=seed, do_ablation=do_ablation)
    return out, abl_res, per_class_f1, hardest, mcnemar_vs, stats, fold_preds


def struct_feature_names(ds4):
    return [
        'age', 'sex_male', 'smoking', 'comorb_count',
        'lab_CRP', 'lab_PCT', 'lab_WBC', 'lab_NEUT', 'lab_HGB', 'lab_ALB', 'lab_PLT',
        'vital_temp', 'vital_pulse', 'vital_resp', 'vital_sbp',
        'med_antibiotic', 'med_steroid', 'med_bronchodilator', 'med_expectorant',
        'med_anticoagulant', 'med_diuretic', 'med_vasoactive', 'med_oxygen_vent', 'med_nutrition',
    ] + [f'sev_{k}' for k in ds4.get('sev_cols', ['FEV1_pct', 'GOLD', 'mMRC', 'SpO2_min', 'PaCO2_hi'])]


def _safe_ent_name(ent):
    return '::'.join(str(x) for x in ent)


def build_4d_xgb_features(ds4, return_feature_names=False, ref_ent2id=None, ref_max_days=None):
    """把 4D 时序超图转成适合小样本的图特征，供 Task1/2 的 XGBoost 头使用。"""
    dims = ['hist', 'evidence', 'treat', 'severity']
    prefix_to_dim = {
        'sym': 'hist', 'diag': 'hist',
        'lab': 'evidence', 'img': 'evidence',
        'med': 'treat',
        'score': 'severity', 'sev': 'severity',
    }
    ent2id = ref_ent2id if ref_ent2id is not None else ds4['ent2id']
    V = len(ent2id)
    max_days = ref_max_days if ref_max_days is not None else ds4['max_days']
    graph_names = []
    for dm in dims:
        graph_names += [
            f'{dm}_unique_entities', f'{dm}_active_days',
            f'{dm}_first_day', f'{dm}_last_day', f'{dm}_mean_daily_count',
        ]
        graph_names += [f'{dm}_day{d}_count' for d in range(max_days)]
    for a, b in combinations(dims, 2):
        graph_names += [f'{a}_{b}_coactive_days'] + [f'{a}_{b}_day{d}_coactive' for d in range(max_days)]
    ent_names = [f'ent::{_safe_ent_name(e)}' for e, _ in sorted(ent2id.items(), key=lambda kv: kv[1])]

    def feat_for_pid(pid):
        by_day = {d: {dm: set() for dm in dims} for d in range(max_days)}
        ent_present = np.zeros(V, dtype=float)
        for d, es in ds4['day_ent_data'].get(pid, []):
            if d >= max_days:
                continue
            for e in es:
                if e in ent2id:
                    ent_present[ent2id[e]] = 1.0
                dm = prefix_to_dim.get(e[0])
                if dm is not None:
                    by_day[d][dm].add(e)
        feats = []
        for dm in dims:
            day_counts = np.array([len(by_day[d][dm]) for d in range(max_days)], dtype=float)
            active = np.where(day_counts > 0)[0]
            total_unique = len(set().union(*[by_day[d][dm] for d in range(max_days)]))
            feats += [total_unique, float(active.size)]
            if active.size:
                feats += [float(active[0]), float(active[-1]), float(day_counts[active].mean())]
            else:
                feats += [0.0, 0.0, 0.0]
            feats += day_counts.tolist()
        for a, b in combinations(dims, 2):
            pair_days = [1.0 if by_day[d][a] and by_day[d][b] else 0.0 for d in range(max_days)]
            feats += [sum(pair_days)] + pair_days
        return np.concatenate([np.array(feats, dtype=float), ent_present])

    X4 = np.stack([feat_for_pid(pid) for pid in ds4['pids']])
    X = np.concatenate([ds4['X_struct'], X4], axis=1)
    if return_feature_names:
        return X, struct_feature_names(ds4) + graph_names + ent_names
    return X


def map_copd3_label_from_diag(diag):
    s = str(diag or '')
    if '慢性阻塞' not in s and 'COPD' not in s and '慢阻肺' not in s:
        return None
    if ('感染' in s) or ('肺炎' in s) or ('下呼吸道' in s):
        return 'C_infection'
    if ('急性加重' in s) or ('AECOPD' in s):
        return 'B_ae'
    return 'A_stable'


def load_external_chain_cases(folder):
    """从测试打分目录加载外部四维链病例，并过滤为 COPD 三类。"""
    bucket_files = {
        'history_chain': 'history_background_similarity.json',
        'evidence_chain': 'clinical_evidence_chain_similarity.json',
        'care_chain': 'treatment_trajectory_similarity.json',
        'severity_chain': 'clinical_severity_similarity.json',
    }
    folder = Path(folder)
    raw_by_bucket = {}
    cases = {}
    for bucket, fname in bucket_files.items():
        raw = json.load(open(folder / fname, encoding='utf-8'))
        per = {}
        for item in raw.get('data', []):
            meta = item.get('meta', {})
            cid = meta.get('case_id')
            diag = meta.get('diag')
            label = map_copd3_label_from_diag(diag)
            if cid is None or label is None:
                continue
            per[cid] = item.get('groups', [])
            cases.setdefault(cid, {
                'group': diag,
                'label': label,
                'ident': {'病例编号': cid, '主诊断': diag, '入院时间': '', '出院时间': ''},
                'groups': {},
                'meta': meta,
            })
        raw_by_bucket[bucket] = per
    keep_ids = set.intersection(*(set(v.keys()) for v in raw_by_bucket.values()))
    out = {}
    for cid in sorted(keep_ids):
        case = cases[cid]
        demo = {}
        for g in raw_by_bucket['history_chain'][cid]:
            if g.get('g') == '患者人口学基础信息' and isinstance(g.get('d'), dict):
                demo = g.get('d')
                break
        case['ident'] = {
            '病例编号': cid,
            '主诊断': case['meta'].get('diag', ''),
            '入院时间': str(demo.get('入院时间', '') or ''),
            '出院时间': str(demo.get('出院时间', '') or ''),
        }
        case['groups'] = {b: raw_by_bucket[b][cid] for b in bucket_files}
        out[cid] = case
    print('外部链级病例载入:', len(out), '例 | 路径:', str(folder))
    return out


def build_external_chain_dataset(cases, max_days=WINDOW):
    """用外部四维链构建与训练集对齐的 29 列结构化特征与 day_ent_data（无原始病历增强）。"""
    pids = sorted(cases)
    X_struct, day_ent_data, meta = [], {}, []
    sev_cols = ['FEV1_pct', 'GOLD', 'mMRC', 'SpO2_min', 'PaCO2_hi']

    for pid in pids:
        case = cases[pid]
        ident = case['ident']
        label3 = {'A_stable': 0, 'B_ae': 1, 'C_infection': 2}[case['label']]
        demo = group_by(case, 'history_chain', '患者人口学基础信息')
        age = first_num(demo.get('年龄'))
        sex = 1 if str(demo.get('性别', '')) == '男' else 0

        hist = group_by(case, 'history_chain', '全维度病史信息')
        chief = str(hist.get('主诉', '') or '')
        present = str(hist.get('现病史', '') or '')
        personal = str(hist.get('个人史', '') or '')
        smoking = 1 if ('烟' in personal or '烟' in present) else 0

        diag = group_by(case, 'evidence_chain', '诊断信息')
        other_dx = list(diag.get('其他诊断', []) or diag.get('other_diag', []) or [])
        comp = group_by(case, 'severity_chain', '并发症信息')
        comp_list = list(comp.get('并发症列表', []) or comp.get('complications', []) or [])
        comorb = sorted(set(str(x) for x in other_dx + comp_list if str(x).strip() and str(x) != '无'))

        labs_info = group_by(case, 'evidence_chain', '检查检验信息')
        lab_items = list(labs_info.get('检验项目列表', []) or [])
        lab_max = {}
        for key, pat in LAB_PAT.items():
            vals = []
            for it in lab_items:
                name = str(it.get('检验项目', '') or '') + str(it.get('检验指标名', '') or '')
                if re.search(pat, name):
                    v = first_num(it.get('检验指标值'))
                    if v is not None:
                        vals.append(v)
            lab_max[key] = max(vals) if vals else None

        admit = parse_dt(ident.get('入院时间', ''))
        lab_daily = {}
        for it in lab_items:
            name = str(it.get('检验项目', '') or '') + str(it.get('检验指标名', '') or '')
            for key, pat in LAB_PAT.items():
                if re.search(pat, name):
                    v = first_num(it.get('检验指标值'))
                    if v is not None:
                        di = 0
                        rdate = parse_dt(it.get('报告日期') or it.get('检查时间') or it.get('接收日期'))
                        if rdate is not None and admit is not None:
                            di = max(0, (rdate - admit).days)
                        di = min(di, max_days - 1)
                        lab_daily.setdefault(di, {})
                        lab_daily[di][key] = max(lab_daily[di].get(key, float('-inf')), v)

        symp = group_by(case, 'evidence_chain', '症状体征信息')
        vt = symp.get('生命体征', {}) if isinstance(symp.get('生命体征'), dict) else {}
        vital_raw_vals = {}
        for vk in ['体温', '脉搏', '呼吸', '血压']:
            sub = vt.get(vk) if isinstance(vt, dict) else None
            if isinstance(sub, dict):
                vital_raw_vals[vk] = str(sub.get('数值', ''))
            else:
                vital_raw_vals[vk] = ''
        if not any(vital_raw_vals.values()):
            pe = str(symp.get('体格检查', '') or '')
            pat_map = {
                '体温': r'体温[:：]\s*([0-9]+(?:\.[0-9]+)?)',
                '脉搏': r'脉搏[:：]\s*([0-9]+(?:\.[0-9]+)?)',
                '呼吸': r'呼吸[:：]\s*([0-9]+(?:\.[0-9]+)?)',
                '血压': r'血压[:：]\s*([0-9]+(?:/[0-9]+)?)',
            }
            for vk, pat in pat_map.items():
                m = re.search(pat, pe)
                if m:
                    vital_raw_vals[vk] = m.group(1)
        vital = {k: first_num(v) for k, v in vital_raw_vals.items()}
        bp_s = None
        if vital_raw_vals.get('血压'):
            m = re.search(r'(\d+)\s*/\s*(\d+)', vital_raw_vals['血压'])
            if m:
                bp_s = float(m.group(1))
            else:
                bp_s = first_num(vital_raw_vals['血压'])

        meds_info = group_by(case, 'care_chain', '药物治疗信息')
        med_orders = list(meds_info.get('用药医嘱列表', []) or [])
        med_courses = list(meds_info.get('用药疗程', []) or [])
        course_by_name = {}
        for c_ in med_courses:
            if isinstance(c_, dict) and c_.get('医嘱名称'):
                course_by_name[str(c_['医嘱名称'])] = c_
        med_cat_count = Counter()
        med_events = []
        for o in med_orders:
            if not isinstance(o, dict):
                continue
            name = str(o.get('医嘱名称', '') or '')
            cat = None
            for c_ in MED_CATS:
                if any(k in name for k in MED_CLASS_KW[c_]):
                    cat = c_
                    break
            if cat is None:
                cls_ = str(o.get('药物分类', '') or '')
                if any(k in name for k in MED_CLASS_KW['抗生素']) or '抗感染' in cls_:
                    cat = '抗生素'
            if cat:
                med_cat_count[cat] += 1
                t = parse_dt(course_by_name.get(name, {}).get('开始时间') if name in course_by_name else None)
                if t is None:
                    t = parse_dt(o.get('开始时间'))
                med_events.append((t, cat))

        sev = group_by(case, 'severity_chain', '疾病分期与严重程度')
        gold = parse_gold(sev.get('疾病分期') or sev.get('stage'))
        if gold is None:
            gold = infer_gold_from_basis(' '.join(str(x) for x in ((sev.get('分期依据') or sev.get('basis') or [])) if str(x).strip()))
        func = group_by(case, 'severity_chain', '功能评分信息')
        mmrc = None
        score_lists = []
        for lst_key in ['入院功能评分', '出院功能评分', 'scores']:
            items = func.get(lst_key, []) or []
            if isinstance(items, dict):
                items = [items]
            if isinstance(items, list):
                score_lists.extend(items)
        for it in score_lists:
            if isinstance(it, dict) and it.get('评分项目'):
                pj = str(it['评分项目'])
                val = first_num(it.get('评分值'))
                if 'mMRC' in pj and mmrc is None and val is not None:
                    mmrc = int(val)
        spo2_min = None
        bg = group_by(case, 'evidence_chain', '检查检验信息')
        for it in list(bg.get('血气分析', []) or []):
            val = first_num(it.get('SpO2') or it.get('spo2_percent') or it.get('氧饱和度'))
            if val is not None:
                spo2_min = val if spo2_min is None else min(spo2_min, val)
        paco2_hi = 0.0
        blood_gas = bg.get('血气分析') or []
        if isinstance(blood_gas, dict):
            blood_gas = [blood_gas]
        for it in blood_gas:
            v = first_num(it.get('PaCO2') or it.get('paco2_mmHg') or it.get('二氧化碳分压'))
            if v is not None and v > 45:
                paco2_hi = 1.0
        fev1_pct = None
        basis_text = ' '.join(str(x) for x in ((sev.get('分期依据') or sev.get('basis') or [])) if str(x).strip())
        m = re.search(r'FEV1[^。；;\n]{0,30}?(\d{1,3}(?:\.\d+)?)\s*[%％]', basis_text)
        if m:
            fev1_pct = float(m.group(1))
        elif gold is not None:
            fev1_pct = {1: 85.0, 2: 65.0, 3: 40.0, 4: 20.0}[gold]

        ents = defaultdict(set)
        sym_text = ' '.join([chief, present, str(symp.get('主要症状', '') or ''), str(symp.get('现病史摘要', '') or '')])
        for kw in SYM_KW:
            if kw in sym_text:
                ents['hist'].add(('sym', kw))
        for dx in comorb:
            ents['hist'].add(('diag', dx))
        for key, v in lab_max.items():
            if v is not None:
                if key == 'CRP' and v > 10: ents['evidence'].add(('lab', 'CRP_high'))
                if key == 'PCT' and v > 0.5: ents['evidence'].add(('lab', 'PCT_high'))
                if key == 'WBC' and v > 10: ents['evidence'].add(('lab', 'WBC_high'))
                if key == 'NEUT' and v > 75: ents['evidence'].add(('lab', 'NEUT_high'))
                if key == 'HGB' and v < 110: ents['evidence'].add(('lab', 'HGB_low'))
                if key == 'ALB' and v < 35: ents['evidence'].add(('lab', 'ALB_low'))
        img_items = list(labs_info.get('影像检查列表', []) or [])
        for it in img_items:
            if isinstance(it, dict):
                ents['evidence'].add(('img', str(it.get('检查项目', '') or '影像')[:20]))
        for cat, n in med_cat_count.items():
            if n > 0:
                ents['treat'].add(('med', cat))
        if gold is not None:
            ents['severity'].add(('score', f'gold_{gold}'))
        if mmrc is not None:
            ents['severity'].add(('score', f'mmrc_{mmrc}'))
        if spo2_min is not None:
            ents['severity'].add(('score', 'spo2_low' if spo2_min < 90 else 'spo2_ok'))
        if paco2_hi:
            ents['severity'].add(('sev', 'paco2_high'))
        if len(comp_list) >= 3:
            ents['severity'].add(('score', 'comorb_heavy'))

        day_e = defaultdict(set)
        for e in ents['hist']:
            day_e[0].add(e)
        for e in ents['severity']:
            day_e[0].add(e)
        for di, kv in lab_daily.items():
            for key, v in kv.items():
                if key == 'CRP' and v > 10: day_e[di].add(('lab', 'CRP_high'))
                if key == 'PCT' and v > 0.5: day_e[di].add(('lab', 'PCT_high'))
                if key == 'WBC' and v > 10: day_e[di].add(('lab', 'WBC_high'))
                if key == 'NEUT' and v > 75: day_e[di].add(('lab', 'NEUT_high'))
                if key == 'HGB' and v < 110: day_e[di].add(('lab', 'HGB_low'))
                if key == 'ALB' and v < 35: day_e[di].add(('lab', 'ALB_low'))
        for it in img_items:
            day_e[0].add(('img', str(it.get('检查项目', '') or '影像')[:20]))
        for t, cat in med_events:
            di = 0
            if t is not None and admit is not None:
                di = max(0, (t - admit).days)
            di = min(di, max_days - 1)
            day_e[di].add(('med', cat))
        day_ent_data[pid] = [(d, sorted(es)) for d, es in sorted(day_e.items())]

        row = [age if age else np.nan, sex, smoking, len(comorb)]
        for key in LAB_PAT:
            row.append(lab_max.get(key) if lab_max.get(key) is not None else np.nan)
        row.append(vital.get('体温') if vital.get('体温') is not None else np.nan)
        row.append(vital.get('脉搏') if vital.get('脉搏') is not None else np.nan)
        row.append(vital.get('呼吸') if vital.get('呼吸') is not None else np.nan)
        row.append(bp_s if bp_s else np.nan)
        for cat in MED_CATS:
            row.append(med_cat_count.get(cat, 0))
        row.append(fev1_pct if fev1_pct is not None else np.nan)
        row.append(gold if gold is not None else np.nan)
        row.append(mmrc if mmrc is not None else np.nan)
        row.append(spo2_min if spo2_min is not None else np.nan)
        row.append(paco2_hi)
        X_struct.append(row)
        meta.append({'pid': pid, 'group': case['group'], 'label3': label3, 'days': None, 'diag': case['ident']['主诊断']})

    return {
        'pids': pids,
        'X_struct': np.array(X_struct, dtype=float),
        'day_ent_data': day_ent_data,
        'meta': meta,
        'sev_cols': sev_cols,
    }


def prep_by_train(Xtr, Xte):
    """用训练集统计量标准化训练/外部特征。"""
    med = np.nanmedian(Xtr, axis=0)
    med[np.isnan(med)] = 0.0
    std = np.nanstd(Xtr, axis=0)
    std[np.isnan(std)] = 1.0
    std[std < 1e-6] = 1.0
    Xtr_i = np.where(np.isnan(Xtr), med, Xtr)
    Xte_i = np.where(np.isnan(Xte), med, Xte)
    return (Xtr_i - med) / std, (Xte_i - med) / std


def run_task12_external_4dxgb(train_ds, external_folder):
    """训练集拟合，外部 COPD 三类链级验证。"""
    ext_cases = load_external_chain_cases(external_folder)
    ext_ds = build_external_chain_dataset(ext_cases, max_days=train_ds['max_days'])
    Xtr4 = build_4d_xgb_features(train_ds, ref_ent2id=train_ds['ent2id'], ref_max_days=train_ds['max_days'])
    Xte4 = build_4d_xgb_features(ext_ds, ref_ent2id=train_ds['ent2id'], ref_max_days=train_ds['max_days'])
    Xtr4s, Xte4s = prep_by_train(Xtr4, Xte4)

    ytr3 = np.array([m['label3'] for m in train_ds['meta']])
    ytr2 = (ytr3 > 0).astype(int)
    yte3 = np.array([m['label3'] for m in ext_ds['meta']])
    yte2 = (yte3 > 0).astype(int)

    m1 = xgb_train_fit(Xtr4s, ytr3, 'cls')
    prob1 = m1.predict_proba(Xte4s)
    pred1 = prob1.argmax(1)
    m2 = xgb_train_fit(Xtr4s, ytr2, 'bin')
    prob2 = m2.predict_proba(Xte4s)[:, 1]

    # LR/XGBoost 基线：仅结构化特征，可在外部链级数据上直接复现
    XtrL, XteL = prep_by_train(train_ds['X_struct'], ext_ds['X_struct'])
    bl1 = xgb_train_fit(XtrL, ytr3, 'cls')
    bl1_prob = bl1.predict_proba(XteL)
    bl1_pred = bl1_prob.argmax(1)
    bl2 = xgb_train_fit(XtrL, ytr2, 'bin')
    bl2_prob = bl2.predict_proba(XteL)[:, 1]

    out = {
        'external_n': len(ext_ds['pids']),
        'class_counts': {str(k): int((yte3 == k).sum()) for k in [0, 1, 2]},
        '4d_xgb': {'cls': metrics_cls(yte3, pred1), 'bin': metrics_bin(yte2, prob2)},
        'lr_xgboost': {'cls': metrics_cls(yte3, bl1_pred), 'bin': metrics_bin(yte2, bl2_prob)},
        'cases': [{'pid': m['pid'], 'diag': m.get('diag', ''), 'label3': int(m['label3'])} for m in ext_ds['meta']],
    }
    fold_preds = {
        '4d_xgb': {'cls_y': [int(x) for x in yte3], 'cls_p': [int(x) for x in pred1], 'bin_y': [int(x) for x in yte2], 'bin_p': [float(x) for x in prob2]},
        'lr_xgboost': {'cls_y': [int(x) for x in yte3], 'cls_p': [int(x) for x in bl1_pred], 'bin_y': [int(x) for x in yte2], 'bin_p': [float(x) for x in bl2_prob]},
    }
    return out, fold_preds


def align_lrraw_struct(ds4):
    """把规则侧结构化特征对齐到当前 4D 病例顺序。"""
    ds = pickle.load(open(f'{BASE}/exp_dataset.pkl', 'rb'))
    row_by_rulepid = {m['pid']: k for k, m in enumerate(ds['meta'])}
    rulepid_by_case = {case_id(p): p for p in ds['pids']}
    rows = [row_by_rulepid[rulepid_by_case[pid]] for pid in ds4['pids']]
    return ds['X_struct'][rows]


def run_task12_xgb_matrix(X, label3, y2, seed=42):
    """对给定特征矩阵执行 Task1/2 的 5-fold XGBoost。"""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results = {'cls': [], 'bin': []}
    fold_preds = {'cls_y': [], 'cls_p': [], 'bin_y': [], 'bin_p': []}
    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X, label3)):
        Xs = ERM.prep_struct(X, tr_idx)
        Xtr, Xte = Xs[tr_idx], Xs[te_idx]
        m1 = xgb_train_fit(Xtr, label3[tr_idx], 'cls')
        prob1 = m1.predict_proba(Xte)
        pred1 = prob1.argmax(1)
        results['cls'].append(metrics_cls(label3[te_idx], pred1))
        fold_preds['cls_y'].extend([int(x) for x in label3[te_idx]])
        fold_preds['cls_p'].extend([int(x) for x in pred1])

        m2 = xgb_train_fit(Xtr, y2[tr_idx], 'bin')
        prob2 = m2.predict_proba(Xte)[:, 1]
        results['bin'].append(metrics_bin(y2[te_idx], prob2))
        fold_preds['bin_y'].extend([int(x) for x in y2[te_idx]])
        fold_preds['bin_p'].extend([float(x) for x in prob2])
        print(f'XGB Task1/2 折{fold_i + 1} 完成', flush=True)
    out = {
        task: {k: float(np.mean([f_[k] for f_ in folds_])) for k in folds_[0]}
        for task, folds_ in results.items()
    }
    return out, fold_preds


def run_v2_4d_xgb_task12(ds4, seed=42):
    """Task1/2 用 4D 时序超图特征 + XGBoost 头，专门解决小样本下 4D 读出头偏弱的问题。"""
    from sklearn.model_selection import StratifiedKFold

    X = build_4d_xgb_features(ds4)
    meta = ds4['meta']
    label3 = np.array([m['label3'] for m in meta])
    y2 = (label3 > 0).astype(int)
    pids = ds4['pids']
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    out = {'4d_hypergraph': {'cls': [], 'bin': []}}
    fold_preds = {'4d_hypergraph': {'cls_y': [], 'cls_p': [], 'bin_y': [], 'bin_p': []}}
    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X, label3)):
        Xs = ERM.prep_struct(X, tr_idx)
        Xtr, Xte = Xs[tr_idx], Xs[te_idx]
        # Task1
        m1 = xgb_train_fit(Xtr, label3[tr_idx], 'cls')
        prob1 = m1.predict_proba(Xte)
        pred1 = prob1.argmax(1)
        out['4d_hypergraph']['cls'].append(metrics_cls(label3[te_idx], pred1))
        fold_preds['4d_hypergraph']['cls_y'].extend([int(x) for x in label3[te_idx]])
        fold_preds['4d_hypergraph']['cls_p'].extend([int(x) for x in pred1])
        # Task2
        m2 = xgb_train_fit(Xtr, y2[tr_idx], 'bin')
        prob2 = m2.predict_proba(Xte)[:, 1]
        out['4d_hypergraph']['bin'].append(metrics_bin(y2[te_idx], prob2))
        fold_preds['4d_hypergraph']['bin_y'].extend([int(x) for x in y2[te_idx]])
        fold_preds['4d_hypergraph']['bin_p'].extend([float(x) for x in prob2])
        print(f'4D-XGB Task1/2 折{fold_i + 1} 完成', flush=True)
    out['4d_hypergraph'] = {
        task: {k: float(np.mean([f_[k] for f_ in folds_])) for k in folds_[0]}
        for task, folds_ in out['4d_hypergraph'].items()
    }
    return out, fold_preds


def run_task12_baselines_seed(ds4, seed=42):
    """重跑与 4D-XGB 同 seed 的 Task1/2 简单基线。"""
    meta = ds4['meta']
    label3 = np.array([m['label3'] for m in meta])
    y2 = (label3 > 0).astype(int)
    out, fold_preds = {}, {}
    mats = {
        'lrraw': align_lrraw_struct(ds4),
        'lr_xgboost': ds4['X_struct'],
    }
    for name, X in mats.items():
        print(f'运行基线: {name} | seed={seed}', flush=True)
        out[name], fold_preds[name] = run_task12_xgb_matrix(X, label3, y2, seed=seed)
    return out, fold_preds


def temporal_split_indices(cases, pids, train_frac=0.7):
    """按入院时间做时间外切分，并保证训练/测试均覆盖 Task1/2 类别。"""
    rows = []
    for i, pid in enumerate(pids):
        a = parse_dt(cases[pid]['ident'].get('入院时间', ''))
        rows.append((i, a))
    rows = [(i, a) for i, a in rows if a is not None]
    rows.sort(key=lambda x: x[1])
    ordered_idx = [i for i, _ in rows]
    ordered_dt = [a for _, a in rows]
    label3 = np.array([{'A_stable': 0, 'B_ae': 1, 'C_infection': 2}[cases[pids[i]]['label']] for i in range(len(pids))])
    y2 = (label3 > 0).astype(int)
    base_k = int(len(ordered_idx) * train_frac)
    best = None
    for delta in range(-15, 16):
        k = min(len(ordered_idx) - 10, max(10, base_k + delta))
        tr_idx = np.array(ordered_idx[:k], dtype=int)
        te_idx = np.array(ordered_idx[k:], dtype=int)
        tr3, te3 = set(label3[tr_idx]), set(label3[te_idx])
        tr2, te2 = set(y2[tr_idx]), set(y2[te_idx])
        if tr3 == {0, 1, 2} and te3 == {0, 1, 2} and tr2 == {0, 1} and te2 == {0, 1}:
            cutoff = ordered_dt[k - 1]
            return tr_idx, te_idx, cutoff
        best = (tr_idx, te_idx, ordered_dt[k - 1])
    return best


def run_task12_xgb_single_split(X, label3, y2, tr_idx, te_idx):
    """在固定训练/测试切分上跑 Task1/2。"""
    Xs = ERM.prep_struct(X, tr_idx)
    Xtr, Xte = Xs[tr_idx], Xs[te_idx]
    m1 = xgb_train_fit(Xtr, label3[tr_idx], 'cls')
    prob1 = m1.predict_proba(Xte)
    pred1 = prob1.argmax(1)
    cls = metrics_cls(label3[te_idx], pred1)
    m2 = xgb_train_fit(Xtr, y2[tr_idx], 'bin')
    prob2 = m2.predict_proba(Xte)[:, 1]
    bin_ = metrics_bin(y2[te_idx], prob2)
    preds = {
        'cls_y': [int(x) for x in label3[te_idx]], 'cls_p': [int(x) for x in pred1],
        'bin_y': [int(x) for x in y2[te_idx]], 'bin_p': [float(x) for x in prob2],
    }
    return {'cls': cls, 'bin': bin_}, preds


def run_task12_temporal(ds4, cases, train_frac=0.7):
    """时间外验证：早期病例训练，后期病例测试。"""
    pids = ds4['pids']
    meta = ds4['meta']
    label3 = np.array([m['label3'] for m in meta])
    y2 = (label3 > 0).astype(int)
    tr_idx, te_idx, cutoff = temporal_split_indices(cases, pids, train_frac=train_frac)
    mats = {
        '4d_hypergraph': build_4d_xgb_features(ds4),
        'lr_xgboost': ds4['X_struct'],
        'lrraw': align_lrraw_struct(ds4),
    }
    out, preds = {}, {}
    for name, X in mats.items():
        print(f'时间外验证: {name}', flush=True)
        out[name], preds[name] = run_task12_xgb_single_split(X, label3, y2, tr_idx, te_idx)
    counts = {
        'train_n': int(len(tr_idx)),
        'test_n': int(len(te_idx)),
        'cutoff_date': cutoff.strftime('%Y-%m-%d'),
        'train_cls_counts': {str(k): int((label3[tr_idx] == k).sum()) for k in sorted(set(label3))},
        'test_cls_counts': {str(k): int((label3[te_idx] == k).sum()) for k in sorted(set(label3))},
    }
    return out, preds, counts


def summarize_task12_3seed_bootstrap(base_path=BASE, seeds=(42, 123, 456), n_boot=2000):
    """汇总 3-seed Task1/2 结果，并给出 bootstrap 95% CI。"""
    rng = np.random.default_rng(20260809)

    def _metric(task, y, p):
        y = np.asarray(y)
        p = np.asarray(p)
        if task == 'cls':
            return metrics_cls(y, p)['macro_f1']
        return metrics_bin(y, p)['auc_roc']

    def _bootstrap_mean(task, pred_list):
        vals = []
        for _ in range(n_boot):
            per_seed = []
            for pred in pred_list:
                y = np.asarray(pred[f'{task}_y'])
                p = np.asarray(pred[f'{task}_p'])
                idx = rng.integers(0, len(y), len(y))
                yb, pb = y[idx], p[idx]
                if task == 'bin' and len(np.unique(yb)) < 2:
                    continue
                per_seed.append(_metric(task, yb, pb))
            if per_seed:
                vals.append(float(np.mean(per_seed)))
        return [float(x) for x in np.percentile(vals, [2.5, 97.5])]

    def _bootstrap_diff(task, pred_list_a, pred_list_b):
        vals = []
        for _ in range(n_boot):
            per_seed = []
            for pa, pb in zip(pred_list_a, pred_list_b):
                y = np.asarray(pa[f'{task}_y'])
                idx = rng.integers(0, len(y), len(y))
                ya = y[idx]
                p_a = np.asarray(pa[f'{task}_p'])[idx]
                p_b = np.asarray(pb[f'{task}_p'])[idx]
                if task == 'bin' and len(np.unique(ya)) < 2:
                    continue
                per_seed.append(_metric(task, ya, p_a) - _metric(task, ya, p_b))
            if per_seed:
                vals.append(float(np.mean(per_seed)))
        return [float(x) for x in np.percentile(vals, [2.5, 97.5])]

    res = {'seed_values': defaultdict(dict), 'summary': defaultdict(dict), 'bootstrap_ci': defaultdict(dict), 'diff_ci': defaultdict(dict)}
    pred_store = defaultdict(list)
    for seed in seeds:
        j4 = json.load(open(f'{base_path}/exp_results_4dchain_v2_seed{seed}_4dxgb_task12.json', encoding='utf-8'))['task_results']['4d_hypergraph']
        jb = json.load(open(f'{base_path}/exp_results_task12_baselines_seed{seed}.json', encoding='utf-8'))['task_results']
        p4 = pickle.load(open(f'{base_path}/exp_foldpreds_v2_seed{seed}_4dxgb_task12.pkl', 'rb'))['4d_hypergraph']
        pb = pickle.load(open(f'{base_path}/exp_foldpreds_task12_baselines_seed{seed}.pkl', 'rb'))
        res['seed_values']['4d_xgb'][str(seed)] = {'cls': float(j4['cls']['macro_f1']), 'bin': float(j4['bin']['auc_roc'])}
        res['seed_values']['lr_xgboost'][str(seed)] = {'cls': float(jb['lr_xgboost']['cls']['macro_f1']), 'bin': float(jb['lr_xgboost']['bin']['auc_roc'])}
        res['seed_values']['lrraw'][str(seed)] = {'cls': float(jb['lrraw']['cls']['macro_f1']), 'bin': float(jb['lrraw']['bin']['auc_roc'])}
        pred_store['4d_xgb'].append(p4)
        pred_store['lr_xgboost'].append(pb['lr_xgboost'])
        pred_store['lrraw'].append(pb['lrraw'])

    for model in ['4d_xgb', 'lr_xgboost', 'lrraw']:
        for task in ['cls', 'bin']:
            vals = np.array([res['seed_values'][model][str(seed)][task] for seed in seeds], dtype=float)
            res['summary'][model][task] = {'mean': float(vals.mean()), 'sd': float(vals.std(ddof=0))}
            res['bootstrap_ci'][model][task] = _bootstrap_mean(task, pred_store[model])
    for baseline in ['lr_xgboost', 'lrraw']:
        for task in ['cls', 'bin']:
            res['diff_ci'][f'4d_xgb_minus_{baseline}'][task] = _bootstrap_diff(task, pred_store['4d_xgb'], pred_store[baseline])
    return res


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def run_task12_shap(ds4, seed=42, topk=20, out_dir=None):
    """用 XGBoost pred_contribs 输出 Task1/2 的全局贡献图与 JSON。"""
    import xgboost as xgb

    out_dir = _ensure_dir(out_dir or f'{BASE}/task12_shap_seed{seed}')
    X, feat_names = build_4d_xgb_features(ds4, return_feature_names=True)
    label3 = np.array([m['label3'] for m in ds4['meta']])
    y2 = (label3 > 0).astype(int)
    Xs = ERM.prep_struct(X, np.arange(len(X)))

    cls_model = xgb_train_fit(Xs, label3, 'cls')
    bin_model = xgb_train_fit(Xs, y2, 'bin')
    dmat = xgb.DMatrix(Xs)
    cls_contrib = cls_model.get_booster().predict(dmat, pred_contribs=True)[:, :, :-1]
    bin_contrib = bin_model.get_booster().predict(dmat, pred_contribs=True)[:, :-1]

    class_names = ['A_stable', 'B_ae', 'C_infection']
    shap_out = {'classification': {}, 'binary': {}}

    cls_global = np.mean(np.abs(cls_contrib), axis=(0, 1))
    top_idx = np.argsort(cls_global)[::-1][:topk]
    shap_out['classification']['global_top'] = [
        {'feature': feat_names[i], 'mean_abs_contrib': float(cls_global[i])} for i in top_idx
    ]
    for ci, cname in enumerate(class_names):
        vals = np.mean(np.abs(cls_contrib[:, ci, :]), axis=0)
        idx = np.argsort(vals)[::-1][:topk]
        shap_out['classification'][cname] = [
            {'feature': feat_names[i], 'mean_abs_contrib': float(vals[i])} for i in idx
        ]

    bin_global = np.mean(np.abs(bin_contrib), axis=0)
    top_idx = np.argsort(bin_global)[::-1][:topk]
    shap_out['binary']['global_top'] = [
        {'feature': feat_names[i], 'mean_abs_contrib': float(bin_global[i])} for i in top_idx
    ]

    def _plot_bar(items, title, out_png):
        labels = [x['feature'] for x in items][::-1]
        vals = [x['mean_abs_contrib'] for x in items][::-1]
        plt.figure(figsize=(10, max(6, len(items) * 0.32)))
        plt.barh(labels, vals, color='#4C72B0')
        plt.title(title)
        plt.xlabel('mean(|contribution|)')
        plt.tight_layout()
        plt.savefig(out_png, dpi=180)
        plt.close()

    _plot_bar(shap_out['classification']['global_top'], '4D-XGB Task1 Global Contributions',
              f'{out_dir}/task1_shap_global_top{topk}.png')
    for cname in class_names:
        _plot_bar(shap_out['classification'][cname], f'4D-XGB Task1 {cname} Contributions',
                  f'{out_dir}/task1_shap_{cname}_top{topk}.png')
    _plot_bar(shap_out['binary']['global_top'], '4D-XGB Task2 Global Contributions',
              f'{out_dir}/task2_shap_global_top{topk}.png')

    json.dump(shap_out, open(f'{out_dir}/task12_shap_summary.json', 'w'), ensure_ascii=False, indent=2)
    return shap_out, out_dir


def run_plot_typical_cases(ds4, out_dir=None):
    """自动选择每类典型病例，并绘制 4 维按天轨迹图。"""
    out_dir = _ensure_dir(out_dir or f'{BASE}/typical_case_trajectories')
    dims = ['hist', 'evidence', 'treat', 'severity']
    prefix_to_dim = {
        'sym': 'hist', 'diag': 'hist',
        'lab': 'evidence', 'img': 'evidence',
        'med': 'treat',
        'score': 'severity', 'sev': 'severity',
    }
    class_names = {0: 'A_stable', 1: 'B_ae', 2: 'C_infection'}
    pids = ds4['pids']
    meta = ds4['meta']
    max_days = ds4['max_days']

    case_stats = []
    for pid, m in zip(pids, meta):
        by_day = {d: {dm: 0 for dm in dims} for d in range(max_days)}
        for d, es in ds4['day_ent_data'].get(pid, []):
            if d >= max_days:
                continue
            for e in es:
                dm = prefix_to_dim.get(e[0])
                if dm is not None:
                    by_day[d][dm] += 1
        total = sum(sum(by_day[d][dm] for dm in dims) for d in range(max_days))
        active = sum(1 for d in range(max_days) if sum(by_day[d][dm] for dm in dims) > 0)
        case_stats.append({'pid': pid, 'label3': m['label3'], 'by_day': by_day, 'total': total, 'active': active})

    chosen = {}
    for lb in [0, 1, 2]:
        items = [x for x in case_stats if x['label3'] == lb]
        med_total = float(np.median([x['total'] for x in items]))
        med_active = float(np.median([x['active'] for x in items]))
        items.sort(key=lambda x: abs(x['total'] - med_total) + abs(x['active'] - med_active))
        chosen[lb] = items[0]

    summary = {}
    for lb, item in chosen.items():
        pid = item['pid']
        arr = np.array([[item['by_day'][d][dm] for dm in dims] for d in range(max_days)], dtype=float)
        plt.figure(figsize=(8, 4.5))
        for j, dm in enumerate(dims):
            plt.plot(range(max_days), arr[:, j], marker='o', linewidth=2, label=dm)
        plt.xticks(range(max_days))
        plt.xlabel('hospital day')
        plt.ylabel('entity count')
        plt.title(f'{class_names[lb]} | {pid}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{out_dir}/{class_names[lb]}_{pid}.png', dpi=180)
        plt.close()
        top_ents = defaultdict(Counter)
        for _, es in ds4['day_ent_data'].get(pid, []):
            for e in es:
                dm = prefix_to_dim.get(e[0])
                if dm is not None:
                    top_ents[dm][_safe_ent_name(e)] += 1
        summary[class_names[lb]] = {
            'pid': pid,
            'group': meta[pids.index(pid)]['group'],
            'label3': int(lb),
            'total_temporal_entities': int(item['total']),
            'active_days': int(item['active']),
            'top_entities_by_dim': {dm: top_ents[dm].most_common(8) for dm in dims},
        }
    json.dump(summary, open(f'{out_dir}/typical_cases_summary.json', 'w'), ensure_ascii=False, indent=2)
    return summary, out_dir



def run_v2_4d_only(ds4, seed=42, task12_only=False, fast_budget=False):
    """仅重跑 4D-HyperGraph，用于验证模型修复是否解决 Task1/2。"""
    import torch
    import torch.nn.functional as F
    from sklearn.model_selection import StratifiedKFold

    pids = ds4['pids']
    meta = ds4['meta']
    label3 = np.array([m['label3'] for m in meta])
    y2 = (label3 > 0).astype(int)
    days = np.array([float(m['days']) if m['days'] else np.nan for m in meta])
    X_struct = ds4['X_struct']
    ent2id = ds4['ent2id']
    day_ent_data = ds4['day_ent_data']
    NS = X_struct.shape[1]
    DIM_ORDER = ['hist', 'evidence', 'treat', 'severity']

    def prep_struct(X, Xtr_idx):
        tr = X[Xtr_idx]
        med = np.nanmedian(tr, axis=0)
        med[np.isnan(med)] = 0.0
        std = np.nanstd(tr, axis=0)
        std[np.isnan(std)] = 1.0
        std[std < 1e-6] = 1.0
        Xi = np.where(np.isnan(X), med, X)
        return (Xi - med) / std

    def day_dim_seq_for(pid, dims=None):
        dim_by_prefix = {
            'sym': 'hist', 'diag': 'hist',
            'lab': 'evidence', 'img': 'evidence',
            'med': 'treat',
            'score': 'severity', 'sev': 'severity',
        }
        out = []
        for d, es in day_ent_data.get(pid, []):
            dmap = defaultdict(list)
            for e in es:
                if e not in ent2id:
                    continue
                dmname = dim_by_prefix.get(e[0])
                if dmname is None:
                    continue
                if dims is not None and dmname not in dims:
                    continue
                dmap[DIM_ORDER.index(dmname)].append(ent2id[e])
            if dmap:
                out.append((d, {di: list(set(ids)) for di, ids in dmap.items()}))
        return out if out else [(0, {})]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(skf.split(np.arange(len(pids)), label3))
    cls_w = torch.tensor([253.0 / (3 * 54), 253.0 / (3 * 89), 253.0 / (3 * 110)])

    results = {'4d_hypergraph': {'cls': [], 'bin': [], 'reg': []}}
    fold_preds = {'4d_hypergraph': {'cls_y': [], 'cls_p': [], 'bin_y': [], 'bin_p': [], 'reg_y': [], 'reg_p': []}}

    old_epochs, old_patience = ERM.EPOCHS, ERM.PATIENCE
    if fast_budget:
        ERM.EPOCHS, ERM.PATIENCE = 50, 8
    try:
        for fold_i, (tr_idx, te_idx) in enumerate(folds):
            Xs = prep_struct(X_struct, tr_idx)
            S_tr = torch.tensor(Xs[tr_idx], dtype=torch.float32)
            S_te = torch.tensor(Xs[te_idx], dtype=torch.float32)
            dtr = [day_dim_seq_for(pids[i]) for i in tr_idx]
            dte = [day_dim_seq_for(pids[i]) for i in te_idx]

            ytr3 = torch.tensor(label3[tr_idx], dtype=torch.long)
            yte3 = label3[te_idx]
            ytr2 = torch.tensor(y2[tr_idx], dtype=torch.long)
            yte2 = y2[te_idx]
            reg_mu = float(np.nanmean(days[tr_idx]))
            reg_sd = float(np.nanstd(days[tr_idx]))
            reg_sd = reg_sd if reg_sd > 1e-6 else 1.0
            ytr3r = torch.tensor((np.nan_to_num(days[tr_idx]) - reg_mu) / reg_sd, dtype=torch.float32)
            yte3r_raw = np.nan_to_num(days[te_idx])
            yte3r = (yte3r_raw - reg_mu) / reg_sd

            m = HyperGraph4D(len(ent2id), NS, nout=3)
            m = train_model(m, (dtr, S_tr), ytr3, (dte, S_te), torch.tensor(yte3, dtype=torch.long), 'cls', fold_i, cls_w)
            with torch.no_grad():
                prob = F.softmax(m(dte, S_te), 1).numpy()
            pred = prob.argmax(1)
            results['4d_hypergraph']['cls'].append(metrics_cls(yte3, pred))
            fold_preds['4d_hypergraph']['cls_y'].extend([int(x) for x in yte3])
            fold_preds['4d_hypergraph']['cls_p'].extend([int(x) for x in pred])

            m2 = HyperGraph4D(len(ent2id), NS, nout=1)
            m2 = train_model(m2, (dtr, S_tr), ytr2, (dte, S_te), torch.tensor(yte2, dtype=torch.long), 'bin', fold_i + 50)
            with torch.no_grad():
                prob2 = torch.sigmoid(m2(dte, S_te).squeeze(1)).numpy()
            results['4d_hypergraph']['bin'].append(metrics_bin(yte2, prob2))
            fold_preds['4d_hypergraph']['bin_y'].extend([int(x) for x in yte2])
            fold_preds['4d_hypergraph']['bin_p'].extend([float(x) for x in prob2])

            if not task12_only:
                m3 = HyperGraph4D(len(ent2id), NS, nout=1)
                m3 = train_model(m3, (dtr, S_tr), ytr3r, (dte, S_te), torch.tensor(yte3r, dtype=torch.float32), 'reg', fold_i + 100)
                with torch.no_grad():
                    pred3 = m3(dte, S_te).squeeze(1).numpy() * reg_sd + reg_mu
                results['4d_hypergraph']['reg'].append(metrics_reg(yte3r_raw, pred3))
                fold_preds['4d_hypergraph']['reg_y'].extend([float(x) for x in yte3r_raw])
                fold_preds['4d_hypergraph']['reg_p'].extend([float(x) for x in pred3])
            print(f'4D-only 折{fold_i + 1} 完成', flush=True)
    finally:
        ERM.EPOCHS, ERM.PATIENCE = old_epochs, old_patience

    out = {'4d_hypergraph': {}}
    for task, folds_ in results['4d_hypergraph'].items():
        if task12_only and task == 'reg':
            continue
        out['4d_hypergraph'][task] = {k: float(np.mean([f_[k] for f_ in folds_])) for k in folds_[0]}
    return out, fold_preds


if __name__ == '__main__':
    cases = R.load_cases()
    raw = load_raw_cases()
    print('原始病历载入:', len(raw), '例 | 四维链病例:', len(cases))
    ds4 = build_v2(cases, raw)
    pickle.dump(ds4, open(f'{BASE}/exp_4dchain_v2_dataset.pkl', 'wb'))
    inspect(ds4)
    if '--build-only' in sys.argv:
        sys.exit(0)
    seed = 42
    only_4d = '--only-4d' in sys.argv
    task12_4dxgb = '--task12-4dxgb' in sys.argv
    task12_baselines = '--task12-baselines' in sys.argv
    task12_temporal = '--task12-temporal' in sys.argv
    task12_summary_ci = '--task12-summary-ci' in sys.argv
    task12_shap = '--task12-shap' in sys.argv
    plot_typical_cases = '--plot-typical-cases' in sys.argv
    task12_external = '--task12-external' in sys.argv
    task12_only = '--task12-only' in sys.argv
    fast_budget = '--fast-4d' in sys.argv
    external_folder = os.environ.get('EXTERNAL_CHAIN_DIR', '/path/to/external_4d_chains')
    for a in sys.argv[1:]:
        if a.startswith('--seed='):
            seed = int(a.split('=')[1])
        elif a.startswith('--external-folder='):
            external_folder = a.split('=', 1)[1]
        elif a == '--no-ablation':
            do_ablation = False
    if task12_shap:
        out, out_dir = run_task12_shap(ds4, seed=seed)
        res_path = f'{out_dir}/task12_shap_summary.json'
    elif plot_typical_cases:
        out, out_dir = run_plot_typical_cases(ds4)
        res_path = f'{out_dir}/typical_cases_summary.json'
    elif task12_summary_ci:
        out = summarize_task12_3seed_bootstrap()
        res_path = f'{BASE}/exp_results_task12_3seed_summary_ci.json'
        json.dump(out, open(res_path, 'w'), ensure_ascii=False, indent=2)
    elif task12_temporal:
        out, fold_preds, split_info = run_task12_temporal(ds4, cases)
        res_path = f'{BASE}/exp_results_task12_temporal.json'
        json.dump({'task_results': out, 'split_info': split_info,
                   'note': 'temporal validation：按入院时间切分，早期训练、后期测试'},
                  open(res_path, 'w'), ensure_ascii=False, indent=2)
        pickle.dump(fold_preds, open(f'{BASE}/exp_foldpreds_task12_temporal.pkl', 'wb'))
    elif task12_external:
        out, fold_preds = run_task12_external_4dxgb(ds4, external_folder)
        stem = Path(external_folder).name
        res_path = f'{BASE}/exp_results_task12_external_{stem}.json'
        json.dump({'task_results': out,
                   'external_folder': external_folder,
                   'note': 'external validation：训练集全量拟合，外部四维链 COPD 三类病例验证；当前仅支持 4D-XGB 与 LR/XGBoost'},
                  open(res_path, 'w'), ensure_ascii=False, indent=2)
        pickle.dump(fold_preds, open(f'{BASE}/exp_foldpreds_task12_external_{stem}.pkl', 'wb'))
    elif task12_baselines:
        out, fold_preds = run_task12_baselines_seed(ds4, seed=seed)
        res_path = f'{BASE}/exp_results_task12_baselines_seed{seed}.json'
        json.dump({'task_results': out, 'seed': seed,
                   'note': 'task12-baselines：与4D-XGB同seed的LR-raw和LR/XGBoost基线'},
                  open(res_path, 'w'), ensure_ascii=False, indent=2)
        pickle.dump(fold_preds, open(f'{BASE}/exp_foldpreds_task12_baselines_seed{seed}.pkl', 'wb'))
    elif task12_4dxgb:
        out, fold_preds = run_v2_4d_xgb_task12(ds4, seed=seed)
        res_path = f'{BASE}/exp_results_4dchain_v2_seed{seed}_4dxgb_task12.json'
        json.dump({'task_results': out, 'seed': seed,
                   'note': 'task12-4dxgb：4D时序超图特征 + XGBoost 头，仅用于 Task1/2'},
                  open(res_path, 'w'), ensure_ascii=False, indent=2)
        pickle.dump(fold_preds, open(f'{BASE}/exp_foldpreds_v2_seed{seed}_4dxgb_task12.pkl', 'wb'))
    elif only_4d:
        out, fold_preds = run_v2_4d_only(ds4, seed=seed, task12_only=task12_only, fast_budget=fast_budget)
        res_path = f'{BASE}/exp_results_4dchain_v2_seed{seed}_4donly.json'
        json.dump({'task_results': out, 'seed': seed,
                   'note': f'only-4d：时序4D修复后仅重跑4d_hypergraph，用于与简单基线比较；task12_only={task12_only}; fast_budget={fast_budget}'},
                  open(res_path, 'w'), ensure_ascii=False, indent=2)
        pickle.dump(fold_preds, open(f'{BASE}/exp_foldpreds_v2_seed{seed}_4donly.pkl', 'wb'))
    else:
        out, abl_res, per_class_f1, hardest, mcnemar_vs, stats, fold_preds = \
            run_v2(ds4, seed=seed, do_ablation=('--no-ablation' not in sys.argv))
        res_path = f'{BASE}/exp_results_4dchain_v2_seed{seed}.json'
        json.dump({'task_results': out, 'ablation': abl_res, 'per_class_f1': per_class_f1,
                   'hardest': hardest, 'mcnemar_vs': mcnemar_vs, 'stats': stats,
                   'seed': seed,
                   'note': 'v3：Dim4更新(新增呼衰/巴氏/加重次数/过去住院实体) + Task3改为CSCS复合评分(GOLD+加重+合并症+呼衰) + 时序窗口固定5天(修复长度泄漏)；模型/折/训练/统计与v1完全一致'},
                  open(res_path, 'w'), ensure_ascii=False, indent=2)
        pickle.dump(fold_preds, open(f'{BASE}/exp_foldpreds_v2_seed{seed}.pkl', 'wb'))
    print('结果已保存:', res_path)
    if isinstance(out, dict):
        for method, r in out.items():
            if isinstance(r, dict) and 'cls' in r:
                print(method, 'cls:', {k: round(v, 4) for k, v in r['cls'].items()})
        for method, r in out.items():
            if isinstance(r, dict) and 'bin' in r:
                print(method, 'bin:', {k: round(v, 4) for k, v in r['bin'].items()})
        for method, r in out.items():
            if isinstance(r, dict) and 'reg' in r:
                print(method, 'reg:', {k: round(v, 4) for k, v in r['reg'].items()})
