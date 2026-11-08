
# -*- coding: utf-8 -*-
"""
import os
exp_4dchain.py — 四维链版监督实验
用 HyperGraphRAG 四维链（Qwen3-14B LLM 抽取，4 链 13 分组）替换「实验执行方案.md」Part A
注：本脚本为早期四维链版本；论文报告的 Task 1/2 结果由 exp_4dchain_v2.py 产生，
    Task 3（CSCS 回归，component-blind 口径）由 exp_task_fix_probe.py 产生。
的提示词抽取表示，作为监督学习输入，复刻 Part C 全部 5 方法 + 三任务 + 5-fold CV + 消融 + 统计。

与 exp_run.py（规则稀疏特征）唯一差异 = 数据表示层：
  - 结构化特征 / 实体 / 按天序列 全部来自四维链 LLM 抽取结果（覆盖度 100%，远高于规则阈值抽取）
  - 模型架构、训练循环、5-fold 种子、评价函数完全复用 exp_run.py

诚实性约束（避免定义泄漏）：
  - 排除 evidence_chain「诊断信息」分组的主诊断/出院诊断文本（A/B/C 分组即按主诊断，否则 Task1 虚高）
  - 排除 severity_chain「预后信息」的住院天数（Task3 回归标签）
  - 保留：其他诊断/并发症（合并症）、检验、症状、生命体征、用药、GOLD/mMRC、诊疗过程

用法:
  python3 exp_4dchain.py --build-only   # 仅构建并自检特征
  python3 exp_4dchain.py                # 完整 5 方法三任务 + 消融
"""
import os, sys, json, re, copy, random, pickle, traceback
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

BASE = os.environ.get('EXPERIMENT_BASE', os.getcwd())
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
sys.path.insert(0, os.environ.get('HYPERGRAPHRAG_PATH',
                                  os.path.join(_REPO, 'third_party', 'HyperGraphRAG')))

import run_hgr_eval as R
import exp_run as ER
from exp_run import (KGGNN, TKG, LSTMModel, HyperGraph4D,
                     train_model, xgb_train_fit,
                     metrics_cls, metrics_bin, metrics_reg)
from scipy.stats import wilcoxon

# ---------------- 常量与解析工具 ----------------
MED_CATS = ['抗生素', '激素', '支气管扩张剂', '祛痰药', '抗凝药', '利尿剂', '血管活性药', '氧疗/通气', '营养支持']
MED_CLASS_KW = {
    '抗生素': ['头孢', '阿莫西林', '哌拉西林', '他唑巴坦', '左氧氟沙星', '莫西沙星', '阿奇霉素', '克拉霉素',
               '万古霉素', '美罗培南', '亚胺培南', '替加环素', '多黏菌素', '利奈唑胺', '氟康唑', '卡泊芬净',
               '奥司他韦', '沙星', '西林', '霉素', '环素', '磺胺', '两性霉素', '抗感染'],
    '激素': ['泼尼松', '甲泼尼龙', '地塞米松', '氢化可的松', '布地奈德', '氟替卡松', '倍氯米松', '曲安奈德', '糖皮质激素'],
    '支气管扩张剂': ['沙丁胺醇', '异丙托溴铵', '噻托溴铵', '乌美溴铵', '维兰特罗', '福莫特罗', '沙美特罗',
                    '氨茶碱', '茶碱', '多索茶碱', '异丙阿托品', '格隆溴铵', '茚达特罗', '奥达特罗', '特布他林'],
    '祛痰药': ['氨溴索', '乙酰半胱氨酸', '羧甲司坦', '溴己新', '祛痰', '化痰', '桃金娘油', '黏液'],
    '抗凝药': ['肝素', '华法林', '利伐沙班', '达比加群', '依诺肝素', '那屈肝素', '比伐卢定'],
    '利尿剂': ['呋塞米', '托拉塞米', '氢氯噻嗪', '螺内酯', '布美他尼'],
    '血管活性药': ['多巴胺', '去甲肾上腺素', '肾上腺素', '间羟胺', '硝酸甘油', '异丙肾上腺素', '多巴酚丁胺'],
    '氧疗/通气': ['氧疗', '无创', '呼吸机', '高流量', '吸氧', 'NIV', 'BiPAP', 'CPAP', '面罩', '氧合'],
    '营养支持': ['肠内营养', '氨基酸', '脂肪乳', '白蛋白', '营养', '葡萄糖'],
}
LAB_PAT = {
    'CRP': r'C反应蛋白|CRP|超敏CRP|hsCRP',
    'PCT': r'降钙素原|PCT',
    'WBC': r'白细胞|WBC',
    'NEUT': r'中性粒细胞|NEUT',
    'HGB': r'血红蛋白|HGB|血红蛋白浓度',
    'ALB': r'白蛋白|ALB',
    'PLT': r'血小板|PLT',
}
SYM_KW = ['咳嗽', '咳痰', '喘憋', '气短', '呼吸困难', '发热', '胸闷', '乏力', '咯血', '胸痛', '喘息', '紫绀']


def parse_dt(s):
    s = str(s).strip()
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d'):
        try:
            return datetime.strptime(s[:19], fmt)
        except Exception:
            continue
    return None


def first_num(v):
    """从任意值里提取第一个数字（浮点），无则 None。"""
    if v is None:
        return None
    m = re.search(r'-?\d+\.?\d*', str(v))
    return float(m.group(0)) if m else None


def parse_gold(v):
    s = str(v).strip()
    m = re.search(r'[IViv]+', s)
    if m:
        return {'I': 1, 'II': 2, 'III': 3, 'IV': 4}.get(m.group(0).upper())
    m = re.search(r'\d', s)
    if m:
        return min(4, max(1, int(m.group(0))))
    return None


def infer_gold_from_basis(basis):
    """从分期依据文本（FEV1%pred 片段）推断 GOLD 1-4 期；无则 None。
    依据 GOLD 2024：FEV1%pred >=80 → I, 50-79 → II, 30-49 → III, <30 → IV。
    """
    if not basis:
        return None
    m = re.search(r'FEV1[^。；;\n]{0,30}?(\d{1,3}(?:\.\d+)?)\s*[%％]', basis)
    if not m:
        m = re.search(r'(\d{1,3}(?:\.\d+)?)\s*[%％]\s*(?:预计值|FEV1|pred)', basis)
    if not m:
        m = re.search(r'(?:预计值|FEV1)[^。；;\n]{0,15}?(\d{1,3}(?:\.\d+)?)', basis)
    if m:
        pct = float(m.group(1))
        if pct >= 80:
            return 1
        if pct >= 50:
            return 2
        if pct >= 30:
            return 3
        return 4
    return None


def severity_grade_to_num(v):
    """轻/中/重/极重度 → 1/2/3/4；无则 None。"""
    s = str(v or '').strip()
    for g, kws in ((1, '轻度'), (2, '中度'), (3, '重度'), (4, '极重度')):
        if kws in s:
            return g
    return None


def group_by(case, bucket, gname):
    for g in case['groups'].get(bucket, []):
        if g['g'] == gname:
            return g['d'] if isinstance(g['d'], dict) else {}
    return {}


# ---------------- 特征构建 ----------------
def build_4dchain_dataset(cases):
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
    max_days = int(np.ceil(np.percentile(days_all, 95)))
    print('四维链版 序列最大长度(95分位住院天数):', max_days)

    for pid in pids:
        case = cases[pid]
        ident = case['ident']
        label3 = {'A_stable': 0, 'B_ae': 1, 'C_infection': 2}[case['label']]
        days = max(0, (parse_dt(ident.get('出院时间', '')) - parse_dt(ident.get('入院时间', ''))).days) \
            if (parse_dt(ident.get('出院时间', '')) and parse_dt(ident.get('入院时间', ''))) else None

        # ---- 人口学 ----
        demo = group_by(case, 'history_chain', '患者人口学基础信息')
        age = first_num(demo.get('年龄'))
        sex = 1 if str(demo.get('性别', '')) == '男' else 0

        # ---- 病史文本（主诉/现病史/个人史）----
        hist = group_by(case, 'history_chain', '全维度病史信息')
        chief = str(hist.get('主诉', '') or '')
        present = str(hist.get('现病史', '') or '')
        personal = str(hist.get('个人史', '') or '')
        smoking = 1 if ('烟' in personal or '烟' in present) else 0

        # ---- 合并症（其他诊断 + 并发症，去重）----
        diag = group_by(case, 'evidence_chain', '诊断信息')
        other_dx = list(diag.get('其他诊断', []) or [])
        comp = group_by(case, 'severity_chain', '并发症信息')
        comp_list = list(comp.get('并发症列表', []) or [])
        comorb = sorted(set(str(x) for x in other_dx + comp_list if str(x).strip() and str(x) != '无'))

        # ---- 检验（从检验项目列表匹配指标，取最大值 + 异常标记）----
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

        # ---- 检验按报告日期展开（H4：提升时序密度；无日期仍归 day0）----
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

        # ---- 生命体征（单点）----
        symp = group_by(case, 'evidence_chain', '症状体征信息')
        vt = symp.get('生命体征', {}) if isinstance(symp.get('生命体征'), dict) else {}
        vital_raw_vals = {}
        for vk in ['体温', '脉搏', '呼吸', '血压']:
            sub = vt.get(vk) if isinstance(vt, dict) else None
            if isinstance(sub, dict):
                vital_raw_vals[vk] = str(sub.get('数值', ''))
            elif isinstance(sub, list):
                raw = []
                for x in sub:
                    if isinstance(x, dict) and x.get('数值') is not None:
                        raw.append(str(x['数值']))
                vital_raw_vals[vk] = raw[0] if raw else ''
            else:
                vital_raw_vals[vk] = ''
        vital = {k: first_num(v) for k, v in vital_raw_vals.items()}
        bp_s = None
        if vital.get('血压') is not None:
            m = re.search(r'(\d+)\s*/\s*(\d+)', vital_raw_vals['血压'])
            if m:
                bp_s = float(m.group(1))
            else:
                bp_s = vital['血压']
        bp_d = None
        if vital.get('血压') is not None:
            m = re.search(r'(\d+)\s*/\s*(\d+)', vital_raw_vals['血压'])
            if m:
                bp_d = float(m.group(2))

        # ---- 用药（医嘱列表 + 疗程时间）----
        meds_info = group_by(case, 'care_chain', '药物治疗信息')
        med_orders = list(meds_info.get('用药医嘱列表', []) or [])
        med_courses = list(meds_info.get('用药疗程', []) or [])
        course_by_name = {}
        for c_ in med_courses:
            if isinstance(c_, dict) and c_.get('医嘱名称'):
                course_by_name[str(c_['医嘱名称'])] = c_
        med_cat_count = Counter()
        med_events = []  # (datetime, cat)
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
                # 用药物分类字段兜底
                cls_ = str(o.get('药物分类', '') or '')
                if any(k in name for k in MED_CLASS_KW['抗生素']) or '抗感染' in cls_:
                    cat = '抗生素'
                else:
                    cat = '营养支持' if any(k in name for k in MED_CLASS_KW['营养支持']) else None
            if cat:
                med_cat_count[cat] += 1
                t = parse_dt(course_by_name.get(name, {}).get('开始时间') if name in course_by_name else None)
                if t is None:
                    t = parse_dt(o.get('开始时间'))
                med_events.append((t, cat))

        # ---- GOLD / mMRC ----
        sev = group_by(case, 'severity_chain', '疾病分期与严重程度')
        gold = parse_gold(sev.get('疾病分期'))
        if gold is None:
            # 疾病分期字段常为空，从分期依据(FEV1%pred 等)推断 GOLD 1-4 期（提示词 Step1.4 允许 FEV1 推断）
            gold = infer_gold_from_basis(' '.join(str(x) for x in (sev.get('分期依据') or []) if str(x).strip()))
        sev_grade = str(sev.get('严重程度分级', '') or '')
        func = group_by(case, 'severity_chain', '功能评分信息')
        mmrc = None
        cat_score = None
        for lst_key in ['入院功能评分', '出院功能评分']:
            for it in (func.get(lst_key, []) or []):
                if isinstance(it, dict) and it.get('评分项目'):
                    pj = str(it['评分项目'])
                    val = first_num(it.get('评分值'))
                    if 'mMRC' in pj and mmrc is None and val is not None:
                        mmrc = int(val)
                    if 'CAT' in pj and cat_score is None and val is not None:
                        cat_score = int(val)

        # ---- 症状实体 ----
        sym_text = ' '.join([chief, present, str(symp.get('主要症状', '') or ''), str(symp.get('现病史摘要', '') or '')])

        # ---- 影像 ----
        img_items = list(labs_info.get('影像检查列表', []) or [])

        # ---- 实体（四维）----
        ents = defaultdict(set)
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
        for it in img_items:
            if isinstance(it, dict):
                ents['evidence'].add(('img', str(it.get('检查项目', '') or '影像')[:20]))
        for cat, n in med_cat_count.items():
            if n > 0:
                ents['treat'].add(('med', cat))
        if gold:
            ents['severity'].add(('score', f'gold_{gold}'))
        if sev_grade:
            ents['severity'].add(('sev', sev_grade))
        if mmrc is not None:
            ents['severity'].add(('score', f'mmrc_{mmrc}'))
        if cat_score is not None:
            ents['severity'].add(('score', f'cat_{cat_score}'))
        if len(comp_list) >= 3:
            ents['severity'].add(('score', 'comorb_heavy'))
        entity_data[pid] = ents
        for e in set(x for es in ents.values() for x in es):
            ent_counter[e] += 1

        # ---- 按天序列（24 维，与 exp_run 同构：8 生命体征 + 7 检验 + 9 用药）----
        seq = np.zeros((max_days, 24), dtype=float)
        # 生命体征（单点 → day0）
        for vk, idx in [('体温', 0), ('脉搏', 1), ('呼吸', 2)]:
            if vital.get(vk) is not None:
                seq[0, idx] = vital[vk]
        if bp_s:
            seq[0, 3] = bp_s
        if bp_d:
            seq[0, 4] = bp_d
        # 检验（按报告日期展开；无日期值并入 day0 取最大）
        for di, kv in lab_daily.items():
            for i, key in enumerate(LAB_PAT):
                if key in kv:
                    seq[di, 8 + i] = kv[key]
        for i, key in enumerate(LAB_PAT):
            if lab_max.get(key) is not None:
                seq[0, 8 + i] = max(seq[0, 8 + i], lab_max[key])
        # 用药（按开始时间分天）
        for t, cat in med_events:
            di = 0
            if t is not None and admit is not None:
                di = max(0, (t - admit).days)
            di = min(di, max_days - 1)
            seq[di, 15 + MED_CATS.index(cat)] += 1
        seq_data[pid] = seq

        # ---- 按天实体序列（TKG用）----
        day_e = defaultdict(set)
        for e in ents['hist']:
            day_e[0].add(e)
        for e in ents['severity']:
            day_e[0].add(e)
        # evidence 检验实体按报告日期展开
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

        # ---- 结构化特征（24 维，与 exp_run 对齐）----
        row = [age if age else np.nan, sex, smoking, len(comorb)]
        for key in LAB_PAT:
            row.append(lab_max.get(key) if lab_max.get(key) is not None else np.nan)
        row.append(vital.get('体温') if vital.get('体温') is not None else np.nan)
        row.append(vital.get('脉搏') if vital.get('脉搏') is not None else np.nan)
        row.append(vital.get('呼吸') if vital.get('呼吸') is not None else np.nan)
        row.append(bp_s if bp_s else np.nan)
        for cat in MED_CATS:
            row.append(med_cat_count.get(cat, 0))
        X_struct.append(row)
        meta.append({'pid': pid, 'group': case['group'], 'label3': label3, 'days': days})

    # ---- 实体词典 ----
    ent2id = {}
    for e, c in ent_counter.most_common():
        if c >= 2:
            ent2id[e] = len(ent2id)
    print('四维链版 实体总数:', len(ent_counter), '| 过滤后(>=2):', len(ent2id))

    X_struct = np.array(X_struct, dtype=float)
    return {
        'pids': pids, 'X_struct': X_struct, 'ent2id': ent2id,
        'entity_data': entity_data, 'seq_data': seq_data, 'max_days': max_days,
        'day_ent_data': day_ent_data, 'meta': meta,
    }


def inspect(ds4):
    print('X_struct shape:', ds4['X_struct'].shape)
    labels = np.array([m['label3'] for m in ds4['meta']])
    print('标签分布:', dict(Counter(labels)))
    days = np.array([m['days'] if m['days'] else np.nan for m in ds4['meta']])
    print('住院天数: mean=%.1f std=%.1f' % (np.nanmean(days), np.nanstd(days)))
    for g in [0, 1, 2]:
        dd = days[labels == g]
        print(f'  组{g} n={len(dd)} days: mean=%.1f std=%.1f' % (np.nanmean(dd), np.nanstd(dd)))
    seq_ok = sum(1 for pid in ds4['pids'] if ds4['seq_data'][pid] is not None)
    print('时序序列可用:', seq_ok, '/', len(ds4['pids']))
    nz = sum(1 for pid in ds4['pids'] if np.any(ds4['seq_data'][pid]))
    tot = len(ds4['pids']) * ds4['max_days'] * 24
    nz_cells = sum(int((ds4['seq_data'][pid] != 0).sum()) for pid in ds4['pids'])
    print('序列非零单元率: %.2f%% (%d/%d)' % (100 * nz_cells / tot, nz_cells, tot))
    for dm in ['hist', 'evidence', 'treat', 'severity']:
        c = sum(1 for pid in ds4['pids'] for e in ds4['entity_data'][pid].get(dm, set()) if e in ds4['ent2id'])
        print(f'  维度{dm} 有效实体关联数:', c)
    per_ent = sum(sum(len(es) for es in ds4['entity_data'][pid].values()) for pid in ds4['pids']) / len(ds4['pids'])
    print('每例实体数 均值: %.1f' % per_ent)


# ---------------- 主流程（复用 exp_run 模型与训练） ----------------
def run4(ds4, seed=42, do_ablation=True):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (f1_score, roc_auc_score)
    from scipy.stats import ttest_rel, wilcoxon, binomtest

    pids = ds4['pids']
    meta = ds4['meta']
    label3 = np.array([m['label3'] for m in meta])
    y2 = (label3 > 0).astype(int)
    days = np.array([float(m['days']) if m['days'] else np.nan for m in meta])
    X_struct = ds4['X_struct']
    seq_data = ds4['seq_data']
    max_days = ds4['max_days']
    ent2id = ds4['ent2id']
    entity_data = ds4['entity_data']
    day_ent_data = ds4['day_ent_data']
    NS = X_struct.shape[1]
    V = len(ent2id)
    DIM_ORDER = ['hist', 'evidence', 'treat', 'severity']

    def prep_struct(X, Xtr_idx):
        tr = X[Xtr_idx]
        med = np.nanmedian(tr, axis=0); med[np.isnan(med)] = 0.0
        std = np.nanstd(tr, axis=0); std[np.isnan(std)] = 1.0; std[std < 1e-6] = 1.0
        Xi = np.where(np.isnan(X), med, X)
        return (Xi - med) / std

    def prep_seq(pid, Xtr_idx):
        tr = np.stack([seq_data[pids[i]] for i in Xtr_idx])
        mu = tr.mean(axis=(0, 1)); sd = tr.std(axis=(0, 1)); sd[sd < 1e-6] = 1.0
        return (seq_data[pid] - mu) / sd

    def pid_ent_ids(pid, dims=None):
        ed = entity_data[pid]
        ids = []
        for dm, es in ed.items():
            if dims is not None and dm not in dims:
                continue
            ids += [ent2id[e] for e in es if e in ent2id]
        return list(set(ids))

    def dim_map(pid, dims=None):
        ed = entity_data[pid]
        dm = {}
        for di, dmname in enumerate(DIM_ORDER):
            if dims is not None and dmname not in dims:
                continue
            dm[di] = list(set(ent2id[e] for e in ed.get(dmname, set()) if e in ent2id))
        return dm

    def day_seq_for(pid):
        return [(d, [ent2id[e] for e in es if e in ent2id]) for d, es in day_ent_data.get(pid, [])]

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
                di = DIM_ORDER.index(dmname)
                dmap[di].append(ent2id[e])
            if dmap:
                out.append((d, {di: list(set(ids)) for di, ids in dmap.items()}))
        if out:
            return out
        dm = dim_map(pid, dims)
        return [(0, dm)] if dm else [(0, {})]

    def build_graph_adj(tr_pids, V):
        A = np.zeros((V, V), dtype=float)
        for pid in tr_pids:
            ids = pid_ent_ids(pid)
            for i in ids:
                for j in ids:
                    A[i, j] += 1.0
        np.fill_diagonal(A, 1.0)
        d = A.sum(1) ** (-0.5); d[np.isinf(d)] = 0
        D = np.diag(d)
        A = D @ A @ D
        return torch.tensor(A, dtype=torch.float32)

    import torch
    import torch.nn.functional as F

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(skf.split(np.arange(len(pids)), label3))
    cls_w = torch.tensor([253.0 / (3 * 54), 253.0 / (3 * 89), 253.0 / (3 * 110)])

    results = {}
    # 逐折预测（供 Step5 统计检验：McNemar / Wilcoxon / 95%CI）
    fold_preds = {m: {'cls_y': [], 'cls_p': [], 'bin_y': [], 'bin_p': [], 'reg_y': [], 'reg_p': []}
                  for m in ['lr_xgboost', 'kg_gnn', 'tkg', 'lstm_transformer', '4d_hypergraph']}
    mcnemar_pairs = defaultdict(lambda: {'a': [], 'b': []})
    cohen = defaultdict(lambda: defaultdict(list))

    for fold_i, (tr_idx, te_idx) in enumerate(folds):
        tr_pids = [pids[i] for i in tr_idx]
        Xs = prep_struct(X_struct, tr_idx)
        Xs_tr = Xs[tr_idx]; Xs_te = Xs[te_idx]
        S_tr = torch.tensor(Xs_tr, dtype=torch.float32)
        S_te = torch.tensor(Xs_te, dtype=torch.float32)
        seq_tr = torch.tensor(np.stack([prep_seq(pids[i], tr_idx) for i in tr_idx]), dtype=torch.float32)
        seq_te = torch.tensor(np.stack([prep_seq(pids[i], tr_idx) for i in te_idx]), dtype=torch.float32)
        adj = build_graph_adj(tr_pids, V)
        ent_tr = [pid_ent_ids(pids[i]) for i in tr_idx]
        ent_te = [pid_ent_ids(pids[i]) for i in te_idx]
        dmap_tr = [dim_map(pids[i]) for i in tr_idx]
        dmap_te = [dim_map(pids[i]) for i in te_idx]
        dseq_tr = [day_seq_for(pids[i]) for i in tr_idx]
        dseq_te = [day_seq_for(pids[i]) for i in te_idx]
        ddim_tr = [day_dim_seq_for(pids[i]) for i in tr_idx]
        ddim_te = [day_dim_seq_for(pids[i]) for i in te_idx]

        ytr3 = torch.tensor(label3[tr_idx], dtype=torch.long)
        yte3 = label3[te_idx]
        ytr2 = torch.tensor(y2[tr_idx], dtype=torch.long)
        yte2 = y2[te_idx]
        # Task3 回归目标 z-score 标准化（H1：深度模型直接拟合原始天数会欠拟合导致 R² 全负）
        reg_mu = float(np.nanmean(days[tr_idx]))
        reg_sd = float(np.nanstd(days[tr_idx]))
        reg_sd = reg_sd if reg_sd > 1e-6 else 1.0
        ytr3r = torch.tensor((np.nan_to_num(days[tr_idx]) - reg_mu) / reg_sd, dtype=torch.float32)
        yte3r_raw = np.nan_to_num(days[te_idx])
        yte3r = (yte3r_raw - reg_mu) / reg_sd

        for method in ['lr_xgboost', 'kg_gnn', 'tkg', 'lstm_transformer', '4d_hypergraph']:
            results.setdefault(method, {}).setdefault('cls', [])
            results[method].setdefault('bin', [])
            results[method].setdefault('reg', [])
            if method == 'lr_xgboost':
                m = xgb_train_fit(Xs_tr, label3[tr_idx], 'cls')
                prob = m.predict_proba(Xs_te); pred = prob.argmax(1)
            elif method == 'kg_gnn':
                m = KGGNN(V, adj, NS, nout=3)
                m = train_model(m, (ent_tr, S_tr), ytr3, (ent_te, S_te), torch.tensor(yte3, dtype=torch.long), 'cls', fold_i, cls_w)
                with torch.no_grad(): prob = F.softmax(m(ent_te, S_te), 1).numpy()
                pred = prob.argmax(1)
            elif method == 'tkg':
                m = TKG(V, NS, nout=3)
                m = train_model(m, (dseq_tr, S_tr), ytr3, (dseq_te, S_te), torch.tensor(yte3, dtype=torch.long), 'cls', fold_i, cls_w)
                with torch.no_grad(): prob = F.softmax(m(dseq_te, S_te), 1).numpy()
                pred = prob.argmax(1)
            elif method == 'lstm_transformer':
                m = LSTMModel(seq_tr.size(2), NS, nout=3)
                m = train_model(m, (seq_tr, S_tr), ytr3, (seq_te, S_te), torch.tensor(yte3, dtype=torch.long), 'cls', fold_i, cls_w)
                with torch.no_grad(): prob = F.softmax(m(seq_te, S_te), 1).numpy()
                pred = prob.argmax(1)
            else:
                m = HyperGraph4D(V, NS, nout=3)
                m = train_model(m, (ddim_tr, S_tr), ytr3, (ddim_te, S_te), torch.tensor(yte3, dtype=torch.long), 'cls', fold_i, cls_w)
                with torch.no_grad(): prob = F.softmax(m(ddim_te, S_te), 1).numpy()
                pred = prob.argmax(1)
            results[method]['cls'].append(metrics_cls(yte3, pred))
            fold_preds[method]['cls_y'].extend([int(x) for x in yte3])
            fold_preds[method]['cls_p'].extend([int(x) for x in pred])
            if method == '4d_hypergraph':
                for bl in ['lr_xgboost', 'kg_gnn', 'tkg', 'lstm_transformer']:
                    mcnemar_pairs[bl]['a'].extend(list(pred))
            else:
                mcnemar_pairs[method]['b'].extend(list(pred))

            if method == 'lr_xgboost':
                m2 = xgb_train_fit(Xs_tr, y2[tr_idx], 'bin')
                prob2 = m2.predict_proba(Xs_te)[:, 1]
            elif method == 'kg_gnn':
                m2 = KGGNN(V, adj, NS, nout=1)
                m2 = train_model(m2, (ent_tr, S_tr), ytr2, (ent_te, S_te), torch.tensor(yte2, dtype=torch.long), 'bin', fold_i + 50)
                with torch.no_grad(): prob2 = torch.sigmoid(m2(ent_te, S_te).squeeze(1)).numpy()
            elif method == 'tkg':
                m2 = TKG(V, NS, nout=1)
                m2 = train_model(m2, (dseq_tr, S_tr), ytr2, (dseq_te, S_te), torch.tensor(yte2, dtype=torch.long), 'bin', fold_i + 50)
                with torch.no_grad(): prob2 = torch.sigmoid(m2(dseq_te, S_te).squeeze(1)).numpy()
            elif method == 'lstm_transformer':
                m2 = LSTMModel(seq_tr.size(2), NS, nout=1)
                m2 = train_model(m2, (seq_tr, S_tr), ytr2, (seq_te, S_te), torch.tensor(yte2, dtype=torch.long), 'bin', fold_i + 50)
                with torch.no_grad(): prob2 = torch.sigmoid(m2(seq_te, S_te).squeeze(1)).numpy()
            else:
                m2 = HyperGraph4D(V, NS, nout=1)
                m2 = train_model(m2, (ddim_tr, S_tr), ytr2, (ddim_te, S_te), torch.tensor(yte2, dtype=torch.long), 'bin', fold_i + 50)
                with torch.no_grad(): prob2 = torch.sigmoid(m2(ddim_te, S_te).squeeze(1)).numpy()
            results[method]['bin'].append(metrics_bin(yte2, prob2))
            fold_preds[method]['bin_y'].extend([int(x) for x in yte2])
            fold_preds[method]['bin_p'].extend([float(x) for x in prob2])

            if method == 'lr_xgboost':
                m3 = xgb_train_fit(Xs_tr, days[tr_idx], 'reg')
                pred3 = m3.predict(Xs_te)
            elif method == 'kg_gnn':
                m3 = KGGNN(V, adj, NS, nout=1)
                m3 = train_model(m3, (ent_tr, S_tr), ytr3r, (ent_te, S_te), torch.tensor(yte3r, dtype=torch.float32), 'reg', fold_i + 100)
                with torch.no_grad(): pred3 = m3(ent_te, S_te).squeeze(1).numpy() * reg_sd + reg_mu
            elif method == 'tkg':
                m3 = TKG(V, NS, nout=1)
                m3 = train_model(m3, (dseq_tr, S_tr), ytr3r, (dseq_te, S_te), torch.tensor(yte3r, dtype=torch.float32), 'reg', fold_i + 100)
                with torch.no_grad(): pred3 = m3(dseq_te, S_te).squeeze(1).numpy() * reg_sd + reg_mu
            elif method == 'lstm_transformer':
                m3 = LSTMModel(seq_tr.size(2), NS, nout=1)
                m3 = train_model(m3, (seq_tr, S_tr), ytr3r, (seq_te, S_te), torch.tensor(yte3r, dtype=torch.float32), 'reg', fold_i + 100)
                with torch.no_grad(): pred3 = m3(seq_te, S_te).squeeze(1).numpy() * reg_sd + reg_mu
            else:
                m3 = HyperGraph4D(V, NS, nout=1)
                m3 = train_model(m3, (ddim_tr, S_tr), ytr3r, (ddim_te, S_te), torch.tensor(yte3r, dtype=torch.float32), 'reg', fold_i + 100)
                with torch.no_grad(): pred3 = m3(ddim_te, S_te).squeeze(1).numpy() * reg_sd + reg_mu
            results[method]['reg'].append(metrics_reg(yte3r_raw, pred3))
            fold_preds[method]['reg_y'].extend([float(x) for x in yte3r_raw])
            fold_preds[method]['reg_p'].extend([float(x) for x in pred3])

        print(f'折{fold_i + 1} 完成', flush=True)

    # ---------------- 消融 ----------------
    # 'full' 不重复训练：Full 即完整 4D 模型，其 5 折指标由主循环逐折结果给出（见汇总段），保证 Full 与主结果一致
    abl_res = {}
    if do_ablation:
        abl = {'wo_patient_history': ['evidence', 'treat', 'severity'],
               'wo_clinical_evidence': ['hist', 'treat', 'severity'],
               'wo_treatment_trajectory': ['hist', 'evidence', 'severity'],
               'wo_dynamic_severity': ['hist', 'evidence', 'treat']}
        for name, dims in abl.items():
            fold_res = {'macro_f1': [], 'auc_roc': [], 'pearson_r': []}
            for fold_i, (tr_idx, te_idx) in enumerate(folds):
                tr_pids = [pids[i] for i in tr_idx]
                adj = build_graph_adj(tr_pids, V)
                Xs = prep_struct(X_struct, tr_idx)
                S_tr = torch.tensor(Xs[tr_idx], dtype=torch.float32)
                S_te = torch.tensor(Xs[te_idx], dtype=torch.float32)
                dtr = [day_dim_seq_for(pids[i], dims) for i in tr_idx]
                dte = [day_dim_seq_for(pids[i], dims) for i in te_idx]
                m = HyperGraph4D(V, NS, nout=3)
                m = train_model(m, (dtr, S_tr), torch.tensor(label3[tr_idx], dtype=torch.long),
                                (dte, S_te), torch.tensor(label3[te_idx], dtype=torch.long), 'cls', fold_i, cls_w)
                with torch.no_grad(): pr = F.softmax(m(dte, S_te), 1).numpy().argmax(1)
                fold_res['macro_f1'].append(f1_score(label3[te_idx], pr, average='macro', zero_division=0))
                m2 = HyperGraph4D(V, NS, nout=1)
                m2 = train_model(m2, (dtr, S_tr), torch.tensor(y2[tr_idx], dtype=torch.long),
                                 (dte, S_te), torch.tensor(y2[te_idx], dtype=torch.long), 'bin', fold_i + 50)
                with torch.no_grad(): p2 = torch.sigmoid(m2(dte, S_te).squeeze(1)).numpy()
                fold_res['auc_roc'].append(roc_auc_score(y2[te_idx], p2))
                reg_mu = float(np.nanmean(days[tr_idx]))
                reg_sd = float(np.nanstd(days[tr_idx]))
                reg_sd = reg_sd if reg_sd > 1e-6 else 1.0
                m3 = HyperGraph4D(V, NS, nout=1)
                m3 = train_model(m3, (dtr, S_tr), torch.tensor((np.nan_to_num(days[tr_idx]) - reg_mu) / reg_sd, dtype=torch.float32),
                                 (dte, S_te), torch.tensor((np.nan_to_num(days[te_idx]) - reg_mu) / reg_sd, dtype=torch.float32), 'reg', fold_i + 100)
                with torch.no_grad(): p3 = m3(dte, S_te).squeeze(1).numpy() * reg_sd + reg_mu
                r, _ = ER.pearsonr(np.nan_to_num(days[te_idx]), p3)
                fold_res['pearson_r'].append(r)
            abl_res[name] = {k: float(np.mean(v)) for k, v in fold_res.items()}
            print('消融', name, abl_res[name], flush=True)

    # ---------------- 汇总 ----------------
    out = {}
    for method, r in results.items():
        out[method] = {}
        for task, folds_ in r.items():
            out[method][task] = {k: float(np.mean([f_[k] for f_ in folds_])) for k in folds_[0]}
            cohen[task][method] = [f_['macro_f1' if task == 'cls' else ('auc_roc' if task == 'bin' else 'pearson_r')] for f_ in folds_]
    # Full = 完整 4D 模型（与主结果逐折一致，不重复训练）
    abl_res['full'] = {'macro_f1': float(np.mean(cohen['cls']['4d_hypergraph'])),
                       'auc_roc': float(np.mean(cohen['bin']['4d_hypergraph'])),
                       'pearson_r': float(np.mean(cohen['reg']['4d_hypergraph']))}

    # per-class F1 直接由主循环逐折预测计算（避免重复训练 fold_i+500 模型）
    tt = np.array(fold_preds['4d_hypergraph']['cls_y']); pp = np.array(fold_preds['4d_hypergraph']['cls_p'])
    per_class_f1 = {f'class_{i}': float(f1_score(tt == i, pp == i, zero_division=0)) for i in range(3)}
    hardest = max(per_class_f1, key=lambda k: -per_class_f1[k])

    # McNemar (4D vs each baseline，5 折合并预测)
    # 多分类定义：n01=4D对且基线错，n10=4D错且基线对（用真实标签计算对错）
    mcnemar_vs = {}
    yt_all = np.array(fold_preds['4d_hypergraph']['cls_y'])
    for bl, d_ in mcnemar_pairs.items():
        a = np.array(d_['a']); b = np.array(d_['b'])
        n01 = int(np.sum((a == yt_all) & (b != yt_all)))
        n10 = int(np.sum((a != yt_all) & (b == yt_all)))
        n_eff = n01 + n10
        pval = float(ER.binomtest(min(n01, n10), n_eff, 0.5).pvalue) if n_eff > 0 else 1.0
        mcnemar_vs[bl] = {'p': pval, 'b': n01, 'c': n10}

    # 统计检验：5 折配对 (4D vs each) — Cohen's d / 95%CI(boot) / t-test / Wilcoxon
    stats = {}
    for bl in ['lr_xgboost', 'kg_gnn', 'tkg', 'lstm_transformer']:
        stats[bl] = {}
        for task, tname in [('cls', 'classification_d'), ('bin', 'prediction_d'), ('reg', 'regression_d')]:
            a = np.array(cohen[task]['4d_hypergraph']); bb = np.array(cohen[task][bl])
            diff = a - bb
            sd = diff.std(ddof=1)
            d = float(diff.mean() / sd) if sd > 1e-9 else 0.0
            t, p = ER.ttest_rel(a, bb)
            rng = np.random.default_rng(42)
            boots = [rng.choice(diff, len(diff), replace=True).mean() / max(sd, 1e-9) for _ in range(2000)]
            ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            w_stat, w_p = wilcoxon(a - bb, alternative='two-sided') if not np.allclose(diff, 0) else (0.0, 1.0)
            # 指标本身 95%CI（bootstrap 均值）
            m_boots = [float(np.mean(rng.choice(a, len(a), replace=True) - rng.choice(bb, len(bb), replace=True))) for _ in range(2000)]
            m_ci = [float(np.percentile(m_boots, 2.5)), float(np.percentile(m_boots, 97.5))]
            stats[bl][tname] = {'d': round(d, 2), 'd_ci95': ci, 't_p': float(p),
                                'wilcoxon_p': float(w_p), 'diff_ci95': m_ci}
    return out, abl_res, per_class_f1, hardest, mcnemar_vs, stats, fold_preds


if __name__ == '__main__':
    if '--build-only' in sys.argv:
        cases = R.load_cases()
        ds4 = build_4dchain_dataset(cases)
        pickle.dump(ds4, open(f'{BASE}/exp_4dchain_dataset.pkl', 'wb'))
        inspect(ds4)
        sys.exit(0)
    seed = 42
    do_ablation = True
    for a in sys.argv[1:]:
        if a.startswith('--seed='):
            seed = int(a.split('=')[1])
        elif a == '--no-ablation':
            do_ablation = False
    # 优先用已构建的数据集
    if os.path.exists(f'{BASE}/exp_4dchain_dataset.pkl'):
        with open(f'{BASE}/exp_4dchain_dataset.pkl', 'rb') as f:
            ds4 = pickle.load(f)
        print('载入已构建的四维链数据集')
    else:
        cases = R.load_cases()
        ds4 = build_4dchain_dataset(cases)
        pickle.dump(ds4, open(f'{BASE}/exp_4dchain_dataset.pkl', 'wb'))
    inspect(ds4)
    out, abl_res, per_class_f1, hardest, mcnemar_vs, stats, fold_preds = run4(ds4, seed=seed, do_ablation=do_ablation)
    res_path = f'{BASE}/exp_results_4dchain_seed{seed}.json'
    json.dump({'task_results': out, 'ablation': abl_res, 'per_class_f1': per_class_f1,
               'hardest': hardest, 'mcnemar_vs': mcnemar_vs, 'stats': stats,
               'seed': seed, 'note': '数据表示层=HyperGraphRAG四维链(LLM抽取,排除诊断/住院天数泄漏)；模型与训练同exp_run.py'},
              open(res_path, 'w'), ensure_ascii=False, indent=2)
    # 保存逐折预测（供 Step5 统计检验复算）
    pickle.dump(fold_preds, open(f'{BASE}/exp_foldpreds_seed{seed}.pkl', 'wb'))
    print('结果已保存:', res_path)
    print('== 4D vs 基线 McNemar (分类) ==')
    for bl, v in mcnemar_vs.items():
        print(f'  vs {bl}: p={v["p"]:.4f} (b={v["b"]}, c={v["c"]})')
    print('== 4D vs 基线 配对统计 (d / t_p / wilcoxon_p) ==')
    for bl, v in stats.items():
        print(f'  vs {bl}: cls d={v["classification_d"]["d"]} t_p={v["classification_d"]["t_p"]:.4f} w_p={v["classification_d"]["wilcoxon_p"]:.4f} | '
              f'bin d={v["prediction_d"]["d"]} t_p={v["prediction_d"]["t_p"]:.4f} w_p={v["prediction_d"]["wilcoxon_p"]:.4f} | '
              f'reg d={v["regression_d"]["d"]} t_p={v["regression_d"]["t_p"]:.4f} w_p={v["regression_d"]["wilcoxon_p"]:.4f}')
    for method, r in out.items():
        print(method, 'cls:', {k: round(v, 4) for k, v in r['cls'].items()})
    for method, r in out.items():
        print(method, 'bin:', {k: round(v, 4) for k, v in r['bin'].items()})
    for method, r in out.items():
        print(method, 'reg:', {k: round(v, 4) for k, v in r['reg'].items()})
