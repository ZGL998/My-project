
# -*- coding: utf-8 -*-
"""
import os
真实实验：从253例COPD患者原始JSON提取特征
- Table 1 基线特征
- 结构化特征（LR/XGBoost用）
- 时序特征（LSTM/Transformer用）
- 四维实体特征（KG+GNN/TKG/4D-HyperGraph用）
"""
import json, glob, os, re, pickle
from collections import Counter, defaultdict
import numpy as np

BASE = os.environ.get('EXPERIMENT_BASE', os.getcwd())
GROUPS = [
    ('A', '慢性阻塞性肺病'),
    ('B', '慢性阻塞性肺病伴有急性加重'),
    ('C', '慢性阻塞性肺病伴有急性下呼吸道感染'),
]
OUT = os.path.join(BASE, 'exp_features.pkl')

# 症状/诊断关键词（呼吸系统）
SYMPTOM_KW = ['咳嗽','咳痰','憋喘','气短','喘息','呼吸困难','胸闷','发热','心悸','乏力','纳差',
              '盗汗','咯血','胸痛','紫绀','水肿','尿少','精神萎靡','嗜睡','意识障碍','咳粉红色泡沫痰']
COPD_KW = ['慢性阻塞性肺','慢阻肺','肺气肿','支气管哮喘','支气管扩张','肺心病','肺动脉高压','呼吸衰竭',
           '心力衰竭','冠心病','高血压','糖尿病','胃食管反流','睡眠呼吸暂停','肺癌','肺结核','肺炎','胸腔积液']
MED_CATEGORY = {
    '抗生素': ['头孢','阿莫西林','青霉素','左氧氟沙星','莫西沙星','环丙沙星','阿奇霉素','多西环素','美罗培南','哌拉西林','亚胺培南','万古霉素','替加环素'],
    '激素': ['甲泼尼龙','泼尼松','布地奈德','地塞米松','氢化可的松','倍氯米松','氟替卡松'],
    '支气管扩张剂': ['沙丁胺醇','异丙托溴铵','噻托溴铵','布地奈德福莫特罗','沙美特罗','特布他林','氨茶碱','多索茶碱','茶碱'],
    '祛痰药': ['氨溴索','乙酰半胱氨酸','羧甲司坦','沐舒坦'],
    '抗凝药': ['低分子肝素','华法林','利伐沙班','达比加群','依诺肝素'],
    '利尿剂': ['呋塞米','托拉塞米','氢氯噻嗪','螺内酯'],
    '血管活性药': ['多巴胺','去甲肾上腺素','硝酸甘油','硝普钠'],
    '氧疗/通气': ['氧疗','吸氧','无创通气','双水平','呼吸机','有创通气'],
    '营养支持': ['肠内营养','瑞能','百普力','氨基酸','脂肪乳'],
}
LAB_MAP = {
    'CRP': ['C反应蛋白', 'CRP'],
    'PCT': ['降钙素原', 'PCT'],
    'WBC': ['白细胞计数', '白细胞', 'WBC'],
    'NEUT': ['中性粒细胞百分比', '中性粒细胞比率'],
    'HGB': ['血红蛋白'],
    'PLT': ['血小板计数', '血小板'],
    'ALB': ['白蛋白'],
    'pH': ['pH'],
    'PaCO2': ['二氧化碳分压', 'PaCO2', 'PCO2'],
    'PaO2': ['氧分压', 'PaO2', 'PO2'],
}

# 排除易混淆项目（如"平均血红蛋白浓度"(MCHC)、"前白蛋白"、"白细胞介素"等）
LAB_EXCLUDE = {
    'WBC': ['白细胞介素', '尿白细胞'],
    'NEUT': ['中性粒细胞绝对值'],
    'HGB': ['平均血红蛋白', '血红蛋白浓度', '血红蛋白含量', '血红蛋白量'],
    'PLT': ['平均血小板', '血小板压积'],
    'ALB': ['前白蛋白', '微量白蛋白', '白蛋白/球蛋白', '白球比'],
}

def parse_num(s):
    s = str(s).replace('↓','').replace('↑','').replace(' ','')
    try:
        return float(s)
    except:
        m = re.search(r'-?\d+\.?\d*', s)
        return float(m.group()) if m else None

def has_smoking_history(ps):
    """吸烟史（含已戒）：排除"无吸烟/不吸烟/否认吸烟"的表述。
    以"吸烟"前4字符是否含 无/不/否 判断，避免把"无吸毒史，吸烟X年"误判为无吸烟史。"""
    if '吸烟' not in ps:
        return 0
    for m in re.finditer('吸烟', ps):
        pre = ps[max(0, m.start() - 4):m.start()]
        if any(x in pre for x in ['无', '不', '否']):
            continue
        return 1
    return 0

def load_all():
    patients = []
    for gname, sub in GROUPS:
        for fp in sorted(glob.glob(f'{BASE}/{sub}/zy@*.json')):
            patients.append((gname, fp))
    return patients

def extract(pid, fp):
    d = json.load(open(fp, encoding='utf-8'))
    feats = {'pid': pid, 'group': None, 'age': None, 'sex': None, 'days': None,
             'smoking': 0, 'golds': [], 'cat_scores': [], 'mmrc_scores': [],
             'fev1pct': [], 'comorb': [], 'meds': [], 'med_times': [],
             'labs': defaultdict(list), 'admit_time': None,
             'symptoms': [], 'vitals': [], 'diag_text': '', 'chief': '', 'hist_text': ''}
    emr = d.get('电子病历') or []
    if emr and isinstance(emr[0], dict):
        e = emr[0]
        feats['admit_time'] = e.get('入院时间')
        feats['age'] = parse_num(e.get('年龄'))
        feats['sex'] = e.get('性别')
        feats['chief'] = str(e.get('主诉', ''))
        feats['hist_text'] = str(e.get('现病史', '')) + str(e.get('既往史', '')) + str(e.get('个人史', ''))
        for kw in SYMPTOM_KW:
            if kw in feats['hist_text'] or kw in feats['chief']:
                feats['symptoms'].append(kw)
        ps = str(e.get('个人史', ''))
        feats['smoking'] = has_smoking_history(ps)
        diag = str(e.get('出院诊断', ''))
        if diag:
            feats['diag_text'] = diag
    home = d.get('病案首页') or []
    if home and isinstance(home[0], dict):
        h = home[0]
        if feats['age'] is None:
            feats['age'] = parse_num(h.get('年龄') or h.get('年龄（岁）'))
        if feats['sex'] is None:
            feats['sex'] = h.get('性别')
        feats['days'] = parse_num(h.get('实际住院（天）'))
        for i in range(1, 41):
            nm = h.get(f'出院其他诊断名称{i}')
            if nm:
                feats['comorb'].append(str(nm).strip())
        main = str(h.get('出院主要诊断名称', ''))
        if main:
            feats['diag_text'] += main
    # 实验室检查
    for r in d.get('实验室检查') or []:
        if not isinstance(r, dict):
            continue
        iname = str(r.get('检验指标名', '')) or str(r.get('英文名称', ''))
        val = r.get('检验指标值')
        t = str(r.get('报告日期', '')) or str(r.get('检查时间', ''))
        for key, pats in LAB_MAP.items():
            if any(p in iname for p in pats):
                if any(ex in iname for ex in LAB_EXCLUDE.get(key, [])):
                    continue
                num = parse_num(val)
                if num is not None:
                    feats['labs'][key].append((t, num))
                break
    # 医嘱
    for r in d.get('医嘱') or []:
        if not isinstance(r, dict):
            continue
        nm = str(r.get('医嘱名称', '')) or str(r.get('医嘱内容', ''))
        for cat, pats in MED_CATEGORY.items():
            if any(p in nm for p in pats):
                feats['meds'].append(cat)
                feats['med_times'].append((str(r.get('开始时间', '')), cat))
                break
    # 护理记录生命体征（取入院首日与出院前日）
    vits = defaultdict(list)
    for r in d.get('护理记录') or []:
        if not isinstance(r, dict):
            continue
        nm = str(r.get('项目名称', ''))
        tt = str(r.get('记录时间', ''))
        v = r.get('项目值(数值)') if r.get('项目值(数值)') not in (None, '') else r.get('项目值(字符)')
        num = parse_num(v)
        if num is None:
            continue
        if '体温' in nm:
            vits['T'].append((tt, num))
        elif '脉搏' in nm:
            vits['P'].append((tt, num))
        elif '呼吸' in nm:
            vits['R'].append((tt, num))
        elif '血压' in nm and ('收缩' in nm or '高压' in nm):
            vits['BP_S'].append((tt, num))
        elif '血压' in nm and ('舒张' in nm or '低压' in nm):
            vits['BP_D'].append((tt, num))
    feats['vitals'] = dict(vits)
    # 病程记录/护理记录/电子病历中的评分
    txt_all = ''
    for r in (d.get('病程记录') or []) + (d.get('护理记录') or []):
        if isinstance(r, dict):
            txt_all += str(r.get('病程记录', '')) + str(r.get('观察记录', '')) + str(r.get('评估内容', ''))
    for r in (d.get('电子病历') or []):
        if isinstance(r, dict):
            txt_all += str(r.get('出院记录', '')) + str(r.get('现病史', '')) + str(r.get('既往史', '')) + str(r.get('出院诊断', '')) + str(r.get('入院诊断', ''))
    for m in re.finditer(r'CAT\s*[评分]{0,2}[为]?\s*[:：]?\s*(\d{1,2})\s*分', txt_all):
        feats['cat_scores'].append(int(m.group(1)))
    for m in re.finditer(r'mMRC\s*[评分]{0,2}?\s*[:：]?\s*(\d{1,2})\s*[分级]?', txt_all):
        feats['mmrc_scores'].append(int(m.group(1)))
    for m in re.finditer(r'GOLD\s*([IVⅠⅡⅢⅣ1-4])\s*级', txt_all):
        feats['golds'].append(m.group(1))
    for m in re.finditer(r'FEV1(?:占预计值|%pred)?\s*[:：]?\s*(\d{1,3}(?:\.\d)?)\s*%', txt_all):
        feats['fev1pct'].append(float(m.group(1)))
    # 入院诊断中的并发症/合并症并入
    return feats

def main():
    patients = load_all()
    print('加载患者:', len(patients))
    all_feats = {}
    for gname, fp in patients:
        pid = os.path.basename(fp)
        try:
            f = extract(pid, fp)
        except Exception as ex:
            print('失败', fp, ex)
            continue
        f['group'] = gname
        f['group_label'] = {'A': 0, 'B': 1, 'C': 2}[gname]
        all_feats[pid] = f
    print('成功提取:', len(all_feats))
    pickle.dump(all_feats, open(OUT, 'wb'))
    # 汇总覆盖度
    from collections import Counter
    gc = Counter(f['group'] for f in all_feats.values())
    print('分组分布:', dict(gc))
    for key in ['age','sex','days','smoking','cat_scores','mmrc_scores','golds','fev1pct','comorb']:
        has = sum(1 for f in all_feats.values() if f[key] not in (None, [], 0, '', 0.0))
        print(f'  {key}: {has}')

if __name__ == '__main__':
    main()
