# -*- coding: utf-8 -*-
"""构建实验数据：结构化特征 + 按天时序序列 + 四维实体/超图结构"""
import pickle, re
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime

BASE = os.environ.get('EXPERIMENT_BASE', os.getcwd())

def load():
    return pickle.load(open(f'{BASE}/exp_features.pkl', 'rb'))

VITAL_KEYS = ['T', 'P', 'R', 'BP_S', 'BP_D']
LAB_KEYS = ['CRP', 'PCT', 'WBC', 'NEUT', 'HGB', 'PLT', 'ALB']
MED_CATS = ['抗生素', '激素', '支气管扩张剂', '祛痰药', '抗凝药', '利尿剂', '血管活性药', '氧疗/通气', '营养支持']

def parse_dt(s):
    s = str(s).strip()
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d'):
        try:
            return datetime.strptime(s[:19], fmt)
        except Exception:
            continue
    return None

def build_daily_seq(f, max_days):
    """按住院日分桶：每日特征向量 [8个生命体征峰值 + 7个实验室值 + 9类用药计数]"""
    admit = parse_dt(f.get('admit_time'))
    if admit is None:
        return None
    D = 8 + len(LAB_KEYS) + len(MED_CATS)
    seq = np.zeros((max_days, D), dtype=float)
    # 生命体征
    for k in VITAL_KEYS:
        for t, v in f['vitals'].get(k, []):
            dt = parse_dt(t)
            if dt is None:
                continue
            di = (dt - admit).days
            if 0 <= di < max_days:
                seq[di, VITAL_KEYS.index(k)] = max(seq[di, VITAL_KEYS.index(k)], v)
    # 实验室（当日最后一次取值）
    for k in LAB_KEYS:
        for t, v in f['labs'].get(k, []):
            dt = parse_dt(t)
            if dt is None:
                continue
            di = (dt - admit).days
            if 0 <= di < max_days:
                seq[di, 8 + LAB_KEYS.index(k)] = v
    # 用药
    for t, cat in f.get('med_times', []):
        dt = parse_dt(t)
        if dt is None:
            continue
        di = (dt - admit).days
        if 0 <= di < max_days:
            seq[di, 8 + len(LAB_KEYS) + MED_CATS.index(cat)] += 1
    return seq

def build(feats_all):
    pids = sorted(feats_all.keys())
    X_struct = []
    seq_data = {}
    entity_data = {}
    ent_counter = Counter()
    meta = []

    # 住院天数（用于序列长度与回归代理标签）
    days_all = [f['days'] if f['days'] else 8 for f in feats_all.values()]
    max_days = int(np.ceil(np.percentile(days_all, 95)))
    print('序列最大长度(95分位住院天数):', max_days)
    day_ent_data = {}  # pid -> list of (day, set(ent))

    for pid, f in feats_all.items():
        ents = defaultdict(set)
        for s in f['symptoms']:
            ents['hist'].add(('sym', s))
        for c in f['comorb']:
            for kw in ['高血压','糖尿病','冠心病','心力衰竭','肺心病','呼吸衰竭','肺癌','肺结核','支气管扩张','肺气肿','哮喘','贫血','肾功能不全','肝功能不全','骨质疏松','胃食管反流','房颤','脑梗死','帕金森','痴呆','前列腺增生','焦虑','抑郁','睡眠呼吸暂停','高脂血症','高尿酸']:
                if kw in c:
                    ents['hist'].add(('diag', kw))
        for key, vals in f['labs'].items():
            nums = [v for _, v in vals if v is not None]
            if not nums:
                continue
            mx = max(nums)
            if key == 'CRP' and mx > 10: ents['evidence'].add(('lab', 'CRP_high'))
            if key == 'PCT' and mx > 0.5: ents['evidence'].add(('lab', 'PCT_high'))
            if key == 'WBC' and mx > 10: ents['evidence'].add(('lab', 'WBC_high'))
            if key == 'WBC' and mx < 3: ents['evidence'].add(('lab', 'WBC_low'))
            if key == 'NEUT' and mx > 75: ents['evidence'].add(('lab', 'NEUT_high'))
            if key == 'HGB' and mx < 110: ents['evidence'].add(('lab', 'HGB_low'))
            if key == 'ALB' and mx < 35: ents['evidence'].add(('lab', 'ALB_low'))
        for m in f['meds']:
            ents['treat'].add(('med', m))
        vt = f['vitals']
        if 'T' in vt:
            ts = [v for _, v in vt['T']]
            if ts and max(ts) >= 37.3: ents['severity'].add(('vit', 'T_fever'))
        if 'R' in vt:
            rs = [v for _, v in vt['R']]
            if rs and max(rs) >= 24: ents['severity'].add(('vit', 'R_tachypnea'))
        if 'P' in vt:
            ps = [v for _, v in vt['P']]
            if ps and max(ps) >= 100: ents['severity'].add(('vit', 'P_tachy'))
        if 'BP_S' in vt:
            ss = [v for _, v in vt['BP_S']]
            if ss and min(ss) < 90: ents['severity'].add(('vit', 'BP_low'))
        if f['cat_scores']:
            ents['severity'].add(('score', f'cat_{max(f["cat_scores"])}'))
        if f['mmrc_scores']:
            ents['severity'].add(('score', f'mmrc_{max(f["mmrc_scores"])}'))
        if f['golds']:
            ents['severity'].add(('score', f'gold_{f["golds"][0]}'))
        entity_data[pid] = ents
        for e in set(e for es in ents.values() for e in es):
            ent_counter[e] += 1

        seq_data[pid] = build_daily_seq(f, max_days)

        # 按天实体序列（TKG用）：病史/证据/严重度归入入院日，治疗按医嘱开始时间
        admit = parse_dt(f.get('admit_time'))
        day_e = defaultdict(set)
        for e in ents['hist']:
            day_e[0].add(e)
        for e in ents['evidence']:
            day_e[0].add(e)
        for e in ents['severity']:
            day_e[0].add(e)
        for t, cat in f.get('med_times', []):
            dt = parse_dt(t)
            di = 0
            if dt is not None and admit is not None:
                di = (dt - admit).days
            di = max(0, min(di, max_days - 1))
            day_e[di].add(('med', cat))
        day_ent_data[pid] = [(d, sorted(es)) for d, es in sorted(day_e.items())]

        row = [f['age'] if f['age'] else np.nan,
               1 if f['sex'] == '男' else 0,
               f['smoking'],
               len(set(f['comorb']))]
        for key in LAB_KEYS:
            vals = [v for _, v in f['labs'][key]]
            row.append(max(vals) if vals else np.nan)
        for vk in VITAL_KEYS[:4]:
            vals = [v for _, v in f['vitals'].get(vk, [])]
            row.append(max(vals) if vals else np.nan)
        mc = Counter(f['meds'])
        for cat in MED_CATS:
            row.append(mc.get(cat, 0))
        X_struct.append(row)
        meta.append({'pid': pid, 'group': f['group'], 'label3': f['group_label'],
                     'days': f['days'], 'cat_last': (max(f['cat_scores']) if f['cat_scores'] else None)})

    ent2id = {}
    for e, c in ent_counter.most_common():
        if c >= 2:
            ent2id[e] = len(ent2id)
    print('实体总数:', len(ent_counter), '| 过滤后(>=2):', len(ent2id))

    X_struct = np.array(X_struct, dtype=float)
    return {
        'pids': pids, 'X_struct': X_struct, 'ent2id': ent2id,
        'entity_data': entity_data, 'seq_data': seq_data, 'max_days': max_days,
        'day_ent_data': day_ent_data, 'meta': meta,
    }

def inspect(ds):
    print('X_struct shape:', ds['X_struct'].shape)
    labels = np.array([m['label3'] for m in ds['meta']])
    print('标签分布:', dict(Counter(labels)))
    days = np.array([m['days'] if m['days'] else np.nan for m in ds['meta']])
    print('住院天数: mean=%.1f std=%.1f' % (np.nanmean(days), np.nanstd(days)))
    for g in [0, 1, 2]:
        dd = days[labels == g]
        print(f'  组{g} n={len(dd)} days: mean=%.1f std=%.1f' % (np.nanmean(dd), np.nanstd(dd)))
    seq_ok = sum(1 for pid in ds['pids'] if ds['seq_data'][pid] is not None)
    print('时序序列可用:', seq_ok, '/', len(ds['pids']))
    dims = ['hist', 'evidence', 'treat', 'severity']
    for dm in dims:
        c = sum(1 for pid in ds['pids'] for e in ds['entity_data'][pid].get(dm, set()) if e in ds['ent2id'])
        print(f'  维度{dm} 有效实体关联数:', c)
    return labels, days

if __name__ == '__main__':
    feats = load()
    ds = build(feats)
    pickle.dump(ds, open(f'{BASE}/exp_dataset.pkl', 'wb'))
    inspect(ds)
