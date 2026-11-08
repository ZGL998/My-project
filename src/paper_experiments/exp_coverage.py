
# -*- coding: utf-8 -*-
"""扫描253例患者数据覆盖度：确定哪些特征/标签可提取。"""
import os
import json, glob, re, os
from collections import Counter

BASE = os.environ.get('EXPERIMENT_BASE', os.getcwd())
GROUPS = [
    ('A', '慢性阻塞性肺病', 0),
    ('B', '慢性阻塞性肺病伴有急性加重', 1),
    ('C', '慢性阻塞性肺病伴有急性下呼吸道感染', 2),
]

def load_patients():
    patients = []
    for gname, sub, glabel in GROUPS:
        for fp in sorted(glob.glob(f'{BASE}/{sub}/zy@*.json')):
            patients.append((os.path.basename(fp), glabel, fp))
    return patients

def main():
    patients = load_patients()
    print('总患者数:', len(patients))
    stats = {
        'has_emr': 0, 'has_progress': 0, 'has_order': 0, 'has_lab': 0, 'has_nursing': 0, 'has_home': 0,
        'cat_any': 0, 'cat_multi': 0, 'mmrc_any': 0, 'fev1_any': 0, 'gold_any': 0,
        'smoking_any': 0, 'cat_first': 0, 'cat_last': 0,
    }
    cat_vals = []
    for name, glabel, fp in patients:
        d = json.load(open(fp, encoding='utf-8'))
        emr = d.get('电子病历') or []
        progress = d.get('病程记录') or []
        orders = d.get('医嘱') or []
        lab = d.get('实验室检查') or []
        nursing = d.get('护理记录') or []
        home = d.get('病案首页') or []
        if emr: stats['has_emr'] += 1
        if progress: stats['has_progress'] += 1
        if orders: stats['has_order'] += 1
        if lab: stats['has_lab'] += 1
        if nursing: stats['has_nursing'] += 1
        if home: stats['has_home'] += 1
        # 病程记录文本
        texts = [str(r.get('病程记录', '')) for r in progress if isinstance(r, dict)]
        joined = '\n'.join(texts)
        if 'CAT' in joined or 'CAT评分' in joined or 'CAT 评分' in joined:
            stats['cat_any'] += 1
            # 提取 CAT 评分数字
            found = re.findall(r'CAT\s*评分?\s*[:：]?\s*(\d{1,2})\s*分', joined)
            if found:
                cat_vals.append(len(found))
                if len(found) >= 2: stats['cat_multi'] += 1
                if len(found) >= 1: stats['cat_first'] += 1
                if len(found) >= 2: stats['cat_last'] += 1
        if 'mMRC' in joined or 'mMRC评分' in joined:
            stats['mmrc_any'] += 1
        if 'FEV1' in joined:
            stats['fev1_any'] += 1
        if 'GOLD' in joined:
            stats['gold_any'] += 1
        # 电子病历个人史吸烟
        for rec in emr:
            if isinstance(rec, dict):
                ph = str(rec.get('个人史', ''))
                if '吸烟' in ph:
                    stats['smoking_any'] += 1
                    break
    print('\n=== 覆盖度统计 (n=%d) ===' % len(patients))
    for k, v in stats.items():
        print(f'  {k}: {v} ({v/len(patients)*100:.1f}%)')
    print('  CAT评分出现次数分布:', Counter(cat_vals))

if __name__ == '__main__':
    main()
