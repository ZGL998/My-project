# -*- coding: utf-8 -*-
"""
Step 2 提取质量验证（程序化近似）
对四维链 251 例，统计 4 个维度 × 13 个字段分组的核心字段非空率/条目量，
作为「人工对照审核 30 例」的程序化代理指标。所有数值来自真实四维链数据。
"""
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.environ.get('EXPERIMENT_BASE', os.getcwd()))
import run_hgr_eval as R

# 维度 → 分组（对应 Step1 四个抽取提示词）
DIM2GROUPS = {
    '患者历史': ['患者人口学基础信息', '全维度病史信息'],
    '临床证据': ['症状体征信息', '检查检验信息', '诊断信息'],
    '治疗轨迹': ['药物治疗信息', '手术治疗信息', '诊疗过程信息', '出院信息'],
    '动态严重度': ['疾病分期与严重程度', '并发症信息', '功能评分信息', '预后信息'],
}
BUCKET_BY_GROUP = {
    '患者人口学基础信息': 'history_chain', '全维度病史信息': 'history_chain',
    '症状体征信息': 'evidence_chain', '检查检验信息': 'evidence_chain', '诊断信息': 'evidence_chain',
    '药物治疗信息': 'care_chain', '手术治疗信息': 'care_chain', '诊疗过程信息': 'care_chain', '出院信息': 'care_chain',
    '疾病分期与严重程度': 'severity_chain', '并发症信息': 'severity_chain',
    '功能评分信息': 'severity_chain', '预后信息': 'severity_chain',
}
# 每个分组关心的关键字段（非空即计为有效）
KEY_FIELDS = {
    '患者人口学基础信息': ['性别', '年龄', '入院时间', '出院时间', '入院科室', '出院科室'],
    '全维度病史信息': ['主诉', '现病史', '既往史', '个人史', '家族史', '过敏史', '婚育史', '首程病例特点'],
    '症状体征信息': ['主要症状', '现病史摘要', '体格检查', '生命体征'],
    '检查检验信息': ['检验项目列表', '影像检查列表', '其他检查列表'],
    '诊断信息': ['入院诊断', '出院诊断', '主要诊断', '其他诊断', 'ICD编码列表', '病理诊断', '诊断依据', '首程诊断推理'],
    '药物治疗信息': ['用药医嘱列表', '用药用法', '用药疗程'],
    '手术治疗信息': ['手术列表', '手术详情', '麻醉记录'],
    '诊疗过程信息': ['治疗原则', '诊疗过程描述', '全过程诊疗时序总结', '治疗效果'],
    '出院信息': ['出院日期', '出院症状', '出院医嘱', '出院结局', '离院方式', '出院诊断'],
    '疾病分期与严重程度': ['疾病分期', '严重程度分级', '分期依据'],
    '并发症信息': ['并发症列表', '并发症数量', '并发症分级'],
    '功能评分信息': ['入院功能评分', '出院功能评分', '评分变化'],
    '预后信息': ['住院天数', '出院结局', '再入院风险', '预后评估'],
}


def nonempty(v):
    if v is None:
        return False
    if isinstance(v, (list, tuple)):
        return len(v) > 0
    if isinstance(v, dict):
        return len(v) > 0
    s = str(v).strip()
    return s != '' and s != '无' and s != '无相关信息' and s != '[]'


def main():
    cases = R.load_cases()
    n = len(cases)
    print(f'病例数: {n}\n')

    # 分组 → {字段 → 非空计数}
    group_stat = defaultdict(lambda: defaultdict(int))
    dim_stat = defaultdict(lambda: {'valid': 0, 'total': 0})
    for pid, case in cases.items():
        for gname, fields in KEY_FIELDS.items():
            bucket = BUCKET_BY_GROUP[gname]
            data = {}
            for g in case['groups'][bucket]:
                if g['g'] == gname:
                    data = g['d'] if isinstance(g['d'], dict) else {}
                    break
            for f in fields:
                if nonempty(data.get(f)):
                    group_stat[gname][f] += 1

    # 输出表格：按维度
    print(f"{'分组':<14}{'字段':<12}{'非空/总数':>12}{'覆盖率':>9}")
    print('-' * 52)
    overall_valid = overall_total = 0
    for dim, groups in DIM2GROUPS.items():
        dim_valid = dim_total = 0
        for gname in groups:
            stat = group_stat[gname]
            for f in KEY_FIELDS[gname]:
                c = stat.get(f, 0)
                dim_valid += c
                dim_total += n
                overall_valid += c
                overall_total += n
                print(f'{gname:<14}{f:<12}{c:>5}/{n:<7}{100*c/n:>8.1f}%')
            print('-' * 52)
        dim_stat[dim]['valid'] = dim_valid
        dim_stat[dim]['total'] = dim_total

    print(f"\n{'维度':<8}{'有效字段单元':>12}{'总字段单元':>12}{'覆盖率':>10}")
    print('-' * 44)
    for dim in DIM2GROUPS:
        v = dim_stat[dim]['valid']
        t = dim_stat[dim]['total']
        print(f'{dim:<8}{v:>12}{t:>12}{100*v/t:>9.1f}%')
    v, t = overall_valid, overall_total
    print('-' * 44)
    print(f'总体   {v:>12}{t:>12}{100*v/t:>9.1f}%')

    # 抽检 30 例（每组 10 例，按 label 分层随机，固定种子）输出覆盖汇总
    import random
    rng = random.Random(42)
    by_label = defaultdict(list)
    for pid, case in cases.items():
        by_label[case['label']].append(pid)
    sample = []
    for label in ['A_stable', 'B_ae', 'C_infection']:
        sample += rng.sample(by_label[label], 10)
    print(f"\n抽检 30 例（每组 10 例，种子 42）各维度有效字段单元占比：")
    for pid in sample:
        pass
    samp_stat = defaultdict(lambda: [0, 0])
    for pid in sample:
        case = cases[pid]
        for dim, groups in DIM2GROUPS.items():
            for gname in groups:
                bucket = BUCKET_BY_GROUP[gname]
                data = {}
                for g in case['groups'][bucket]:
                    if g['g'] == gname:
                        data = g['d'] if isinstance(g['d'], dict) else {}
                        break
                for f in KEY_FIELDS[gname]:
                    samp_stat[dim][1] += 1
                    if nonempty(data.get(f)):
                        samp_stat[dim][0] += 1
    for dim in DIM2GROUPS:
        v, t = samp_stat[dim]
        print(f'  {dim:<8} {v}/{t} = {100*v/t:.1f}%')


if __name__ == '__main__':
    main()
