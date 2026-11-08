import argparse
from collections import defaultdict
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
COPD_ROOT = SCRIPT_DIR / "出院主诊断_COPD三类"
DEFAULT_RULES = COPD_ROOT / "四维相似链处理规则.json"
DEFAULT_INPUT_DIR = COPD_ROOT / "慢性阻塞性肺病"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "四维相似链输出"


RULE_FILE_MAP = {
    "病史背景": "history_background_similarity.json",
    "临床证据链": "clinical_evidence_chain_similarity.json",
    "诊疗轨迹": "treatment_trajectory_similarity.json",
    "临床严重度": "clinical_severity_similarity.json",
}

GROUP_PLACEHOLDERS = {
    "患者人口学基础信息": "无相关信息",
    "全维度病史信息": "无相关病史",
    "症状体征信息": "无相关症状体征",
    "检查检验信息": "无相关检查结果",
    "诊断信息": "无相关诊断记录",
    "药物治疗信息": "无相关用药记录",
    "手术治疗信息": "无手术史",
    "诊疗过程信息": "无相关治疗记录",
    "出院信息": "无相关出院记录",
    "疾病分期与严重程度": "无相关分期记录",
    "并发症信息": "无相关并发症",
    "功能评分信息": "无相关功能评分",
    "预后信息": "无相关预后评估",
}

DIRECT_IDENTIFIER_LABELS = [
    "姓名",
    "现住址",
    "地址",
    "联系方式",
    "联系电话",
    "住院号/门诊号",
    "住院号",
    "门诊号",
    "身份证号",
    "身份证号码",
    "唯一号",
    "病案号",
    "出生日期",
]

SENSITIVE_DEMOGRAPHIC_LABELS = {
    "职业",
    "民族",
    "婚姻状态",
    "婚姻状况",
    "婚姻",
    "国籍",
    "籍贯",
    "籍贯省（自治区、直辖市）",
    "入院途径",
}

MISSING_TOKENS = {
    "-",
    "--",
    "---",
    "——",
    "/",
    "／",
    "\\",
    "nan",
    "none",
    "null",
    "n/a",
    "na",
    "未提供",
    "未记录",
    "未填写",
    "不详",
}

PROCESS_SUMMARY_KEYWORDS = [
    "患者诉",
    "今日查房",
    "查房",
    "病情变化",
    "一般情况",
    "病例特点",
    "诊疗计划",
    "现患者",
]

DIAGNOSIS_SUMMARY_KEYWORDS = [
    "诊断",
    "考虑",
    "符合",
    "依据",
    "鉴别",
    "分析",
    "判断",
]

TREATMENT_ACTION_KEYWORDS = [
    "给予",
    "予以",
    "继续",
    "完善",
    "安排",
    "建议",
    "嘱",
    "观察",
    "监测",
    "复查",
    "治疗",
    "会诊",
    "吸入",
    "平喘",
    "祛痰",
    "抗感染",
    "出院",
]

RESULT_SUMMARY_KEYWORDS = [
    "好转",
    "改善",
    "缓解",
    "稳定",
    "未诉",
    "无明显异常变化",
    "一般情况尚可",
    "较前",
    "出院",
]

EVIDENCE_REFERENCE_KEYWORDS = [
    "CT",
    "胸部CT",
    "肺部CT",
    "影像",
    "肺功能",
    "血气",
    "病理",
    "支气管舒张试验",
    "6MWT",
]

HISTORY_SYMPTOM_KEYWORDS = [
    "咳嗽",
    "咳痰",
    "气短",
    "胸闷",
    "喘",
    "喘息",
    "呼吸困难",
    "发热",
    "乏力",
    "胸痛",
    "活动后",
    "受凉",
]

NON_SURGICAL_PROCEDURE_KEYWORDS = [
    "分析",
    "CT",
    "MRI",
    "MR",
    "超声",
    "彩超",
    "心电图",
    "肺功能",
    "检查",
    "检验",
    "测定",
    "评估",
    "化验",
    "胸片",
    "X线",
    "血气",
    "镜检",
]

STAFF_ROLE_LABELS = [
    "副主任医师",
    "主任医师",
    "主治医师",
    "主诊医师",
    "上级医师",
    "住院总医师",
    "总住院医师",
    "住院总",
    "住院医师",
    "专培医师",
    "医师",
    "医生",
    "护士长",
    "责任护士",
    "护士",
]

COMMON_SINGLE_CHAR_SURNAMES = set(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
    "戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐"
    "费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元顾孟平黄和"
    "穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜"
    "阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞"
    "万支柯昝管卢莫房裘缪解应宗丁宣贲邓郁单杭洪包诸左石崔吉钮龚程嵇邢"
    "滑裴陆荣翁荀羊於惠甄曲家封芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山"
    "谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘厉戎祖武符刘景詹束龙叶幸司韶"
    "黎薄印宿白怀蒲邰从鄂索咸籍赖卓蔺屠蒙池乔阴胥能苍双闻莘党翟谭贡劳"
    "逄姬申扶堵冉宰郦雍郤璩桑桂濮牛寿通边扈燕冀郏浦尚农温别庄晏柴瞿阎"
    "充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡国文寇广禄阙"
    "东欧殳沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾毋沙乜养鞠须"
    "丰巢关蒯相查后荆红游竺权逯盖益桓公代"
)

COMMON_COMPOUND_SURNAMES = {
    "万俟",
    "司马",
    "上官",
    "欧阳",
    "夏侯",
    "诸葛",
    "闻人",
    "东方",
    "赫连",
    "皇甫",
    "尉迟",
    "公羊",
    "澹台",
    "公冶",
    "宗政",
    "濮阳",
    "淳于",
    "单于",
    "太叔",
    "申屠",
    "公孙",
    "仲孙",
    "轩辕",
    "令狐",
    "钟离",
    "宇文",
    "长孙",
    "慕容",
    "司徒",
    "司空",
}

STAFF_ROLE_PATTERN = re.compile(
    r"(" + "|".join(map(re.escape, STAFF_ROLE_LABELS)) + r")(?!\d)"
)

STAFF_NAME_STOPWORDS = {
    "患者",
    "报告",
    "记录",
    "查房",
    "首次",
    "常规",
    "日常",
    "床旁",
    "仅供",
    "签名",
    "医生",
    "医师",
    "护士",
    "主任",
    "主治",
    "住院",
    "责任",
}


def normalize_key(value):
    return re.sub(r"\s+", "", str(value or ""))


def clean_scalar(value):
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.lower() in MISSING_TOKENS or text in MISSING_TOKENS:
            return None
        return text
    return value


def sanitize(value):
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return clean_scalar(value)


def compact_structure(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            cleaned = compact_structure(item)
            if cleaned is None:
                continue
            if isinstance(cleaned, (dict, list)) and not cleaned:
                continue
            result[str(key)] = cleaned
        return result
    if isinstance(value, list):
        items = []
        for item in value:
            cleaned = compact_structure(item)
            if cleaned is None:
                continue
            if isinstance(cleaned, (dict, list)) and not cleaned:
                continue
            items.append(cleaned)
        return items
    return clean_scalar(value)


def build_case_id(file_path):
    stem = Path(file_path).stem
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:12].upper()
    return f"CASE_{digest}"


def collect_direct_identifiers(data):
    basic = first_record(data, "基本数据")
    home = first_record(data, "病案首页")
    identifiers = []
    field_names = [
        "姓名",
        "现住址",
        "地址",
        "联系方式",
        "联系电话",
        "住院号/门诊号",
        "住院号",
        "门诊号",
        "身份证号码",
        "身份证号",
        "唯一号",
        "病案号",
        "出生日期",
    ]
    for field_name in field_names:
        value = get_value(basic, field_name) or get_value(home, field_name)
        value = clean_scalar(value)
        if value is not None:
            identifiers.append(str(value))
    seen = set()
    deduped = []
    for item in identifiers:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def looks_like_staff_name(name):
    name = (name or "").strip()
    if len(name) < 2 or len(name) > 4:
        return False
    return not any(token in name for token in STAFF_NAME_STOPWORDS)


def is_likely_person_name(name):
    if len(name) == 2:
        return name[0] in COMMON_SINGLE_CHAR_SURNAMES
    if len(name) == 3:
        return name[:2] in COMMON_COMPOUND_SURNAMES or name[0] in COMMON_SINGLE_CHAR_SURNAMES
    if len(name) == 4:
        return name[:2] in COMMON_COMPOUND_SURNAMES
    return False


def split_staff_context(candidate_text):
    text = (candidate_text or "").strip()
    for length in (4, 3, 2):
        if len(text) < length:
            continue
        name = text[-length:]
        leading_text = text[:-length]
        if any(token in leading_text for token in ("主任", "主治", "主诊", "专培", "医师", "医生", "护士")):
            continue
        if looks_like_staff_name(name) and is_likely_person_name(name):
            return leading_text, name
    return None, None


def split_leading_staff_name(candidate_text):
    text = (candidate_text or "").strip()
    for length in (4, 3, 2):
        if len(text) < length:
            continue
        name = text[:length]
        if looks_like_staff_name(name) and is_likely_person_name(name):
            return name, text[length:]
    return None, None


def is_cjk_char(char):
    return "\u4e00" <= char <= "\u9fa5"


def is_name_only_text(text):
    normalized = clean_scalar(text)
    if normalized is None:
        return False
    normalized = normalized.strip(" /,，;；、")
    if not normalized:
        return False
    parts = [part for part in re.split(r"[/、]", normalized) if part]
    if not parts:
        return False
    if len(parts) == 1 and not re.fullmatch(r"[\u4e00-\u9fa5]{2,4}", parts[0]):
        return False
    return all(looks_like_staff_name(part) and is_likely_person_name(part) for part in parts)


def make_staff_aliaser():
    mapping = {}
    counters = defaultdict(int)

    def alias_for(name, role):
        key = (name, role)
        if key not in mapping:
            counters[role] += 1
            mapping[key] = f"{role}{counters[role]}"
        return mapping[key]

    return alias_for


def redact_staff_mentions(text, staff_aliaser):
    if staff_aliaser is None:
        return text
    pieces = []
    last_end = 0
    for match in STAFF_ROLE_PATTERN.finditer(text):
        role_end = match.end()
        right = role_end
        while right < len(text) and right - role_end < 4 and is_cjk_char(text[right]):
            right += 1
        candidate_text = text[role_end:right]
        name, trailing_text = split_leading_staff_name(candidate_text)
        if name is None:
            continue
        pieces.append(text[last_end:match.start()])
        pieces.append(staff_aliaser(name, match.group(1)))
        pieces.append(trailing_text)
        last_end = right
    if pieces:
        pieces.append(text[last_end:])
        text = "".join(pieces)
    pieces = []
    last_end = 0
    for match in STAFF_ROLE_PATTERN.finditer(text):
        role_start = match.start()
        left = role_start
        while left > last_end and role_start - left < 8 and is_cjk_char(text[left - 1]):
            left -= 1
        candidate_text = text[left:role_start]
        leading_text, name = split_staff_context(candidate_text)
        if name is None:
            continue
        pieces.append(text[last_end:left])
        pieces.append(f"{leading_text}{staff_aliaser(name, match.group(1))}")
        last_end = match.end()
    if not pieces:
        return text
    pieces.append(text[last_end:])
    return "".join(pieces)


def redact_text(value, identifiers, staff_aliaser=None):
    value = clean_scalar(value)
    if value is None:
        return None
    text = str(value)
    for token in sorted(identifiers, key=len, reverse=True):
        if len(token.strip()) >= 2:
            text = text.replace(token, "[已脱敏]")
    if staff_aliaser is not None:
        text = redact_staff_mentions(text, staff_aliaser)
        text = re.sub(
            r"([\u4e00-\u9fa5]{2,3})主治上级医师",
            lambda match: staff_aliaser(match.group(1), "上级医师"),
            text,
        )
        text = re.sub(r"[\u4e00-\u9fa5]{2,3}副主主治医师(\d+)", r"副主任医师\1", text)
        text = re.sub(
            r"[\u4e00-\u9fa5]{2,3}((?:副主任医师|主任医师|主治医师|主诊医师|上级医师|住院医师|专培医师)\d+)",
            r"\1",
            text,
        )
        text = re.sub(r"[\u4e00-\u9fa5]{2,3}(主治|主任|副主|上级)(?=\.\.\.)", r"\1", text)
        text = re.sub(
            r"[\u4e00-\u9fa5]{2,3}((?:副主任医师|主任医师|主治医师|主诊医师|住院医师|专培医师)\d+)(?:主任医师|主治医师|主诊医师|住院医师|专培医师|治医师|诊医师|培医师|医师)",
            r"\1",
            text,
        )
    text = re.sub(r"姓名[:：]?\s*\S+", "", text)
    text = re.sub(r"现住址[:：]?\s*\S+", "", text)
    text = re.sub(r"地址[:：]?\s*\S+", "", text)
    text = re.sub(r"联系方式[:：]?\s*\S+", "", text)
    text = re.sub(r"联系电话[:：]?\s*\S+", "", text)
    text = re.sub(r"住院号/门诊号[:：]?\s*\S+", "", text)
    text = re.sub(r"住院号[:：]?\s*\S+", "", text)
    text = re.sub(r"门诊号[:：]?\s*\S+", "", text)
    text = re.sub(r"唯一号[:：]?\s*\S+", "", text)
    text = re.sub(r"病案号[:：]?\s*\S+", "", text)
    text = re.sub(r"身份证(?:号码|号)?[:：]?\s*\S+", "", text)
    text = re.sub(r"职业[:：]?\s*\S+", "", text)
    text = re.sub(r"民族[:：]?\s*\S+", "", text)
    text = re.sub(r"婚姻(?:状态|状况)?[:：]?\s*\S+", "", text)
    text = re.sub(r"国籍[:：]?\s*\S+", "", text)
    text = re.sub(r"籍贯(?:省（自治区、直辖市）)?[:：]?\s*\S+", "", text)
    text = re.sub(r"入院途径[:：]?\s*\S+", "", text)
    text = re.sub(r"生于[^，。；;\n]{2,30}[，,]?\s*", "", text)
    text = re.sub(r"(?:祖籍|籍贯|户籍|家住|来自|居住于)[^，。；;\n]{2,30}[，,]?\s*", "", text)
    text = re.sub(r"久居[^，。；;\n]{0,30}[，,]?\s*", "", text)
    text = re.sub(r"(?:\d+\.)?\s*[^。；;\n]{0,30}门诊(?:时间)?[:：][^。；;\n]*(?:[。；;]|$)", "", text)
    text = re.sub(r"(?:[\u4e00-\u9fa5A-Za-z0-9]+)?门诊(?:时间)?\s*周[一二三四五六日天][^。；;\n]*(?:[。；;]|$)", "", text)
    text = re.sub(r"科室电话[:：]?\s*[^。；;\n]{3,30}", "", text)
    text = re.sub(r"(?:住院|医院|经治|主管)?医师签名[:：]?\s*", "", text)
    text = re.sub(r"(?<!\d)(1[3-9]\d{9})(?!\d)", "", text)
    text = re.sub(r"(?<!\d)(\d{15,18}[0-9Xx]?)(?!\d)", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if is_name_only_text(text):
        return None
    return text or None


def deidentify(value, identifiers, staff_aliaser=None):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in DIRECT_IDENTIFIER_LABELS or key in SENSITIVE_DEMOGRAPHIC_LABELS:
                continue
            result[str(key)] = deidentify(item, identifiers, staff_aliaser)
        return result
    if isinstance(value, list):
        return [deidentify(item, identifiers, staff_aliaser) for item in value]
    if isinstance(value, str):
        return redact_text(value, identifiers, staff_aliaser)
    return value


def is_empty(value):
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def first_record(data, key):
    value = data.get(key)
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and value:
        return value[0]
    return {}


def list_records(data, key):
    value = data.get(key)
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []


def get_value(record, *candidates):
    normalized = {normalize_key(k): v for k, v in record.items()}
    for candidate in candidates:
        value = normalized.get(normalize_key(candidate))
        value = clean_scalar(value)
        if value is not None:
            return value
    return None


def collect_pattern_values(record, pattern):
    rows = []
    for key, value in record.items():
        if re.fullmatch(pattern, normalize_key(key)):
            value = clean_scalar(value)
            if value is not None:
                rows.append((key, value))
    return rows


def parse_date(value):
    value = clean_scalar(value)
    if value is None:
        return None
    text = str(value).replace("/", "-")
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y%m%d"):
        try:
            return datetime.strptime(text[: len(fmt)], fmt)
        except ValueError:
            continue
    return None


def truncate(text, limit=160):
    text = clean_scalar(text)
    if text is None:
        return None
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def normalize_text_block(text):
    text = clean_scalar(text)
    if text is None:
        return None
    return re.sub(r"\s+", " ", text)


def dedupe_texts(values):
    results = []
    seen = set()
    for value in values:
        value = clean_scalar(value)
        if value is None:
            continue
        normalized = normalize_text_block(value)
        if normalized is None:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        results.append(normalized)
    return results


def clean_section_text(text):
    text = normalize_text_block(text)
    if text is None:
        return None
    text = re.sub(r"^[一二三四五六七八九十\d]+[、.．:：]\s*", "", text)
    text = re.sub(r"^[(（]?[一二三四五六七八九十\d]+[)）]\s*", "", text)
    text = re.sub(r"(?:住院|医院|经治|主管)?医师签名[:：]?\s*$", "", text)
    text = re.sub(r"\s*[一二三四五六七八九十\d]+[、.．]\s*$", "", text)
    return text.strip(" ：:;；，,")


def split_sentences(text):
    text = normalize_text_block(text)
    if text is None:
        return []
    parts = re.split(r"(?<=[。；;！？!?])\s*", text)
    return [part.strip() for part in parts if clean_scalar(part)]


def split_clauses(text):
    text = normalize_text_block(text)
    if text is None:
        return []
    parts = re.split(r"[，,、；;]\s*", text)
    return [part.strip() for part in parts if clean_scalar(part)]


def compact_text_items(values, max_items=4, max_chars=240):
    items = dedupe_texts(values)
    if not items:
        return None
    picked = []
    total = 0
    for item in items:
        candidate_len = len(item) + (1 if picked else 0)
        if picked and (len(picked) >= max_items or total + candidate_len > max_chars):
            break
        if not picked and len(item) > max_chars:
            return item[: max_chars - 3] + "..."
        picked.append(item)
        total += candidate_len
    return "；".join(picked) if picked else None


def summarize_present_history(text):
    selected = select_sentences(text, include_keywords=HISTORY_SYMPTOM_KEYWORDS, exclude_test_details=True)
    if not selected:
        selected = select_sentences(text, exclude_test_details=True)[:3]
    if not selected:
        clauses = split_clauses(text)
        prioritized = [clause for clause in clauses if any(keyword in clause for keyword in HISTORY_SYMPTOM_KEYWORDS)]
        selected = prioritized[:3] or clauses[:3]
    if len(selected) == 1:
        clauses = split_clauses(selected[0])
        if len(clauses) >= 2:
            prioritized = [clause for clause in clauses if any(keyword in clause for keyword in HISTORY_SYMPTOM_KEYWORDS)]
            selected = prioritized[:3] or clauses[:3]
    return compact_text_items(selected, max_items=3, max_chars=220)


def summarize_first_course_characteristics(text):
    selected = []
    for sentence in select_sentences(text, exclude_test_details=True):
        sentence = re.sub(r"^\[[^\]]+\]", "", sentence).strip()
        if any(keyword in sentence for keyword in ["辅助检查", "肺功能", "胸部CT", "血气", "FEV", "FVC", "DLCO", "影像"]):
            continue
        selected.append(sentence)
    if not selected:
        selected = select_sentences(text, exclude_test_details=True)[:3]
    return compact_text_items(selected, max_items=3, max_chars=220)


def summarize_outcome_text(discharge_sections):
    return compact_text_items(
        [
            discharge_sections.get("出院情况"),
            compact_text_items(
                select_sentences(discharge_sections.get("诊疗经过"), include_keywords=RESULT_SUMMARY_KEYWORDS, exclude_test_details=True),
                max_items=2,
                max_chars=160,
            ),
        ],
        max_items=2,
        max_chars=180,
    )


def normalize_physical_exam(text):
    text = normalize_text_block(text)
    if text is None:
        return None
    text = re.split(r"辅助检查[:：]", text, maxsplit=1)[0]
    return text.strip()


def is_non_surgical_item(name):
    name = clean_scalar(name) or ""
    if not name:
        return False
    if any(keyword in name for keyword in NON_SURGICAL_PROCEDURE_KEYWORDS):
        return True
    return False


def strip_record_prefix(text, record_name=None):
    text = normalize_text_block(text)
    if text is None:
        return None
    text = re.sub(r"^\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}(?::\d{2})?)?\s*", "", text)
    name = clean_scalar(record_name)
    if name:
        text = re.sub(r"^\s*" + re.escape(name) + r"\s*", "", text, count=1)
    return text.strip()


def parse_labeled_sections(text, labels):
    text = normalize_text_block(text)
    if text is None:
        return {}
    positions = []
    for label in labels:
        matches = []
        for marker in (f"{label}：", f"{label}:"):
            idx = text.find(marker)
            if idx >= 0:
                matches.append((idx, len(marker)))
        if matches:
            idx, marker_len = min(matches, key=lambda item: item[0])
            positions.append((idx, label, marker_len))
    positions.sort(key=lambda item: item[0])
    sections = {}
    for index, (start, label, marker_len) in enumerate(positions):
        content_start = start + marker_len
        content_end = positions[index + 1][0] if index + 1 < len(positions) else len(text)
        content = clean_section_text(text[content_start:content_end])
        if content:
            sections[label] = content
    return sections


def extract_first_course_sections(data):
    for row in list_records(data, "病程记录"):
        record_name = get_value(row, "记录名称")
        if not record_name or "首次病程记录" not in record_name:
            continue
        text = strip_record_prefix(get_value(row, "病程记录"), record_name)
        sections = parse_labeled_sections(
            text,
            ["病例特点", "入院诊断", "诊断依据", "鉴别诊断", "诊疗计划"],
        )
        return {
            "日期": get_value(row, "日期"),
            "记录名称": record_name,
            "原文": text,
            "分块": sections,
        }
    return {}


def extract_discharge_sections(data):
    emr = first_record(data, "电子病历")
    text = normalize_text_block(get_value(emr, "出院记录"))
    return parse_labeled_sections(
        text,
        ["入院情况", "入院诊断", "诊疗经过", "出院情况", "出院诊断", "出院建议", "出院带药"],
    )


def is_structured_test_detail(sentence):
    sentence = clean_scalar(sentence)
    if sentence is None:
        return False
    detail_tokens = [
        "血常规",
        "CRP",
        "PCT",
        "D-二聚体",
        "肺功能",
        "血气",
        "尿",
        "肿瘤标记物",
        "免疫球蛋白",
        "FEV",
        "FVC",
        "DLCO",
        "mmHg",
        "mmol",
        "ng/ml",
        "uL",
        "HPF",
        "IU/ml",
        "↑",
        "↓",
    ]
    hits = sum(1 for token in detail_tokens if token in sentence)
    digit_count = len(re.findall(r"\d", sentence))
    return hits >= 2 and digit_count >= 4


def select_sentences(text, include_keywords=None, exclude_test_details=False, record_name=None):
    base_text = strip_record_prefix(text, record_name)
    selected = []
    for sentence in split_sentences(base_text):
        sentence = clean_section_text(sentence)
        if sentence is None:
            continue
        if include_keywords and not any(keyword in sentence for keyword in include_keywords):
            continue
        if exclude_test_details and is_structured_test_detail(sentence):
            continue
        selected.append(sentence)
    return dedupe_texts(selected)


def infer_treatment_event_type(record_name):
    name = clean_scalar(record_name) or ""
    if "首次病程记录" in name:
        return "入院评估"
    if "首次查房" in name:
        return "首次查房"
    if "常规查房" in name:
        return "治疗调整"
    if "日常病程记录" in name:
        return "病情随访"
    if "出院" in name:
        return "出院总结"
    return "病程记录"


def should_include_course_record(record_name):
    name = clean_scalar(record_name) or ""
    include_keywords = ["病程记录", "查房记录"]
    exclude_keywords = ["筛查表", "预约表", "同意书", "告知书", "评估表", "申请单", "知情同意"]
    if any(keyword in name for keyword in exclude_keywords):
        return False
    return any(keyword in name for keyword in include_keywords)


def extract_evidence_references(text, limit=3):
    sentences = select_sentences(text, include_keywords=EVIDENCE_REFERENCE_KEYWORDS, exclude_test_details=True)
    return [truncate(sentence, 120) for sentence in sentences[:limit]]


def build_course_event(row, first_course_sections=None):
    record_name = normalize_text_block(get_value(row, "记录名称"))
    text = get_value(row, "病程记录")
    if text is None and record_name is None:
        return None
    if not should_include_course_record(record_name):
        return None
    event = {
        "日期": get_value(row, "日期"),
        "记录名称": record_name,
        "事件类型": infer_treatment_event_type(record_name),
    }
    if record_name and "首次病程记录" in record_name and first_course_sections:
        sections = first_course_sections.get("分块") or {}
        event.update(
            {
                "过程摘要": compact_text_items(
                    [truncate(sections.get("病例特点"), 160), truncate(sections.get("诊疗计划"), 120)],
                    max_items=2,
                    max_chars=260,
                ),
                "诊断判断": compact_text_items(
                    [
                        f"入院诊断:{'、'.join(diagnosis_split(sections.get('入院诊断')))}" if sections.get("入院诊断") else None,
                        f"诊断依据:{truncate(sections.get('诊断依据'), 160)}" if sections.get("诊断依据") else None,
                        f"鉴别诊断:{truncate(sections.get('鉴别诊断'), 120)}" if sections.get("鉴别诊断") else None,
                    ],
                    max_items=3,
                    max_chars=320,
                ),
                "治疗动作": compact_text_items(
                    [truncate(sections.get("诊疗计划"), 180)],
                    max_items=1,
                    max_chars=220,
                ),
                "证据引用": extract_evidence_references(
                    join_nonempty([sections.get("病例特点"), sections.get("诊断依据")], sep="。")
                ),
            }
        )
    else:
        symptom_sentences = select_sentences(text, include_keywords=PROCESS_SUMMARY_KEYWORDS, exclude_test_details=True, record_name=record_name)
        if not symptom_sentences:
            symptom_sentences = select_sentences(text, exclude_test_details=True, record_name=record_name)[:2]
        event.update(
            {
                "过程摘要": compact_text_items(symptom_sentences, max_items=3, max_chars=220),
                "诊断判断": compact_text_items(
                    select_sentences(text, include_keywords=DIAGNOSIS_SUMMARY_KEYWORDS, exclude_test_details=True, record_name=record_name),
                    max_items=2,
                    max_chars=220,
                ),
                "治疗动作": compact_text_items(
                    select_sentences(text, include_keywords=TREATMENT_ACTION_KEYWORDS, exclude_test_details=True, record_name=record_name),
                    max_items=2,
                    max_chars=220,
                ),
                "证据引用": extract_evidence_references(text),
                "结果/转归": compact_text_items(
                    select_sentences(text, include_keywords=RESULT_SUMMARY_KEYWORDS, exclude_test_details=True, record_name=record_name),
                    max_items=2,
                    max_chars=180,
                ),
            }
        )
    return compact_structure(event)


def build_longitudinal_summary(record_events, discharge_sections, discharge_date=None):
    timeline = []
    for event in record_events:
        if not isinstance(event, dict):
            continue
        node = compact_structure(
            {
                "时间": event.get("日期"),
                "事件类型": event.get("事件类型"),
                "核心内容": event.get("过程摘要"),
                "诊断判断": event.get("诊断判断"),
                "治疗动作": event.get("治疗动作"),
                "证据引用": event.get("证据引用"),
                "结果/转归": event.get("结果/转归"),
            }
        )
        if node:
            timeline.append(node)
    discharge_node = compact_structure(
        {
            "时间": discharge_date,
            "事件类型": "出院总结",
            "核心内容": compact_text_items(
                [discharge_sections.get("出院情况"), discharge_sections.get("出院建议")],
                max_items=2,
                max_chars=240,
            ),
            "诊断判断": compact_text_items(
                [
                    f"出院诊断:{discharge_sections.get('出院诊断')}" if discharge_sections.get("出院诊断") else None,
                    compact_text_items(
                        select_sentences(discharge_sections.get("诊疗经过"), include_keywords=DIAGNOSIS_SUMMARY_KEYWORDS, exclude_test_details=True),
                        max_items=2,
                        max_chars=180,
                    ),
                ],
                max_items=2,
                max_chars=260,
            ),
            "治疗动作": compact_text_items(
                [
                    compact_text_items(
                        select_sentences(discharge_sections.get("诊疗经过"), include_keywords=TREATMENT_ACTION_KEYWORDS, exclude_test_details=True),
                        max_items=3,
                        max_chars=180,
                    ),
                    discharge_sections.get("出院带药"),
                ],
                max_items=2,
                max_chars=260,
            ),
            "证据引用": extract_evidence_references(discharge_sections.get("诊疗经过")),
            "结果/转归": compact_text_items(
                [
                    discharge_sections.get("出院情况"),
                    compact_text_items(
                        select_sentences(discharge_sections.get("诊疗经过"), include_keywords=RESULT_SUMMARY_KEYWORDS, exclude_test_details=True),
                        max_items=2,
                        max_chars=180,
                    ),
                ],
                max_items=2,
                max_chars=240,
            ),
        }
    )
    if discharge_node:
        timeline.append(discharge_node)
    return timeline


def join_nonempty(parts, sep="；"):
    values = [str(part) for part in parts if not is_empty(part)]
    return sep.join(values) if values else None


def diagnosis_split(text):
    text = clean_scalar(text)
    if text is None:
        return []
    text = text.replace("初步诊断:", "").replace("初步诊断：", "")
    tokens = re.split(r"[\n；;、]|(?<=\S)\s{2,}", text)
    results = []
    for token in tokens:
        token = clean_scalar(token)
        if token and token not in results:
            results.append(token)
    return results


def latest_nursing_values(nursing_rows):
    latest = {}
    for row in nursing_rows:
        name = clean_scalar(row.get("项目名称"))
        if name is None:
            continue
        record_time = clean_scalar(row.get("记录时间"))
        numeric_value = clean_scalar(row.get("项目值(数值)"))
        text_value = clean_scalar(row.get("项目值(字符)"))
        value = numeric_value if numeric_value is not None else text_value
        if value is None:
            continue
        payload = {
            "记录时间": record_time,
            "数值": value,
            "单位": clean_scalar(row.get("测量单位")),
        }
        previous = latest.get(name)
        if previous is None or str(record_time or "") >= str(previous.get("记录时间") or ""):
            latest[name] = payload
    return latest


def classify_medication(name):
    name = clean_scalar(name) or ""
    categories = {
        "支气管扩张剂": ["沙丁胺醇", "异丙托溴铵", "噻托溴铵", "特布他林", "复方异丙托溴铵", "沙美特罗", "福莫特罗"],
        "糖皮质激素": ["布地奈德", "甲泼尼龙", "泼尼松", "地塞米松", "氢化可的松"],
        "抗生素": ["阿奇霉素", "头孢", "左氧氟沙星", "莫西沙星", "哌拉西林", "美罗培南"],
        "祛痰平喘": ["氨溴索", "乙酰半胱氨酸", "茶碱", "多索茶碱"],
        "抗纤维化": ["尼达尼布", "吡非尼酮"],
        "降糖药": ["阿卡波糖", "二甲双胍", "胰岛素", "达格列净"],
        "降压药": ["替米沙坦", "氨氯地平", "倍他乐克", "缬沙坦"],
    }
    for category, keywords in categories.items():
        if any(keyword in name for keyword in keywords):
            return category
    return "其他"


def is_non_zero_dose(value):
    value = clean_scalar(value)
    if value is None:
        return False
    text = str(value).strip()
    try:
        return float(text) != 0
    except ValueError:
        return text not in {"0", "0.0", "0.00"}


def is_medication_order(name, route=None, dose=None, unit=None):
    name = clean_scalar(name) or ""
    route = clean_scalar(route) or ""
    unit = clean_scalar(unit) or ""
    exclude_keywords = [
        "护理", "饮食", "检查", "检验", "检测", "CT", "MR", "MRI", "超声", "彩超", "病理", "血常规",
        "尿常规", "便常规", "血气", "肺功能", "心电图", "会诊", "测", "监测", "康复", "雾化器",
        "病危", "病重", "留观", "巡视", "评估", "吸氧", "导管", "插管", "复查", "筛查", "护理常规",
        "床旁", "标本", "宣教", "营养", "血糖", "禁食", "灌肠", "导尿", "预约",
    ]
    include_keywords = [
        "片", "胶囊", "注射液", "注射用", "颗粒", "滴丸", "气雾剂", "吸入剂", "混悬液", "口服液",
        "针", "药", "酊", "膏", "乳膏", "贴", "栓", "丸", "雾化吸入", "滴眼液", "滴鼻液",
        "胰岛素", "沙丁胺醇", "异丙托溴铵", "布地奈德", "甲泼尼龙", "泼尼松", "地塞米松",
        "莫西沙星", "左氧氟沙星", "头孢", "阿奇霉素", "氨溴索", "乙酰半胱氨酸", "茶碱",
        "尼达尼布", "吡非尼酮", "替米沙坦", "氨氯地平", "阿卡波糖", "二甲双胍",
    ]
    route_keywords = ["口服", "静滴", "静注", "注射", "皮下注射", "雾化", "吸入", "外用", "滴眼", "滴鼻", "泵入"]
    unit_keywords = ["mg", "g", "ml", "ug", "μg", "IU", "U", "片", "粒", "支", "袋", "吸", "喷", "丸", "揿", "瓶"]

    if any(keyword in name for keyword in exclude_keywords):
        return False
    if any(keyword in name for keyword in include_keywords):
        return True
    if any(keyword in route for keyword in route_keywords):
        return True
    if any(keyword.lower() in str(unit).lower() for keyword in unit_keywords) and is_non_zero_dose(dose):
        return True
    if classify_medication(name) != "其他":
        return True
    return False


def extract_scores(texts):
    patterns = [
        r"(Barthel|mMRC|CAT|ADL|APACHE|BODE|NRS2002|MNA)[^\d]{0,10}(\d+(?:\.\d+)?)",
        r"(6分钟步行|6MWT)[^\d]{0,10}(\d+(?:\.\d+)?)",
    ]
    matches = []
    for text in texts:
        raw = clean_scalar(text)
        if raw is None:
            continue
        for pattern in patterns:
            for metric, score in re.findall(pattern, raw, flags=re.IGNORECASE):
                matches.append({"评分项目": metric, "评分值": score, "来源片段": truncate(raw, 120)})
    deduped = []
    seen = set()
    for item in matches:
        key = (item["评分项目"], item["评分值"], item["来源片段"])
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def extract_stage_info(texts):
    joined = " ".join([text for text in texts if clean_scalar(text)])
    stage_matches = re.findall(r"(?:GOLD\s*|慢性阻塞性肺疾病(?:急性加重)?)([IVⅠⅡⅢⅣ1-4]+(?:级)?)", joined, flags=re.IGNORECASE)
    severity_matches = re.findall(r"(轻度|中度|重度|极重度)[^\n，。；;]{0,16}(?:阻塞性通气障碍|弥散功能障碍|慢阻肺)", joined)
    basis_fragments = []
    for pattern in [r"FEV1[^。\n；;]{0,60}", r"FVC[^。\n；;]{0,60}", r"阻塞性通气障碍[^。\n；;]{0,40}", r"弥散功能障碍[^。\n；;]{0,40}", r"急性加重[^。\n；;]{0,40}"]:
        basis_fragments.extend(re.findall(pattern, joined, flags=re.IGNORECASE))
    return {
        "疾病分期": stage_matches[0] if stage_matches else None,
        "严重程度分级": severity_matches[0] if severity_matches else None,
        "分期依据": basis_fragments[:8],
    }


def build_patient_meta(data, file_path):
    basic = first_record(data, "基本数据")
    home = first_record(data, "病案首页")
    return {
        "病例编号": build_case_id(file_path),
        "性别": get_value(basic, "性别") or get_value(home, "性别"),
        "年龄": get_value(basic, "年龄", "年龄（岁）") or get_value(home, "年龄", "年龄（岁）"),
        "入院时间": get_value(basic, "入院时间") or get_value(home, "入院时间", "入院时间.1"),
        "出院时间": get_value(basic, "出院时间") or get_value(home, "出院时间", "出院时间.1"),
        "主诊断": get_value(home, "出院主要诊断名称"),
    }


def normalize_surgery_item(item):
    if not isinstance(item, dict):
        return None
    item = compact_structure(item)
    if not item:
        return None
    name = clean_scalar(
        item.get("手术名称")
        or item.get("术式名称")
        or item.get("手术操作名称")
        or item.get("操作名称")
    )
    if is_non_surgical_item(name):
        return None
    date = clean_scalar(item.get("手术日期") or item.get("手术时间") or item.get("日期"))
    level = clean_scalar(item.get("手术级别"))
    anesthesia = clean_scalar(item.get("麻醉方式"))
    if name is None and date is None and level is None and anesthesia is None:
        return None
    result = {}
    if name is not None:
        result["手术名称"] = name
    if date is not None:
        result["手术日期"] = date
    if level is not None:
        result["手术级别"] = level
    if anesthesia is not None:
        result["麻醉方式"] = anesthesia
    return result or None


def pick_present_keys(item, keys):
    return {key: item[key] for key in keys if key in item and not is_empty(item[key])}


def extract_demographics(data):
    basic = first_record(data, "基本数据")
    home = first_record(data, "病案首页")
    payload = {
        "性别": get_value(basic, "性别") or get_value(home, "性别"),
        "年龄": get_value(basic, "年龄", "年龄（岁）") or get_value(home, "年龄", "年龄（岁）"),
        "入院时间": get_value(basic, "入院时间") or get_value(home, "入院时间", "入院时间.1"),
        "出院时间": get_value(basic, "出院时间") or get_value(home, "出院时间", "出院时间.1"),
        "入院科室": get_value(home, "入院科别") or get_value(basic, "就诊科室名"),
        "出院科室": get_value(home, "出院科别"),
    }
    summary = join_nonempty(
        [
            payload.get("性别"),
            f"{payload.get('年龄')}岁" if payload.get("年龄") else None,
            f"入院{payload.get('入院时间')}" if payload.get("入院时间") else None,
            f"出院{payload.get('出院时间')}" if payload.get("出院时间") else None,
        ]
    ) or GROUP_PLACEHOLDERS["患者人口学基础信息"]
    return payload, summary


def extract_history_info(data):
    emr = first_record(data, "电子病历")
    first_course = extract_first_course_sections(data)
    first_course_sections = first_course.get("分块") or {}
    payload = {
        "主诉": get_value(emr, "主诉"),
        "现病史": get_value(emr, "现病史"),
        "既往史": get_value(emr, "既往史"),
        "个人史": get_value(emr, "个人史"),
        "家族史": get_value(emr, "家族史"),
        "过敏史": get_value(emr, "过敏史"),
        "婚育史": get_value(emr, "婚育史"),
        "首程病例特点": summarize_first_course_characteristics(first_course_sections.get("病例特点")),
    }
    summary = join_nonempty(
        [
            f"主诉:{truncate(payload.get('主诉'), 40)}" if payload.get("主诉") else None,
            f"现病史:{truncate(payload.get('现病史'), 60)}" if payload.get("现病史") else None,
            f"既往史:{truncate(payload.get('既往史'), 60)}" if payload.get("既往史") else None,
            f"个人史:{truncate(payload.get('个人史'), 60)}" if payload.get("个人史") else None,
            f"首程病例特点:{truncate(payload.get('首程病例特点'), 60)}" if payload.get("首程病例特点") else None,
        ]
    ) or GROUP_PLACEHOLDERS["全维度病史信息"]
    return payload, summary


def extract_symptoms_signs(data):
    emr = first_record(data, "电子病历")
    nursing = latest_nursing_values(list_records(data, "护理记录"))
    payload = {
        "主要症状": get_value(emr, "主诉"),
        "现病史摘要": summarize_present_history(get_value(emr, "现病史")),
        "体格检查": normalize_physical_exam(get_value(emr, "体格检查")),
        "生命体征": {
            key: nursing.get(key)
            for key in ["体温", "脉搏", "呼吸", "血压", "血氧饱和度", "SPO2"]
            if nursing.get(key)
        },
    }
    summary = join_nonempty(
        [
            f"症状:{truncate(payload.get('主要症状'), 40)}" if payload.get("主要症状") else None,
            f"体检:{truncate(payload.get('体格检查'), 60)}" if payload.get("体格检查") else None,
            f"生命体征项数:{len(payload['生命体征'])}" if payload["生命体征"] else None,
        ]
    ) or GROUP_PLACEHOLDERS["症状体征信息"]
    return payload, summary


def extract_tests_exams(data):
    labs = []
    for row in list_records(data, "实验室检查"):
        indicator = get_value(row, "检验指标名")
        value = get_value(row, "检验指标值")
        if indicator is None and value is None:
            continue
        labs.append(
            {
                "报告日期": get_value(row, "报告日期", "检查时间", "接收日期"),
                "检验项目": get_value(row, "检验项目"),
                "检验指标名": indicator,
                "检验指标值": value,
                "单位": get_value(row, "单位"),
                "参考范围": get_value(row, "参考范围"),
            }
        )
    exams = []
    pulm_tests = []
    for row in list_records(data, "影像学检查"):
        item = {
            "检查时间": get_value(row, "检查时间"),
            "检查项目": get_value(row, "检查项目"),
            "检查部位": get_value(row, "检查部位"),
            "影像所见": get_value(row, "影像所见"),
            "检查结论": get_value(row, "检查结论"),
        }
        item = compact_structure(item)
        if item:
            exams.append(item)
        joined = " ".join([str(v) for v in item.values() if not is_empty(v)]) if item else ""
        if any(keyword in joined for keyword in ["肺功能", "通气", "弥散", "FEV", "FVC"]):
            pulm_tests.append(item)
    payload = {
        "检验项目列表": labs,
        "影像检查列表": exams,
        "其他检查列表": pulm_tests,
    }
    summary = join_nonempty(
        [
            f"实验室检查{len(labs)}项" if labs else None,
            f"影像检查{len(exams)}项" if exams else None,
            f"肺功能相关检查{len(pulm_tests)}项" if pulm_tests else None,
        ]
    ) or GROUP_PLACEHOLDERS["检查检验信息"]
    return payload, summary


def extract_diagnosis_info(data):
    emr = first_record(data, "电子病历")
    home = first_record(data, "病案首页")
    first_course = extract_first_course_sections(data)
    first_course_sections = first_course.get("分块") or {}
    discharge_sections = extract_discharge_sections(data)
    other_diagnoses = [value for _, value in collect_pattern_values(home, r"出院其他诊断名称\d+")]
    icd_codes = []
    for key, value in home.items():
        normalized = normalize_key(key)
        if normalized.startswith("出院其他诊断编码") or normalized == "出院主要诊断编码":
            value = clean_scalar(value)
            if value is not None:
                icd_codes.append(value)
    pathology = [value for _, value in collect_pattern_values(home, r"病理诊断名称\d*")]
    payload = {
        "入院诊断": diagnosis_split(get_value(emr, "入院诊断")),
        "出院诊断": diagnosis_split(get_value(emr, "出院诊断")),
        "主要诊断": get_value(home, "出院主要诊断名称"),
        "其他诊断": other_diagnoses,
        "ICD编码列表": icd_codes,
        "病理诊断": pathology,
        "诊断依据": compact_text_items(
            [
                first_course_sections.get("诊断依据"),
                compact_text_items(
                    select_sentences(discharge_sections.get("诊疗经过"), include_keywords=DIAGNOSIS_SUMMARY_KEYWORDS, exclude_test_details=True),
                    max_items=2,
                    max_chars=180,
                ),
            ],
            max_items=2,
            max_chars=260,
        ),
        "首程诊断推理": {
            "病例特点": summarize_first_course_characteristics(first_course_sections.get("病例特点")),
            "入院诊断": diagnosis_split(first_course_sections.get("入院诊断")),
            "诊断依据": first_course_sections.get("诊断依据"),
            "鉴别诊断": truncate(first_course_sections.get("鉴别诊断"), 180),
        },
    }
    summary = join_nonempty(
        [
            f"主诊断:{payload.get('主要诊断')}" if payload.get("主要诊断") else None,
            f"出院诊断:{'、'.join(payload.get('出院诊断')[:6])}" if payload.get("出院诊断") else None,
            f"其他诊断{len(other_diagnoses)}项" if other_diagnoses else None,
        ]
    ) or GROUP_PLACEHOLDERS["诊断信息"]
    return payload, summary


def extract_medication_info(data):
    orders = []
    for row in list_records(data, "医嘱"):
        name = get_value(row, "医嘱名称")
        if name is None:
            continue
        route = get_value(row, "用法")
        dose = get_value(row, "剂量")
        unit = get_value(row, "单位")
        if not is_medication_order(name, route=route, dose=dose, unit=unit):
            continue
        item = {
            "开始时间": get_value(row, "开始时间"),
            "结束时间": get_value(row, "结束时间"),
            "医嘱名称": name,
            "频次": get_value(row, "频次"),
            "用法": route,
            "剂量": dose,
            "单位": unit,
            "药物分类": classify_medication(name),
        }
        item = compact_structure(item)
        if item:
            orders.append(item)
    names = []
    for item in orders:
        name = item["医嘱名称"]
        if name not in names:
            names.append(name)
    payload = {
        "用药医嘱列表": orders,
        "用药用法": [pick_present_keys(item, ["医嘱名称", "频次", "用法", "剂量", "单位"]) for item in orders],
        "用药疗程": [pick_present_keys(item, ["医嘱名称", "开始时间", "结束时间"]) for item in orders],
    }
    summary = join_nonempty(
        [
            f"医嘱{len(orders)}条" if orders else None,
            f"核心医嘱:{'、'.join(names[:10])}" if names else None,
        ]
    ) or GROUP_PLACEHOLDERS["药物治疗信息"]
    return payload, summary


def extract_surgery_info(data):
    home = first_record(data, "病案首页")
    surgeries = []
    for row in list_records(data, "手术记录"):
        item = normalize_surgery_item(row)
        if item:
            surgeries.append(item)
    for index in range(1, 41):
        name = get_value(home, f"其他手术操作名称{index}")
        date = get_value(home, f"其他手术操作日期{index}")
        item = normalize_surgery_item({"手术名称": name, "手术日期": date})
        if item:
            surgeries.append(item)
    main_name = get_value(home, "主要手术操作名称")
    main_item = normalize_surgery_item(
        {
            "手术名称": main_name,
            "手术日期": get_value(home, "主要手术操作日期"),
            "手术级别": get_value(home, "主要手术操作级别"),
            "麻醉方式": get_value(home, "主要手术操作麻醉方式"),
        }
    )
    if main_item:
        surgeries.insert(0, main_item)
    deduped_surgeries = []
    seen = set()
    for item in surgeries:
        sig = (
            clean_scalar(item.get("手术名称")) if isinstance(item, dict) else None,
            clean_scalar(item.get("手术日期")) if isinstance(item, dict) else None,
            clean_scalar(item.get("麻醉方式")) if isinstance(item, dict) else None,
        )
        if sig in seen:
            continue
        seen.add(sig)
        deduped_surgeries.append(item)
    surgeries = deduped_surgeries
    payload = {
        "手术列表": surgeries,
        "手术详情": surgeries,
        "麻醉记录": [
            {
                "手术名称": item.get("手术名称"),
                "麻醉方式": item.get("麻醉方式"),
            }
            for item in surgeries
            if isinstance(item, dict) and item.get("麻醉方式")
        ],
    }
    summary = (
        f"手术{len(surgeries)}项：{'、'.join([str(item.get('手术名称')) for item in surgeries[:5] if isinstance(item, dict) and item.get('手术名称')])}"
        if surgeries
        else GROUP_PLACEHOLDERS["手术治疗信息"]
    )
    return payload, summary


def extract_treatment_process_info(data):
    first_course = extract_first_course_sections(data)
    discharge_sections = extract_discharge_sections(data)
    basic = first_record(data, "基本数据")
    home = first_record(data, "病案首页")
    records = []
    for row in list_records(data, "病程记录"):
        item = build_course_event(row, first_course_sections=first_course)
        if item:
            records.append(item)
    discharge_date = get_value(basic, "出院时间") or get_value(home, "出院时间", "出院时间.1")
    longitudinal_summary = build_longitudinal_summary(records, discharge_sections, discharge_date=discharge_date)
    payload = {
        "治疗原则": compact_text_items(
            [
                compact_text_items(
                    select_sentences(discharge_sections.get("诊疗经过"), include_keywords=TREATMENT_ACTION_KEYWORDS, exclude_test_details=True),
                    max_items=3,
                    max_chars=200,
                ),
                (first_course.get("分块") or {}).get("诊疗计划"),
            ],
            max_items=2,
            max_chars=260,
        ),
        "诊疗过程描述": records,
        "全过程诊疗时序总结": longitudinal_summary,
        "治疗效果": summarize_outcome_text(discharge_sections),
    }
    record_names = [item["记录名称"] for item in records if item.get("记录名称")]
    summary = join_nonempty(
        [
            f"病程记录{len(records)}条" if records else None,
            f"记录类型:{'、'.join(record_names[:8])}" if record_names else None,
            f"时序节点{len(longitudinal_summary)}项" if longitudinal_summary else None,
        ]
    ) or GROUP_PLACEHOLDERS["诊疗过程信息"]
    return payload, summary


def extract_discharge_info(data):
    basic = first_record(data, "基本数据")
    emr = first_record(data, "电子病历")
    home = first_record(data, "病案首页")
    discharge_sections = extract_discharge_sections(data)
    discharge_text = get_value(emr, "出院诊断") or get_value(home, "出院主要诊断名称")
    payload = {
        "出院日期": get_value(basic, "出院时间") or get_value(home, "出院时间", "出院时间.1"),
        "出院症状": summarize_outcome_text(discharge_sections),
        "出院医嘱": compact_text_items(
            [
                discharge_sections.get("出院建议"),
                discharge_sections.get("出院带药"),
                get_value(emr, "出院带药") or get_value(emr, "出院医嘱"),
            ],
            max_items=2,
            max_chars=240,
        ),
        "出院结局": get_value(home, "主要诊断出院情况") or get_value(home, "离院方式"),
        "离院方式": get_value(home, "离院方式"),
        "出院诊断": discharge_text,
        "出院诊断列表": diagnosis_split(discharge_text) or [get_value(home, "出院主要诊断名称")],
    }
    summary = join_nonempty(
        [
            f"出院日期:{payload.get('出院日期')}" if payload.get("出院日期") else None,
            f"出院结局:{payload.get('出院结局')}" if payload.get("出院结局") else None,
            f"出院诊断:{truncate(payload.get('出院诊断'), 80)}" if payload.get("出院诊断") else None,
        ]
    ) or GROUP_PLACEHOLDERS["出院信息"]
    return payload, summary


def extract_complications_info(data):
    diagnosis_payload, _ = extract_diagnosis_info(data)
    main_diagnosis = diagnosis_payload.get("主要诊断")
    diagnosis_pool = diagnosis_payload.get("出院诊断") or []
    diagnosis_pool.extend(diagnosis_payload.get("其他诊断") or [])
    complications = []
    for item in diagnosis_pool:
        item = clean_scalar(item)
        if item is None:
            continue
        if main_diagnosis and item == main_diagnosis:
            continue
        if "慢性阻塞性肺病" in item or "慢性阻塞性肺疾病" in item:
            continue
        if item not in complications:
            complications.append(item)
    severity = "重度" if any(keyword in " ".join(complications) for keyword in ["呼吸衰竭", "心力衰竭", "脓毒", "感染"]) else None
    payload = {
        "并发症列表": complications,
        "并发症数量": len(complications),
        "并发症分级": severity,
    }
    summary = join_nonempty(
        [
            f"并发症{len(complications)}项" if complications else None,
            f"并发症:{'、'.join(complications[:8])}" if complications else None,
        ]
    ) or GROUP_PLACEHOLDERS["并发症信息"]
    return payload, summary


def extract_function_score_info(data):
    emr = first_record(data, "电子病历")
    texts = [
        get_value(emr, "出院记录"),
        get_value(emr, "现病史"),
        *(row.get("病程记录") for row in list_records(data, "病程记录")),
    ]
    scores = extract_scores(texts)
    payload = {
        "入院功能评分": scores,
        "出院功能评分": scores,
        "评分变化": None,
    }
    if scores:
        score_parts = [f"{item['评分项目']}={item['评分值']}" for item in scores[:6]]
        summary = f"功能评分{len(scores)}项：{'、'.join(score_parts)}"
    else:
        summary = GROUP_PLACEHOLDERS["功能评分信息"]
    return payload, summary


def extract_prognosis_info(data):
    basic = first_record(data, "基本数据")
    home = first_record(data, "病案首页")
    admission = get_value(basic, "入院时间") or get_value(home, "入院时间", "入院时间.1")
    discharge = get_value(basic, "出院时间") or get_value(home, "出院时间", "出院时间.1")
    stay_days = get_value(home, "实际住院（天）")
    if stay_days is None:
        admission_dt = parse_date(admission)
        discharge_dt = parse_date(discharge)
        if admission_dt and discharge_dt:
            stay_days = (discharge_dt.date() - admission_dt.date()).days
    discharge_outcome = get_value(home, "主要诊断出院情况") or get_value(home, "离院方式")
    readmit = get_value(home, "是否有出院31日内再住院计划")
    payload = {
        "住院天数": stay_days,
        "出院结局": discharge_outcome,
        "再入院风险": readmit,
        "预后评估": get_value(home, "出院31天再住院计划目的"),
    }
    summary = join_nonempty(
        [
            f"住院天数:{stay_days}" if stay_days is not None else None,
            f"出院结局:{discharge_outcome}" if discharge_outcome else None,
            f"再住院计划:{readmit}" if readmit else None,
        ]
    ) or GROUP_PLACEHOLDERS["预后信息"]
    return payload, summary


def extract_severity_info(data):
    emr = first_record(data, "电子病历")
    home = first_record(data, "病案首页")
    texts = [
        get_value(emr, "入院诊断"),
        get_value(emr, "出院诊断"),
        get_value(emr, "出院记录"),
        get_value(home, "出院主要诊断名称"),
    ]
    for row in list_records(data, "影像学检查"):
        texts.append(get_value(row, "检查结论"))
        texts.append(get_value(row, "影像所见"))
    info = extract_stage_info(texts)
    payload = {
        "疾病分期": info["疾病分期"],
        "严重程度分级": info["严重程度分级"],
        "分期依据": info["分期依据"],
    }
    summary = join_nonempty(
        [
            f"疾病分期:{payload.get('疾病分期')}" if payload.get("疾病分期") else None,
            f"严重程度:{payload.get('严重程度分级')}" if payload.get("严重程度分级") else None,
            f"依据条目:{len(payload.get('分期依据') or [])}" if payload.get("分期依据") else None,
        ]
    ) or GROUP_PLACEHOLDERS["疾病分期与严重程度"]
    return payload, summary


GROUP_EXTRACTORS = {
    "患者人口学基础信息": extract_demographics,
    "全维度病史信息": extract_history_info,
    "症状体征信息": extract_symptoms_signs,
    "检查检验信息": extract_tests_exams,
    "诊断信息": extract_diagnosis_info,
    "药物治疗信息": extract_medication_info,
    "手术治疗信息": extract_surgery_info,
    "诊疗过程信息": extract_treatment_process_info,
    "出院信息": extract_discharge_info,
    "疾病分期与严重程度": extract_severity_info,
    "并发症信息": extract_complications_info,
    "功能评分信息": extract_function_score_info,
    "预后信息": extract_prognosis_info,
}


def load_rules(rule_path):
    rules = json.loads(rule_path.read_text(encoding="utf-8"))
    current_dimension = None
    grouped = {key: [] for key in RULE_FILE_MAP}
    for row in rules:
        dimension = clean_scalar(row.get("四维链"))
        if dimension:
            current_dimension = dimension
        if current_dimension not in grouped:
            continue
        grouped[current_dimension].append(
            {
                "字段分组": row.get("字段分组"),
                "核心字段": row.get("核心字段"),
                "数据处理逻辑": row.get("数据处理逻辑"),
                "相似度匹配核心依据": row.get("相似度匹配核心依据"),
            }
        )
    return grouped


def build_group_payload(group_name, rule, extracted_data, summary):
    return {
        "字段分组": group_name,
        "核心字段": rule.get("核心字段"),
        "数据处理逻辑": rule.get("数据处理逻辑"),
        "相似度匹配核心依据": rule.get("相似度匹配核心依据"),
        "提取结果": compact_structure(extracted_data) or {},
        "相似链条": summary or GROUP_PLACEHOLDERS.get(group_name) or "无相关信息",
    }


def build_case_dimension(data, file_path, dimension_rules):
    identifiers = collect_direct_identifiers(data)
    staff_aliaser = make_staff_aliaser()
    patient_meta = build_patient_meta(data, file_path)
    field_groups = []
    merged_parts = []
    for rule in dimension_rules:
        group_name = rule.get("字段分组")
        extractor = GROUP_EXTRACTORS.get(group_name)
        if extractor is None:
            extracted_data = {}
            summary = GROUP_PLACEHOLDERS.get(group_name, "无相关信息")
        else:
            extracted_data, summary = extractor(data)
        extracted_data = deidentify(extracted_data, identifiers, staff_aliaser)
        summary = redact_text(summary, identifiers, staff_aliaser)
        field_groups.append(build_group_payload(group_name, rule, extracted_data, summary))
        merged_parts.append(f"{group_name}: {summary}")
    return {
        "病例标识": compact_structure(deidentify(patient_meta, identifiers, staff_aliaser)) or {},
        "字段分组明细": field_groups,
        "合并相似链条": redact_text("\n".join(merged_parts), identifiers, staff_aliaser),
    }


def build_outputs(rule_json, input_dir, output_dir):
    grouped_rules = load_rules(rule_json)
    json_files = sorted(
        [path for path in input_dir.glob("*.json") if path.is_file() and not path.name.startswith("_")]
    )
    outputs = {dimension: [] for dimension in RULE_FILE_MAP}
    skipped_files = []
    for json_file in json_files:
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            skipped_files.append(
                {
                    "file_name": json_file.name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        for dimension, rules in grouped_rules.items():
            outputs[dimension].append(build_case_dimension(data, json_file, rules))
    output_dir.mkdir(parents=True, exist_ok=True)
    for dimension, file_name in RULE_FILE_MAP.items():
        payload = {
            "相似度维度": dimension,
            "来源目录": str(input_dir),
            "规则文件": str(rule_json),
            "病例数量": len(outputs[dimension]),
            "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "跳过文件数": len(skipped_files),
            "跳过文件": skipped_files,
            "病例相似链数据": outputs[dimension],
        }
        target = output_dir / file_name
        target.write_text(json.dumps(compact_structure(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="按照四维相似链规则批量构建 COPD 病例相似链 JSON。")
    parser.add_argument(
        "--rules",
        default=str(DEFAULT_RULES),
        help="规则 JSON 路径",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="待处理病例 JSON 文件夹",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="导出目录",
    )
    args = parser.parse_args()

    rule_json = Path(args.rules)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    build_outputs(rule_json, input_dir, output_dir)
    print(output_dir)
    for file_name in RULE_FILE_MAP.values():
        print(output_dir / file_name)


if __name__ == "__main__":
    main()
