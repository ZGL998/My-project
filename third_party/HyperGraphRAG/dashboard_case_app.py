import argparse
import ast
import hashlib
import json
import math
import mmap
import os
import re
import threading
import time
import urllib.error
import urllib.request
import unicodedata
import uuid
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import networkx as nx
import numpy as np

try:
    from xinference.client import Client
except Exception:
    Client = None

try:
    from resp_rules import get_default_diagnosis_mapper, get_default_rule_set
except Exception:
    get_default_diagnosis_mapper = None
    get_default_rule_set = None


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_HTML_FILE = REPO_ROOT / "templates" / "case_dashboard_v2.html"
XINFERENCE_ENDPOINT = os.environ.get("XINFERENCE_ENDPOINT", "http://127.0.0.1:9997")
XINFERENCE_EMBED_MODEL = os.environ.get("XINFERENCE_EMBED_MODEL", "Qwen3-Embedding-8B")
XINFERENCE_RERANK_MODEL = os.environ.get("XINFERENCE_RERANK_MODEL", "Qwen3-Reranker-0.6B")
XINFERENCE_LOADING_RETRY_ATTEMPTS = max(
    1, int(os.environ.get("XINFERENCE_LOADING_RETRY_ATTEMPTS", "4"))
)
XINFERENCE_LOADING_RETRY_SLEEP_SECONDS = max(
    0.0, float(os.environ.get("XINFERENCE_LOADING_RETRY_SLEEP_SECONDS", "3"))
)
XINFERENCE_RERANK_QUERY_MAX_CHARS = max(
    384, int(os.environ.get("XINFERENCE_RERANK_QUERY_MAX_CHARS", "1200"))
)
XINFERENCE_RERANK_QUERY_CHAIN_CHAR_BUDGET = max(
    96, int(os.environ.get("XINFERENCE_RERANK_QUERY_CHAIN_CHAR_BUDGET", "220"))
)
XINFERENCE_RERANK_DOC_MAX_CHARS = max(
    256, int(os.environ.get("XINFERENCE_RERANK_DOC_MAX_CHARS", "560"))
)
XINFERENCE_RERANK_DOC_CHAIN_CHAR_BUDGET = max(
    72, int(os.environ.get("XINFERENCE_RERANK_DOC_CHAIN_CHAR_BUDGET", "140"))
)
XINFERENCE_EMBED_TEXT_MAX_CHARS = max(
    256, int(os.environ.get("XINFERENCE_EMBED_TEXT_MAX_CHARS", "2800"))
)
RERANK_CANDIDATE_POOL_LIMIT = max(4, int(os.environ.get("RERANK_CANDIDATE_POOL_LIMIT", "8")))
RERANK_CANDIDATE_GAP_THRESHOLD = max(
    0.02, float(os.environ.get("RERANK_CANDIDATE_GAP_THRESHOLD", "0.12"))
)
QUERY_EMBED_REQUEST_TIMEOUT_SECONDS = max(
    5.0, float(os.environ.get("QUERY_EMBED_REQUEST_TIMEOUT_SECONDS", "8"))
)
QUERY_EMBED_MODE = str(os.environ.get("QUERY_EMBED_MODE", "summary") or "summary").strip().lower()
QUERY_EMBED_BATCH_SIZE = max(
    1, int(os.environ.get("QUERY_EMBED_BATCH_SIZE", "2"))
)
QUERY_EMBED_FAILURE_CACHE_TTL_SECONDS = max(
    30.0, float(os.environ.get("QUERY_EMBED_FAILURE_CACHE_TTL_SECONDS", "180"))
)
QUERY_EMBED_CACHE_MAX_ITEMS = max(
    32, int(os.environ.get("QUERY_EMBED_CACHE_MAX_ITEMS", "256"))
)
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_.+-]{2,}")
DEBUG_SELF_MATCH_CASE_IDS = {"CASE_A83DF3C908CC", "CASE_88F1919F35D2"}

# #region debug-point A:similar-cases-timeout-helper
def _debug_emit(hypothesis_id: str, location: str, msg: str, data: dict[str, Any] | None = None) -> None:
    session_env_path = REPO_ROOT / ".dbg" / "dashboard-request-stall.env"
    env_path = REPO_ROOT / ".dbg" / "self-match-score-gap.env"
    legacy_env_path = REPO_ROOT / ".dbg" / "similar-cases-timeout.env"
    server_url = "http://127.0.0.1:7777/event"
    session_id = "dashboard-request-stall"
    try:
        chosen_env_path = (
            session_env_path
            if session_env_path.exists()
            else (env_path if env_path.exists() else legacy_env_path)
        )
        if chosen_env_path.exists():
            env_content = chosen_env_path.read_text(encoding="utf-8", errors="ignore")
            for line in env_content.splitlines():
                if line.startswith("DEBUG_SERVER_URL="):
                    server_url = line.split("=", 1)[1].strip() or server_url
                elif line.startswith("DEBUG_SESSION_ID="):
                    session_id = line.split("=", 1)[1].strip() or session_id
        request = urllib.request.Request(
            server_url,
            data=json.dumps(
                {
                    "sessionId": session_id,
                    "runId": "pre-fix",
                    "hypothesisId": hypothesis_id,
                    "location": location,
                    "msg": f"[DEBUG] {msg}",
                    "data": data or {},
                    "ts": int(time.time() * 1000),
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request, timeout=1.5).read()
    except Exception:
        pass

# #endregion

QUERY_EMBED_CACHE: OrderedDict[str, dict[str, dict[str, list[float]]]] = OrderedDict()
QUERY_EMBED_FAILURE_CACHE: OrderedDict[str, float] = OrderedDict()
QUERY_EMBED_CACHE_LOCK = threading.Lock()


def _build_query_embed_cache_key(
    chain_queries: dict[str, str],
    core_chain_texts: dict[str, str],
) -> str:
    payload = {
        "model": XINFERENCE_EMBED_MODEL,
        "chains": {bucket: str(chain_queries.get(bucket, "")).strip() for bucket in CHAIN_LABELS},
        "core_chains": {bucket: str(core_chain_texts.get(bucket, "")).strip() for bucket in CHAIN_LABELS},
    }
    return hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _get_query_embed_cache(cache_key: str) -> dict[str, dict[str, list[float]]] | None:
    with QUERY_EMBED_CACHE_LOCK:
        cached = QUERY_EMBED_CACHE.get(cache_key)
        if cached is None:
            return None
        QUERY_EMBED_CACHE.move_to_end(cache_key)
        return {
            "query_vecs": {bucket: list(vector) for bucket, vector in (cached.get("query_vecs") or {}).items()},
            "query_core_vecs": {
                bucket: list(vector) for bucket, vector in (cached.get("query_core_vecs") or {}).items()
            },
        }


def _put_query_embed_cache(
    cache_key: str,
    *,
    query_vecs: dict[str, np.ndarray],
    query_core_vecs: dict[str, np.ndarray],
) -> None:
    with QUERY_EMBED_CACHE_LOCK:
        QUERY_EMBED_CACHE[cache_key] = {
            "query_vecs": {bucket: vector.astype(np.float32).tolist() for bucket, vector in query_vecs.items()},
            "query_core_vecs": {
                bucket: vector.astype(np.float32).tolist() for bucket, vector in query_core_vecs.items()
            },
        }
        QUERY_EMBED_CACHE.move_to_end(cache_key)
        QUERY_EMBED_FAILURE_CACHE.pop(cache_key, None)
        while len(QUERY_EMBED_CACHE) > QUERY_EMBED_CACHE_MAX_ITEMS:
            QUERY_EMBED_CACHE.popitem(last=False)


def _get_query_embed_failure(cache_key: str) -> bool:
    with QUERY_EMBED_CACHE_LOCK:
        failed_at = QUERY_EMBED_FAILURE_CACHE.get(cache_key)
        if failed_at is None:
            return False
        if (time.time() - failed_at) > QUERY_EMBED_FAILURE_CACHE_TTL_SECONDS:
            QUERY_EMBED_FAILURE_CACHE.pop(cache_key, None)
            return False
        QUERY_EMBED_FAILURE_CACHE.move_to_end(cache_key)
        return True


def _put_query_embed_failure(cache_key: str) -> None:
    with QUERY_EMBED_CACHE_LOCK:
        QUERY_EMBED_FAILURE_CACHE[cache_key] = time.time()
        QUERY_EMBED_FAILURE_CACHE.move_to_end(cache_key)
        while len(QUERY_EMBED_FAILURE_CACHE) > QUERY_EMBED_CACHE_MAX_ITEMS:
            QUERY_EMBED_FAILURE_CACHE.popitem(last=False)

CHAIN_MAP = {
    "人口学背景": "history_chain",
    "病史背景": "history_chain",
    "症状体征证据": "evidence_chain",
    "影像检查证据": "evidence_chain",
    "实验室检查证据": "evidence_chain",
    "诊断推理证据": "evidence_chain",
    "药物治疗方案": "care_chain",
    "全过程诊疗时序事件": "care_chain",
    "出院结局与随访": "care_chain",
    "出院带药方案": "care_chain",
    "疾病严重度评估": "severity_chain",
    "并发症负担": "severity_chain",
    "功能评分评估": "severity_chain",
    "预后风险评估": "severity_chain",
}

CHAIN_LABELS = {
    "history_chain": "病史链",
    "evidence_chain": "证据链",
    "care_chain": "诊疗链",
    "severity_chain": "严重度链",
}

RERANK_TEXT_NOISE_PATTERNS = (
    re.compile(r"\[(?:\d+)\]"),
    re.compile(r"(?:异常检验|正常检验|检验指标|影像检查|其他诊断|医嘱|手术)\d+项"),
    re.compile(r"(?:等|共)\d+(?:项|条|次|例)"),
    re.compile(r"(?:详见原文|见附件|内容略)"),
)

RERANK_CHAIN_KEYWORDS = {
    "history_chain": (
        "慢阻肺",
        "copd",
        "急性加重",
        "咳嗽",
        "咳痰",
        "喘",
        "胸闷",
        "气促",
        "发热",
    ),
    "evidence_chain": (
        "血气",
        "pao2",
        "paco2",
        "spo2",
        "fev1",
        "ct",
        "影像",
        "渗出",
        "感染",
        "crp",
        "pct",
        "白细胞",
    ),
    "care_chain": (
        "抗感染",
        "激素",
        "雾化",
        "支气管扩张剂",
        "无创",
        "有创",
        "吸氧",
        "通气",
        "机械通气",
    ),
    "severity_chain": (
        "呼吸衰竭",
        "低氧",
        "高碳酸",
        "酸中毒",
        "重症",
        "机械通气",
        "休克",
        "意识",
        "入住icu",
        "icu",
    ),
}

DATASET_CONFIGS = {
    "chronic": {
        "label": "慢性阻塞性肺病",
        "working_dir": REPO_ROOT / "expr" / "copd_chronic_custom_kg",
        "custom_kg": REPO_ROOT / "data" / "copd_custom_kg" / "copd_chronic_custom_kg.json",
    },
    "ae": {
        "label": "COPD 急性加重",
        "working_dir": REPO_ROOT / "expr" / "copd_ae_custom_kg",
        "custom_kg": REPO_ROOT / "data" / "copd_custom_kg" / "copd_ae_custom_kg.json",
    },
    "infection": {
        "label": "COPD 感染",
        "working_dir": REPO_ROOT / "expr" / "copd_infection_custom_kg",
        "custom_kg": REPO_ROOT / "data" / "copd_custom_kg" / "copd_infection_custom_kg.json",
    },
}
ADDITIONAL_DATASET_MANIFEST = REPO_ROOT / "data" / "non_copd_dataset_manifest.json"
TESTSET_ROOT = Path(os.environ.get('TESTSET_ROOT', '/path/to/testset'))
SIMILAR_FEEDBACK_DIR = REPO_ROOT / "validation_reports" / "similar_feedback"
TESTSET_IGNORE_DIRS = {"bysj", "__pycache__"}
TESTSET_FILE_PATTERNS = {
    "history_chain": ["history_background_similarity*.json*"],
    "evidence_chain": ["clinical_evidence_chain_similarity*.json*"],
    "care_chain": ["treatment_trajectory_similarity*.json*"],
    "severity_chain": ["clinical_severity_similarity*.json*"],
}


def report_debug_event(
    *,
    hypothesis_id: str,
    location: str,
    msg: str,
    data: dict[str, Any] | None = None,
    run_id: str = "pre-fix",
) -> None:
    env_path = REPO_ROOT / ".dbg" / "embedding-handle-error.env"
    server_url = "http://127.0.0.1:7777/event"
    session_id = "embedding-handle-error"
    try:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("DEBUG_SERVER_URL="):
                    server_url = line.split("=", 1)[1].strip() or server_url
                elif line.startswith("DEBUG_SESSION_ID="):
                    session_id = line.split("=", 1)[1].strip() or session_id
        payload = {
            "sessionId": session_id,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": msg,
            "data": data or {},
            "ts": int(time.time() * 1000),
        }
        urllib.request.urlopen(
            urllib.request.Request(
                server_url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ),
            timeout=1.5,
        ).read()
    except Exception:
        pass


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_resp_diagnosis_mapper():
    if get_default_diagnosis_mapper is None:
        raise RuntimeError("resp_rules diagnosis mapper is unavailable in the current environment.")
    return get_default_diagnosis_mapper("v1")


@lru_cache(maxsize=1)
def get_resp_rule_set():
    if get_default_rule_set is None:
        raise RuntimeError("resp_rules rule set is unavailable in the current environment.")
    return get_default_rule_set("v1")


def build_diagnosis_mapping_payload(diagnosis_text: str, max_matches: int = 3) -> dict[str, Any]:
    raw_text = str(diagnosis_text or "").strip()
    payload = {
        "raw_text": raw_text,
        "status": "empty" if not raw_text else "unmapped",
        "normalized_text": "",
        "icd10": "",
        "concept_id": "",
        "template": "",
        "family": "",
        "match_method": "",
        "match_score": 0.0,
        "matched_pattern": "",
        "matches": [],
        "error": "",
    }
    if not raw_text:
        return payload
    try:
        result = get_resp_diagnosis_mapper().map(raw_text, max_matches=max_matches)
    except Exception as exc:
        payload["status"] = "mapper_error"
        payload["error"] = str(exc)
        return payload

    primary = result.primary_match
    if primary is None:
        return payload

    payload.update(
        {
            "status": "matched",
            "normalized_text": primary.mapping.normalized_text,
            "icd10": primary.mapping.icd10,
            "concept_id": primary.mapping.concept_id,
            "template": primary.mapping.template,
            "family": primary.mapping.family,
            "match_method": primary.match_method,
            "match_score": round(float(primary.match_score), 6),
            "matched_pattern": primary.matched_pattern,
            "matches": [
                {
                    "normalized_text": match.mapping.normalized_text,
                    "icd10": match.mapping.icd10,
                    "concept_id": match.mapping.concept_id,
                    "template": match.mapping.template,
                    "family": match.mapping.family,
                    "matched_pattern": match.matched_pattern,
                    "match_method": match.match_method,
                    "match_score": round(float(match.match_score), 6),
                }
                for match in result.matches
            ],
        }
    )
    return payload


def compute_diagnosis_bonus(
    diagnosis_text: str,
    query_mapping: dict[str, Any],
    case_mapping: dict[str, Any],
    case_similarity_text: str,
) -> tuple[float, str]:
    diagnosis = str(diagnosis_text or "").strip()
    if diagnosis and diagnosis in case_similarity_text:
        return 0.03, f"主诊断文本与候选病例直接匹配 `{diagnosis}`。"

    query_concept = str(query_mapping.get("concept_id", "")).strip()
    case_concept = str(case_mapping.get("concept_id", "")).strip()
    if query_concept and query_concept == case_concept:
        normalized = str(query_mapping.get("normalized_text", "")).strip() or diagnosis
        return 0.03, f"标准化诊断一致，均映射为 `{normalized}`。"

    query_template = str(query_mapping.get("template", "")).strip()
    case_template = str(case_mapping.get("template", "")).strip()
    if query_template and query_template == case_template:
        return 0.025, f"诊断模板一致，均归入 `{query_template}`。"

    query_family = str(query_mapping.get("family", "")).strip()
    case_family = str(case_mapping.get("family", "")).strip()
    if query_family and query_family == case_family:
        return 0.015, f"诊断家族一致，均归入 `{query_family}`。"

    return 0.0, ""


def build_semantic_weight_plan(
    query_mapping: dict[str, Any],
    active_buckets: list[str],
) -> dict[str, Any]:
    fallback_weights = {
        "history_chain": 0.25,
        "evidence_chain": 0.25,
        "care_chain": 0.25,
        "severity_chain": 0.25,
    }
    selected_weights = dict(fallback_weights)
    source = "hardcoded_default"
    error = ""
    template = str(query_mapping.get("template", "")).strip()
    family = str(query_mapping.get("family", "")).strip()

    try:
        rule_set = get_resp_rule_set()
        if rule_set.scoring.semantic_chain_weights:
            selected_weights = {
                bucket: float(weight)
                for bucket, weight in rule_set.scoring.semantic_chain_weights.items()
                if bucket in CHAIN_LABELS
            }
            source = "default_weights"
        template_override = rule_set.scoring.template_overrides.get(template, {}) if template else {}
        override_weights = (
            template_override.get("semantic_chain_weights")
            if isinstance(template_override, dict)
            else None
        )
        if isinstance(override_weights, dict) and override_weights:
            selected_weights.update(
                {
                    bucket: float(weight)
                    for bucket, weight in override_weights.items()
                    if bucket in CHAIN_LABELS
                }
            )
            source = f"template:{template}"
    except Exception as exc:
        error = str(exc)

    normalized_weights: dict[str, float] = {}
    valid_buckets = [bucket for bucket in active_buckets if bucket in CHAIN_LABELS]
    total_weight = sum(max(0.0, float(selected_weights.get(bucket, 0.0))) for bucket in valid_buckets)
    if valid_buckets and total_weight > 0.0:
        normalized_weights = {
            bucket: float(selected_weights.get(bucket, 0.0)) / total_weight
            for bucket in valid_buckets
        }
    elif valid_buckets:
        uniform_weight = 1.0 / len(valid_buckets)
        normalized_weights = {bucket: uniform_weight for bucket in valid_buckets}
        source = f"{source}|uniform_active"

    return {
        "template": template,
        "family": family,
        "source": source,
        "error": error,
        "raw_weights": {bucket: round(float(selected_weights.get(bucket, 0.0)), 6) for bucket in CHAIN_LABELS},
        "normalized_weights": {bucket: round(weight, 6) for bucket, weight in normalized_weights.items()},
    }


def get_template_scoring_override(template_id: str) -> tuple[dict[str, Any], str]:
    template_key = str(template_id or "").strip()
    try:
        rule_set = get_resp_rule_set()
    except Exception as exc:
        return {}, str(exc)
    if template_key:
        override = rule_set.scoring.template_overrides.get(template_key, {})
    else:
        override = {}
    if not override:
        override = (
            rule_set.scoring.template_overrides.get("__default__", {})
            or rule_set.scoring.template_overrides.get("default", {})
            or {}
        )
    return (override if isinstance(override, dict) else {}), ""


def normalize_lookup_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower().strip()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", normalized)


LAB_NUMERIC_THRESHOLDS: dict[str, dict[str, float]] = {
    "lab.wbc": {"high": 10.0, "low": 4.0},
    "lab.neutrophil_ratio": {"high": 75.0},
    "lab.eosinophil_ratio": {"high": 5.0},
    "lab.crp": {"high": 8.0},
    "lab.pct": {"high": 0.25},
    "lab.esr": {"high": 20.0},
    "lab.ldh": {"high": 250.0},
    "bg.pao2": {"low": 60.0},
    "bg.paco2": {"high": 50.0},
    "bg.ph": {"low": 7.35, "high": 7.45},
    "bg.spo2": {"low": 90.0},
    "pf.fev1_fvc": {"low": 0.7},
    "pf.fev1_pred": {"low": 80.0},
    "lab.d_dimer": {"high": 0.5},
    "lab.cea": {"high": 5.0},
}


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def determine_numeric_lab_flag(item_id: str, value: float) -> str:
    thresholds = LAB_NUMERIC_THRESHOLDS.get(item_id, {})
    if "low" in thresholds and value < float(thresholds["low"]):
        return "low"
    if "high" in thresholds and value > float(thresholds["high"]):
        return "high"
    return ""


def extract_numeric_lab_findings(raw_text: str, lookup: dict[str, Any]) -> dict[str, dict[str, Any]]:
    findings: dict[str, dict[str, Any]] = {}
    text = str(raw_text or "")
    if not text:
        return findings
    for item_id, aliases in lookup.get("lab_aliases_by_id", {}).items():
        for alias in aliases:
            alias_text = str(alias or "").strip()
            if not alias_text:
                continue
            match = re.search(
                rf"{re.escape(alias_text)}\s*[:：=]?\s*([<>]?\d+(?:\.\d+)?)\s*([A-Za-z%/0-9^.+-]*)",
                text,
                re.IGNORECASE,
            )
            if not match:
                continue
            value_text = str(match.group(1)).strip().lstrip("<>")
            value = safe_float(value_text)
            if value is None:
                continue
            unit = str(match.group(2) or "").strip()
            flag = determine_numeric_lab_flag(item_id, value)
            if not flag:
                window_start = max(0, match.start() - 12)
                window_end = min(len(text), match.end() + 12)
                context = text[window_start:window_end].lower()
                abnormal_flags = lookup.get("lab_flags_by_id", {}).get(item_id, {})
                for candidate_flag, labels in abnormal_flags.items():
                    if any(str(label).lower() in context for label in labels):
                        flag = candidate_flag
                        break
            findings[item_id] = {
                "value": round(value, 6),
                "unit": unit,
                "flag": flag,
                "source": "numeric",
                "matched_alias": alias_text,
            }
            break
    return findings


@lru_cache(maxsize=1)
def get_resp_lookup_tables() -> dict[str, Any]:
    rule_set = get_resp_rule_set()

    def build_patterns(entries: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
        seen: set[tuple[str, str]] = set()
        rows: list[tuple[str, str, str]] = []
        for alias, item_id, extra in entries:
            normalized = normalize_lookup_text(alias)
            if not normalized or (item_id, normalized) in seen:
                continue
            seen.add((item_id, normalized))
            rows.append((normalized, item_id, extra))
        rows.sort(key=lambda item: len(item[0]), reverse=True)
        return rows

    concept_entries: list[tuple[str, str, str]] = []
    for concept in rule_set.concepts.values():
        for alias in (concept.normalized_text, *concept.aliases):
            concept_entries.append((alias, concept.concept_id, concept.category))

    lab_entries: list[tuple[str, str, str]] = []
    lab_aliases_by_id: dict[str, tuple[str, ...]] = {}
    lab_flags_by_id: dict[str, dict[str, tuple[str, ...]]] = {}
    for lab in rule_set.labs.values():
        aliases = tuple(alias for alias in (lab.canonical_name, *lab.aliases) if str(alias).strip())
        lab_aliases_by_id[lab.item_id] = aliases
        lab_flags_by_id[lab.item_id] = {
            str(flag).strip(): tuple(str(label).strip() for label in labels if str(label).strip())
            for flag, labels in lab.abnormal_flags.items()
        }
        for alias in aliases:
            lab_entries.append((alias, lab.item_id, "lab"))

    imaging_entries: list[tuple[str, str, str]] = []
    imaging_families_by_id: dict[str, tuple[str, ...]] = {}
    for imaging in rule_set.imagings.values():
        imaging_families_by_id[imaging.item_id] = tuple(imaging.families)
        for alias in (imaging.normalized_text, *imaging.aliases):
            imaging_entries.append((alias, imaging.item_id, "imaging"))

    drug_entries: list[tuple[str, str, str]] = []
    drug_class_by_id: dict[str, str] = {}
    drug_intents_by_id: dict[str, tuple[str, ...]] = {}
    for drug in rule_set.drugs.values():
        drug_class_by_id[drug.drug_id] = drug.drug_class
        drug_intents_by_id[drug.drug_id] = tuple(drug.treatment_intents)
        for alias in (drug.canonical_name, *drug.aliases):
            drug_entries.append((alias, drug.drug_id, drug.drug_class))

    # ---- Pre-compute concept alias embeddings for semantic matching ----
    concept_alias_texts: list[str] = []
    concept_alias_to_id: list[str] = []
    for alias, concept_id, category in concept_entries:
        normalized = normalize_lookup_text(alias)
        if normalized:
            concept_alias_texts.append(normalized)
            concept_alias_to_id.append(concept_id)

    concept_alias_vecs: np.ndarray | None = None
    if concept_alias_texts:
        try:
            response = create_embedding_with_retry(concept_alias_texts, timeout=120)
            concept_alias_vecs = np.asarray(extract_embedding_vectors(response), dtype=np.float32)
            print(f"[lookup] Pre-computed {len(concept_alias_vecs)} concept alias embeddings (dim={concept_alias_vecs.shape[1]})")
        except Exception as exc:
            print(f"[lookup] Concept embedding pre-compute failed, falling back to exact match only: {exc}")
            concept_alias_vecs = None

    return {
        "concept_patterns": build_patterns(concept_entries),
        "lab_patterns": build_patterns(lab_entries),
        "imaging_patterns": build_patterns(imaging_entries),
        "drug_patterns": build_patterns(drug_entries),
        "lab_aliases_by_id": lab_aliases_by_id,
        "lab_flags_by_id": lab_flags_by_id,
        "imaging_families_by_id": imaging_families_by_id,
        "drug_class_by_id": drug_class_by_id,
        "drug_intents_by_id": drug_intents_by_id,
        "concept_alias_texts": concept_alias_texts,
        "concept_alias_to_id": concept_alias_to_id,
        "concept_alias_vecs": concept_alias_vecs,
    }


def extract_structured_profile(
    diagnosis_text: str,
    chain_texts: dict[str, str],
    diagnosis_mapping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    diagnosis_mapping = diagnosis_mapping or {}
    lookup = get_resp_lookup_tables()
    joined_text = "\n".join(
        part
        for part in [
            str(diagnosis_text or "").strip(),
            *[str(chain_texts.get(bucket, "")).strip() for bucket in CHAIN_LABELS],
        ]
        if part
    )
    normalized_text = normalize_lookup_text(joined_text)
    numeric_lab_findings = extract_numeric_lab_findings(joined_text, lookup)

    concept_ids: set[str] = set()
    categories: dict[str, set[str]] = defaultdict(set)
    matched_alias_set: set[str] = set()
    alias_to_concept: dict[str, tuple[str, str]] = {}
    for alias, concept_id, category in lookup["concept_patterns"]:
        if alias and alias in normalized_text:
            concept_ids.add(concept_id)
            categories[category].add(concept_id)
            matched_alias_set.add(alias)
            alias_to_concept[alias] = (concept_id, category)

    # ---- Semantic concept matching fallback (Phase 3) ----
    # When few concepts found by exact match, try embedding-based fuzzy matching
    MIN_EXACT_CONCEPTS = 3
    SEMANTIC_SIMILARITY_THRESHOLD = 0.85
    HIGH_CONFIDENCE_THRESHOLD = 0.90  # Above this, skip Jaccard check (genuine synonyms)
    MIN_JACCARD = 0.35  # For lower-confidence matches, require some character overlap
    concept_alias_vecs = lookup.get("concept_alias_vecs")

    if len(concept_ids) < MIN_EXACT_CONCEPTS and concept_alias_vecs is not None:
        leftover_text = normalized_text
        for alias in sorted(matched_alias_set, key=len, reverse=True):
            leftover_text = leftover_text.replace(alias, "", 1)

        # Skip semantic matching if leftover is too short (noise only)
        chinese_chars = sum(1 for c in leftover_text if "\u4e00" <= c <= "\u9fff")
        if chinese_chars >= 4:
            text_ngrams = [leftover_text[i:j] for i in range(len(leftover_text) - 2)
                           for j in range(i + 3, min(i + 7, len(leftover_text) + 1))]
            text_ngrams = list(dict.fromkeys(text_ngrams))

            if text_ngrams:
                try:
                    batch_size = 64
                    alias_vecs_norm = concept_alias_vecs / (np.linalg.norm(concept_alias_vecs, axis=1, keepdims=True) + 1e-10)
                    alias_texts = lookup["concept_alias_texts"]
                    alias_ids = lookup["concept_alias_to_id"]

                    for start in range(0, len(text_ngrams), batch_size):
                        batch = text_ngrams[start:start + batch_size]
                        try:
                            resp = create_embedding_with_retry(batch, timeout=60)
                            batch_vecs = np.asarray(extract_embedding_vectors(resp), dtype=np.float32)
                        except Exception:
                            continue
                        batch_vecs_norm = batch_vecs / (np.linalg.norm(batch_vecs, axis=1, keepdims=True) + 1e-10)
                        sims = np.dot(batch_vecs_norm, alias_vecs_norm.T)
                        for i, sim_row in enumerate(sims):
                            best_idx = int(np.argmax(sim_row))
                            best_sim = float(sim_row[best_idx])
                            if best_sim >= SEMANTIC_SIMILARITY_THRESHOLD:
                                cid = alias_ids[best_idx]
                                alias_str = alias_texts[best_idx]
                                ngram_str = batch[i]
                                alias_len = len(alias_str)
                                ngram_len = len(ngram_str)
                                if alias_len > ngram_len * 2.5:
                                    continue
                                if best_sim < HIGH_CONFIDENCE_THRESHOLD:
                                    alias_chars = set(alias_str)
                                    ngram_chars = set(ngram_str)
                                    jaccard = len(alias_chars & ngram_chars) / len(alias_chars | ngram_chars) if alias_chars | ngram_chars else 0
                                    if jaccard < MIN_JACCARD:
                                        continue
                                if cid not in concept_ids and ngram_str != alias_str:
                                    concept_ids.add(cid)
                                    for alias, cp, cat in lookup["concept_patterns"]:
                                        if cp == cid:
                                            categories[cat].add(cid)
                                            break
                except Exception as exc:
                    pass  # Semantic matching is best-effort

    lab_ids = {item_id for alias, item_id, _ in lookup["lab_patterns"] if alias and alias in normalized_text}
    imaging_ids = {
        item_id for alias, item_id, _ in lookup["imaging_patterns"] if alias and alias in normalized_text
    }
    drug_ids = {item_id for alias, item_id, _ in lookup["drug_patterns"] if alias and alias in normalized_text}
    drug_classes = {
        lookup["drug_class_by_id"].get(drug_id, "")
        for drug_id in drug_ids
        if lookup["drug_class_by_id"].get(drug_id, "")
    }
    drug_intents = {
        treatment_intent
        for drug_id in drug_ids
        for treatment_intent in lookup.get("drug_intents_by_id", {}).get(drug_id, ())
        if treatment_intent
    }
    diagnosis_concept_ids = {
        str(diagnosis_mapping.get("concept_id", "")).strip()
    } - {""}
    blood_gas_ids = {item_id for item_id in lab_ids if item_id.startswith("bg.")}
    pulmonary_function_ids = {item_id for item_id in lab_ids if item_id.startswith("pf.")}
    infection_lab_ids = {
        item_id
        for item_id in lab_ids
        if item_id.startswith("lab.") and item_id not in {"lab.d_dimer", "lab.cea"}
    }
    imaging_family_ids = {
        family_id
        for imaging_id in imaging_ids
        for family_id in lookup.get("imaging_families_by_id", {}).get(imaging_id, ())
        if family_id
    }
    lab_flag_ids = {
        f"{item_id}:{str(payload.get('flag', '')).strip()}"
        for item_id, payload in numeric_lab_findings.items()
        if str(payload.get("flag", "")).strip()
    }
    if any(flag_id in lab_flag_ids for flag_id in {"bg.pao2:low", "bg.spo2:low"}):
        categories["complication"].add("resp.complication.hypoxemia")
    if "bg.paco2:high" in lab_flag_ids:
        categories["complication"].add("resp.complication.hypercapnia")
    if blood_gas_ids & {"bg.pao2", "bg.paco2"} and (
        "resp.complication.hypoxemia" in categories["complication"]
        or "resp.complication.hypercapnia" in categories["complication"]
    ):
        categories["acute_event"].add("resp.event.respiratory_failure")
    if "resp.support.imv" in categories["support"] or "resp.support.niv" in categories["support"]:
        categories["acute_event"].add("resp.event.mechanical_ventilation")
    if "img.pleural_effusion" in imaging_ids:
        categories["complication"].add("resp.complication.pleural_effusion")
    if "img.pneumothorax" in imaging_ids:
        categories["complication"].add("resp.complication.pneumothorax")
    if "img.ground_glass" in imaging_ids and (
        "resp.event.mechanical_ventilation" in categories["acute_event"]
        or "resp.support.imv" in categories["support"]
    ):
        categories["complication"].add("resp.complication.ards")

    return {
        "template": str(diagnosis_mapping.get("template", "")).strip(),
        "family": str(diagnosis_mapping.get("family", "")).strip(),
        "diagnosis_concept_ids": sorted(diagnosis_concept_ids),
        "concept_ids": sorted(concept_ids),
        "symptom_ids": sorted(categories.get("symptom", set()) | categories.get("sign", set())),
        "event_ids": sorted(categories.get("acute_event", set())),
        "complication_ids": sorted(categories.get("complication", set())),
        "support_ids": sorted(categories.get("support", set())),
        "procedure_ids": sorted(categories.get("procedure", set())),
        "lab_ids": sorted(lab_ids),
        "infection_lab_ids": sorted(infection_lab_ids),
        "blood_gas_ids": sorted(blood_gas_ids),
        "pulmonary_function_ids": sorted(pulmonary_function_ids),
        "lab_flag_ids": sorted(lab_flag_ids),
        "numeric_lab_findings": numeric_lab_findings,
        "imaging_ids": sorted(imaging_ids),
        "imaging_family_ids": sorted(imaging_family_ids),
        "drug_ids": sorted(drug_ids),
        "drug_classes": sorted(drug_classes),
        "drug_intents": sorted(drug_intents),
    }


@lru_cache(maxsize=1)
def get_resp_semantic_label_catalog() -> dict[str, dict[str, str]]:
    rule_set = get_resp_rule_set()
    diagnosis_by_concept: dict[str, str] = {}
    for mapping in rule_set.diagnosis_mappings:
        if mapping.concept_id and mapping.normalized_text and mapping.concept_id not in diagnosis_by_concept:
            diagnosis_by_concept[mapping.concept_id] = mapping.normalized_text

    concept_by_id = {
        concept_id: str(concept.normalized_text or concept_id).strip()
        for concept_id, concept in rule_set.concepts.items()
        if str(concept_id).strip()
    }
    lab_by_id = {
        item_id: str(lab.canonical_name or item_id).strip()
        for item_id, lab in rule_set.labs.items()
        if str(item_id).strip()
    }
    imaging_by_id = {
        item_id: str(imaging.normalized_text or item_id).strip()
        for item_id, imaging in rule_set.imagings.items()
        if str(item_id).strip()
    }
    drug_by_id = {
        drug_id: str(drug.canonical_name or drug_id).strip()
        for drug_id, drug in rule_set.drugs.items()
        if str(drug_id).strip()
    }
    return {
        "diagnosis_by_concept": diagnosis_by_concept,
        "concept_by_id": concept_by_id,
        "lab_by_id": lab_by_id,
        "imaging_by_id": imaging_by_id,
        "drug_by_id": drug_by_id,
    }


def _unique_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    rows: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        rows.append(text)
    return rows


def build_semantic_core_chain_texts(
    profile: dict[str, Any],
    diagnosis_mapping: dict[str, Any] | None = None,
) -> dict[str, str]:
    diagnosis_mapping = diagnosis_mapping or {}
    catalog = get_resp_semantic_label_catalog()

    def label_for(item_id: str) -> str:
        text = str(item_id or "").strip()
        if not text:
            return ""
        if ":" in text:
            base_id, flag = text.split(":", 1)
            base_label = (
                catalog["lab_by_id"].get(base_id)
                or catalog["concept_by_id"].get(base_id)
                or catalog["imaging_by_id"].get(base_id)
                or base_id
            )
            flag_label = {"high": "升高", "low": "降低", "positive": "阳性", "negative": "阴性"}.get(
                flag.strip().lower(),
                flag.strip(),
            )
            return f"{base_label} {flag_label}".strip()
        return (
            catalog["diagnosis_by_concept"].get(text)
            or catalog["concept_by_id"].get(text)
            or catalog["lab_by_id"].get(text)
            or catalog["imaging_by_id"].get(text)
            or catalog["drug_by_id"].get(text)
            or text
        )

    normalized_diagnosis = (
        str(diagnosis_mapping.get("normalized_text", "")).strip()
        or next(iter(profile_value_set(profile, "diagnosis_concept_ids")), "")
    )
    diagnosis_label = label_for(normalized_diagnosis) if normalized_diagnosis.startswith("resp.") else normalized_diagnosis

    template_override, _ = get_template_scoring_override(str(diagnosis_mapping.get("template", "")).strip())
    semantic_core_override = (
        template_override.get("semantic_core_config")
        if isinstance(template_override, dict)
        else {}
    )
    semantic_core_override = semantic_core_override if isinstance(semantic_core_override, dict) else {}
    default_bucket_fields = {
        "history_chain": ["diagnosis_concept_ids", "symptom_ids"],
        "evidence_chain": [
            "infection_lab_ids",
            "blood_gas_ids",
            "pulmonary_function_ids",
            "imaging_ids",
            "lab_flag_ids",
        ],
        "care_chain": ["drug_classes", "drug_intents", "support_ids", "procedure_ids", "drug_ids"],
        "severity_chain": ["event_ids", "complication_ids", "lab_flag_ids"],
    }

    def build_bucket_terms(bucket: str) -> list[str]:
        bucket_override = semantic_core_override.get(bucket, {})
        field_names = default_bucket_fields[bucket]
        prepend_diagnosis = bucket == "history_chain"
        max_terms = 0
        if isinstance(bucket_override, dict):
            override_fields = bucket_override.get("fields")
            if isinstance(override_fields, list) and override_fields:
                field_names = [str(name).strip() for name in override_fields if str(name).strip()]
            prepend_diagnosis = bool(bucket_override.get("prepend_diagnosis", prepend_diagnosis))
            try:
                max_terms = max(0, int(bucket_override.get("max_terms", 0) or 0))
            except (TypeError, ValueError):
                max_terms = 0
        raw_terms: list[str] = [diagnosis_label] if prepend_diagnosis and diagnosis_label else []
        for field_name in field_names:
            raw_terms.extend(label_for(item_id) for item_id in profile.get(field_name, []))
        normalized_terms = _unique_texts(raw_terms)
        if max_terms > 0:
            normalized_terms = normalized_terms[:max_terms]
        return normalized_terms

    history_terms = build_bucket_terms("history_chain")
    evidence_terms = build_bucket_terms("evidence_chain")
    care_terms = build_bucket_terms("care_chain")
    severity_terms = build_bucket_terms("severity_chain")
    return {
        "history_chain": " ".join(history_terms),
        "evidence_chain": " ".join(evidence_terms),
        "care_chain": " ".join(care_terms),
        "severity_chain": " ".join(severity_terms),
    }


def profile_value_set(profile: dict[str, Any], key: str) -> set[str]:
    value = profile.get(key, [])
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    text = str(value or "").strip()
    return {text} if text else set()


PROFILE_VECTOR_KEYS = (
    "diagnosis_concept_ids",
    "concept_ids",
    "symptom_ids",
    "event_ids",
    "complication_ids",
    "support_ids",
    "procedure_ids",
    "lab_ids",
    "infection_lab_ids",
    "blood_gas_ids",
    "pulmonary_function_ids",
    "lab_flag_ids",
    "imaging_ids",
    "imaging_family_ids",
    "drug_ids",
    "drug_classes",
    "drug_intents",
)


def build_profile_denoise_report(
    before: dict[str, set[str]],
    after: dict[str, set[str]],
    template_id: str,
    steps: list[str],
) -> dict[str, Any]:
    removed: dict[str, list[str]] = {}
    for key in PROFILE_VECTOR_KEYS:
        diff = sorted(before.get(key, set()) - after.get(key, set()))
        if diff:
            removed[key] = diff
    return {
        "applied": bool(steps),
        "template": template_id,
        "steps": steps,
        "removed": removed,
    }


def apply_template_profile_denoise(
    profile: dict[str, Any],
    template_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile_copy = dict(profile or {})
    template_key = str(template_id or "").strip()
    if not template_key:
        return profile_copy, {"applied": False, "template": "", "steps": [], "removed": {}}

    try:
        rule_set = get_resp_rule_set()
    except Exception as exc:
        return profile_copy, {
            "applied": False,
            "template": template_key,
            "steps": [],
            "removed": {},
            "error": str(exc),
        }

    template_config = rule_set.get_template(template_key)
    if template_config is None or not template_config.denoise:
        return profile_copy, {"applied": False, "template": template_key, "steps": [], "removed": {}}

    before = {key: profile_value_set(profile_copy, key) for key in PROFILE_VECTOR_KEYS}
    working = {key: set(values) for key, values in before.items()}
    steps: list[str] = []

    def intersect_field(field: str, allowed: set[str], step_text: str) -> None:
        current = working.get(field, set())
        if not allowed or not current:
            return
        kept = current & allowed
        if kept and kept != current:
            working[field] = kept
            steps.append(step_text)

    def keep_respiratory_intents(intents: set[str], step_text: str) -> None:
        if not intents:
            return
        respiratory_keywords = (
            "airway",
            "broncho",
            "infection",
            "oxygen",
            "ventilation",
            "resp",
            "exacerbation",
            "inflammation",
        )
        kept = {item for item in intents if any(keyword in item.lower() for keyword in respiratory_keywords)}
        if kept and kept != intents:
            working["drug_intents"] = kept
            steps.append(step_text)

    allowed_symptoms = set(template_config.key_symptoms)
    allowed_labs = set(template_config.key_evidence.get("labs", ()))
    allowed_bg = set(template_config.key_evidence.get("blood_gas", ()))
    allowed_pf = set(template_config.key_evidence.get("pulmonary_function", ()))
    allowed_imaging = set(template_config.key_evidence.get("imaging", ()))
    allowed_drug_classes = set(template_config.key_treatments.get("drug_classes", ()))
    allowed_supports = set(template_config.key_treatments.get("supports", ()))
    allowed_procedures = set(template_config.key_treatments.get("procedures", ()))
    allowed_events = set(template_config.severity_markers.get("events", ()))
    allowed_complications = set(template_config.severity_markers.get("complications", ()))
    allowed_lab_flags = set(template_config.severity_markers.get("lab_flags", ()))

    if template_config.denoise.get("suppress_generic_physical_exam"):
        intersect_field(
            "symptom_ids",
            allowed_symptoms | {"resp.sign.tachypnea"},
            "按模板保留核心症状体征",
        )

    if template_config.denoise.get("suppress_unrelated_comorbidity_lists"):
        intersect_field("lab_ids", allowed_labs, "按模板收敛实验室证据")
        intersect_field("infection_lab_ids", allowed_labs, "按模板收敛感染实验室证据")
        intersect_field("blood_gas_ids", allowed_bg, "按模板收敛血气证据")
        intersect_field("pulmonary_function_ids", allowed_pf, "按模板收敛肺功能证据")
        intersect_field("imaging_ids", allowed_imaging, "按模板收敛影像证据")
        intersect_field("event_ids", allowed_events, "按模板收敛事件证据")
        intersect_field("complication_ids", allowed_complications, "按模板收敛并发症证据")
        intersect_field("lab_flag_ids", allowed_lab_flags, "按模板收敛检验标志")

    if template_config.denoise.get("suppress_chronic_non_respiratory_drugs"):
        intersect_field("drug_classes", allowed_drug_classes, "按模板保留核心治疗类别")
        intersect_field("support_ids", allowed_supports, "按模板保留核心支持治疗")
        intersect_field("procedure_ids", allowed_procedures, "按模板保留核心操作")
        keep_respiratory_intents(working.get("drug_intents", set()), "按语义保留呼吸相关治疗意图")

    # Derive dependent fields from the denoised evidence collections.
    kept_lab_like_ids = (
        working.get("lab_ids", set())
        | working.get("blood_gas_ids", set())
        | working.get("pulmonary_function_ids", set())
    )
    if kept_lab_like_ids:
        filtered_numeric = {
            key: value
            for key, value in (profile_copy.get("numeric_lab_findings") or {}).items()
            if key in kept_lab_like_ids
        }
        profile_copy["numeric_lab_findings"] = filtered_numeric
        if working.get("lab_flag_ids"):
            working["lab_flag_ids"] = {
                item for item in working["lab_flag_ids"] if item.split(":", 1)[0] in kept_lab_like_ids
            }

    if working.get("imaging_ids"):
        lookup = get_resp_lookup_tables()
        working["imaging_family_ids"] = {
            family_id
            for imaging_id in working["imaging_ids"]
            for family_id in lookup.get("imaging_families_by_id", {}).get(imaging_id, ())
            if family_id
        }

    concept_ids = (
        working.get("diagnosis_concept_ids", set())
        | working.get("symptom_ids", set())
        | working.get("event_ids", set())
        | working.get("complication_ids", set())
        | working.get("support_ids", set())
        | working.get("procedure_ids", set())
    )
    working["concept_ids"] = concept_ids

    for key in PROFILE_VECTOR_KEYS:
        profile_copy[key] = sorted(working.get(key, set()))

    report = build_profile_denoise_report(before=before, after=working, template_id=template_key, steps=steps)
    return profile_copy, report


def restricted_overlap_score(
    query_values: set[str],
    case_values: set[str],
    template_values: set[str],
) -> tuple[float, dict[str, Any], bool]:
    if not template_values:
        return 0.0, {"query_hits": [], "case_hits": [], "shared_hits": [], "template_values": []}, False
    query_hits = query_values & template_values
    case_hits = case_values & template_values
    shared_hits = query_hits & case_hits
    union_hits = query_hits | case_hits
    # Use Dice coefficient (F1) instead of Jaccard to be robust against larger template key sets.
    # Dice = 2 * |intersection| / (|query_hits| + |case_hits|) — penalizes less for template-only concepts.
    dice_denom = len(query_hits) + len(case_hits)
    score = (2.0 * len(shared_hits) / dice_denom) if dice_denom > 0 else 0.0
    score = min(1.0, max(0.0, score))
    return (
        round(score, 6),
        {
            "query_hits": sorted(query_hits),
            "case_hits": sorted(case_hits),
            "shared_hits": sorted(shared_hits),
            "template_values": sorted(template_values),
        },
        True,
    )


RULE_CONDITION_PROFILE_KEYS: dict[str, str] = {
    "diagnosis_any": "diagnosis_concept_ids",
    "symptom_any": "symptom_ids",
    "event_any": "event_ids",
    "complication_any": "complication_ids",
    "support_any": "support_ids",
    "imaging_any": "imaging_ids",
    "lab_any": "lab_ids",
    "lab_flag_any": "lab_flag_ids",
    "drug_class_any": "drug_classes",
    "drug_intent_any": "drug_intents",
    "procedure_any": "procedure_ids",
    "concept_any": "concept_ids",
}


def collect_rule_condition_values(condition: dict[str, Any], collected: dict[str, set[str]]) -> None:
    if not isinstance(condition, dict) or not condition:
        return
    if "all" in condition:
        for item in condition.get("all") or []:
            collect_rule_condition_values(item, collected)
        return
    if "any" in condition:
        for item in condition.get("any") or []:
            collect_rule_condition_values(item, collected)
        return

    condition_key = next(iter(condition.keys()), "")
    if not condition_key:
        return
    values = {str(item).strip() for item in condition.get(condition_key, []) or [] if str(item).strip()}
    if values:
        collected.setdefault(condition_key, set()).update(values)


def build_template_rule_feature_space(template_id: str) -> dict[str, set[str]]:
    try:
        rule_set = get_resp_rule_set()
    except Exception:
        return {}
    template_config = rule_set.get_template(str(template_id or "").strip())
    if template_config is None or not template_config.rules:
        return {}

    feature_space: dict[str, set[str]] = {}
    for rule in template_config.rules:
        collect_rule_condition_values(getattr(rule, "when", {}) or {}, feature_space)
    return {key: values for key, values in feature_space.items() if values}


def profile_values_for_rule_condition(profile: dict[str, Any], condition_key: str) -> set[str]:
    if condition_key == "evidence_any":
        return (
            profile_value_set(profile, "lab_ids")
            | profile_value_set(profile, "blood_gas_ids")
            | profile_value_set(profile, "pulmonary_function_ids")
            | profile_value_set(profile, "imaging_ids")
        )
    profile_key = RULE_CONDITION_PROFILE_KEYS.get(condition_key, "")
    if not profile_key:
        return set()
    return profile_value_set(profile, profile_key)


def compute_rule_evidence_overlap(
    template_id: str,
    query_profile: dict[str, Any],
    case_profile: dict[str, Any],
) -> dict[str, Any]:
    feature_space = build_template_rule_feature_space(template_id)
    if not feature_space:
        return {
            "score": 0.0,
            "available": False,
            "feature_space": {},
            "details": {},
        }

    template_override, override_error = get_template_scoring_override(template_id)
    condition_weight_override = (
        template_override.get("rule_evidence_condition_weights")
        if isinstance(template_override, dict)
        else {}
    )
    condition_weight_override = (
        condition_weight_override if isinstance(condition_weight_override, dict) else {}
    )
    condition_scores: list[tuple[str, float, dict[str, Any], bool]] = []
    for condition_key, template_values in feature_space.items():
        score, details, available = restricted_overlap_score(
            profile_values_for_rule_condition(query_profile, condition_key),
            profile_values_for_rule_condition(case_profile, condition_key),
            template_values,
        )
        if available:
            condition_scores.append((condition_key, score, details, available))
    if condition_scores:
        weighted_total = 0.0
        weighted_score = 0.0
        normalized_condition_weights: dict[str, float] = {}
        for condition_key, score, _details, _available in condition_scores:
            weight = max(0.0, float(condition_weight_override.get(condition_key, 1.0)))
            weighted_total += weight
            weighted_score += score * weight
        if weighted_total > 0:
            normalized_condition_weights = {
                condition_key: round(
                    max(0.0, float(condition_weight_override.get(condition_key, 1.0))) / weighted_total,
                    6,
                )
                for condition_key, _score, _details, _available in condition_scores
            }
            averaged = {
                "score": round(weighted_score / weighted_total, 6),
                "available": True,
                "details": {
                    name: {"score": round(score, 6), **detail}
                    for name, score, detail, _ in condition_scores
                },
            }
        else:
            averaged = average_named_scores(condition_scores)
            normalized_condition_weights = {}
    else:
        averaged = average_named_scores(condition_scores)
        normalized_condition_weights = {}
    return {
        "score": round(float(averaged.get("score", 0.0) or 0.0), 6),
        "available": bool(averaged.get("available")),
        "feature_space": {key: sorted(values) for key, values in feature_space.items()},
        "details": averaged.get("details", {}),
        "normalized_condition_weights": normalized_condition_weights,
        "error": override_error,
    }


def compute_profile_overlap_score(
    query_profile: dict[str, Any],
    case_profile: dict[str, Any],
    profile_keys: list[str],
) -> dict[str, Any]:
    overlap_scores: list[tuple[str, float, dict[str, Any], bool]] = []
    for profile_key in profile_keys:
        query_values = profile_value_set(query_profile, profile_key)
        case_values = profile_value_set(case_profile, profile_key)
        template_values = query_values | case_values
        if not template_values:
            continue
        score, details, available = restricted_overlap_score(query_values, case_values, template_values)
        if available:
            overlap_scores.append((profile_key, score, details, available))

    averaged = average_named_scores(overlap_scores)
    return {
        "score": round(float(averaged.get("score", 0.0) or 0.0), 6),
        "available": bool(averaged.get("available")),
        "details": averaged.get("details", {}),
    }


def average_named_scores(items: list[tuple[str, float, dict[str, Any], bool]]) -> dict[str, Any]:
    active = [item for item in items if item[3]]
    if not active:
        return {"score": 0.0, "available": False, "details": {}}
    return {
        "score": round(sum(item[1] for item in active) / len(active), 6),
        "available": True,
        "details": {
            name: {"score": round(score, 6), **detail}
            for name, score, detail, _ in active
        },
    }


def get_template_for_scoring(
    query_mapping: dict[str, Any],
    case_mapping: dict[str, Any],
):
    try:
        rule_set = get_resp_rule_set()
    except Exception:
        return None, ""
    template_id = str(query_mapping.get("template", "")).strip() or str(
        case_mapping.get("template", "")
    ).strip()
    if not template_id:
        return None, ""
    return rule_set.get_template(template_id), template_id


def build_structured_match_score(
    query_profile: dict[str, Any],
    case_profile: dict[str, Any],
    query_mapping: dict[str, Any],
    case_mapping: dict[str, Any],
) -> dict[str, Any]:
    template_config, template_id = get_template_for_scoring(query_mapping, case_mapping)
    try:
        rule_set = get_resp_rule_set()
        slot_weights = dict(rule_set.scoring.structured_slot_weights or {})
    except Exception as exc:
        rule_set = None
        slot_weights = {}
        rule_set_error = str(exc)
    else:
        rule_set_error = ""

    slot_weights = {
        "history_slots": float(slot_weights.get("history_slots", 0.25)),
        "evidence_slots": float(slot_weights.get("evidence_slots", 0.25)),
        "care_slots": float(slot_weights.get("care_slots", 0.25)),
        "severity_slots": float(slot_weights.get("severity_slots", 0.25)),
    }
    if template_id:
        template_override = rule_set.scoring.template_overrides.get(template_id, {}) if rule_set else {}
        override_slot_weights = (
            template_override.get("structured_slot_weights")
            if isinstance(template_override, dict)
            else None
        )
        if isinstance(override_slot_weights, dict) and override_slot_weights:
            slot_weights.update(
                {
                    key: float(value)
                    for key, value in override_slot_weights.items()
                    if key in slot_weights
                }
            )

    def slot_profile_key(slot_name: str) -> str:
        key_map = {
            "labs": "infection_lab_ids",
            "blood_gas": "blood_gas_ids",
            "pulmonary_function": "pulmonary_function_ids",
            "imaging": "imaging_ids",
            "drug_classes": "drug_classes",
            "drug_intents": "drug_intents",
            "supports": "support_ids",
            "procedures": "procedure_ids",
            "events": "event_ids",
            "complications": "complication_ids",
            "lab_flags": "lab_flag_ids",
        }
        return key_map.get(slot_name, "concept_ids")

    if template_config is not None:
        history_score = restricted_overlap_score(
            profile_value_set(query_profile, "symptom_ids"),
            profile_value_set(case_profile, "symptom_ids"),
            set(template_config.key_symptoms),
        )
        evidence_parts = average_named_scores(
            [
                (
                    slot_name,
                    *restricted_overlap_score(
                        profile_value_set(query_profile, slot_profile_key(slot_name)),
                        profile_value_set(case_profile, slot_profile_key(slot_name)),
                        set(slot_values),
                    ),
                )
                for slot_name, slot_values in template_config.key_evidence.items()
            ]
        )
        care_parts = average_named_scores(
            [
                (
                    slot_name,
                    *restricted_overlap_score(
                        profile_value_set(query_profile, slot_profile_key(slot_name)),
                        profile_value_set(case_profile, slot_profile_key(slot_name)),
                        set(slot_values),
                    ),
                )
                for slot_name, slot_values in template_config.key_treatments.items()
            ]
        )
        severity_parts = average_named_scores(
            [
                (
                    slot_name,
                    *restricted_overlap_score(
                        profile_value_set(query_profile, slot_profile_key(slot_name)),
                        profile_value_set(case_profile, slot_profile_key(slot_name)),
                        set(slot_values),
                    ),
                )
                for slot_name, slot_values in template_config.severity_markers.items()
            ]
        )
        slot_results = {
            "history_slots": {
                "score": history_score[0],
                "available": history_score[2],
                "details": history_score[1],
            },
            "evidence_slots": evidence_parts,
            "care_slots": care_parts,
            "severity_slots": severity_parts,
        }
        source = f"template:{template_id}"
    else:
        generic_pairs = {
            "history_slots": restricted_overlap_score(
                profile_value_set(query_profile, "symptom_ids"),
                profile_value_set(case_profile, "symptom_ids"),
                profile_value_set(query_profile, "symptom_ids") | profile_value_set(case_profile, "symptom_ids"),
            ),
            "evidence_slots": restricted_overlap_score(
                profile_value_set(query_profile, "lab_ids")
                | profile_value_set(query_profile, "blood_gas_ids")
                | profile_value_set(query_profile, "pulmonary_function_ids")
                | profile_value_set(query_profile, "imaging_ids"),
                profile_value_set(case_profile, "lab_ids")
                | profile_value_set(case_profile, "blood_gas_ids")
                | profile_value_set(case_profile, "pulmonary_function_ids")
                | profile_value_set(case_profile, "imaging_ids"),
                (
                    profile_value_set(query_profile, "lab_ids")
                    | profile_value_set(case_profile, "lab_ids")
                    | profile_value_set(query_profile, "blood_gas_ids")
                    | profile_value_set(case_profile, "blood_gas_ids")
                    | profile_value_set(query_profile, "pulmonary_function_ids")
                    | profile_value_set(case_profile, "pulmonary_function_ids")
                    | profile_value_set(query_profile, "imaging_ids")
                    | profile_value_set(case_profile, "imaging_ids")
                ),
            ),
            "care_slots": restricted_overlap_score(
                profile_value_set(query_profile, "drug_classes")
                | profile_value_set(query_profile, "drug_intents")
                | profile_value_set(query_profile, "support_ids")
                | profile_value_set(query_profile, "procedure_ids"),
                profile_value_set(case_profile, "drug_classes")
                | profile_value_set(case_profile, "drug_intents")
                | profile_value_set(case_profile, "support_ids")
                | profile_value_set(case_profile, "procedure_ids"),
                (
                    profile_value_set(query_profile, "drug_classes")
                    | profile_value_set(case_profile, "drug_classes")
                    | profile_value_set(query_profile, "drug_intents")
                    | profile_value_set(case_profile, "drug_intents")
                    | profile_value_set(query_profile, "support_ids")
                    | profile_value_set(case_profile, "support_ids")
                    | profile_value_set(query_profile, "procedure_ids")
                    | profile_value_set(case_profile, "procedure_ids")
                ),
            ),
            "severity_slots": restricted_overlap_score(
                profile_value_set(query_profile, "event_ids")
                | profile_value_set(query_profile, "complication_ids")
                | profile_value_set(query_profile, "lab_flag_ids"),
                profile_value_set(case_profile, "event_ids")
                | profile_value_set(case_profile, "complication_ids")
                | profile_value_set(case_profile, "lab_flag_ids"),
                (
                    profile_value_set(query_profile, "event_ids")
                    | profile_value_set(case_profile, "event_ids")
                    | profile_value_set(query_profile, "complication_ids")
                    | profile_value_set(case_profile, "complication_ids")
                    | profile_value_set(query_profile, "lab_flag_ids")
                    | profile_value_set(case_profile, "lab_flag_ids")
                ),
            ),
        }
        slot_results = {
            name: {"score": score, "available": available, "details": details}
            for name, (score, details, available) in generic_pairs.items()
        }
        source = "generic_overlap"

    available_slots = [name for name, item in slot_results.items() if item.get("available")]
    total_weight = sum(slot_weights.get(name, 0.0) for name in available_slots)
    normalized_slot_weights = {
        name: (slot_weights.get(name, 0.0) / total_weight if total_weight > 0 else 0.0)
        for name in available_slots
    }
    score = sum(
        slot_results[name]["score"] * normalized_slot_weights.get(name, 0.0)
        for name in available_slots
    )
    template_override = rule_set.scoring.template_overrides.get(template_id, {}) if (rule_set and template_id) else {}
    density_config = (
        template_override.get("structured_density_bonus")
        if isinstance(template_override, dict)
        else None
    )
    density_bonus = 0.0
    density_bonus_report: dict[str, Any] = {"applied": False}
    if isinstance(density_config, dict):
        required_slots = [
            str(name).strip()
            for name in density_config.get("required_slots", [])
            if str(name).strip() in slot_results
        ]
        min_slot_score = float(density_config.get("min_slot_score", 0.55) or 0.55)
        min_avg_score = float(density_config.get("min_avg_score", 0.6) or 0.6)
        bonus_value = float(density_config.get("bonus", 0.0) or 0.0)
        active_required_scores = [
            float(slot_results[name]["score"])
            for name in required_slots
            if slot_results.get(name, {}).get("available")
        ]
        if required_slots and len(active_required_scores) == len(required_slots):
            avg_required_score = sum(active_required_scores) / len(active_required_scores)
            if min(active_required_scores) >= min_slot_score and avg_required_score >= min_avg_score:
                density_bonus = max(0.0, bonus_value)
                density_bonus_report = {
                    "applied": density_bonus > 0,
                    "required_slots": required_slots,
                    "required_scores": {
                        name: round(float(slot_results[name]["score"]), 6) for name in required_slots
                    },
                    "min_slot_score": round(min_slot_score, 6),
                    "min_avg_score": round(min_avg_score, 6),
                    "avg_required_score": round(avg_required_score, 6),
                    "bonus": round(density_bonus, 6),
                }
            else:
                density_bonus_report = {
                    "applied": False,
                    "required_slots": required_slots,
                    "required_scores": {
                        name: round(float(slot_results[name]["score"]), 6) for name in required_slots
                    },
                    "min_slot_score": round(min_slot_score, 6),
                    "min_avg_score": round(min_avg_score, 6),
                    "bonus": round(bonus_value, 6),
                }
    score = min(1.0, score + density_bonus)
    # Apply structured_score floor to prevent data-sparsity penalization
    STRUCTURED_MIN_FLOOR = 0.35
    if score < STRUCTURED_MIN_FLOOR:
        score = STRUCTURED_MIN_FLOOR

    return {
        "score": round(score, 6),
        "available": bool(available_slots),
        "source": source,
        "template": template_id,
        "error": rule_set_error,
        "raw_slot_weights": {name: round(float(value), 6) for name, value in slot_weights.items()},
        "normalized_slot_weights": {
            name: round(weight, 6) for name, weight in normalized_slot_weights.items()
        },
        "density_bonus": round(density_bonus, 6),
        "density_bonus_report": density_bonus_report,
        "slot_results": slot_results,
    }


def evaluate_rule_condition(condition: dict[str, Any], profile: dict[str, Any]) -> bool:
    if "all" in condition:
        return all(evaluate_rule_condition(item, profile) for item in condition.get("all") or [])
    if "any" in condition:
        return any(evaluate_rule_condition(item, profile) for item in condition.get("any") or [])

    condition_key = next(iter(condition.keys()), "")
    values = {str(item).strip() for item in condition.get(condition_key, []) if str(item).strip()}
    if not values:
        return False
    key_map = {
        "diagnosis_any": "diagnosis_concept_ids",
        "symptom_any": "symptom_ids",
        "event_any": "event_ids",
        "complication_any": "complication_ids",
        "support_any": "support_ids",
        "imaging_any": "imaging_ids",
        "lab_any": "lab_ids",
        # evidence_any is used by some stable/chronic templates to express
        # shared structured evidence across labs, blood gas, pulmonary function,
        # and imaging findings.
        "evidence_any": "",
        "lab_flag_any": "lab_flag_ids",
        "drug_class_any": "drug_classes",
        "drug_intent_any": "drug_intents",
        "procedure_any": "procedure_ids",
        "concept_any": "concept_ids",
    }
    if condition_key == "evidence_any":
        evidence_values = (
            profile_value_set(profile, "lab_ids")
            | profile_value_set(profile, "blood_gas_ids")
            | profile_value_set(profile, "pulmonary_function_ids")
            | profile_value_set(profile, "imaging_ids")
        )
        return bool(evidence_values & values)
    profile_key = key_map.get(condition_key, "")
    if not profile_key:
        return False
    return bool(profile_value_set(profile, profile_key) & values)


def evaluate_template_rule_hits(template_id: str, profile: dict[str, Any]) -> dict[str, Any]:
    try:
        rule_set = get_resp_rule_set()
    except Exception as exc:
        return {"available": False, "template": template_id, "hits": [], "error": str(exc)}
    template_config = rule_set.get_template(template_id)
    if template_config is None or not template_config.rules:
        return {"available": False, "template": template_id, "hits": [], "error": ""}

    hits = []
    for rule in template_config.rules:
        if evaluate_rule_condition(rule.when, profile):
            hits.append(
                {
                    "rule_id": rule.rule_id,
                    "score": round(float(rule.then.get("add_score", 0.0) or 0.0), 6),
                    "rationale": getattr(rule, "rationale", ""),
                    "evidence_level": getattr(rule, "evidence_level", ""),
                    "recommendation_strength": getattr(rule, "recommendation_strength", ""),
                    "citations": list(getattr(rule, "citations", ()) or ()),
                    "standard_mappings": list(getattr(rule, "standard_mappings", ()) or ()),
                    "tags": list(getattr(rule, "tags", ()) or ()),
                }
            )
    return {"available": True, "template": template_id, "hits": hits, "error": ""}


def build_rule_match_score(
    template_id: str,
    query_profile: dict[str, Any],
    case_profile: dict[str, Any],
) -> dict[str, Any]:
    if not template_id:
        return {
            "score": 0.0,
            "available": False,
            "template": "",
            "query_rule_hits": [],
            "case_rule_hits": [],
            "shared_rule_ids": [],
            "raw_shared_score": 0.0,
            "error": "",
        }
    query_hits = evaluate_template_rule_hits(template_id, query_profile)
    case_hits = evaluate_template_rule_hits(template_id, case_profile)
    if not query_hits.get("available") or not case_hits.get("available"):
        return {
            "score": 0.0,
            "available": False,
            "template": template_id,
            "query_rule_hits": query_hits.get("hits", []),
            "case_rule_hits": case_hits.get("hits", []),
            "shared_rule_ids": [],
            "raw_shared_score": 0.0,
            "error": query_hits.get("error") or case_hits.get("error") or "",
        }

    query_score_map = {item["rule_id"]: float(item["score"]) for item in query_hits.get("hits", [])}
    case_score_map = {item["rule_id"]: float(item["score"]) for item in case_hits.get("hits", [])}
    shared_rule_ids = sorted(set(query_score_map) & set(case_score_map))
    raw_shared_score = sum(min(query_score_map[rule_id], case_score_map[rule_id]) for rule_id in shared_rule_ids)
    # 分母以查询侧命中规则为基准（查询侧对齐）：
    # 候选比查询命中更多规则（case 独有命中）不应扩大分母而压低共享分数，
    # 候选多出的证据通过 rule_evidence_overlap（Dice 重叠）另行计分。
    max_possible_score = sum(query_score_map.values())
    shared_rule_score = raw_shared_score / max_possible_score if max_possible_score > 0 else 0.0
    evidence_overlap = compute_rule_evidence_overlap(template_id, query_profile, case_profile)
    evidence_score = float(evidence_overlap.get("score", 0.0) or 0.0)
    template_override, override_error = get_template_scoring_override(template_id)
    mix_override = (
        template_override.get("rule_match_weights")
        if isinstance(template_override, dict)
        else {}
    )
    mix_override = mix_override if isinstance(mix_override, dict) else {}
    shared_weight = max(0.0, float(mix_override.get("shared_rule_score", 0.7)))
    evidence_weight = max(0.0, float(mix_override.get("rule_evidence_score", 0.3)))
    if evidence_overlap.get("available"):
        total_weight = shared_weight + evidence_weight
        if total_weight > 0:
            normalized_shared_weight = shared_weight / total_weight
            normalized_evidence_weight = evidence_weight / total_weight
        else:
            normalized_shared_weight = 0.7
            normalized_evidence_weight = 0.3
        blended_score = (
            normalized_shared_weight * shared_rule_score
            + normalized_evidence_weight * evidence_score
        )
        score = max(shared_rule_score, blended_score)
    else:
        normalized_shared_weight = 1.0
        normalized_evidence_weight = 0.0
        score = shared_rule_score
    return {
        "score": round(score, 6),
        "available": bool(query_hits.get("hits") or case_hits.get("hits") or evidence_overlap.get("available")),
        "template": template_id,
        "query_rule_hits": query_hits.get("hits", []),
        "case_rule_hits": case_hits.get("hits", []),
        "shared_rule_ids": shared_rule_ids,
        "shared_rule_score": round(shared_rule_score, 6),
        "rule_evidence_score": round(evidence_score, 6),
        "rule_evidence_overlap": evidence_overlap,
        "normalized_rule_weights": {
            "shared_rule_score": round(normalized_shared_weight, 6),
            "rule_evidence_score": round(normalized_evidence_weight, 6),
        },
        "raw_shared_score": round(raw_shared_score, 6),
        "error": override_error,
    }


def compute_diagnosis_alignment(
    diagnosis_text: str,
    query_mapping: dict[str, Any],
    case_mapping: dict[str, Any],
    case_similarity_text: str,
    query_profile: dict[str, Any],
    case_profile: dict[str, Any],
) -> dict[str, Any]:
    diagnosis = str(diagnosis_text or "").strip()
    query_concept = str(query_mapping.get("concept_id", "")).strip()
    case_concept = str(case_mapping.get("concept_id", "")).strip()
    query_template = str(query_mapping.get("template", "")).strip()
    case_template = str(case_mapping.get("template", "")).strip()
    query_family = str(query_mapping.get("family", "")).strip()
    case_family = str(case_mapping.get("family", "")).strip()
    phenotype_overlap = compute_profile_overlap_score(
        query_profile=query_profile,
        case_profile=case_profile,
        profile_keys=[
            "diagnosis_concept_ids",
            "event_ids",
            "complication_ids",
            "imaging_family_ids",
        ],
    )

    def build_alignment_result(
        *,
        match_level: str,
        diagnosis_bounds: tuple[float, float],
        hierarchy_bounds: tuple[float, float],
        diagnosis_bonus: float,
        reason: str,
    ) -> dict[str, Any]:
        if phenotype_overlap.get("available"):
            phenotype_score = float(phenotype_overlap.get("score", 0.0) or 0.0)
        else:
            phenotype_score = 1.0 if match_level == "concept_exact" else 0.5
        disease_consistency_score = diagnosis_bounds[0] + (
            diagnosis_bounds[1] - diagnosis_bounds[0]
        ) * phenotype_score
        diagnosis_hierarchy_score = hierarchy_bounds[0] + (
            hierarchy_bounds[1] - hierarchy_bounds[0]
        ) * phenotype_score
        return {
            "match_level": match_level,
            "diagnosis_hierarchy_score": round(diagnosis_hierarchy_score, 6),
            "disease_consistency_score": round(disease_consistency_score, 6),
            "diagnosis_bonus": diagnosis_bonus,
            "reason": reason,
            "phenotype_overlap_score": round(phenotype_score, 6),
            "phenotype_overlap": phenotype_overlap,
            "available": True,
        }

    if query_concept and case_concept and query_concept == case_concept:
        return build_alignment_result(
            match_level="concept_exact",
            diagnosis_bounds=(0.92, 1.0),
            hierarchy_bounds=(0.92, 1.0),
            diagnosis_bonus=0.03,
            reason=f"标准化诊断一致，均映射为 `{query_mapping.get('normalized_text', diagnosis)}`。",
        )
    if diagnosis and diagnosis in case_similarity_text:
        return build_alignment_result(
            match_level="raw_text_match",
            diagnosis_bounds=(0.75, 0.85),
            hierarchy_bounds=(0.85, 0.95),
            diagnosis_bonus=0.03,
            reason=f"主诊断文本与候选病例直接匹配 `{diagnosis}`。",
        )
    if query_template and case_template and query_template == case_template:
        return build_alignment_result(
            match_level="template_match",
            diagnosis_bounds=(0.82, 0.95),
            hierarchy_bounds=(0.72, 0.85),
            diagnosis_bonus=0.025,
            reason=f"诊断模板一致，均归入 `{query_template}`。",
        )
    if query_family and case_family and query_family == case_family:
        return build_alignment_result(
            match_level="family_match",
            diagnosis_bounds=(0.72, 0.85),
            hierarchy_bounds=(0.52, 0.65),
            diagnosis_bonus=0.015,
            reason=f"诊断家族一致，均归入 `{query_family}`。",
        )
    return {
        "match_level": "none",
        "diagnosis_hierarchy_score": 0.0,
        "disease_consistency_score": 0.0,
        "diagnosis_bonus": 0.0,
        "reason": "",
        "phenotype_overlap_score": 0.0,
        "phenotype_overlap": phenotype_overlap,
        "available": bool(diagnosis or query_concept or case_concept or query_family or case_family),
    }


def build_final_fusion_score(
    components: dict[str, float],
    availability: dict[str, bool],
    *,
    template_id: str = "",
) -> dict[str, Any]:
    fallback_weights = {
        "semantic_score": 0.10,
        "structured_score": 0.35,
        "guideline_rule_score": 0.25,
        "disease_consistency_score": 0.15,
        "diagnosis_hierarchy_score": 0.15,
    }
    source = "hardcoded_default"
    error = ""
    try:
        rule_set = get_resp_rule_set()
        final_weights = {
            key: float(value)
            for key, value in (rule_set.scoring.final_score_weights or {}).items()
            if key in fallback_weights
        }
        if final_weights:
            fallback_weights.update(final_weights)
            source = "resp_rules.final_score_weights"
        template_override = rule_set.scoring.template_overrides.get(template_id, {}) if template_id else {}
        override_weights = (
            template_override.get("final_score_weights")
            if isinstance(template_override, dict)
            else None
        )
        if isinstance(override_weights, dict) and override_weights:
            fallback_weights.update(
                {
                    key: float(value)
                    for key, value in override_weights.items()
                    if key in fallback_weights
                }
            )
            source = f"{source}|template:{template_id}" if source else f"template:{template_id}"
    except Exception as exc:
        error = str(exc)

    configured_weights = {key: float(value) for key, value in fallback_weights.items()}
    effective_weights = dict(configured_weights)
    active_keys = [key for key, available in availability.items() if available and key in effective_weights]

    adaptive_adjustments: list[dict[str, Any]] = []
    diagnosis_pair = ("disease_consistency_score", "diagnosis_hierarchy_score")
    if all(key in active_keys for key in diagnosis_pair):
        disease_score = float(components.get("disease_consistency_score", 0.0) or 0.0)
        hierarchy_score = float(components.get("diagnosis_hierarchy_score", 0.0) or 0.0)
        pair_weight = sum(effective_weights.get(key, 0.0) for key in diagnosis_pair)
        if (
            pair_weight > 0.24
            and min(disease_score, hierarchy_score) >= 0.92
            and abs(disease_score - hierarchy_score) <= 0.015
        ):
            rule_score = float(components.get("guideline_rule_score", 0.0) or 0.0)
            # When guideline_rule_score is near ceiling, redirect its share to structured_score
            RULE_NEAR_CEILING = 0.95
            preferred_targets = {
                "structured_score": 0.5,
                "guideline_rule_score": 0.3,
                "semantic_score": 0.2,
            }
            if rule_score >= RULE_NEAR_CEILING and "guideline_rule_score" in preferred_targets:
                # Rule is near max — redirect its 30% share to structured (70%) + semantic (30%)
                rule_share = preferred_targets.pop("guideline_rule_score")
                preferred_targets["structured_score"] = preferred_targets.get("structured_score", 0.0) + rule_share * 0.7
                preferred_targets["semantic_score"] = preferred_targets.get("semantic_score", 0.0) + rule_share * 0.3
            targets = [
                key
                for key in preferred_targets
                if key in active_keys and float(components.get(key, 0.0) or 0.0) >= 0.45
            ]
            if targets:
                # Weaken dedup instead of removing it: keep diagnosis anti-double-counting,
                # but allow slightly more diagnosis alignment contribution for close matches.
                capped_pair_weight = 0.27
                excess_weight = pair_weight - capped_pair_weight
                scale = capped_pair_weight / pair_weight if pair_weight > 0 else 1.0
                for key in diagnosis_pair:
                    effective_weights[key] = effective_weights.get(key, 0.0) * scale
                target_total = sum(preferred_targets[key] for key in targets)
                for key in targets:
                    effective_weights[key] = effective_weights.get(key, 0.0) + (
                        excess_weight * preferred_targets[key] / target_total
                    )
                adaptive_adjustments.append(
                    {
                        "type": "diagnosis_alignment_dedup",
                        "reason": "疾病一致性与诊断层级同时高分且高度重合，减少重复加权并回流到结构化/规则/语义分项。",
                        "pair_scores": {
                            "disease_consistency_score": round(disease_score, 6),
                            "diagnosis_hierarchy_score": round(hierarchy_score, 6),
                        },
                        "pair_weight_before": round(pair_weight, 6),
                        "pair_weight_after": round(capped_pair_weight, 6),
                        "redistributed_targets": targets,
                        "rule_near_ceiling": bool(rule_score >= RULE_NEAR_CEILING),
                    }
                )

    total_weight = sum(effective_weights.get(key, 0.0) for key in active_keys)
    normalized_weights = {
        key: (effective_weights.get(key, 0.0) / total_weight if total_weight > 0 else 0.0)
        for key in active_keys
    }
    score = sum(float(components.get(key, 0.0)) * normalized_weights.get(key, 0.0) for key in active_keys)
    return {
        "score": round(score, 6),
        "source": source,
        "error": error,
        "configured_weights": {
            key: round(float(value), 6) for key, value in configured_weights.items()
        },
        "raw_weights": {key: round(float(value), 6) for key, value in effective_weights.items()},
        "normalized_weights": {key: round(float(value), 6) for key, value in normalized_weights.items()},
        "components": {key: round(float(value), 6) for key, value in components.items()},
        "availability": dict(availability),
        "adaptive_adjustments": adaptive_adjustments,
    }


def build_evidence_coverage(
    *,
    chain_scores: dict[str, dict[str, float]],
    score_breakdown: dict[str, Any],
    case: dict[str, Any],
    query_diagnosis_mapping: dict[str, Any],
) -> dict[str, Any]:
    active_buckets = sorted(bucket for bucket in chain_scores.keys() if bucket in CHAIN_LABELS)

    bucket_stats: dict[str, Any] = {}
    any_candidate_fragments = False
    for bucket in CHAIN_LABELS:
        items = ((case.get("chains") or {}).get(bucket) or []) if isinstance(case.get("chains"), dict) else []
        fragment_count = 0
        for item in items:
            if isinstance(item, dict):
                fragment_count += len(item.get("evidence_fragments") or [])
        any_candidate_fragments = any_candidate_fragments or fragment_count > 0
        bucket_stats[bucket] = {
            "query_active": bucket in active_buckets,
            "candidate_hyperedges": len(items),
            "candidate_evidence_fragments": int(fragment_count),
            "bucket_score": float((chain_scores.get(bucket) or {}).get("final", 0.0) or 0.0),
        }

    structured_match = score_breakdown.get("structured_match") or {}
    slot_results = structured_match.get("slot_results") or {}
    structured_available = bool(structured_match.get("available"))

    structured_slots = ("history_slots", "evidence_slots", "care_slots", "severity_slots")
    structured_slot_details: dict[str, Any] = {}
    structured_slot_scores: list[float] = []
    for slot_name in structured_slots:
        slot = slot_results.get(slot_name) or {}
        details = slot.get("details") or {}
        shared_hits = details.get("shared_hits") or []
        query_hits = details.get("query_hits") or []
        case_hits = details.get("case_hits") or []
        template_values = details.get("template_values") or []
        slot_available = bool(slot.get("available"))
        slot_score = float(slot.get("score", 0.0) or 0.0)
        if slot_available:
            structured_slot_scores.append(slot_score)
        structured_slot_details[slot_name] = {
            "available": slot_available,
            "score": round(slot_score, 6),
            "shared_hit_count": len(shared_hits),
            "query_hit_count": len(query_hits),
            "case_hit_count": len(case_hits),
            "template_value_count": len(template_values),
        }

    structured_strength = (
        (sum(structured_slot_scores) / len(structured_slot_scores)) if structured_slot_scores else 0.0
    )

    rule_match = score_breakdown.get("rule_match") or {}
    rule_available = bool(rule_match.get("available"))
    shared_rule_ids = list(rule_match.get("shared_rule_ids") or [])
    rule_strength = max(
        float(score_breakdown.get("guideline_rule_score", 0.0) or 0.0),
        float(rule_match.get("shared_rule_score", 0.0) or 0.0),
        float(rule_match.get("rule_evidence_score", 0.0) or 0.0),
    )

    diagnosis_alignment = score_breakdown.get("diagnosis_alignment") or {}
    diagnosis_available = bool(diagnosis_alignment.get("available"))
    diagnosis_strength = max(
        float(diagnosis_alignment.get("phenotype_overlap_score", 0.0) or 0.0),
        float(score_breakdown.get("disease_consistency_score", 0.0) or 0.0),
        float(score_breakdown.get("diagnosis_hierarchy_score", 0.0) or 0.0),
    )

    semantic_strength = float(score_breakdown.get("semantic_score", 0.0) or 0.0)
    fusion_plan = score_breakdown.get("fusion_plan") or {}
    semantic_weights = fusion_plan.get("normalized_weights") or {}
    if active_buckets and isinstance(semantic_weights, dict):
        weighted_sum = 0.0
        total_weight = 0.0
        for bucket in active_buckets:
            weight = float(semantic_weights.get(bucket, 0.0) or 0.0)
            if weight <= 0:
                continue
            total_weight += weight
            weighted_sum += weight * float((chain_scores.get(bucket) or {}).get("final", 0.0) or 0.0)
        if total_weight > 0.0:
            semantic_strength = weighted_sum / total_weight

    coverage_score = (
        0.15 * semantic_strength
        + 0.45 * structured_strength
        + 0.25 * rule_strength
        + 0.15 * diagnosis_strength
    )
    coverage_score = max(0.0, min(1.0, float(coverage_score)))

    warnings: list[str] = []
    if not active_buckets:
        warnings.append("查询侧四桶语义链为空")
    if not structured_available:
        warnings.append("结构化证据不可用或抽取为空")
    if not rule_available:
        warnings.append("模板规则证据不可用或未命中")
    match_level = str(diagnosis_alignment.get("match_level", "") or "")
    query_concept = str(query_diagnosis_mapping.get("concept_id", "") or "").strip()
    if not query_concept and match_level in {"raw_text_match", "none", ""}:
        warnings.append("主诊断未完成标准化映射")
    if not any_candidate_fragments:
        warnings.append("候选病例缺少可解析证据片段")

    return {
        "score": round(coverage_score, 6),
        "strength": {
            "semantic": round(float(semantic_strength), 6),
            "structured": round(float(structured_strength), 6),
            "rules": round(float(rule_strength), 6),
            "diagnosis": round(float(diagnosis_strength), 6),
        },
        "availability": {
            "semantic": bool(active_buckets),
            "structured": structured_available,
            "rules": rule_available,
            "diagnosis": diagnosis_available,
            "candidate_fragments": any_candidate_fragments,
        },
        "details": {
            "buckets": bucket_stats,
            "structured_slots": structured_slot_details,
            "shared_rule_count": len(shared_rule_ids),
            "query_rule_hit_count": len(rule_match.get("query_rule_hits") or []),
            "case_rule_hit_count": len(rule_match.get("case_rule_hits") or []),
            "diagnosis_match_level": match_level,
            "template": str(structured_match.get("template", "") or ""),
        },
        "warnings": warnings,
    }


def sanitize_rerank_text(text: str, *, bucket: str = "", max_len: int = 160) -> str:
    compact = " ".join(str(text or "").split())
    if not compact:
        return ""
    cleaned = compact
    for pattern in RERANK_TEXT_NOISE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,，;；。")
    if not cleaned:
        cleaned = compact

    segments = [
        segment.strip(" ,，;；。")
        for segment in re.split(r"[\n;；。]+", cleaned)
        if segment.strip(" ,，;；。")
    ]
    if not segments:
        return shorten_text(cleaned, max_len)

    keywords = tuple(keyword.lower() for keyword in RERANK_CHAIN_KEYWORDS.get(bucket, ()))
    prioritized: list[str] = []
    fallback: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        if len(segment) <= 2 or segment in seen:
            continue
        seen.add(segment)
        lowered = segment.lower()
        if keywords and any(keyword in lowered for keyword in keywords):
            prioritized.append(segment)
        else:
            fallback.append(segment)
    ordered_segments = prioritized + fallback
    return shorten_text("；".join(ordered_segments), max_len)


def build_dataset_configs() -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {
        dataset_id: dict(config) for dataset_id, config in DATASET_CONFIGS.items()
    }
    if not ADDITIONAL_DATASET_MANIFEST.exists():
        return configs

    payload = load_json(ADDITIONAL_DATASET_MANIFEST)
    for item in payload.get("datasets", []):
        dataset_id = str(item.get("dataset_id", "")).strip()
        label = str(item.get("label", "")).strip()
        working_dir = str(item.get("working_dir", "")).strip()
        custom_kg = str(item.get("custom_kg", "")).strip()
        if not all([dataset_id, label, working_dir, custom_kg]):
            continue
        configs[dataset_id] = {
            "label": label,
            "working_dir": Path(working_dir),
            "custom_kg": Path(custom_kg),
            "stats": item.get("stats") or {},
            "group": str(item.get("group", "")).strip(),
        }
    return configs


def normalize_scope_match_text(text: str) -> str:
    normalized = str(text or "").strip().lower()
    normalized = normalized.replace(" ", "").replace("疾病", "病")
    normalized = normalized.replace("慢性阻塞性肺疾病", "慢性阻塞性肺病").replace("慢阻肺病", "慢阻肺")
    return normalized


def resolve_auto_dataset_ids(
    dataset_configs: dict[str, dict[str, Any]],
    *,
    diagnosis_text: str,
    diagnosis_mapping: dict[str, Any] | None = None,
) -> list[str]:
    mapping = diagnosis_mapping or {}
    normalized_text = normalize_scope_match_text(mapping.get("normalized_text", ""))
    raw_text = normalize_scope_match_text(diagnosis_text)
    terms = {
        raw_text,
        normalized_text,
        normalize_scope_match_text(mapping.get("family", "")),
        normalize_scope_match_text(mapping.get("template", "")),
    }
    terms = {term for term in terms if term}
    if not terms:
        return []

    joined_terms = " ".join(sorted(terms))
    if any(keyword in joined_terms for keyword in ("慢阻肺", "copd")):
        if any(keyword in joined_terms for keyword in ("急性加重", "aecopd")):
            return [dataset_id for dataset_id in ("ae", "infection", "chronic") if dataset_id in dataset_configs]
        if "感染" in joined_terms:
            return [dataset_id for dataset_id in ("infection", "ae", "chronic") if dataset_id in dataset_configs]
        return [dataset_id for dataset_id in ("chronic",) if dataset_id in dataset_configs]

    # For pneumonia_hospital (HAP), the auto-scope frequently resolves to
    # only a "肺炎"-labelled dataset that contains exclusively CAP cases
    # and zero HAP cases.  Broaden to all non-COPD datasets so that the
    # search actually sees the ~2200 HAP candidates spread across other
    # non-COPD datasets.
    query_template = str(mapping.get("template", "")).strip()
    if query_template == "pneumonia_hospital":
        non_copd_ids: list[str] = []
        for dataset_id, config in dataset_configs.items():
            if str(config.get("group", "")).strip().lower() not in ("copd",):
                label_norm = normalize_scope_match_text(config.get("label", ""))
                if label_norm and ("肺炎" in label_norm or "感染" in label_norm or "呼吸" in label_norm):
                    non_copd_ids.append(dataset_id)
        if non_copd_ids:
            return non_copd_ids

    ranked_matches: list[tuple[tuple[int, int, int, int], str]] = []
    for dataset_id, config in dataset_configs.items():
        label_norm = normalize_scope_match_text(config.get("label", ""))
        if not label_norm:
            continue
        if any(term in label_norm or label_norm in term for term in terms):
            group = str(config.get("group", "")).strip().lower()
            exact_normalized_match = int(bool(normalized_text and label_norm == normalized_text))
            exact_raw_match = int(bool(raw_text and label_norm == raw_text))
            non_test_priority = 1 if group != "test_17" else 0
            specificity = len(label_norm)
            ranked_matches.append(
                (
                    (exact_normalized_match, exact_raw_match, non_test_priority, specificity),
                    dataset_id,
                )
            )
    ranked_matches.sort(key=lambda item: item[0], reverse=True)
    return [dataset_id for _, dataset_id in ranked_matches]


def json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_response(handler: BaseHTTPRequestHandler, text: str, status: int = 200):
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def clean_node_name(node_id: str) -> str:
    text = sanitize_graphml_text(node_id).strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text


def sanitize_graphml_text(value: Any) -> str:
    text = str(value or "")
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", text)


def normalize_graph_node_id(name: str) -> str:
    normalized = sanitize_graphml_text(name).upper().strip()
    return f'"{normalized}"'


def classify_node(node_id: str, attrs: dict[str, Any]) -> str:
    label = clean_node_name(node_id)
    if label.startswith("HYPEREDGE::"):
        return "hyperedge"
    entity_type = str(attrs.get("entity_type", "")).strip().lower()
    return entity_type if entity_type and entity_type != "unknown" else "entity"


def shorten_text(text: str, max_len: int = 160) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_len:
        return compact
    return f"{compact[: max_len - 3]}..."


def _flatten_group_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        rows: list[str] = []
        for key, item in value.items():
            nested = _flatten_group_value(item)
            if not nested:
                continue
            if len(nested) == 1 and not isinstance(item, (dict, list)):
                rows.append(f"{key}:{nested[0]}")
            else:
                rows.append(f"{key}: {'；'.join(nested[:12])}")
        return rows
    if isinstance(value, list):
        rows: list[str] = []
        for item in value[:20]:
            nested = _flatten_group_value(item)
            if nested:
                rows.append("；".join(nested[:12]))
        return rows
    text = normalize_text_value(value)
    return [text] if text else []


def build_query_chain_from_groups(bucket: str, raw_text: str, group_payloads: Any) -> str:
    parts: list[str] = [str(raw_text or "").strip()] if str(raw_text or "").strip() else []
    if not isinstance(group_payloads, list):
        return "\n".join(part for part in parts if part)
    for group in group_payloads:
        if not isinstance(group, dict):
            continue
        group_name = str(group.get("g") or group.get("字段分组") or "").strip()
        summary = normalize_text_value(group.get("s") or group.get("摘要"))
        flattened = _flatten_group_value(group.get("d") if "d" in group else group.get("提取结果"))
        if group_name and summary:
            parts.append(f"{group_name}: {summary}")
        elif summary:
            parts.append(summary)
        if flattened:
            parts.append(f"{group_name or bucket}明细: " + "；".join(flattened[:40]))
    return shorten_text("\n".join(part for part in parts if part), 6000)


def normalize_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        try:
            if math.isnan(value):
                return ""
        except Exception:
            pass
    text = str(value).strip()
    if text.lower() in {"", "null", "none", "nan", "n/a"}:
        return ""
    return text


def tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in TOKEN_RE.findall(text.lower()):
        tokens.add(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", token):
            for idx in range(len(token) - 1):
                tokens.add(token[idx : idx + 2])
    return tokens


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if not denom:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def extract_embedding_vectors(response: Any) -> list[list[float]]:
    if isinstance(response, dict):
        data = response.get("data")
        if isinstance(data, list) and data:
            vectors = []
            for item in data:
                if isinstance(item, dict) and "embedding" in item:
                    vectors.append(item["embedding"])
            if vectors:
                return vectors
    if isinstance(response, list):
        if response and isinstance(response[0], dict) and "embedding" in response[0]:
            return [item["embedding"] for item in response]
        if response and isinstance(response[0], (list, tuple)):
            return response
    raise ValueError(f"Unsupported embedding response format: {type(response)}")


def _xinference_rest_url(path: str) -> str:
    return f"{XINFERENCE_ENDPOINT.rstrip('/')}{path}"


def _xinference_rest_request(
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 300,
):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(_xinference_rest_url(path), data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")
        detail = body.strip() or str(exc)
        raise RuntimeError(f"Xinference REST {path} failed: HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"Xinference REST {path} failed: {exc}") from exc
    if not body.strip():
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


@lru_cache(maxsize=1)
def _xinference_rest_models() -> list[dict[str, Any]]:
    payload = _xinference_rest_request("/v1/models")
    rows = payload.get("data") if isinstance(payload, dict) else None
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _resolve_xinference_rest_model_uid(model_uid: str) -> str:
    candidates = xinference_model_uid_candidates(model_uid)
    for row in _xinference_rest_models():
        row_id = str(row.get("id", "")).strip()
        row_name = str(row.get("model_name", "")).strip()
        if row_id in candidates:
            return row_id
        if row_name in candidates:
            return row_id or row_name
    return str(model_uid or "").strip()


def create_embedding_via_rest(payload: Any, *, timeout: float = 300):
    model_uid = _resolve_xinference_rest_model_uid(XINFERENCE_EMBED_MODEL)
    return _xinference_rest_request(
        "/v1/embeddings",
        payload={
            "model": model_uid,
            "input": payload,
        },
        timeout=timeout,
    )


def rerank_via_rest(
    *,
    documents: list[str],
    query: str,
    top_n: int,
    return_documents: bool = False,
):
    model_uid = _resolve_xinference_rest_model_uid(XINFERENCE_RERANK_MODEL)
    return _xinference_rest_request(
        "/v1/rerank",
        payload={
            "model": model_uid,
            "documents": documents,
            "query": query,
            "top_n": top_n,
            "return_documents": return_documents,
        },
    )


def get_embed_model():
    if Client is None:
        raise RuntimeError("xinference is not installed in the current Python environment.")
    return get_xinference_model_handle(XINFERENCE_EMBED_MODEL)


def get_rerank_model():
    if Client is None:
        raise RuntimeError("xinference is not installed in the current Python environment.")
    return get_xinference_model_handle(XINFERENCE_RERANK_MODEL)


def xinference_model_uid_candidates(model_uid: str) -> list[str]:
    normalized = str(model_uid or "").strip()
    if not normalized:
        return []
    if normalized.endswith("-0"):
        return [normalized]
    return [normalized, f"{normalized}-0"]


def is_xinference_model_not_found_error(exc: Exception) -> bool:
    message = str(exc or "").strip().lower()
    return "model not found in the model list" in message


def get_xinference_model_handle(model_uid: str):
    client = Client(XINFERENCE_ENDPOINT)
    last_exc: Exception | None = None
    # #region debug-point A:model-handle-entry
    report_debug_event(
        hypothesis_id="A",
        location="dashboard_case_app.py:get_xinference_model_handle",
        msg="[DEBUG] resolving xinference model handle",
        data={
            "requested_uid": str(model_uid or ""),
            "candidates": xinference_model_uid_candidates(model_uid),
            "endpoint": XINFERENCE_ENDPOINT,
        },
    )
    # #endregion
    for candidate_uid in xinference_model_uid_candidates(model_uid):
        try:
            model = client.get_model(candidate_uid)
            # #region debug-point A:model-handle-success
            report_debug_event(
                hypothesis_id="A",
                location="dashboard_case_app.py:get_xinference_model_handle",
                msg="[DEBUG] xinference get_model succeeded",
                data={
                    "requested_uid": str(model_uid or ""),
                    "candidate_uid": candidate_uid,
                    "handle_type": type(model).__name__,
                },
            )
            # #endregion
            return model
        except Exception as exc:
            last_exc = exc
            # #region debug-point E:model-handle-error
            report_debug_event(
                hypothesis_id="E",
                location="dashboard_case_app.py:get_xinference_model_handle",
                msg="[DEBUG] xinference get_model failed",
                data={
                    "requested_uid": str(model_uid or ""),
                    "candidate_uid": candidate_uid,
                    "error": str(exc),
                    "not_found": is_xinference_model_not_found_error(exc),
                },
            )
            # #endregion
            if not is_xinference_model_not_found_error(exc):
                raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Xinference model uid is empty: {model_uid!r}")


def is_xinference_loading_error(exc: Exception) -> bool:
    message = str(exc or "").strip().lower()
    return "model is loading" in message or (
        "please retry later" in message and "loading" in message
    )


def xinference_call_with_retry(request_factory):
    last_exc: Exception | None = None
    for attempt in range(1, XINFERENCE_LOADING_RETRY_ATTEMPTS + 1):
        try:
            return request_factory()
        except Exception as exc:
            last_exc = exc
            if (
                not is_xinference_loading_error(exc)
                or attempt >= XINFERENCE_LOADING_RETRY_ATTEMPTS
            ):
                raise
            time.sleep(XINFERENCE_LOADING_RETRY_SLEEP_SECONDS)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Xinference request failed without exception details.")


def create_embedding_with_retry(payload: Any, *, timeout: float = 300):
    # #region debug-point B:embedding-request
    report_debug_event(
        hypothesis_id="B",
        location="dashboard_case_app.py:create_embedding_with_retry",
        msg="[DEBUG] create_embedding requested",
        data={
            "payload_type": type(payload).__name__,
            "payload_size": len(payload) if isinstance(payload, list) else 1,
            "sample_preview": (
                str(payload[0])[:160]
                if isinstance(payload, list) and payload
                else str(payload)[:160]
            ),
            "model_uid": XINFERENCE_EMBED_MODEL,
        },
    )
    # #endregion
    try:
        if Client is None:
            response = xinference_call_with_retry(lambda: create_embedding_via_rest(payload, timeout=timeout))
        else:
            response = xinference_call_with_retry(lambda: get_embed_model().create_embedding(payload))
        # #region debug-point C:embedding-success
        report_debug_event(
            hypothesis_id="C",
            location="dashboard_case_app.py:create_embedding_with_retry",
            msg="[DEBUG] create_embedding succeeded",
            data={
                "payload_type": type(payload).__name__,
                "payload_size": len(payload) if isinstance(payload, list) else 1,
                "response_type": type(response).__name__,
            },
        )
        # #endregion
        return response
    except Exception as exc:
        # #region debug-point D:embedding-error
        report_debug_event(
            hypothesis_id="D",
            location="dashboard_case_app.py:create_embedding_with_retry",
            msg="[DEBUG] create_embedding failed",
            data={
                "payload_type": type(payload).__name__,
                "payload_size": len(payload) if isinstance(payload, list) else 1,
                "error": str(exc),
            },
        )
        # #endregion
        raise


def rerank_with_retry(
    *,
    documents: list[str],
    query: str,
    top_n: int,
    return_documents: bool = False,
):
    if Client is None:
        return xinference_call_with_retry(
            lambda: rerank_via_rest(
                documents=documents,
                query=query,
                top_n=top_n,
                return_documents=return_documents,
            )
        )
    return xinference_call_with_retry(
        lambda: get_rerank_model().rerank(
            documents=documents,
            query=query,
            top_n=top_n,
            return_documents=return_documents,
        )
    )


def extract_rerank_rows(response: Any) -> list[Any]:
    if response is None:
        return []
    if isinstance(response, dict):
        rows = response.get("results")
        if isinstance(rows, list):
            return rows
    if hasattr(response, "results"):
        return list(getattr(response, "results"))
    if isinstance(response, list):
        return response
    raise ValueError(f"Unsupported rerank response format: {type(response)}")


def rerank_row_value(row: Any, key: str, default=None):
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def normalize_rerank_score(score: float) -> float:
    if 0.0 <= score <= 1.0:
        return float(score)
    if score >= 0:
        return 1.0 - math.exp(-score)
    return 1.0 / (1.0 + math.exp(-score))


POST_RANK_RESCUE_CONFIG: dict[str, dict[str, float]] = {
    # Keep the rescue small enough to preserve cross-disease exploration while
    # nudging obviously under-ranked critical templates back into the top slots.
    "resp_failure_type_1": {
        "same_template_bonus": 0.035,
        "shared_rule_bonus": 0.012,
        "rule_only_bonus": 0.008,
        "same_family_shared_rule_bonus": 0.005,
        "cross_family_no_rule_penalty": -0.018,
        "unmapped_no_rule_penalty": -0.012,
        "max_abs_delta": 0.05,
    },
    "ards": {
        "same_template_bonus": 0.04,
        "shared_rule_bonus": 0.015,
        "rule_only_bonus": 0.01,
        "same_family_shared_rule_bonus": 0.006,
        "cross_family_no_rule_penalty": -0.02,
        "unmapped_no_rule_penalty": -0.014,
        "max_abs_delta": 0.055,
    },
    "pneumonia_hospital": {
        "same_template_bonus": 0.07,
        "shared_rule_bonus": 0.015,
        "rule_only_bonus": 0.01,
        "same_family_shared_rule_bonus": 0.005,
        "cross_family_no_rule_penalty": -0.025,
        "unmapped_no_rule_penalty": -0.015,
        "max_abs_delta": 0.08,
    },
    "pe_acute": {
        "same_template_bonus": 0.025,
        "shared_rule_bonus": 0.01,
        "rule_only_bonus": 0.006,
        "same_family_shared_rule_bonus": 0.004,
        "cross_family_no_rule_penalty": -0.015,
        "unmapped_no_rule_penalty": -0.01,
        "max_abs_delta": 0.035,
    },
    "copd_infection": {
        "same_template_bonus": 0.03,
        "shared_rule_bonus": 0.012,
        "rule_only_bonus": 0.008,
        "same_family_shared_rule_bonus": 0.005,
        "cross_family_no_rule_penalty": -0.018,
        "unmapped_no_rule_penalty": -0.012,
        "max_abs_delta": 0.04,
    },
}


def build_rerank_document(case: dict[str, Any], per_chain_limit: int = 120) -> str:
    diagnosis_mapping = case.get("diagnosis_mapping") or {}
    normalized_diagnosis = (
        str(diagnosis_mapping.get("normalized_text", "")).strip() or case["diagnosis"]
    )
    core_chain_texts = case.get("core_chain_texts") or {}
    parts = [
        f"病例ID: {case['case_id']}",
        f"主诊断: {normalized_diagnosis}",
        f"性别: {case['gender']}",
        f"年龄: {case['age']}",
    ]
    for bucket, label in CHAIN_LABELS.items():
        bucket_budget = min(per_chain_limit, XINFERENCE_RERANK_DOC_CHAIN_CHAR_BUDGET)
        if bucket == "care_chain":
            bucket_budget = min(bucket_budget, 96)
        elif bucket == "severity_chain":
            bucket_budget = min(bucket_budget, 108)
        selected_text = sanitize_rerank_text(
            core_chain_texts.get(bucket, "") or case["chain_texts"].get(bucket, ""),
            bucket=bucket,
            max_len=bucket_budget,
        )
        if selected_text:
            parts.append(f"{label}: {selected_text}")
    return shorten_text("\n".join(parts), XINFERENCE_RERANK_DOC_MAX_CHARS)


def build_rerank_query_text(
    *,
    diagnosis: str,
    chain_texts: dict[str, str],
    core_chain_texts: dict[str, str] | None = None,
    extra_note: str = "",
    diagnosis_mapping: dict[str, Any] | None = None,
) -> str:
    core_chain_texts = core_chain_texts or {}
    diagnosis_mapping = diagnosis_mapping or {}
    normalized_diagnosis = str(diagnosis_mapping.get("normalized_text", "")).strip()
    parts = [f"主诊断: {normalized_diagnosis or diagnosis}" if (normalized_diagnosis or diagnosis) else ""]
    for bucket, label in CHAIN_LABELS.items():
        core_text = str(core_chain_texts.get(bucket, "")).strip()
        raw_text = str(chain_texts.get(bucket, "")).strip()
        bucket_budget = XINFERENCE_RERANK_QUERY_CHAIN_CHAR_BUDGET
        if bucket == "care_chain":
            bucket_budget = min(bucket_budget, 120)
        elif bucket == "severity_chain":
            bucket_budget = min(bucket_budget, 160)
        selected_text = sanitize_rerank_text(
            core_text or raw_text,
            bucket=bucket,
            max_len=bucket_budget,
        )
        if selected_text:
            parts.append(f"{label}: {selected_text}")
    if extra_note:
        parts.append(
            "补充说明: "
            + sanitize_rerank_text(
                extra_note,
                max_len=XINFERENCE_RERANK_QUERY_CHAIN_CHAR_BUDGET,
            )
        )
    return shorten_text(
        "\n".join(part for part in parts if part),
        XINFERENCE_RERANK_QUERY_MAX_CHARS,
    )


def parse_section_item(section: str | None, item: str):
    if section == "entities":
        match = re.match(r"\[(.*?)\]\s*(.*)", item)
        if match:
            return {"entity_type": match.group(1).strip(), "name": match.group(2).strip()}
        return {"entity_type": "UNKNOWN", "name": item.strip()}
    if section == "time_info":
        if "=" in item:
            key, value = item.split("=", 1)
            return {"key": key.strip(), "value": value.strip()}
        return {"key": "time", "value": item.strip()}
    return item.strip()


def parse_chunk_content(content: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {"entities": [], "time_info": [], "evidence_fragments": []}
    section: str | None = None
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "涉及实体:":
            section = "entities"
            continue
        if line == "时间信息:":
            section = "time_info"
            continue
        if line == "证据片段:":
            section = "evidence"
            continue
        if line.startswith("数据集目录:"):
            parsed["dataset"] = line.split(":", 1)[1].strip()
            section = None
            continue
        if line.startswith("病例ID:"):
            parsed["case_id"] = line.split(":", 1)[1].strip()
            section = None
            continue
        if line.startswith("超边ID:"):
            parsed["hyperedge_id"] = line.split(":", 1)[1].strip()
            section = None
            continue
        if line.startswith("关系类型:"):
            parsed["relation_type"] = line.split(":", 1)[1].strip()
            section = None
            continue
        if line.startswith("超边摘要:"):
            parsed["summary"] = line.split(":", 1)[1].strip()
            section = None
            continue
        if line.startswith("主诊断:"):
            parsed["diagnosis"] = line.split(":", 1)[1].strip()
            section = None
            continue
        if line.startswith("人口学:"):
            parsed["demographics"] = line.split(":", 1)[1].strip()
            section = None
            continue
        if line.startswith("- "):
            value = parse_section_item(section, line[2:].strip())
            if section == "entities":
                parsed["entities"].append(value)
            elif section == "time_info":
                parsed["time_info"].append(value)
            elif section == "evidence":
                parsed["evidence_fragments"].append(value)
    return parsed


def parse_demographics(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in text.split(";"):
        part = part.strip()
        if "=" in part:
            key, value = part.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def detect_chain_bucket(relation_type: str, evidence_fragments: list[str]) -> str:
    if relation_type in CHAIN_MAP:
        return CHAIN_MAP[relation_type]
    merged = "\n".join(evidence_fragments)
    if "病史背景" in merged:
        return "history_chain"
    if "临床证据链" in merged:
        return "evidence_chain"
    if "诊疗轨迹" in merged:
        return "care_chain"
    if "临床严重度" in merged:
        return "severity_chain"
    return "evidence_chain"


def ensure_graph_from_custom_kg(custom_kg_path: Path, graph_file: Path):
    if graph_file.exists():
        return
    graph_file.parent.mkdir(parents=True, exist_ok=True)
    data = load_json(custom_kg_path)
    graph = nx.Graph()

    for entity in data.get("entities", []):
        entity_name = normalize_graph_node_id(entity.get("entity_name", "UNKNOWN"))
        graph.add_node(
            entity_name,
            entity_type=sanitize_graphml_text(entity.get("entity_type", "UNKNOWN")),
            description=sanitize_graphml_text(entity.get("description", "No description provided")),
            source_id=sanitize_graphml_text(entity.get("source_id", "UNKNOWN")),
        )

    for relation in data.get("relationships", []):
        src_id = normalize_graph_node_id(relation.get("src_id", "UNKNOWN"))
        tgt_id = normalize_graph_node_id(relation.get("tgt_id", "UNKNOWN"))
        if not graph.has_node(src_id):
            graph.add_node(
                src_id,
                entity_type="UNKNOWN",
                description="UNKNOWN",
                source_id=sanitize_graphml_text(relation.get("source_id", "UNKNOWN")),
            )
        if not graph.has_node(tgt_id):
            graph.add_node(
                tgt_id,
                entity_type="UNKNOWN",
                description="UNKNOWN",
                source_id=sanitize_graphml_text(relation.get("source_id", "UNKNOWN")),
            )
        graph.add_edge(
            src_id,
            tgt_id,
            weight=float(relation.get("weight", 1.0)),
            description=sanitize_graphml_text(relation.get("description", "")),
            keywords=sanitize_graphml_text(relation.get("keywords", "")),
            source_id=sanitize_graphml_text(relation.get("source_id", "UNKNOWN")),
        )
    nx.write_graphml(graph, graph_file)


def build_case_index(custom_kg_path: Path) -> dict[str, Any]:
    data = load_json(custom_kg_path)
    cases: dict[str, dict[str, Any]] = {}
    hyperedges: list[dict[str, Any]] = []

    for chunk in data.get("chunks", []):
        parsed = parse_chunk_content(chunk.get("content", ""))
        case_id = parsed.get("case_id")
        if not case_id:
            continue
        bucket = detect_chain_bucket(parsed.get("relation_type", ""), parsed.get("evidence_fragments", []))
        demographics = parse_demographics(parsed.get("demographics", ""))
        hyperedge = {
            "case_id": case_id,
            "hyperedge_id": parsed.get("hyperedge_id", ""),
            "relation_type": parsed.get("relation_type", ""),
            "summary": parsed.get("summary", ""),
            "diagnosis": parsed.get("diagnosis", ""),
            "demographics": demographics,
            "entities": parsed.get("entities", []),
            "time_info": parsed.get("time_info", []),
            "evidence_fragments": parsed.get("evidence_fragments", []),
            "chain_bucket": bucket,
            "raw_content": chunk.get("content", ""),
        }
        hyperedges.append(hyperedge)

        case = cases.setdefault(
            case_id,
            {
                "case_id": case_id,
                "diagnosis": parsed.get("diagnosis", ""),
                "gender": demographics.get("性别", ""),
                "age": demographics.get("年龄", ""),
                "dataset": parsed.get("dataset", ""),
                "chains": {bucket_name: [] for bucket_name in CHAIN_LABELS},
                "entity_counter": Counter(),
                "relation_type_counts": Counter(),
            },
        )
        case["chains"][bucket].append(hyperedge)
        case["relation_type_counts"][hyperedge["relation_type"]] += 1
        for entity in hyperedge["entities"]:
            case["entity_counter"][entity["name"]] += 1

    rows = []
    for case in cases.values():
        case["diagnosis_mapping"] = build_diagnosis_mapping_payload(case["diagnosis"])
        case["top_entities"] = [
            {"name": name, "count": count}
            for name, count in case["entity_counter"].most_common(12)
        ]
        case["similarity_text"] = "\n".join(
            [
                f"病例ID: {case['case_id']}",
                f"主诊断: {case['diagnosis']}",
                f"性别: {case['gender']}",
                f"年龄: {case['age']}",
                *[
                    f"{CHAIN_LABELS[bucket]}: "
                    + " ".join(
                        f"{item['relation_type']} {item['summary']} {' '.join(item['evidence_fragments'][:4])}"
                        for item in case["chains"][bucket]
                    )
                    for bucket in CHAIN_LABELS
                ],
            ]
        )
        case["chain_texts"] = {
            bucket: " ".join(
                f"{item['relation_type']} {item['summary']} {' '.join(item['evidence_fragments'][:6])}"
                for item in case["chains"][bucket]
            )
            for bucket in CHAIN_LABELS
        }
        case["structured_profile"] = extract_structured_profile(
            diagnosis_text=case["diagnosis"],
            chain_texts=case["chain_texts"],
            diagnosis_mapping=case["diagnosis_mapping"],
        )
        case["effective_structured_profile"], case["denoise_report"] = apply_template_profile_denoise(
            case["structured_profile"],
            str(case["diagnosis_mapping"].get("template", "")).strip(),
        )
        case["core_chain_texts"] = build_semantic_core_chain_texts(
            case["effective_structured_profile"],
            case["diagnosis_mapping"],
        )
        case["chain_token_sets"] = {
            bucket: tokenize(case["chain_texts"][bucket]) for bucket in CHAIN_LABELS
        }
        case["core_chain_token_sets"] = {
            bucket: tokenize(case["core_chain_texts"][bucket]) for bucket in CHAIN_LABELS
        }
        case["token_set"] = tokenize(case["similarity_text"])
        case["relation_type_counts"] = dict(case["relation_type_counts"])
        case["chain_overview"] = {
            bucket: [
                {
                    "hyperedge_id": item["hyperedge_id"],
                    "relation_type": item["relation_type"],
                    "summary": item["summary"],
                    "evidence_preview": shorten_text(
                        "；".join(item["evidence_fragments"]) or item.get("raw_content", ""),
                        520,
                    ),
                    "evidence_full": "；".join(item["evidence_fragments"])
                    or item.get("raw_content", ""),
                }
                for item in case["chains"][bucket][:12]
            ]
            for bucket in CHAIN_LABELS
        }
        if case["case_id"] in DEBUG_SELF_MATCH_CASE_IDS:
            # #region debug-point A:self-case-index
            _debug_emit(
                "A",
                "build_case_index",
                "indexed self-match candidate case",
                {
                    "case_id": case["case_id"],
                    "diagnosis": case["diagnosis"],
                    "dataset": case["dataset"],
                    "bucket_hyperedge_counts": {
                        bucket: len(case["chains"][bucket]) for bucket in CHAIN_LABELS
                    },
                    "chain_text_lengths": {
                        bucket: len(str(case["chain_texts"].get(bucket, ""))) for bucket in CHAIN_LABELS
                    },
                    "core_chain_lengths": {
                        bucket: len(str(case["core_chain_texts"].get(bucket, ""))) for bucket in CHAIN_LABELS
                    },
                    "structured_profile": case["structured_profile"],
                    "effective_structured_profile": case["effective_structured_profile"],
                    "chain_text_preview": {
                        bucket: shorten_text(str(case["chain_texts"].get(bucket, "")), 240)
                        for bucket in CHAIN_LABELS
                    },
                    "core_chain_preview": {
                        bucket: shorten_text(str(case["core_chain_texts"].get(bucket, "")), 240)
                        for bucket in CHAIN_LABELS
                    },
                },
            )
            # #endregion
        rows.append(
            {
                "case_id": case["case_id"],
                "diagnosis": case["diagnosis"],
                "normalized_diagnosis": case["diagnosis_mapping"].get("normalized_text", ""),
                "diagnosis_concept_id": case["diagnosis_mapping"].get("concept_id", ""),
                "template_hint": case["diagnosis_mapping"].get("template", ""),
                "family_hint": case["diagnosis_mapping"].get("family", ""),
                "structured_summary": {
                    "symptom_count": len(case["structured_profile"].get("symptom_ids", [])),
                    "lab_count": len(case["structured_profile"].get("lab_ids", [])),
                    "blood_gas_count": len(case["structured_profile"].get("blood_gas_ids", [])),
                    "pulmonary_function_count": len(
                        case["structured_profile"].get("pulmonary_function_ids", [])
                    ),
                    "imaging_count": len(case["structured_profile"].get("imaging_ids", [])),
                    "drug_class_count": len(case["structured_profile"].get("drug_classes", [])),
                    "drug_intent_count": len(case["structured_profile"].get("drug_intents", [])),
                    "support_count": len(case["structured_profile"].get("support_ids", [])),
                    "lab_flag_count": len(case["structured_profile"].get("lab_flag_ids", [])),
                },
                "gender": case["gender"],
                "age": case["age"],
                "dataset": case["dataset"],
                "hyperedge_count": sum(len(case["chains"][bucket]) for bucket in CHAIN_LABELS),
                "history_count": len(case["chains"]["history_chain"]),
                "evidence_count": len(case["chains"]["evidence_chain"]),
                "care_count": len(case["chains"]["care_chain"]),
                "severity_count": len(case["chains"]["severity_chain"]),
                "top_entities": case["top_entities"][:6],
            }
        )
    rows.sort(key=lambda row: row["case_id"])
    hyperedges.sort(key=lambda row: (row["case_id"], row["hyperedge_id"]))
    return {"cases": cases, "case_rows": rows, "hyperedges": hyperedges}


def node_payload(graph: nx.Graph, node_id: str, x: float, y: float) -> dict[str, Any]:
    attrs = graph.nodes[node_id]
    kind = classify_node(node_id, attrs)
    return {
        "id": str(node_id),
        "display_name": clean_node_name(node_id),
        "label": clean_node_name(node_id).replace("HYPEREDGE::", ""),
        "kind": kind,
        "degree": int(graph.degree(node_id)),
        "x": float(x),
        "y": float(y),
        "data": {
            "entity_type": attrs.get("entity_type", ""),
            "description": attrs.get("description", ""),
            "source_id": attrs.get("source_id", ""),
        },
    }


def edge_payload(graph: nx.Graph, src: str, tgt: str) -> dict[str, Any]:
    attrs = graph.edges[(src, tgt)]
    return {
        "id": f"{src}__{tgt}",
        "source": str(src),
        "target": str(tgt),
        "label": shorten_text(attrs.get("keywords", ""), 24),
        "data": {
            "description": attrs.get("description", ""),
            "keywords": attrs.get("keywords", ""),
            "source_id": attrs.get("source_id", ""),
            "weight": attrs.get("weight", 1.0),
        },
    }


def build_graph_sample(graph: nx.Graph, max_nodes: int) -> dict[str, Any]:
    ranked_nodes = sorted(graph.degree, key=lambda item: (-item[1], str(item[0])))
    selected: set[str] = set()
    for seed, _ in ranked_nodes[:12]:
        selected.add(seed)
        neighbors = sorted(graph.neighbors(seed), key=lambda item: (-graph.degree(item), str(item)))
        for neighbor in neighbors:
            selected.add(neighbor)
            if len(selected) >= max_nodes:
                break
        if len(selected) >= max_nodes:
            break
    subgraph = graph.subgraph(selected).copy()
    layout = nx.spring_layout(subgraph, seed=42, k=1.2 / max(1, math.sqrt(max(1, subgraph.number_of_nodes()))))
    return {
        "nodes": [node_payload(subgraph, node_id, *layout[node_id]) for node_id in subgraph.nodes],
        "edges": [edge_payload(subgraph, src, tgt) for src, tgt in subgraph.edges],
        "center_node": "",
        "center_node_id": "",
        "stats": {
            "node_count": subgraph.number_of_nodes(),
            "edge_count": subgraph.number_of_edges(),
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "mode": "sample",
        },
    }


def build_graph_neighborhood(graph: nx.Graph, center: str, hops: int, max_nodes: int) -> dict[str, Any]:
    center_node = center if graph.has_node(center) else ""
    if not center_node:
        lower = center.lower()
        matches = [node for node in graph.nodes if lower in clean_node_name(node).lower()]
        center_node = matches[0] if matches else ""
    if not center_node:
        return {"nodes": [], "edges": [], "center_node": "", "center_node_id": ""}

    selected = {center_node}
    queue = deque([(center_node, 0)])
    while queue and len(selected) < max_nodes:
        node, depth = queue.popleft()
        if depth >= hops:
            continue
        neighbors = sorted(graph.neighbors(node), key=lambda item: (-graph.degree(item), str(item)))
        for neighbor in neighbors:
            if neighbor in selected:
                continue
            selected.add(neighbor)
            queue.append((neighbor, depth + 1))
            if len(selected) >= max_nodes:
                break

    subgraph = graph.subgraph(selected).copy()
    layout = nx.spring_layout(subgraph, seed=42, k=1.1 / max(1, math.sqrt(max(1, subgraph.number_of_nodes()))))
    return {
        "nodes": [node_payload(subgraph, node_id, *layout[node_id]) for node_id in subgraph.nodes],
        "edges": [edge_payload(subgraph, src, tgt) for src, tgt in subgraph.edges],
        "center_node": clean_node_name(center_node),
        "center_node_id": str(center_node),
        "stats": {
            "node_count": subgraph.number_of_nodes(),
            "edge_count": subgraph.number_of_edges(),
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "mode": "expand",
        },
    }


def search_nodes(graph: nx.Graph, query: str, limit: int = 20) -> list[dict[str, str]]:
    lower = query.lower().strip()
    if not lower:
        return []
    rows = []
    for node in graph.nodes:
        display = clean_node_name(node)
        if lower in display.lower():
            rows.append(
                {
                    "id": str(node),
                    "display_name": display,
                    "kind": classify_node(node, graph.nodes[node]),
                }
            )
    rows.sort(key=lambda row: (0 if row["kind"] == "hyperedge" else 1, len(row["display_name"])))
    return rows[:limit]


@dataclass
class SimilarityService:
    cases: dict[str, dict[str, Any]]
    cache_path: Path

    def __post_init__(self):
        self.lock = threading.Lock()
        self.case_vectors: dict[str, list[float]] | None = None
        self.chain_vectors: dict[str, dict[str, list[float]]] | None = None
        self.core_chain_vectors: dict[str, dict[str, list[float]]] | None = None
        self.status = "未初始化"

    def _load_cache(self):
        if not self.cache_path.exists():
            return None
        payload = load_json(self.cache_path)
        if payload.get("model_name") != XINFERENCE_EMBED_MODEL:
            return None
        vectors = payload.get("vectors", {})
        chain_vectors = payload.get("chain_vectors", {})
        core_chain_vectors = payload.get("core_chain_vectors", {})
        if set(vectors.keys()) != set(self.cases.keys()):
            return None
        if not chain_vectors:
            return {
                "vectors": vectors,
                "chain_vectors": {},
                "core_chain_vectors": {},
                "legacy_cache": True,
            }
        for bucket in CHAIN_LABELS:
            if set(chain_vectors.get(bucket, {}).keys()) != set(self.cases.keys()):
                return None
            if core_chain_vectors and set(core_chain_vectors.get(bucket, {}).keys()) != set(self.cases.keys()):
                return None
        return {
            "vectors": vectors,
            "chain_vectors": chain_vectors,
            "core_chain_vectors": core_chain_vectors,
            "legacy_cache": not bool(core_chain_vectors),
        }

    def _save_cache(
        self,
        vectors: dict[str, list[float]],
        chain_vectors: dict[str, dict[str, list[float]]],
        core_chain_vectors: dict[str, dict[str, list[float]]],
    ):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(
                {
                    "model_name": XINFERENCE_EMBED_MODEL,
                    "vectors": vectors,
                    "chain_vectors": chain_vectors,
                    "core_chain_vectors": core_chain_vectors,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _embed_texts(
        self,
        texts: list[str],
        batch_size: int = 8,
        *,
        request_timeout: float = 300,
        allow_batch_fallback_to_single: bool = True,
    ) -> list[list[float]]:
        embed_started_at = time.perf_counter()
        vectors: list[list[float]] = []
        normalized_texts = [
            shorten_text(str(text).strip() or "无内容", XINFERENCE_EMBED_TEXT_MAX_CHARS)
            for text in texts
        ]
        # #region debug-point C:embed-texts-entry
        _debug_emit(
            "C",
            "SimilarityService._embed_texts:entry",
            "embed_texts started",
            {
                "text_count": len(normalized_texts),
                "batch_size": batch_size,
                "sample_lengths": [len(text) for text in normalized_texts[:3]],
            },
        )
        # #endregion
        for start in range(0, len(normalized_texts), batch_size):
            batch = normalized_texts[start : start + batch_size]
            try:
                response = create_embedding_with_retry(batch, timeout=request_timeout)
                vectors.extend(extract_embedding_vectors(response))
                continue
            except Exception:
                if not allow_batch_fallback_to_single:
                    raise
            for text in batch:
                response = create_embedding_with_retry(text, timeout=request_timeout)
                vectors.extend(extract_embedding_vectors(response))
        # #region debug-point C:embed-texts-return
        _debug_emit(
            "C",
            "SimilarityService._embed_texts:return",
            "embed_texts finished",
            {
                "elapsed_ms": round((time.perf_counter() - embed_started_at) * 1000, 2),
                "text_count": len(normalized_texts),
                "vector_count": len(vectors),
            },
        )
        # #endregion
        return vectors

    def _ensure_vectors(self):
        with self.lock:
            ensure_started_at = time.perf_counter()
            # #region debug-point C:ensure-vectors-entry
            _debug_emit(
                "C",
                "SimilarityService._ensure_vectors:entry",
                "ensure_vectors started",
                {
                    "cache_path": str(self.cache_path),
                    "case_count": len(self.cases),
                    "has_case_vectors": bool(self.case_vectors),
                },
            )
            # #endregion
            if self.case_vectors:
                # #region debug-point C:ensure-vectors-hit-memory
                _debug_emit(
                    "C",
                    "SimilarityService._ensure_vectors:memory-hit",
                    "ensure_vectors reused in-memory vectors",
                    {
                        "elapsed_ms": round((time.perf_counter() - ensure_started_at) * 1000, 2),
                        "status": self.status,
                    },
                )
                # #endregion
                return
            cached = self._load_cache()
            if cached is not None:
                self.case_vectors = cached["vectors"]
                self.chain_vectors = cached["chain_vectors"]
                self.core_chain_vectors = cached.get("core_chain_vectors") or {}
                self.status = (
                    "已从旧版缓存加载，核心链向量缺失时回退原始语义"
                    if cached.get("legacy_cache")
                    else "已从缓存加载 embedding 与核心链向量"
                )
                # #region debug-point C:ensure-vectors-cache-hit
                _debug_emit(
                    "C",
                    "SimilarityService._ensure_vectors:cache-hit",
                    "ensure_vectors loaded cache",
                    {
                        "elapsed_ms": round((time.perf_counter() - ensure_started_at) * 1000, 2),
                        "legacy_cache": bool(cached.get("legacy_cache")),
                        "status": self.status,
                    },
                )
                # #endregion
                return
            try:
                case_ids = list(self.cases.keys())
                # #region debug-point C:ensure-vectors-rebuild
                _debug_emit(
                    "C",
                    "SimilarityService._ensure_vectors:rebuild",
                    "ensure_vectors rebuilding cache",
                    {
                        "case_count": len(case_ids),
                        "bucket_count": len(CHAIN_LABELS),
                    },
                )
                # #endregion
                similarity_texts = [self.cases[case_id]["similarity_text"] for case_id in case_ids]
                case_vectors = self._embed_texts(similarity_texts)
                chain_vectors = {bucket: {} for bucket in CHAIN_LABELS}
                core_chain_vectors = {bucket: {} for bucket in CHAIN_LABELS}
                for bucket in CHAIN_LABELS:
                    bucket_texts = [
                        self.cases[case_id]["chain_texts"].get(bucket)
                        or self.cases[case_id]["similarity_text"]
                        for case_id in case_ids
                    ]
                    bucket_vectors = self._embed_texts(bucket_texts)
                    for case_id, vector in zip(case_ids, bucket_vectors):
                        chain_vectors[bucket][case_id] = vector
                    core_bucket_texts = [
                        self.cases[case_id].get("core_chain_texts", {}).get(bucket) or "无内容"
                        for case_id in case_ids
                    ]
                    core_bucket_vectors = self._embed_texts(core_bucket_texts)
                    for case_id, vector in zip(case_ids, core_bucket_vectors):
                        core_chain_vectors[bucket][case_id] = vector
                self.case_vectors = {
                    case_id: vector for case_id, vector in zip(case_ids, case_vectors)
                }
                self.chain_vectors = chain_vectors
                self.core_chain_vectors = core_chain_vectors
                self._save_cache(self.case_vectors, self.chain_vectors, self.core_chain_vectors)
                self.status = f"已使用 {XINFERENCE_EMBED_MODEL} 重建 embedding 与核心链缓存"
                # #region debug-point C:ensure-vectors-rebuild-finished
                _debug_emit(
                    "C",
                    "SimilarityService._ensure_vectors:rebuild-finished",
                    "ensure_vectors rebuilt cache",
                    {
                        "elapsed_ms": round((time.perf_counter() - ensure_started_at) * 1000, 2),
                        "status": self.status,
                    },
                )
                # #endregion
                return
            except Exception as exc:
                # Keep the service retryable after transient Xinference loading/OOM failures.
                self.case_vectors = None
                self.chain_vectors = None
                self.core_chain_vectors = None
                self.status = f"embedding 构建失败，已回退词项相似度: {exc}"
                # #region debug-point C:ensure-vectors-error
                _debug_emit(
                    "C",
                    "SimilarityService._ensure_vectors:error",
                    "ensure_vectors failed",
                    {
                        "elapsed_ms": round((time.perf_counter() - ensure_started_at) * 1000, 2),
                        "error": str(exc),
                        "status": self.status,
                    },
                )
                # #endregion
                return

    def _top_overlap_terms(self, query_text: str, case_text: str, limit: int = 6) -> list[str]:
        query_terms = Counter(TOKEN_RE.findall(query_text.lower()))
        case_terms = Counter(TOKEN_RE.findall(case_text.lower()))
        return [token for token, _ in (query_terms & case_terms).most_common(limit)][:limit]

    def _apply_rerank(
        self,
        query_text: str,
        top_k: int,
        query_diagnosis_mapping: dict[str, Any],
        scored: list[
            tuple[
                float,
                dict[str, Any],
                list[str],
                dict[str, dict[str, float]],
                float,
                dict[str, Any],
            ]
        ],
    ) -> tuple[
        list[
            tuple[
                float,
                dict[str, Any],
                list[str],
                dict[str, dict[str, float]],
                float,
                dict[str, Any],
                float,
                float,
            ]
        ],
        str,
        bool,
    ]:
        scored_with_rerank = [
            (score, case, reasons, chain_scores, diagnosis_bonus, score_breakdown, 0.0, score)
            for score, case, reasons, chain_scores, diagnosis_bonus, score_breakdown in scored
        ]
        if not query_text.strip() or not scored_with_rerank:
            return scored_with_rerank, "", False
        query_template = str(query_diagnosis_mapping.get("template", "")).strip()
        query_family = str(query_diagnosis_mapping.get("family", "")).strip()
        pool_limit = min(len(scored_with_rerank), max(top_k, RERANK_CANDIDATE_POOL_LIMIT))
        candidate_pool = scored_with_rerank[:pool_limit]
        gated_candidates = []
        top_score = float(candidate_pool[0][0]) if candidate_pool else 0.0
        for row in candidate_pool:
            score, case, _, _, _, score_breakdown, _, _ = row
            case_mapping = case.get("diagnosis_mapping") or {}
            case_template = str(case_mapping.get("template", "")).strip()
            case_family = str(case_mapping.get("family", "")).strip()
            rule_match = score_breakdown.get("rule_match") or {}
            shared_rule_ids = list(rule_match.get("shared_rule_ids") or [])
            structured_score = float(score_breakdown.get("structured_score", 0.0) or 0.0)
            rule_score = float(score_breakdown.get("guideline_rule_score", 0.0) or 0.0)
            semantic_score = float(score_breakdown.get("semantic_score", 0.0) or 0.0)
            disease_score = float(score_breakdown.get("disease_consistency_score", 0.0) or 0.0)
            hierarchy_score = float(score_breakdown.get("diagnosis_hierarchy_score", 0.0) or 0.0)
            same_template = bool(query_template and case_template and query_template == case_template)
            same_family = bool(query_family and case_family and query_family == case_family)
            clinically_supported = any(
                (
                    same_template,
                    same_family,
                    bool(shared_rule_ids),
                    rule_score >= 0.08,
                    structured_score >= 0.28,
                    disease_score >= 0.72,
                    hierarchy_score >= 0.55,
                )
            )
            low_value_cross_family = (
                bool(query_family and case_family and query_family != case_family)
                and not shared_rule_ids
                and rule_score < 0.05
                and structured_score < 0.18
                and disease_score < 0.5
                and semantic_score <= max(structured_score, 0.32)
            )
            near_front = float(score) >= max(0.2, top_score - RERANK_CANDIDATE_GAP_THRESHOLD)
            if clinically_supported and not low_value_cross_family:
                gated_candidates.append(row)
            elif near_front and len(gated_candidates) < max(top_k, 2):
                gated_candidates.append(row)
        if len(gated_candidates) < min(len(candidate_pool), max(top_k, 3)):
            selected_ids = {item[1]["case_id"] for item in gated_candidates}
            for row in candidate_pool:
                if row[1]["case_id"] in selected_ids:
                    continue
                gated_candidates.append(row)
                selected_ids.add(row[1]["case_id"])
                if len(gated_candidates) >= min(len(candidate_pool), max(top_k, 3)):
                    break

        initial_candidate_count = min(len(gated_candidates), max(top_k, 3))
        candidate_count_plan = []
        for size in (
            initial_candidate_count,
            min(initial_candidate_count, max(top_k - 1, 2)),
            1,
        ):
            if size > 0 and size not in candidate_count_plan:
                candidate_count_plan.append(size)

        rerank_rows = None
        candidates = []
        last_error = ""
        for candidate_count in candidate_count_plan:
            candidate_slice = gated_candidates[:candidate_count]
            for per_chain_limit in (120, 80, 40):
                documents = [
                    build_rerank_document(case, per_chain_limit=per_chain_limit)
                    for _, case, _, _, _, _, _, _ in candidate_slice
                ]
                try:
                    rerank_response = rerank_with_retry(
                        documents=documents,
                        query=query_text,
                        top_n=candidate_count,
                        return_documents=False,
                    )
                    rerank_rows = extract_rerank_rows(rerank_response)
                    candidates = candidate_slice
                    break
                except Exception as exc:
                    last_error = str(exc)
                    # Candidate count and document truncation cannot recover a model
                    # that is still warming up; exit early to avoid multiplying wait time.
                    if is_xinference_loading_error(exc):
                        return scored_with_rerank, last_error, False
            if rerank_rows is not None:
                break

        if rerank_rows is None:
            return scored_with_rerank, last_error, False

        rerank_scores: dict[str, float] = {}
        for row in rerank_rows:
            index = rerank_row_value(row, "index")
            relevance_score = rerank_row_value(row, "relevance_score", 0.0)
            if index is None or not (0 <= int(index) < len(candidates)):
                continue
            _, case, _, _, _, _, _, _ = candidates[int(index)]
            rerank_scores[case["case_id"]] = normalize_rerank_score(float(relevance_score))

        rescored = []
        for score, case, reasons, chain_scores, diagnosis_bonus, score_breakdown, _, _ in scored_with_rerank:
            rerank_score = rerank_scores.get(case["case_id"], 0.0)
            final_score = score
            if case["case_id"] in rerank_scores:
                final_score = min(1.0, 0.7 * score + 0.3 * rerank_score)
            rescored.append(
                (
                    final_score,
                    case,
                    reasons,
                    chain_scores,
                    diagnosis_bonus,
                    score_breakdown,
                    rerank_score,
                    score,
                )
            )
        rescored.sort(key=lambda item: item[0], reverse=True)
        return rescored, "", bool(rerank_scores)

    def _apply_post_rank_adjustment(
        self,
        query_diagnosis_mapping: dict[str, Any],
        scored_rows: list[
            tuple[
                float,
                dict[str, Any],
                list[str],
                dict[str, dict[str, float]],
                float,
                dict[str, Any],
                float,
                float,
            ]
        ],
        *,
        stage: str = "post_rerank",
    ) -> list[
        tuple[
            float,
            dict[str, Any],
            list[str],
            dict[str, dict[str, float]],
            float,
            dict[str, Any],
            float,
            float,
        ]
    ]:
        adjustment_key = (
            "pre_rerank_adjustment" if stage == "pre_rerank" else "post_rank_adjustment"
        )
        query_template = str(query_diagnosis_mapping.get("template", "")).strip()
        query_family = str(query_diagnosis_mapping.get("family", "")).strip()
        config = POST_RANK_RESCUE_CONFIG.get(query_template)
        if not config or not scored_rows:
            return scored_rows

        adjusted_rows = []
        for row in scored_rows:
            # Support both 6-tuple (pre-rerank) and 8-tuple (post-rerank)
            if len(row) >= 8:
                final_score, case, reasons, chain_scores, diagnosis_bonus, score_breakdown, rerank_score, base_score = row
            else:
                # Pre-rerank 6-tuple: (final_score, case, reasons, chain_scores, diagnosis_bonus, score_breakdown)
                final_score, case, reasons, chain_scores, diagnosis_bonus, score_breakdown = row
                rerank_score = None
                base_score = final_score
            breakdown = dict(score_breakdown or {})
            case_mapping = case.get("diagnosis_mapping") or {}
            case_template = str(case_mapping.get("template", "")).strip()
            case_family = str(case_mapping.get("family", "")).strip()
            rule_match = breakdown.get("rule_match") or {}
            shared_rule_ids = list(rule_match.get("shared_rule_ids") or [])
            rule_score = float(breakdown.get("guideline_rule_score", 0.0) or 0.0)
            semantic_score = float(breakdown.get("semantic_score", 0.0) or 0.0)
            structured_score = float(breakdown.get("structured_score", 0.0) or 0.0)

            delta = 0.0
            adjustment_reasons: list[str] = []

            if case_template and case_template == query_template:
                delta += float(config.get("same_template_bonus", 0.0))
                adjustment_reasons.append("候选模板与查询模板一致")
                if shared_rule_ids:
                    delta += float(config.get("shared_rule_bonus", 0.0))
                    adjustment_reasons.append("共享目标模板规则命中")
                elif rule_score > 0:
                    delta += float(config.get("rule_only_bonus", 0.0))
                    adjustment_reasons.append("候选存在目标模板规则证据")
            elif case_family and query_family and case_family == query_family and shared_rule_ids:
                delta += float(config.get("same_family_shared_rule_bonus", 0.0))
                adjustment_reasons.append("同家族且共享目标模板规则")
            elif not shared_rule_ids and math.isclose(rule_score, 0.0, abs_tol=1e-9):
                if not case_template and semantic_score >= structured_score:
                    delta += float(config.get("unmapped_no_rule_penalty", 0.0))
                    adjustment_reasons.append("未映射模板且仅语义相似")
                elif case_family and query_family and case_family != query_family and semantic_score >= structured_score:
                    delta += float(config.get("cross_family_no_rule_penalty", 0.0))
                    adjustment_reasons.append("跨家族且缺少目标模板规则")

            max_abs_delta = float(config.get("max_abs_delta", 0.05) or 0.05)
            delta = max(-max_abs_delta, min(max_abs_delta, delta))
            adjusted_score = min(1.0, max(0.0, final_score + delta))
            breakdown[adjustment_key] = {
                "applied": bool(adjustment_reasons and not math.isclose(delta, 0.0, abs_tol=1e-9)),
                "target_template": query_template,
                "delta": round(delta, 6),
                "reasons": adjustment_reasons,
                "score_before_adjustment": round(float(final_score), 6),
                "score_after_adjustment": round(float(adjusted_score), 6),
            }
            if len(row) >= 8:
                adjusted_rows.append(
                    (
                        adjusted_score,
                        case,
                        reasons,
                        chain_scores,
                        diagnosis_bonus,
                        breakdown,
                        rerank_score,
                        base_score,
                    )
                )
            else:
                # Preserve 6-tuple format for pre-rerank consumption
                adjusted_rows.append(
                    (
                        adjusted_score,
                        case,
                        reasons,
                        chain_scores,
                        diagnosis_bonus,
                        breakdown,
                    )
                )

        adjusted_rows.sort(key=lambda item: item[0], reverse=True)
        return adjusted_rows

    def search(self, payload: dict[str, Any], top_k: int) -> dict[str, Any]:
        trace_id = uuid.uuid4().hex[:12]
        search_started_at = time.perf_counter()
        debug_query_case_id = str(payload.get("query_case_id", "")).strip()
        diagnosis = str(payload.get("diagnosis", "")).strip()
        history_chain = str(payload.get("history_chain", "")).strip()
        evidence_chain = str(payload.get("evidence_chain", "")).strip()
        care_chain = str(payload.get("care_chain", "")).strip()
        severity_chain = str(payload.get("severity_chain", "")).strip()
        extra_note = str(payload.get("extra_note", "")).strip()
        query_diagnosis_mapping = build_diagnosis_mapping_payload(diagnosis)
        query_text = "\n".join(
            part
            for part in [
                f"主诊断: {diagnosis}" if diagnosis else "",
                f"病史链: {history_chain}" if history_chain else "",
                f"证据链: {evidence_chain}" if evidence_chain else "",
                f"诊疗链: {care_chain}" if care_chain else "",
                f"严重度链: {severity_chain}" if severity_chain else "",
                f"补充说明: {extra_note}" if extra_note else "",
            ]
            if part
        )
        chain_queries = {
            "history_chain": history_chain,
            "evidence_chain": evidence_chain,
            "care_chain": care_chain,
            "severity_chain": severity_chain,
        }
        for bucket in CHAIN_LABELS:
            group_payloads = payload.get(bucket.replace("_chain", "_groups"))
            if group_payloads:
                chain_queries[bucket] = build_query_chain_from_groups(
                    bucket,
                    chain_queries[bucket],
                    group_payloads,
                )
        query_structured_profile = extract_structured_profile(
            diagnosis_text=diagnosis,
            chain_texts=chain_queries,
            diagnosis_mapping=query_diagnosis_mapping,
        )
        query_effective_structured_profile, base_query_denoise_report = apply_template_profile_denoise(
            query_structured_profile,
            str(query_diagnosis_mapping.get("template", "")).strip(),
        )
        query_core_chain_texts = build_semantic_core_chain_texts(
            query_effective_structured_profile,
            query_diagnosis_mapping,
        )
        if debug_query_case_id:
            # #region debug-point C:self-query-profile
            _debug_emit(
                "C",
                "SimilarityService.search:query-profile",
                "captured self-match query profile",
                {
                    "traceId": trace_id,
                    "query_case_id": debug_query_case_id,
                    "diagnosis": diagnosis,
                    "query_chain_lengths": {
                        bucket: len(str(chain_queries.get(bucket, ""))) for bucket in CHAIN_LABELS
                    },
                    "query_chain_preview": {
                        bucket: shorten_text(str(chain_queries.get(bucket, "")), 240)
                        for bucket in CHAIN_LABELS
                    },
                    "query_structured_profile": query_structured_profile,
                    "query_effective_structured_profile": query_effective_structured_profile,
                    "query_core_chain_texts": query_core_chain_texts,
                },
            )
            # #endregion
        rerank_query_text = build_rerank_query_text(
            diagnosis=diagnosis,
            chain_texts=chain_queries,
            core_chain_texts=query_core_chain_texts,
            extra_note=extra_note,
            diagnosis_mapping=query_diagnosis_mapping,
        )
        active_query_buckets = [bucket for bucket, text in chain_queries.items() if text]
        semantic_weight_plan = build_semantic_weight_plan(
            query_mapping=query_diagnosis_mapping,
            active_buckets=active_query_buckets,
        )
        query_tokens_map = {bucket: tokenize(text) for bucket, text in chain_queries.items() if text}
        query_core_tokens_map = {
            bucket: tokenize(text)
            for bucket, text in query_core_chain_texts.items()
            if str(text).strip()
        }
        query_vec_map: dict[str, np.ndarray] = {}
        query_core_vec_map: dict[str, np.ndarray] = {}
        query_embed_cache_key = _build_query_embed_cache_key(chain_queries, query_core_chain_texts)
        embedding_error = ""
        # #region debug-point B:similarity-search-entry
        _debug_emit(
            "B",
            "SimilarityService.search:entry",
            "enter similarity search",
            {
                "traceId": trace_id,
                "diagnosis": diagnosis,
                "top_k": top_k,
                "case_count": len(self.cases),
                "active_query_buckets": active_query_buckets,
                "query_text_len": len(query_text),
                "rerank_query_len": len(rerank_query_text),
            },
        )
        # #endregion
        embedding_started_at = time.perf_counter()
        cached_query_vectors = _get_query_embed_cache(query_embed_cache_key)
        if cached_query_vectors is not None:
            query_vec_map = {
                bucket: np.array(vector, dtype=np.float32)
                for bucket, vector in (cached_query_vectors.get("query_vecs") or {}).items()
            }
            query_core_vec_map = {
                bucket: np.array(vector, dtype=np.float32)
                for bucket, vector in (cached_query_vectors.get("query_core_vecs") or {}).items()
            }
        elif _get_query_embed_failure(query_embed_cache_key):
            embedding_error = "query embedding skipped after recent timeout"
        else:
            try:
                self._ensure_vectors()
                query_vector_targets: list[tuple[str, bool, bool]] = []
                for bucket in CHAIN_LABELS:
                    raw_text = str(chain_queries.get(bucket, "")).strip()
                    core_text = str(query_core_chain_texts.get(bucket, "")).strip()
                    has_chain_vectors = (
                        bool(raw_text)
                        and self.chain_vectors is not None
                        and bucket in self.chain_vectors
                        and bool(self.chain_vectors[bucket])
                    )
                    has_core_vectors = (
                        bool(core_text)
                        and self.core_chain_vectors is not None
                        and bucket in self.core_chain_vectors
                        and bool(self.core_chain_vectors[bucket])
                    )
                    if has_chain_vectors or has_core_vectors:
                        query_vector_targets.append((bucket, has_chain_vectors, has_core_vectors))
                if query_vector_targets:
                    if QUERY_EMBED_MODE == "bucket":
                        query_embedding_items = []
                        for bucket, has_chain_vectors, has_core_vectors in query_vector_targets:
                            raw_text = str(chain_queries.get(bucket, "")).strip()
                            core_text = str(query_core_chain_texts.get(bucket, "")).strip()
                            semantic_parts = []
                            if core_text:
                                semantic_parts.append(core_text)
                            if raw_text and raw_text != core_text:
                                semantic_parts.append(raw_text)
                            semantic_text = "\n".join(part for part in semantic_parts if part) or raw_text or core_text
                            if semantic_text:
                                query_embedding_items.append(
                                    (bucket, semantic_text, has_chain_vectors, has_core_vectors)
                                )
                        query_vectors = self._embed_texts(
                            [text for _, text, _, _ in query_embedding_items],
                            batch_size=min(QUERY_EMBED_BATCH_SIZE, len(query_embedding_items)),
                            request_timeout=QUERY_EMBED_REQUEST_TIMEOUT_SECONDS,
                            allow_batch_fallback_to_single=False,
                        )
                        for (bucket, _, has_chain_vectors, has_core_vectors), vector in zip(
                            query_embedding_items,
                            query_vectors,
                        ):
                            vector_array = np.array(vector, dtype=np.float32)
                            if has_chain_vectors:
                                query_vec_map[bucket] = vector_array
                            if has_core_vectors:
                                query_core_vec_map[bucket] = vector_array
                    else:
                        summary_query_text = (
                            rerank_query_text
                            or query_text
                            or "\n".join(
                                str(query_core_chain_texts.get(bucket, "")).strip()
                                for bucket, _, _ in query_vector_targets
                                if str(query_core_chain_texts.get(bucket, "")).strip()
                            )
                        )
                        shared_query_vector = np.array(
                            self._embed_texts(
                                [summary_query_text],
                                batch_size=1,
                                request_timeout=QUERY_EMBED_REQUEST_TIMEOUT_SECONDS,
                                allow_batch_fallback_to_single=False,
                            )[0],
                            dtype=np.float32,
                        )
                        for bucket, has_chain_vectors, has_core_vectors in query_vector_targets:
                            if has_chain_vectors:
                                query_vec_map[bucket] = shared_query_vector
                            if has_core_vectors:
                                query_core_vec_map[bucket] = shared_query_vector
                if query_vec_map or query_core_vec_map:
                    _put_query_embed_cache(
                        query_embed_cache_key,
                        query_vecs=query_vec_map,
                        query_core_vecs=query_core_vec_map,
                    )
            except Exception as exc:
                embedding_error = str(exc)
                _put_query_embed_failure(query_embed_cache_key)
        # #region debug-point C:similarity-search-embedding
        _debug_emit(
            "C",
            "SimilarityService.search:embedding",
            "embedding stage finished",
            {
                "traceId": trace_id,
                "elapsed_ms": round((time.perf_counter() - embedding_started_at) * 1000, 2),
                "embedding_error": embedding_error,
                "query_vec_buckets": sorted(query_vec_map.keys()),
                "query_core_vec_buckets": sorted(query_core_vec_map.keys()),
                "status": self.status,
            },
        )
        # #endregion

        scored = []
        scoring_started_at = time.perf_counter()
        for case_id, case in self.cases.items():
            chain_scores: dict[str, dict[str, float]] = {}
            case_profile = case.get("structured_profile") or extract_structured_profile(
                diagnosis_text=case["diagnosis"],
                chain_texts=case.get("chain_texts", {}),
                diagnosis_mapping=case.get("diagnosis_mapping") or {},
            )
            _, active_template_id = get_template_for_scoring(
                query_mapping=query_diagnosis_mapping,
                case_mapping=case.get("diagnosis_mapping") or {},
            )
            effective_query_profile, query_denoise_report = apply_template_profile_denoise(
                query_structured_profile,
                active_template_id,
            )
            effective_case_profile, case_denoise_report = apply_template_profile_denoise(
                case_profile,
                active_template_id,
            )
            for bucket in CHAIN_LABELS:
                query_text_bucket = chain_queries[bucket]
                if not query_text_bucket:
                    continue
                lexical = 0.0
                bucket_tokens = query_tokens_map.get(bucket, set())
                case_tokens = case["chain_token_sets"].get(bucket, set())
                if bucket_tokens and case_tokens:
                    lexical = len(bucket_tokens & case_tokens) / math.sqrt(len(bucket_tokens) * len(case_tokens))
                core_lexical = 0.0
                query_core_tokens = query_core_tokens_map.get(bucket, set())
                case_core_tokens = case.get("core_chain_token_sets", {}).get(bucket, set())
                if query_core_tokens and case_core_tokens:
                    core_lexical = len(query_core_tokens & case_core_tokens) / math.sqrt(
                        len(query_core_tokens) * len(case_core_tokens)
                    )
                embed_score = 0.0
                core_embed_score = 0.0
                if (
                    bucket in query_vec_map
                    and self.chain_vectors is not None
                    and bucket in self.chain_vectors
                    and case_id in self.chain_vectors[bucket]
                ):
                    case_vec = np.array(self.chain_vectors[bucket][case_id], dtype=np.float32)
                    embed_score = max(0.0, cosine_similarity(query_vec_map[bucket], case_vec))
                if (
                    bucket in query_core_vec_map
                    and self.core_chain_vectors is not None
                    and bucket in self.core_chain_vectors
                    and case_id in self.core_chain_vectors[bucket]
                ):
                    case_core_vec = np.array(self.core_chain_vectors[bucket][case_id], dtype=np.float32)
                    core_embed_score = max(
                        0.0,
                        cosine_similarity(query_core_vec_map[bucket], case_core_vec),
                    )

                fused_embedding = 0.0
                if embed_score and core_embed_score:
                    fused_embedding = 0.65 * embed_score + 0.35 * core_embed_score
                elif core_embed_score:
                    fused_embedding = core_embed_score
                else:
                    fused_embedding = embed_score

                fused_lexical = lexical
                if core_lexical:
                    fused_lexical = 0.75 * lexical + 0.25 * core_lexical

                if fused_embedding:
                    final_bucket_score = 0.72 * fused_embedding + 0.28 * fused_lexical
                else:
                    final_bucket_score = fused_lexical
                chain_scores[bucket] = {
                    "final": round(final_bucket_score, 6),
                    "embedding": round(embed_score, 6),
                    "core_embedding": round(core_embed_score, 6),
                    "embedding_fused": round(fused_embedding, 6),
                    "lexical": round(lexical, 6),
                    "core_lexical": round(core_lexical, 6),
                    "lexical_fused": round(fused_lexical, 6),
                    "weight": round(
                        float(semantic_weight_plan["normalized_weights"].get(bucket, 0.0)),
                        6,
                    ),
                }
            semantic_score = sum(
                chain_scores[bucket]["final"] * float(semantic_weight_plan["normalized_weights"].get(bucket, 0.0))
                for bucket in chain_scores
            )
            diagnosis_alignment = compute_diagnosis_alignment(
                diagnosis_text=diagnosis,
                query_mapping=query_diagnosis_mapping,
                case_mapping=case.get("diagnosis_mapping") or {},
                case_similarity_text=case["similarity_text"],
                query_profile=effective_query_profile,
                case_profile=effective_case_profile,
            )
            structured_match = build_structured_match_score(
                query_profile=effective_query_profile,
                case_profile=effective_case_profile,
                query_mapping=query_diagnosis_mapping,
                case_mapping=case.get("diagnosis_mapping") or {},
            )
            rule_match = build_rule_match_score(
                template_id=structured_match.get("template", ""),
                query_profile=effective_query_profile,
                case_profile=effective_case_profile,
            )
            fusion_components = {
                "semantic_score": semantic_score,
                "structured_score": float(structured_match.get("score", 0.0)),
                "guideline_rule_score": float(rule_match.get("score", 0.0)),
                "disease_consistency_score": float(
                    diagnosis_alignment.get("disease_consistency_score", 0.0)
                ),
                "diagnosis_hierarchy_score": float(
                    diagnosis_alignment.get("diagnosis_hierarchy_score", 0.0)
                ),
            }
            fusion_availability = {
                "semantic_score": bool(active_query_buckets),
                "structured_score": bool(structured_match.get("available")),
                "guideline_rule_score": bool(rule_match.get("available")),
                "disease_consistency_score": bool(diagnosis_alignment.get("available")),
                "diagnosis_hierarchy_score": bool(diagnosis_alignment.get("available")),
            }
            fusion_plan = build_final_fusion_score(
                components=fusion_components,
                availability=fusion_availability,
                template_id=active_template_id,
            )
            diagnosis_bonus = float(diagnosis_alignment.get("diagnosis_bonus", 0.0))
            final_score = float(fusion_plan.get("score", 0.0))
            reasons = []
            if diagnosis_alignment.get("reason"):
                reasons.append(str(diagnosis_alignment.get("reason")))
            if semantic_weight_plan["normalized_weights"]:
                reasons.append(
                    "语义链权重采用 "
                    + ", ".join(
                        f"{CHAIN_LABELS[bucket]}={weight:.2f}"
                        for bucket, weight in (
                            (
                                bucket,
                                float(semantic_weight_plan["normalized_weights"].get(bucket, 0.0)),
                            )
                            for bucket in CHAIN_LABELS
                            if bucket in semantic_weight_plan["normalized_weights"]
                        )
                    )
                    + "。"
                )
            if structured_match.get("available"):
                reasons.append(
                    f"结构化匹配分为 {float(structured_match.get('score', 0.0)):.4f}，来源 `{structured_match.get('source', '')}`。"
                )
            if query_denoise_report.get("applied"):
                reasons.append(
                    "查询侧去噪："
                    + "、".join(str(item) for item in query_denoise_report.get("steps", []))
                    + "。"
                )
            if case_denoise_report.get("applied"):
                reasons.append(
                    "候选侧去噪："
                    + "、".join(str(item) for item in case_denoise_report.get("steps", []))
                    + "。"
                )
            if rule_match.get("shared_rule_ids"):
                reasons.append(
                    "共享模板规则命中："
                    + "、".join(str(item) for item in rule_match.get("shared_rule_ids", []))
                    + "。"
                )
            elif rule_match.get("available"):
                reasons.append(
                    "规则未共享命中：查询侧命中 "
                    + "、".join(
                        str(item.get("rule_id"))
                        for item in rule_match.get("query_rule_hits", [])
                        if item.get("rule_id")
                    )
                    + "；候选侧命中 "
                    + "、".join(
                        str(item.get("rule_id"))
                        for item in rule_match.get("case_rule_hits", [])
                        if item.get("rule_id")
                    )
                    + "。"
                )
            history_terms = self._top_overlap_terms(
                history_chain,
                "\n".join(item["raw_content"] for item in case["chains"]["history_chain"]),
            )
            if history_terms:
                reasons.append(f"病史链重合要点：{'、'.join(history_terms[:4])}。")
            evidence_terms = self._top_overlap_terms(
                evidence_chain,
                "\n".join(item["raw_content"] for item in case["chains"]["evidence_chain"]),
            )
            if evidence_terms:
                reasons.append(f"证据链重合要点：{'、'.join(evidence_terms[:4])}。")
            care_terms = self._top_overlap_terms(
                care_chain,
                "\n".join(item["raw_content"] for item in case["chains"]["care_chain"]),
            )
            if care_terms:
                reasons.append(f"诊疗链重合要点：{'、'.join(care_terms[:4])}。")
            severity_terms = self._top_overlap_terms(
                severity_chain,
                "\n".join(item["raw_content"] for item in case["chains"]["severity_chain"]),
            )
            if severity_terms:
                reasons.append(f"严重度链重合要点：{'、'.join(severity_terms[:4])}。")
            if case["chains"]["severity_chain"]:
                reasons.append(f"严重度链摘要：{case['chains']['severity_chain'][0]['summary']}。")
            reasons.append(
                "融合评分组成："
                f"语义 {semantic_score:.4f}，结构化 {float(structured_match.get('score', 0.0)):.4f}，"
                f"规则 {float(rule_match.get('score', 0.0)):.4f}，"
                f"疾病一致性 {float(diagnosis_alignment.get('disease_consistency_score', 0.0)):.4f}，"
                f"诊断层级 {float(diagnosis_alignment.get('diagnosis_hierarchy_score', 0.0)):.4f}，"
                f"最终基础分 {final_score:.4f}。"
            )
            scored.append(
                (
                    final_score,
                    case,
                    reasons,
                    chain_scores,
                    diagnosis_bonus,
                    {
                        "semantic_score": round(semantic_score, 6),
                        "structured_score": round(float(structured_match.get("score", 0.0)), 6),
                        "guideline_rule_score": round(float(rule_match.get("score", 0.0)), 6),
                        "disease_consistency_score": round(
                            float(diagnosis_alignment.get("disease_consistency_score", 0.0)),
                            6,
                        ),
                        "diagnosis_hierarchy_score": round(
                            float(diagnosis_alignment.get("diagnosis_hierarchy_score", 0.0)),
                            6,
                        ),
                        "structured_match": structured_match,
                        "rule_match": rule_match,
                        "diagnosis_alignment": diagnosis_alignment,
                        "fusion_plan": fusion_plan,
                        "applied_chain_weights": semantic_weight_plan["normalized_weights"],
                        "weight_source": semantic_weight_plan["source"],
                        "query_effective_profile": effective_query_profile,
                        "case_effective_profile": effective_case_profile,
                        "query_denoise_report": query_denoise_report,
                        "case_denoise_report": case_denoise_report,
                        "query_core_chain_texts": query_core_chain_texts,
                        "case_core_chain_texts": case.get("core_chain_texts", {}),
                        "base_query_denoise_report": base_query_denoise_report,
                    },
                )
            )
            if debug_query_case_id and case_id == debug_query_case_id:
                # #region debug-point D:self-score-breakdown
                _debug_emit(
                    "D",
                    "SimilarityService.search:self-score",
                    "captured self-match scoring breakdown",
                    {
                        "traceId": trace_id,
                        "query_case_id": debug_query_case_id,
                        "case_id": case_id,
                        "semantic_weight_plan": semantic_weight_plan,
                        "chain_scores": chain_scores,
                        "query_profile": effective_query_profile,
                        "case_profile": effective_case_profile,
                        "query_denoise_report": query_denoise_report,
                        "case_denoise_report": case_denoise_report,
                        "structured_match": structured_match,
                        "rule_match": rule_match,
                        "diagnosis_alignment": diagnosis_alignment,
                        "fusion_components": fusion_components,
                        "fusion_availability": fusion_availability,
                        "fusion_plan": fusion_plan,
                        "case_chain_lengths": {
                            bucket: len(str(case.get("chain_texts", {}).get(bucket, "")))
                            for bucket in CHAIN_LABELS
                        },
                        "case_chain_preview": {
                            bucket: shorten_text(str(case.get("chain_texts", {}).get(bucket, "")), 240)
                            for bucket in CHAIN_LABELS
                        },
                        "case_core_chain_texts": case.get("core_chain_texts", {}),
                    },
                )
                # #endregion

        scored.sort(key=lambda item: item[0], reverse=True)
        # #region debug-point D:similarity-search-scoring
        _debug_emit(
            "D",
            "SimilarityService.search:scoring",
            "case scoring loop finished",
            {
                "traceId": trace_id,
                "elapsed_ms": round((time.perf_counter() - scoring_started_at) * 1000, 2),
                "scored_count": len(scored),
                "top_score_before_rerank": round(float(scored[0][0]), 6) if scored else 0.0,
            },
        )
        # #endregion
        rerank_error = ""
        rerank_applied = False
        rerank_started_at = time.perf_counter()
        # Stage 1: Pre-rerank adjustment — lightly boost same-template
        # candidates into the rerank candidate pool so they are not excluded
        # before the reranker even sees them.
        scored = self._apply_post_rank_adjustment(
            query_diagnosis_mapping=query_diagnosis_mapping,
            scored_rows=scored,
            stage="pre_rerank",
        )
        scored_with_rerank, rerank_error, rerank_applied = self._apply_rerank(
            query_text=rerank_query_text,
            top_k=top_k,
            query_diagnosis_mapping=query_diagnosis_mapping,
            scored=scored,
        )
        # Stage 2: Post-rerank adjustment — after the reranker blends scores
        # (0.7*base + 0.3*rerank), same-template candidates may still be
        # pushed down by semantically-similar cross-template results.
        # Re-apply the adjustment to give a final ordering nudge so that
        # same-template / rule-evidenced candidates stay on top.
        if rerank_applied:
            scored_with_rerank = self._apply_post_rank_adjustment(
                query_diagnosis_mapping=query_diagnosis_mapping,
                scored_rows=scored_with_rerank,
                stage="post_rerank",
            )
        if debug_query_case_id:
            self_row = next((row for row in scored_with_rerank if row[1].get("case_id") == debug_query_case_id), None)
            # #region debug-point E:self-result
            _debug_emit(
                "E",
                "SimilarityService.search:self-result",
                "captured self-match final result",
                {
                    "traceId": trace_id,
                    "query_case_id": debug_query_case_id,
                    "top_case_ids": [row[1].get("case_id") for row in scored_with_rerank[: min(5, len(scored_with_rerank))]],
                    "self_row": {
                        "final_score": self_row[0],
                        "case_id": self_row[1].get("case_id"),
                        "rerank_score": self_row[6],
                        "base_score": self_row[7],
                        "score_breakdown": self_row[5],
                    }
                    if self_row
                    else None,
                    "rerank_applied": rerank_applied,
                    "rerank_error": rerank_error,
                },
            )
            # #endregion
        # #region debug-point E:similarity-search-rerank
        _debug_emit(
            "E",
            "SimilarityService.search:rerank",
            "rerank and post-rank finished",
            {
                "traceId": trace_id,
                "elapsed_ms": round((time.perf_counter() - rerank_started_at) * 1000, 2),
                "rerank_applied": rerank_applied,
                "rerank_error": rerank_error,
                "candidate_count": len(scored_with_rerank),
            },
        )
        # #endregion
        results = []
        for (
            final_score,
            case,
            reasons,
            chain_scores,
            diagnosis_bonus,
            score_breakdown,
            rerank_score,
            base_score,
        ) in scored_with_rerank[:top_k]:
            post_rank_adjustment = score_breakdown.get("post_rank_adjustment", {})
            pre_rerank_adjustment = score_breakdown.get("pre_rerank_adjustment", {})
            if rerank_applied and rerank_score:
                reasons = list(reasons) + [
                    f"Qwen3 reranker 复排分为 {rerank_score:.4f}，基础检索分为 {base_score:.4f}。"
                ]
            if pre_rerank_adjustment.get("applied"):
                reasons = list(reasons) + [
                    "预排序调整："
                    + "、".join(str(item) for item in pre_rerank_adjustment.get("reasons", []))
                    + f"，分值调整 {float(pre_rerank_adjustment.get('delta', 0.0)):+.4f}（进入复排池前）。"
                ]
            if post_rank_adjustment.get("applied"):
                reasons = list(reasons) + [
                    "排序后调："
                    + "、".join(str(item) for item in post_rank_adjustment.get("reasons", []))
                    + f"，分值调整 {float(post_rank_adjustment.get('delta', 0.0)):+.4f}。"
                ]
            evidence_coverage = build_evidence_coverage(
                chain_scores=chain_scores,
                score_breakdown=score_breakdown,
                case=case,
                query_diagnosis_mapping=query_diagnosis_mapping,
            )
            results.append(
                {
                    "case_id": case["case_id"],
                    "diagnosis": case["diagnosis"],
                    "normalized_diagnosis": (case.get("diagnosis_mapping") or {}).get(
                        "normalized_text", ""
                    ),
                    "diagnosis_concept_id": (case.get("diagnosis_mapping") or {}).get(
                        "concept_id", ""
                    ),
                    "diagnosis_icd10": (case.get("diagnosis_mapping") or {}).get("icd10", ""),
                    "template_hint": (case.get("diagnosis_mapping") or {}).get("template", ""),
                    "family_hint": (case.get("diagnosis_mapping") or {}).get("family", ""),
                    "diagnosis_mapping": case.get("diagnosis_mapping") or {},
                    "gender": case["gender"],
                    "age": case["age"],
                    "score": round(final_score, 6),
                    "final_score": round(final_score, 6),
                    "base_score": round(base_score, 6),
                    "rerank_score": round(rerank_score, 6),
                    "diagnosis_bonus": round(diagnosis_bonus, 6),
                    "semantic_score": score_breakdown.get("semantic_score", 0.0),
                    "structured_score": score_breakdown.get("structured_score", 0.0),
                    "guideline_rule_score": score_breakdown.get("guideline_rule_score", 0.0),
                    "disease_consistency_score": score_breakdown.get(
                        "disease_consistency_score", 0.0
                    ),
                    "diagnosis_hierarchy_score": score_breakdown.get(
                        "diagnosis_hierarchy_score", 0.0
                    ),
                    "structured_match": score_breakdown.get("structured_match", {}),
                    "rule_match": score_breakdown.get("rule_match", {}),
                    "diagnosis_alignment": score_breakdown.get("diagnosis_alignment", {}),
                    "fusion_plan": score_breakdown.get("fusion_plan", {}),
                    "post_rank_adjustment": post_rank_adjustment,
                    "pre_rerank_adjustment": pre_rerank_adjustment,
                    "applied_chain_weights": score_breakdown.get("applied_chain_weights", {}),
                    "weight_source": score_breakdown.get("weight_source", ""),
                    "structured_profile": case.get("structured_profile") or {},
                    "effective_structured_profile": score_breakdown.get("case_effective_profile", {}),
                    "query_effective_profile": score_breakdown.get("query_effective_profile", {}),
                    "query_denoise_report": score_breakdown.get("query_denoise_report", {}),
                    "case_denoise_report": score_breakdown.get("case_denoise_report", {}),
                    "query_core_chain_texts": score_breakdown.get("query_core_chain_texts", {}),
                    "case_core_chain_texts": score_breakdown.get("case_core_chain_texts", {}),
                    "chain_scores": chain_scores,
                    "evidence_coverage": evidence_coverage,
                    "evidence_coverage_score": float(evidence_coverage.get("score", 0.0) or 0.0),
                    "reasons": reasons,
                    "chain_overview": case["chain_overview"],
                    "chain_counts": {bucket: len(case["chains"][bucket]) for bucket in CHAIN_LABELS},
                }
            )

        response_payload = {
            "query_text": query_text,
            "rerank_query_text": rerank_query_text,
            "query_diagnosis_mapping": query_diagnosis_mapping,
            "query_structured_profile": query_structured_profile,
            "query_effective_structured_profile": (
                results[0].get("query_effective_profile", {})
                if results
                else query_structured_profile
            ),
            "query_denoise_report": (
                results[0].get("query_denoise_report", {})
                if results
                else {"applied": False, "template": "", "steps": [], "removed": {}}
            ),
            "query_core_chain_texts": (
                results[0].get("query_core_chain_texts", {})
                if results
                else query_core_chain_texts
            ),
            "semantic_weight_plan": semantic_weight_plan,
            "strategy": (
                f"{self.status} + {XINFERENCE_RERANK_MODEL}复排"
                if (query_vec_map or query_core_vec_map) and rerank_applied
                else self.status
                if (query_vec_map or query_core_vec_map)
                else f"词项相似度回退 + {XINFERENCE_RERANK_MODEL}复排"
                if rerank_applied
                else "词项相似度回退"
            ),
            "embedding_error": embedding_error,
            "rerank_error": rerank_error,
            "results": results,
        }
        # #region debug-point E:similarity-search-return
        _debug_emit(
            "E",
            "SimilarityService.search:return",
            "similarity search returning",
            {
                "traceId": trace_id,
                "elapsed_ms": round((time.perf_counter() - search_started_at) * 1000, 2),
                "result_count": len(results),
                "strategy": response_payload.get("strategy", ""),
                "embedding_error": embedding_error,
                "rerank_error": rerank_error,
            },
        )
        # #endregion
        return response_payload


@dataclass
class DatasetStore:
    dataset_id: str
    label: str
    working_dir: Path
    custom_kg_path: Path

    def __post_init__(self):
        self.graph_file = self.working_dir / "graph_chunk_entity_relation.graphml"
        self.embed_cache = self.working_dir / "case_similarity_embeddings.json"
        ensure_graph_from_custom_kg(self.custom_kg_path, self.graph_file)
        try:
            self.graph = nx.read_graphml(self.graph_file)
        except Exception:
            self.graph_file.unlink(missing_ok=True)
            ensure_graph_from_custom_kg(self.custom_kg_path, self.graph_file)
            self.graph = nx.read_graphml(self.graph_file)
        self.case_index = build_case_index(self.custom_kg_path)
        self.similarity = SimilarityService(self.case_index["cases"], self.embed_cache)

    def meta(self):
        started_at = time.perf_counter()
        # #region debug-point A:dataset-meta-entry
        _debug_emit(
            "A",
            "DatasetStore.meta",
            "[DEBUG] dataset meta requested",
            {"dataset_id": self.dataset_id, "label": self.label},
        )
        # #endregion
        payload = {
            "dataset_id": self.dataset_id,
            "dataset_label": self.label,
            "graph": {"nodes": self.graph.number_of_nodes(), "edges": self.graph.number_of_edges()},
            "cases": len(self.case_index["case_rows"]),
            "hyperedges": len(self.case_index["hyperedges"]),
        }
        # #region debug-point A:dataset-meta-return
        _debug_emit(
            "A",
            "DatasetStore.meta:return",
            "[DEBUG] dataset meta resolved",
            {
                "dataset_id": self.dataset_id,
                "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "cases": payload["cases"],
                "hyperedges": payload["hyperedges"],
            },
        )
        # #endregion
        return payload

    def graph_sample(self, max_nodes: int):
        return build_graph_sample(self.graph, max_nodes=max_nodes)

    def graph_expand(self, center: str, hops: int, max_nodes: int):
        return build_graph_neighborhood(self.graph, center=center, hops=hops, max_nodes=max_nodes)

    def cases_rows(self):
        return {"rows": self.case_index["case_rows"]}

    def case_detail(self, case_id: str):
        case = self.case_index["cases"].get(case_id)
        if not case:
            return None
        return {
            "case_id": case["case_id"],
            "diagnosis": case["diagnosis"],
            "normalized_diagnosis": (case.get("diagnosis_mapping") or {}).get("normalized_text", ""),
            "diagnosis_concept_id": (case.get("diagnosis_mapping") or {}).get("concept_id", ""),
            "diagnosis_icd10": (case.get("diagnosis_mapping") or {}).get("icd10", ""),
            "template_hint": (case.get("diagnosis_mapping") or {}).get("template", ""),
            "family_hint": (case.get("diagnosis_mapping") or {}).get("family", ""),
            "diagnosis_mapping": case.get("diagnosis_mapping") or {},
            "gender": case["gender"],
            "age": case["age"],
            "dataset": case["dataset"],
            "top_entities": case["top_entities"],
            "relation_type_counts": case["relation_type_counts"],
            "structured_profile": case.get("structured_profile") or {},
            "chains": {
                bucket: [
                    {
                        "hyperedge_id": item["hyperedge_id"],
                        "relation_type": item["relation_type"],
                        "summary": item["summary"],
                        "time_info": item["time_info"],
                        "entities": item["entities"],
                        "evidence_fragments": item["evidence_fragments"],
                    }
                    for item in case["chains"][bucket]
                ]
                for bucket in CHAIN_LABELS
            },
        }

    def search_nodes(self, query: str):
        return {"rows": search_nodes(self.graph, query)}

    def similar_cases(self, payload: dict[str, Any], top_k: int):
        return self.similarity.search(payload, top_k=top_k)


class TestCaseRepository:
    LARGE_JSON_FAST_SCAN_THRESHOLD_BYTES = max(
        64 * 1024 * 1024,
        int(os.environ.get("TESTSET_LARGE_JSON_FAST_SCAN_THRESHOLD_BYTES", str(256 * 1024 * 1024))),
    )
    LARGE_JSON_FAST_SCAN_CHUNK_BYTES = max(
        1024 * 1024,
        int(os.environ.get("TESTSET_LARGE_JSON_FAST_SCAN_CHUNK_BYTES", str(8 * 1024 * 1024))),
    )
    FULL_CHAIN_CHAR_LIMITS = {
        "history_chain": 2400,
        "evidence_chain": 3600,
        "care_chain": 2200,
        "severity_chain": 1600,
    }
    PREVIEW_CHAIN_CHAR_LIMITS = {
        "history_chain": 420,
        "evidence_chain": 520,
        "care_chain": 420,
        "severity_chain": 320,
    }
    FULL_TEXT_VALUE_MAX_CHARS = 600
    PREVIEW_TEXT_VALUE_MAX_CHARS = 180
    FULL_RECORD_TEXT_MAX_CHARS = 360
    FULL_DETAIL_LIST_LIMITS = {
        "abn_lab": 28,
        "异常检验": 28,
        "检验项目列表": 28,
        "normal_lab": 10,
        "正常检验": 10,
        "img": 16,
        "影像检查列表": 16,
        "other_diag": 20,
        "其他诊断": 20,
        "诊断信息": 20,
        "drugs": 24,
        "用药医嘱列表": 24,
        "surgeries": 12,
        "手术列表": 12,
        "complications": 24,
        "并发症列表": 24,
        "scores": 10,
        "功能评分": 10,
        "basis": 10,
        "分期依据": 10,
    }
    PREVIEW_DETAIL_LIST_LIMITS = {
        "abn_lab": 10,
        "异常检验": 10,
        "检验项目列表": 10,
        "normal_lab": 3,
        "正常检验": 3,
        "img": 4,
        "影像检查列表": 4,
        "other_diag": 6,
        "其他诊断": 6,
        "诊断信息": 6,
        "drugs": 10,
        "用药医嘱列表": 10,
        "surgeries": 4,
        "手术列表": 4,
        "complications": 10,
        "并发症列表": 10,
        "scores": 4,
        "功能评分": 4,
        "basis": 4,
        "分期依据": 4,
    }
    FOCUS_FILTER_KEYS = {
        "abn_lab",
        "异常检验",
        "检验项目列表",
        "normal_lab",
        "正常检验",
        "norm_lab_cnt",
        "normal_lab_cnt",
        "img",
        "exams",
        "影像检查",
        "影像检查列表",
        "drugs",
        "drug_names",
        "用药医嘱",
        "用药医嘱列表",
        "order",
        "出院医嘱",
        "surgeries",
        "main_diag",
        "dis_diag",
        "other_diag",
        "其他诊断",
        "诊断信息",
        "诊断列表",
        "icd",
        "complications",
        "并发症列表",
        "并发症",
        "主诉",
        "现病史",
        "既往史",
        "个人史",
        "主要症状",
        "体格检查",
    }
    RESP_DIAG_KEYWORDS = (
        "慢性阻塞性肺",
        "慢阻肺",
        "copd",
        "aecopd",
        "呼吸衰竭",
        "肺炎",
        "肺部感染",
        "支气管扩张",
        "支扩",
        "哮喘",
        "肺气肿",
        "胸腔积液",
        "气胸",
        "肺结核",
        "间质性肺疾病",
        "肺纤维化",
        "高碳酸",
        "低氧",
        "肺栓塞",
        "胸膜",
    )
    RESP_LAB_KEYWORDS = (
        "血气",
        "氧分压",
        "二氧化碳分压",
        "氧饱和度",
        "pao2",
        "paco2",
        "spo2",
        "sao2",
        "fev1",
        "fvc",
        "fev1/fvc",
        "crp",
        "降钙素原",
        "pct",
        "白细胞计数",
        "中性粒",
        "嗜酸",
        "d-二聚体",
        "d_dimer",
        "乳酸",
    )
    RESP_LAB_NEGATIVE_KEYWORDS = (
        "尿白细胞",
        "尿胆原",
        "前列腺",
        "雌二醇",
        "促黄体",
        "cd45",
        "psa",
        "肌酐",
        "胆红素",
        "甲状腺",
    )
    RESP_IMAGING_KEYWORDS = (
        "肺",
        "胸部",
        "胸片",
        "支气管",
        "气道",
        "胸腔",
        "胸膜",
        "磨玻璃",
        "实变",
        "渗出",
        "肺气肿",
        "肺大泡",
        "结节",
        "空洞",
        "纤维化",
        "间质",
        "积液",
        "气胸",
        "肺栓塞",
        "肺不张",
    )
    RESP_IMAGING_NEGATIVE_KEYWORDS = (
        "盆腔",
        "腮腺",
        "甲状腺",
        "髋关节",
        "肩关节",
        "腹部",
        "肝胆",
        "肾",
        "前列腺",
        "胸椎",
    )
    RESP_DRUG_KEYWORDS = (
        "雾化",
        "吸入",
        "支气管扩张",
        "沙丁胺醇",
        "异丙托溴铵",
        "噻托溴铵",
        "布地奈德",
        "激素",
        "糖皮质激素",
        "抗感染",
        "抗菌",
        "抗生素",
        "吸氧",
        "氧疗",
        "无创",
        "有创",
        "机械通气",
        "通气",
        "祛痰",
        "乙酰半胱氨酸",
        "氨溴索",
        "茶碱",
        "孟鲁司特",
    )
    RESP_DRUG_NEGATIVE_KEYWORDS = (
        "滴眼液",
        "胰岛素",
        "乳果糖",
        "降压",
        "护胃",
    )
    RESP_PROCEDURE_KEYWORDS = (
        "支气管镜",
        "肺叶切除",
        "肺段切除",
        "肺楔形切除",
        "胸腔镜",
        "气管镜",
        "穿刺活检",
        "ebus",
        "tbna",
        "bt-pna",
        "超声支气管镜",
        "无创通气",
        "有创通气",
        "机械通气",
    )
    RESP_HISTORY_KEYWORDS = (
        "慢阻肺",
        "copd",
        "aecopd",
        "咳嗽",
        "咳痰",
        "喘",
        "气促",
        "胸闷",
        "呼吸困难",
        "发热",
        "吸烟",
        "戒烟",
        "肺气肿",
        "支气管",
        "氧疗",
        "雾化",
        "无创",
        "有创",
        "机械通气",
        "呼吸衰竭",
        "肺炎",
        "肺部感染",
    )

    def __init__(self, root: Path, dataset_configs: dict[str, dict[str, Any]] | None = None):
        self.root = root
        self.dataset_configs = dataset_configs or build_dataset_configs()
        self.testsets = self._discover_testsets()
        self.cache_dir = REPO_ROOT / "validation_reports" / "testset_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict[str, Any]] = {}
        self._rows_cache: dict[str, dict[str, Any]] = {}
        self._detail_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def _discover_testsets(self) -> dict[str, dict[str, Any]]:
        configs: dict[str, dict[str, Any]] = {}
        if not self.root.exists():
            return configs
        for directory in sorted(self.root.iterdir(), key=lambda item: item.name):
            if not directory.is_dir() or directory.name in TESTSET_IGNORE_DIRS:
                continue
            if not any(self._resolve_chain_file(directory, bucket) for bucket in CHAIN_LABELS):
                continue
            testset_id = re.sub(r"[^a-z0-9]+", "_", directory.name.lower()).strip("_") or directory.name
            suffix = 2
            base_id = testset_id
            while testset_id in configs:
                testset_id = f"{base_id}_{suffix}"
                suffix += 1
            configs[testset_id] = {
                "id": testset_id,
                "label": directory.name,
                "path": directory,
                "scope_hint": self._infer_scope_hint(directory.name),
            }
        return configs

    def _infer_scope_hint(self, label: str) -> str:
        text = str(label).strip()
        if "急性加重" in text:
            return "ae"
        if "感染" in text:
            return "infection"
        if "慢阻肺" in text or "慢性阻塞性肺疾病" in text:
            return "chronic"
        return "all"

    def _resolve_chain_file(self, directory: Path, bucket: str) -> Path | None:
        for pattern in TESTSET_FILE_PATTERNS.get(bucket, []):
            matches = sorted(path for path in directory.glob(pattern) if path.is_file())
            if matches:
                return matches[0]
        return None

    def _testset_source_signature(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        signature = []
        directory = config.get("path")
        if not isinstance(directory, Path):
            return signature
        for bucket in CHAIN_LABELS:
            chain_file = self._resolve_chain_file(directory, bucket)
            if chain_file is None:
                continue
            stat = chain_file.stat()
            signature.append(
                {
                    "bucket": bucket,
                    "path": str(chain_file),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "size": int(stat.st_size),
                }
            )
        return signature

    def _cache_file_path(self, testset_id: str, suffix: str) -> Path:
        raw_id = str(testset_id)
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw_id).strip("_") or "testset"
        digest = hashlib.md5(raw_id.encode("utf-8")).hexdigest()[:10]
        return self.cache_dir / f"{safe_id}.{digest}.{suffix}.json"

    def _detail_cache_file_path(self, testset_id: str, case_key: str) -> Path:
        raw_id = str(testset_id)
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw_id).strip("_") or "testset"
        prefix = hashlib.md5(raw_id.encode("utf-8")).hexdigest()[:10]
        digest = hashlib.md5(case_key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{safe_id}.{prefix}.detail.{digest}.json"

    def _build_large_json_case_needles(self, record: dict[str, Any], case_key: str) -> list[bytes]:
        file_name = normalize_text_value(record.get("file_name", ""))
        unique_id = normalize_text_value(record.get("unique_id", ""))
        visit_id = normalize_text_value(record.get("visit_id", ""))
        needles: list[str] = []
        if file_name and unique_id and visit_id:
            needles.append(
                f'"meta":{{"f":{json.dumps(file_name, ensure_ascii=False)},"id":{json.dumps(unique_id, ensure_ascii=False)},"vid":{json.dumps(visit_id, ensure_ascii=False)}}}'
            )
            needles.append(
                f'"meta":{{"f":{json.dumps(file_name, ensure_ascii=False)},"id":{json.dumps(unique_id, ensure_ascii=False)},"vid":{json.dumps(visit_id, ensure_ascii=False)}'
            )
            needles.append(
                f'"f":{json.dumps(file_name, ensure_ascii=False)},"id":{json.dumps(unique_id, ensure_ascii=False)},"vid":{json.dumps(visit_id, ensure_ascii=False)}'
            )
        if unique_id and visit_id:
            needles.append(
                f'"id":{json.dumps(unique_id, ensure_ascii=False)},"vid":{json.dumps(visit_id, ensure_ascii=False)}'
            )
        if case_key:
            needles.append(case_key)
        return [needle.encode("utf-8") for needle in needles if needle]

    def _find_large_json_needle_offset(self, path: Path, needle: bytes) -> int:
        if not needle:
            return -1
        try:
            with path.open("rb") as fh:
                with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    return mm.find(needle)
        except (OSError, ValueError):
            return -1

    def _find_large_json_object_start(self, path: Path, offset: int) -> int:
        markers = (b'{"meta":', b'{"\xe7\x97\x85\xe4\xbe\x8b\xe6\xa0\x87\xe8\xaf\x86":')
        window = self.LARGE_JSON_FAST_SCAN_CHUNK_BYTES
        end = max(offset + max(len(marker) for marker in markers), 0)
        with path.open("rb") as fh:
            while end > 0:
                start = max(0, end - window)
                fh.seek(start)
                data = fh.read(end - start)
                best = -1
                for marker in markers:
                    idx = data.rfind(marker)
                    if idx > best:
                        best = idx
                if best != -1:
                    return start + best
                if start == 0:
                    break
                end = start + max(len(marker) for marker in markers) - 1
        return -1

    def _find_large_json_object_end(self, path: Path, start_offset: int) -> int:
        chunk_size = self.LARGE_JSON_FAST_SCAN_CHUNK_BYTES
        depth = 0
        in_string = False
        escape = False
        seen_open = False
        offset = start_offset
        with path.open("rb") as fh:
            fh.seek(start_offset)
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    return -1
                for index, byte in enumerate(chunk):
                    if in_string:
                        if escape:
                            escape = False
                        elif byte == 92:
                            escape = True
                        elif byte == 34:
                            in_string = False
                        continue
                    if byte == 34:
                        in_string = True
                    elif byte == 123:
                        depth += 1
                        seen_open = True
                    elif byte == 125:
                        depth -= 1
                        if seen_open and depth == 0:
                            return offset + index
                offset += len(chunk)

    def _load_large_json_case_item(self, path: Path, record: dict[str, Any], case_key: str) -> dict[str, Any] | None:
        for needle in self._build_large_json_case_needles(record, case_key):
            found_offset = self._find_large_json_needle_offset(path, needle)
            if found_offset < 0:
                continue
            start_offset = self._find_large_json_object_start(path, found_offset)
            if start_offset < 0:
                continue
            end_offset = self._find_large_json_object_end(path, start_offset)
            if end_offset < start_offset:
                continue
            with path.open("rb") as fh:
                fh.seek(start_offset)
                payload = fh.read(end_offset - start_offset + 1)
            try:
                item = json.loads(payload.decode("utf-8"))
            except Exception:
                continue
            current_key, _meta = self._extract_case_identity(item)
            if current_key == case_key:
                return item
        return None

    def _load_disk_cache(self, path: Path, signature: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            payload = load_json(path)
        except Exception:
            return None
        if payload.get("signature") != signature:
            return None
        data = payload.get("data")
        return data if isinstance(data, dict) else None

    def _save_disk_cache(self, path: Path, signature: list[dict[str, Any]], data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"signature": signature, "data": data}, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load_testset(self, testset_id: str) -> dict[str, Any] | None:
        if testset_id in self._cache:
            return self._cache[testset_id]
        config = self.testsets.get(testset_id)
        if not config:
            return None

        cases: dict[str, dict[str, Any]] = {}
        for bucket in CHAIN_LABELS:
            chain_file = self._resolve_chain_file(config["path"], bucket)
            if chain_file is None:
                continue
            payload = load_json(chain_file)
            items = payload.get("data") or payload.get("病例相似链数据") or []
            for item in items:
                case_key, meta = self._extract_case_identity(item)
                if not case_key:
                    continue
                record = cases.setdefault(
                    case_key,
                    {
                        "case_key": case_key,
                        "file_name": normalize_text_value(meta.get("file_name", "")),
                        "unique_id": normalize_text_value(meta.get("unique_id", "")),
                        "visit_id": normalize_text_value(meta.get("visit_id", "")),
                        "name": normalize_text_value(meta.get("name", "")),
                        "scope_hint": config["scope_hint"],
                        "diagnosis": "",
                        "history_chain": "",
                        "history_chain_preview": "",
                        "evidence_chain": "",
                        "evidence_chain_preview": "",
                        "care_chain": "",
                        "care_chain_preview": "",
                        "severity_chain": "",
                        "severity_chain_preview": "",
                    },
                )
                if not record["diagnosis"]:
                    record["diagnosis"] = self._extract_diagnosis(
                        item,
                        meta,
                        scope_hint=config["scope_hint"],
                    )
                diagnosis_for_render = record["diagnosis"] or self._extract_diagnosis(
                    item,
                    meta,
                    scope_hint=config["scope_hint"],
                )
                diagnosis_mapping = build_diagnosis_mapping_payload(diagnosis_for_render, max_matches=1)
                chain_bundle = self._extract_chain_bundle(
                    item,
                    diagnosis_mapping=diagnosis_mapping,
                    scope_hint=config["scope_hint"],
                )
                if chain_bundle["full"]:
                    record[bucket] = chain_bundle["full"]
                if chain_bundle["preview"]:
                    record[f"{bucket}_preview"] = chain_bundle["preview"]

        rows = []
        for record in cases.values():
            preview_parts = [
                shorten_text(record["history_chain"], 72) if record["history_chain"] else "",
                shorten_text(record["evidence_chain"], 72) if record["evidence_chain"] else "",
            ]
            preview = " | ".join(part for part in preview_parts if part)
            rows.append(
                {
                    "case_key": record["case_key"],
                    "case_label": self._build_case_label(record),
                    "diagnosis": record["diagnosis"],
                    "file_name": record["file_name"],
                    "unique_id": record["unique_id"],
                    "visit_id": record["visit_id"],
                    "preview": preview,
                }
            )

        rows.sort(
            key=lambda row: (
                self._sort_visit_id(row.get("visit_id", "")),
                row.get("file_name", ""),
                row.get("unique_id", ""),
            )
        )
        payload = {
            "id": config["id"],
            "label": config["label"],
            "scope_hint": config["scope_hint"],
            "case_count": len(rows),
            "rows": rows,
            "cases": cases,
        }
        self._cache[testset_id] = payload
        return payload

    def _build_testset_case_record(
        self,
        *,
        case_key: str,
        meta: dict[str, str],
        scope_hint: str,
        diagnosis: str = "",
    ) -> dict[str, Any]:
        return {
            "case_key": case_key,
            "case_id": normalize_text_value(meta.get("case_id", "")),
            "file_name": normalize_text_value(meta.get("file_name", "")),
            "unique_id": normalize_text_value(meta.get("unique_id", "")),
            "visit_id": normalize_text_value(meta.get("visit_id", "")),
            "name": normalize_text_value(meta.get("name", "")),
            "scope_hint": scope_hint,
            "diagnosis": diagnosis,
            "history_chain": "",
            "history_chain_preview": "",
            "evidence_chain": "",
            "evidence_chain_preview": "",
            "evidence_groups": [],
            "care_chain": "",
            "care_chain_preview": "",
            "care_groups": [],
            "severity_chain": "",
            "severity_chain_preview": "",
            "severity_groups": [],
            "history_groups": [],
        }

    def _load_testset_rows(self, testset_id: str) -> dict[str, Any] | None:
        started_at = time.perf_counter()
        if testset_id in self._rows_cache:
            # #region debug-point B:testset-rows-memory-hit
            _debug_emit(
                "B",
                "TestCaseRepository._load_testset_rows:memory_hit",
                "[DEBUG] testset rows cache hit",
                {
                    "testset_id": testset_id,
                    "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
                },
            )
            # #endregion
            return self._rows_cache[testset_id]
        config = self.testsets.get(testset_id)
        if not config:
            return None

        signature = self._testset_source_signature(config)
        disk_rows = self._load_disk_cache(self._cache_file_path(testset_id, "rows"), signature)
        if disk_rows is not None:
            self._rows_cache[testset_id] = disk_rows
            # #region debug-point B:testset-rows-disk-hit
            _debug_emit(
                "B",
                "TestCaseRepository._load_testset_rows:disk_hit",
                "[DEBUG] testset rows disk cache hit",
                {
                    "testset_id": testset_id,
                    "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
                    "case_count": len(disk_rows.get("rows", [])),
                },
            )
            # #endregion
            return disk_rows

        anchor_bucket = "history_chain"
        chain_file = self._resolve_chain_file(config["path"], anchor_bucket)
        if chain_file is None:
            for candidate_bucket in CHAIN_LABELS:
                chain_file = self._resolve_chain_file(config["path"], candidate_bucket)
                if chain_file is not None:
                    anchor_bucket = candidate_bucket
                    break
        if chain_file is None:
            return None

        payload = load_json(chain_file)
        items = payload.get("data") or payload.get("病例相似链数据") or []
        cases: dict[str, dict[str, Any]] = {}
        rows = []
        for item in items:
            case_key, meta = self._extract_case_identity(item)
            if not case_key:
                continue
            diagnosis = self._extract_diagnosis(item, meta, scope_hint=config["scope_hint"])
            record = self._build_testset_case_record(
                case_key=case_key,
                meta=meta,
                scope_hint=config["scope_hint"],
                diagnosis=diagnosis,
            )
            diagnosis_mapping = build_diagnosis_mapping_payload(diagnosis, max_matches=1)
            chain_bundle = self._extract_chain_bundle(
                item,
                diagnosis_mapping=diagnosis_mapping,
                scope_hint=config["scope_hint"],
            )
            if chain_bundle["full"]:
                record[anchor_bucket] = chain_bundle["full"]
            if chain_bundle["preview"]:
                record[f"{anchor_bucket}_preview"] = chain_bundle["preview"]

            preview = shorten_text(
                record.get(f"{anchor_bucket}_preview") or record.get(anchor_bucket) or "",
                144,
            )
            rows.append(
                {
                    "case_key": record["case_key"],
                    "case_label": self._build_case_label(record),
                    "diagnosis": record["diagnosis"],
                    "file_name": record["file_name"],
                    "unique_id": record["unique_id"],
                    "visit_id": record["visit_id"],
                    "preview": preview,
                }
            )
            cases[case_key] = record

        rows.sort(
            key=lambda row: (
                self._sort_visit_id(row.get("visit_id", "")),
                row.get("file_name", ""),
                row.get("unique_id", ""),
            )
        )
        lightweight_payload = {
            "id": config["id"],
            "label": config["label"],
            "scope_hint": config["scope_hint"],
            "case_count": len(rows),
            "rows": rows,
            "cases": cases,
        }
        self._rows_cache[testset_id] = lightweight_payload
        self._save_disk_cache(self._cache_file_path(testset_id, "rows"), signature, lightweight_payload)
        # #region debug-point B:testset-rows-built
        _debug_emit(
            "B",
            "TestCaseRepository._load_testset_rows:built",
            "[DEBUG] testset rows built from source",
            {
                "testset_id": testset_id,
                "anchor_bucket": anchor_bucket,
                "case_count": len(rows),
                "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
            },
        )
        # #endregion
        return lightweight_payload

    def _load_case_detail(self, testset_id: str, case_key: str) -> dict[str, Any] | None:
        started_at = time.perf_counter()
        cache_key = (testset_id, case_key)
        if cache_key in self._detail_cache:
            # #region debug-point B:testset-detail-memory-hit
            _debug_emit(
                "B",
                "TestCaseRepository._load_case_detail:memory_hit",
                "[DEBUG] testset detail memory cache hit",
                {
                    "testset_id": testset_id,
                    "case_key": case_key,
                    "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
                },
            )
            # #endregion
            return self._detail_cache[cache_key]
        config = self.testsets.get(testset_id)
        if not config:
            return None

        signature = self._testset_source_signature(config)
        detail_cache_path = self._detail_cache_file_path(testset_id, case_key)
        disk_detail = self._load_disk_cache(detail_cache_path, signature)
        if disk_detail is not None:
            self._detail_cache[cache_key] = disk_detail
            # #region debug-point B:testset-detail-disk-hit
            _debug_emit(
                "B",
                "TestCaseRepository._load_case_detail:disk_hit",
                "[DEBUG] testset detail disk cache hit",
                {
                    "testset_id": testset_id,
                    "case_key": case_key,
                    "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
                },
            )
            # #endregion
            return disk_detail

        rows_payload = self._load_testset_rows(testset_id)
        base_record = (rows_payload or {}).get("cases", {}).get(case_key)
        record = dict(base_record) if isinstance(base_record, dict) else None
        matched = False
        for bucket in CHAIN_LABELS:
            bucket_started_at = time.perf_counter()
            chain_file = self._resolve_chain_file(config["path"], bucket)
            if chain_file is None:
                continue
            target_item = None
            target_meta: dict[str, str] = {}
            items_count: int | None = None
            chain_file_size = chain_file.stat().st_size if chain_file.exists() else 0
            use_large_scan = chain_file_size >= self.LARGE_JSON_FAST_SCAN_THRESHOLD_BYTES
            if use_large_scan and record is not None:
                target_item = self._load_large_json_case_item(chain_file, record, case_key)
                if target_item is not None:
                    _matched_key, target_meta = self._extract_case_identity(target_item)
                # #region debug-point B:testset-detail-large-scan
                _debug_emit(
                    "B",
                    "TestCaseRepository._load_case_detail:large_scan",
                    "[DEBUG] large json fast scan finished",
                    {
                        "testset_id": testset_id,
                        "case_key": case_key,
                        "bucket": bucket,
                        "file_size_bytes": chain_file_size,
                        "matched": target_item is not None,
                        "elapsed_ms": round((time.perf_counter() - bucket_started_at) * 1000, 2),
                    },
                )
                # #endregion
            else:
                payload = load_json(chain_file)
                items = payload.get("data") or payload.get("病例相似链数据") or []
                items_count = len(items)
                for item in items:
                    current_key, meta = self._extract_case_identity(item)
                    if current_key != case_key:
                        continue
                    target_item = item
                    target_meta = meta
                    break
            if target_item is None:
                continue
            matched = True
            if record is None:
                record = self._build_testset_case_record(
                    case_key=case_key,
                    meta=target_meta,
                    scope_hint=config["scope_hint"],
                )
            if not record["diagnosis"]:
                record["diagnosis"] = self._extract_diagnosis(
                    target_item,
                    target_meta,
                    scope_hint=config["scope_hint"],
                )
            diagnosis_mapping = build_diagnosis_mapping_payload(record["diagnosis"], max_matches=1)
            chain_bundle = self._extract_chain_bundle(
                target_item,
                diagnosis_mapping=diagnosis_mapping,
                scope_hint=config["scope_hint"],
            )
            if chain_bundle["full"]:
                record[bucket] = chain_bundle["full"]
            if chain_bundle["preview"]:
                record[f"{bucket}_preview"] = chain_bundle["preview"]
            record[bucket.replace("_chain", "_groups")] = list(
                target_item.get("字段分组明细") or target_item.get("groups") or []
            )
            for field in ("file_name", "unique_id", "visit_id", "name"):
                if not record.get(field):
                    record[field] = normalize_text_value(target_meta.get(field, ""))
            # #region debug-point B:testset-detail-bucket-match
            _debug_emit(
                "B",
                "TestCaseRepository._load_case_detail:bucket_match",
                "[DEBUG] testset detail bucket matched",
                {
                    "testset_id": testset_id,
                    "case_key": case_key,
                    "bucket": bucket,
                    "items_count": items_count,
                    "file_size_bytes": chain_file_size,
                    "scan_mode": "large_json_fast_scan" if use_large_scan else "load_json",
                    "elapsed_ms": round((time.perf_counter() - bucket_started_at) * 1000, 2),
                },
            )
            # #endregion

        if not matched or record is None:
            # #region debug-point B:testset-detail-not-found
            _debug_emit(
                "B",
                "TestCaseRepository._load_case_detail:not_found",
                "[DEBUG] testset detail source scan found no match",
                {
                    "testset_id": testset_id,
                    "case_key": case_key,
                    "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
                },
            )
            # #endregion
            return None
        self._detail_cache[cache_key] = record
        self._save_disk_cache(detail_cache_path, signature, record)
        # #region debug-point B:testset-detail-built
        _debug_emit(
            "B",
            "TestCaseRepository._load_case_detail:built",
            "[DEBUG] testset detail built from source",
            {
                "testset_id": testset_id,
                "case_key": case_key,
                "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "diagnosis": record.get("diagnosis", ""),
            },
        )
        # #endregion
        return record

    def _quick_case_count(self, config: dict[str, Any]) -> int | None:
        directory = config.get("path")
        if not isinstance(directory, Path):
            return None
        for bucket in CHAIN_LABELS:
            chain_file = self._resolve_chain_file(directory, bucket)
            if chain_file is None:
                continue
            try:
                payload = load_json(chain_file)
            except Exception:
                return None
            items = payload.get("data") or payload.get("病例相似链数据") or []
            if isinstance(items, list):
                return len(items)
        return None

    def _sort_visit_id(self, value: str):
        text = str(value).strip()
        if text.isdigit():
            return (0, int(text))
        return (1, text)

    def _flatten_diagnosis_value(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            flattened: list[str] = []
            for item in value:
                flattened.extend(self._flatten_diagnosis_value(item))
            return flattened
        text = normalize_text_value(value)
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                return self._flatten_diagnosis_value(parsed)
        parts = [
            part.strip(" []'\"")
            for part in re.split(r"[;,；，]\s*|\s{2,}", text)
            if part.strip(" []'\"")
        ]
        if not parts:
            return [text]
        return parts

    def _collect_diagnosis_candidates(self, item: dict[str, Any], meta: dict[str, str]) -> list[str]:
        candidates: list[str] = []
        for value in (
            meta.get("diagnosis", ""),
            normalize_text_value((item.get("meta") or {}).get("diag") or ""),
            normalize_text_value((item.get("病例标识") or {}).get("主诊断") or ""),
        ):
            candidates.extend(self._flatten_diagnosis_value(value))

        groups = item.get("groups") or item.get("字段分组明细") or []
        for group in groups:
            details = group.get("d") or group.get("提取结果") or {}
            if not isinstance(details, dict):
                continue
            for key in (
                "主要诊断",
                "主诊断",
                "入院诊断",
                "出院诊断",
                "main_diag",
                "diag",
                "诊断信息",
                "诊断列表",
                "其他诊断",
                "other_diag",
            ):
                if key in details:
                    candidates.extend(self._flatten_diagnosis_value(details.get(key)))
        unique_candidates: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = normalize_text_value(candidate)
            if not normalized:
                continue
            if "/" in normalized:
                suffix = normalized.split("/", 1)[1].strip()
                if suffix:
                    normalized = suffix
            normalized = re.sub(r"\s+", " ", normalized).strip(" ,，;；")
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_candidates.append(normalized)
        return unique_candidates

    def _focus_respiratory_label(self, text: str) -> str:
        candidate = normalize_text_value(text)
        if not candidate:
            return ""
        exact_terms = [
            "慢性阻塞性肺疾病急性加重伴急性下呼吸道感染",
            "慢性阻塞性肺疾病急性加重",
            "慢性阻塞性肺病伴有急性加重",
            "慢性阻塞性肺疾病稳定期",
            "慢性阻塞性肺疾病",
            "慢性阻塞性肺病",
            "AECOPD",
            "COPD",
            "支气管扩张症合并感染",
            "支气管扩张伴感染",
            "支气管扩张症",
            "支气管扩张",
            "支气管哮喘急性发作",
            "支气管哮喘",
            "社区获得性肺炎",
            "医院获得性肺炎",
            "肺部感染",
            "重症肺炎",
            "肺炎",
            "II型呼吸衰竭",
            "I型呼吸衰竭",
            "呼吸衰竭",
            "肺栓塞",
            "胸腔积液",
            "气胸",
            "肺结核",
            "间质性肺疾病",
            "肺纤维化",
            "阻塞性睡眠呼吸暂停低通气综合征",
            "肥胖低通气综合征",
        ]
        for term in exact_terms:
            if term.lower() in candidate.lower():
                return term
        regex_patterns = [
            r"慢性阻塞性肺(?:疾病|病)[^,，;；]{0,24}",
            r"(?:I|II|ⅰ|ⅱ|1|2)型呼吸衰竭",
            r"支气管扩张[^,，;；]{0,20}",
            r"哮喘[^,，;；]{0,20}",
            r"肺炎[^,，;；]{0,20}",
            r"肺部感染[^,，;；]{0,20}",
            r"胸腔积液[^,，;；]{0,20}",
            r"气胸[^,，;；]{0,20}",
            r"肺栓塞[^,，;；]{0,20}",
            r"间质性肺疾病[^,，;；]{0,20}",
            r"肺纤维化[^,，;；]{0,20}",
        ]
        for pattern in regex_patterns:
            match = re.search(pattern, candidate, re.IGNORECASE)
            if match:
                return match.group(0).strip(" ,，;；")
        return candidate

    def _pick_best_diagnosis(self, candidates: list[str], scope_hint: str) -> str:
        if not candidates:
            return ""
        scope = str(scope_hint or "").strip()
        respiratory_keywords = (
            "慢性阻塞性肺",
            "慢阻肺",
            "copd",
            "aecopd",
            "呼吸衰竭",
            "肺炎",
            "肺部感染",
            "支气管扩张",
            "支扩",
            "哮喘",
            "肺气肿",
            "胸腔积液",
            "气胸",
            "肺结核",
            "间质性肺疾病",
            "肺纤维化",
            "高碳酸血症",
            "低氧血症",
            "睡眠呼吸暂停",
            "肺栓塞",
        )
        best_score = -10**9
        best_candidate = candidates[0]
        for candidate in candidates:
            candidate_lower = candidate.lower()
            score = 0
            has_respiratory_keyword = any(keyword in candidate_lower for keyword in respiratory_keywords)
            if has_respiratory_keyword:
                score += 4
            mapping: dict[str, Any] = {}
            should_map = has_respiratory_keyword or any(
                token in candidate_lower
                for token in ("肺", "呼吸", "支气管", "哮喘", "胸", "气道", "低氧", "高碳酸")
            )
            if should_map:
                mapping = build_diagnosis_mapping_payload(candidate, max_matches=1)
                if mapping.get("status") == "matched":
                    score += 8
            template = str(mapping.get("template", "")).strip()
            family = str(mapping.get("family", "")).strip()
            if scope == "ae":
                if template in {"copd_ae", "copd_infection", "resp_failure_type_1", "resp_failure_type_2"}:
                    score += 6
                if "急性加重" in candidate or "呼吸衰竭" in candidate:
                    score += 3
            elif scope == "chronic":
                if family == "airway_obstructive":
                    score += 5
                if template in {"copd_stable", "copd_ae", "copd_infection", "bronchiectasis_infection"}:
                    score += 3
            elif scope == "infection":
                if family == "respiratory_infection":
                    score += 5
            if "姓名" in candidate or len(candidate) <= 1:
                score -= 10
            score += min(len(candidate), 40) / 100.0
            if score > best_score:
                best_score = score
                best_candidate = candidate
        return best_candidate

    def _extract_case_identity(self, item: dict[str, Any]) -> tuple[str, dict[str, str]]:
        meta = item.get("meta") or item.get("病例标识") or {}
        file_name = normalize_text_value(meta.get("f") or meta.get("文件名") or "")
        unique_id = normalize_text_value(meta.get("id") or meta.get("唯一号") or "")
        visit_id = normalize_text_value(meta.get("vid") or meta.get("visit_id") or "")
        name = normalize_text_value(meta.get("name") or meta.get("姓名") or "")
        key = "|".join([file_name, unique_id, visit_id]).strip("|")
        if not key:
            return "", {}
        return key, {
            "case_id": normalize_text_value(
                meta.get("case_id") or meta.get("病例编号") or meta.get("caseId") or ""
            ),
            "file_name": file_name,
            "unique_id": unique_id,
            "visit_id": visit_id,
            "name": name,
            "diagnosis": normalize_text_value(meta.get("diag") or meta.get("主诊断") or ""),
        }

    def _build_render_focus_context(
        self,
        diagnosis_mapping: dict[str, Any] | None = None,
        scope_hint: str = "",
    ) -> dict[str, Any]:
        diagnosis_mapping = diagnosis_mapping or {}
        template_id = str(diagnosis_mapping.get("template", "")).strip()
        family = str(diagnosis_mapping.get("family", "")).strip()
        allowed_lab_ids: set[str] = set()
        allowed_imaging_ids: set[str] = set()
        allowed_drug_classes: set[str] = set()
        allowed_drug_intents: set[str] = set()
        if template_id:
            try:
                rule_set = get_resp_rule_set()
                template_config = rule_set.get_template(template_id)
            except Exception:
                template_config = None
            if template_config is not None:
                allowed_lab_ids.update(template_config.key_evidence.get("labs", ()))
                allowed_lab_ids.update(template_config.key_evidence.get("blood_gas", ()))
                allowed_lab_ids.update(template_config.key_evidence.get("pulmonary_function", ()))
                allowed_imaging_ids.update(template_config.key_evidence.get("imaging", ()))
                allowed_drug_classes.update(template_config.key_treatments.get("drug_classes", ()))
                allowed_drug_intents.update(template_config.key_treatments.get("supports", ()))
                allowed_drug_intents.update(template_config.key_treatments.get("procedures", ()))
        return {
            "template": template_id,
            "family": family,
            "scope_hint": str(scope_hint or "").strip(),
            "allowed_lab_ids": allowed_lab_ids,
            "allowed_imaging_ids": allowed_imaging_ids,
            "allowed_drug_classes": allowed_drug_classes,
            "allowed_drug_intents": allowed_drug_intents,
        }

    def _detail_item_text(self, item: Any) -> str:
        if isinstance(item, dict):
            parts = [
                normalize_text_value(item.get("检验项目") or item.get("p") or ""),
                normalize_text_value(item.get("检验指标名") or item.get("n") or item.get("医嘱名称") or ""),
                normalize_text_value(item.get("检验指标值") or item.get("v") or item.get("剂量") or item.get("d") or ""),
                normalize_text_value(item.get("单位") or item.get("u") or ""),
                normalize_text_value(item.get("用法") or item.get("r") or ""),
                normalize_text_value(item.get("频次") or item.get("f") or ""),
                normalize_text_value(item.get("药物分类") or item.get("c") or ""),
                normalize_text_value(item.get("内容") or ""),
                normalize_text_value(item.get("参考范围") or item.get("rr") or ""),
            ]
            merged = " ".join(part for part in parts if part)
            if merged:
                return merged
            return " ".join(
                normalize_text_value(value)
                for value in item.values()
                if normalize_text_value(value)
            )
        return normalize_text_value(item)

    def _is_respiratory_text(self, text: str) -> bool:
        normalized = normalize_text_value(text)
        if not normalized:
            return False
        lowered = normalized.lower()
        if self._is_respiratory_diagnosis_text(normalized):
            return True
        return any(
            keyword in lowered
            for keyword in (
                *self.RESP_HISTORY_KEYWORDS,
                *self.RESP_LAB_KEYWORDS,
                *self.RESP_IMAGING_KEYWORDS,
                *self.RESP_DRUG_KEYWORDS,
                *self.RESP_PROCEDURE_KEYWORDS,
            )
        )

    def _filter_focus_text(self, text: Any) -> str:
        normalized = normalize_text_value(text)
        if not normalized:
            return ""
        if self._is_respiratory_text(normalized):
            return normalized
        segments = [
            segment.strip()
            for segment in re.split(r"[；;。！？!\?\n]+", normalized)
            if segment.strip()
        ]
        kept = [segment for segment in segments if self._is_respiratory_text(segment)]
        return "；".join(kept)

    def _match_lookup_ids(self, text: str, pattern_key: str) -> set[str]:
        normalized = normalize_lookup_text(text)
        if not normalized:
            return set()
        lookup = get_resp_lookup_tables()
        return {
            item_id
            for alias, item_id, _ in lookup.get(pattern_key, [])
            if alias and alias in normalized
        }

    def _is_respiratory_diagnosis_text(self, text: str) -> bool:
        normalized = normalize_text_value(text)
        if not normalized:
            return False
        lowered = normalized.lower()
        if any(keyword in lowered for keyword in self.RESP_DIAG_KEYWORDS):
            return True
        should_map = any(token in lowered for token in ("肺", "呼吸", "支气管", "哮喘", "胸", "低氧", "高碳酸"))
        if not should_map:
            return False
        mapping = build_diagnosis_mapping_payload(normalized, max_matches=1)
        return str(mapping.get("status", "")).strip() == "matched"

    def _should_keep_focus_item(self, key: str, item: Any, focus_context: dict[str, Any]) -> bool:
        key_text = normalize_text_value(key)
        if key_text not in self.FOCUS_FILTER_KEYS:
            return True
        text = self._detail_item_text(item)
        if not text:
            return False
        lowered = text.lower()
        if key_text in {"norm_lab_cnt", "normal_lab_cnt", "icd"}:
            return False
        if key_text in {"main_diag", "dis_diag", "other_diag", "其他诊断", "诊断信息", "诊断列表"}:
            return self._is_respiratory_diagnosis_text(text)
        if key_text in {"complications", "并发症列表", "并发症"}:
            return self._is_respiratory_diagnosis_text(text)
        if key_text in {"主诉", "主要症状"}:
            # 主诉/主要症状为病例核心标识，只要存在即保留
            return bool(normalize_text_value(text))
        if key_text in {"现病史", "既往史", "个人史", "体格检查"}:
            return bool(self._filter_focus_text(text))
        if key_text in {"abn_lab", "异常检验", "检验项目列表", "normal_lab", "正常检验"}:
            if any(keyword in lowered for keyword in self.RESP_LAB_NEGATIVE_KEYWORDS):
                return False
            matched_lab_ids = self._match_lookup_ids(text, "lab_patterns")
            if matched_lab_ids & set(focus_context.get("allowed_lab_ids", set())):
                return True
            if any(item_id.startswith(("bg.", "pf.")) for item_id in matched_lab_ids):
                return True
            return any(keyword in lowered for keyword in self.RESP_LAB_KEYWORDS)
        if key_text in {"img", "exams", "影像检查", "影像检查列表"}:
            if any(keyword in lowered for keyword in self.RESP_IMAGING_NEGATIVE_KEYWORDS):
                return False
            matched_imaging_ids = self._match_lookup_ids(text, "imaging_patterns")
            if matched_imaging_ids & set(focus_context.get("allowed_imaging_ids", set())):
                return True
            return any(keyword in lowered for keyword in self.RESP_IMAGING_KEYWORDS)
        if key_text in {"drugs", "drug_names", "用药医嘱", "用药医嘱列表", "order", "出院医嘱"}:
            if any(keyword in lowered for keyword in self.RESP_DRUG_NEGATIVE_KEYWORDS):
                return False
            lookup = get_resp_lookup_tables()
            matched_drug_ids = self._match_lookup_ids(text, "drug_patterns")
            matched_classes = {
                lookup.get("drug_class_by_id", {}).get(drug_id, "")
                for drug_id in matched_drug_ids
                if lookup.get("drug_class_by_id", {}).get(drug_id, "")
            }
            matched_intents = {
                intent
                for drug_id in matched_drug_ids
                for intent in lookup.get("drug_intents_by_id", {}).get(drug_id, ())
                if intent
            }
            if matched_classes & set(focus_context.get("allowed_drug_classes", set())):
                return True
            if matched_intents & set(focus_context.get("allowed_drug_intents", set())):
                return True
            category = normalize_text_value(item.get("药物分类") or item.get("c") or "").lower() if isinstance(item, dict) else ""
            if category and any(keyword in category for keyword in ("支气管扩张", "糖皮质激素", "抗感染", "祛痰")):
                return True
            return any(keyword in lowered for keyword in self.RESP_DRUG_KEYWORDS)
        if key_text in {"surgeries"}:
            return any(keyword in lowered for keyword in self.RESP_PROCEDURE_KEYWORDS)
        return True

    def _extract_chain_bundle(
        self,
        item: dict[str, Any],
        *,
        diagnosis_mapping: dict[str, Any] | None = None,
        scope_hint: str = "",
    ) -> dict[str, str]:
        focus_context = self._build_render_focus_context(
            diagnosis_mapping=diagnosis_mapping,
            scope_hint=scope_hint,
        )
        groups = item.get("groups") or item.get("字段分组明细") or []
        full_parts = []
        preview_parts = []
        for group in groups:
            label = normalize_text_value(group.get("g") or group.get("字段分组") or "")
            full_text = self._render_group_text(group, mode="full", focus_context=focus_context)
            preview_text = self._render_group_text(group, mode="preview", focus_context=focus_context)
            if full_text:
                full_parts.append(f"{label}: {full_text}" if label else full_text)
            if preview_text:
                preview_parts.append(f"{label}: {preview_text}" if label else preview_text)
        full_text = "\n".join(full_parts)
        preview_text = "\n".join(preview_parts)
        bucket = self._infer_bucket_from_item(item)
        full_limit = self.FULL_CHAIN_CHAR_LIMITS.get(bucket, 2400)
        preview_limit = self.PREVIEW_CHAIN_CHAR_LIMITS.get(bucket, 420)
        direct = normalize_text_value(item.get("chain") or item.get("合并相似链条") or "")
        direct_full = shorten_text(direct, full_limit) if direct else ""
        direct_preview = shorten_text(direct, preview_limit) if direct else ""
        if full_text:
            full_text = shorten_text(full_text, full_limit)
        if preview_text:
            preview_text = shorten_text(preview_text, preview_limit)
        if direct_preview and (
            not preview_text
            or len(preview_text) >= preview_limit
            or len(preview_text) >= max(len(direct_preview) * 2, len(direct_preview) + 120)
        ):
            preview_text = direct_preview
        if full_text:
            return {
                "full": full_text,
                "preview": preview_text or full_text,
            }
        return {
            "full": direct_full,
            "preview": direct_preview if direct_full else direct_full,
        }

    def _infer_bucket_from_item(self, item: dict[str, Any]) -> str:
        label = normalize_text_value((item.get("meta") or {}).get("chain_type") or "")
        if label in CHAIN_MAP:
            return CHAIN_MAP[label]
        groups = item.get("groups") or item.get("字段分组明细") or []
        for group in groups:
            group_label = normalize_text_value(group.get("g") or group.get("字段分组") or "")
            if group_label in CHAIN_MAP:
                return CHAIN_MAP[group_label]
        return ""

    def _render_group_text(
        self,
        group: dict[str, Any],
        mode: str = "full",
        focus_context: dict[str, Any] | None = None,
    ) -> str:
        details = group.get("d") or group.get("提取结果") or {}
        rendered = self._render_detail_mapping(details, mode=mode, focus_context=focus_context)
        if rendered:
            return rendered
        return self._compact_render_text(group.get("s") or group.get("相似链条") or "", mode=mode)

    def _render_detail_mapping(
        self,
        details: Any,
        mode: str = "full",
        focus_context: dict[str, Any] | None = None,
    ) -> str:
        if not isinstance(details, dict):
            return self._compact_render_text(details, mode=mode)
        fields = []
        for key, value in details.items():
            rendered = self._render_detail_value(key, value, mode=mode, focus_context=focus_context)
            if not rendered:
                continue
            fields.append(f"{self._pretty_key(key)}: {rendered}")
        return "；".join(fields)

    def _render_detail_value(
        self,
        key: str,
        value: Any,
        mode: str = "full",
        focus_context: dict[str, Any] | None = None,
    ) -> str:
        if isinstance(value, dict):
            nested = []
            for sub_key, sub_value in value.items():
                rendered = self._render_detail_value(
                    sub_key,
                    sub_value,
                    mode=mode,
                    focus_context=focus_context,
                )
                if rendered:
                    nested.append(f"{self._pretty_key(sub_key)}={rendered}")
            return "；".join(nested)
        if isinstance(value, list):
            return self._render_detail_list(key, value, mode=mode, focus_context=focus_context)
        if focus_context and not self._should_keep_focus_item(key, value, focus_context):
            return ""
        # 主诉/主要症状是病例核心标识，必须完整保留，不能因呼吸道关键词过滤被丢弃
        if focus_context and normalize_text_value(key) in {
            "主诉",
            "主要症状",
        }:
            filtered = normalize_text_value(value)
            if not filtered:
                return ""
            return self._compact_render_text(filtered, mode=mode, key=key)
        if focus_context and normalize_text_value(key) in {
            "现病史",
            "既往史",
            "个人史",
            "体格检查",
        }:
            filtered = self._filter_focus_text(value)
            if not filtered:
                return ""
            return self._compact_render_text(filtered, mode=mode, key=key)
        return self._compact_render_text(value, mode=mode, key=key)

    def _render_detail_list(
        self,
        key: str,
        values: list[Any],
        mode: str = "full",
        focus_context: dict[str, Any] | None = None,
    ) -> str:
        cleaned = []
        for item in values:
            if focus_context and not self._should_keep_focus_item(key, item, focus_context):
                continue
            rendered = self._render_detail_list_item(item, mode=mode)
            if rendered:
                cleaned.append(rendered)
        cleaned = self._unique_preserve_order(cleaned)
        if not cleaned:
            return ""
        if mode == "full":
            return "；".join(cleaned)
        limit = self._list_limit(key, mode=mode)
        visible = cleaned[:limit]
        suffix = f" 等{len(cleaned)}项" if len(cleaned) > limit else ""
        return "；".join(visible) + suffix

    def _render_detail_list_item(self, item: Any, mode: str = "full") -> str:
        if isinstance(item, dict):
            return self._format_record(item, mode=mode)
        return normalize_text_value(item)

    def _unique_preserve_order(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        unique_items: list[str] = []
        for item in items:
            text = normalize_text_value(item)
            if not text or text in seen:
                continue
            seen.add(text)
            unique_items.append(text)
        return unique_items

    def _join_record_parts(self, parts: list[str], max_len: int | None = 220) -> str:
        merged = "，".join(part for part in parts if part)
        if max_len is None:
            return merged
        return shorten_text(merged, max_len)

    def _format_record(self, item: dict[str, Any], mode: str = "full") -> str:
        project = normalize_text_value(item.get("检验项目") or item.get("p") or "")
        metric = normalize_text_value(item.get("检验指标名") or item.get("n") or item.get("医嘱名称") or "")
        value = normalize_text_value(item.get("检验指标值") or item.get("v") or item.get("剂量") or item.get("d") or "")
        unit = normalize_text_value(item.get("单位") or item.get("u") or "")
        route = normalize_text_value(item.get("用法") or item.get("r") or "")
        freq = normalize_text_value(item.get("频次") or item.get("f") or "")
        category = normalize_text_value(item.get("药物分类") or item.get("c") or "")
        reference = normalize_text_value(item.get("参考范围") or item.get("rr") or "")
        report_time = normalize_text_value(item.get("报告日期") or item.get("dt") or item.get("开始时间") or "")
        outcome = normalize_text_value(item.get("结束时间") or item.get("outcome") or "")
        content = normalize_text_value(item.get("内容") or item.get("c") or "")
        flag = normalize_text_value(item.get("a") or item.get("flag") or "")
        flag_label = self._normalize_flag_label(flag)
        display_value = f"{value}{unit}" if value else unit
        if display_value and flag_label:
            display_value = f"{display_value}（{flag_label}）"

        max_len = self.FULL_RECORD_TEXT_MAX_CHARS if mode == "full" else 150

        if project.startswith("[") and content:
            return self._join_record_parts([f"{project}: {content}"], max_len=max_len)
        if metric.startswith("[") and content:
            return self._join_record_parts([f"{metric}: {content}"], max_len=max_len)

        if metric and (value or unit or route or freq or category or reference):
            head = f"{project}-{metric}" if project else metric
            if display_value:
                head += f"={display_value}"
            extras = [part for part in [route, freq, category, f"参考{reference}" if reference else "", report_time] if part]
            return self._join_record_parts([head, *extras], max_len=max_len)
        if metric and content and content != category:
            return self._join_record_parts([f"{metric}: {content}"], max_len=max_len)
        if metric:
            extras = [part for part in [route, freq, display_value, category, report_time, outcome] if part]
            return self._join_record_parts([metric, *extras], max_len=max_len)
        if content:
            return self._join_record_parts([content], max_len=max_len)

        parts = []
        for key, value in item.items():
            rendered = self._render_detail_value(key, value, mode=mode)
            if rendered:
                parts.append(f"{self._pretty_key(key)}={rendered}")
        merged = "；".join(self._unique_preserve_order(parts))
        if max_len is None:
            return merged
        return shorten_text(merged, max_len)

    def _compact_render_text(self, value: Any, mode: str = "full", key: str = "") -> str:
        text = normalize_text_value(value)
        if not text:
            return ""
        limit = self.FULL_TEXT_VALUE_MAX_CHARS if mode == "full" else self.PREVIEW_TEXT_VALUE_MAX_CHARS
        key_text = str(key)
        if key_text in {"体格检查", "现病史", "检查所见", "影像所见", "内容"}:
            limit = min(limit, 420 if mode == "full" else 150)
        return shorten_text(text, limit)

    def _normalize_flag_label(self, flag: str) -> str:
        text = normalize_text_value(flag).lower()
        if not text or text in {"0", "n", "normal"}:
            return ""
        if text in {"1", "abnormal", "abn", "a"}:
            return "异常"
        if text in {"h", "high", "up", "↑"}:
            return "升高"
        if text in {"l", "low", "down", "↓"}:
            return "降低"
        if text in {"+", "positive", "pos", "阳性"}:
            return "阳性"
        if text in {"-", "negative", "neg", "阴性"}:
            return "阴性"
        return normalize_text_value(flag)

    def _list_limit(self, key: str, mode: str = "full") -> int:
        key_text = str(key)
        if mode == "preview":
            return self.PREVIEW_DETAIL_LIST_LIMITS.get(key_text, 6)
        return self.FULL_DETAIL_LIST_LIMITS.get(key_text, 18)

    def _pretty_key(self, key: Any) -> str:
        text = normalize_text_value(key)
        if not text:
            return ""
        pretty_map = {
            "abn_lab": "异常检验",
            "normal_lab": "正常检验",
            "img": "影像检查",
            "other_diag": "其他诊断",
            "drugs": "用药医嘱",
            "surgeries": "手术治疗",
            "complications": "并发症",
            "stage": "疾病分期",
            "severity": "严重程度",
            "basis": "分期依据",
            "scores": "功能评分",
            "stay_days": "住院天数",
            "readmit": "再入院风险",
            "cnt": "数量",
            "principle": "治疗原则",
            "result": "治疗结果",
            "order": "出院医嘱",
            "dt": "日期",
            "outcome": "结局",
        }
        if text in pretty_map:
            return pretty_map[text]
        return text.replace("_", " ")

    def _extract_diagnosis(
        self,
        item: dict[str, Any],
        meta: dict[str, str],
        scope_hint: str = "",
    ) -> str:
        candidates = self._collect_diagnosis_candidates(item, meta)
        best = self._pick_best_diagnosis(candidates, scope_hint=scope_hint)
        return self._focus_respiratory_label(best)

    def _build_case_label(self, record: dict[str, Any]) -> str:
        prefix = (
            normalize_text_value(record.get("diagnosis"))
            or normalize_text_value(record.get("file_name"))
            or normalize_text_value(record.get("case_key"))
        )
        suffix = " / ".join(
            part
            for part in [
                f"visit {record['visit_id']}" if normalize_text_value(record.get("visit_id")) else "",
                normalize_text_value(record.get("unique_id", "")),
            ]
            if part
        )
        return f"{prefix} ({suffix})" if suffix else str(prefix)

    def list_testsets(self) -> dict[str, Any]:
        rows = []
        for testset_id, config in self.testsets.items():
            cached = self._cache.get(testset_id)
            case_count = cached["case_count"] if cached else self._quick_case_count(config)
            rows.append(
                {
                    "id": config["id"],
                    "label": config["label"],
                    "case_count": case_count,
                    "scope_hint": config["scope_hint"],
                }
            )
        return {"rows": rows}

    def list_cases(self, testset_id: str) -> dict[str, Any]:
        payload = self._load_testset_rows(testset_id)
        if payload is None:
            return {"error": "testset not found"}
        return {
            "testset": {
                "id": payload["id"],
                "label": payload["label"],
                "case_count": payload["case_count"],
                "scope_hint": payload["scope_hint"],
            },
            "rows": payload["rows"],
        }

    def case_detail(self, testset_id: str, case_key: str) -> dict[str, Any] | None:
        started_at = time.perf_counter()
        # #region debug-point B:testset-case-detail-entry
        _debug_emit(
            "B",
            "TestCaseRepository.case_detail:entry",
            "[DEBUG] testset case detail requested",
            {"testset_id": testset_id, "case_key": case_key},
        )
        # #endregion
        payload = self._load_testset_rows(testset_id)
        if payload is None:
            # #region debug-point B:testset-case-detail-missing-testset
            _debug_emit(
                "B",
                "TestCaseRepository.case_detail:missing_testset",
                "[DEBUG] testset rows missing",
                {
                    "testset_id": testset_id,
                    "case_key": case_key,
                    "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
                },
            )
            # #endregion
            return None
        record = self._load_case_detail(testset_id, case_key)
        if record is None:
            # #region debug-point B:testset-case-detail-missing-case
            _debug_emit(
                "B",
                "TestCaseRepository.case_detail:missing_case",
                "[DEBUG] test case record missing",
                {
                    "testset_id": testset_id,
                    "case_key": case_key,
                    "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
                },
            )
            # #endregion
            return None
        diagnosis_mapping = build_diagnosis_mapping_payload(record["diagnosis"])
        auto_dataset_ids = resolve_auto_dataset_ids(
            self.dataset_configs,
            diagnosis_text=record["diagnosis"],
            diagnosis_mapping=diagnosis_mapping,
        )
        recommended_scope_hint = auto_dataset_ids[0] if auto_dataset_ids else payload["scope_hint"]
        # #region debug-point B:testset-case-detail-return
        _debug_emit(
            "B",
            "TestCaseRepository.case_detail:return",
            "[DEBUG] testset case detail resolved",
            {
                "testset_id": testset_id,
                "case_key": case_key,
                "diagnosis": record["diagnosis"],
                "recommended_scope_hint": recommended_scope_hint,
                "testset_scope_hint": payload["scope_hint"],
                "auto_dataset_ids": auto_dataset_ids[:5],
                "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
            },
        )
        # #endregion
        return {
            "testset_id": payload["id"],
            "testset_label": payload["label"],
            "scope_hint": recommended_scope_hint,
            "testset_scope_hint": payload["scope_hint"],
            "case_key": record["case_key"],
            "case_id": record.get("case_id", ""),
            "case_label": self._build_case_label(record),
            "file_name": record["file_name"],
            "unique_id": record["unique_id"],
            "visit_id": record["visit_id"],
            "diagnosis": record["diagnosis"],
            "history_chain": record["history_chain"],
            "history_chain_preview": record["history_chain_preview"] or record["history_chain"],
            "history_groups": record.get("history_groups", []),
            "evidence_chain": record["evidence_chain"],
            "evidence_chain_preview": record["evidence_chain_preview"] or record["evidence_chain"],
            "evidence_groups": record.get("evidence_groups", []),
            "care_chain": record["care_chain"],
            "care_chain_preview": record["care_chain_preview"] or record["care_chain"],
            "care_groups": record.get("care_groups", []),
            "severity_chain": record["severity_chain"],
            "severity_chain_preview": record["severity_chain_preview"] or record["severity_chain"],
            "severity_groups": record.get("severity_groups", []),
        }


class SimilarFeedbackStore:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "similar_case_feedback_events.jsonl"
        self.summary_json_path = self.output_dir / "similar_case_feedback_summary.json"
        self.summary_md_path = self.output_dir / "similar_case_feedback_summary.md"
        self._lock = threading.Lock()

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        event = {
            "event_id": uuid.uuid4().hex,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query_id": str(payload.get("query_id", "")).strip(),
            "query_text": str(payload.get("query_text", "")).strip(),
            "search_scope": str(payload.get("search_scope", "")).strip(),
            "testset_id": str(payload.get("testset_id", "")).strip(),
            "testset_label": str(payload.get("testset_label", "")).strip(),
            "query_case_key": str(payload.get("query_case_key", "")).strip(),
            "query_case_label": str(payload.get("query_case_label", "")).strip(),
            "rank": int(payload.get("rank", 0) or 0),
            "rated_case_id": str(payload.get("rated_case_id", "")).strip(),
            "rated_case_label": str(payload.get("rated_case_label", "")).strip(),
            "rated_dataset_id": str(payload.get("rated_dataset_id", "")).strip(),
            "rated_dataset_label": str(payload.get("rated_dataset_label", "")).strip(),
            "final_score": float(payload.get("final_score", 0.0) or 0.0),
            "base_score": float(payload.get("base_score", 0.0) or 0.0),
            "rerank_score": float(payload.get("rerank_score", 0.0) or 0.0),
            "human_score": int(payload.get("human_score", 0) or 0),
            "human_note": str(payload.get("human_note", "")).strip(),
        }
        if event["human_score"] < 1 or event["human_score"] > 5:
            raise ValueError("human_score must be between 1 and 5")

        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            summary = self._regenerate_summary()

        return {
            "ok": True,
            "event_id": event["event_id"],
            "summary": summary,
            "summary_json_path": str(self.summary_json_path),
            "summary_md_path": str(self.summary_md_path),
        }

    def get_summary(self) -> dict[str, Any]:
        if self.summary_json_path.exists():
            return load_json(self.summary_json_path)
        return self._regenerate_summary()

    def _load_events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        events = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def _aggregate_scores(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"count": 0, "avg_score": 0.0}
        total = sum(float(item.get("human_score", 0) or 0) for item in items)
        return {"count": len(items), "avg_score": round(total / len(items), 4)}

    def _regenerate_summary(self) -> dict[str, Any]:
        events = self._load_events()
        score_distribution = {str(score): 0 for score in range(1, 6)}
        by_testset: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        by_query_case: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        by_recommended_case: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

        for event in events:
            score_distribution[str(int(event.get("human_score", 0) or 0))] = (
                score_distribution.get(str(int(event.get("human_score", 0) or 0)), 0) + 1
            )
            testset_label = event.get("testset_label") or event.get("testset_id") or "未指定测试集"
            by_testset[testset_label].append(event)
            query_case_label = event.get("query_case_label") or event.get("query_case_key") or "未指定测试病例"
            by_query_case[query_case_label].append(event)
            recommended_label = " / ".join(
                part
                for part in [
                    event.get("rated_dataset_label") or event.get("rated_dataset_id") or "",
                    event.get("rated_case_id") or "",
                    event.get("rated_case_label") or "",
                ]
                if part
            ) or "未识别推荐病例"
            by_recommended_case[recommended_label].append(event)

        summary = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_ratings": len(events),
            "avg_score": round(
                sum(float(item.get("human_score", 0) or 0) for item in events) / len(events),
                4,
            )
            if events
            else 0.0,
            "score_distribution": score_distribution,
            "by_testset": [
                {"label": label, **self._aggregate_scores(items)}
                for label, items in sorted(
                    by_testset.items(),
                    key=lambda pair: (-len(pair[1]), pair[0]),
                )
            ],
            "by_query_case": [
                {"label": label, **self._aggregate_scores(items)}
                for label, items in sorted(
                    by_query_case.items(),
                    key=lambda pair: (-len(pair[1]), pair[0]),
                )[:20]
            ],
            "by_recommended_case": [
                {"label": label, **self._aggregate_scores(items)}
                for label, items in sorted(
                    by_recommended_case.items(),
                    key=lambda pair: (-len(pair[1]), pair[0]),
                )[:20]
            ],
            "latest_events": events[-20:],
        }
        self.summary_json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.summary_md_path.write_text(
            self._render_markdown_summary(summary),
            encoding="utf-8",
        )
        return summary

    def _render_markdown_summary(self, summary: dict[str, Any]) -> str:
        lines = [
            "# 相似病例检索人工评分统计",
            "",
            f"- 生成时间: {summary.get('generated_at', '-')}",
            f"- 总评分次数: {summary.get('total_ratings', 0)}",
            f"- 平均分: {summary.get('avg_score', 0.0)}",
            "",
            "## 评分分布",
            "",
        ]
        for score, count in summary.get("score_distribution", {}).items():
            lines.append(f"- {score} 分: {count}")

        lines.extend(["", "## 按测试集汇总", ""])
        for item in summary.get("by_testset", []):
            lines.append(
                f"- {item['label']}: {item['count']} 次, 平均分 {item['avg_score']}"
            )

        lines.extend(["", "## 按测试病例汇总", ""])
        for item in summary.get("by_query_case", []):
            lines.append(
                f"- {item['label']}: {item['count']} 次, 平均分 {item['avg_score']}"
            )

        lines.extend(["", "## 按推荐病例汇总", ""])
        for item in summary.get("by_recommended_case", []):
            lines.append(
                f"- {item['label']}: {item['count']} 次, 平均分 {item['avg_score']}"
            )

        lines.extend(["", "## 最近 20 条评分", ""])
        for item in summary.get("latest_events", []):
            note = shorten_text(item.get("human_note", ""), 80) if item.get("human_note") else ""
            note_part = f" | 备注: {note}" if note else ""
            lines.append(
                "- "
                + " | ".join(
                    part
                    for part in [
                        item.get("timestamp", "-"),
                        item.get("testset_label") or item.get("testset_id") or "未指定测试集",
                        item.get("query_case_label") or item.get("query_case_key") or "未指定测试病例",
                        f"Top {item.get('rank', 0)}",
                        item.get("rated_case_id") or "未识别推荐病例",
                        f"{item.get('human_score', 0)} 分",
                    ]
                    if part
                )
                + note_part
            )
        return "\n".join(lines) + "\n"


class DashboardStore:
    def __init__(self):
        self.dataset_configs = build_dataset_configs()
        self.datasets: dict[str, DatasetStore] = {}
        self.testcase_repo = TestCaseRepository(TESTSET_ROOT, self.dataset_configs)
        self.feedback_store = SimilarFeedbackStore(SIMILAR_FEEDBACK_DIR)
        self.default_dataset_id = (
            "chronic"
            if "chronic" in self.dataset_configs
            else next(iter(self.dataset_configs))
        )

    def get_dataset(self, dataset_id: str | None) -> DatasetStore:
        target_id = dataset_id or self.default_dataset_id
        if target_id not in self.dataset_configs:
            target_id = self.default_dataset_id
        if target_id not in self.datasets:
            config = self.dataset_configs[target_id]
            self.datasets[target_id] = DatasetStore(
                dataset_id=target_id,
                label=config["label"],
                working_dir=config["working_dir"],
                custom_kg_path=config["custom_kg"],
            )
        return self.datasets[target_id]

    def meta(self, dataset_id: str | None):
        current = self.get_dataset(dataset_id)
        payload = current.meta()
        dataset_rows = []
        for ds_id, config in self.dataset_configs.items():
            if ds_id in self.datasets:
                store = self.datasets[ds_id]
                dataset_rows.append(
                    {
                        "id": ds_id,
                        "label": store.label,
                        "graph_nodes": store.graph.number_of_nodes(),
                        "graph_edges": store.graph.number_of_edges(),
                        "cases": len(store.case_index["case_rows"]),
                        "loaded": True,
                        "group": config.get("group", ""),
                    }
                )
                continue
            stats = config.get("stats") or {}
            dataset_rows.append(
                {
                    "id": ds_id,
                    "label": config["label"],
                    "graph_nodes": int(stats.get("graph_nodes", 0) or 0),
                    "graph_edges": int(stats.get("graph_edges", 0) or 0),
                    "cases": int(
                        stats.get("cases", stats.get("case_count", 0)) or 0
                    ),
                    "loaded": False,
                    "group": config.get("group", ""),
                }
            )
        payload["datasets"] = dataset_rows
        payload["testset_count"] = len(self.testcase_repo.testsets)
        return payload

    def list_testsets(self):
        return self.testcase_repo.list_testsets()

    def testset_cases(self, testset_id: str):
        return self.testcase_repo.list_cases(testset_id)

    def testset_case_detail(self, testset_id: str, case_key: str):
        return self.testcase_repo.case_detail(testset_id, case_key)

    def save_feedback(self, payload: dict[str, Any]):
        return self.feedback_store.submit(payload)

    def feedback_summary(self):
        return self.feedback_store.get_summary()

    def _hydrate_query_payload_from_testset(self, payload: dict[str, Any]) -> dict[str, Any]:
        hydrated = dict(payload)
        testset_id = str(hydrated.get("testset_id", "")).strip()
        case_key = str(hydrated.get("query_case_key", "")).strip()
        if not testset_id or not case_key:
            return hydrated
        detail = self.testset_case_detail(testset_id, case_key)
        if not isinstance(detail, dict):
            return hydrated
        if not str(hydrated.get("diagnosis", "")).strip():
            hydrated["diagnosis"] = detail.get("diagnosis", "")
        for bucket in CHAIN_LABELS:
            if not str(hydrated.get(bucket, "")).strip():
                hydrated[bucket] = detail.get(bucket, "")
            group_key = bucket.replace("_chain", "_groups")
            if not hydrated.get(group_key):
                hydrated[group_key] = detail.get(group_key, [])
        if not str(hydrated.get("query_case_id", "")).strip() and detail.get("case_id"):
            hydrated["query_case_id"] = detail.get("case_id", "")
        return hydrated

    def similar_cases(self, payload: dict[str, Any], top_k: int):
        payload = self._hydrate_query_payload_from_testset(payload)
        query_id = uuid.uuid4().hex
        request_started_at = time.perf_counter()
        merged_results: list[dict[str, Any]] = []
        strategies: list[str] = []
        embedding_errors: list[str] = []
        rerank_errors: list[str] = []
        query_diagnosis_mapping = build_diagnosis_mapping_payload(
            str(payload.get("diagnosis", "")).strip()
        )
        query_chain_texts = {
            bucket: str(payload.get(bucket, "")).strip()
            for bucket in CHAIN_LABELS
        }
        query_structured_profile = extract_structured_profile(
            diagnosis_text=str(payload.get("diagnosis", "")).strip(),
            chain_texts=query_chain_texts,
            diagnosis_mapping=query_diagnosis_mapping,
        )
        query_effective_structured_profile, query_denoise_report = apply_template_profile_denoise(
            query_structured_profile,
            str(query_diagnosis_mapping.get("template", "")).strip(),
        )
        query_core_chain_texts = build_semantic_core_chain_texts(
            query_effective_structured_profile,
            query_diagnosis_mapping,
        )
        active_query_buckets = [
            bucket
            for bucket in CHAIN_LABELS
            if str(payload.get(bucket, "")).strip()
        ]
        semantic_weight_plan = build_semantic_weight_plan(
            query_mapping=query_diagnosis_mapping,
            active_buckets=active_query_buckets,
        )
        requested_dataset_scope = str(payload.get("dataset_scope", "all") or "all").strip()
        dataset_scope = requested_dataset_scope
        if dataset_scope == "all":
            auto_dataset_ids = resolve_auto_dataset_ids(
                self.dataset_configs,
                diagnosis_text=str(payload.get("diagnosis", "")).strip(),
                diagnosis_mapping=query_diagnosis_mapping,
            )
            # Filter out large test datasets (test17_*) to avoid heavy mmap loading
            auto_dataset_ids = [did for did in auto_dataset_ids if not did.startswith("test17_")]
            if auto_dataset_ids:
                dataset_scope = "auto"
                target_datasets = [
                    (dataset_id, self.get_dataset(dataset_id))
                    for dataset_id in auto_dataset_ids
                ]
            else:
                # Also filter fallback (all datasets) to skip test17_
                fallback_ids = [did for did in self.dataset_configs if not did.startswith("test17_")]
                target_datasets = [
                    (dataset_id, self.get_dataset(dataset_id))
                    for dataset_id in fallback_ids
                ]
        else:
            target_store = self.get_dataset(dataset_scope)
            target_datasets = [(target_store.dataset_id, target_store)]

        # #region debug-point A:dashboard-similar-cases-entry
        _debug_emit(
            "A",
            "DashboardStore.similar_cases:entry",
            "dashboard similar-cases request started",
            {
                "traceId": query_id,
                "requested_dataset_scope": requested_dataset_scope,
                "dataset_scope": dataset_scope,
                "target_dataset_ids": [dataset_id for dataset_id, _ in target_datasets],
                "top_k": top_k,
                "diagnosis": str(payload.get("diagnosis", "")).strip(),
            },
        )
        # #endregion
        for dataset_id, dataset in target_datasets:
            dataset_payload = dict(payload)
            dataset_payload["dataset"] = dataset_id
            dataset_started_at = time.perf_counter()
            dataset_result = dataset.similar_cases(dataset_payload, top_k=top_k)
            # #region debug-point A:dashboard-similar-cases-dataset
            _debug_emit(
                "A",
                "DashboardStore.similar_cases:dataset",
                "dataset similarity finished",
                {
                    "traceId": query_id,
                    "dataset_id": dataset_id,
                    "elapsed_ms": round((time.perf_counter() - dataset_started_at) * 1000, 2),
                    "strategy": dataset_result.get("strategy", ""),
                    "embedding_error": dataset_result.get("embedding_error", ""),
                    "rerank_error": dataset_result.get("rerank_error", ""),
                    "result_count": len(dataset_result.get("results", [])),
                },
            )
            # #endregion
            strategies.append(f"{dataset.label}: {dataset_result.get('strategy', '-')}")
            if dataset_result.get("embedding_error"):
                embedding_errors.append(
                    f"{dataset.label}: {dataset_result.get('embedding_error', '')}"
                )
            if dataset_result.get("rerank_error"):
                rerank_errors.append(
                    f"{dataset.label}: {dataset_result.get('rerank_error', '')}"
                )
            for item in dataset_result.get("results", []):
                merged_item = dict(item)
                merged_item["dataset_id"] = dataset_id
                merged_item["dataset_label"] = dataset.label
                merged_results.append(merged_item)

        merged_results.sort(key=lambda item: item.get("final_score", 0.0), reverse=True)
        query_text = "\n".join(
            part
            for part in [
                f"主诊断: {str(payload.get('diagnosis', '')).strip()}"
                if str(payload.get("diagnosis", "")).strip()
                else "",
                f"病史链: {str(payload.get('history_chain', '')).strip()}"
                if str(payload.get("history_chain", "")).strip()
                else "",
                f"证据链: {str(payload.get('evidence_chain', '')).strip()}"
                if str(payload.get("evidence_chain", "")).strip()
                else "",
                f"诊疗链: {str(payload.get('care_chain', '')).strip()}"
                if str(payload.get("care_chain", "")).strip()
                else "",
                f"严重度链: {str(payload.get('severity_chain', '')).strip()}"
                if str(payload.get("severity_chain", "")).strip()
                else "",
                f"补充说明: {str(payload.get('extra_note', '')).strip()}"
                if str(payload.get("extra_note", "")).strip()
                else "",
            ]
            if part
        )
        response_payload = {
            "query_id": query_id,
            "query_text": query_text,
            "query_diagnosis_mapping": query_diagnosis_mapping,
            "query_structured_profile": query_structured_profile,
            "query_effective_structured_profile": query_effective_structured_profile,
            "query_denoise_report": query_denoise_report,
            "query_core_chain_texts": query_core_chain_texts,
            "semantic_weight_plan": semantic_weight_plan,
            "requested_dataset_scope": requested_dataset_scope,
            "resolved_dataset_scope": dataset_scope,
            "resolved_dataset_ids": [dataset_id for dataset_id, _ in target_datasets],
            "strategy": " | ".join(strategies),
            "embedding_error": " | ".join(embedding_errors),
            "rerank_error": " | ".join(rerank_errors),
            "results": merged_results[:top_k],
        }
        # #region debug-point A:dashboard-similar-cases-return
        _debug_emit(
            "A",
            "DashboardStore.similar_cases:return",
            "dashboard similar-cases request finished",
            {
                "traceId": query_id,
                "elapsed_ms": round((time.perf_counter() - request_started_at) * 1000, 2),
                "strategy": response_payload.get("strategy", ""),
                "embedding_error": response_payload.get("embedding_error", ""),
                "rerank_error": response_payload.get("rerank_error", ""),
                "merged_result_count": len(response_payload.get("results", [])),
            },
        )
        # #endregion
        return response_payload


def make_handler(store: DashboardStore, html_file: Path):
    html = html_file.read_text(encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request_started_at = time.perf_counter()
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            dataset_id = params.get("dataset", [store.default_dataset_id])[0]
            # #region debug-point C:http-get-entry
            _debug_emit(
                "C",
                "Handler.do_GET:entry",
                "[DEBUG] http get request started",
                {"path": parsed.path, "dataset_id": dataset_id, "query": parsed.query},
            )
            # #endregion

            if parsed.path == "/":
                text_response(self, html)
                return
            if parsed.path == "/api/meta":
                payload = store.meta(dataset_id)
                # #region debug-point C:http-get-meta-ready
                _debug_emit(
                    "C",
                    "Handler.do_GET:meta_ready",
                    "[DEBUG] meta payload ready",
                    {
                        "path": parsed.path,
                        "dataset_id": dataset_id,
                        "elapsed_ms": round((time.perf_counter() - request_started_at) * 1000, 2),
                        "payload_keys": sorted(payload.keys()),
                    },
                )
                # #endregion
                json_response(self, payload)
                return
            if parsed.path == "/api/testsets":
                json_response(self, store.list_testsets())
                return
            if parsed.path == "/api/testset-cases":
                testset_id = params.get("testset_id", [""])[0]
                payload = store.testset_cases(testset_id)
                status = 404 if payload.get("error") else 200
                json_response(self, payload, status=status)
                return
            if parsed.path == "/api/testset-case":
                testset_id = params.get("testset_id", [""])[0]
                case_key = params.get("case_key", [""])[0]
                detail = store.testset_case_detail(testset_id, case_key)
                # #region debug-point C:http-get-testset-case-ready
                _debug_emit(
                    "C",
                    "Handler.do_GET:testset_case_ready",
                    "[DEBUG] testset-case payload ready",
                    {
                        "testset_id": testset_id,
                        "case_key": case_key,
                        "detail_found": detail is not None,
                        "elapsed_ms": round((time.perf_counter() - request_started_at) * 1000, 2),
                    },
                )
                # #endregion
                if detail is None:
                    json_response(self, {"error": "test case not found"}, status=404)
                    return
                json_response(self, detail)
                return
            if parsed.path == "/api/similar-feedback-summary":
                json_response(self, store.feedback_summary())
                return
            current = store.get_dataset(dataset_id)
            if parsed.path == "/api/graph":
                max_nodes = int(params.get("max_nodes", ["120"])[0] or 120)
                json_response(self, current.graph_sample(max_nodes=max_nodes))
                return
            if parsed.path == "/api/graph-expand":
                center = params.get("center", [""])[0]
                hops = int(params.get("hops", ["1"])[0] or 1)
                max_nodes = int(params.get("max_nodes", ["60"])[0] or 60)
                json_response(self, current.graph_expand(center=center, hops=hops, max_nodes=max_nodes))
                return
            if parsed.path == "/api/cases":
                json_response(self, current.cases_rows())
                return
            if parsed.path == "/api/case":
                case_id = params.get("case_id", [""])[0]
                detail = current.case_detail(case_id)
                if detail is None:
                    json_response(self, {"error": "case not found"}, status=404)
                    return
                json_response(self, detail)
                return
            if parsed.path == "/api/search-nodes":
                query = params.get("q", [""])[0]
                json_response(self, current.search_nodes(query))
                return
            json_response(self, {"error": "not found"}, status=404)

        def do_POST(self):
            parsed = urlparse(self.path)
            request_started_at = time.perf_counter()
            # #region debug-point A:http-post-entry
            _debug_emit(
                "A",
                "Handler.do_POST:entry",
                "http post received",
                {
                    "path": parsed.path,
                    "content_length": int(self.headers.get("Content-Length", "0") or 0),
                },
            )
            # #endregion
            if parsed.path not in {"/api/similar-cases", "/api/similar-feedback"}:
                json_response(self, {"error": "not found"}, status=404)
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(content_length) if content_length else b"{}"
            body = json.loads(payload.decode("utf-8"))
            # #region debug-point A:http-post-body
            _debug_emit(
                "A",
                "Handler.do_POST:body",
                "http post body parsed",
                {
                    "path": parsed.path,
                    "elapsed_ms": round((time.perf_counter() - request_started_at) * 1000, 2),
                    "body_keys": sorted(body.keys()),
                },
            )
            # #endregion
            if parsed.path == "/api/similar-cases":
                top_k = int(body.get("top_k", 5) or 5)
                json_response(self, store.similar_cases(body, top_k=top_k))
                return
            try:
                json_response(self, store.save_feedback(body))
            except Exception as exc:
                json_response(self, {"error": str(exc)}, status=400)

        def log_message(self, fmt: str, *args):
            return

    return Handler


def parse_args():
    parser = argparse.ArgumentParser(description="Run the multi-disease HyperGraphRAG dashboard server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--html-file", type=Path, default=DEFAULT_HTML_FILE)
    return parser.parse_args()


def main():
    args = parse_args()
    store = DashboardStore()
    handler = make_handler(store, args.html_file.resolve())
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Dashboard running at http://{args.host}:{args.port}/")
    for ds_id, dataset in store.datasets.items():
        print(
            f"[{ds_id}] cases={len(dataset.case_index['case_rows'])} "
            f"nodes={dataset.graph.number_of_nodes()} edges={dataset.graph.number_of_edges()}"
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
