from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .loader import get_default_rule_set
from .models import DiagnosisMappingEntry, RespiratoryRuleSet

NON_ALNUM_RE = re.compile(r"[^0-9a-z\u4e00-\u9fff]+")

MATCH_PRIORITY = {
    "exact": 4,
    "pattern_contains_query": 3,
    "query_contains_pattern": 2,
}

MATCH_BASE_SCORE = {
    "exact": 1.00,
    "pattern_contains_query": 0.88,
    "query_contains_pattern": 0.80,
}


def normalize_diagnosis_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower().strip()
    compact = NON_ALNUM_RE.sub("", normalized)
    return compact


@dataclass(frozen=True)
class DiagnosisMatch:
    mapping: DiagnosisMappingEntry
    matched_pattern: str
    match_method: str
    match_score: float


@dataclass(frozen=True)
class DiagnosisMappingResult:
    query_text: str
    normalized_query: str
    matches: tuple[DiagnosisMatch, ...]

    @property
    def primary_match(self) -> DiagnosisMatch | None:
        return self.matches[0] if self.matches else None

    @property
    def primary_template(self) -> str:
        match = self.primary_match
        return match.mapping.template if match else ""

    @property
    def primary_family(self) -> str:
        match = self.primary_match
        return match.mapping.family if match else ""

    @property
    def primary_concept_id(self) -> str:
        match = self.primary_match
        return match.mapping.concept_id if match else ""


class DiagnosisMapper:
    def __init__(self, rule_set: RespiratoryRuleSet):
        self.rule_set = rule_set
        self._compiled = [
            {
                "mapping": mapping,
                "patterns": tuple(
                    (pattern, normalize_diagnosis_text(pattern))
                    for pattern in mapping.raw_patterns
                    if normalize_diagnosis_text(pattern)
                ),
            }
            for mapping in rule_set.diagnosis_mappings
        ]

    def map(self, diagnosis_text: str, max_matches: int = 5) -> DiagnosisMappingResult:
        query_text = str(diagnosis_text or "").strip()
        normalized_query = normalize_diagnosis_text(query_text)
        if not normalized_query:
            return DiagnosisMappingResult(query_text=query_text, normalized_query="", matches=())

        matches: list[DiagnosisMatch] = []
        seen: set[tuple[str, str]] = set()

        for item in self._compiled:
            mapping: DiagnosisMappingEntry = item["mapping"]
            best: DiagnosisMatch | None = None
            for raw_pattern, normalized_pattern in item["patterns"]:
                match_method = self._detect_match_method(normalized_query, normalized_pattern)
                if not match_method:
                    continue
                match_score = self._build_match_score(
                    mapping=mapping,
                    match_method=match_method,
                    normalized_query=normalized_query,
                    normalized_pattern=normalized_pattern,
                )
                candidate = DiagnosisMatch(
                    mapping=mapping,
                    matched_pattern=raw_pattern,
                    match_method=match_method,
                    match_score=match_score,
                )
                if best is None or self._match_sort_key(candidate) > self._match_sort_key(best):
                    best = candidate

            if best is None:
                continue
            dedupe_key = (best.mapping.concept_id, best.match_method)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            matches.append(best)

        matches.sort(key=self._match_sort_key, reverse=True)
        limited = tuple(matches[: max(1, int(max_matches))]) if matches else ()
        return DiagnosisMappingResult(
            query_text=query_text,
            normalized_query=normalized_query,
            matches=limited,
        )

    def route_template(self, diagnosis_text: str, fallback_template: str = "") -> str:
        result = self.map(diagnosis_text, max_matches=1)
        if result.primary_template:
            return result.primary_template
        return str(fallback_template or "").strip()

    def route_family(self, diagnosis_text: str, fallback_family: str = "") -> str:
        result = self.map(diagnosis_text, max_matches=1)
        if result.primary_family:
            return result.primary_family
        return str(fallback_family or "").strip()

    def _detect_match_method(self, normalized_query: str, normalized_pattern: str) -> str:
        if not normalized_query or not normalized_pattern:
            return ""
        if normalized_query == normalized_pattern:
            return "exact"
        if normalized_query in normalized_pattern:
            return "pattern_contains_query"
        if normalized_pattern in normalized_query:
            return "query_contains_pattern"
        return ""

    def _build_match_score(
        self,
        mapping: DiagnosisMappingEntry,
        match_method: str,
        normalized_query: str,
        normalized_pattern: str,
    ) -> float:
        base_score = MATCH_BASE_SCORE.get(match_method, 0.0)
        specificity_bonus = min(0.08, len(normalized_pattern) / max(24.0, len(normalized_query) or 1.0))
        confidence_bonus = min(0.10, max(0.0, float(mapping.confidence)) * 0.10)
        return round(min(1.0, base_score + specificity_bonus + confidence_bonus), 6)

    def _match_sort_key(self, match: DiagnosisMatch) -> tuple[float, float, float, str]:
        pattern_length = len(normalize_diagnosis_text(match.matched_pattern))
        return (
            float(MATCH_PRIORITY.get(match.match_method, 0)),
            float(match.match_score),
            float(pattern_length),
            match.mapping.normalized_text,
        )


def get_default_diagnosis_mapper(
    version: str = "v1",
    versions_root: str | Path | None = None,
) -> DiagnosisMapper:
    rule_set = get_default_rule_set(version=version, versions_root=str(versions_root) if versions_root else None)
    return DiagnosisMapper(rule_set)
