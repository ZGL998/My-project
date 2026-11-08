from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .models import (
    ConceptDefinition,
    DiagnosisMappingEntry,
    DrugDefinition,
    FamilyDefinition,
    GovernanceMeta,
    ImagingDefinition,
    LabDefinition,
    RespiratoryRuleSet,
    ScoringConfig,
    TemplateConfig,
    TemplateRule,
    TemplateSummary,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_RULE_VERSIONS_DIR = PACKAGE_ROOT.parent / "data" / "respiratory_rules" / "versions"


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if value is None:
        return ()
    text = str(value).strip()
    return (text,) if text else ()


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_tuple_mapping(value: Any) -> dict[str, tuple[str, ...]]:
    payload = _as_mapping(value)
    return {str(key).strip(): _as_str_tuple(items) for key, items in payload.items() if str(key).strip()}


def _as_dict_tuple(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    rows: list[dict[str, Any]] = []
    for item in value:
        entry = _as_mapping(item)
        if entry:
            rows.append(entry)
    return tuple(rows)


class RespiratoryRuleLoader:
    def __init__(self, versions_root: Path | None = None):
        self.versions_root = Path(versions_root) if versions_root else DEFAULT_RULE_VERSIONS_DIR

    def available_versions(self) -> list[str]:
        if not self.versions_root.exists():
            return []
        return sorted(path.name for path in self.versions_root.iterdir() if path.is_dir())

    def load(self, version: str = "v1") -> RespiratoryRuleSet:
        version_dir = self.resolve_version_dir(version)
        warnings: list[str] = []

        taxonomy_payload = self._load_yaml(version_dir / "disease_taxonomy.yaml")
        diagnosis_payload = self._load_yaml(version_dir / "diagnosis_mapping.yaml")
        concept_payload = self._load_yaml(version_dir / "concept_dictionary.yaml")
        lab_payload = self._load_yaml(version_dir / "lab_dictionary.yaml")
        imaging_payload = self._load_yaml(version_dir / "imaging_dictionary.yaml")
        drug_payload = self._load_yaml(version_dir / "drug_dictionary.yaml")
        default_scoring_payload = self._load_yaml(version_dir / "scoring" / "default_weights.yaml")
        template_scoring_payload = self._load_yaml(version_dir / "scoring" / "template_weights.yaml")
        governance_payload = self._load_yaml(version_dir / "governance" / "version_meta.yaml")

        families = self._parse_families(taxonomy_payload)
        taxonomy_templates = self._parse_taxonomy_templates(taxonomy_payload)
        diagnosis_mappings = self._parse_diagnosis_mappings(diagnosis_payload, taxonomy_templates, families, warnings)
        concepts = self._parse_concepts(concept_payload)
        labs = self._parse_labs(lab_payload)
        imagings = self._parse_imagings(imaging_payload)
        drugs = self._parse_drugs(drug_payload)
        templates = self._parse_template_files(version_dir / "templates", taxonomy_templates, families, warnings)
        scoring = self._parse_scoring(default_scoring_payload, template_scoring_payload)
        governance = self._parse_governance(governance_payload)

        for family_id, family in families.items():
            for template_id in family.templates:
                if template_id not in taxonomy_templates:
                    warnings.append(
                        f"Family `{family_id}` references undefined taxonomy template `{template_id}`."
                    )

        for template_id in governance.bootstrap_templates:
            if template_id not in templates:
                warnings.append(f"Bootstrap template `{template_id}` has no template config file.")

        return RespiratoryRuleSet(
            version=str(governance.version or taxonomy_payload.get("version") or version_dir.name),
            root_dir=version_dir,
            families=families,
            taxonomy_templates=taxonomy_templates,
            diagnosis_mappings=diagnosis_mappings,
            concepts=concepts,
            labs=labs,
            imagings=imagings,
            drugs=drugs,
            templates=templates,
            scoring=scoring,
            governance=governance,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def resolve_version_dir(self, version: str = "v1") -> Path:
        requested = str(version or "").strip() or "v1"
        candidate = Path(requested)
        if candidate.is_absolute():
            if not candidate.exists():
                raise FileNotFoundError(f"Respiratory rule version path not found: {candidate}")
            return candidate
        version_dir = self.versions_root / requested
        if not version_dir.exists():
            raise FileNotFoundError(
                f"Respiratory rule version `{requested}` not found under `{self.versions_root}`."
            )
        return version_dir

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Required respiratory rule config is missing: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}

    def _parse_families(self, payload: dict[str, Any]) -> dict[str, FamilyDefinition]:
        families: dict[str, FamilyDefinition] = {}
        for family_id, item in _as_mapping(payload.get("families")).items():
            family_key = str(family_id).strip()
            if not family_key:
                continue
            entry = _as_mapping(item)
            families[family_key] = FamilyDefinition(
                family_id=family_key,
                label=str(entry.get("label", "")).strip(),
                description=str(entry.get("description", "")).strip(),
                templates=_as_str_tuple(entry.get("templates")),
            )
        return families

    def _parse_taxonomy_templates(self, payload: dict[str, Any]) -> dict[str, TemplateSummary]:
        templates: dict[str, TemplateSummary] = {}
        for template_id, item in _as_mapping(payload.get("templates")).items():
            template_key = str(template_id).strip()
            if not template_key:
                continue
            entry = _as_mapping(item)
            templates[template_key] = TemplateSummary(
                template_id=template_key,
                label=str(entry.get("label", "")).strip(),
                family=str(entry.get("family", "")).strip(),
            )
        return templates

    def _parse_diagnosis_mappings(
        self,
        payload: dict[str, Any],
        taxonomy_templates: dict[str, TemplateSummary],
        families: dict[str, FamilyDefinition],
        warnings: list[str],
    ) -> tuple[DiagnosisMappingEntry, ...]:
        mappings: list[DiagnosisMappingEntry] = []
        for item in payload.get("mappings") or []:
            entry = _as_mapping(item)
            mapping = DiagnosisMappingEntry(
                raw_patterns=_as_str_tuple(entry.get("raw_patterns")),
                normalized_text=str(entry.get("normalized_text", "")).strip(),
                icd10=str(entry.get("icd10", "")).strip(),
                concept_id=str(entry.get("concept_id", "")).strip(),
                family=str(entry.get("family", "")).strip(),
                template=str(entry.get("template", "")).strip(),
                confidence=float(entry.get("confidence", 0.0) or 0.0),
            )
            if not mapping.raw_patterns or not mapping.normalized_text:
                continue
            if mapping.family and mapping.family not in families:
                warnings.append(
                    f"Diagnosis mapping `{mapping.normalized_text}` references unknown family `{mapping.family}`."
                )
            if mapping.template and mapping.template not in taxonomy_templates:
                warnings.append(
                    f"Diagnosis mapping `{mapping.normalized_text}` references unknown template `{mapping.template}`."
                )
            mappings.append(mapping)
        return tuple(mappings)

    def _parse_concepts(self, payload: dict[str, Any]) -> dict[str, ConceptDefinition]:
        concepts: dict[str, ConceptDefinition] = {}
        for item in payload.get("concepts") or []:
            entry = _as_mapping(item)
            concept_id = str(entry.get("concept_id", "")).strip()
            if not concept_id:
                continue
            concepts[concept_id] = ConceptDefinition(
                concept_id=concept_id,
                category=str(entry.get("category", "")).strip(),
                normalized_text=str(entry.get("normalized_text", "")).strip(),
                aliases=_as_str_tuple(entry.get("aliases")),
                respiratory_relevance=str(entry.get("respiratory_relevance", "")).strip(),
                raw_payload=entry,
            )
        return concepts

    def _parse_labs(self, payload: dict[str, Any]) -> dict[str, LabDefinition]:
        labs: dict[str, LabDefinition] = {}
        for item in payload.get("items") or payload.get("labs") or []:
            entry = _as_mapping(item)
            item_id = str(entry.get("item_id", "")).strip()
            if not item_id:
                continue
            labs[item_id] = LabDefinition(
                item_id=item_id,
                canonical_name=str(entry.get("canonical_name", "")).strip(),
                aliases=_as_str_tuple(entry.get("aliases")),
                default_unit=str(entry.get("default_unit", "")).strip(),
                abnormal_flags=_as_mapping(entry.get("abnormal_flags")),
                disease_relevance=_as_mapping(entry.get("disease_relevance")),
                raw_payload=entry,
            )
        return labs

    def _parse_imagings(self, payload: dict[str, Any]) -> dict[str, ImagingDefinition]:
        imagings: dict[str, ImagingDefinition] = {}
        for item in payload.get("items") or payload.get("imagings") or payload.get("findings") or []:
            entry = _as_mapping(item)
            item_id = str(entry.get("item_id") or entry.get("finding_id") or "").strip()
            if not item_id:
                continue
            imagings[item_id] = ImagingDefinition(
                item_id=item_id,
                normalized_text=str(entry.get("normalized_text") or entry.get("canonical_name") or "").strip(),
                aliases=_as_str_tuple(entry.get("aliases")),
                families=_as_str_tuple(entry.get("families")),
                raw_payload=entry,
            )
        return imagings

    def _parse_drugs(self, payload: dict[str, Any]) -> dict[str, DrugDefinition]:
        drugs: dict[str, DrugDefinition] = {}
        for item in payload.get("items") or payload.get("drugs") or []:
            entry = _as_mapping(item)
            drug_id = str(entry.get("drug_id", "")).strip()
            if not drug_id:
                continue
            drugs[drug_id] = DrugDefinition(
                drug_id=drug_id,
                canonical_name=str(entry.get("canonical_name", "")).strip(),
                aliases=_as_str_tuple(entry.get("aliases")),
                drug_class=str(entry.get("drug_class", "")).strip(),
                mechanism_class=str(entry.get("mechanism_class", "")).strip(),
                treatment_intents=_as_str_tuple(
                    entry.get("treatment_intents") or entry.get("treatment_intent")
                ),
                respiratory_relevance=str(entry.get("respiratory_relevance", "")).strip(),
                raw_payload=entry,
            )
        return drugs

    def _parse_template_files(
        self,
        template_dir: Path,
        taxonomy_templates: dict[str, TemplateSummary],
        families: dict[str, FamilyDefinition],
        warnings: list[str],
    ) -> dict[str, TemplateConfig]:
        templates: dict[str, TemplateConfig] = {}
        if not template_dir.exists():
            return templates

        for path in sorted(template_dir.glob("*.yaml")):
            entry = self._load_yaml(path)
            template_id = str(entry.get("template", "")).strip() or path.stem
            family = str(entry.get("family", "")).strip()
            template_label = str(entry.get("label", "")).strip()
            taxonomy_summary = taxonomy_templates.get(template_id)

            if taxonomy_summary and family and taxonomy_summary.family and family != taxonomy_summary.family:
                warnings.append(
                    f"Template file `{template_id}` family `{family}` mismatches taxonomy family `{taxonomy_summary.family}`."
                )
            if family and family not in families:
                warnings.append(f"Template file `{template_id}` references unknown family `{family}`.")

            rules: list[TemplateRule] = []
            for raw_rule in entry.get("rules") or []:
                rule_payload = _as_mapping(raw_rule)
                rule_id = str(rule_payload.get("rule_id", "")).strip()
                if not rule_id:
                    continue
                rules.append(
                    TemplateRule(
                        rule_id=rule_id,
                        when=_as_mapping(rule_payload.get("when")),
                        then=_as_mapping(rule_payload.get("then")),
                        rationale=str(rule_payload.get("rationale", "")).strip(),
                        evidence_level=str(rule_payload.get("evidence_level", "")).strip(),
                        recommendation_strength=str(
                            rule_payload.get("recommendation_strength", "")
                        ).strip(),
                        citations=_as_dict_tuple(rule_payload.get("citations")),
                        standard_mappings=_as_dict_tuple(rule_payload.get("standard_mappings")),
                        tags=_as_str_tuple(rule_payload.get("tags")),
                    )
                )

            templates[template_id] = TemplateConfig(
                template_id=template_id,
                label=template_label or (taxonomy_summary.label if taxonomy_summary else ""),
                family=family or (taxonomy_summary.family if taxonomy_summary else ""),
                diagnosis_concepts=_as_str_tuple(entry.get("diagnosis_concepts")),
                key_symptoms=_as_str_tuple(entry.get("key_symptoms")),
                key_evidence=_as_tuple_mapping(entry.get("key_evidence")),
                key_treatments=_as_tuple_mapping(entry.get("key_treatments")),
                severity_markers=_as_tuple_mapping(entry.get("severity_markers")),
                denoise=_as_mapping(entry.get("denoise")),
                rules=tuple(rules),
                citations=_as_dict_tuple(entry.get("citations")),
                standard_mappings=_as_dict_tuple(entry.get("standard_mappings")),
                guideline_notes=_as_str_tuple(entry.get("guideline_notes")),
                raw_payload=entry,
            )

        for template_id in taxonomy_templates:
            if template_id not in templates:
                warnings.append(f"Taxonomy template `{template_id}` has no config file under `{template_dir}`.")
        return templates

    def _parse_scoring(
        self,
        default_payload: dict[str, Any],
        template_payload: dict[str, Any],
    ) -> ScoringConfig:
        return ScoringConfig(
            version=str(default_payload.get("version") or template_payload.get("version") or "").strip(),
            final_score_weights={
                key: float(value)
                for key, value in _as_mapping(default_payload.get("final_score_weights")).items()
            },
            semantic_chain_weights={
                key: float(value)
                for key, value in _as_mapping(default_payload.get("semantic_chain_weights")).items()
            },
            structured_slot_weights={
                key: float(value)
                for key, value in _as_mapping(default_payload.get("structured_slot_weights")).items()
            },
            rule_caps={
                key: float(value) for key, value in _as_mapping(default_payload.get("rule_caps")).items()
            },
            template_overrides=_as_mapping(template_payload.get("templates")),
        )

    def _parse_governance(self, payload: dict[str, Any]) -> GovernanceMeta:
        return GovernanceMeta(
            version=str(payload.get("version", "")).strip(),
            created_at=str(payload.get("created_at", "")).strip(),
            status=str(payload.get("status", "")).strip(),
            owners=_as_str_tuple(payload.get("owners")),
            scope=str(payload.get("scope", "")).strip(),
            sources=_as_mapping(payload.get("sources")),
            bootstrap_templates=_as_str_tuple(payload.get("bootstrap_templates")),
            change_policy=_as_mapping(payload.get("change_policy")),
        )


@lru_cache(maxsize=4)
def get_default_rule_set(version: str = "v1", versions_root: str | None = None) -> RespiratoryRuleSet:
    loader = RespiratoryRuleLoader(Path(versions_root) if versions_root else None)
    return loader.load(version=version)
