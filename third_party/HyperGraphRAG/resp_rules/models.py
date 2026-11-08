from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FamilyDefinition:
    family_id: str
    label: str
    description: str
    templates: tuple[str, ...] = ()


@dataclass(frozen=True)
class TemplateSummary:
    template_id: str
    label: str
    family: str


@dataclass(frozen=True)
class DiagnosisMappingEntry:
    raw_patterns: tuple[str, ...]
    normalized_text: str
    icd10: str
    concept_id: str
    family: str
    template: str
    confidence: float = 0.0


@dataclass(frozen=True)
class ConceptDefinition:
    concept_id: str
    category: str
    normalized_text: str
    aliases: tuple[str, ...] = ()
    respiratory_relevance: str = ""
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LabDefinition:
    item_id: str
    canonical_name: str
    aliases: tuple[str, ...] = ()
    default_unit: str = ""
    abnormal_flags: dict[str, Any] = field(default_factory=dict)
    disease_relevance: dict[str, Any] = field(default_factory=dict)
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImagingDefinition:
    item_id: str
    normalized_text: str
    aliases: tuple[str, ...] = ()
    families: tuple[str, ...] = ()
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DrugDefinition:
    drug_id: str
    canonical_name: str
    aliases: tuple[str, ...] = ()
    drug_class: str = ""
    mechanism_class: str = ""
    treatment_intents: tuple[str, ...] = ()
    respiratory_relevance: str = ""
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TemplateRule:
    rule_id: str
    when: dict[str, Any] = field(default_factory=dict)
    then: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    evidence_level: str = ""
    recommendation_strength: str = ""
    citations: tuple[dict[str, Any], ...] = ()
    standard_mappings: tuple[dict[str, Any], ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class TemplateConfig:
    template_id: str
    label: str
    family: str
    diagnosis_concepts: tuple[str, ...] = ()
    key_symptoms: tuple[str, ...] = ()
    key_evidence: dict[str, tuple[str, ...]] = field(default_factory=dict)
    key_treatments: dict[str, tuple[str, ...]] = field(default_factory=dict)
    severity_markers: dict[str, tuple[str, ...]] = field(default_factory=dict)
    denoise: dict[str, Any] = field(default_factory=dict)
    rules: tuple[TemplateRule, ...] = ()
    citations: tuple[dict[str, Any], ...] = ()
    standard_mappings: tuple[dict[str, Any], ...] = ()
    guideline_notes: tuple[str, ...] = ()
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoringConfig:
    version: str
    final_score_weights: dict[str, float] = field(default_factory=dict)
    semantic_chain_weights: dict[str, float] = field(default_factory=dict)
    structured_slot_weights: dict[str, float] = field(default_factory=dict)
    rule_caps: dict[str, float] = field(default_factory=dict)
    template_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class GovernanceMeta:
    version: str
    created_at: str = ""
    status: str = ""
    owners: tuple[str, ...] = ()
    scope: str = ""
    sources: dict[str, Any] = field(default_factory=dict)
    bootstrap_templates: tuple[str, ...] = ()
    change_policy: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RespiratoryRuleSet:
    version: str
    root_dir: Path
    families: dict[str, FamilyDefinition]
    taxonomy_templates: dict[str, TemplateSummary]
    diagnosis_mappings: tuple[DiagnosisMappingEntry, ...]
    concepts: dict[str, ConceptDefinition]
    labs: dict[str, LabDefinition]
    imagings: dict[str, ImagingDefinition]
    drugs: dict[str, DrugDefinition]
    templates: dict[str, TemplateConfig]
    scoring: ScoringConfig
    governance: GovernanceMeta
    warnings: tuple[str, ...] = ()

    def get_template(self, template_id: str) -> TemplateConfig | None:
        return self.templates.get(str(template_id).strip())

    def get_family(self, family_id: str) -> FamilyDefinition | None:
        return self.families.get(str(family_id).strip())

    def has_template(self, template_id: str) -> bool:
        return str(template_id).strip() in self.templates
