# 4D Temporal Hypergraph — Paper Code Export

> **Version:** v5 — four-dimensional **temporal hypergraph representation** (JBI submission)
> **What changed from v4:** the paper's contribution is now framed as the *representation* (a four-dimensional
> temporal hypergraph) with a swappable read-out head. This release adds (i) a **hyperedge-family ablation**
> that separates the contribution of the hypergraph structure itself from a plain entity bag, (ii) a
> **quantitative cross-dimension coupling** re-construction, (iii) a **read-out comparison**
> (MLP / DeepSets / Set Transformer / HGNN, same features), and (iv) a **component-blind** dataset builder.
> The headline protocol was expanded from 3 seeds to **10 seeds**. Task 3 (severity regression) is reported in
> the **component-blind** setting — every directly measured quantity entering the CSCS definition is withheld
> from the input — which removes the definitional overlap with the target.

This folder is a curated GitHub-ready export of the code used for the manuscript.

## What is included

- `src/paper_experiments/`
  Core experiment scripts:
  - internal Task1/Task2 experiments and the 4D chain pipelines (`exp_4dchain.py`, `exp_4dchain_v2.py`)
  - base baselines (`exp_run.py`, `exp_lrraw.py`, `exp_lrraw_stats.py`), ablation (`exp_abl.py`),
    feature extraction (`exp_extract.py`, `exp_dataset.py`, `exp_coverage.py`)
  - **hyperedge-family ablation** (`exp_hyperedge_family_ablation.py`) — structure-vs-entity-bag ablation and
    model-free (Kruskal-Wallis) structural tests → `_hyperedge_ablation.json`
  - **quantitative cross-dimension coupling** (`exp_cross_quant_coupling.py`) — 7 coupling statistics per
    dimension pair, paired tests against the binary construction → `_cross_quant*.json`
  - **read-out comparison** (`exp_hgnn_vs_set_v2.py`) — S-only MLP / DeepSets / Set Transformer / HGNN on the
    same features and folds → `_hgnn_main.json`
  - **component-blind dataset** (`build_component_blind_ds.py`) — removes severity columns and `score`/`sev`
    entities for the leakage ablation
  - **Task 3 implementation** (`exp_task_fix_probe.py`) — the component-blind CSCS severity regression, plus the
    component-inclusive (definition-leaking) control arm; its component-blind arm is the pre-audit (205-dim) value,
    and the corrected (177-dim) result is produced by `exp_fig3_truecb_10seed.py`
  - external cohort preparation (manifest building, re-extraction prep, format conversion) and sensitivity subsets
  - HyperGraphRAG retrieval evaluation (`run_hgr_eval.py`, `step2_validate.py`)
  - paper supplementary analyses (`_paper_supplements.py`)
  - manuscript figure generation (`generate_figure7.py` — Figure 7)
  - reproducibility check (`exp_repro_check.py`, schema-aware)

- `src/reextraction_support/` — support code for rebuilding 4D similarity chains from raw external records.

- `configs/` — rule/config files used by the re-extraction pipeline.

- `docs/`
  - `四维抽取提示词.md` — full text of the four dimension-specific LLM extraction prompts (Section 2.3 / Text S1)
  - `四维链信息提取规则清单.md` — method notes for the 4D chain extraction rule set
  - `paper_supplements_report.md` — consolidated report of the supplementary analyses with the numbers to cite

- `results/paper_supplements/` — figures and machine-readable outputs from `_paper_supplements.py`.

- `results/final/` — machine-readable outputs backing the final manuscript tables and the new Figure 8:
  - `exp_results_task12_10seed_summary_ci.json` — 10-seed Task1/Task2 summary with bootstrap CIs
  - `exp_results_p1_headroom.json` — decision-rule / nested-CV head-room analysis
  - `_hyperedge_ablation.json` — hyperedge-family ablation + model-free structural tests (Table 5, Figure 8)
  - `_cross_quant.json`, `_cross_quant_paired.json` — quantitative cross-dimension coupling
  - `_hgnn_main.json` — read-out comparison (Table 6)
  - `fig8_hyperedge_family_gains.png` — hyperedge-family gains (Figure 8)

- `third_party/HyperGraphRAG/` — minimal vendored dependency files required by `run_hgr_eval.py`.

## What is intentionally excluded

- raw patient data, de-identified case JSON collections, patient-level CSV manifests
- `.pkl` cached datasets and prediction dumps
- experiment logs, generated figures/intermediate outputs (except `results/paper_supplements/` and `results/final/`)
- generated custom-KG JSON assets used by retrieval experiments
- manuscript-only working files

## Paths and environment variables

No absolute paths are hard-coded. Every entry point resolves its inputs from environment variables
(falling back to the current working directory):

| Variable | Meaning | Used by |
|---|---|---|
| `EXPERIMENT_BASE` | experiment workspace (datasets, results) | all scripts |
| `HYPERGRAPHRAG_PATH` | vendored HyperGraphRAG package (defaults to `<repo>/third_party/HyperGraphRAG`) | `exp_4dchain*.py`, `_paper_supplements.py`, `run_hgr_eval.py` |
| `DS4_PATH` | path to `exp_4dchain_v2_dataset.pkl` | ablation / coupling / read-out / repro scripts |
| `HG_OUT_DIR` | output directory for the new probe JSONs | `exp_hyperedge_family_ablation.py`, `exp_cross_quant_coupling.py` |
| `EVENT_OUT`, `HG_OUT` | output directory / filename for the read-out comparison | `exp_hgnn_vs_set_v2.py` |
| `FIG_OUT` | output path for the manuscript figure (defaults to `$EXPERIMENT_BASE/Figure7_4D_Hypergraph_Structure.png`) | `generate_figure7.py` |
| `TRAIN_ROOT` | 4D-chain training-set root (private data) | `run_hgr_eval.py`, `build_core30_reextract_prep.py` |
| `RAW_DATA_ROOT` | raw external CSV root (private data) | `build_external_*`, `build_core30_reextract_prep.py` |
| `TESTSET_ROOT` | external test-set root (private data) | `third_party/HyperGraphRAG/dashboard_case_app.py` |
| `EXTERNAL_CHAIN_DIR` | external 4D-chain folder for external validation | `exp_4dchain_v2.py --task12-external` |

Example:

```bash
export EXPERIMENT_BASE="/path/to/your/experiment/directory"
export DS4_PATH="$EXPERIMENT_BASE/exp_4dchain_v2_dataset.pkl"
```

## Note on omitted retrieval assets

The original workflow referenced large generated COPD custom-KG JSON files under
`HyperGraphRAG/data/copd_custom_kg/`. Those are generated runtime artifacts, not source code, and are not copied.

## Reproducibility and number provenance

All numbers reported in the manuscript are produced by the scripts in this repository.

- **Seeds.** The headline protocol uses **10 seeds** `42, 123, 456, 7, 13, 99, 202, 404, 777, 1234` with
  `StratifiedKFold(5, shuffle=True, random_state=seed)` (50 fold evaluations). Bootstrap CIs use 2,000 resamples.
  The read-out comparison (`exp_hgnn_vs_set_v2.py`) uses the earlier 3-seed subset (`42, 123, 456`).
- **XGBoost read-out.** `n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9`,
  `n_jobs=4`. Note: XGBoost multi-threading introduces a small non-determinism (≈ ±0.003 AUC across repeated
  runs of the same configuration); conclusions are unaffected. Set `n_jobs=1` for bit-level reproducibility.
- **Inputs required.** The pipeline consumes (i) the de-identified case JSON collection and (ii) the
  four-dimensional (History / Evidence / Treatment / Severity) LLM-extracted chain JSONs.
  `exp_4dchain_v2.py --build-only` builds `exp_4dchain_v2_dataset.pkl` from these; they are excluded here for
  privacy reasons, so the dataset must be supplied by the user.
- **Script → result mapping**
  - `exp_4dchain_v2.py` — main Task1/Task2 results, 4D-XGB, dimension ablation, external validation
  - `exp_run.py` — base baselines (KG+GNN, TKG, LSTM, LR/XGBoost, 4D-HyperGraph) and temporal hold-out
  - `exp_lrraw.py` / `exp_lrraw_stats.py` — LR-raw baseline metrics and paired statistics
  - `exp_abl.py` — end-to-end dimension ablation
  - `exp_hyperedge_family_ablation.py` — hyperedge-family ablation + model-free structural tests
    (Table 5, Figure 8; mirrors `results/final/_hyperedge_ablation.json`)
  - `exp_cross_quant_coupling.py` — quantitative cross-dimension coupling, paired vs the binary construction
  - `exp_hgnn_vs_set_v2.py` — read-out comparison at fixed features (Table 6)
  - `build_component_blind_ds.py` — component-blind dataset for the severity-leakage ablation
  - `exp_task_fix_probe.py` — Task 3 component-blind CSCS regression (pre-audit arm) and the Task 2 support
    numbers; the corrected (177-dim) result is produced by `exp_fig3_truecb_10seed.py`
  - `build_external_*`, `merge_core30_4d_outputs.py`, `convert_core30_to_external_chain_format.py` — external
    cohort manifests (Core30 / Balanced87 / Strict72) and format conversion
  - `_paper_supplements.py` — supplementary analyses (writes `results/paper_supplements/`)
  - `generate_figure7.py` — regenerates manuscript Figure 7 from the statistics in its constants block
    (patient count, per-dimension entity counts, hyperedge count, coverage)
  - `exp_repro_check.py` — dataset integrity check (patient set, labels, folds, hypergraph statistics)

## Key results (10-seed protocol)

| Task | Method | Metric | Mean ± SD | 95% CI |
|---|---|---|---|---|
| Task 1 (3-way) | 4D-XGB | macro-F1 | 0.486 ± 0.032 | [0.464, 0.505] |
| Task 1 (3-way) | LR/XGBoost | macro-F1 | 0.433 ± 0.029 | [0.411, 0.451] |
| Task 1 (3-way) | LR-raw | macro-F1 | 0.441 ± 0.021 | [0.419, 0.460] |
| Task 2 (A/B/C>0) | 4D-XGB | AUC-ROC | 0.708 ± 0.021 | [0.683, 0.733] |
| Task 2 (A/B/C>0) | LR/XGBoost | AUC-ROC | 0.698 ± 0.018 | [0.672, 0.723] |
| Task 2 (A/B/C>0) | LR-raw | AUC-ROC | 0.696 ± 0.018 | [0.671, 0.722] |

Paired bootstrap differences (4D-XGB − baseline): Task 1 macro-F1 vs LR/XGBoost `[+0.035, +0.073]`,
vs LR-raw `[+0.022, +0.068]` (both exclude 0). Task 2 AUC vs LR/XGBoost `[-0.004, +0.023]`,
vs LR-raw `[-0.009, +0.031]` (both include 0 → statistical tie).

**Task 3 — component-blind CSCS severity regression** (4D representation, XGBoost readout; 5-fold).
The corrected component-blind setting withholds the severity structural columns, the `score`/`sev` entities, **and
the 28 severity-derived graph statistics** (see `_task3_sev_graph_leak.json`). Under this corrected setting the 4D
representation does **not** recover CSCS: per-seed mean R² = −0.013, r = 0.272, MAE = 0.152, RMSE = 0.186
(`_task3_true_cb_flat_compare.json`); the Figure 3 pooled scatter gives R² = 0.057, r = 0.287, MAE = 0.148,
RMSE = 0.182 (`fig3_scatter_points.csv`, 10-seed; `fig3_predicted_actual_trueCB.csv`, 3-seed). The apparent
R² = 0.383 reported earlier is fully explained by the 28 severity-derived graph statistics that indirectly encode
the withheld severity entities; removing them jointly drives R² to ≈ 0. Produced by `exp_fig3_truecb_10seed.py`
(corrected 10-seed OOF) and `exp_task_fix_probe.py` (component-inclusive control arm).

**Hyperedge-family ablation** (same builder / same folds / fixed XGBoost read-out, 10 seeds; macro-F1):

| Arm | Features | macro-F1 | AUC-ROC |
|---|---|---|---|
| S (structured only) | 29 | 0.4219 | 0.6291 |
| S + CARD (hyperedge cardinality/count) | 37 | 0.4444 | 0.6446 |
| S + TEMP (temporal hyperedges) | 61 | 0.4751 | 0.6524 |
| S + CROSS (binary cross-dimension) | 65 | 0.4354 | 0.6337 |
| S + GRAPH (all structure) | 105 | 0.4696 | 0.6517 |
| S + ENT (entity bag, no structure) | 152 | 0.4480 | 0.6448 |
| Full 4D hypergraph (S+GRAPH+ENT) | 228 | 0.4897 | 0.6654 |

Paired difference for the full model: − S `[+0.049, +0.087]`, − S+ENT `[+0.024, +0.057]` (structure adds
independent signal beyond an entity bag), − S+TEMP `[-0.001, +0.029]` (temporal hyperedges carry most of it).

**Quantitative cross-dimension coupling** (71 features) reaches macro-F1 0.4701 / AUC 0.6610 versus the binary
construction (65 features) 0.4354 / 0.6337 — paired differences macro-F1 `[+0.020, +0.054]`, AUC `[+0.015, +0.037]`.

**Read-out comparison** (identical features/folds, 3 seeds): S-only MLP 0.658, DeepSets 0.646,
Set Transformer 0.652, HGNN 0.660 AUC — all pairwise ΔAUC CIs include 0, i.e. the read-out head is swappable.

**Model-free structural differences** across the clinical subgroups (Kruskal-Wallis):
mean hyperedge cardinality p = 0.0024, hyperedge count p = 0.0052, mean temporal span p = 0.0010,
cross-dimension co-activation rate p = 0.0126 (max cardinality p = 0.264).

**Hypergraph statistics** (current pipeline, 253 patients): extraction-stage entity union **182**
(history 12, evidence 144, treatment 9, severity 17) → modelling vocabulary **123** entity nodes
(history 12, evidence 84, treatment 9, severity 18; entities documented in fewer than two patients removed),
1,378 per-day per-dimension hyperedges (mean 5.45 per patient), 2.94 mean active days within the 5-day window,
58.9% window coverage.

## Reproducibility verification

```bash
python src/paper_experiments/exp_repro_check.py --dataset /path/to/exp_4dchain_v2_dataset.pkl
```

This prints the patient set, label distribution, fold splits, entity vocabulary, hyperedge count and temporal
coverage, alongside the manuscript values for comparison.

## Suggested repository structure

```bash
cd "<this folder>"
git init
git add .
```

## Dependency notes

- Root-level `requirements.txt` lists the main Python dependencies.
- `third_party/HyperGraphRAG/requirements.txt` preserves the vendored component's dependency list.

## Quick navigation

See `FILE_INDEX.md` for a short description of the main scripts and how they map to the paper workflow.
