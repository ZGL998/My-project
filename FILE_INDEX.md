# File Index

## Core experiment scripts

- `src/paper_experiments/exp_extract.py`  
  Extracts structured, sequential, and entity-level features from the internal cohort.

- `src/paper_experiments/exp_dataset.py`  
  Builds the base experiment dataset object from extracted features.

- `src/paper_experiments/exp_run.py`  
  Runs the main baseline experiments for Task1/Task2.

- `src/paper_experiments/exp_4dchain.py`  
  Runs the original 4D chain experiment pipeline.

- `src/paper_experiments/exp_4dchain_v2.py`  
  Runs the revised 4D chain experiment pipeline with severity enhancement and external validation utilities.

- `src/paper_experiments/exp_abl.py`  
  Runs the ablation study on the end-to-end 4D model.

- `src/paper_experiments/exp_lrraw.py`  
  Runs the rule-based structured baseline.

- `src/paper_experiments/exp_lrraw_stats.py`  
  Computes paired statistics for the rule-based structured baseline.

- `src/paper_experiments/exp_coverage.py`  
  Summarizes extraction coverage and field presence.

- `src/paper_experiments/_paper_supplements.py`  
  Reproduces the five paper supplementary analyses (same-readout argument, SHAP feature attribution, external confusion matrices, feature-level incremental analysis, Task 2 decision-threshold analysis; all using the final post-rollback number convention). Figures and data are written to `results/paper_supplements/`.

- `src/paper_experiments/exp_hyperedge_family_ablation.py`  
  Hyperedge-family ablation (CARD / TEMP / CROSS / full structure / entity bag), same builder, same folds and a
  fixed XGBoost read-out over 10 seeds, plus model-free Kruskal-Wallis tests of the structural statistics across
  the three clinical subgroups. Writes `_hyperedge_ablation.json` (Table 5, Figure 8).

- `src/paper_experiments/exp_cross_quant_coupling.py`  
  Replaces the binary cross-dimension coupling with seven quantitative coupling statistics per dimension pair and
  runs paired tests against the binary construction. Writes `_cross_quant.json` and `_cross_quant_paired.json`.

- `src/paper_experiments/exp_hgnn_vs_set_v2.py`  
  Read-out comparison at fixed features and folds: S-only MLP, DeepSets, Set Transformer and the 4D hypergraph
  network (fold-internal ensembling, class-weighted loss). Writes `_hgnn_main.json` (Table 6).

- `src/paper_experiments/build_component_blind_ds.py`  
  Builds the component-blind dataset used for the severity-leakage ablation: drops the severity structural
  columns and all `('score', *)` / `('sev', *)` entities, and zeroes the SpO₂ sequence column.

- `src/paper_experiments/exp_task_fix_probe.py`  
  Implements Task 3 — the component-blind CSCS severity regression — together with the component-inclusive
  control arm and the Task 2 support numbers. Note: its component-blind arm is the pre-audit (205-dim) value;
  the corrected (177-dim) result is produced by `exp_fig3_truecb_10seed.py`.

- `src/paper_experiments/exp_fig3_truecb_10seed.py`  
  Builds the corrected component-blind feature matrix (177 dims = 24 structured + 48 non-severity graph stats +
  105 non-severity entities) and runs the 10-seed × 5-fold XGBoost OOF regression for the CSCS target, pooling
  per-patient predictions for Figure 3. Writes `fig3_truecb_oof_10seed.npz` and `fig3_scatter_points.csv`.

- `src/paper_experiments/export_fig3_scatter.py`  
  Pools per-seed OOF predictions into 253 scatter points, computes the pooled R² / Pearson r / MAE / RMSE
  self-check, fits the linear trend (slope / intercept), and writes `fig3_scatter_points.csv`.

- `src/paper_experiments/_fill_withholding_mcnemar.py`  
  Exports the component-blind withholding list (`withholding_list.json`: 5 withheld columns / 18 severity entities /
  28 severity-derived graph statistics) and the per-seed three-class McNemar test (`mcnemar_per_seed.json`).

- `src/paper_experiments/exp_repro_check.py`  
  Schema-aware dataset integrity check: patient set, label distribution, fold splits, entity vocabulary,
  hyperedge count and temporal coverage, printed next to the manuscript values.

- `src/paper_experiments/dump_table1_and_entities.py`  
  Prints Table 1 cohort characteristics (age / sex / smoking / comorbidity / labs / vitals / medications /
  severity) by clinical subgroup, plus the entity distribution across types and dimensions.

- `src/paper_experiments/generate_figure7.py`  
  Regenerates manuscript Figure 7 (four-dimensional temporal hypergraph structure) from the statistics in its
  constants block: patient count, per-dimension entity counts, hyperedge count, mean active days and window
  coverage. Output path is controlled by `FIG_OUT`.

## Retrieval evaluation

- `src/paper_experiments/run_hgr_eval.py`  
  Runs HyperGraphRAG retrieval-style evaluation on the internal three-group cohort.

- `src/paper_experiments/step2_validate.py`  
  Validation helper script for the HyperGraphRAG evaluation output.

## External cohort preparation

- `src/paper_experiments/build_external_raw_copd_manifest.py`  
  Builds the external Beijing cohort manifest and derives balanced subsets.

- `src/paper_experiments/build_core30_reextract_prep.py`  
  Assembles raw-text case packages and prepares 4D re-extraction directories for external subsets.

- `src/paper_experiments/merge_core30_4d_outputs.py`  
  Merges per-group 4D outputs into one cohort-level package.

- `src/paper_experiments/convert_core30_to_external_chain_format.py`  
  Converts merged 4D outputs into the format required by `exp_4dchain_v2.py --task12-external`.

- `src/paper_experiments/build_external_strict72_subset.py`  
  Builds the stricter external sensitivity subset with stronger infection-label support.

## Re-extraction support

- `src/reextraction_support/build_similarity_chains.py`  
  Original script used to rebuild 4D similarity-chain outputs from case JSON files.

- `configs/四维相似链处理规则.json`  
  Rule configuration file used by the re-extraction support script.

- `docs/四维链信息提取规则清单.md`  
  Human-readable notes for the 4D chain extraction rule set.

- `docs/四维抽取提示词.md`  
  Full text of the four dimension-specific LLM extraction prompts used to build the 4D chains (corresponds to Section 2.3 and Multimedia Appendix Text S1 of the manuscript).

- `docs/paper_supplements_report.md`  
  Consolidated report of the five supplementary analyses (same-readout, SHAP, external confusion matrices, feature-level incremental, decision threshold), with honest caveats and the exact numbers to cite in the manuscript (final post-rollback convention).

- `docs/可复现性数据补全回传.md`  
  Reproducibility data completion report: the component-blind withholding list (5 columns / 18 entities /
  28 severity-derived graph statistics) and the per-seed three-class McNemar test for §4.4 / §4.7.

- `docs/论文数据溯源对照表.md`  
  Paper-to-repository provenance table: maps every number in the manuscript data list to its producing script
  and backing result file, with privacy-exclusion notes.

## Results (paper supplements)

- `results/paper_supplements/fig1_same_readout.png`  
  Same-readout comparison (Figure 7 of the manuscript).

- `results/paper_supplements/fig2_shap_by_group.png`  
  SHAP feature attribution by feature group (Figure 8 of the manuscript).

- `results/paper_supplements/fig5_feature_incremental.png`  
  Feature-level incremental analysis (Figure 9 of the manuscript).

- `results/paper_supplements/fig6_threshold.png`  
  Task 2 decision-threshold analysis (Figure 10 of the manuscript).

- `results/paper_supplements/shap_by_group.json`, `feature_incremental.json`, `temporal_3seed.json`  
  Machine-readable outputs of the corresponding analyses.

## Results (final manuscript)

- `results/final/exp_results_task12_10seed_summary_ci.json`  
  10-seed Task 1 (macro-F1) and Task 2 (AUC-ROC) summary with bootstrap 95% CIs and paired differences.

- `results/final/exp_results_task12_3seed_summary_ci.json`  
  3-seed Task 1 / Task 2 summary (bootstrap CIs), the earlier protocol used for the seed-42 reference tables.

- `results/final/exp_results_4dchain_v2_seed42.json`, `results/final/_audit_earlystop_fix_seed42.json`  
  Seed-42 reference numbers for Table 2 (Task 1) and Table 3 (Task 2): XGB read-out rows and the early-stopping
  corrected neural rows (KG+GNN / TKG / LSTM / 4D-HyperGraph).

- `results/final/_audit_earlystop_fix_seed42_abl.json`  
  Dimension-deletion ablation (Table D.1): macro-F1 / AUC-ROC / Pearson r for the full model and the four
  leave-one-dimension-out variants.

- `results/final/exp_results_task12_temporal.json`  
  Temporal hold-out (train 162 / test 91, split 2024-10-22) Task 1 / Task 2 metrics.

- `results/final/exp_results_task12_external_*.json`  
  External-validation results for the four queues (Balanced-30 / Balanced-87 / Strict-72 / Prevalence-65),
  backing Figure 4.

- `results/final/_strict_external.json`, `results/final/_strict_internal.json`  
  Strict-72 sensitivity analyses (internal and external), including the 300-candidate retrieval pool.

- `results/final/exp_results_p1_headroom.json`  
  Head-room analysis: fixed decision-rule tuning and nested-CV hyper-parameter search (both CIs include 0).

- `results/final/_hyperedge_ablation.json`  
  Hyperedge-family ablation table, paired bootstrap CIs, and model-free structural statistics (Table 5, Figure 8).

- `results/final/_cross_quant.json`, `results/final/_cross_quant_paired.json`  
  Quantitative cross-dimension coupling arms/differences, and the paired quantitative-vs-binary test.

- `results/final/_hgnn_main.json`  
  Read-out comparison metrics (Table 6).

- `results/final/_task3_sev_graph_leak.json`  
  Leakage audit for Task 3: the component-blind arm (205 dims) vs the arm with the 28 severity-derived graph
  statistics removed (177 dims), listing the exact 28 statistics that drive R² from 0.383 to ≈ 0.

- `results/final/_task3_true_cb_flat_compare.json`  
  Corrected component-blind Task 3 comparison (4D vs flat-entity vs structured-only) backing Table A.2's blind
  rows: 4D-XGB R² = −0.013 / r = 0.272, flat-entity R² = −0.018, S-only R² = −0.086.

- `results/final/_task3_full_structured_baseline.json`  
  Extended structured baseline for Task 3 (34 features): R² = −0.022 / r = 0.283.

- `results/final/_task3_comorb_fix_compare.json`  
  Comorbidity-fix comparison for the CSCS target (pre/post the comorbidity-count bug fix).

- `results/final/_cb_withhold_list.json`  
  Component-blind withholding-list check (23 withheld features = 5 severity columns + 18 severity entities), with
  a 205-dimension leak scan over the CSCS-defining quantities.

- `results/final/withholding_list.json`  
  Component-blind withholding list for §4.4: the 5 withheld severity structured columns, the 18 withheld
  severity entities, and the 28 severity-derived graph statistics (from the leakage audit).

- `results/final/mcnemar_per_seed.json`  
  Per-seed three-class McNemar test (4D-XGB vs LR/XGBoost and LR-raw): directional discordant counts b/c and
  exact binomial p-values for the 10 seeds.

- `results/final/fig3_scatter_points.csv`, `results/final/fig3_predicted_actual_trueCB.csv`  
  Figure 3 scatter data (253 patients, `actual_cscs / predicted_cscs`): the former is the corrected
  component-blind 10-seed pooled prediction, the latter the 3-seed pooled prediction.

- `results/final/fig3_truecb_oof_10seed.npz`  
  Per-seed out-of-fold CSCS predictions for the corrected component-blind 10-seed run (`seeds` / `actual` /
  `oof` array), used to reproduce the Figure 3 scatter points.

- `results/final/fig8_hyperedge_family_gains.png`  
  Per-family gains over the structured-only baseline (Figure 8).

## Third-party dependency snapshot

- `third_party/HyperGraphRAG/dashboard_case_app.py`  
  Entry point used by `run_hgr_eval.py` to access retrieval and case-ranking logic.

- `third_party/HyperGraphRAG/hypergraphrag/`  
  Core HyperGraphRAG Python package.

- `third_party/HyperGraphRAG/resp_rules/`  
  Respiratory rule mapping and loading utilities.

- `third_party/HyperGraphRAG/templates/`  
  HTML templates referenced by `dashboard_case_app.py`.

- `third_party/HyperGraphRAG/data/copd_custom_kg/`  
  COPD custom KG configuration JSON files referenced by the retrieval module.
