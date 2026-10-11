# Held-out Identifiability Final Audit

**FINAL STATUS: PASS**

- Queries: 11,505
- Candidates: 4,308
- Pairs per semantic condition: 49,563,540
- Candidate universe: `df64a26bff45406e2fd4cfbcfb5b3e54691dfb6acac9bce7ed231014e13d9eb1`
- Text2Pos exact-source locks: {1: 1372, 5: 3676, 10: 4901}
- Text2Loc exact-source locks: {1: 3428, 5: 6917, 10: 8206}
- Bootstrap: 10000 scene-stratified, source-cell-clustered replicates
- Seed: 20260901
- Unsupported queries: 0
- Fine semantic localization: not reported; continuous valid-region support unavailable

## Gates
- PASS — query_count_11505
- PASS — candidate_count_4308
- PASS — query_ids_unique
- PASS — model_rows_23010
- PASS — pairs_color
- PASS — pairs_no_color
- PASS — no_unsupported_queries
- PASS — restart_test
- PASS — text2pos_six_count_lock
- PASS — text2loc_six_count_lock
- PASS — valid_primary_shape
- PASS — valid_no_color_shape
- PASS — query_model_unique
- PASS — finite_csv::FIG_metric_disagreement.csv
- PASS — finite_csv::FIG_source_vs_semantic_by_V.csv
- PASS — finite_csv::FIG_valid_set_distribution.csv
- PASS — finite_csv::TABLE_ambiguity_strata.csv
- PASS — finite_csv::TABLE_bootstrap_intervals.csv
- PASS — finite_csv::TABLE_identifiability_bound.csv
- PASS — finite_csv::TABLE_metric_disagreement.csv
- PASS — finite_csv::TABLE_overall_metrics.csv
- PASS — finite_csv::TABLE_scene_robustness.csv
- PASS — finite_csv::TABLE_valid_set_distribution.csv
