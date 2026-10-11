# Data provenance and external resources

## Dataset and evaluation population

The experiments use KITTI360Pose, based on the KITTI-360 dataset. Source data and dataset-specific permissions remain with the dataset providers. No raw point clouds or original model checkpoints are redistributed in this repository.

The held-out test population includes 11,505 descriptions from scenes 0003, 0005, and 0009 evaluated against the same ordered candidate universe of 4,308 cells. The stored primary color-aware compatibility predicate yields 4,794 ambiguous queries (`|V(t)| > 1`), including 3,780 where the generating source is supported. The controlled 30 m-cell study covers 564,768 query–source evaluations and 1,192 distinct text templates.

## Frozen evidence

- `data/frozen/PREPARED_PRIMARY_INPUTS.npz`: 4,794 query identifiers, ordered candidates, source indices, scene-specific valid-set counts, bit-packed compatibility, source membership, and Text2Pos/Text2Loc top-10 predictions.
- `data/frozen/QUERY_LEVEL_NULL_PROBABILITIES.csv.gz`: query-aligned source/semantic outcomes and null probabilities used for the matched comparison.
- `data/frozen/TEXT2LOCPP_QUERY_LEVEL_NULL_PROBABILITIES.parquet`: derived query-level null probabilities for the Text2Loc++ statistical extension, not a complete model-ranking record.
- `data/frozen/chains/`: archived constrained-chain captures and diagnostics.
- `results/controlled_and_generator/`, `results/heldout/`, `results/null_control/`, and `results/text2locpp/`: numerical experiment summaries and accompanying tables.

The original parent experiment records identify the ordered candidate-universe SHA-256 as `df64a26bff45406e2fd4cfbcfb5b3e54691dfb6acac9bce7ed231014e13d9eb1`. Hashes of original external parent artifacts are records of the generation run, rather than checks against files supplied in this distribution.

## Integrity

`provenance/frozen_artifact_sha256.csv` records hashes of the included original analysis code, frozen data and result files. `python scripts/run_all_checks.py` checks the presence, length and SHA-256 digest of those files, and verifies source and semantic retrieval outcomes against query-aligned arrays. The original experiment package is staged under its recorded source-tree hashes for regression testing.

## Independent reproduction

The supplied files enable numerical reproduction of the primary ambiguous-cohort Text2Pos/Text2Loc comparison and arithmetic checks of the controlled, null and Text2Loc++ summary tables. Regeneration from raw point clouds, model retraining or full Text2Loc++ ranking recomputation require the corresponding upstream artifacts.
