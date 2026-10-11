# Reproducing the reported results

## Offline frozen-evidence evaluation

**Environment:** Python 3.12 and the dependencies in `requirements.txt`. This mode runs on CPU and requires no original KITTI360Pose files.

```bash
python -m pip install -r requirements.txt
python scripts/run_all_checks.py
```

`run_all_checks.py` performs archive reconstruction and integrity verification, checks the original source and frozen data, recalculates source and semantic Recall from the included ambiguous-cohort arrays, reconciles the tabulated null effects, and exercises the original unit and synthetic integration tests. The script terminates with an error on a failed check. An independent machine-readable report can be produced with:

```bash
python scripts/verify_release.py --report verification_report.json
```

The score archive under `results/controlled_and_generator/jobs/template_analysis_primary/` is assembled from byte-verified chunks by `scripts/restore_large_archive.py`. Both the part checksums and reconstructed output checksum are checked before the file is used. The integrity manifest is maintained separately at `provenance/frozen_artifact_sha256.csv`.

## Experimental modules

| Module | Purpose |
| --- | --- |
| `src/kpose_diagnostics/` | Spatial geometry, generator agreement, exact-input multiplicity, compatible-set computation and controlled experiment drivers |
| `src/article_evidence/` | Controlled factorial analysis, scheduled execution and resumable experimental tasks |
| `src/source_matched_null/` | Source-membership-matched bipartite switch-chain sampling and statistical inference |
| `scripts/verify_release.py` | Independent query-aligned frozen-data and table checks |
| `scripts/check_manuscript_alignment.py` | Comparison of LaTeX numerical tables with archived result values |
| `scripts/run_historical.py` | Prepared-data replay of the archived experimental pipeline |

## Prepared-data replay

The generator and controlled-study runners require the original prepared KITTI360Pose data, with `cells/` and `poses/` subdirectories. To replay the archived two-track experiment, stage the verified source and create a durable output directory:

```bash
python scripts/run_historical.py core-tests --stage /path/to/staged
python scripts/run_historical.py executor-tests --stage /path/to/staged
python scripts/run_historical.py plan --stage /path/to/staged --dataset /path/to/prepared --out /path/to/output
python scripts/run_historical.py track --stage /path/to/staged --out /path/to/output --track 0
python scripts/run_historical.py track --stage /path/to/staged --out /path/to/output --track 1
python scripts/run_historical.py status --stage /path/to/staged --out /path/to/output
python scripts/run_historical.py finalize --stage /path/to/staged --out /path/to/output
```

The two tracks may execute in separate environments while sharing a persistent output directory. A stopped track may be restarted with the same command; completed stages are checked against their saved signatures. These commands do not download KITTI360Pose, train public baselines, or infer Text2Loc++ rankings.

For Colab, use `notebooks/10_Historical_Article_Track_0.ipynb` and `notebooks/10_Historical_Article_Track_1.ipynb`. `notebooks/00_Verify_Frozen_Evidence.ipynb` provides the dataset-independent validation path.

## Interpretation of verification

Recalculated metrics from frozen predictions establish evaluation reproducibility and preserve the original model outputs. Regenerating the model predictions themselves requires upstream checkpoints and the native training/inference environments. The frozen Text2Loc++ extension provides statistical tables and derived query-level probabilities but not the original complete candidate-ranking matrix.
