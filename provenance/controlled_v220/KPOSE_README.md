# kpose_diagnostics v1.1.0

Robust diagnostics for the KITTI360Pose / Text2Pos prepared dataset.

This revision fixes the Colab and real-dataset ingestion failures in v1.0.0. The two core measurements are unchanged in purpose:

- **B1 — generating-cell validity:** tests whether each relation query's generating cell belongs to its own valid set, with a reproduction gate against the manuscript's primary-sequence statistics.
- **B2 — multi-hint ambiguity:** measures valid-set size for KITTI360Pose multi-hint descriptions.

## What v1.1.0 fixes

1. **Folder-aware dataset loading.** Both formats are accepted:

   ```text
   # Prepared / per-record layout
   DATA/
     cells/<scene>/*.pkl
     poses/<scene>/*.pkl

   # Legacy aggregate convenience
   DATA/
     cells/<scene>.pkl
     poses/<scene>.pkl
   ```

2. **Dataset-root discovery.** `kpose_diagnostics.layout` resolves a dataset root, a `cells/` or `poses/` directory, or a scene directory.
3. **Correct cell center fallback.** `bbox_w` is used when no explicit center is stored; only then does the loader fall back to the mean of object centroids.
4. **B2 description compatibility.** Structured object-level descriptions are preferred. If descriptions are plain text but explicit ground-truth object–hint match indices are present, B2 uses those indices and the pose's own cell instead of parsing English text.
5. **Object lookup safety.** For structured descriptions, the referenced object is looked up in the pose's own cell first, with a global-ID fallback only when needed.
6. **Color is no longer silently ignored.** Label-only B2 is the default. `--use-color` runs only when textual colors are genuinely available for every required object/hint; otherwise it exits with an explicit schema error.
7. **Schema inspection is RAM-safe for folder datasets.** `inspect_inputs` samples one record from a per-record directory rather than loading the whole dataset just to inspect field names.
8. **Corrected Colab notebook.** The v1.0.0 notebook had collapsed code-cell line breaks and could not parse. The new notebook has executable cells and auto-discovers both code and data.

## Install

```bash
pip install numpy
pip install pandas   # optional; only for CSV side outputs
```

Python 3.9+ is supported. The diagnostics do not require PyTorch or PyTorch Geometric.

## Recommended Colab path

Use `run_diagnostics_colab_fixed.ipynb` from this release. Set `DATA_BASE` only if auto-discovery does not find the desired prepared dataset. For the standard baseline, the data root commonly has a name like:

```text
k360_30-10_scG_pd10_pc4_spY_all_nm6/
```

The directory itself should contain sibling `cells/` and `poses/` folders.

## Command-line path

From the extracted release directory:

```bash
bash run_all.sh /path/to/k360_30-10_scG_pd10_pc4_spY_all_nm6
```

An optional scene can be supplied as argument 2; the default is:

```text
2013_05_28_drive_0000_sync
```

### Inspect only

```bash
python -m kpose_diagnostics.cli.inspect_inputs \
  --cells /DATA/cells/2013_05_28_drive_0000_sync \
  --poses /DATA/poses/2013_05_28_drive_0000_sync \
  --out results/schema_report.json
```

The `--cells` and `--poses` arguments may each be either a scene directory or an aggregate `.pkl` file.

### B1 only

```bash
python -m kpose_diagnostics.cli.run_gt_validity \
  --cells /DATA/cells/2013_05_28_drive_0000_sync \
  --scene 2013_05_28_drive_0000_sync \
  --out-dir results/b1_gt_validity
```

For the manuscript's primary scene, the script compares recomputed values with the recorded reproduction-gate values. A gate failure is intentionally a nonzero exit: do not quote B1 headline numbers until that mismatch is reconciled.

### B2 only

```bash
python -m kpose_diagnostics.cli.run_multihint \
  --cells /DATA/cells/2013_05_28_drive_0000_sync \
  --poses /DATA/poses/2013_05_28_drive_0000_sync \
  --scene 2013_05_28_drive_0000_sync \
  --out-dir results/b2_multihint
```

Run label-only first. Color is optional:

```bash
python -m kpose_diagnostics.cli.run_multihint ... --use-color
```

If textual color information cannot be recovered from the pickle schema, v1.1.0 refuses to invent a palette or silently degrade to label-only matching.

## Self-validation

```bash
python -m compileall -q kpose_diagnostics
python -m kpose_diagnostics.tests.test_core
python -m kpose_diagnostics.tests.smoke_test
```

The smoke test exercises both aggregate scene pickles and per-record scene folders, including missing-origin-class unpickling.

## Outputs

B1 writes, among other files:

- `results/b1_gt_validity/gt_validity_results.json`
- `results/b1_gt_validity/per_query.npz`
- `results/b1_gt_validity/run.log`

B2 writes:

- `results/b2_multihint/multihint_results.json`
- `results/b2_multihint/per_description.npz`
- `results/b2_multihint/run.log`

Schema inspection writes `results/schema_report.json`.

## Exit codes

- `0`: clean completion
- `2`: schema/input problem
- `3`: a strict scientific/reference gate failed and downstream scientific output was not produced
- `4`: an explicitly nonfatal **expected reference mismatch** (used only where requested, e.g. theta=90 reproduction reporting); this is not an execution failure

For KPOSE verification, `--skip-gate` exists only as a legacy debug escape hatch.
It must not be used to accept zero ambiguity/recall values as scientific results.

## Important integrity rule

The code does not silently guess required fields. When the serialized schema does not support a defensible extraction route, it stops and records what was found. This is deliberate: a visible schema error is preferable to a plausible but scientifically invalid result.

---

## v2.2.0 reproducibility repair

- KPOSE verification follows the Text2Pos generator semantics: object points and candidate positions are compared in normalized cell coordinates; `offset_closest = pose - closest_point`; `on-top` uses the normalized `< 0.05` threshold; and cardinal labels use the generator's exact axis/tie order.
- Missing or invalid direction labels fail loudly. No unknown value silently defaults to north.
- Full object points are retained for exact closest-point evaluation; point subsampling is rejected by the exact KPOSE path.
- Text2Pos numeric RGB is mapped with the official eight colour centres before the colour ablation.
- Failed KPOSE verification writes `verification_failure.json` with return code, reason, gate metrics, confusion matrix, command, version/code hash, and input fingerprints.
- Theta values other than 90 degrees are never compared to the theta=90 reference. Return status distinguishes success, expected reference mismatch, and execution failure.
- `run_parallel` uses durable job signatures and internal KPOSE checkpoints. Parsed scenes, content fingerprints, grid indexes, primary measurements, and convergence results are reused after interruption.
- `--only` is part of the durable plan and is packed across tracks, so a targeted repair rerun can be split efficiently across two Colab runtimes.
- `--merge` embeds result JSON, logs and failure diagnostics, and writes `MERGED_RESULTS.json`, `RUN_SUMMARY.txt`, `MANIFEST.json`, and a self-contained zip archive.

### Recommended two-Colab targeted rerun

Create the plan once, then run tracks 0 and 1 against the same Drive output directory:

```bash
python -m kpose_diagnostics.cli.run_parallel --base <dataset> --out <results> \
  --tracks 2 --only 'kpose_*,theta_45,theta_22.5,theta_10' --plan
python -m kpose_diagnostics.cli.run_parallel --base <dataset> --out <results> \
  --tracks 2 --track 0 --only 'kpose_*,theta_45,theta_22.5,theta_10'
python -m kpose_diagnostics.cli.run_parallel --base <dataset> --out <results> \
  --tracks 2 --track 1 --only 'kpose_*,theta_45,theta_22.5,theta_10'
python -m kpose_diagnostics.cli.run_parallel --out <results> --merge
```

If interrupted, rerun the same track command. Durable completed work is reused.
