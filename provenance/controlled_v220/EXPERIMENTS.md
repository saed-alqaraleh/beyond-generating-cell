# Experiment register

## Running

The jobs differ in cost by more than an order of magnitude, so they are
packed across tracks rather than split by kind.

**Two runtimes** (two Colab notebooks, or two terminals). The assignment is
deterministic, so the tracks partition the work with no coordination:

```bash
python -m kpose_diagnostics.cli.run_parallel --base <dataset> --out results_all --tracks 2 --plan
python -m kpose_diagnostics.cli.run_parallel --base <dataset> --out results_all --tracks 2 --track 0   # runtime 1
python -m kpose_diagnostics.cli.run_parallel --base <dataset> --out results_all --tracks 2 --track 1   # runtime 2
```

**One runtime**, concurrent subprocesses:

```bash
python -m kpose_diagnostics.cli.run_parallel --base <dataset> --out results_all --tracks 1 --track 0 --workers 2
```

**Collect**, once the tracks finish:

```bash
python -m kpose_diagnostics.cli.run_parallel --out results_all --merge
```

`--status` prints progress at any time, including mid-run. `run_everything.sh`
remains available for a serial run.

Expect roughly 15–40 minutes serial, and appreciably less across two tracks;
the single dominant job is the primary KITTI360Pose run with its resolution
sweep, which the packer isolates.

### What makes concurrent runs safe

- Every job writes to its own output directory, so no two jobs write one file.
- Each track gets its own scene cache, so the cache is never written twice at
  once. The cost is re-parsing the scene once per track — seconds.
- A finished job writes `_JOB_DONE.json`; any invocation skips it. An
  interrupted run is resumed by reissuing the same command, and a job cannot
  run twice even if the plan changes between invocations.
- The track count is recorded in `PLAN.json`, and a mismatched `--tracks` is
  refused rather than silently producing a different partition.
- A failing job is recorded and does not stop its track.

| # | Experiment | Answers | Status |
|---|---|---|---|
| E1 | T1/T2 gap | Does the metric or the model produce the headline? | done |
| E2 | Template-query ambiguity `V` | How many cells satisfy each query? | done |
| E3 | Generating-cell validity `R` | Is a query true of its own source cell? | done (0.740) |
| E4 | θ sweep | Is ambiguity an artefact of a loose predicate? | done |
| E5 | Multi-sequence | Is it particular to one scene? | done (4 scenes) |
| **E6** | **Predicate verification** | **Does our rule reproduce the generator?** | **new** |
| **E7** | **KITTI360Pose ambiguity** | **How ambiguous are the benchmark's own descriptions?** | **new** |
| **E8** | **KITTI360Pose validity** | **Are they true of their own source cell?** | **new** |
| **E9** | **recall@k metric ceiling** | **What can the published metric mean?** | **new** |
| E10 | Colour ablation | What do appearance attributes buy? | new |
| E11 | Resolution convergence | Is E9 stable in grid spacing? | new |

## The four new measurements

**E6 — verification gate.** Every description carries the direction its
generator assigned. We recompute that direction from the stored pose and
object geometry and compare. Agreement is swept over the proximity
threshold δ. Nothing downstream is reported unless agreement is high: a
predicate that does not reproduce the generator measures its own
disagreement, which is precisely the error that invalidated our first
attempt at this measurement.

**E7 — ambiguity of the benchmark's own descriptions.** `V` is the number
of cells containing at least one position consistent with every hint. Note
the formulation: a description describes a *position*, so a cell counts as
valid when some position inside it is consistent, not when the cell centre
happens to be. Testing at the centre asks a different and much stricter
question and would report `V = 0` even for the generating cell.

**E8 — generating-cell validity for those descriptions.** The same
question E3 asks of the template family. Where a description is not
satisfied by its own source cell, exact retrieval is unattainable and
`E[1/V]` overstates the bound.

**E9 — the metric ceiling.** This is the measurement with the widest
reach. Published work reports `recall@k m`. E7 already yields, for each
description, the *set of positions* consistent with it. The fraction of
that set within `k` metres of the true pose is the recall an oracle would
achieve — one that identifies the described region exactly but cannot
order within it. It is fixed by the descriptions and the map before any
model is trained, and it bounds what a reported recall figure can mean.
E11 re-runs it at several grid resolutions so convergence is visible
rather than assumed.

## Outputs to send back

```
results_all/RUN_SUMMARY.txt
results_all/b1_*/gt_validity_results.json
results_all/kpose_*/kpose_full_results.json
results_all/theta_*/gt_validity_results.json
```

## If a stage fails

`RUN_SUMMARY.txt` records which. The two informative failure modes:

- **exit 3 from `run_kpose_full`** — the verification gate failed. This is
  itself a finding: the generation rule differs from the documented one.
  `verification_failure.json` holds the confusion matrix, from which the
  actual rule can be recovered.
- **exit 2** — a schema field could not be resolved. The message names what
  was tried and what exists.
