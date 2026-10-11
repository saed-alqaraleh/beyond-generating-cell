
# Source-membership-matched semantic-null control v2.1

This package is the audited correction to the source-membership-matched control.
It fixes the Markov-chain clock so rejected switch proposals remain explicit
self-transitions; retained states are separated by fixed proposal counts rather
than fixed accepted-move counts.

## Scientific correction

For every primary query with `V_i > 1`, the null now fixes the released-source
position to its observed semantic membership:

- if `source_supported[i] == 1`, the source edge is pinned in every null set;
- if `source_supported[i] == 0`, the source position is a structural zero.

Only the remaining free edges are randomized.

Within each candidate scene, constrained bipartite 2×2 switches preserve:

1. exact per-query scene-specific cardinality;
2. exact total `V_i`;
3. exact candidate-cell column degrees;
4. the complete source-support vector.

The same randomized realization is scored against the frozen Text2Pos and
Text2Loc top-10 rankings. No model inference is rerun and no ranking is
reconstructed.

Primary cohort: `V>1`, n=4,794.
Diagnostic cohort: `V>1` and source-supported, n=3,780.

## Required frozen parent inputs

The runner hard-locks:

- held-out evidence ZIP SHA-256  
  `961934e5031459785663185c80dafd453451565bdd55c19d59de6930536446ac`
- primary color valid-set SHA-256  
  `fb062c7b743a39a368552ea85be6cbfecfa83136201fe9948b3af136f9561234`
- candidate-universe SHA-256  
  `df64a26bff45406e2fd4cfbcfb5b3e54691dfb6acac9bce7ed231014e13d9eb1`
- frozen Text2Pos/Text2Loc prediction hashes.

The older unconditional null-control archive is optional. If found, it is retained
and clearly labelled as a secondary analysis; its absence does not block the
corrected primary control. By default the runner
looks for:

`CARDINALITY_MATCHED_NULL_CONTROL_FINAL_v1.zip`

under:

`/content/drive/MyDrive/or_fusion_verification/heldout_identifiability_v1`

If its name/path differs, pass `--legacy-null-zip`.

## Colab execution

Upload/extract this code package, then run:

```bash
python -m source_matched_null.tests
```

Small real-data smoke:

```bash
python run_source_membership_matched_null_v2.py --mode smoke
```

Full corrected control:

```bash
python run_source_membership_matched_null_v2.py --mode full --workers 4
```

If the legacy v1 archive has another path:

```bash
python run_source_membership_matched_null_v2.py \
  --mode full \
  --workers 4 \
  --legacy-null-zip /content/drive/MyDrive/.../YOUR_V1_NULL_ARCHIVE.zip
```

Audit-only after completion:

```bash
python run_source_membership_matched_null_v2.py --mode audit-only
```

Figure-only:

```bash
python run_source_membership_matched_null_v2.py --mode figure-only
```

## Resumability

Each chain has an independent deterministic seed and writes an atomic
checkpoint containing:

- exact free-edge arrays;
- NumPy RNG state;
- burn-in progress;
- retained-repetition progress;
- query-level hit accumulators;
- repetition-level Recall numerators;
- provenance hashes.

A restart resumes the exact Markov trajectory from the last valid checkpoint.

## Mandatory gates

The full run refuses to create a PASS archive unless:

- primary `V>1` n = 4,794;
- source-supported primary n = 3,780;
- all frozen source Recall locks reproduce;
- all existing H1 source/semantic counts reproduce;
- every retained null realization satisfies  
  `null_contains_source[i,r] == source_supported[i]`;
- every query's total and per-scene cardinalities remain exact;
- candidate column degrees remain exact;
- all diagnostic source-supported null Recalls are at least source Recall;
- the explicit conservative full-cohort lower bounds pass;
- all four chains return 250 retained realizations;
- split-Rhat ≤ 1.10 for all six primary null-Recall chains;
- 10,000 paired scene-stratified source-cell-clustered bootstrap replicates finish;
- manifests and hashes pass.

The v2 final archive is created only after all mandatory integrity gates pass:

`SOURCE_MEMBERSHIP_MATCHED_SEMANTIC_NULL_CONTROL_FINAL_v2_1.zip`

The earlier unconditional configuration-model null is copied into the v2
archive under `LEGACY_UNCONSTRAINED_SECONDARY/` and is explicitly marked
secondary.

## v2.1 audit correction

The earlier v2 implementation advanced burn-in/thinning by a fixed number of
**accepted** switches. That records the embedded jump chain. Because switch
acceptance can vary across feasible states, the jump chain can weight states by
acceptance rate. v2.1 instead advances by a fixed number of **proposal**
transitions, including rejected proposals as self-loops. This is the correct
clock for the symmetric constrained-switch Markov chain.

The ordinary clustered-bootstrap sign fraction is retained only as a descriptive
diagnostic and is not called a p-value. Confirmatory inference is based on the
six familywise simultaneous 95% clustered-bootstrap intervals.
