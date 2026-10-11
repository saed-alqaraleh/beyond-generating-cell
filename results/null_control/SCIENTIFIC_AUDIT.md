# Scientific audit — source-membership-matched semantic-null control v2.1

**STATUS: PASS**

- Query cardinality is preserved exactly.
- Per-query candidate-scene cardinalities are preserved exactly.
- The complete source-membership vector is preserved in every retained realization.
- Candidate-cell column degrees are preserved exactly within scenes by constrained bipartite switches.
- Text2Pos/Text2Loc rankings are loaded from the frozen prediction files; no inference is rerun.
- The same null realization is scored for both models.
- Four chains × 250 retained = 1000 total null realizations.
- Cluster bootstrap replicates: 10000.
- The older unconditional configuration-model control is retained as a labelled secondary analysis when available; it is not required for the primary corrected control.
- Rejected switch proposals remain on the Markov-chain clock as self-transitions; retained states are separated by fixed proposal counts, not fixed accepted-move counts.

The archive must not be interpreted as proving uniform sampling over all feasible constrained incidence matrices; finite switch-chain convergence is assessed empirically.
