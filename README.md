# Beyond the Generating Cell

**Identifiability-aware set-valued evaluation for text-to-point-cloud localization**

Research code, frozen evaluation inputs, numerical results, and verification tools accompanying *Beyond the Generating Cell: Identifiability-Aware Set-Valued Evaluation for Text-to-Point-Cloud Localization*.

The evaluation distinguishes recovery of the benchmark's designated source cell from retrieval of any map cell satisfying the query's declared spatial-semantic constraints. The package includes the exact text-only source-identifiability calculation, generator-aware compatibility diagnostics, evaluation of frozen Text2Pos/Text2Loc rankings, and a source-membership-matched semantic null. Summary results for Text2Loc++ are included with their statistical audits.

## Installation and verification

Python **3.12** is required for the archived execution suite. No GPU or external dataset is needed for the included frozen-evidence checks.

```bash
git clone https://github.com/saed-alqaraleh/beyond-generating-cell.git
cd beyond-generating-cell
python -m pip install -r requirements.txt
python scripts/run_all_checks.py
```

The verification command restores one checksummed score archive from its constituent parts, validates the file hashes, recomputes query-level retrieval and compatibility statistics from the frozen arrays, checks the reported numerical tables, and runs the experimental code's regression and synthetic end-to-end tests. It exits with a nonzero status if any validation fails.

The same procedure runs automatically under [GitHub Actions](.github/workflows/reproducibility.yml). For a browser-based environment, see [the verification notebook](notebooks/00_Verify_Frozen_Evidence.ipynb).

## Research components

| Component | Included artifacts | Verification scope |
| --- | --- | --- |
| Controlled spatial-relation analysis | Generator predicates, enumeration outputs, fusion statistics | Exact-source bound, population counts, original regression tests |
| Held-out evaluation | 11,505-query audit summaries and 4,794 ambiguous-query compatibility arrays | Query-aligned source/semantic Recall@1/5/10 for frozen Text2Pos/Text2Loc top-10 rankings |
| Matched semantic null | Switching algorithms, four chain outputs, query-level probabilities, statistical tables | Marginal invariants, result consistency, numerical contrasts and regression tests |
| Text2Loc++ evaluation | Derived per-query null-probability table, statistical summaries and reported comparisons | Aggregate consistency and reported-effect arithmetic |

The held-out candidate universe consists of **4,308 cells**; **4,794 of 11,505 queries** have more than one generator-compatible candidate. These counts are verified against the included research artifacts.

## Code and results

- [`src/`](src/) — spatial predicates, identifiability analysis, experimental execution, and matched-null algorithms
- [`scripts/`](scripts/) — reproducibility checks, archive integrity checks, and execution entry points
- [`notebooks/`](notebooks/) — executable frozen-evidence verification and prepared-data experiment workflows
- [`data/frozen/`](data/frozen/) — query-aligned compatibility inputs and archived null-chain files
- [`results/`](results/) — numeric study outputs, tables, and statistical analyses
- [`provenance/`](provenance/) — source identity, SHA-256 records, experiment configurations, and integrity manifests

The [reproduction guide](REPRODUCIBILITY.md) provides command-line and Colab instructions. [Data provenance](DATA_PROVENANCE.md) documents the exact contents and external data dependencies.

## Data and reproducibility scope

The included bit-packed compatibility matrices and frozen predictions permit independent reevaluation of the ambiguous-query Text2Pos/Text2Loc source and semantic metrics without rerunning model inference. The complete controlled study and source-matched null also include archived numerical outputs and executable implementations.

A full computation starting from original KITTI360Pose point clouds and model checkpoints requires the separately distributed upstream datasets and baseline implementations. The original Text2Loc++ candidate rankings and raw per-query semantic audit are not included; its published summary statistics are verifiable from the supplied aggregate and derived artifacts. The package does not redistribute original point clouds or pretrained weights.

## Upstream resources and licensing

The author-developed evaluation software is released under the MIT license. [License scope](LICENSE_SCOPE.md) distinguishes this software from dataset-derived artifacts and externally licensed material. [Third-party notices](THIRD_PARTY_NOTICES.md) document upstream resources and applicable restrictions.

## Citation

The research article title and author information are recorded in [`CITATION.cff`](CITATION.cff). Software-version identifiers and permanent archive citations can be added when a tagged archival release is deposited.
