
from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import ensure_runtime_deps, make_figure


def main():
    ensure_runtime_deps()
    import pandas as pd
    p = argparse.ArgumentParser()
    p.add_argument("final_dir", type=Path)
    args = p.parse_args()
    primary = pd.read_csv(
        args.final_dir / "TABLE_source_membership_matched_null_primary.csv"
    ).to_dict("records")
    diag = pd.read_csv(
        args.final_dir / "TABLE_source_supported_diagnostic.csv"
    ).to_dict("records")
    make_figure(primary, diag, args.final_dir)
    print("FIGURE REGENERATION: PASS")


if __name__ == "__main__":
    main()
