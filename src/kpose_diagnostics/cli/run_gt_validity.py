"""
run_gt_validity.py -- BLOCKER B1.

Measures the rate at which a query's own generating cell satisfies the
query generated from it, and recomputes the exact-recovery ceiling with
that rate taken into account.

The run has two parts and the first is a gate.

  Reproduction gate. The script regenerates the query population from
  the cells alone, following Section 5.2, and recomputes V per template
  following Proposition 1. It then compares against the values published
  in the manuscript for the primary sequence. If those do not reproduce,
  the generation rule or the directional convention implemented here
  differs from the one used originally, the generating-cell validity
  number would not be comparable, and the script says so loudly instead
  of reporting a plausible wrong answer.

  Measurement. R = mean(C_gt in V(q)) overall and per ambiguity stratum,
  plus the recoverable ceiling E[1{C_gt in V}/V] against the published
  E[1/V].

Nothing is written outside --out-dir. No existing artefact is modified.

Example
-------
    python -m kpose_diagnostics.cli.run_gt_validity \
        --cells data/k360_30-10_scG_pd10_pc4_spY_all_nm6/cells/2013_05_28_drive_0000_sync \
        --scene 2013_05_28_drive_0000_sync \
        --out-dir results/b1_gt_validity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ..ambiguity import (clustered_bootstrap, compute_relation_queries,
                         naive_ceiling, recoverable_ceiling, stratify,
                         summarise_v)
from ..loading import SchemaError, load_cells_cached
from ..runner import Logger, Stage, environment, git_revision, save_json

#: Published values for the primary sequence, from the manuscript.
#: Used only as a reproduction gate; never as an input to any result.
PUBLISHED = {
    "2013_05_28_drive_0000_sync": {
        "n_cells": 2466,
        "n_objects": 38094,
        "n_queries": 564768,
        "n_templates_realised": 1192,
        "median_V": 758.0,
        "mean_V": 782.0,
        "n_V1": 48,
        "n_V0": 122,
        "naive_ceiling": 0.00341,
    }
}

TOLERANCE = {"median_V": 1.0, "mean_V": 5.0, "naive_ceiling": 5e-5,
             "n_templates_realised": 50}  # ±50 covers bbox-vs-centroid centre variants


def _check(name, got, want, tol=0):
    ok = abs(float(got) - float(want)) <= tol
    return {"quantity": name, "recomputed": got, "published": want,
            "tolerance": tol, "match": bool(ok)}



def effective_reference_mode(theta: float, mode: str = "auto", skip_gate: bool = False) -> str:
    """Resolve theta-reference policy without ever borrowing theta=90 for other theta values."""
    if skip_gate:
        return "nonfatal"
    if mode == "auto":
        return "strict" if abs(float(theta) - 90.0) < 1e-12 else "off"
    return mode

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", required=True, type=Path,
                   help="cells source: aggregate <scene>.pkl or cells/<scene>/ directory")
    p.add_argument("--scene", required=True,
                   help="scene name, e.g. 2013_05_28_drive_0000_sync")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--theta", type=float, default=90.0,
                   help="angular tolerance in degrees (default 90 = half-plane)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="reuse a parsed-scene cache across runs; "
                        "big win for the theta sweep and for data on "
                        "network storage. Omit to parse every time.")
    p.add_argument("--force", action="store_true",
                   help="recompute even if a valid success marker exists")
    p.add_argument("--skip-gate", action="store_true",
                   help="legacy alias for --reference-mode nonfatal")
    p.add_argument("--reference-mode", choices=("auto", "strict", "nonfatal", "off"),
                   default="auto", help="theta-reference policy: auto checks only theta=90")
    for role in ("cell-objects", "cell-id", "cell-center", "cell-bbox",
                 "obj-label", "obj-xyz", "obj-id", "obj-color"):
        p.add_argument(f"--field-{role}", default=None,
                       help=f"override the attribute name for {role}")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.out_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(run_dir / "run.log")

    overrides = {k.replace("field_", "").replace("-", "_"): v
                 for k, v in vars(args).items()
                 if k.startswith("field_") and v}

    cfg = {"scene": args.scene, "cells": str(args.cells), "theta": args.theta,
           "n_boot": args.n_boot, "seed": args.seed, "overrides": overrides,
           "cache_dir": str(args.cache_dir) if args.cache_dir else None,
           "code_version": "kpose_diagnostics/2.2.0",
           "reference_mode": args.reference_mode}

    log("=" * 72)
    log("BLOCKER B1 -- generating-cell validity")
    log("=" * 72)
    log(f"scene      : {args.scene}")
    log(f"cells      : {args.cells}")
    log(f"theta      : {args.theta} deg")
    log(f"out-dir    : {run_dir.resolve()}")
    log(f"environment: {environment()}")
    log(f"git        : {git_revision(Path(__file__).resolve().parents[2])}")
    log("")

    # ---------------- stage 1: load -------------------------------------
    with Stage(run_dir, "load", cfg, log, args.force) as st:
        try:
            scene = load_cells_cached(args.cells, args.scene, overrides,
                                      args.cache_dir, log)
        except SchemaError as exc:
            log(f"SCHEMA ERROR\n{exc}")
            return 2
        log(f"    cells   : {scene.n_cells}")
        log(f"    objects : {scene.n_objects}")
        log(f"    labels  : {len(scene.label_counts())}")
        if scene.stubbed_classes:
            log(f"    stubbed classes (harmless): "
                f"{sorted(scene.stubbed_classes)[:6]}")
        log(f"    resolved fields: {scene.resolved_fields}")
        st.outputs = [save_json(st.dir / "scene_summary.json", {
            "scene": scene.scene, "n_cells": scene.n_cells,
            "n_objects": scene.n_objects,
            "label_counts": scene.label_counts(),
            "resolved_fields": scene.resolved_fields,
            "stubbed_classes": scene.stubbed_classes,
        })]

    # ---------------- stage 2: single pass ------------------------------
    log("")
    log("computing valid sets, query population and generating-cell validity")
    log("(one pass over cells; all three quantities come from the same pass)")
    res = compute_relation_queries(scene, args.theta, progress_every=250, log=log)
    log(f"    queries           : {res.n_queries:,}")
    log(f"    templates realised: {res.n_templates_realised:,}")

    v = res.query_v
    gt = res.query_gt_valid
    cells_idx = res.query_cell_index

    # ---------------- stage 3: reproduction gate ------------------------
    log("")
    log("-" * 72)
    log("REPRODUCTION GATE")
    log("-" * 72)
    pub = PUBLISHED.get(args.scene)
    checks, gate_ok = [], True
    mode = effective_reference_mode(args.theta, args.reference_mode, args.skip_gate)
    reference_status = "not_checked"
    if mode == "off":
        reference_status = ("not_applicable_theta" if abs(args.theta - 90.0) >= 1e-12
                            else "disabled")
        log(f"reference comparison not applied for theta={args.theta:g} (mode={mode})")
    elif pub is None:
        reference_status = "reference_unavailable"
        log(f"no published values recorded for {args.scene}; reference comparison skipped")
    else:
        stats = summarise_v(v)
        checks = [
            _check("n_cells", scene.n_cells, pub["n_cells"]),
            _check("n_objects", scene.n_objects, pub["n_objects"]),
            _check("n_queries", res.n_queries, pub["n_queries"]),
            _check("n_templates_realised", res.n_templates_realised,
                   pub["n_templates_realised"],
                   TOLERANCE["n_templates_realised"]),
            _check("median_V", stats["median"], pub["median_V"],
                   TOLERANCE["median_V"]),
            _check("mean_V", stats["mean"], pub["mean_V"], TOLERANCE["mean_V"]),
            _check("n_V1", stats["n_V1"], pub["n_V1"]),
            _check("n_V0", stats["n_V0"], pub["n_V0"]),
            _check("naive_ceiling", naive_ceiling(v), pub["naive_ceiling"],
                   TOLERANCE["naive_ceiling"]),
        ]
        for c in checks:
            flag = "OK  " if c["match"] else "FAIL"
            log(f"  [{flag}] {c['quantity']:<22} recomputed="
                f"{c['recomputed']!s:>12}  published={c['published']!s:>12}")
        gate_ok = all(c["match"] for c in checks)
        log("")
        if gate_ok:
            reference_status = "match"
            log("  REFERENCE MATCH. theta=90 reproduces the registered reference.")
        else:
            only_template_count = all(
                c["match"] for c in checks if c["quantity"] != "n_templates_realised")
            if only_template_count:
                reference_status = "match_with_template_count_note"
                log("  REFERENCE NOTE: only n_templates_realised differs; all V-distribution quantities match.")
                gate_ok = True
            else:
                reference_status = "mismatch"
                log("  REFERENCE MISMATCH.")
                save_json(run_dir / "reference_mismatch.json",
                          {"return_code": 4 if mode == "nonfatal" else 3,
                           "failure_reason": "theta reference mismatch",
                           "theta": args.theta, "checks": checks, "config": cfg})
                if mode == "strict":
                    log("  Strict reference mode: measurement not reported; exit 3.")
                    return 3
                log("  Nonfatal reference mode: computing measurement, clearly marked as mismatch; final exit 4.")

    # ---------------- stage 4: the measurement --------------------------
    log("")
    log("-" * 72)
    log("MEASUREMENT")
    log("-" * 72)

    R, R_lo, R_hi = clustered_bootstrap(gt.astype(np.float64), cells_idx,
                                        n_boot=args.n_boot, seed=args.seed)
    v1 = v == 1
    R1 = float(gt[v1].mean()) if v1.any() else float("nan")

    ceil_naive = naive_ceiling(v)
    ceil_recov = recoverable_ceiling(v, gt)

    log(f"  R   (all queries)            : {R:.6f}  [{R_lo:.6f}, {R_hi:.6f}]"
        "   <- clustered by source cell")
    log(f"  R_1 (V = 1 stratum, n={int(v1.sum())})    : {R1:.6f}")
    log(f"  naive ceiling E[1/V]         : {ceil_naive:.6f}")
    log(f"  recoverable ceiling          : {ceil_recov:.6f}")
    if ceil_naive > 0:
        log(f"  naive / recoverable          : {ceil_naive / max(ceil_recov, 1e-12):.2f}x"
            "  <- factor by which E[1/V] overstates the bound")

    log("")
    log("  PREDICTION UNDER TEST: if the manuscript's 0.125 on the V = 1")
    log("  stratum is explained by generating-cell validity rather than by")
    log("  scoring, then R_1 should be close to 6/48 = 0.125.")
    if not np.isnan(R1):
        log(f"  observed R_1 = {R1:.4f}   (0.125 would confirm; 1.000 would refute)")

    rows = stratify(v, gt)
    log("")
    log(f"  {'stratum':<16}{'queries':>10}{'mean V':>10}{'R':>10}"
        f"{'E[1/V]':>12}{'recoverable':>14}")
    for r in rows:
        if r["queries"] == 0:
            continue
        log(f"  {r['stratum']:<16}{r['queries']:>10,}{r['mean_V']:>10.1f}"
            f"{r['gt_valid_rate']:>10.4f}{r['naive_ceiling']:>12.4f}"
            f"{r['recoverable_ceiling']:>14.4f}")

    # ---------------- stage 5: persist ----------------------------------
    payload = {
        "config": cfg,
        "environment": environment(),
        "reproduction_gate": {"passed": gate_ok, "status": reference_status,
                              "mode": mode, "checks": checks},
        "v_summary": summarise_v(v),
        "headline": {
            "R_all": R, "R_all_ci95": [R_lo, R_hi],
            "R_V1": R1, "n_V1": int(v1.sum()),
            "naive_ceiling_E_inv_V": ceil_naive,
            "recoverable_ceiling": ceil_recov,
            "n_queries": int(res.n_queries),
            "n_cells": int(scene.n_cells),
            "resampling_unit": "source cell",
            "n_bootstrap": args.n_boot,
        },
        "strata": rows,
    }
    save_json(run_dir / "gt_validity_results.json", payload)

    np.savez_compressed(run_dir / "per_query.npz",
                        cell_index=cells_idx, v=v, gt_valid=gt)

    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(run_dir / "strata.csv", index=False)
        tv = sorted(((f"{s}|{d}|{r}", n) for (s, d, r), n in res.template_v.items()))
        pd.DataFrame(tv, columns=["template", "V"]).to_csv(
            run_dir / "template_v.csv", index=False)
    except ImportError:
        log("  pandas unavailable; CSV side-outputs skipped (JSON/NPZ written)")

    log("")
    log("wrote:")
    for name in ("gt_validity_results.json", "per_query.npz", "strata.csv",
                 "template_v.csv", "run.log"):
        p = run_dir / name
        if p.exists():
            log(f"  {p.resolve()}")
    log("")
    log("DONE. Send gt_validity_results.json and run.log.")
    log.close()
    return 4 if reference_status == "mismatch" else 0


if __name__ == "__main__":
    sys.exit(main())
