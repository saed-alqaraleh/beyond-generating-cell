"""KITTI360Pose description ambiguity with a generator-faithful verification gate.

This lightweight runner performs verification + cell-level ambiguity only.
For the full verification/ambiguity/validity/metric-ceiling pipeline use
``run_kpose_full``.  Both use the same exact Text2Pos closest-point predicate.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..ambiguity import naive_ceiling, summarise_v
from ..kpose_predicate import (DEFAULT_DELTA, calibrate_delta,
                               normalize_direction_label,
                               verify_against_stored)
from ..loading import SchemaError, describe, get, load_cells, load_pickle, load_pickle_entries
from ..runner import (Logger, code_tree_hash, environment, fingerprint_path,
                      save_json)
from .run_kpose_full import build_index


def _iter(raw):
    if isinstance(raw, dict):
        return list(raw.values())
    if isinstance(raw, (list, tuple)):
        return list(raw)
    raise SchemaError(f"poses pickle holds {type(raw).__name__}")


def _resolve(obj, names):
    have = set(vars(obj)) if hasattr(obj, "__dict__") else set()
    if isinstance(obj, dict):
        have |= set(obj)
    for n in names:
        if n in have or hasattr(obj, n):
            return n
    return None


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells", required=True, type=Path)
    p.add_argument("--poses", required=True, type=Path)
    p.add_argument("--scene", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--delta", type=float, default=DEFAULT_DELTA,
                   help="on-top threshold in normalized cell units (official 0.05)")
    p.add_argument("--grid", type=int, default=12)
    p.add_argument("--cell-size", type=float, default=30.0)
    p.add_argument("--use-color", action="store_true")
    p.add_argument("--min-agreement", type=float, default=0.95)
    p.add_argument("--skip-gate", action="store_true",
                   help="DEBUG ONLY: continue but mark output scientifically invalid")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "run.log")
    log("=" * 72)
    log("KITTI360Pose DESCRIPTION AMBIGUITY -- generator-faithful v2.2.0")
    log("=" * 72)

    try:
        # Full object points are required: Text2Pos directions use the exact
        # closest point, not an object centroid or a point subsample.
        scene = load_cells(args.cells, args.scene, None, keep_points=-1)
        poses_raw = load_pickle(args.poses)[0] if args.poses.is_file() else load_pickle_entries(args.poses)[0]
    except (SchemaError, FileNotFoundError) as exc:
        log(f"LOAD/SCHEMA ERROR: {exc}")
        return 2
    poses = _iter(poses_raw)
    if not poses:
        log("poses source contained no entries")
        return 2

    probe = poses[0]
    f_desc = _resolve(probe, ("descriptions", "descriptors", "hints"))
    d_probe = None
    if f_desc:
        for p_ in poses[:200]:
            ds = list(get(p_, f_desc))
            if ds:
                d_probe = ds[0]
                break
    if d_probe is None:
        save_json(out / "poses_schema.json", describe(probe, 4))
        log("could not resolve non-empty descriptions")
        return 2
    f_dlab = _resolve(d_probe, ("object_label", "label", "cls"))
    f_ddir = _resolve(d_probe, ("direction", "direction_text", "relation"))
    f_dcol = _resolve(d_probe, ("object_color_text", "color_text", "color"))
    f_doff = _resolve(d_probe, ("offset_closest",))
    if not (f_dlab and f_ddir and f_doff):
        save_json(out / "descr_schema.json", describe(d_probe, 4))
        log("description must contain object label, direction, and offset_closest")
        return 2

    fp_cache = out.parent / "_fingerprints"
    fingerprints = {"cells": fingerprint_path(args.cells, fp_cache),
                    "poses": fingerprint_path(args.poses, fp_cache)}
    code_hash = code_tree_hash(Path(__file__).resolve().parents[2])

    records = []
    try:
        for p_ in poses:
            for d in get(p_, f_desc):
                records.append((np.asarray(get(d, f_doff), float).ravel()[:2],
                                normalize_direction_label(get(d, f_ddir))))
        report = verify_against_stored(records, args.delta)
    except (ValueError, TypeError) as exc:
        save_json(out / "verification_failure.json", {
            "return_code": 2, "failure_reason": str(exc), "gate_metrics": None,
            "confusion_matrix": {},
            "command": [sys.executable, "-m", "kpose_diagnostics.cli.run_kpose_ambiguity"] + list(sys.argv[1:]),
            "version": "2.2.0", "code_hash": code_hash,
            "input_fingerprint": fingerprints})
        log(f"verification input error: {exc}")
        return 2

    sweep = calibrate_delta(records)
    for line in report.summary().splitlines():
        log(line)
    gate_ok = report.agreement >= args.min_agreement
    if not gate_ok:
        diag = {
            "return_code": 3,
            "failure_reason": "KPOSE generator-verification agreement below min_agreement",
            "gate_metrics": {"agreement": report.agreement,
                             "min_agreement": args.min_agreement,
                             "n_compared": report.n_compared,
                             "delta": args.delta, "delta_sweep": sweep,
                             "per_direction": {k: list(v) for k, v in report.per_direction.items()}},
            "confusion_matrix": {f"{a}->{b}": n for (a, b), n in report.confusion.items()},
            "command": [sys.executable, "-m", "kpose_diagnostics.cli.run_kpose_ambiguity"] + list(sys.argv[1:]),
            "version": "2.2.0", "code_hash": code_hash,
            "input_fingerprint": fingerprints,
        }
        save_json(out / "verification_failure.json", diag)
        if not args.skip_gate:
            log("GATE FAILED; wrote verification_failure.json; no ambiguity result computed.")
            return 3
        log("DEBUG ONLY: continuing after failed gate; result marked scientifically invalid.")

    centres = np.asarray([c.center[:2] for c in scene.cells], float)
    packed, _, n_slots = build_index(scene, centres, args.grid, args.cell_size,
                                     args.delta, args.use_color, log)
    per_cell = args.grid * args.grid
    sig_v = {}
    v_list, nh_list = [], []
    for p_ in poses:
        hints = []
        for d in get(p_, f_desc):
            label = str(get(d, f_dlab))
            direction = normalize_direction_label(get(d, f_ddir))
            color = (str(get(d, f_dcol)).strip().lower()
                     if args.use_color and f_dcol else None)
            hints.append((label, color, direction))
        if not hints:
            continue
        sig = tuple(sorted(set(hints)))
        if sig not in sig_v:
            acc = None
            missing = False
            for h in sig:
                a = packed.get(h)
                if a is None:
                    missing = True
                    break
                acc = a if acc is None else (acc & a)
            if missing or acc is None:
                sig_v[sig] = 0
            else:
                slots = np.flatnonzero(np.unpackbits(acc)[:n_slots])
                sig_v[sig] = int(np.unique(slots // per_cell).size)
        v_list.append(sig_v[sig])
        nh_list.append(len(hints))

    v = np.asarray(v_list, dtype=np.int32)
    nh = np.asarray(nh_list, dtype=np.int32)
    stats = summarise_v(v)
    rows = []
    by_n = defaultdict(list)
    for k, vi in zip(nh, v):
        by_n[int(k)].append(int(vi))
    for k in sorted(by_n):
        a = np.asarray(by_n[k])
        rows.append({"n_hints": k, "n_descriptions": int(a.size),
                     "median_V": float(np.median(a)),
                     "pct_unique": float(100.0 * np.mean(a == 1))})

    save_json(out / "kpose_ambiguity_results.json", {
        "config": {"scene": args.scene, "delta": args.delta,
                   "grid": args.grid, "cell_size": args.cell_size,
                   "use_color": args.use_color,
                   "code_version": "kpose_diagnostics/2.2.0",
                   "code_hash": code_hash,
                   "input_fingerprint": fingerprints,
                   "scientifically_valid": bool(gate_ok)},
        "environment": environment(),
        "verification": {"agreement": report.agreement,
                         "n_compared": report.n_compared,
                         "passed": gate_ok,
                         "delta_sweep": sweep,
                         "per_direction": {k: list(vv) for k, vv in report.per_direction.items()},
                         "confusion": {f"{a}->{b}": n for (a, b), n in report.confusion.items()}},
        "v_summary": stats,
        "naive_ceiling": naive_ceiling(v),
        "by_hint_count": rows,
    })
    np.savez_compressed(out / "per_description.npz", v=v, n_hints=nh)
    log(f"wrote {(out / 'kpose_ambiguity_results.json').resolve()}")
    log.close()
    return 0 if gate_ok else 4


if __name__ == "__main__":
    sys.exit(main())
