"""
run_multihint.py -- ambiguity of the KITTI360Pose multi-hint descriptions.

This is the measurement that connects the protocol to the query family
the benchmark itself uses. For every description in the selected ``poses/<scene>`` source
it computes V, the number of cells in the same map that satisfy every
hint, and reports the distribution alongside the single-relation family
measured in the manuscript.

Directions are derived geometrically, not parsed from text: for each
referenced object, the direction of the described position relative to
that object's centroid is computed with the same sigma_v convention used
throughout the manuscript. Nothing depends on the wording of the hint or
on a ``direction`` string being present in the pickle.

Template invariance (Proposition 1) extends to hint sets: V depends only
on the multiset of hints, so descriptions are grouped by canonical
signature and V computed once per signature. The realised speed-up is
reported.

Example
-------
    python -m kpose_diagnostics.cli.run_multihint \
        --cells data/k360_30-10_scG_pd10_pc4_spY_all_nm6/cells/2013_05_28_drive_0000_sync \
        --poses data/k360_30-10_scG_pd10_pc4_spY_all_nm6/poses/2013_05_28_drive_0000_sync \
        --scene 2013_05_28_drive_0000_sync \
        --out-dir results/b2_multihint
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..ambiguity import (Hint, compute_multihint_v, hint_delta,
                         hint_signature, multihint_gt_validity,
                         naive_ceiling, summarise_v)
from ..geometry import DIRECTIONS, satisfies
from ..loading import (SchemaError, SceneData, describe, get, load_cells_cached,
                       load_pickle_entries, pick)
from ..runner import Logger, Stage, environment, save_json


def _iter(raw):
    if isinstance(raw, dict):
        yield from raw.values()
    elif isinstance(raw, (list, tuple)):
        yield from raw
    else:
        raise SchemaError(f"poses pickle holds {type(raw).__name__}")


# Attribute names for the 2D object-offset field, tried in order.
_OFFSET_CANDIDATES = ('offset_center', 'offset_closest',
                      'best_offset_center', 'best_offset_closest')


def _resolve_offset(obj) -> str | None:
    """Return the attribute name for the 2D object-offset vector, or None.

    Checks both instance dict and class hierarchy via hasattr, because
    the permissive unpickler creates stub classes whose attributes may
    live on the class rather than the instance.
    """
    have = set(vars(obj)) if hasattr(obj, '__dict__') else set()
    if isinstance(obj, dict):
        have |= set(obj)
    for name in _OFFSET_CANDIDATES:
        if name in have or hasattr(obj, name):
            return name
    return None


def _direction_of_position(position_xy: np.ndarray,
                           object_xy: np.ndarray) -> str | None:
    """Direction of ``position`` relative to ``object``, dominant axis.

    Uses :func:`hint_delta` so that the name produced here and the test
    applied to candidate cells share one convention -- the manuscript's.
    A hint states a single direction, so the 45-degree dominant-axis
    criterion is the right reading. Returns None for an exact diagonal or
    a zero displacement, which the caller counts rather than guessing.
    """
    ok = satisfies(hint_delta(position_xy, object_xy)[None, :], 45.0)[0]
    idx = np.flatnonzero(ok)
    return DIRECTIONS[int(idx[0])] if idx.size else None


def extract_descriptions(poses_raw, scene: SceneData, log,
                         overrides: dict) -> tuple[list[list[Hint]], list, dict]:
    """Build hint lists from poses, deriving directions from geometry.

    Two input encodings are supported:
      * structured description objects carrying label / object id; and
      * ground-truth ``matches`` pairs ``(object_index, hint_index)`` tied to
        the pose's own cell. The latter avoids parsing natural-language text.
    """
    poses = list(_iter(poses_raw))
    if not poses:
        raise SchemaError("poses pickle contained no entries")

    probe = poses[0]
    f_pos = pick(probe, "pose_position", overrides)
    try:
        f_cell = pick(probe, "pose_cell_id", overrides)
    except SchemaError:
        if overrides.get("pose_cell_id"):
            raise
        f_cell = None

    # Description objects are preferred because they preserve object ids even
    # when a pose does not point at its generating cell. If they are strings
    # (or absent), fall back to explicit object-hint match indices when stored.
    try:
        f_desc = pick(probe, "pose_descrs", overrides)
    except SchemaError:
        if overrides.get("pose_descrs"):
            raise
        f_desc = None
    try:
        f_matches = pick(probe, "pose_matches", overrides)
    except SchemaError:
        if overrides.get("pose_matches"):
            raise
        f_matches = None

    d_probe = None
    if f_desc:
        for p in poses[:200]:
            try:
                ds = list(get(p, f_desc))
            except TypeError:
                ds = []
            if ds:
                d_probe = ds[0]
                break

    structured = d_probe is not None and not isinstance(d_probe, (str, bytes))
    f_dlab = f_doid = f_dcol = f_ddir = None
    f_doffset = None
    if d_probe is not None:
        f_doffset = overrides.get("descr_offset") or _resolve_offset(d_probe)
        if f_doffset:
            log(f"    offset field resolved: {f_doffset} (object-centroid relative to cell centre)")
        else:
            log("    NOTE: no offset_center field found on description; will rely on object-id lookup")

    if structured:
        try:
            f_dlab = pick(d_probe, "descr_label", overrides)
        except SchemaError:
            # A non-string container can still be an opaque wrapper. In that
            # case the explicit matches representation remains a valid route.
            if overrides.get("descr_label") or f_matches is None:
                raise
            structured = False
        if structured:
            try:
                f_doid = pick(d_probe, "descr_obj_id", overrides)
            except SchemaError:
                if overrides.get("descr_obj_id"):
                    raise
                f_doid = None
            try:
                f_dcol = pick(d_probe, "descr_color", overrides)
            except SchemaError:
                if overrides.get("descr_color"):
                    raise
                f_dcol = None
            try:
                f_ddir = pick(d_probe, "descr_dir", overrides)
            except SchemaError:
                f_ddir = None

    if not structured and f_matches is None:
        kind = type(d_probe).__name__ if d_probe is not None else "none"
        raise SchemaError(
            "could not recover object-level hints from the pose records. "
            f"The descriptions field contains {kind} entries and no explicit "
            "matches field was found. Run inspect_inputs and provide the "
            "schema report; this version deliberately does not guess object "
            "labels or directions by parsing free-form text."
        )
    if not structured and f_cell is None:
        raise SchemaError(
            "pose match-index fallback requires a pose cell id, but none was "
            "resolved. Supply --field-pose-cell-id after running inspect_inputs."
        )

    mode = "structured_descriptions" if structured else "object_hint_matches"
    log(f"    pose extraction mode: {mode}")
    log(f"    resolved pose fields: position={f_pos} descriptions={f_desc} "
        f"cell={f_cell} matches={f_matches} label={f_dlab} object_id={f_doid} "
        f"offset={f_doffset} color={f_dcol} direction={f_ddir}")

    # Prefer the referenced object in the pose's own cell. The same global
    # instance can occur in overlapping cells with clipped point subsets.
    by_cell_id = {str(c.cell_id): c for c in scene.cells}
    centroid_of: dict[str, np.ndarray] = {}
    label_of: dict[str, str] = {}
    color_of: dict[str, str | None] = {}
    for cell in scene.cells:
        for oid, c, lab, col in zip(cell.object_ids, cell.centroids,
                                    cell.labels, cell.colors):
            key = str(oid)
            centroid_of.setdefault(key, c)
            label_of.setdefault(key, str(lab))
            color_of.setdefault(key, col)

    descriptions, meta = [], []
    n_no_dir = n_unmatched = n_dir_disagree = 0
    dir_confusion: dict = {}
    n_bad_matches = n_missing_cells = 0
    for p in poses:
        pos = np.asarray(get(p, f_pos), dtype=np.float64).ravel()[:3]
        pose_cid = str(get(p, f_cell)) if f_cell else None
        own_cell = by_cell_id.get(pose_cid) if pose_cid is not None else None
        own_centroid = {}
        own_label = {}
        own_color = {}
        if own_cell is not None:
            own_centroid = {str(oid): c for oid, c in zip(own_cell.object_ids, own_cell.centroids)}
            own_label = {str(oid): str(lab) for oid, lab in zip(own_cell.object_ids, own_cell.labels)}
            own_color = {str(oid): col for oid, col in zip(own_cell.object_ids, own_cell.colors)}

        hints: list[Hint] = []
        if structured:
            for d in get(p, f_desc):
                label = str(get(d, f_dlab))
                raw_color = get(d, f_dcol) if f_dcol else None
                color = str(raw_color).strip().lower() if raw_color is not None else None
                # Use offset_center when available -- it gives the object
                # position directly and requires no global-id lookup.
                # Fall back to the id-based lookup only when offset is absent.
                c = None
                oid = None
                if f_doffset:
                    pass  # handled in the offset block below
                elif f_doid:
                    try:
                        oid = str(get(d, f_doid))
                        c = own_centroid.get(oid)
                        if c is None:
                            c = centroid_of.get(oid)
                            if f_dlab is None:
                                label = label_of.get(oid, label)
                            if color is None:
                                color = color_of.get(oid)
                    except AttributeError:
                        pass  # object_id unreachable on this stub class
                if f_doffset and c is None:
                    try:
                        raw_off = get(d, f_doffset)
                        offset = np.asarray(raw_off, dtype=np.float64).ravel()[:2]
                        # offset = object_xy - cell_centre_xy
                        # cell centre is in cell.center, but we don't have the
                        # cell object here. However:
                        #   direction(pose, object) = direction(pose, cell_centre + offset)
                        # The pose's own cell centre is stored in by_cell_id;
                        # use it when available, else fall back to global centroid lookup.
                        if own_cell is not None:
                            obj_xy = own_cell.center[:2] + offset
                            c = np.array([obj_xy[0], obj_xy[1], 0.0])
                        # else: c remains None, counted as unmatched below
                    except Exception:
                        pass
                geom_dir = (_direction_of_position(pos[:2], c[:2])
                            if c is not None else None)
                stored_dir = None
                if f_ddir:
                    candidate = str(get(d, f_ddir)).strip().lower()
                    if candidate in DIRECTIONS:
                        stored_dir = candidate
                if geom_dir is not None and stored_dir is not None:
                    dir_confusion[(stored_dir, geom_dir)] = (
                        dir_confusion.get((stored_dir, geom_dir), 0) + 1)
                    if geom_dir != stored_dir:
                        n_dir_disagree += 1
                # Geometry only. A stored direction is recorded for
                # comparison but never substituted: the two are produced by
                # different conventions, and silently mixing them would
                # corrupt the intersections that define V.
                if c is None:
                    n_unmatched += 1
                    continue
                if geom_dir is None:
                    n_no_dir += 1
                    continue
                hints.append(Hint(label=label, direction=geom_dir, color=color))
        else:
            if own_cell is None:
                n_missing_cells += 1
                continue
            try:
                pairs = np.asarray(get(p, f_matches), dtype=np.int64)
            except Exception:
                n_bad_matches += 1
                continue
            if pairs.size == 0:
                continue
            if pairs.ndim == 1:
                if pairs.size != 2:
                    n_bad_matches += 1
                    continue
                pairs = pairs.reshape(1, 2)
            if pairs.ndim != 2 or pairs.shape[1] < 2:
                n_bad_matches += 1
                continue
            # Some loaders include dustbin/invalid matches. Keep only genuine
            # object indices and one object per hint index, in hint order.
            usable = [(int(a), int(b)) for a, b in pairs[:, :2]
                      if 0 <= int(a) < len(own_cell.labels) and int(b) >= 0]
            seen_hints = set()
            for obj_idx, hint_idx in sorted(usable, key=lambda x: x[1]):
                if hint_idx in seen_hints:
                    continue
                seen_hints.add(hint_idx)
                c = own_cell.centroids[obj_idx]
                direction = _direction_of_position(pos[:2], c[:2])
                if direction is None:
                    n_no_dir += 1
                    continue
                color = (own_cell.colors[obj_idx]
                         if obj_idx < own_cell.colors.size else None)
                hints.append(Hint(label=str(own_cell.labels[obj_idx]),
                                  direction=direction, color=color))

        if hints:
            descriptions.append(hints)
            meta.append({"cell_id": pose_cid, "n_hints": len(hints)})

    diag = {
        "input_mode": mode,
        "n_poses": len(poses),
        "n_descriptions_usable": len(descriptions),
        "hints_dropped_unmatched_object": n_unmatched,
        "hints_dropped_degenerate_direction": n_no_dir,
        "geometry_vs_stored_direction_disagreements": n_dir_disagree,
        "geometry_vs_stored_direction_confusion":
            {f"{a}->{b}": n for (a, b), n in sorted(dir_confusion.items())},
        "poses_missing_resolved_cell": n_missing_cells,
        "poses_with_invalid_matches": n_bad_matches,
        "resolved_fields": {"pose_position": f_pos, "pose_descrs": f_desc,
                            "pose_cell_id": f_cell, "pose_matches": f_matches,
                            "descr_label": f_dlab, "descr_obj_id": f_doid,
                            "descr_color": f_dcol, "descr_dir": f_ddir},
    }
    if n_unmatched:
        log(f"    NOTE {n_unmatched} hints referenced objects absent from the "
            f"cell file; they were dropped and are reported, not silently ignored")
    if n_missing_cells:
        log(f"    NOTE {n_missing_cells} poses referenced a cell id absent from the loaded scene")
    if n_bad_matches:
        log(f"    NOTE {n_bad_matches} poses carried an unreadable match array")
    return descriptions, meta, diag


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", required=True, type=Path)
    p.add_argument("--poses", required=True, type=Path)
    p.add_argument("--scene", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--theta", type=float, default=90.0)
    p.add_argument("--use-color", action="store_true",
                   help="also require the hinted colour to match")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="reuse a parsed-scene cache across runs; "
                        "big win for the theta sweep and for data on "
                        "network storage. Omit to parse every time.")
    p.add_argument("--force", action="store_true")
    for role in ("pose-position", "pose-descrs", "pose-cell-id",
                 "pose-matches", "descr-label", "descr-obj-id", "descr-color",
                 "cell-objects", "cell-id", "cell-center", "cell-bbox",
                 "obj-label", "obj-xyz", "obj-id", "obj-color"):
        p.add_argument(f"--field-{role}", default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.out_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(run_dir / "run.log")

    overrides = {k.replace("field_", "").replace("-", "_"): v
                 for k, v in vars(args).items()
                 if k.startswith("field_") and v}
    cfg = {"scene": args.scene, "cells": str(args.cells),
           "poses": str(args.poses), "theta": args.theta,
           "use_color": args.use_color, "overrides": overrides,
           "cache_dir": str(args.cache_dir) if args.cache_dir else None,
           "code_version": "kpose_diagnostics/2.2.0"}

    log("=" * 72)
    log("MULTI-HINT AMBIGUITY (KITTI360Pose description family)")
    log("=" * 72)
    log(f"environment: {environment()}")

    with Stage(run_dir, "load", cfg, log, args.force) as st:
        try:
            scene = load_cells_cached(args.cells, args.scene, overrides,
                                      args.cache_dir, log)
            poses_raw, stubbed = load_pickle_entries(args.poses)
        except SchemaError as exc:
            log(f"SCHEMA ERROR\n{exc}")
            return 2
        log(f"    cells {scene.n_cells}, objects {scene.n_objects}")
        st.outputs = [save_json(st.dir / "poses_schema.json",
                                describe(poses_raw, max_depth=4))]

    log("")
    log("extracting hints (directions derived geometrically, not parsed)")
    try:
        descriptions, meta, diag = extract_descriptions(poses_raw, scene, log,
                                                        overrides)
    except SchemaError as exc:
        log(f"SCHEMA ERROR\n{exc}")
        log("A schema report was written to load/poses_schema.json -- send it "
            "and the correct --field-* names can be supplied in one line.")
        return 2

    log(f"    usable descriptions: {diag['n_descriptions_usable']:,} "
        f"of {diag['n_poses']:,} poses")
    n_hints = np.array([len(h) for h in descriptions])
    log(f"    hints per description: mean {n_hints.mean():.2f}, "
        f"range {n_hints.min()}-{n_hints.max()}")

    if args.use_color:
        total_obj = sum(c.colors.size for c in scene.cells)
        known_obj = sum(sum(x is not None for x in c.colors) for c in scene.cells)
        total_hints = sum(len(h) for h in descriptions)
        known_hints = sum(sum(x.color is not None for x in h) for h in descriptions)
        log(f"    color coverage: objects {known_obj}/{total_obj}, hints {known_hints}/{total_hints}")
        if known_obj != total_obj or known_hints != total_hints:
            log("COLOR SCHEMA ERROR: --use-color requested, but textual colors are not "
                "available for every object/hint. This version refuses to silently "
                "ignore color. Run the label-only B2 result, or make the original "
                "Text2Pos color-text method importable.")
            return 2

    log("")
    log("computing V per description")
    v, mh_diag = compute_multihint_v(scene, descriptions, args.theta,
                                     args.use_color, None, log=log)
    log(f"    signature grouping gave a "
        f"{mh_diag['speedup_vs_per_description']:.1f}x reduction in work")

    stats = summarise_v(v)
    ceiling = naive_ceiling(v)

    # The same question B1 asks, asked of this query family.
    gt_ids = [m.get("cell_id") for m in meta]
    gt_valid, gt_resolved = multihint_gt_validity(
        scene, descriptions, gt_ids, mh_diag.get("_sig_mask", {}), args.use_color)
    mh_diag.pop("_sig_mask", None)
    n_res = int(gt_resolved.sum())
    R_b2 = float(gt_valid[gt_resolved].mean()) if n_res else float("nan")
    recoverable = (float(np.where(gt_valid[gt_resolved] & (v[gt_resolved] > 0),
                                  1.0 / np.maximum(v[gt_resolved], 1), 0.0).mean())
                   if n_res else float("nan"))

    log("")
    log("-" * 72)
    log("RESULT")
    log("-" * 72)
    log(f"  descriptions           : {stats['n_queries']:,}")
    log(f"  median V               : {stats['median']:.1f}   "
        f"(single-relation family: 758)")
    log(f"  mean V (SD)            : {stats['mean']:.1f} ({stats['sd']:.1f})")
    log(f"  uniquely identifying   : {stats['n_V1']:,} "
        f"({stats['pct_V1']:.4f}%)   (single-relation: 0.0085%)")
    log(f"  V <= 10                : {stats['n_V_le_10']:,} "
        f"({stats['pct_V_le_10']:.2f}%)")
    log(f"  V = 0                  : {stats['n_V0']:,}")
    log(f"  implied ceiling E[1/V] : {ceiling:.5f}   (single-relation: 0.00341)")
    log("")
    log(f"  generating cell satisfies its own description: {R_b2:.4f} "
        f"(of {n_res:,} resolved)")
    log(f"  recoverable ceiling E[1{{gt in V}}/V]        : {recoverable:.5f}")
    log("    A description states a relation to the described position, but a")
    log("    cell is tested at its centre, so the two can differ. Where the")
    log("    generating cell fails its own description, E[1/V] overstates the")
    log("    ceiling -- the same effect B1 measures for the other query family.")
    if diag.get("geometry_vs_stored_direction_disagreements"):
        log("")
        log(f"  NOTE geometry and the stored direction field disagree on "
            f"{diag['geometry_vs_stored_direction_disagreements']:,} hints.")
        log(f"       confusion (stored -> geometric): "
            f"{diag['geometry_vs_stored_direction_confusion']}")
        log("       Only the geometric direction is used. A systematic pattern")
        log("       here indicates a convention difference, not noisy data.")

    by_n: dict[int, list[int]] = defaultdict(list)
    for k, vi in zip(n_hints, v):
        by_n[int(k)].append(int(vi))
    log("")
    log(f"  {'hints':>6}{'descriptions':>14}{'median V':>12}{'% unique':>11}")
    rows_by_hints = []
    for k in sorted(by_n):
        arr = np.asarray(by_n[k])
        row = {"n_hints": k, "n_descriptions": int(arr.size),
               "median_V": float(np.median(arr)),
               "pct_unique": float(100.0 * (arr == 1).mean())}
        rows_by_hints.append(row)
        log(f"  {k:>6}{arr.size:>14,}{row['median_V']:>12.1f}"
            f"{row['pct_unique']:>11.3f}")

    payload = {"config": cfg, "environment": environment(),
               "extraction": diag, "grouping": mh_diag,
               "v_summary": stats, "ceiling_E_inv_V": ceiling,
               "generating_cell_validity": {
                   "R": R_b2, "n_resolved": n_res,
                   "recoverable_ceiling": recoverable},
               "by_hint_count": rows_by_hints,
               "comparison_single_relation": {
                   "median_V": 758, "pct_unique": 0.0085, "ceiling": 0.00341}}
    save_json(run_dir / "multihint_results.json", payload)
    np.savez_compressed(run_dir / "per_description.npz",
                        v=v, n_hints=n_hints)

    try:
        import pandas as pd
        pd.DataFrame(rows_by_hints).to_csv(run_dir / "by_hint_count.csv",
                                           index=False)
    except ImportError:
        pass

    log("")
    log(f"wrote {(run_dir / 'multihint_results.json').resolve()}")
    log("DONE.")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
