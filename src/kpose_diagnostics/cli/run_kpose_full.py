"""
run_kpose_full.py -- every KITTI360Pose measurement in one pass.

Four stages, sharing one expensive computation (a per-cell grid of
candidate positions with the direction of every object resolved at each).

  Stage 1  Verification. Recompute each hint's direction from the stored
           pose and object geometry and compare against the direction
           string the generator wrote. Also sweep the proximity
           threshold delta. Nothing downstream is reported unless this
           agrees with the generator, because a predicate that does not
           reproduce the generator measures its own disagreement rather
           than the benchmark's ambiguity.

  Stage 2  Cell-level ambiguity. V = the number of cells containing at
           least one position consistent with every hint. This is the
           quantity that bounds coarse retrieval.

  Stage 3  Generating-cell validity. Whether the cell a description was
           generated from is among its valid cells. Where it is not,
           exact retrieval is unattainable for that description by any
           method, and E[1/V] overstates the bound.

  Stage 4  Metric ceiling. The published evaluation for this task is
           recall@k m: a prediction counts as correct within k metres of
           the true position. Stage 2 already yields, for every
           description, the set of positions consistent with it. The
           fraction of that set lying within k metres of the true pose is
           the recall@k an oracle would achieve -- one that identifies
           the described region exactly but cannot order within it. It is
           a property of the descriptions and the map, fixed before any
           model is trained, and to our knowledge has not been reported.

Resolution is a stated parameter, not a hidden one. --grid controls the
spacing of candidate positions inside each cell; the area and recall
figures are reported together with it, and --grid-sweep re-runs Stage 4
at several resolutions so the reader can see the numbers converge.

Example
-------
    python -m kpose_diagnostics.cli.run_kpose_full \\
        --cells .../cells/2013_05_28_drive_0000_sync.pkl \\
        --poses .../poses/2013_05_28_drive_0000_sync.pkl \\
        --scene 2013_05_28_drive_0000_sync \\
        --out-dir results/kpose_full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..ambiguity import clustered_bootstrap, naive_ceiling, summarise_v
from ..kpose_predicate import (DEFAULT_DELTA, OUTCOMES, calibrate_delta,
                               direction_code_from_offsets, normalize_direction_label,
                               verify_against_stored)
from ..loading import (SchemaError, describe, get, load_cells, load_pickle,
                       load_pickle_entries)
from ..runner import (Logger, code_tree_hash, environment, fingerprint_path,
                      save_json)

#: Distance thresholds used by the published evaluation for this task.
RECALL_THRESHOLDS = (5.0, 10.0, 15.0)


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


def _nearest_offsets(points_norm: np.ndarray, object_points_xy: np.ndarray,
                     chunk_points: int = 4096) -> np.ndarray:
    """Exact ``position - closest_object_point`` offsets with bounded memory."""
    q = np.asarray(points_norm, dtype=np.float64)[:, :2]
    pts = np.asarray(object_points_xy, dtype=np.float64)[:, :2]
    if pts.size == 0:
        raise SchemaError("KPOSE object has no points; closest-point predicate undefined")
    best_d2 = np.full(q.shape[0], np.inf, dtype=np.float64)
    best = np.zeros((q.shape[0], 2), dtype=np.float64)
    for lo in range(0, pts.shape[0], chunk_points):
        block = pts[lo:lo + chunk_points]
        d2 = ((q[:, None, :] - block[None, :, :]) ** 2).sum(axis=2)
        jj = np.argmin(d2, axis=1)
        vv = d2[np.arange(q.shape[0]), jj]
        upd = vv < best_d2
        if np.any(upd):
            best_d2[upd] = vv[upd]
            best[upd] = block[jj[upd]]
    return q - best


def _direction_codes(points_norm, points_xy, delta):
    """Exact Text2Pos direction codes for grid positions and every object."""
    n_obj = len(points_xy)
    code = np.empty((len(points_norm), n_obj), dtype=np.int8)
    for j, pts in enumerate(points_xy):
        code[:, j] = direction_code_from_offsets(
            _nearest_offsets(points_norm, pts), delta=delta)
    return code


def build_index(scene, centres, grid, cell_size, delta, use_color, log):
    """Build atomic KPOSE hint bitmaps using generator-faithful coordinates.

    Object points in official KITTI360Pose cells are normalised to [0,1].
    Candidate positions are therefore evaluated in the same normalised frame.
    A separate world-coordinate grid is retained for metric recall/area outputs.
    """
    n_cells = scene.n_cells
    per_cell = grid * grid
    n_slots = n_cells * per_cell
    atomic: dict = {}
    coords = np.zeros((n_slots, 2), dtype=np.float32)

    axis = np.linspace(0.0, 1.0, grid, dtype=np.float64)
    points_norm = np.stack(np.meshgrid(axis, axis, indexing="ij"),
                           axis=-1).reshape(-1, 2)

    for ci, cell in enumerate(scene.cells):
        if cell.bbox_w is None:
            # Synthetic/legacy fallback: use the declared cell size around the
            # world center, but object geometry still must be local-normalised.
            half = float(cell.cell_size or cell_size) / 2.0
            lo = np.asarray(cell.center[:2], float) - half
            scale = float(cell.cell_size or cell_size)
        else:
            bb = np.asarray(cell.bbox_w, float).ravel()
            lo = bb[:2]
            scale = float(cell.cell_size)
        world_grid = lo[None, :] + points_norm * scale
        base = ci * per_cell
        coords[base:base + per_cell] = world_grid.astype(np.float32)
        if cell.labels.size == 0:
            continue
        if len(cell.points_xy) != cell.labels.size:
            raise SchemaError(
                f"cell {cell.cell_id}: retained points for {len(cell.points_xy)} "
                f"objects but {cell.labels.size} labels; rerun with full points")
        code = _direction_codes(points_norm, cell.points_xy, delta)
        for j in range(cell.labels.size):
            lab = str(cell.labels[j])
            col = (str(cell.colors[j]).strip().lower()
                   if use_color and j < cell.colors.size
                   and cell.colors[j] is not None else None)
            cj = code[:, j]
            for c_id, name in enumerate(OUTCOMES):
                sel = np.flatnonzero(cj == c_id)
                if sel.size == 0:
                    continue
                key = (lab, col, name)
                arr = atomic.get(key)
                if arr is None:
                    arr = np.zeros(n_slots, dtype=bool)
                    atomic[key] = arr
                arr[base + sel] = True
        if log and (ci + 1) % 250 == 0:
            log(f"        indexed {ci + 1}/{n_cells} cells")

    packed = {k: np.packbits(v) for k, v in atomic.items()}
    return packed, coords, n_slots


def _sig(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _atomic_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def _load_pickle_checkpoint(path: Path, signature: str):
    meta = path.with_suffix(path.suffix + ".json")
    if not (path.is_file() and meta.is_file()):
        return None
    try:
        m = json.loads(meta.read_text(encoding="utf-8"))
        if m.get("signature") != signature:
            return None
        with path.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def _save_pickle_checkpoint(path: Path, signature: str, obj) -> None:
    _atomic_pickle(path, obj)
    meta = path.with_suffix(path.suffix + ".json")
    tmp = meta.with_suffix(meta.suffix + ".tmp")
    tmp.write_text(json.dumps({"signature": signature,
                               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                              indent=2), encoding="utf-8")
    tmp.replace(meta)


def _load_scene_resumable(cells: Path, scene_name: str, keep_points: int,
                          cache_dir: Path, cells_fp: dict, log):
    """Parse the expensive scene once, safe for two Colab runtimes."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _sig({"cells_sha256": cells_fp["sha256"], "scene": scene_name,
                "keep_points": keep_points, "format": 2})[:20]
    cp = cache_dir / f"scene_{scene_name}_{key}.pkl"
    if cp.is_file():
        try:
            with cp.open("rb") as fh:
                scene = pickle.load(fh)
            log(f"scene cache hit: {cp}")
            return scene
        except Exception:
            cp.unlink(missing_ok=True)
    lock = cp.with_suffix(".lock")
    owner = False
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        owner = True
    except FileExistsError:
        pass
    if not owner:
        log(f"scene cache is being built by another runtime; waiting for {cp.name}")
        for _ in range(600):  # up to 20 minutes
            if cp.is_file():
                try:
                    with cp.open("rb") as fh:
                        scene = pickle.load(fh)
                    log(f"scene cache became available: {cp}")
                    return scene
                except Exception:
                    pass
            time.sleep(2)
        log("scene-cache wait timed out; parsing independently")
    scene = load_cells(cells, scene_name, None, keep_points=keep_points)
    if owner:
        try:
            _atomic_pickle(cp, scene)
            log(f"scene cached: {cp}")
        finally:
            lock.unlink(missing_ok=True)
    return scene


def _build_index_resumable(scene, centres, grid, cell_size, delta, use_color,
                           cache_dir: Path, base_signature: dict, log):
    """Build each grid index once and reuse it after restarts."""
    sig_payload = dict(base_signature)
    sig_payload.update({"grid": int(grid), "cell_size": float(cell_size),
                        "delta": float(delta), "use_color": bool(use_color),
                        "index_format": 2})
    signature = _sig(sig_payload)
    cp = cache_dir / "indexes" / f"index_g{grid}_{signature[:20]}.pkl"
    cached = _load_pickle_checkpoint(cp, signature)
    if cached is not None:
        log(f"    grid-index checkpoint hit: {cp}")
        return cached
    value = build_index(scene, centres, grid, cell_size, delta, use_color, log)
    _save_pickle_checkpoint(cp, signature, value)
    log(f"    grid-index checkpoint saved: {cp}")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", required=True, type=Path)
    p.add_argument("--poses", required=True, type=Path)
    p.add_argument("--scene", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--delta", type=float, default=DEFAULT_DELTA,
                   help="on-top threshold in normalized cell units (official: 0.05)")
    p.add_argument("--keep-points", type=int, default=-1,
                   help="points retained per object; -1 keeps all (required for exact KPOSE)")
    p.add_argument("--grid", type=int, default=16,
                   help="candidate positions per axis inside each cell")
    p.add_argument("--cell-size", type=float, default=30.0)
    p.add_argument("--grid-sweep", type=str, default="8,12,16,24",
                   help="resolutions for the Stage 4 convergence check")
    p.add_argument("--use-color", action="store_true")
    p.add_argument("--min-agreement", type=float, default=0.95)
    p.add_argument("--skip-gate", action="store_true")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260809)
    p.add_argument("--max-descriptions", type=int, default=0,
                   help="cap for a quick smoke run; 0 means all")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="shared durable cache for parsed scenes and grid indexes")
    p.add_argument("--checkpoint-every", type=int, default=250,
                   help="persist primary per-description progress every N poses")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "run.log")

    log("=" * 72)
    log("KITTI360Pose -- verification, ambiguity, validity, metric ceiling")
    log("=" * 72)
    log(f"scene       : {args.scene}")
    log(f"grid        : {args.grid}x{args.grid} per {args.cell_size} m cell")
    log(f"environment : {environment()}")
    log("")

    # ------------------------------------------------ provenance + load
    if args.keep_points != -1:
        log("ERROR: exact KPOSE requires --keep-points -1; point subsampling changes the closest-point predicate.")
        return 2
    fp_cache = out.parent / "_fingerprints"
    try:
        input_fingerprint = {
            "cells": fingerprint_path(args.cells, fp_cache),
            "poses": fingerprint_path(args.poses, fp_cache),
        }
    except FileNotFoundError as exc:
        log(f"INPUT ERROR: {exc}")
        return 2
    code_root = Path(__file__).resolve().parents[2]
    code_hash = code_tree_hash(code_root)
    cache_dir = args.cache_dir or (out.parent / "cache" / "kpose_shared")
    try:
        scene = _load_scene_resumable(args.cells, args.scene, args.keep_points,
                                      cache_dir, input_fingerprint["cells"], log)
        poses_raw = (load_pickle(args.poses)[0] if args.poses.is_file()
                     else load_pickle_entries(args.poses)[0])
    except (SchemaError, FileNotFoundError) as exc:
        log(f"SCHEMA/LOAD ERROR\n{exc}")
        return 2
    log(f"cells {scene.n_cells:,}, objects {scene.n_objects:,}")

    poses = _iter(poses_raw)
    if args.max_descriptions:
        poses = poses[:args.max_descriptions]
    probe = poses[0]
    f_pos = _resolve(probe, ("pose_w", "pose", "position", "xyz"))
    f_desc = _resolve(probe, ("descriptions", "descriptors", "hints"))
    f_cell = _resolve(probe, ("cell_id", "cell", "cell_idx"))
    d_probe = None
    for p_ in poses[:200]:
        ds = list(get(p_, f_desc)) if f_desc else []
        if ds:
            d_probe = ds[0]
            break
    if not (f_pos and f_desc and d_probe is not None):
        log("could not resolve pose/description fields")
        save_json(out / "poses_schema.json", describe(probe, 4))
        return 2
    f_dlab = _resolve(d_probe, ("object_label", "label", "cls"))
    f_ddir = _resolve(d_probe, ("direction", "direction_text", "relation"))
    f_dcol = _resolve(d_probe, ("object_color_text", "color_text", "color"))
    f_doff = _resolve(d_probe, ("offset_closest",))
    log(f"fields: label={f_dlab} direction={f_ddir} color={f_dcol} "
        f"offset={f_doff} cell={f_cell}")
    if not (f_dlab and f_ddir and f_doff):
        log("stored direction and offset_closest are required for Stage 1; aborting")
        save_json(out / "poses_schema.json", describe(d_probe, 4))
        return 2

    by_id = {str(c.cell_id): i for i, c in enumerate(scene.cells)}
    centres = np.asarray([c.center[:2] for c in scene.cells], float)

    # ------------------------------------------------ Stage 1
    log("")
    log("-" * 72)
    log("STAGE 1 -- does the implemented rule reproduce the generator?")
    log("-" * 72)
    records = []
    try:
        for p_ in poses:
            for d in get(p_, f_desc):
                stored = normalize_direction_label(get(d, f_ddir))
                off = np.asarray(get(d, f_doff), float).ravel()[:2]
                if off.size < 2 or not np.isfinite(off).all():
                    raise ValueError(f"invalid offset_closest {off!r}")
                records.append((off, stored))
    except (ValueError, TypeError) as exc:
        failure = {
            "return_code": 2,
            "failure_reason": f"invalid direction/offset record: {exc}",
            "gate_metrics": None,
            "confusion_matrix": {},
            "command": [sys.executable, "-m", "kpose_diagnostics.cli.run_kpose_full"] + list(sys.argv[1:]),
            "version": "2.2.0",
            "code_hash": code_hash,
            "input_fingerprint": input_fingerprint,
        }
        save_json(out / "verification_failure.json", failure)
        log(f"invalid verification record: {exc}")
        return 2

    log(f"    hint records: {len(records):,}")
    log("    agreement vs normalized proximity threshold delta:")
    sweep = calibrate_delta(records)
    for d, a in sweep:
        mark = "  <-- selected" if abs(d - args.delta) < 1e-12 else ""
        log(f"        delta = {d:>6g} normalized   agreement = {a:.4f}{mark}")
    report = verify_against_stored(records, args.delta)
    log("")
    for line in report.summary().splitlines():
        log("    " + line)
    gate_ok = report.agreement >= args.min_agreement
    log("")
    if gate_ok:
        log("    GATE PASSED -- downstream numbers describe the benchmark.")
    else:
        log("    GATE FAILED -- confusion matrix (stored -> recomputed):")
        for (a, b), n in sorted(report.confusion.items(),
                                key=lambda kv: -kv[1])[:12]:
            log(f"        {a:<8} -> {b:<8} {n:,}")
        failure = {
            "return_code": 3,
            "failure_reason": "KPOSE generator-verification agreement below min_agreement",
            "gate_metrics": {
                "agreement": report.agreement,
                "min_agreement": args.min_agreement,
                "n_compared": report.n_compared,
                "delta": args.delta,
                "delta_sweep": sweep,
                "per_direction": {k: list(v) for k, v in report.per_direction.items()},
            },
            "confusion_matrix": {f"{a}->{b}": n for (a, b), n in report.confusion.items()},
            "command": [sys.executable, "-m", "kpose_diagnostics.cli.run_kpose_full"] + list(sys.argv[1:]),
            "version": "2.2.0",
            "code_hash": code_hash,
            "input_fingerprint": input_fingerprint,
        }
        save_json(out / "verification_failure.json", failure)
        if not args.skip_gate:
            log("    Wrote verification_failure.json. Exit 3; downstream scientific results were NOT computed.")
            return 3
        log("    DEBUG ONLY: continuing after failed gate; outputs will be marked scientifically_valid=false.")

    # ------------------------------------------------ index
    log("")
    log(f"building position index at {args.grid}x{args.grid} per cell")
    base_signature = {"version": "2.2.0", "code_hash": code_hash,
                      "cells_sha256": input_fingerprint["cells"]["sha256"],
                      "scene": args.scene}
    packed, coords, n_slots = _build_index_resumable(
        scene, centres, args.grid, args.cell_size, args.delta,
        args.use_color, cache_dir, base_signature, log)
    per_cell = args.grid * args.grid
    spacing = args.cell_size / max(args.grid - 1, 1)
    log(f"    {len(packed):,} atomic keys, {n_slots:,} slots, "
        f"spacing {spacing:.2f} m")

    zero = np.zeros(next(iter(packed.values())).shape, dtype=np.uint8)

    def consistent_slots(sig):
        acc = None
        for h in sig:
            arr = packed.get(h)
            if arr is None:
                return np.zeros(0, dtype=np.int64)
            acc = arr if acc is None else (acc & arr)
        if acc is None:
            return np.zeros(0, dtype=np.int64)
        return np.flatnonzero(np.unpackbits(acc)[:n_slots])

    # ------------------------------------------------ Stages 2-4
    log("")
    log("-" * 72)
    log("STAGES 2-4 -- ambiguity, validity, metric ceiling")
    log("-" * 72)

    cache: dict = {}
    progress_sig = _sig({**base_signature, "poses_sha256": input_fingerprint["poses"]["sha256"],
                         "grid": args.grid, "delta": args.delta, "use_color": args.use_color,
                         "max_descriptions": args.max_descriptions, "measurement_format": 2})
    progress_path = out / "checkpoints" / f"primary_{progress_sig[:20]}.pkl"
    state = _load_pickle_checkpoint(progress_path, progress_sig)
    if state is None:
        start_pose_idx = 0
        v_list, nh_list, gtv_list, cl_list = [], [], [], []
        area_list = []
        rec = {k: [] for k in RECALL_THRESHOLDS}
        signatures_seen = set()
    else:
        start_pose_idx = int(state["next_pose_idx"])
        v_list = state["v_list"]; nh_list = state["nh_list"]
        gtv_list = state["gtv_list"]; cl_list = state["cl_list"]
        area_list = state["area_list"]; rec = state["rec"]
        signatures_seen = state.get("signatures_seen", set())
        log(f"    primary measurement checkpoint hit: resume at pose {start_pose_idx}/{len(poses)}")
    n_done = len(v_list)

    for p_idx in range(start_pose_idx, len(poses)):
        p_ = poses[p_idx]
        pos = np.asarray(get(p_, f_pos), float).ravel()[:2]
        ci = by_id.get(str(get(p_, f_cell))) if f_cell else None
        hints = []
        for d in get(p_, f_desc):
            lab = str(get(d, f_dlab))
            dd = normalize_direction_label(get(d, f_ddir))
            col = (str(get(d, f_dcol)).strip().lower()
                   if args.use_color and f_dcol else None)
            hints.append((lab, col, dd))
        if hints and ci is not None:
            sig = tuple(sorted(set(hints)))
            signatures_seen.add(sig)
            if sig not in cache:
                slots = consistent_slots(sig)
                cells_hit = np.unique(slots // per_cell) if slots.size else np.zeros(0, dtype=np.int64)
                cache[sig] = (slots, cells_hit)
            slots, cells_hit = cache[sig]
            v_list.append(int(cells_hit.size))
            nh_list.append(len(hints))
            gtv_list.append(bool(np.any(cells_hit == ci)))
            cl_list.append(ci)
            if slots.size:
                pts = coords[slots]
                uniq = np.unique(np.round(pts / spacing).astype(np.int64), axis=0)
                area_list.append(float(uniq.shape[0] * spacing * spacing))
                dist = np.sqrt(((coords[slots] - pos[None, :]) ** 2).sum(1))
                for k in RECALL_THRESHOLDS:
                    rec[k].append(float((dist <= k).mean()))
            else:
                area_list.append(0.0)
                for k in RECALL_THRESHOLDS:
                    rec[k].append(0.0)
            n_done += 1
            if n_done % 1000 == 0:
                log(f"        {n_done:,} descriptions processed")

        if args.checkpoint_every > 0 and ((p_idx + 1) % args.checkpoint_every == 0):
            _save_pickle_checkpoint(progress_path, progress_sig, {
                "next_pose_idx": p_idx + 1, "v_list": v_list, "nh_list": nh_list,
                "gtv_list": gtv_list, "cl_list": cl_list, "area_list": area_list,
                "rec": rec, "signatures_seen": signatures_seen})

    _save_pickle_checkpoint(progress_path, progress_sig, {
        "next_pose_idx": len(poses), "v_list": v_list, "nh_list": nh_list,
        "gtv_list": gtv_list, "cl_list": cl_list, "area_list": area_list,
        "rec": rec, "signatures_seen": signatures_seen})

    v = np.asarray(v_list, dtype=np.int32)
    nh = np.asarray(nh_list, dtype=np.int32)
    gtv = np.asarray(gtv_list, dtype=bool)
    cl = np.asarray(cl_list, dtype=np.int64)
    area = np.asarray(area_list, float)
    stats = summarise_v(v)

    R, R_lo, R_hi = clustered_bootstrap(gtv.astype(float), cl,
                                        n_boot=args.n_boot, seed=args.seed)
    naive = naive_ceiling(v)
    recoverable = float(np.where(gtv & (v > 0), 1.0 / np.maximum(v, 1), 0.0).mean())

    log("")
    log("  STAGE 2 -- cell-level ambiguity")
    log(f"    descriptions           : {stats['n_queries']:,}")
    log(f"    distinct signatures    : {len(signatures_seen):,}")
    log(f"    hints per description  : mean {nh.mean():.2f}, "
        f"range {nh.min()}-{nh.max()}")
    log(f"    median V               : {stats['median']:.1f}"
        "     (template family: 758)")
    log(f"    mean V (SD)            : {stats['mean']:.1f} ({stats['sd']:.1f})")
    log(f"    uniquely identifying   : {stats['n_V1']:,} "
        f"({stats['pct_V1']:.3f}%)   (template family: 0.0085%)")
    log(f"    V <= 10                : {stats['n_V_le_10']:,} "
        f"({stats['pct_V_le_10']:.2f}%)")
    log(f"    V = 0                  : {stats['n_V0']:,}")

    log("")
    log("  STAGE 3 -- generating-cell validity")
    log(f"    R                      : {R:.4f}  [{R_lo:.4f}, {R_hi:.4f}]")
    log(f"    naive ceiling E[1/V]   : {naive:.5f}")
    log(f"    recoverable ceiling    : {recoverable:.5f}")
    if recoverable > 0:
        log(f"    overstatement factor   : {naive / recoverable:.2f}x")

    log("")
    log("  STAGE 4 -- metric ceiling for recall@k")
    log(f"    consistent region area : median {np.median(area):,.0f} m^2, "
        f"mean {area.mean():,.0f} m^2")
    ceil_rows = []
    for k in RECALL_THRESHOLDS:
        arr = np.asarray(rec[k])
        pt, lo, hi = clustered_bootstrap(arr, cl, n_boot=args.n_boot,
                                         seed=args.seed)
        ceil_rows.append({"threshold_m": k, "oracle_recall": pt,
                          "ci95": [lo, hi]})
        log(f"    oracle recall@{k:>4.0f} m    : {pt:.4f}  [{lo:.4f}, {hi:.4f}]")
    log("")
    log("    This is the recall a method would achieve if it identified the")
    log("    described region exactly and then picked uniformly within it.")
    log("    It is fixed by the descriptions and the map before training.")

    # -------- resolution convergence for Stage 4
    conv = []
    if args.grid_sweep:
        log("")
        log("  Stage 4 convergence in grid resolution")
        for g in [int(x) for x in args.grid_sweep.split(",") if x.strip()]:
            if g == args.grid:
                conv.append({"grid": g,
                             "oracle_recall": {str(k): ceil_rows[i]["oracle_recall"]
                                               for i, k in enumerate(RECALL_THRESHOLDS)}})
                log(f"    grid {g:>3}: " + "  ".join(
                    f"@{k:.0f}m={ceil_rows[i]['oracle_recall']:.4f}"
                    for i, k in enumerate(RECALL_THRESHOLDS)))
                continue
            conv_sig = _sig({**base_signature, "poses_sha256": input_fingerprint["poses"]["sha256"],
                             "grid": g, "delta": args.delta, "use_color": args.use_color,
                             "convergence_format": 2})
            conv_path = out / "checkpoints" / f"convergence_g{g}_{conv_sig[:20]}.pkl"
            cached_row = _load_pickle_checkpoint(conv_path, conv_sig)
            if cached_row is not None:
                row = cached_row
                log(f"    convergence checkpoint hit for grid {g}")
            else:
                pk, co, ns = _build_index_resumable(
                    scene, centres, g, args.cell_size, args.delta,
                    args.use_color, cache_dir, base_signature, log)
                loc = {}
                vals = {k: [] for k in RECALL_THRESHOLDS}
                for p_ in poses:
                    ci = by_id.get(str(get(p_, f_cell))) if f_cell else None
                    if ci is None:
                        continue
                    pos = np.asarray(get(p_, f_pos), float).ravel()[:2]
                    hh = []
                    for d in get(p_, f_desc):
                        hh.append((str(get(d, f_dlab)),
                                   (str(get(d, f_dcol)).strip().lower()
                                    if args.use_color and f_dcol else None),
                                   normalize_direction_label(get(d, f_ddir))))
                    if not hh:
                        continue
                    sg = tuple(sorted(set(hh)))
                    if sg not in loc:
                        acc = None
                        miss = False
                        for h in sg:
                            a2 = pk.get(h)
                            if a2 is None:
                                miss = True
                                break
                            acc = a2 if acc is None else (acc & a2)
                        loc[sg] = (np.zeros(0, dtype=np.int64) if miss or acc is None
                                   else np.flatnonzero(np.unpackbits(acc)[:ns]))
                    sl = loc[sg]
                    if sl.size:
                        dd2 = np.sqrt(((co[sl] - pos[None, :]) ** 2).sum(1))
                        for k in RECALL_THRESHOLDS:
                            vals[k].append(float((dd2 <= k).mean()))
                    else:
                        for k in RECALL_THRESHOLDS:
                            vals[k].append(0.0)
                row = {"grid": g, "oracle_recall": {
                    str(k): float(np.mean(vals[k])) for k in RECALL_THRESHOLDS}}
                _save_pickle_checkpoint(conv_path, conv_sig, row)
            conv.append(row)
            log(f"    grid {g:>3}: " + "  ".join(
                f"@{k:.0f}m={row['oracle_recall'][str(k)]:.4f}"
                for k in RECALL_THRESHOLDS))

    by_n = defaultdict(list)
    for k, vi in zip(nh, v):
        by_n[int(k)].append(int(vi))
    rows = [{"n_hints": k, "n_descriptions": len(by_n[k]),
             "median_V": float(np.median(by_n[k])),
             "pct_unique": float(100.0 * np.mean(np.asarray(by_n[k]) == 1))}
            for k in sorted(by_n)]

    save_json(out / "kpose_full_results.json", {
        "config": {"scene": args.scene, "delta": args.delta,
                   "grid": args.grid, "cell_size": args.cell_size,
                   "spacing_m": spacing, "use_color": args.use_color,
                   "keep_points": args.keep_points,
                   "code_version": "kpose_diagnostics/2.2.0",
                   "code_hash": code_hash,
                   "input_fingerprint": input_fingerprint,
                   "scientifically_valid": bool(gate_ok)},
        "environment": environment(),
        "stage1_verification": {
            "agreement": report.agreement, "n_compared": report.n_compared,
            "passed": gate_ok, "delta_sweep": sweep,
            "per_direction": {k: list(val)
                              for k, val in report.per_direction.items()},
            "confusion": {f"{a}->{b}": n
                          for (a, b), n in report.confusion.items()}},
        "stage2_ambiguity": {"v_summary": stats, "n_signatures": len(signatures_seen),
                             "by_hint_count": rows},
        "stage3_validity": {"R": R, "R_ci95": [R_lo, R_hi],
                            "naive_ceiling": naive,
                            "recoverable_ceiling": recoverable,
                            "overstatement": (naive / recoverable
                                              if recoverable > 0 else None)},
        "stage4_metric_ceiling": {
            "region_area_m2": {"median": float(np.median(area)),
                               "mean": float(area.mean()),
                               "p25": float(np.percentile(area, 25)),
                               "p75": float(np.percentile(area, 75))},
            "oracle_recall": ceil_rows,
            "convergence": conv},
    })
    np.savez_compressed(out / "per_description.npz",
                        v=v, n_hints=nh, gt_valid=gtv, cell_index=cl,
                        area=area,
                        **{f"recall_{int(k)}": np.asarray(rec[k])
                           for k in RECALL_THRESHOLDS})
    log("")
    log(f"wrote {(out / 'kpose_full_results.json').resolve()}")
    log("DONE.")
    log.close()
    return 0 if gate_ok else 4


if __name__ == "__main__":
    sys.exit(main())
