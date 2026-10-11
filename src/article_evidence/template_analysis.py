"""Full primary template-query analysis required by manuscript_v10.

This module intentionally lives OUTSIDE kpose_diagnostics.  The validated
v2.2.0 core is imported read-only so its content hash remains identical to the
hash used by the already-completed scientific jobs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import ALPHAS, PRIMARY_SCENE
from .common import atomic_json, atomic_npz, utc_now, write_csv

from kpose_diagnostics.ambiguity import (STRATA, compute_relation_queries,
                                         naive_ceiling, recoverable_ceiling,
                                         summarise_v)
from kpose_diagnostics.geometry import DIRECTIONS, s_dir
from kpose_diagnostics.loading import load_cells_cached
from kpose_diagnostics.runner import fingerprint_path


BOOT_SEED = 20260808
N_BOOT = 2000
N_PERM = 10000
RANK_SEED = 20260808
RANK_N = 500


def _alpha_key(a: float) -> str:
    return f"{a:g}"


def _group_indices(labels: np.ndarray) -> dict[str, np.ndarray]:
    by: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels.tolist()):
        by[str(lab)].append(i)
    return {k: np.asarray(v, dtype=np.int32) for k, v in by.items()}


def build_population(scene):
    """Build manuscript query order and a canonical template table."""
    per_cell_pairs = []
    templates = set()
    for cell in scene.cells:
        labs = sorted(set(str(x) for x in cell.labels.tolist()))
        pairs = [(a, b) for a in labs for b in labs if a != b]
        per_cell_pairs.append(pairs)
        for s, r in pairs:
            for d in DIRECTIONS:
                templates.add((s, d, r))
    template_list = sorted(templates)
    tindex = {t: i for i, t in enumerate(template_list)}
    q_cell, q_tpl = [], []
    for ci, pairs in enumerate(per_cell_pairs):
        for s, r in pairs:
            for d in DIRECTIONS:
                q_cell.append(ci)
                q_tpl.append(tindex[(s, d, r)])
    return template_list, tindex, np.asarray(q_cell, np.int32), np.asarray(q_tpl, np.int32)


def _matrix_meta(scene, template_list, cells_fp, core_hash, executor_hash):
    return {"schema": 1, "scene": scene.scene, "n_cells": scene.n_cells,
            "n_objects": scene.n_objects, "n_templates": len(template_list),
            "cells_fingerprint": cells_fp, "core_hash": core_hash,
            "executor_hash": executor_hash}


def build_score_matrices(scene, template_list, tindex, out: Path, meta: dict, log=print):
    """Build exact object/relation scores with atomic 50-cell checkpoints.

    Each relation chunk is an independent compressed archive. An interruption can
    therefore lose at most the current chunk and never leaves a completed chunk
    partially written.
    """
    stage = out / "score_matrices"
    chunks = stage / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    meta_path = stage / "META.json"
    old_ok = False
    if meta_path.is_file():
        try:
            old_ok = json.loads(meta_path.read_text()) == meta
        except Exception:
            old_ok = False
    if not old_ok:
        import shutil
        shutil.rmtree(chunks, ignore_errors=True)
        chunks.mkdir(parents=True, exist_ok=True)
        atomic_json(meta_path, meta)

    # Object score is cheap and deterministic: two distinct query labels, each
    # contributing one binary presence indicator.
    obj = np.empty((len(template_list), scene.n_cells), dtype=np.float32)
    subj = [t[0] for t in template_list]
    ref = [t[2] for t in template_list]
    for ci, cell in enumerate(scene.cells):
        present = set(str(x) for x in cell.labels.tolist())
        obj[:, ci] = np.asarray([(int(a in present) + int(b in present)) / 2.0
                                 for a, b in zip(subj, ref)], dtype=np.float32)

    chunk_size = 50
    rel_parts = []
    for start in range(0, scene.n_cells, chunk_size):
        stop = min(scene.n_cells, start + chunk_size)
        cp = chunks / f"rel_{start:05d}_{stop:05d}.npz"
        part = None
        if cp.is_file():
            try:
                z = np.load(cp, allow_pickle=False)
                candidate = z["s_rel"]
                if candidate.shape == (len(template_list), stop-start):
                    part = candidate
            except Exception:
                part = None
        if part is None:
            part = np.zeros((len(template_list), stop-start), dtype=np.float64)
            for ci in range(start, stop):
                cell = scene.cells[ci]
                by = _group_indices(cell.labels)
                labs = sorted(by)
                for a in labs:
                    cs = cell.centroids[by[a], :2]
                    for b in labs:
                        if b == a:
                            continue
                        ct = cell.centroids[by[b], :2]
                        delta = ct[None, :, :] - cs[:, None, :]
                        vals = np.asarray(s_dir(delta), dtype=np.float64).mean(axis=(0, 1))
                        for di, d in enumerate(DIRECTIONS):
                            ti = tindex.get((a, d, b))
                            if ti is not None:
                                part[ti, ci-start] = float(vals[di])
            atomic_npz(cp, s_rel=part)
            log(f"score matrix chunk complete: cells {start}:{stop}")
        else:
            log(f"score matrix chunk reused: cells {start}:{stop}")
        rel_parts.append(np.asarray(part, dtype=np.float64))
    rel = np.concatenate(rel_parts, axis=1) if rel_parts else np.zeros((len(template_list), 0))
    return obj, rel

def _cluster_bootstrap_all(q_cell, t1, t2, n_cells, n_boot=N_BOOT, seed=BOOT_SEED):
    """One paired source-cell bootstrap shared by every alpha."""
    n_alpha = t1.shape[0]
    qcount = np.bincount(q_cell, minlength=n_cells).astype(np.int64)
    succ1 = np.empty((n_cells, n_alpha), dtype=np.int64)
    succ2 = np.empty((n_cells, n_alpha), dtype=np.int64)
    for a in range(n_alpha):
        succ1[:, a] = np.bincount(q_cell, weights=t1[a].astype(np.int8), minlength=n_cells).astype(np.int64)
        succ2[:, a] = np.bincount(q_cell, weights=t2[a].astype(np.int8), minlength=n_cells).astype(np.int64)
    rng = np.random.default_rng(seed)
    b1 = np.empty((n_boot, n_alpha), dtype=np.float64)
    b2 = np.empty((n_boot, n_alpha), dtype=np.float64)
    # Match kpose_diagnostics.clustered_bootstrap exactly: only source cells
    # that actually generated at least one query are bootstrap clusters.
    active = np.flatnonzero(qcount > 0).astype(np.int32)
    n_groups = int(active.size)
    chunk = 100
    for start in range(0, n_boot, chunk):
        stop = min(n_boot, start + chunk)
        local = rng.integers(0, n_groups, size=(stop-start, n_groups), dtype=np.int32)
        pick = active[local]
        den = qcount[pick].sum(axis=1).astype(np.float64)
        b1[start:stop] = succ1[pick].sum(axis=1) / den[:, None]
        b2[start:stop] = succ2[pick].sum(axis=1) / den[:, None]
    return b1, b2, succ1, succ2, qcount


def _perm_pvalue(diff_success_by_cell: np.ndarray, n_queries: int, observed: float,
                 seed: int, n_perm: int=N_PERM) -> float:
    rng = np.random.default_rng(seed)
    extreme = 0
    chunk = 500
    d = np.asarray(diff_success_by_cell, dtype=np.float64)
    for start in range(0, n_perm, chunk):
        m = min(chunk, n_perm-start)
        signs = rng.integers(0, 2, size=(m, d.size), dtype=np.int8) * 2 - 1
        stats = (signs @ d) / float(n_queries)
        extreme += int(np.count_nonzero(np.abs(stats) >= abs(observed) - 1e-15))
    return float((extreme + 1) / (n_perm + 1))


def _strata_rows(v, gt_valid, q_cell, t1, t2, alphas):
    rows = []
    for name, lo, hi in STRATA:
        m = (v >= lo) & (v <= hi)
        n = int(m.sum())
        if not n:
            continue
        base = {"stratum": name, "lo": lo, "hi": hi, "queries": n,
                "mean_V": float(v[m].mean()), "R": float(gt_valid[m].mean()),
                "naive_ceiling": naive_ceiling(v[m]),
                "recoverable_ceiling": recoverable_ceiling(v[m], gt_valid[m])}
        for ai, a in enumerate(alphas):
            row = dict(base)
            row.update({"alpha": float(a), "T1": float(t1[ai, m].mean()),
                        "T2": float(t2[ai, m].mean())})
            rows.append(row)
    return rows


def _rank_sample(scores_obj, scores_rel, template_list, q_tpl, q_cell, alpha=0.5,
                 seed=RANK_SEED, n=RANK_N):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(q_cell.size, size=min(n, q_cell.size), replace=False))
    cells = np.arange(scores_obj.shape[1])
    rows = []
    for qi in idx:
        ti = int(q_tpl[qi]); gt = int(q_cell[qi])
        row_scores = alpha * scores_obj[ti].astype(np.float64) + (1-alpha) * scores_rel[ti]
        g = float(row_scores[gt])
        rank = 1 + int(np.count_nonzero(row_scores > g)) + int(np.count_nonzero((row_scores == g) & (cells < gt)))
        top = int(np.argmax(row_scores))
        t = template_list[ti]
        rows.append({"query_index": int(qi), "template_index": ti,
                     "subject": t[0], "direction": t[1], "reference": t[2],
                     "gt_cell_index": gt, "predicted_cell_index": top,
                     "gt_score": g, "rank": rank,
                     "gt_score_tie_size": int(np.count_nonzero(row_scores == g))})
    ranks = np.asarray([r["rank"] for r in rows], dtype=np.int32)
    summary = {"alpha": alpha, "sample_n": len(rows), "seed": seed,
               "sample_query_indices_saved": True,
               "mean_rank": float(ranks.mean()), "median_rank": float(np.median(ranks)),
               "min_rank": int(ranks.min()), "max_rank": int(ranks.max()),
               "rank1": int(np.sum(ranks == 1)), "rank_le5": int(np.sum(ranks <= 5)),
               "rank_le10": int(np.sum(ranks <= 10)), "rank_le50": int(np.sum(ranks <= 50)),
               "rank_le100": int(np.sum(ranks <= 100))}
    return rows, summary


def run(args) -> int:
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"
    lf = log_path.open("a", encoding="utf-8")
    t0 = time.time()
    def log(s=""):
        line = f"[{time.time()-t0:8.1f}s] {s}"
        print(line, flush=True); lf.write(line+"\n"); lf.flush()

    cells_fp = fingerprint_path(args.cells, args.fingerprint_cache)
    global_fp = json.loads(Path(args.global_fingerprint_json).read_text()) if args.global_fingerprint_json else None
    log(f"scene={args.scene}  cells={args.cells}")
    scene = load_cells_cached(args.cells, args.scene, cache_dir=args.cache_dir, log=log)
    template_list, tindex, q_cell, q_tpl = build_population(scene)
    log(f"population: cells={scene.n_cells} objects={scene.n_objects} queries={q_cell.size} templates={len(template_list)}")

    # Core cross-check is a scientific integrity gate, not an optional diagnostic.
    qcore = compute_relation_queries(scene, 90.0, progress_every=0, log=log)
    if not np.array_equal(q_cell, qcore.query_cell_index):
        raise RuntimeError("query-order mismatch against v2.2.0 core")

    meta = _matrix_meta(scene, template_list, cells_fp, args.core_hash, args.executor_hash)
    obj, rel = build_score_matrices(scene, template_list, tindex, out, meta, log)
    valid = rel > 0.0
    tpl_v = valid.sum(axis=1).astype(np.int32)
    q_v = tpl_v[q_tpl]
    q_gt = valid[q_tpl, q_cell]
    if not np.array_equal(q_v, qcore.query_v):
        bad = int(np.count_nonzero(q_v != qcore.query_v))
        raise RuntimeError(f"V mismatch against v2.2.0 core on {bad} queries")
    if not np.array_equal(q_gt, qcore.query_gt_valid):
        bad = int(np.count_nonzero(q_gt != qcore.query_gt_valid))
        raise RuntimeError(f"generating-cell-validity mismatch against v2.2.0 core on {bad} queries")
    log("integrity gate: query order, V and generating-cell validity agree exactly with core")

    alphas = np.asarray(ALPHAS, dtype=np.float64)
    n_a, n_t, n_q = len(alphas), len(template_list), q_cell.size
    pred_tpl = np.empty((n_a, n_t), dtype=np.int32)
    ties_tpl = np.empty((n_a, n_t), dtype=np.int32)
    for ai, a in enumerate(alphas):
        score = a * obj.astype(np.float64) + (1.0-a) * rel
        mx = score.max(axis=1)
        pred_tpl[ai] = np.argmax(score, axis=1).astype(np.int32)  # stable-first by fixed cell order
        ties_tpl[ai] = np.count_nonzero(score == mx[:, None], axis=1).astype(np.int32)
        log(f"alpha={a:g}: template argmax/ties complete")

    predicted = pred_tpl[:, q_tpl]
    tie_size = ties_tpl[:, q_tpl]
    t1 = predicted == q_cell[None, :]
    t2 = valid[q_tpl[None, :], predicted]

    boot1, boot2, succ1, succ2, qcount = _cluster_bootstrap_all(q_cell, t1, t2, scene.n_cells,
                                                                 args.n_boot, args.seed)
    alpha_rows = []
    for ai, a in enumerate(alphas):
        alpha_rows.append({
            "alpha": float(a), "T1": float(t1[ai].mean()),
            "T1_ci_lo": float(np.quantile(boot1[:, ai], .025)),
            "T1_ci_hi": float(np.quantile(boot1[:, ai], .975)),
            "T2": float(t2[ai].mean()),
            "T2_ci_lo": float(np.quantile(boot2[:, ai], .025)),
            "T2_ci_hi": float(np.quantile(boot2[:, ai], .975)),
            "tie_fraction": float(np.mean(tie_size[ai] > 1)),
            "mean_tied_set_size": float(tie_size[ai].mean()),
            "median_tied_set_size": float(np.median(tie_size[ai])),
            "max_tied_set_size": int(tie_size[ai].max()),
        })
    write_csv(out / "alpha_sweep.csv", alpha_rows)

    # Every pair gets a paired source-cell-bootstrap comparison for T1 and T2.
    pair_rows = []
    for i in range(n_a):
        for j in range(i+1, n_a):
            for metric, arr, boot in (("T1", t1, boot1), ("T2", t2, boot2)):
                point = float(arr[i].mean() - arr[j].mean())
                bd = boot[:, i] - boot[:, j]
                pair_rows.append({"metric": metric, "alpha_a": float(alphas[i]),
                                  "alpha_b": float(alphas[j]), "difference_a_minus_b": point,
                                  "ci_lo": float(np.quantile(bd, .025)),
                                  "ci_hi": float(np.quantile(bd, .975)),
                                  "bootstrap_clusters": "source_cell", "n_boot": args.n_boot})
    write_csv(out / "paired_all_alpha_bootstrap.csv", pair_rows)

    # The manuscript explicitly reports four T1 permutation contrasts.
    requested_pairs = ((0.0, 1.0), (0.0, 0.5), (0.5, 1.0), (0.4, 1.0))
    perm_rows = []
    amap = {round(float(a), 8): i for i, a in enumerate(alphas)}
    for pi, (aa, ab) in enumerate(requested_pairs):
        i, j = amap[aa], amap[ab]
        point = float(t1[i].mean() - t1[j].mean())
        p = _perm_pvalue(succ1[:, i]-succ1[:, j], n_q, point,
                         seed=args.seed + 1000 + pi, n_perm=args.n_perm)
        bd = boot1[:, i]-boot1[:, j]
        perm_rows.append({"alpha_a": aa, "alpha_b": ab,
                          "difference_a_minus_b": point,
                          "ci_lo": float(np.quantile(bd,.025)),
                          "ci_hi": float(np.quantile(bd,.975)),
                          "permutation_p_two_sided": p, "n_permutations": args.n_perm,
                          "cluster_unit": "source_cell"})
    write_csv(out / "paired_manuscript_comparisons.csv", perm_rows)

    strata_rows = _strata_rows(q_v, q_gt, q_cell, t1, t2, alphas)
    write_csv(out / "ambiguity_stratified_accuracy.csv", strata_rows)

    hist_vals, hist_counts = np.unique(q_v, return_counts=True)
    write_csv(out / "V_exact_distribution.csv",
              [{"V": int(v), "queries": int(n), "fraction": float(n/n_q)}
               for v,n in zip(hist_vals, hist_counts)])
    write_csv(out / "template_V.csv",
              [{"template_index": i, "subject": t[0], "direction": t[1],
                "reference": t[2], "V": int(tpl_v[i])}
               for i,t in enumerate(template_list)])

    rank_rows, rank_summary = _rank_sample(obj, rel, template_list, q_tpl, q_cell,
                                           alpha=args.rank_alpha, seed=args.rank_seed,
                                           n=args.rank_n)
    write_csv(out / "rank500_alpha0.5.csv", rank_rows)

    ai04 = amap[0.4]
    p04 = predicted[ai04]
    labels_present = obj[q_tpl, p04] >= 1.0 - 1e-12
    correct = t2[ai04]
    outcomes = {
        "correct": int(correct.sum()),
        "missing_object": int((~labels_present).sum()),
        "wrong_direction": int((labels_present & ~correct).sum()),
    }
    if sum(outcomes.values()) != n_q:
        raise RuntimeError("outcome classification does not sum to query total")
    write_csv(out / "outcomes_alpha0.4.csv",
              [{"outcome": k, "count": v, "fraction": float(v/n_q)} for k,v in outcomes.items()])

    # Chance and ambiguity ceilings.
    a05 = amap[0.5]
    chance_rows = [
        {"method": "uniform_random_cell", "T1": 1.0/scene.n_cells, "kind": "analytic_chance"},
        {"method": "uniform_within_valid_set_naive", "T1": naive_ceiling(q_v), "kind": "analytic_ceiling"},
        {"method": "uniform_within_valid_set_recoverable", "T1": recoverable_ceiling(q_v, q_gt), "kind": "corrected_analytic_ceiling"},
        {"method": "observed_alpha_0.5", "T1": float(t1[a05].mean()), "kind": "observed"},
    ]
    write_csv(out / "chance_baselines.csv", chance_rows)

    # Per-cell data needed to independently reconstruct clustered inference.
    write_csv(out / "source_cell_counts.csv",
              [{"cell_index": i, "cell_id": scene.cells[i].cell_id,
                "query_count": int(qcount[i]), "gt_valid_count": int(np.sum(q_gt[q_cell == i]))}
               for i in range(scene.n_cells)])

    label_counts=scene.label_counts()
    total_objects=sum(label_counts.values())
    write_csv(out / "label_distribution.csv",
              [{"label":lab,"count":int(n),"fraction":float(n/total_objects)}
               for lab,n in label_counts.items()])

    atomic_npz(out / "per_query_all_alpha.npz",
               source_cell=q_cell, template_index=q_tpl, V=q_v,
               generating_cell_valid=q_gt, alphas=alphas,
               predicted_cell=predicted, tie_size=tie_size,
               T1=t1, T2=t2)
    atomic_npz(out / "per_template_scores.npz",
               alphas=alphas, s_obj=obj.astype(np.float32), s_rel=rel,
               valid_theta90=valid, template_V=tpl_v,
               predicted_cell_by_alpha=pred_tpl,
               tie_size_by_alpha=ties_tpl)
    atomic_npz(out / "cluster_bootstrap_samples.npz",
               alphas=alphas, T1_accuracy=boot1, T2_accuracy=boot2)

    summary = {
        "schema_version": 1, "generated_utc": utc_now(),
        "config": {"scene": args.scene, "alphas": alphas.tolist(),
                   "n_boot": args.n_boot, "bootstrap_seed": args.seed,
                   "n_permutations": args.n_perm, "rank_n": args.rank_n,
                   "rank_seed": args.rank_seed, "rank_alpha": args.rank_alpha,
                   "tie_rule": "stable-first fixed cell order"},
        "provenance": {"core_version": "kpose_diagnostics/2.2.0",
                       "core_hash": args.core_hash, "executor_hash": args.executor_hash,
                       "cells_fingerprint": cells_fp, "global_input_fingerprint": global_fp},
        "population": {"cells": scene.n_cells, "objects": scene.n_objects,
                       "queries": n_q, "templates_generated": len(template_list),
                       "templates_realised": int(np.count_nonzero(tpl_v > 0))},
        "integrity": {"core_query_order_exact": True, "core_V_exact": True,
                      "core_generating_validity_exact": True,
                      "arrays_consistent_with_aggregates": True},
        "V_summary": summarise_v(q_v),
        "R": {"value": float(q_gt.mean()),
              "naive_ceiling": naive_ceiling(q_v),
              "recoverable_ceiling": recoverable_ceiling(q_v, q_gt)},
        "alpha_sweep": alpha_rows, "rank_sample": rank_summary,
        "alpha_0.4_outcomes": outcomes,
        "outputs": ["alpha_sweep.csv", "paired_all_alpha_bootstrap.csv",
                    "paired_manuscript_comparisons.csv", "ambiguity_stratified_accuracy.csv",
                    "V_exact_distribution.csv", "template_V.csv", "rank500_alpha0.5.csv",
                    "outcomes_alpha0.4.csv", "chance_baselines.csv", "source_cell_counts.csv",
                    "per_query_all_alpha.npz", "per_template_scores.npz",
                    "cluster_bootstrap_samples.npz"]}
    atomic_json(out / "template_analysis_results.json", summary)
    atomic_json(out / "_JOB_DONE.json", {"job": "template_analysis_primary", "ok": True,
                                          "completed": True, "returncode": 0,
                                          "core_hash": args.core_hash,
                                          "executor_hash": args.executor_hash,
                                          "finished_utc": utc_now(),
                                          "result": str(out / "template_analysis_results.json")})
    log("DONE")
    lf.close()
    return 0


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", type=Path, required=True)
    p.add_argument("--scene", default=PRIMARY_SCENE)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--fingerprint-cache", type=Path, required=True)
    p.add_argument("--global-fingerprint-json", type=Path)
    p.add_argument("--core-hash", required=True)
    p.add_argument("--executor-hash", required=True)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--seed", type=int, default=BOOT_SEED)
    p.add_argument("--n-perm", type=int, default=N_PERM)
    p.add_argument("--rank-n", type=int, default=RANK_N)
    p.add_argument("--rank-seed", type=int, default=RANK_SEED)
    p.add_argument("--rank-alpha", type=float, default=.5)
    return p


def main(argv=None):
    return run(build_parser().parse_args(argv))

if __name__ == "__main__":
    raise SystemExit(main())
