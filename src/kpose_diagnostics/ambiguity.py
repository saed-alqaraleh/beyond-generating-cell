"""
ambiguity.py -- valid-set sizes, generating-cell validity, and ceilings.

Two computations, both exact over the full query population.

A. Object-relation queries (the manuscript's own query family).
   Section 5.2 generates, for every cell, every ordered pair of distinct
   co-occurring labels, and each of four directional predicates, one
   query "[subject] [direction] of [reference]" with that cell as ground
   truth. By Proposition 1 the valid set depends only on the template
   tau = (subject, direction, reference), so V is computed once per
   template.

   The key efficiency point: a single pass over cells yields, at the same
   time, (i) the template -> valid-cell-set map and (ii) for every
   generated query, whether its own generating cell is in that set. The
   second quantity is the blocker measurement and costs nothing extra.

B. Multi-hint descriptions (the KITTI360Pose family).
   A description is a set of hints, each relating one referenced object
   to the described position. A cell satisfies the description iff every
   hint is satisfiable inside it. Template invariance still holds -- the
   valid set depends only on the multiset of hints -- so descriptions are
   grouped by canonical signature and V computed once per signature.

Sampling unit. Queries generated from the same cell share that cell's
configuration and are not independent. Every interval reported here uses
a bootstrap that resamples *cells*, never queries.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .geometry import DIRECTIONS, satisfies
from .loading import CellRecord, SceneData


# --------------------------------------------------------------------------
# A. Object-relation queries
# --------------------------------------------------------------------------

def _cell_satisfied_templates(cell: CellRecord, theta_deg: float) -> set:
    """Templates (subj, dir, ref) satisfied by at least one instance pair.

    "Satisfied" matches the manuscript's per-relation score being > 0:
    the per-relation score averages over instance pairs and is positive
    iff at least one pair meets the directional predicate.
    """
    out: set = set()
    if cell.labels.size == 0:
        return out

    by_label: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(cell.labels):
        by_label[lab].append(i)

    labels = sorted(by_label)
    for subj in labels:
        si = np.asarray(by_label[subj])
        cs = cell.centroids[si]                       # (A, 3)
        for ref in labels:
            if ref == subj:
                continue
            ti = np.asarray(by_label[ref])
            ct = cell.centroids[ti]                   # (B, 3)
            # dc = c_t - c_s for every ordered instance pair
            delta = ct[None, :, :2] - cs[:, None, :2]  # (A, B, 2)
            ok = satisfies(delta, theta_deg)           # (A, B, 4)
            any_pair = ok.any(axis=(0, 1))             # (4,)
            for d, flag in enumerate(any_pair):
                if flag:
                    out.add((subj, DIRECTIONS[d], ref))
    return out


def _cell_cooccurring_pairs(cell: CellRecord) -> list[tuple[str, str]]:
    """Ordered pairs of distinct labels co-occurring in the cell."""
    labels = sorted(set(cell.labels.tolist()))
    return [(a, b) for a in labels for b in labels if a != b]


@dataclass
class RelationQueryResult:
    """Population-level results for the object-relation query family."""

    theta_deg: float
    n_cells: int
    n_queries: int
    n_templates_generated: int
    n_templates_realised: int
    #: template -> number of cells satisfying it
    template_v: dict
    #: per query, aligned arrays
    query_cell_index: np.ndarray      # int32, index of generating cell
    query_v: np.ndarray               # int32, V of the query's template
    query_gt_valid: np.ndarray        # bool, generating cell in valid set


def compute_relation_queries(scene: SceneData,
                             theta_deg: float = 90.0,
                             progress_every: int = 250,
                             log=print) -> RelationQueryResult:
    """Single pass: valid sets, query population, and generating-cell validity."""
    n_cells = scene.n_cells
    template_cells: dict[tuple, set] = defaultdict(set)
    per_cell_satisfied: list[set] = []
    per_cell_pairs: list[list[tuple[str, str]]] = []

    for ci, cell in enumerate(scene.cells):
        sat = _cell_satisfied_templates(cell, theta_deg)
        per_cell_satisfied.append(sat)
        per_cell_pairs.append(_cell_cooccurring_pairs(cell))
        for tpl in sat:
            template_cells[tpl].add(ci)
        if progress_every and (ci + 1) % progress_every == 0:
            log(f"    pass 1: {ci + 1}/{n_cells} cells")

    template_v = {t: len(cs) for t, cs in template_cells.items()}

    cell_idx, v_vals, gt_valid = [], [], []
    n_generated_templates = set()
    for ci, pairs in enumerate(per_cell_pairs):
        sat = per_cell_satisfied[ci]
        for subj, ref in pairs:
            for d in DIRECTIONS:
                tpl = (subj, d, ref)
                n_generated_templates.add(tpl)
                cell_idx.append(ci)
                v_vals.append(template_v.get(tpl, 0))
                gt_valid.append(tpl in sat)

    return RelationQueryResult(
        theta_deg=theta_deg,
        n_cells=n_cells,
        n_queries=len(cell_idx),
        n_templates_generated=len(n_generated_templates),
        n_templates_realised=len(template_v),
        template_v=template_v,
        query_cell_index=np.asarray(cell_idx, dtype=np.int32),
        query_v=np.asarray(v_vals, dtype=np.int32),
        query_gt_valid=np.asarray(gt_valid, dtype=bool),
    )


# --------------------------------------------------------------------------
# Ceilings
# --------------------------------------------------------------------------

def naive_ceiling(v: np.ndarray) -> float:
    """E[1/V], the manuscript's published ceiling.

    Queries with V = 0 are assigned 1/V = 1, matching Section 6.1.
    """
    v = np.asarray(v, dtype=np.float64)
    inv = np.where(v > 0, 1.0 / np.maximum(v, 1.0), 1.0)
    return float(inv.mean())


def recoverable_ceiling(v: np.ndarray, gt_valid: np.ndarray) -> float:
    """E[1{C_gt in V} / V].

    The exact-recovery accuracy attainable by a procedure that identifies
    the valid set but cannot order within it. Differs from
    :func:`naive_ceiling` exactly on queries whose generating cell does
    not satisfy the query generated from it, where no procedure can
    succeed.
    """
    v = np.asarray(v, dtype=np.float64)
    gt = np.asarray(gt_valid, dtype=bool)
    inv = np.where((v > 0) & gt, 1.0 / np.maximum(v, 1.0), 0.0)
    return float(inv.mean())


# --------------------------------------------------------------------------
# Stratification and clustered inference
# --------------------------------------------------------------------------

STRATA = (
    ("V = 0", 0, 0),
    ("V = 1 (unique)", 1, 1),
    ("2-5", 2, 5),
    ("6-10", 6, 10),
    ("11-50", 11, 50),
    ("51-200", 51, 200),
    ("201-500", 201, 500),
    ("501-1000", 501, 1000),
    ("> 1000", 1001, 10**9),
)


def stratify(v: np.ndarray, gt_valid: np.ndarray) -> list[dict]:
    """Per-stratum query counts, mean V, and generating-cell validity rate."""
    v = np.asarray(v)
    gt = np.asarray(gt_valid, dtype=bool)
    rows = []
    for name, lo, hi in STRATA:
        m = (v >= lo) & (v <= hi)
        n = int(m.sum())
        if n == 0:
            rows.append({"stratum": name, "queries": 0, "mean_V": float("nan"),
                         "gt_valid_rate": float("nan"),
                         "naive_ceiling": float("nan"),
                         "recoverable_ceiling": float("nan")})
            continue
        rows.append({
            "stratum": name,
            "queries": n,
            "mean_V": float(v[m].mean()),
            "gt_valid_rate": float(gt[m].mean()),
            "naive_ceiling": naive_ceiling(v[m]),
            "recoverable_ceiling": recoverable_ceiling(v[m], gt[m]),
        })
    return rows


def clustered_bootstrap(values: np.ndarray,
                        cluster_index: np.ndarray,
                        statistic=np.mean,
                        n_boot: int = 2000,
                        seed: int = 20260808,
                        alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap resampling *clusters* (source cells).

    Queries from one cell are not independent; resampling queries would
    understate uncertainty. Returns ``(point, lo, hi)``.
    """
    values = np.asarray(values, dtype=np.float64)
    cluster_index = np.asarray(cluster_index)
    point = float(statistic(values))

    order = np.argsort(cluster_index, kind="stable")
    sorted_vals = values[order]
    sorted_keys = cluster_index[order]
    boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
    groups = np.split(sorted_vals, boundaries)
    n_groups = len(groups)
    if n_groups < 2:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, n_groups, size=n_groups)
        stats[b] = statistic(np.concatenate([groups[i] for i in pick]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def summarise_v(v: np.ndarray) -> dict:
    """Population statistics for V, matching Table 3 of the manuscript."""
    v = np.asarray(v)
    n = v.size
    return {
        "n_queries": int(n),
        "min": int(v.min()), "p5": float(np.percentile(v, 5)),
        "p25": float(np.percentile(v, 25)),
        "median": float(np.median(v)),
        "p75": float(np.percentile(v, 75)),
        "p95": float(np.percentile(v, 95)),
        "max": int(v.max()),
        "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if n > 1 else 0.0,
        "n_V0": int((v == 0).sum()),
        "n_V1": int((v == 1).sum()),
        "pct_V1": float(100.0 * (v == 1).sum() / n) if n else float("nan"),
        "n_V_le_10": int((v <= 10).sum()),
        "pct_V_le_10": float(100.0 * (v <= 10).sum() / n) if n else float("nan"),
        "n_V_le_50": int((v <= 50).sum()),
        "pct_V_le_50": float(100.0 * (v <= 50).sum() / n) if n else float("nan"),
    }


# --------------------------------------------------------------------------
# B. Multi-hint descriptions
# --------------------------------------------------------------------------

@dataclass
class Hint:
    """One hint: the described position lies in ``direction`` of a ``label``."""

    label: str
    direction: str
    color: str | None = None

    def key(self, use_color: bool) -> tuple:
        return (self.label, self.direction, self.color if use_color else None)


def hint_signature(hints: list[Hint], use_color: bool) -> tuple:
    """Canonical, order-independent template signature for a description."""
    return tuple(sorted(h.key(use_color) for h in hints))


def hint_delta(reference_xy: np.ndarray, object_xy: np.ndarray) -> np.ndarray:
    """Displacement for a hint, in the manuscript's convention.

    A hint reads "the position is [direction] of the [object]", so the
    position is the *subject* and the object is the *reference*. Equation
    (3) measures dc = c_reference - c_subject, hence

        dc = object_centroid - position.

    Both the direction derived when a description is parsed and the test
    applied to a candidate cell must use this same expression. Version
    1.1.0 used the negation on both sides, which is self-consistent and
    leaves V unchanged -- a global relabelling of the axes -- but reverses
    every direction name relative to the manuscript, which makes any
    comparison against a stored direction field read as total
    disagreement. Aligning it here makes that comparison meaningful.
    """
    return np.asarray(object_xy, dtype=np.float64) - np.asarray(
        reference_xy, dtype=np.float64)


def cells_satisfying_hint(scene: SceneData,
                          hint: Hint,
                          theta_deg: float,
                          use_color: bool,
                          colors: list[np.ndarray] | None = None) -> np.ndarray:
    """Boolean mask over cells: does the cell admit this hint?

    A cell admits ``hint`` iff it holds an object of the required label
    (and colour, when used) such that the cell centre stands in the hinted
    relation to that object. The cell centre plays the role of the
    described position, which is the direct cell-level analogue of the
    position-level relation a hint expresses.

    Kept as the reference implementation; :func:`build_atomic_index`
    computes every hint at once and is what the CLI uses. The two are
    asserted equal in the test suite.
    """
    d_index = DIRECTIONS.index(hint.direction)
    mask = np.zeros(scene.n_cells, dtype=bool)
    for ci, cell in enumerate(scene.cells):
        if cell.labels.size == 0:
            continue
        sel = cell.labels == hint.label
        if use_color:
            palette = colors[ci] if colors is not None else cell.colors
            sel = sel & (palette == hint.color)
        if not sel.any():
            continue
        cs = cell.centroids[sel][:, :2]
        delta = hint_delta(cell.center[None, :2], cs)
        if satisfies(delta, theta_deg)[:, d_index].any():
            mask[ci] = True
    return mask


def build_atomic_index(scene: SceneData,
                       theta_deg: float,
                       use_color: bool) -> dict:
    """Compute the cell mask of every atomic hint in one vectorised pass.

    The per-hint loop in :func:`cells_satisfying_hint` walks all cells once
    per hint. With 18 labels, 4 directions and a colour palette this is
    hundreds of passes over the scene. Every atomic mask is instead
    derivable from a single (n_objects, 4) predicate evaluation, because a
    cell admits a hint exactly when one of its own objects satisfies it.

    Returns ``{(label, direction, color_or_None): bool mask over cells}``.
    Absent keys denote hints no cell admits.
    """
    n_cells = scene.n_cells
    cell_idx, labels, colors, xy, centres = [], [], [], [], []
    for ci, cell in enumerate(scene.cells):
        centres.append(cell.center[:2])
        if cell.labels.size == 0:
            continue
        cell_idx.append(np.full(cell.labels.size, ci, dtype=np.int64))
        labels.append(cell.labels)
        colors.append(cell.colors if cell.colors.size == cell.labels.size
                      else np.full(cell.labels.size, None, dtype=object))
        xy.append(cell.centroids[:, :2])

    if not cell_idx:
        return {}

    cell_idx = np.concatenate(cell_idx)
    labels = np.concatenate(labels)
    colors = np.concatenate(colors)
    xy = np.concatenate(xy)
    centres = np.asarray(centres, dtype=np.float64)

    ok = satisfies(hint_delta(centres[cell_idx], xy), theta_deg)  # (N, 4)

    index: dict = {}
    for d, name in enumerate(DIRECTIONS):
        sel = np.flatnonzero(ok[:, d])
        if sel.size == 0:
            continue
        groups: dict = {}
        for i in sel:
            key = (labels[i], name, colors[i] if use_color else None)
            groups.setdefault(key, []).append(cell_idx[i])
        for key, cs in groups.items():
            mask = np.zeros(n_cells, dtype=bool)
            mask[np.asarray(cs, dtype=np.int64)] = True
            index[key] = mask
    return index


def compute_multihint_v(scene: SceneData,
                        descriptions: list[list[Hint]],
                        theta_deg: float = 90.0,
                        use_color: bool = False,
                        colors: list[np.ndarray] | None = None,
                        log=print) -> tuple[np.ndarray, dict]:
    """V for every description, via signature grouping.

    Returns ``(v_per_description, diagnostics)``.
    """
    atomic = build_atomic_index(scene, theta_deg, use_color)
    empty = np.zeros(scene.n_cells, dtype=bool)
    needed = {h.key(use_color) for hints in descriptions for h in hints}
    log(f"    {len(atomic)} atomic hints indexed in one pass; "
        f"{len(needed)} referenced by the descriptions")

    sig_v: dict[tuple, int] = {}
    sig_mask: dict[tuple, np.ndarray] = {}
    v_out = np.empty(len(descriptions), dtype=np.int32)
    for i, hints in enumerate(descriptions):
        sig = hint_signature(hints, use_color)
        if sig not in sig_v:
            mask = np.ones(scene.n_cells, dtype=bool)
            for h in hints:
                mask &= atomic.get(h.key(use_color), empty)
            sig_mask[sig] = mask
            sig_v[sig] = int(mask.sum())
        v_out[i] = sig_v[sig]

    diagnostics = {
        "n_descriptions": len(descriptions),
        "n_distinct_signatures": len(sig_v),
        "n_atomic_hints": len(atomic),
        "_sig_mask": sig_mask,
        "speedup_vs_per_description": (len(descriptions) / len(sig_v)
                                       if sig_v else float("nan")),
        "hints_per_description_mean": (float(np.mean([len(h) for h in descriptions]))
                                       if descriptions else float("nan")),
    }
    return v_out, diagnostics


def multihint_gt_validity(scene: SceneData,
                          descriptions: list[list[Hint]],
                          gt_cell_ids: list,
                          sig_mask: dict,
                          use_color: bool) -> tuple[np.ndarray, np.ndarray]:
    """Does each description's own generating cell satisfy it?

    The same question B1 asks of the single-relation family, asked of the
    multi-hint family. It is not redundant: a description states the
    relation of an object to the *described position*, whereas a cell is
    tested at its *centre*, and the two differ by up to half a cell. Where
    the generating cell fails its own description, exact recovery is
    unattainable and E[1/V] overstates the ceiling by that margin --
    exactly the effect B1 exists to expose.

    Returns ``(gt_valid, resolved)``; ``resolved`` is False where the
    generating cell id could not be matched, and those entries are
    excluded rather than counted as failures.
    """
    pos_of = {str(c.cell_id): i for i, c in enumerate(scene.cells)}
    gt_valid = np.zeros(len(descriptions), dtype=bool)
    resolved = np.zeros(len(descriptions), dtype=bool)
    for i, (hints, cid) in enumerate(zip(descriptions, gt_cell_ids)):
        ci = pos_of.get(str(cid)) if cid is not None else None
        if ci is None:
            continue
        resolved[i] = True
        mask = sig_mask.get(hint_signature(hints, use_color))
        if mask is not None:
            gt_valid[i] = bool(mask[ci])
    return gt_valid, resolved
