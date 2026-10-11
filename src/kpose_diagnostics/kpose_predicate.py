"""Generator-faithful KITTI360Pose direction predicate.

The official Text2Pos generator operates in *cell-normalised coordinates* and
assigns the description direction from ``offset_closest = pose - closest_point``.
The proximity threshold is 0.05 in those normalised coordinates.  Cardinal
labels are then chosen from the same closest-point offset.  The original code
uses four sequential ``if`` statements, so exact |dx| == |dy| ties are finally
assigned to north/south (the y-axis tests run last).

This module intentionally validates labels strictly.  An unknown, missing, or
invalid label is an error; it must never silently become ``north`` or any other
category.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

OUTCOMES = ("on-top", "east", "west", "north", "south")
OUTCOME_TO_CODE = {name: i for i, name in enumerate(OUTCOMES)}
CODE_TO_OUTCOME = dict(enumerate(OUTCOMES))
DEFAULT_DELTA = 0.05  # official Text2Pos threshold, in normalised cell units


def normalize_direction_label(value: object) -> str:
    """Return a canonical direction label, raising on invalid input."""
    if value is None:
        raise ValueError("direction label is missing (None)")
    label = str(value).strip().lower().replace("_", "-")
    aliases = {"ontop": "on-top", "on top": "on-top"}
    label = aliases.get(label, label)
    if label not in OUTCOME_TO_CODE:
        raise ValueError(
            f"invalid direction label {value!r}; expected one of {OUTCOMES}"
        )
    return label


def label_to_code(value: object) -> int:
    return OUTCOME_TO_CODE[normalize_direction_label(value)]


def code_to_label(code: int) -> str:
    try:
        return CODE_TO_OUTCOME[int(code)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid direction code {code!r}; expected 0..4") from exc


def direction_from_offset(offset_closest_xy: np.ndarray,
                          delta: float = DEFAULT_DELTA) -> str:
    """Reproduce Text2Pos ``get_direction`` from ``pose - closest_point``.

    Parameters
    ----------
    offset_closest_xy:
        Two-vector in the cell-normalised coordinate system.
    delta:
        On-top threshold in the same normalised units.  The official value is
        0.05.
    """
    if not np.isfinite(delta) or delta < 0:
        raise ValueError(f"delta must be a finite non-negative value, got {delta}")
    v = np.asarray(offset_closest_xy, dtype=np.float64).ravel()
    if v.size < 2 or not np.isfinite(v[:2]).all():
        raise ValueError(f"invalid offset_closest vector: {offset_closest_xy!r}")
    dx, dy = float(v[0]), float(v[1])
    if np.hypot(dx, dy) < delta:
        return "on-top"
    # Important: official sequential-if tie behaviour is equivalent to using
    # x only when |dx| > |dy|; equality is ultimately overwritten by y.
    if abs(dx) > abs(dy):
        return "east" if dx >= 0 else "west"
    return "north" if dy >= 0 else "south"


def direction_code_from_offsets(offsets_xy: np.ndarray,
                                delta: float = DEFAULT_DELTA) -> np.ndarray:
    """Vectorised :func:`direction_from_offset`, returning int8 codes."""
    a = np.asarray(offsets_xy, dtype=np.float64)
    if a.ndim < 2 or a.shape[-1] < 2:
        raise ValueError(f"expected (...,2) offsets, got {a.shape}")
    xy = a[..., :2]
    if not np.isfinite(xy).all():
        raise ValueError("offset array contains NaN/Inf")
    dx, dy = xy[..., 0], xy[..., 1]
    use_x = np.abs(dx) > np.abs(dy)
    out = np.where(use_x, np.where(dx >= 0, 1, 2),
                   np.where(dy >= 0, 3, 4)).astype(np.int8)
    out[np.hypot(dx, dy) < delta] = 0
    return out


def direction_of(position_xy: np.ndarray,
                 closest_point_xy: np.ndarray,
                 min_distance: float | None = None,
                 delta: float = DEFAULT_DELTA) -> str:
    """Compatibility wrapper using a position and its closest object point.

    ``min_distance`` is accepted for backward compatibility, but cardinal
    direction is always computed from the closest-point vector, matching the
    generator.  If supplied, ``min_distance`` only decides the on-top branch.
    """
    pos = np.asarray(position_xy, float).ravel()[:2]
    closest = np.asarray(closest_point_xy, float).ravel()[:2]
    off = pos - closest
    if min_distance is not None and float(min_distance) < delta:
        return "on-top"
    return direction_from_offset(off, delta=delta)


def directions_of(position_xy: np.ndarray,
                  closest_points_xy: np.ndarray,
                  min_distances: np.ndarray | None = None,
                  delta: float = DEFAULT_DELTA) -> np.ndarray:
    pos = np.asarray(position_xy, float).ravel()[:2]
    pts = np.asarray(closest_points_xy, float)[:, :2]
    codes = direction_code_from_offsets(pos[None, :] - pts, delta=delta)
    if min_distances is not None:
        md = np.asarray(min_distances, float)
        codes[md < delta] = 0
    return np.asarray([code_to_label(c) for c in codes], dtype=object)


def min_distance_to_points(position_xy: np.ndarray, points_xy: np.ndarray) -> float:
    p = np.asarray(points_xy, float)[:, :2]
    q = np.asarray(position_xy, float).ravel()[:2]
    if p.size == 0:
        return float("inf")
    return float(np.sqrt(((p - q[None, :]) ** 2).sum(axis=1)).min())


@dataclass
class VerificationReport:
    n_compared: int
    n_agree: int
    confusion: dict
    per_direction: dict
    delta: float

    @property
    def agreement(self) -> float:
        return self.n_agree / self.n_compared if self.n_compared else float("nan")

    @property
    def faithful(self) -> bool:
        return bool(self.n_compared and self.agreement >= 0.95)

    def summary(self) -> str:
        lines = [
            f"compared      : {self.n_compared:,} hints",
            f"agreement     : {self.agreement:.4f}   "
            f"({'FAITHFUL' if self.faithful else 'NOT FAITHFUL'})",
            f"delta         : {self.delta} normalised-cell units",
            "per stored direction:",
        ]
        for d in OUTCOMES:
            n, k = self.per_direction.get(d, (0, 0))
            if n:
                lines.append(f"    {d:<8} {k:>7,}/{n:<7,}  ({k/n:.4f})")
        return "\n".join(lines)


def verify_offsets(records: Iterable[tuple[np.ndarray, object]],
                   delta: float = DEFAULT_DELTA) -> VerificationReport:
    """Verify ``(offset_closest_xy, stored_label)`` records strictly."""
    confusion: dict = {}
    per_direction: dict = {}
    n = k = 0
    for offset, stored_raw in records:
        stored = normalize_direction_label(stored_raw)
        got = direction_from_offset(offset, delta)
        n += 1
        agree = got == stored
        k += int(agree)
        confusion[(stored, got)] = confusion.get((stored, got), 0) + 1
        tn, tk = per_direction.get(stored, (0, 0))
        per_direction[stored] = (tn + 1, tk + int(agree))
    return VerificationReport(n, k, confusion, per_direction, delta)


def verify_against_stored(records, delta: float = DEFAULT_DELTA) -> VerificationReport:
    """Backward-compatible verifier.

    Accepts either ``(offset, stored)`` or the former four-field
    ``(position, closest_point, min_distance, stored)`` form. Invalid labels
    always raise instead of being skipped.
    """
    converted = []
    for rec in records:
        if len(rec) == 2:
            converted.append((rec[0], rec[1]))
        elif len(rec) == 4:
            pos, closest, mind, stored = rec
            off = np.asarray(pos, float).ravel()[:2] - np.asarray(closest, float).ravel()[:2]
            # Preserve the old explicit min-distance branch if it was given.
            if mind is not None and float(mind) < delta:
                off = np.zeros(2, dtype=float)
            converted.append((off, stored))
        else:
            raise ValueError(f"verification record must have 2 or 4 fields, got {rec!r}")
    return verify_offsets(converted, delta)


def calibrate_delta(records, candidates=(0.015, 0.025, 0.05, 0.075, 0.10)) -> list:
    return [(float(d), verify_against_stored(records, float(d)).agreement)
            for d in candidates]
