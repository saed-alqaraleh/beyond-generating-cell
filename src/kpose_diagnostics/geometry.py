"""
geometry.py -- directional predicates for KITTI360Pose cell diagnostics.

Implements exactly the definitions in the manuscript:

  Eq. (3)  S_dir(o_s, o_t, v) = max(0, sigma_v(dc) / (|dx| + |dy| + eps))
           with  dc = c_t - c_s  and  sigma_v selecting
           -dy (north), +dy (south), +dx (east), -dx (west).

  Sec 4.2  S^theta_dir(o_s, o_t, v) = 1 iff the angle between (dx, dy)
           and the queried axis is at most theta, else 0.
           theta = 90 deg recovers the support of Eq. (3);
           theta = 45 deg is the dominant-axis criterion.

CONVENTION NOTE (important, and validated at run time):
  A query reads "[subject] [direction] of [reference]". The subject is
  o_s, the reference is o_t, and the displacement is measured from the
  subject to the reference. This is transcribed directly from the
  manuscript. `run_gt_validity.py` reproduces the published population
  statistics as a check on this convention; if the convention were
  flipped, the reproduction gate would fail loudly rather than produce
  a plausible wrong answer.

Everything here is pure: no I/O, no globals, no randomness.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6

#: Cardinal names, in the fixed order used for bit-packing.
DIRECTIONS = ("north", "south", "east", "west")

#: sigma_v as row vectors acting on (dx, dy). Index matches DIRECTIONS.
#: north -> -dy, south -> +dy, east -> +dx, west -> -dx
_SIGMA = np.array(
    [
        [0.0, -1.0],   # north
        [0.0, +1.0],   # south
        [+1.0, 0.0],   # east
        [-1.0, 0.0],   # west
    ],
    dtype=np.float64,
)


def sigma(delta_xy: np.ndarray) -> np.ndarray:
    """sigma_v(dc) for all four directions.

    Parameters
    ----------
    delta_xy : (..., 2) array of (dx, dy) displacements.

    Returns
    -------
    (..., 4) array; column order follows ``DIRECTIONS``.
    """
    delta_xy = np.asarray(delta_xy, dtype=np.float64)
    if delta_xy.shape[-1] != 2:
        raise ValueError(f"expected trailing dim 2, got {delta_xy.shape}")
    return delta_xy @ _SIGMA.T


def s_dir(delta_xy: np.ndarray) -> np.ndarray:
    """Graded direction score of Eq. (3), for all four directions.

    Returns a (..., 4) array in [0, 1].
    """
    delta_xy = np.asarray(delta_xy, dtype=np.float64)
    denom = np.abs(delta_xy[..., 0]) + np.abs(delta_xy[..., 1]) + EPS
    return np.maximum(0.0, sigma(delta_xy) / denom[..., None])


def satisfies(delta_xy: np.ndarray, theta_deg: float = 90.0) -> np.ndarray:
    """Binary directional predicate for all four directions.

    A displacement satisfies direction ``v`` iff

        sigma_v(dc) > 0   AND   angle((dx, dy), axis_v) <= theta.

    The explicit ``sigma_v > 0`` term makes ``theta = 90`` exactly the
    support of Eq. (3): a displacement lying *on* the perpendicular axis
    has sigma_v = 0, hence S_dir = 0, hence does not satisfy. Using
    ``angle <= 90`` alone would wrongly admit it.

    A zero displacement satisfies nothing.

    Parameters
    ----------
    delta_xy : (..., 2) array of (dx, dy).
    theta_deg : angular tolerance in degrees, in (0, 90].

    Returns
    -------
    (..., 4) boolean array; column order follows ``DIRECTIONS``.
    """
    if not (0.0 < theta_deg <= 90.0):
        raise ValueError(f"theta must be in (0, 90], got {theta_deg}")

    delta_xy = np.asarray(delta_xy, dtype=np.float64)
    sig = sigma(delta_xy)
    positive = sig > 0.0

    norm = np.linalg.norm(delta_xy[..., :2], axis=-1)
    nonzero = norm > 0.0

    if theta_deg >= 90.0:
        return positive & nonzero[..., None]

    # cos(angle to axis_v) = sigma_v / ||d||  because each axis is a unit vector.
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_angle = np.where(nonzero[..., None], sig / norm[..., None], -1.0)
    return positive & nonzero[..., None] & (cos_angle >= np.cos(np.radians(theta_deg)))


def analytic_acceptance_rate(theta_deg: float) -> float:
    """Expected acceptance rate under uniformly distributed displacements.

    Section 4.2: a single directional predicate accepts with probability
    theta / 180 degrees. Used as a correctness check on ``satisfies``.
    """
    return theta_deg / 180.0


def pack_directions(mask4: np.ndarray) -> int:
    """Pack a length-4 boolean direction mask into a small integer."""
    mask4 = np.asarray(mask4, dtype=bool).ravel()
    if mask4.size != 4:
        raise ValueError(f"expected 4 flags, got {mask4.size}")
    return int(sum(1 << i for i, flag in enumerate(mask4) if flag))


def unpack_directions(bits: int) -> tuple:
    """Inverse of :func:`pack_directions`."""
    return tuple(bool(bits >> i & 1) for i in range(4))
