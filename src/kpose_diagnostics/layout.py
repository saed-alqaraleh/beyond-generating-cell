"""Dataset-layout discovery for KITTI360Pose diagnostics.

Supports both common prepared-data layouts:

1. Aggregate scene pickle (legacy/local convenience)::

      DATA/cells/2013_..._sync.pkl
      DATA/poses/2013_..._sync.pkl

2. Official Text2Pos prepared layout (one pickle per record)::

      DATA/cells/2013_..._sync/*.pkl
      DATA/poses/2013_..._sync/*.pkl

The functions are deliberately dependency-free so they can be used from
Colab before loading any dataset objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DatasetSources:
    base: Path
    scene: str
    cells: Path
    poses: Path
    layout: str


def _scene_names(kind_dir: Path) -> set[str]:
    out: set[str] = set()
    if not kind_dir.is_dir():
        return out
    for p in kind_dir.iterdir():
        if p.is_dir() and any(p.glob("*.pkl")):
            out.add(p.name)
        elif p.is_file() and p.suffix == ".pkl":
            out.add(p.stem)
    return out


def resolve_dataset_sources(base: Path, scene: str | None = None) -> DatasetSources:
    """Resolve cells/poses sources under a prepared KITTI360Pose root.

    ``base`` may be the dataset root itself, its ``cells``/``poses`` child,
    or a scene subdirectory. Both aggregate and directory layouts are accepted.
    """
    base = Path(base).expanduser().resolve()
    candidates: list[Path] = []
    if base.name in {"cells", "poses"}:
        candidates.append(base.parent)
    elif base.parent.name in {"cells", "poses"}:
        candidates.append(base.parent.parent)
    candidates.append(base)

    # One extra level is useful when the user points at a generic data folder.
    if base.is_dir():
        try:
            for p in base.iterdir():
                if p.is_dir() and (p / "cells").is_dir() and (p / "poses").is_dir():
                    candidates.append(p)
        except PermissionError:
            pass

    seen = set()
    errors = []
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        cdir, pdir = cand / "cells", cand / "poses"
        if not cdir.is_dir() or not pdir.is_dir():
            errors.append(f"{cand}: missing cells/ or poses/")
            continue
        scenes = sorted(_scene_names(cdir) & _scene_names(pdir))
        chosen = scene
        if chosen is None:
            if len(scenes) == 1:
                chosen = scenes[0]
            elif "2013_05_28_drive_0000_sync" in scenes:
                chosen = "2013_05_28_drive_0000_sync"
            elif scenes:
                chosen = scenes[0]
        if not chosen or chosen not in scenes:
            errors.append(f"{cand}: scene {scene!r} not found; available={scenes[:20]}")
            continue

        cf = cdir / f"{chosen}.pkl"
        pf = pdir / f"{chosen}.pkl"
        cd = cdir / chosen
        pd = pdir / chosen
        if cf.is_file() and pf.is_file():
            return DatasetSources(cand, chosen, cf, pf, "aggregate_scene_pickles")
        if cd.is_dir() and pd.is_dir() and any(cd.glob("*.pkl")) and any(pd.glob("*.pkl")):
            return DatasetSources(cand, chosen, cd, pd, "per_record_pickles")
        errors.append(f"{cand}: found scene name but cells/poses layouts do not pair")

    raise FileNotFoundError(
        "Could not resolve a prepared KITTI360Pose dataset. Expected either "
        "cells/<scene>.pkl + poses/<scene>.pkl or "
        "cells/<scene>/*.pkl + poses/<scene>/*.pkl.\n" + "\n".join(errors[:20])
    )


def discover_dataset_roots(search_roots: Iterable[Path], max_depth: int = 6) -> list[Path]:
    """Find directories that directly contain sibling ``cells`` and ``poses``.

    Traversal is depth-limited and prunes large irrelevant folders commonly
    found on Drive. Returned paths are unique and sorted by a small heuristic
    that prefers names beginning with ``k360``.
    """
    found: set[Path] = set()
    prune = {".git", "node_modules", "__pycache__", ".ipynb_checkpoints"}
    for start in search_roots:
        start = Path(start).expanduser()
        if not start.is_dir():
            continue
        stack = [(start, 0)]
        while stack:
            cur, depth = stack.pop()
            try:
                names = {p.name: p for p in cur.iterdir() if p.is_dir()}
            except (PermissionError, OSError):
                continue
            if "cells" in names and "poses" in names:
                found.add(cur.resolve())
                continue
            if depth >= max_depth:
                continue
            for name, p in names.items():
                if name not in prune and not name.startswith("."):
                    stack.append((p, depth + 1))
    return sorted(found, key=lambda p: (0 if p.name.startswith("k360") else 1, len(p.parts), str(p)))
