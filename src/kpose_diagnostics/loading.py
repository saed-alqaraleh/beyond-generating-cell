"""
loading.py -- schema-discovering loaders for KITTI360Pose artefacts.

Two problems this module solves.

1. The KITTI360Pose ``cells/*.pkl`` and ``poses/*.pkl`` files contain
   instances of classes defined in the Text2Pos repository
   (``datapreparation.kitti360pose.imports`` and friends). A plain
   ``pickle.load`` therefore raises ModuleNotFoundError unless that repo
   is importable. :class:`PermissiveUnpickler` substitutes lightweight
   stand-ins for any missing class, so the data can be read with numpy
   alone. Attribute values survive intact; only methods are lost, and we
   need none.

2. The exact attribute names in those classes are not assumed anywhere
   in this package. :func:`describe` walks a loaded object and reports
   what is actually there; the resolvers below then locate the fields we
   need by trying documented candidates and, failing that, by shape and
   dtype. If a required field cannot be located the loader raises with a
   message naming what it looked for and what it found, rather than
   guessing and producing a plausible wrong number.
"""

from __future__ import annotations

import hashlib
import io
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# --------------------------------------------------------------------------
# 1. Reading pickles without the originating package
# --------------------------------------------------------------------------

class _Stub:
    """Stand-in for a class that cannot be imported.

    Unpickling an ordinary instance does ``cls.__new__(cls)`` then updates
    ``__dict__``, so an empty subclass of ``object`` is sufficient.
    """

    __slots__ = ("__dict__",)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        keys = ", ".join(sorted(self.__dict__)[:6])
        return f"<{type(self).__name__} {keys}>"


class _StubDict(dict):
    """Stand-in for dict subclasses such as ``easydict.EasyDict``."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


_DICTLIKE_HINTS = ("easydict", "EasyDict", "AttrDict", "Munch", "DotDict")


class PermissiveUnpickler(pickle.Unpickler):
    """Unpickler that stubs out classes it cannot import.

    ``missing`` records every substitution so the caller can report which
    classes were stubbed. Real imports are always preferred: a stub is
    created only after the import genuinely fails.
    """

    def __init__(self, file: io.BufferedReader) -> None:
        super().__init__(file)
        self.missing: dict[str, str] = {}
        self._cache: dict[tuple[str, str], type] = {}

    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            pass

        key = (module, name)
        if key in self._cache:
            return self._cache[key]

        base = _StubDict if any(h in module or h in name for h in _DICTLIKE_HINTS) else _Stub
        stub = type(name, (base,), {"__module__": module})
        self._cache[key] = stub
        self.missing[f"{module}.{name}"] = base.__name__
        return stub


def load_pickle(path: Path) -> tuple[Any, dict[str, str]]:
    """Load a pickle, stubbing unavailable classes.

    Returns ``(object, stubbed_classes)``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    with path.open("rb") as fh:
        up = PermissiveUnpickler(fh)
        obj = up.load()
    return obj, dict(up.missing)


def pickle_files(path: Path) -> list[Path]:
    """Return pickle files represented by ``path``.

    A file denotes an aggregate pickle. A directory denotes the official
    prepared KITTI360Pose per-record layout and is read in filename order.
    """
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.pkl"))
        if not files:
            files = sorted(path.rglob("*.pkl"))
        if files:
            return files
        raise FileNotFoundError(f"no .pkl files under directory: {path}")
    raise FileNotFoundError(f"no such file or directory: {path}")


def load_pickle_entries(path: Path) -> tuple[list[Any], dict[str, str]]:
    """Load either one aggregate pickle or a directory of record pickles.

    Lists/tuples and dict values are flattened one level. A single pickled
    object becomes one entry. Missing-class substitutions are accumulated.
    """
    entries: list[Any] = []
    missing: dict[str, str] = {}
    files = pickle_files(path)
    for fp in files:
        raw, stubbed = load_pickle(fp)
        missing.update(stubbed)
        if isinstance(raw, dict):
            entries.extend(raw.values())
        elif isinstance(raw, (list, tuple)):
            entries.extend(raw)
        else:
            entries.append(raw)
    return entries, missing


# --------------------------------------------------------------------------
# 2. Schema discovery
# --------------------------------------------------------------------------

def _summarise(value: Any, depth: int, max_depth: int) -> Any:
    if isinstance(value, np.ndarray):
        return {"__type__": "ndarray", "shape": list(value.shape),
                "dtype": str(value.dtype)}
    if isinstance(value, (str, bytes)):
        text = value.decode("utf8", "replace") if isinstance(value, bytes) else value
        return {"__type__": type(value).__name__, "len": len(value),
                "sample": text[:80]}
    if isinstance(value, (int, float, bool, type(None))):
        return {"__type__": type(value).__name__, "value": value}
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        out = {"__type__": type(value).__name__, "len": len(seq)}
        if seq and depth < max_depth:
            out["item0"] = _summarise(seq[0], depth + 1, max_depth)
        return out
    if isinstance(value, dict):
        out = {"__type__": "dict", "len": len(value),
               "keys": [str(k) for k in list(value)[:12]]}
        if value and depth < max_depth:
            first = next(iter(value.values()))
            out["value0"] = _summarise(first, depth + 1, max_depth)
        return out
    if hasattr(value, "__dict__"):
        out = {"__type__": f"{type(value).__module__}.{type(value).__name__}",
               "attrs": {}}
        if depth < max_depth:
            for k, v in sorted(vars(value).items()):
                out["attrs"][k] = _summarise(v, depth + 1, max_depth)
        else:
            out["attrs"] = sorted(vars(value))
        return out
    return {"__type__": type(value).__name__}


def describe(obj: Any, max_depth: int = 3) -> dict:
    """Structural summary of a loaded artefact, safe to print or store."""
    return _summarise(obj, 0, max_depth)


# --------------------------------------------------------------------------
# 3. Field resolution -- documented candidates first, then shape/dtype
# --------------------------------------------------------------------------

#: Attribute names observed in the Text2Pos / Text2Loc lineage, tried in order.
#: These are *candidates*, not assumptions: resolution failure is an error.
CANDIDATES = {
    "cell_objects":  ("objects", "instances", "objs", "cell_objects"),
    "cell_id":       ("id", "cell_id", "idx", "name"),
    "cell_bbox":     ("bbox_w", "bbox", "bbox_world", "extent"),
    "cell_center":   ("center", "cell_center", "centroid", "pose_w"),
    "obj_label":     ("label", "cls", "semantic_label", "class_name", "name"),
    "obj_xyz":       ("xyz", "points", "xyz_w", "points_w", "pts"),
    "obj_id":        ("id", "object_id", "instance_id", "obj_id"),
    "obj_color":     ("color_text", "object_color_text", "rgb", "color"),
    "pose_position": ("pose_w", "pose", "position", "xyz", "pose_world"),
    "pose_cell_id":  ("cell_id", "cell", "cell_idx", "gt_cell_id"),
    "pose_descrs":   ("descriptions", "descriptors", "hints", "description"),
    "pose_hints":     ("hint_descriptions", "hint_description", "hint_texts"),
    "pose_matches":   ("matches", "gt_matches", "object_hint_matches"),
    "descr_obj_id":  ("object_id", "obj_id", "instance_id", "id"),
    "descr_label":   ("object_label", "label", "cls", "class_name"),
    "descr_dir":     ("direction", "direction_text", "relation", "spatial"),
    "descr_color":   ("object_color_text", "color_text", "color"),
    "descr_offset":  ("offset_center", "offset_closest",
                      "best_offset_center", "best_offset_closest"),
}


#: Roles the package can do without. ``cell_center`` falls back to the mean
#: of object centroids; ``obj_color`` is needed only with --use-color;
#: ``descr_dir`` is never needed because directions are derived
#: geometrically. Unresolved OPTIONAL roles are informational, not errors.
OPTIONAL_ROLES = frozenset({"cell_center", "cell_bbox", "obj_id", "descr_offset",
                            "obj_color", "descr_obj_id", "descr_dir", "descr_color",
                            "pose_cell_id", "pose_descrs", "pose_hints",
                            "pose_matches"})


class SchemaError(RuntimeError):
    """Raised when a required field cannot be located in the real data."""


def resolve(obj: Any, role: str, extra: Sequence[str] = ()) -> str:
    """Find the attribute (or key) on ``obj`` that plays ``role``.

    Raises :class:`SchemaError` listing what was tried and what exists.
    """
    names = tuple(extra) + CANDIDATES.get(role, ())
    available = set(vars(obj)) if hasattr(obj, "__dict__") else set()
    if isinstance(obj, dict):
        available |= {str(k) for k in obj}

    for name in names:
        if name in available or (not isinstance(obj, dict) and hasattr(obj, name)):
            return name
    raise SchemaError(
        f"could not resolve role {role!r}.\n"
        f"  tried:     {list(names)}\n"
        f"  available: {sorted(available)}\n"
        f"Pass the correct name via the matching --field-* CLI option, or "
        f"run `python -m kpose_diagnostics.cli.inspect_inputs` and send the "
        f"schema report."
    )


def pick(obj: Any, role: str, overrides: dict | None = None) -> str:
    """Resolve ``role``, honouring an explicit override.

    If the caller supplied an override for this role it must exist on the
    object. Falling back to a guessed candidate would defeat the purpose
    of the override and could silently produce a wrong number, so a bad
    override raises instead.
    """
    overrides = overrides or {}
    want = overrides.get(role)
    if want is None:
        return resolve(obj, role)

    available = set(vars(obj)) if hasattr(obj, "__dict__") else set()
    if isinstance(obj, dict):
        available |= {str(k) for k in obj}
    if want in available or (not isinstance(obj, dict) and hasattr(obj, want)):
        return want
    raise SchemaError(
        f"--field-{role.replace('_', '-')} was given as {want!r}, but that "
        f"attribute does not exist.\n"
        f"  available: {sorted(available)}\n"
        f"Run `python -m kpose_diagnostics.cli.inspect_inputs` to see the "
        f"real field names."
    )


def get(obj: Any, name: str) -> Any:
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    return getattr(obj, name)


def as_xyz(value: Any) -> np.ndarray:
    """Coerce a point or point set to float64 ``(N, 3)``."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise SchemaError(f"expected (N,3) coordinates, got shape {arr.shape}")
    return arr[:, :3]


# --------------------------------------------------------------------------
# 4. Normalised in-memory representation
# --------------------------------------------------------------------------

@dataclass
class CellRecord:
    """One map cell, reduced to what the diagnostics need.

    ``centroids`` and ``points_xy`` preserve the coordinate system stored in
    the cell pickle.  For official KITTI360Pose cells this is normalised to
    [0,1] relative to ``bbox_w``; ``bbox_w`` and ``cell_size`` retain the
    world transform needed for metric-distance outputs.
    """

    cell_id: str
    center: np.ndarray                    # (3,) world-space cell center
    labels: np.ndarray                    # (M,) object labels, dtype=object
    centroids: np.ndarray                 # (M, 3), cell-local for KPOSE
    object_ids: np.ndarray                # (M,)
    colors: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    points_xy: list = field(default_factory=list)
    bbox_w: np.ndarray | None = None       # (6,) world-space bbox
    cell_size: float = 30.0


@dataclass
class SceneData:
    scene: str
    cells: list[CellRecord]
    stubbed_classes: dict[str, str] = field(default_factory=dict)
    resolved_fields: dict[str, str] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_objects(self) -> int:
        return int(sum(c.labels.size for c in self.cells))

    def label_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cell in self.cells:
            for lab in cell.labels:
                counts[lab] = counts.get(lab, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _iter_cells(raw: Any):
    if isinstance(raw, dict):
        yield from raw.values()
    elif isinstance(raw, (list, tuple)):
        yield from raw
    else:
        raise SchemaError(
            f"cells pickle holds {type(raw).__name__}; expected list or dict"
        )


def _bbox_center(value: Any) -> np.ndarray:
    """Best-effort center of a 2D/3D world-space bounding box."""
    a = np.asarray(value, dtype=np.float64)
    if a.shape == (2, 3):
        return a.mean(axis=0)
    if a.shape == (3, 2):
        return a.mean(axis=1)
    a = a.ravel()
    if a.size >= 6:
        return (a[:3] + a[3:6]) / 2.0
    if a.size == 4:
        return np.array([(a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0, 0.0])
    raise SchemaError(f"could not interpret cell bounding box with shape {np.asarray(value).shape}")


# Official Text2Pos colour centres / names (utils.py in the CVPR 2022 code).
_TEXT2POS_COLORS = np.array([
    [47.2579917, 49.75368454, 42.4153065],
    [136.32696657, 136.95241796, 126.02741229],
    [87.49822126, 91.69058836, 80.14558512],
    [213.91030679, 216.25033052, 207.24611073],
    [110.39218852, 112.91977458, 103.68638249],
    [27.47505158, 28.43996795, 25.16840296],
    [66.65951839, 70.22342483, 60.20395996],
    [171.00852191, 170.05737735, 155.00130334],
], dtype=np.float64) / 255.0
_TEXT2POS_COLOR_NAMES = np.array(
    ["dark-green", "gray", "gray-green", "bright-gray",
     "gray", "black", "green", "beige"], dtype=object)

def _color_text(value: Any) -> str | None:
    """Return Text2Pos's exact discrete colour word.

    Strings are preserved. Numeric RGB arrays are averaged exactly as
    ``Object3d.get_color_text`` does, then mapped to the nearest official
    Text2Pos colour centre in L2 distance.
    """
    try:
        if callable(value):
            value = value()
    except Exception:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf8", "replace")
    if isinstance(value, str):
        out = value.strip().lower()
        return out or None
    a = np.asarray(value)
    if a.ndim == 0 and a.dtype.kind in {"U", "S", "O"}:
        out = str(a.item()).strip().lower()
        return out or None
    if a.ndim >= 1 and a.shape[-1] >= 3 and np.issubdtype(a.dtype, np.number):
        rgb = np.asarray(a[..., :3], dtype=np.float64).reshape(-1, 3).mean(axis=0)
        if np.nanmax(np.abs(rgb)) > 1.5:
            rgb = rgb / 255.0
        if not np.isfinite(rgb).all():
            return None
        return str(_TEXT2POS_COLOR_NAMES[np.argmin(np.linalg.norm(_TEXT2POS_COLORS-rgb, axis=1))])
    return None


def load_cells(path: Path, scene: str, overrides: dict | None = None,
               keep_points: int = 0) -> SceneData:
    """Load a cells pickle into :class:`SceneData`.

    Object centroids are the arithmetic mean of the object's points, as
    specified in Section 4.2 of the manuscript.
    """
    overrides = overrides or {}
    cells_raw, stubbed = load_pickle_entries(path)
    if not cells_raw:
        raise SchemaError(f"{path} contained no cells")

    probe = cells_raw[0]
    f_objs = pick(probe, "cell_objects", overrides)
    f_cid = pick(probe, "cell_id", overrides)

    objs_probe = list(get(probe, f_objs))
    if not objs_probe:
        for cand in cells_raw[1:50]:
            objs_probe = list(get(cand, f_objs))
            if objs_probe:
                break
    if not objs_probe:
        raise SchemaError("no cell among the first 50 contained any objects")

    o_probe = objs_probe[0]
    f_lab = pick(o_probe, "obj_label", overrides)
    f_xyz = pick(o_probe, "obj_xyz", overrides)
    try:
        f_oid = pick(o_probe, "obj_id", overrides)
    except SchemaError:
        if overrides.get("obj_id"):
            raise
        f_oid = None

    try:
        f_ctr = pick(probe, "cell_center", overrides)
    except SchemaError:
        if overrides.get("cell_center"):
            raise
        f_ctr = None
    try:
        f_bbox = pick(probe, "cell_bbox", overrides)
    except SchemaError:
        if overrides.get("cell_bbox"):
            raise
        f_bbox = None
    try:
        f_col = pick(o_probe, "obj_color", overrides)
    except SchemaError:
        if overrides.get("obj_color"):
            raise
        f_col = None

    cells: list[CellRecord] = []
    for idx, craw in enumerate(cells_raw):
        objs = list(get(craw, f_objs))
        labels, cents, oids, colors = [], [], [], []
        pts_keep = []
        for j, o in enumerate(objs):
            pts = as_xyz(get(o, f_xyz))
            if pts.size == 0:
                continue
            labels.append(str(get(o, f_lab)))
            cents.append(pts.mean(axis=0))
            if keep_points:
                xy = pts[:, :2]
                # keep_points < 0 means retain every point (required for
                # generator-faithful closest-point KPOSE diagnostics).
                if keep_points > 0 and xy.shape[0] > keep_points:
                    step = max(1, xy.shape[0] // keep_points)
                    xy = xy[::step][:keep_points]
                pts_keep.append(np.ascontiguousarray(xy, dtype=np.float32))
            oids.append(str(get(o, f_oid)) if f_oid else f"{idx}:{j}")
            colors.append(_color_text(get(o, f_col)) if f_col else None)

        cents_arr = (np.asarray(cents, dtype=np.float64)
                     if cents else np.empty((0, 3), dtype=np.float64))
        bbox_w = None
        if f_bbox is not None:
            bb = np.asarray(get(craw, f_bbox), dtype=np.float64).ravel()
            if bb.size >= 6:
                bbox_w = bb[:6]
                center = _bbox_center(bbox_w)
            else:
                raise SchemaError(f"cell bbox has {bb.size} values; expected >=6")
        elif f_ctr is not None:
            center = as_xyz(get(craw, f_ctr))[0]
        elif cents_arr.size:
            center = cents_arr.mean(axis=0)
        else:
            center = np.zeros(3)
        raw_cell_size = getattr(craw, "cell_size", None)
        if raw_cell_size is not None:
            cell_size = float(raw_cell_size)
        elif bbox_w is not None:
            cell_size = float(np.max(bbox_w[3:6] - bbox_w[:3]))
        else:
            cell_size = 30.0

        cells.append(CellRecord(
            points_xy=pts_keep,
            cell_id=str(get(craw, f_cid)),
            center=center,
            labels=np.asarray(labels, dtype=object),
            centroids=cents_arr,
            object_ids=np.asarray(oids, dtype=object),
            colors=np.asarray(colors, dtype=object),
            bbox_w=bbox_w,
            cell_size=cell_size,
        ))

    return SceneData(
        scene=scene,
        cells=cells,
        stubbed_classes=stubbed,
        resolved_fields={
            "cell_objects": f_objs, "cell_id": f_cid, "obj_label": f_lab,
            "obj_xyz": f_xyz, "obj_id": f_oid or "<synthesised>",
            "obj_color": f_col or "<unavailable>",
            "cell_bbox": f_bbox or "<unavailable>",
            "cell_center": f_ctr or (f"<center of {f_bbox}>" if f_bbox else "<mean of object centroids>"),
        },
    )


def write_json(path: Path, payload: dict) -> None:
    """Atomic JSON write (temp file then replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------
# 5. Scene cache
# --------------------------------------------------------------------------

def _source_fingerprint(path: Path) -> str:
    """Identity of the input, from file sizes and modification times.

    Cheap enough to compute on a directory of thousands of records, and
    sufficient to invalidate the cache when the data changes. Hashing the
    bytes would be more rigorous but would cost as much as the load the
    cache exists to avoid.
    """
    parts = []
    for fp in pickle_files(path):
        st = fp.stat()
        parts.append(f"{fp.name}:{st.st_size}:{int(st.st_mtime)}")
    blob = "|".join(sorted(parts)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def save_scene_cache(cache_path: Path, scene: SceneData, fingerprint: str) -> Path:
    """Persist a parsed scene as one compressed archive (atomic write)."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lengths = np.asarray([c.labels.size for c in scene.cells], dtype=np.int64)
    # numpy appends ".npz" when the target lacks it, so the temporary name
    # must already end in ".npz" or the atomic rename would target a file
    # that was never created.
    tmp = cache_path.with_name(cache_path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        fingerprint=np.asarray(fingerprint),
        scene=np.asarray(scene.scene),
        cell_ids=np.asarray([c.cell_id for c in scene.cells], dtype=object),
        centers=np.asarray([c.center for c in scene.cells], dtype=np.float64),
        lengths=lengths,
        labels=(np.concatenate([c.labels for c in scene.cells])
                if scene.cells else np.empty(0, dtype=object)),
        centroids=(np.concatenate([c.centroids for c in scene.cells])
                   if scene.cells else np.empty((0, 3))),
        object_ids=(np.concatenate([c.object_ids for c in scene.cells])
                    if scene.cells else np.empty(0, dtype=object)),
        colors=(np.concatenate([c.colors if c.colors.size == c.labels.size
                                else np.full(c.labels.size, None, dtype=object)
                                for c in scene.cells])
                if scene.cells else np.empty(0, dtype=object)),
        resolved=np.asarray(json.dumps(scene.resolved_fields)),
        stubbed=np.asarray(json.dumps(scene.stubbed_classes)),
        allow_pickle=True,
    )
    tmp.replace(cache_path)
    return cache_path


def load_scene_cache(cache_path: Path, fingerprint: str) -> SceneData | None:
    """Return the cached scene, or None when absent or stale."""
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        return None
    try:
        z = np.load(cache_path, allow_pickle=True)
        if str(z["fingerprint"]) != fingerprint:
            return None
        lengths = z["lengths"]
        offs = np.concatenate([[0], np.cumsum(lengths)])
        labels, centroids = z["labels"], z["centroids"]
        oids, colors, centers = z["object_ids"], z["colors"], z["centers"]
        ids = z["cell_ids"]
        cells = [
            CellRecord(cell_id=str(ids[i]), center=centers[i],
                       labels=labels[offs[i]:offs[i + 1]],
                       centroids=centroids[offs[i]:offs[i + 1]],
                       object_ids=oids[offs[i]:offs[i + 1]],
                       colors=colors[offs[i]:offs[i + 1]])
            for i in range(len(lengths))
        ]
        return SceneData(scene=str(z["scene"]), cells=cells,
                         stubbed_classes=json.loads(str(z["stubbed"])),
                         resolved_fields=json.loads(str(z["resolved"])))
    except Exception:                                   # noqa: BLE001
        return None                                     # a bad cache is ignored


def load_cells_cached(path: Path, scene_name: str,
                      overrides: dict | None = None,
                      cache_dir: Path | None = None,
                      log=None) -> SceneData:
    """:func:`load_cells` with a parsed-scene cache.

    Parsing is identical either way; the cache only avoids repeating it.
    A cache built under different --field-* overrides is not reused,
    because the overrides change what is parsed.
    """
    say = log or (lambda *_: None)
    if cache_dir is None:
        return load_cells(path, scene_name, overrides)

    cache_dir = Path(cache_dir)
    key = hashlib.sha256(
        json.dumps({"path": str(Path(path).resolve()),
                    "overrides": overrides or {}},
                   sort_keys=True).encode("utf-8")).hexdigest()[:12]
    cache_path = cache_dir / f"scene_{scene_name}_{key}.npz"

    fp = _source_fingerprint(path)
    cached = load_scene_cache(cache_path, fp)
    if cached is not None:
        say(f"    scene cache hit: {cache_path}")
        return cached

    scene = load_cells(path, scene_name, overrides)
    try:
        save_scene_cache(cache_path, scene, fp)
        say(f"    scene cached to {cache_path}")
    except OSError as exc:
        say(f"    could not write scene cache ({exc}); continuing")
    return scene
