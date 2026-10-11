from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Iterable


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_bytes(path: Path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
    return path


def atomic_text(path: Path, text: str) -> Path:
    return atomic_bytes(path, text.encode("utf-8"))


def atomic_json(path: Path, payload) -> Path:
    return atomic_text(path, json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> Path:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
    return path


def copy_atomic(src: Path, dst: Path) -> Path:
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dst.name + ".", suffix=".tmp", dir=dst.parent)
    os.close(fd)
    try:
        shutil.copy2(src, tmp_name)
        os.replace(tmp_name, dst)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
    return dst


def executor_hash(root: Path) -> str:
    root = Path(root).resolve()
    h = hashlib.sha256()
    exts = {".py", ".md", ".txt", ".sh", ".ipynb"}
    for fp in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in exts):
        if "__pycache__" in fp.parts:
            continue
        rel = fp.relative_to(root).as_posix()
        h.update(rel.encode() + b"\0")
        h.update(bytes.fromhex(sha256_file(fp)))
    return h.hexdigest()


def rel_manifest(root: Path, exclude_names: set[str] | None = None) -> list[dict]:
    root = Path(root)
    exclude_names = exclude_names or set()
    rows = []
    for fp in sorted(p for p in root.rglob("*") if p.is_file()):
        if fp.name in exclude_names:
            continue
        rows.append({"path": fp.relative_to(root).as_posix(),
                     "size_bytes": int(fp.stat().st_size),
                     "sha256": sha256_file(fp)})
    return rows


def atomic_npz(path: Path, **arrays) -> Path:
    """Write a compressed NumPy archive atomically."""
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz.tmp", dir=path.parent)
    os.close(fd)
    # np.savez appends .npz when the filename lacks that suffix; write to an open file instead.
    try:
        with open(tmp_name, "wb") as fh:
            np.savez_compressed(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
    return path
