"""
runner.py -- staging, resume markers, logging and provenance.

Every stage writes ``_STAGE_SUCCESS.json`` only after its outputs exist
and validate. A rerun with the same configuration hash skips completed
stages; a changed configuration produces a different hash and therefore a
different run directory, so completed work is never silently overwritten.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def config_hash(config: dict, length: int = 12) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]


def git_revision(root: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=root, capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() or "not-a-git-repo"
    except Exception:                                   # noqa: BLE001
        return "unavailable"


class Logger:
    """Timestamped, immediately flushed logging to console and file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.t0 = time.time()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8")
        else:
            self._fh = None

    def __call__(self, message: str = "") -> None:
        stamp = f"[{time.time() - self.t0:7.1f}s]"
        line = f"{stamp} {message}"
        print(line, flush=True)
        if self._fh is not None:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


class Stage:
    """Context manager for a resumable stage.

    Usage::

        with Stage(run_dir, "gt_validity", cfg, log) as st:
            if st.skipped:
                ...
            else:
                ... produce outputs ...
                st.outputs = [path1, path2]
    """

    def __init__(self, run_dir: Path, name: str, config: dict, log: Logger,
                 force: bool = False) -> None:
        self.dir = Path(run_dir) / name
        self.name = name
        self.config = config
        self.log = log
        self.force = force
        self.outputs: list[Path] = []
        self.marker = self.dir / "_STAGE_SUCCESS.json"
        self.skipped = False

    def _valid_marker(self) -> bool:
        if not self.marker.is_file():
            return False
        try:
            data = json.loads(self.marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        if data.get("config_hash") != config_hash(self.config):
            return False
        return all(Path(p).is_file() for p in data.get("outputs", []))

    def __enter__(self) -> "Stage":
        self.dir.mkdir(parents=True, exist_ok=True)
        if not self.force and self._valid_marker():
            self.skipped = True
            self.log(f"stage {self.name}: already complete, skipping "
                     f"(delete {self.marker} to force)")
        else:
            self.log(f"stage {self.name}: starting")
        self._t = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.log(f"stage {self.name}: FAILED -- {exc_type.__name__}: {exc}")
            return False
        if self.skipped:
            return False
        missing = [str(p) for p in self.outputs if not Path(p).is_file()]
        if missing:
            self.log(f"stage {self.name}: outputs missing, marker NOT written: "
                     f"{missing}")
            raise RuntimeError(f"stage {self.name} declared outputs it did not write")
        payload = {
            "stage": self.name,
            "config_hash": config_hash(self.config),
            "config": self.config,
            "outputs": [str(p) for p in self.outputs],
            "elapsed_s": round(time.time() - self._t, 2),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        tmp = self.marker.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.marker)
        self.log(f"stage {self.name}: complete in {payload['elapsed_s']}s")
        return False


def environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def save_json(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path

# ---------------------------------------------------------------------------
# Reproducibility fingerprints
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    """SHA-256 of one file using bounded memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def fingerprint_path(path: Path, cache_dir: Path | None = None) -> dict:
    """Content fingerprint a file or directory, caching per-file hashes.

    Cache validity uses absolute path + size + mtime_ns.  Thus the expensive
    byte hash is paid once for unchanged inputs, including across Colab reruns.
    The returned tree hash is independent of the absolute parent path.
    """
    path = Path(path).resolve()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no files to fingerprint under {path}")
    cache_path = None
    cache = {}
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "sha256_cache.json"
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            cache = {}
    rows = []
    changed = False
    for fp in files:
        st = fp.stat()
        key = str(fp)
        sig = f"{st.st_size}:{st.st_mtime_ns}"
        rec = cache.get(key)
        if not rec or rec.get("sig") != sig:
            digest = sha256_file(fp)
            cache[key] = {"sig": sig, "sha256": digest}
            changed = True
        else:
            digest = rec["sha256"]
        rel = fp.name if path.is_file() else fp.relative_to(path).as_posix()
        rows.append((rel, int(st.st_size), digest))
    if cache_path is not None and changed:
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(cache_path)
    tree = hashlib.sha256()
    for rel, size, digest in rows:
        tree.update(f"{rel}\0{size}\0{digest}\n".encode("utf-8"))
    return {"path": str(path), "n_files": len(rows),
            "size_bytes": int(sum(r[1] for r in rows)),
            "sha256": tree.hexdigest(), "method": "sha256-content-tree-v1"}


def code_tree_hash(root: Path) -> str:
    """Stable hash of source/configuration files in the package tree."""
    root = Path(root).resolve()
    h = hashlib.sha256()
    exts = {".py", ".sh", ".md", ".txt"}
    for fp in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in exts):
        if "__pycache__" in fp.parts:
            continue
        h.update(fp.relative_to(root).as_posix().encode("utf-8") + b"\0")
        h.update(bytes.fromhex(sha256_file(fp)))
    return h.hexdigest()
