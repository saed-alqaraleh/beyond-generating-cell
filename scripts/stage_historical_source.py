#!/usr/bin/env python3
"""Verify and stage the *original* historical experiment runner, without changing it.

Why this exists
---------------
The August 2026 article-evidence runner checks the SHA-256 of the entire
v2.2.0 KPOSE source tree, including filenames, paths and documentation.
Moving the identical .py files under the public monorepo's `src/` changes that
hash. Silencing the guard would invalidate the original run's provenance.
Instead, this program verifies the archived, originally validated bundle and
extracts it into the original layout in a user-selected work directory.

It never removes an existing staging directory and never edits research code.
Repeat invocations verify and reuse an existing staging directory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = ROOT / "provenance" / "historical_source_bundles"
LOCK = ARCHIVE_DIR / "BUNDLE_LOCK.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for piece in iter(lambda: source.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def safe_extract(archive: zipfile.ZipFile, dest: Path) -> None:
    """Extract trusted local archive only after validating all member paths."""
    import stat
    base = dest.resolve()
    for item in archive.infolist():
        target = (base / item.filename).resolve()
        if not target.is_relative_to(base):
            raise ValueError(f"Unsafe archive path: {item.filename!r}")
        if stat.S_ISLNK(item.external_attr >> 16):
            raise ValueError(f"Unexpected archive symlink: {item.filename!r}")
    if archive.testzip() is not None:
        raise ValueError("Historical source ZIP failed its CRC test")
    archive.extractall(base)


def tree_hash(root: Path, extensions: set[str], mode: str) -> str:
    """Reproduce original `code_tree_hash` and `executor_hash` algorithms."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in extensions):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        if mode == "core":
            digest.update(relative.encode("utf-8") + b"\0")
        else:
            digest.update(relative.encode() + b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def verify(destination: Path) -> dict:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    archive_path = ARCHIVE_DIR / lock["archive_name"]
    if sha256(archive_path) != lock["sha256"]:
        raise RuntimeError("Historical bundle SHA-256 mismatch; refuse to execute")
    parent = destination / "article_evidence_v220_runner"
    core_zip = parent / "core" / "kpose_diagnostics_v2.2.0_repaired.zip"
    core_root = destination / "verified_core" / "kpose_diagnostics_v2.2.0_repaired"
    if not parent.is_dir() or not core_root.is_dir():
        raise FileNotFoundError("Staging incomplete: run without --verify-only first")
    if sha256(core_zip) != lock["expected_embedded_core_zip_sha256"]:
        raise RuntimeError("Embedded original core ZIP mismatch")
    core = tree_hash(core_root, {".py", ".sh", ".md", ".txt"}, "core")
    executor = tree_hash(parent, {".py", ".md", ".txt", ".sh", ".ipynb"}, "executor")
    if core != lock["expected_core_tree_hash"]:
        raise RuntimeError(f"Original core identity mismatch: {core}")
    if executor != lock["expected_executor_tree_hash"]:
        raise RuntimeError(f"Original executor identity mismatch: {executor}")
    return {"status":"PASS", "core_hash":core,"executor_hash":executor,
            "historical_code_root":str(parent),"historical_core_root":str(core_root)}


def stage(destination: Path) -> dict:
    destination = destination.expanduser().resolve()
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    archive_path = ARCHIVE_DIR / lock["archive_name"]
    if sha256(archive_path) != lock["sha256"]:
        raise RuntimeError("Original archived code checksum mismatch")
    parent = destination / "article_evidence_v220_runner"
    core_root = destination / "verified_core" / "kpose_diagnostics_v2.2.0_repaired"
    if not parent.exists():
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as z:
            safe_extract(z, destination)
    if not core_root.exists():
        with zipfile.ZipFile(parent / "core" / "kpose_diagnostics_v2.2.0_repaired.zip") as z:
            safe_extract(z, destination / "verified_core")
    return verify(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True,
                        help="Local scratch directory, NOT the Drive results directory")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify previously staged original source without extraction")
    args = parser.parse_args()
    try:
        result = verify(args.destination.resolve()) if args.verify_only else stage(args.destination)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"STAGING FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
