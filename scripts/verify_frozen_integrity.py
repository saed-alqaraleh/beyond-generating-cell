#!/usr/bin/env python3
"""Fail-closed SHA-256 audit for *every* bundled source and frozen evidence file.

The lock table is independent of new documentation and verification scripts.
Never regenerate this table automatically as part of verification; doing so
would allow silent edits to experimental results or archived Python code.
"""
from __future__ import annotations
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "provenance/frozen_artifact_sha256.csv"
FROZEN_DIRS = (
    "src/", "data/frozen/", "results/heldout/", "results/null_control/",
    "results/text2locpp/", "results/controlled_and_generator/",
    "provenance/historical_source_bundles/",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    with MANIFEST.open(newline="", encoding="utf-8") as infile:
        rows = list(csv.DictReader(infile))
    documented = {r["relative_path"]: r for r in rows}
    if len(documented) != len(rows) or not rows:
        raise ValueError("Duplicate or missing frozen artifact manifest entries")
    actual = set()
    for dirname in FROZEN_DIRS:
        for p in (ROOT / dirname).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}:
                actual.add(p.relative_to(ROOT).as_posix())
    missing = sorted(set(documented) - actual)
    unexpected = sorted(actual - set(documented))
    incorrect = []
    for name in sorted(set(documented) & actual):
        record = documented[name]
        path = ROOT / name
        if path.stat().st_size != int(record["bytes"]) or sha256(path) != record["sha256"]:
            incorrect.append(name)
    report = {"status":"PASS" if not any((missing, unexpected, incorrect)) else "FAIL",
              "locked_files":len(rows), "missing":missing, "unexpected":unexpected,
              "modified":incorrect}
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
