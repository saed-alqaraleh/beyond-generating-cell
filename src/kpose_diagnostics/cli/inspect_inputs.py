"""
inspect_inputs.py -- report what is actually inside the pickles.

Run this first. It loads the artefacts, stubbing any class it cannot
import, and writes a structural report. Nothing is assumed about the
schema: whatever the report shows is what the other scripts will use, and
any field they cannot locate is named explicitly in the error.

    python -m kpose_diagnostics.cli.inspect_inputs \
        --cells .../cells/2013_05_28_drive_0000_sync.pkl \
        --poses .../poses/2013_05_28_drive_0000_sync.pkl \
        --out schema_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..loading import (CANDIDATES, OPTIONAL_ROLES, describe, get, load_pickle,
                       load_pickle_entries, pickle_files)


def _first(raw):
    if isinstance(raw, dict):
        return next(iter(raw.values()), None)
    if isinstance(raw, (list, tuple)):
        return raw[0] if raw else None
    return raw


def _roles(obj, prefix):
    """Which candidate names are present on this object, per role."""
    if obj is None:
        return {}
    have = set(vars(obj)) if hasattr(obj, "__dict__") else set()
    if isinstance(obj, dict):
        have |= {str(k) for k in obj}
    out = {}
    for role, names in CANDIDATES.items():
        if not role.startswith(prefix):
            continue
        hit = [n for n in names if n in have]
        out[role] = {"found": hit, "resolved_to": hit[0] if hit else None,
                     "required": role not in OPTIONAL_ROLES}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", type=Path)
    p.add_argument("--poses", type=Path)
    p.add_argument("--out", type=Path, default=Path("schema_report.json"))
    p.add_argument("--depth", type=int, default=4)
    args = p.parse_args(argv)

    if not args.cells and not args.poses:
        p.error("give at least one of --cells / --poses")

    report = {}
    for tag, path in (("cells", args.cells), ("poses", args.poses)):
        if path is None:
            continue
        print(f"\n=== {tag}: {path} ===", flush=True)
        try:
            files = pickle_files(path)
            # For a many-file dataset directory, schema inspection only needs
            # the first record. Loading every cell/pose merely to inspect the
            # schema can consume unnecessary RAM in Colab. Aggregate .pkl
            # files are still read normally and then sampled below.
            if path.is_dir():
                raw_one, stubbed = load_pickle(files[0])
                raw = raw_one
            else:
                raw, stubbed = load_pickle_entries(path)
        except Exception as exc:                       # noqa: BLE001
            print(f"  FAILED to load: {type(exc).__name__}: {exc}")
            report[tag] = {"error": f"{type(exc).__name__}: {exc}"}
            continue

        container = type(raw).__name__
        n = len(raw) if hasattr(raw, "__len__") else "n/a"
        print(f"  container: {container}, length {n}")
        if stubbed:
            print(f"  classes stubbed (expected, harmless): {sorted(stubbed)}")

        first = _first(raw)
        entry = {"container": container, "length": n,
                 "source_files": len(files),
                 "stubbed_classes": stubbed,
                 "structure": describe(first, args.depth)}

        if tag == "cells":
            entry["roles"] = _roles(first, "cell_")
            f_objs = entry["roles"].get("cell_objects", {}).get("resolved_to")
            if f_objs:
                objs = list(get(first, f_objs))
                if objs:
                    entry["object_roles"] = _roles(objs[0], "obj_")
                    entry["object_structure"] = describe(objs[0], args.depth)
        else:
            entry["roles"] = _roles(first, "pose_")
            f_d = entry["roles"].get("pose_descrs", {}).get("resolved_to")
            f_m = entry["roles"].get("pose_matches", {}).get("resolved_to")
            if f_d:
                try:
                    ds = list(get(first, f_d))
                except TypeError:
                    ds = []
                if ds:
                    if isinstance(ds[0], (str, bytes)):
                        entry["description_encoding"] = "text"
                        entry["description_sample"] = str(ds[0])[:200]
                        if f_m:
                            entry["extraction_route"] = "object_hint_matches"
                        else:
                            entry["extraction_error"] = (
                                "descriptions are text but no pose matches field was found")
                    else:
                        entry["description_encoding"] = "structured"
                        entry["extraction_route"] = "structured_descriptions"
                        entry["descr_roles"] = _roles(ds[0], "descr_")
                        entry["descr_structure"] = describe(ds[0], args.depth)
            elif f_m:
                entry["extraction_route"] = "object_hint_matches"
            else:
                entry["extraction_error"] = (
                    "neither structured descriptions nor a pose matches field was found")

        for group in ("roles", "object_roles", "descr_roles"):
            for role, info in entry.get(group, {}).items():
                if info["resolved_to"]:
                    mark, tail = "OK  ", info["resolved_to"]
                elif info["required"]:
                    mark, tail = "MISS", "NOT FOUND -- required"
                else:
                    mark, tail = "--  ", "absent (optional; a fallback is used)"
                print(f"  [{mark}] {role:<16} -> {tail}")
        report[tag] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {args.out.resolve()}")
    missing = [f"{t}.{r}" for t, e in report.items()
               for g in ("roles", "object_roles", "descr_roles")
               for r, i in e.get(g, {}).items()
               if not i["resolved_to"] and i["required"]]
    missing.extend(f"{t}.object_level_hints" for t, e in report.items()
                   if e.get("extraction_error"))
    optional = [f"{t}.{r}" for t, e in report.items()
                for g in ("roles", "object_roles", "descr_roles")
                for r, i in e.get(g, {}).items()
                if not i["resolved_to"] and not i["required"]]
    if optional:
        print(f"\nOptional roles absent (fallbacks apply): {optional}")
    if missing:
        print(f"\nREQUIRED roles unresolved: {missing}")
        print("Send schema_report.json; the right --field-* values follow from it.")
        return 1
    print("\nAll required roles resolved. The run scripts can proceed unmodified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
