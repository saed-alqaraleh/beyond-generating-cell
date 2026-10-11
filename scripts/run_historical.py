#!/usr/bin/env python3
"""Supported entry point for historical v2.2.0 experiments (no notebook patches).

Examples (prepared data + adequate resources are required for experiment modes):
    python scripts/run_historical.py core-tests --stage /tmp/bgc_historical
    python scripts/run_historical.py executor-tests --stage /tmp/bgc_historical
    python scripts/run_historical.py plan --stage /tmp/bgc_historical --dataset /path/to/prepared --out /path/to/out
    python scripts/run_historical.py track --stage /tmp/bgc_historical --out /path/to/out --track 0
    python scripts/run_historical.py status --stage /tmp/bgc_historical --out /path/to/out

A stopped track is restarted with the same command, output directory and inputs.
Original stage signatures, locks, checkpoint and result files are left intact.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path
from stage_historical_source import stage


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("core-tests", "executor-tests", "plan", "track", "status", "finalize"))
    p.add_argument("--stage", type=Path, required=True, help="Local stage directory containing verified historical source")
    p.add_argument("--dataset", type=Path, help="Original prepared KITTI360Pose data root (cells/ and poses/)")
    p.add_argument("--out", type=Path, help="Durable output directory; use Google Drive in Colab")
    p.add_argument("--existing", type=Path, help="Prior v2.2.0 reusable job archive root for plan")
    p.add_argument("--track", type=int, choices=(0,1), help="Parallel track number, 0 or 1")
    args = p.parse_args()
    verified = stage(args.stage)
    runner_root = Path(verified["historical_code_root"])
    core_root = Path(verified["historical_core_root"])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(core_root), str(runner_root), environment.get("PYTHONPATH", "")])
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if args.command == "core-tests":
        cmd = [sys.executable, "-m", "unittest", "discover", "-s", str(core_root / "kpose_diagnostics/tests"), "-t", str(core_root), "-v"]
    elif args.command == "executor-tests":
        cmd = [sys.executable, "-m", "unittest", "article_evidence.tests.test_executor", "-v"]
    else:
        if args.out is None:
            p.error("--out is required for experiment/status/finalize commands")
        args.out = args.out.expanduser().resolve()
        cmd = [sys.executable, "-u", "-m", "article_evidence.runner", args.command if args.command!="track" else "run-track", "--out", str(args.out)]
        if args.command == "plan":
            if args.dataset is None:
                p.error("--dataset is required for planning")
            data = args.dataset.expanduser().resolve()
            if not (data / "cells").is_dir() or not (data / "poses").is_dir():
                p.error("Dataset must contain cells/ and poses/ directories")
            cmd += ["--base", str(data), "--core-root", str(core_root)]
            if args.existing:
                cmd += ["--existing", str(args.existing.expanduser().resolve())]
        elif args.command == "track":
            if args.track is None:
                p.error("--track 0 or --track 1 is required")
            if not (args.out / "PLAN.json").is_file():
                p.error("PLAN.json missing; run the plan command before a track")
            cmd += ["--track", str(args.track)]
        elif args.command == "finalize":
            if not (args.out / "PLAN.json").is_file():
                p.error("PLAN.json missing; cannot finalize")
            cmd += ["--core-archive", str(runner_root / "core/kpose_diagnostics_v2.2.0_repaired.zip")]
    print("Running verified historical command:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=runner_root, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
