"""
run_parallel.py -- run every experiment across N parallel tracks.

The jobs differ in cost by more than an order of magnitude, so splitting
them by kind would leave one track idle for most of the run. This module
enumerates the jobs, assigns them to tracks by greedy longest-processing-
time packing, and runs one track per invocation.

Two ways to get parallelism, both supported:

  Separate runtimes (two Colab notebooks, or two terminals)
      Run one track in each. Assignment is deterministic given the same
      --tracks and dataset, so the tracks partition the work with no
      coordination and no shared state.

          python -m kpose_diagnostics.cli.run_parallel ... --track 0
          python -m kpose_diagnostics.cli.run_parallel ... --track 1

  One runtime
      --workers N runs a track's jobs in N concurrent subprocesses.

          python -m kpose_diagnostics.cli.run_parallel ... --workers 2

Safety properties that matter when two processes run at once:

  * Every job writes to its own output directory, so no two jobs write
    the same file.
  * Expensive KPOSE scene and grid-index caches are shared safely using
    atomic files/locks; completed checkpoints are reused across restarts.
  * A job that has completed writes ``_JOB_DONE.json`` with a content
    signature. Matching completed jobs are reused; interrupted/execution-
    failure jobs are rerun and resume from their internal checkpoints.
  * A failing job is recorded and does not stop its track.

Finally, ``--merge`` collects every result into one file.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .. import __version__
from ..runner import code_tree_hash, fingerprint_path, sha256_file


# ---------------------------------------------------------------------
# Job model
# ---------------------------------------------------------------------

@dataclass
class Job:
    """One experiment invocation.

    ``cost`` is a relative estimate used only for packing. It does not
    need to be accurate, only roughly ordered: greedy packing is robust
    to moderate error, and the done-markers make a bad estimate cost
    time rather than correctness.
    """

    jid: str
    module: str
    args: list
    cost: float
    out: Path
    result_file: str
    needs: tuple = ()

    def done_marker(self) -> Path:
        return self.out / "_JOB_DONE.json"

    def is_done(self) -> bool:
        m = self.done_marker()
        if not m.is_file():
            return False
        try:
            rec = json.loads(m.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return bool(rec.get("ok")) and (self.out / self.result_file).is_file()


def _src(base: Path, kind: str, scene: str):
    f = base / kind / f"{scene}.pkl"
    if f.is_file():
        return f
    d = base / kind / scene
    if d.is_dir():
        return d
    return None


def discover_scenes(base: Path) -> list:
    """Scenes with cells present, primary sequence first."""
    cells_dir = base / "cells"
    if not cells_dir.is_dir():
        return []
    names = set()
    for p in cells_dir.iterdir():
        names.add(p.name[:-4] if p.name.endswith(".pkl") else p.name)
    scenes = sorted(n for n in names if _src(base, "cells", n))
    primary = "2013_05_28_drive_0000_sync"
    if primary in scenes:
        scenes.remove(primary)
        scenes.insert(0, primary)
    return scenes


def build_jobs(base: Path, out: Path, scenes: list, cache_root: Path) -> list:
    """Enumerate every job, with cost estimates for packing.

    Costs are in arbitrary units calibrated so that the primary
    KITTI360Pose run with its resolution sweep -- much the largest single
    job -- dominates, which is what makes packing worth doing.
    """
    jobs: list = []
    if not scenes:
        return jobs
    primary = scenes[0]

    def cache_for(track_hint: str) -> str:
        return str(cache_root / track_hint)

    # --- template-query family (cells only, cheap) --------------------
    for i, s in enumerate(scenes):
        c = _src(base, "cells", s)
        if c is None:
            continue
        a = ["--cells", str(c), "--scene", s,
             "--out-dir", str(out / f"b1_{s}")]
        if s != primary:
            a.append("--skip-gate")
        jobs.append(Job(f"b1_{s}", "kpose_diagnostics.cli.run_gt_validity",
                        a, 3.0, out / f"b1_{s}",
                        "gt_validity_results.json", ("cells",)))

    # --- angular robustness on the primary scene ----------------------
    c = _src(base, "cells", primary)
    if c is not None:
        for t in ("90", "45", "22.5", "10"):
            jobs.append(Job(
                f"theta_{t}", "kpose_diagnostics.cli.run_gt_validity",
                ["--cells", str(c), "--scene", primary, "--theta", t,
                 "--reference-mode", ("nonfatal" if t == "90" else "off"),
                 "--out-dir", str(out / f"theta_{t}")],
                3.0, out / f"theta_{t}",
                "gt_validity_results.json", ("cells",)))

    # --- benchmark descriptions (cells + poses, expensive) ------------
    for s in scenes:
        c, p = _src(base, "cells", s), _src(base, "poses", s)
        if c is None or p is None:
            continue
        primary_run = (s == primary)
        a = ["--cells", str(c), "--poses", str(p), "--scene", s,
             "--out-dir", str(out / f"kpose_{s}")]
        if primary_run:
            a += ["--grid-sweep", "8,12,16,24"]      # convergence check
        else:
            a += ["--grid-sweep", ""]
        jobs.append(Job(f"kpose_{s}", "kpose_diagnostics.cli.run_kpose_full",
                        a, 40.0 if primary_run else 12.0,
                        out / f"kpose_{s}", "kpose_full_results.json",
                        ("cells", "poses")))

    # --- appearance ablation on the primary scene ---------------------
    c, p = _src(base, "cells", primary), _src(base, "poses", primary)
    if c is not None and p is not None:
        jobs.append(Job(
            "kpose_color", "kpose_diagnostics.cli.run_kpose_full",
            ["--cells", str(c), "--poses", str(p), "--scene", primary,
             "--use-color", "--grid-sweep", "",
             "--out-dir", str(out / "kpose_color")],
            12.0, out / "kpose_color",
            "kpose_full_results.json", ("cells", "poses")))

    return jobs


def assign(jobs: list, n_tracks: int) -> dict:
    """Greedy longest-processing-time packing.

    Sorting by descending cost and repeatedly placing the next job on
    the least-loaded track is the standard LPT heuristic; for this job
    mix it puts the one dominant run alone and fills the other track
    with the remainder.
    """
    order = sorted(jobs, key=lambda j: (-j.cost, j.jid))
    loads = [0.0] * n_tracks
    tracks: dict = {i: [] for i in range(n_tracks)}
    for j in order:
        t = min(range(n_tracks), key=lambda i: (loads[i], i))
        tracks[t].append(j)
        loads[t] += j.cost
    return tracks


# ---------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------

def _job_signature(job: Job, code_hash: str) -> str:
    blob = json.dumps({"module": job.module, "args": job.args,
                       "version": __version__, "code_hash": code_hash},
                      sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _classify_returncode(rc: int) -> str:
    if rc == 0:
        return "success"
    if rc == 4:
        return "expected_reference_mismatch"
    if rc == 3:
        return "gate_failure"
    return "execution_failure"


def run_job(job: Job, cache_dir: Path, log_dir: Path, code_hash: str, force: bool = False) -> dict:
    job.out.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    args = list(job.args)
    if job.module.endswith("run_gt_validity"):
        args += ["--cache-dir", str(cache_dir)]
    elif job.module.endswith("run_kpose_full"):
        # Shared across tracks; run_kpose_full uses atomic files + locks.
        args += ["--cache-dir", str(job.out.parent / "cache" / "kpose_shared")]
    cmd = [sys.executable, "-u", "-m", job.module] + args
    logfile = log_dir / f"{job.jid}.log"
    signature = _job_signature(job, code_hash)

    marker = job.done_marker()
    if marker.is_file() and not force:
        try:
            old = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old = {}
        if old.get("signature") == signature and old.get("completed"):
            # Completed scientific success, reference mismatch, or gate failure
            # is durable. Interrupted/execution-failure jobs are rerun.
            return {**old, "reused": True}

    t0 = time.time()
    with logfile.open("w", encoding="utf-8") as fh:
        fh.write("COMMAND: " + " ".join(cmd) + "\n\n")
        fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
    status = _classify_returncode(rc)
    completed = status in {"success", "expected_reference_mismatch", "gate_failure"}
    rec = {
        "job": job.jid, "returncode": rc, "status": status,
        "ok": status == "success", "completed": completed,
        "signature": signature, "version": __version__, "code_hash": code_hash,
        "command": cmd, "elapsed_s": round(time.time() - t0, 1),
        "log": str(logfile), "result": str(job.out / job.result_file),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = marker.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    tmp.replace(marker)
    return rec


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None
    except OSError:
        return None


def _manifest(root: Path, exclude_names=()) -> dict:
    rows = []
    excluded = set(exclude_names)
    for fp in sorted(p for p in root.rglob("*") if p.is_file()):
        if fp.name in excluded or fp.suffix == ".zip":
            continue
        rel = fp.relative_to(root).as_posix()
        rows.append({"path": rel, "size_bytes": fp.stat().st_size,
                     "sha256": sha256_file(fp)})
    return {"version": __version__, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": rows}


def merge(out: Path) -> dict:
    """Create a self-contained merge, summary, manifest, and final archive."""
    out = Path(out)
    plan = {}
    if (out / "PLAN.json").is_file():
        try:
            plan = json.loads((out / "PLAN.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            plan = {}
    expected = plan.get("jobs", [])
    if not expected:
        expected = [{"jid": p.parent.name, "out": str(p.parent),
                     "result_file": ""} for p in sorted(out.glob("*/_JOB_DONE.json"))]

    merged = {
        "schema_version": 2, "project_version": __version__,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_hash": plan.get("code_hash"),
        "input_fingerprint": plan.get("input_fingerprint"),
        "jobs": {},
    }
    summary = [f"kpose_diagnostics {__version__}",
               f"generated_utc: {merged['generated_utc']}", ""]
    for spec in expected:
        jid = spec["jid"]
        jdir = Path(spec.get("out") or (out / jid))
        marker = jdir / "_JOB_DONE.json"
        try:
            status = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {
                "job": jid, "status": "missing", "completed": False, "returncode": None}
        except json.JSONDecodeError:
            status = {"job": jid, "status": "invalid_done_marker", "completed": False, "returncode": None}
        rf = Path(status.get("result") or (jdir / spec.get("result_file", "")))
        result_obj = None
        result_error = None
        if rf.is_file():
            try:
                result_obj = json.loads(rf.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                result_error = f"invalid JSON: {exc}"
        else:
            result_error = "result file missing"
        log_path = Path(status.get("log") or (out / "logs" / f"{jid}.log"))
        diagnostics = {}
        for name in ("verification_failure.json", "reference_mismatch.json", "gate_failure.json"):
            dp = jdir / name
            if dp.is_file():
                try:
                    diagnostics[name] = json.loads(dp.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    diagnostics[name] = {"error": "invalid JSON", "raw_text": _read_text(dp)}
        merged["jobs"][jid] = {
            "status": status,
            "result": result_obj,
            "result_error": result_error,
            "log_text": _read_text(log_path),
            "diagnostics": diagnostics,
        }
        summary.append(f"{jid:<42} {status.get('status','unknown'):<30} rc={status.get('returncode')}")

    merged_path = out / "MERGED_RESULTS.json"
    tmp = merged_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
    tmp.replace(merged_path)
    (out / "RUN_SUMMARY.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    manifest = _manifest(out, exclude_names={"MANIFEST.json"})
    manifest["code_hash"] = merged.get("code_hash")
    manifest["input_fingerprint"] = merged.get("input_fingerprint")
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    archive = out.parent / f"{out.name}_SELF_CONTAINED_v{__version__}.zip"
    tmp_archive = archive.with_suffix(".tmp.zip")
    with zipfile.ZipFile(tmp_archive, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as zf:
        for fp in sorted(p for p in out.rglob("*") if p.is_file()):
            zf.write(fp, arcname=f"{out.name}/{fp.relative_to(out).as_posix()}")
    tmp_archive.replace(archive)
    merged["archive"] = str(archive.resolve())
    return merged

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, help="dataset root holding cells/ and poses/")
    p.add_argument("--out", type=Path, default=Path("results_all"))
    p.add_argument("--tracks", type=int, default=2)
    p.add_argument("--track", type=int, default=None,
                   help="which track this invocation runs (0-based)")
    p.add_argument("--workers", type=int, default=1,
                   help="concurrent subprocesses within this invocation")
    p.add_argument("--plan", action="store_true", help="print the plan and exit")
    p.add_argument("--status", action="store_true", help="print progress and exit")
    p.add_argument("--merge", action="store_true",
                   help="collect all results into MERGED_RESULTS.json")
    p.add_argument("--force", action="store_true", help="ignore done-markers")
    p.add_argument("--only", default="",
                   help="comma-separated job globs, e.g. 'kpose_*,theta_45,theta_22.5,theta_10'")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.merge:
        m = merge(out)
        ok = sum(1 for v in m["jobs"].values() if v["status"].get("ok"))
        print(f"merged {len(m['jobs'])} jobs ({ok} scientific successes) -> "
              f"{(out / 'MERGED_RESULTS.json').resolve()}")
        print(f"archive -> {m['archive']}")
        return 0

    if not args.base:
        print("--base is required unless --merge is given")
        return 2

    scenes = discover_scenes(args.base)
    if not scenes:
        print(f"no scenes found under {args.base}/cells")
        return 2
    cache_root = out / "cache"
    jobs = build_jobs(args.base, out, scenes, cache_root)
    if args.only:
        pats = [x.strip() for x in args.only.split(",") if x.strip()]
        jobs = [j for j in jobs if any(fnmatch.fnmatch(j.jid, pat) for pat in pats)]
        if not jobs:
            print(f"--only matched no jobs: {args.only}")
            return 2
    tracks = assign(jobs, max(args.tracks, 1))
    code_hash = code_tree_hash(Path(__file__).resolve().parents[2])

    # PLAN is durable shared state for two Colabs. Input content hashing can be
    # expensive, so it is cached and performed only when the plan is first made.
    plan_file = out / "PLAN.json"
    if plan_file.is_file():
        try:
            plan = json.loads(plan_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("PLAN.json is invalid; use a fresh --out or repair the plan")
            return 2
    else:
        fp_cache = out / "_fingerprints"
        input_fp = {}
        for kind in ("cells", "poses"):
            src = args.base / kind
            if src.exists():
                input_fp[kind] = fingerprint_path(src, fp_cache)
        plan = {
            "version": __version__, "code_hash": code_hash,
            "tracks": args.tracks, "n_jobs": len(jobs), "scenes": scenes,
            "only": args.only,
            "input_fingerprint": input_fp,
            "assignment": {str(t): [j.jid for j in tracks[t]] for t in sorted(tracks)},
            "jobs": [{"jid": j.jid, "module": j.module, "args": j.args,
                      "out": str(j.out), "result_file": j.result_file}
                     for j in jobs],
        }
        tmp = plan_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        tmp.replace(plan_file)

    if plan.get("tracks") not in (None, args.tracks):
        print(f"This output was planned with --tracks {plan.get('tracks')}; use that value.")
        return 2
    if (plan.get("only") or "") != (args.only or ""):
        print(f"This output was planned with --only {plan.get('only')!r}; use the same selection ")
        print("or use a fresh --out directory.")
        return 2
    if plan.get("code_hash") not in (None, code_hash):
        print("Code has changed since PLAN.json was created. Use a new --out directory ")
        print("or intentionally remove PLAN.json to re-plan with the repaired code.")
        return 2

    if args.status:
        complete = 0
        for j in jobs:
            marker = j.done_marker()
            status = "pending"
            if marker.is_file():
                try:
                    status = json.loads(marker.read_text(encoding="utf-8")).get("status", "unknown")
                except json.JSONDecodeError:
                    status = "invalid_marker"
            complete += status in {"success", "expected_reference_mismatch", "gate_failure"}
            print(f"  {j.jid:<42} {status}")
        print(f"{complete}/{len(jobs)} jobs durably complete")
        return 0

    if args.plan or args.track is None:
        print(f"scenes: {len(scenes)}  ({', '.join(scenes)})")
        print(f"jobs  : {len(jobs)}   tracks: {args.tracks}\n")
        for t in sorted(tracks):
            load = sum(j.cost for j in tracks[t])
            print(f"  track {t}  (relative load {load:.0f})")
            for j in tracks[t]:
                mark = "done" if j.is_done() else "    "
                print(f"      {mark}  {j.jid:<34} cost {j.cost:>5.0f}")
            print()
        if args.plan:
            print("Run each track in its own runtime, for example:")
            for t in sorted(tracks):
                extra = f" --only {args.only!r}" if args.only else ""
                print(f"  python -m kpose_diagnostics.cli.run_parallel "
                      f"--base {args.base} --out {out} "
                      f"--tracks {args.tracks} --track {t}{extra}")
        return 0

    mine = tracks.get(args.track, [])
    pending = mine
    print(f"track {args.track}: {len(pending)} of {len(mine)} jobs to run")
    if not pending:
        print("nothing to do")
        return 0

    cache_dir = cache_root / f"track{args.track}"
    log_dir = out / "logs"
    results = []

    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        # Threads only supervise subprocesses, so the GIL is irrelevant;
        # the work happens in separate interpreters.
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_job, j, cache_dir / f"w{i % args.workers}",
                              log_dir, code_hash, args.force): j
                    for i, j in enumerate(pending)}
            for f in futs:
                pass
            for f, j in futs.items():
                rec = f.result()
                results.append(rec)
                print(f"  {rec['status']:<30} {j.jid:<34}"
                      f" {rec.get('elapsed_s',0):>7.1f}s  rc={rec['returncode']}")
    else:
        for j in pending:
            print(f"  running {j.jid} ...", flush=True)
            rec = run_job(j, cache_dir, log_dir, code_hash, args.force)
            results.append(rec)
            print(f"  {rec['status']:<30} {j.jid:<34}"
                  f" {rec.get('elapsed_s',0):>7.1f}s  rc={rec['returncode']}")

    bad = [r for r in results if r["status"] == "execution_failure"]
    print(f"\ntrack {args.track} finished: {len(results)} completed, "
          f"{len(bad)} execution failures")
    for r in results:
        if r["status"] != "success":
            print(f"  {r['status'].upper()} {r['job']} (rc={r['returncode']}) -- see {r['log']}")
    print("\nWhen every track is done, collect with:")
    print(f"  python -m kpose_diagnostics.cli.run_parallel "
          f"--out {out} --merge")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
