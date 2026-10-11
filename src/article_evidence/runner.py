"""Orchestrate the missing/reusable v2.2.0 article-evidence experiments.

The core kpose_diagnostics source is never modified by this package.  Existing
successful jobs are reusable only after their marker, manifest hashes and
MERGED_RESULTS embedding all verify.  Everything else is rerun into separate
per-track directories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from . import (ALPHAS, EXPECTED_CELLS_TREE_SHA256, EXPECTED_CORE_HASH,
               EXPECTED_POSES_TREE_SHA256, PRIMARY_SCENE, __version__)
from .common import (atomic_json, atomic_text, copy_atomic, executor_hash,
                     read_json, rel_manifest, sha256_file, utc_now, write_csv)

from kpose_diagnostics.geometry import DIRECTIONS, s_dir, satisfies, sigma
from kpose_diagnostics.loading import load_cells_cached
from kpose_diagnostics.runner import code_tree_hash, fingerprint_path


REUSE_JOBS = (
    "kpose_2013_05_28_drive_0000_sync",
    "kpose_color",
    "theta_45",
    "theta_22.5",
    "theta_10",
)
MISSING_SCENES = (
    "2013_05_28_drive_0002_sync",
    "2013_05_28_drive_0003_sync",
    "2013_05_28_drive_0004_sync",
)
MANUSCRIPT_SCENE_REFERENCE = {
    "2013_05_28_drive_0000_sync": {"cells":2466,"objects":38094,"queries":564768,"median_V":758,"naive_ceiling":0.0034},
    "2013_05_28_drive_0002_sync": {"cells":2935,"objects":42802,"queries":700040,"median_V":977,"naive_ceiling":0.0018},
    "2013_05_28_drive_0003_sync": {"cells":447,"objects":4440,"queries":49128,"median_V":170,"naive_ceiling":0.0320},
    "2013_05_28_drive_0004_sync": {"cells":2386,"objects":35477,"queries":598384,"median_V":657,"naive_ceiling":0.0030},
}


def _src(base: Path, kind: str, scene: str) -> Path:
    f = base / kind / f"{scene}.pkl"
    if f.is_file():
        return f
    d = base / kind / scene
    if d.is_dir():
        return d
    raise FileNotFoundError(f"missing {kind} for {scene} under {base}")


def _required_files(jid: str) -> list[str]:
    if jid.startswith("theta_"):
        return ["_JOB_DONE.json", "gt_validity_results.json", "per_query.npz",
                "strata.csv", "template_v.csv", "run.log"]
    return ["_JOB_DONE.json", "kpose_full_results.json", "per_description.npz", "run.log"]


def verify_reusable(existing: Path, jid: str, expected_fp: dict, core_hash: str) -> dict:
    """Verify marker + manifest + embedded result; never trust filenames alone."""
    rec = {"job": jid, "verified": False, "checks": [], "source": str(existing)}
    try:
        manifest = read_json(existing / "MANIFEST.json")
        merged = read_json(existing / "MERGED_RESULTS.json")
    except Exception as exc:
        rec["failure"] = f"cannot read manifest/merge: {exc}"
        return rec

    def check(name, ok, detail=None):
        rec["checks"].append({"check": name, "pass": bool(ok), "detail": detail})
        return bool(ok)

    check("manifest_core_hash", manifest.get("code_hash") == core_hash,
          {"got": manifest.get("code_hash"), "expected": core_hash})
    check("merged_core_hash", merged.get("code_hash") == core_hash,
          {"got": merged.get("code_hash"), "expected": core_hash})
    check("manifest_input_fingerprint", manifest.get("input_fingerprint") == expected_fp)
    check("merged_input_fingerprint", merged.get("input_fingerprint") == expected_fp)
    job_emb = (merged.get("jobs") or {}).get(jid)
    check("embedded_job_present", isinstance(job_emb, dict))
    jdir = existing / jid
    marker_path = jdir / "_JOB_DONE.json"
    try:
        marker = read_json(marker_path)
    except Exception as exc:
        marker = {}
        check("done_marker_readable", False, str(exc))
    else:
        check("done_marker_readable", True)
        check("done_marker_success", marker.get("ok") is True and marker.get("returncode") == 0,
              marker.get("status"))
        check("done_marker_version", marker.get("version") == "2.2.0", marker.get("version"))
        check("done_marker_core_hash", marker.get("code_hash") == core_hash, marker.get("code_hash"))

    by_path = {r.get("path"): r for r in manifest.get("files", [])}
    for name in _required_files(jid):
        rel = f"{jid}/{name}"
        fp = existing / rel
        m = by_path.get(rel)
        ok = fp.is_file() and m is not None
        if ok:
            ok = int(fp.stat().st_size) == int(m.get("size_bytes", -1)) and sha256_file(fp) == m.get("sha256")
        check(f"manifest_hash:{name}", ok)

    result_name = "gt_validity_results.json" if jid.startswith("theta_") else "kpose_full_results.json"
    result_path = jdir / result_name
    try:
        result_obj = read_json(result_path)
    except Exception as exc:
        result_obj = None
        check("result_json_readable", False, str(exc))
    else:
        check("result_json_readable", True)
        if job_emb:
            check("embedded_result_exact", job_emb.get("result") == result_obj)
            check("embedded_status_exact", job_emb.get("status") == marker)
        if jid.startswith("kpose_"):
            s1 = result_obj.get("stage1_verification", {})
            cfg = result_obj.get("config", {})
            check("generator_agreement_1.0", float(s1.get("agreement", -1)) == 1.0)
            check("generator_gate_passed", s1.get("passed") is True)
            check("scientifically_valid", cfg.get("scientifically_valid") is True)
            check("result_core_hash", cfg.get("code_hash") == core_hash)
        else:
            gate = result_obj.get("reproduction_gate", {})
            check("theta_measurement_success", gate.get("passed") is True)
            # Completed non-90 theta points must explicitly be measurement-only.
            check("theta_reference_not_borrowed", gate.get("status") in {"not_applicable_theta", "not_checked"})

    rec["verified"] = all(c["pass"] for c in rec["checks"])
    return rec


def copy_verified_job(existing: Path, out: Path, jid: str) -> None:
    dst = out / "reused" / "jobs" / jid
    dst.mkdir(parents=True, exist_ok=True)
    for name in _required_files(jid):
        src = existing / jid / name
        if src.is_file():
            copy_atomic(src, dst / name)
    plog = existing / "logs" / f"{jid}.log"
    if plog.is_file():
        copy_atomic(plog, dst / "parallel.log")


def make_job(jid: str, kind: str, base: Path, out: Path, track: int | None = None) -> dict:
    scene = PRIMARY_SCENE
    if jid == "theta_90":
        return {"jid":jid,"kind":"core","module":"kpose_diagnostics.cli.run_gt_validity",
                "result_file":"gt_validity_results.json","cost":3.0,
                "args":["--cells",str(_src(base,"cells",scene)),"--scene",scene,"--theta","90",
                        "--reference-mode","off"]}
    if jid.startswith("multiscene_"):
        scene = jid[len("multiscene_"):]
        return {"jid":jid,"kind":"core","module":"kpose_diagnostics.cli.run_gt_validity",
                "result_file":"gt_validity_results.json","cost":3.0,
                "args":["--cells",str(_src(base,"cells",scene)),"--scene",scene,"--theta","90",
                        "--reference-mode","off"]}
    if jid == "template_analysis_primary":
        return {"jid":jid,"kind":"custom","module":"article_evidence.template_analysis",
                "result_file":"template_analysis_results.json","cost":12.0,
                "args":["--cells",str(_src(base,"cells",scene)),"--scene",scene]}
    if jid in {"theta_45","theta_22.5","theta_10"}:
        t = jid.split("_",1)[1]
        return {"jid":jid,"kind":"core","module":"kpose_diagnostics.cli.run_gt_validity",
                "result_file":"gt_validity_results.json","cost":3.0,
                "args":["--cells",str(_src(base,"cells",scene)),"--scene",scene,"--theta",t,
                        "--reference-mode","off"]}
    if jid == "kpose_2013_05_28_drive_0000_sync":
        return {"jid":jid,"kind":"core","module":"kpose_diagnostics.cli.run_kpose_full",
                "result_file":"kpose_full_results.json","cost":40.0,
                "args":["--cells",str(_src(base,"cells",scene)),"--poses",str(_src(base,"poses",scene)),
                        "--scene",scene,"--grid-sweep","8,12,16,24"]}
    if jid == "kpose_color":
        return {"jid":jid,"kind":"core","module":"kpose_diagnostics.cli.run_kpose_full",
                "result_file":"kpose_full_results.json","cost":12.0,
                "args":["--cells",str(_src(base,"cells",scene)),"--poses",str(_src(base,"poses",scene)),
                        "--scene",scene,"--use-color","--grid-sweep",""]}
    raise KeyError(jid)


def create_plan(args) -> dict:
    base, out, core_root = args.base.resolve(), args.out.resolve(), args.core_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    core_hash = code_tree_hash(core_root)
    if core_hash != EXPECTED_CORE_HASH:
        raise RuntimeError(f"validated core hash changed: {core_hash} != {EXPECTED_CORE_HASH}")
    exec_root = Path(__file__).resolve().parents[1]
    ex_hash = executor_hash(exec_root)
    fp_cache = out / "provenance" / "fingerprint_cache"
    cells_fp = fingerprint_path(base / "cells", fp_cache)
    poses_fp = fingerprint_path(base / "poses", fp_cache)
    if cells_fp["sha256"] != EXPECTED_CELLS_TREE_SHA256:
        raise RuntimeError(f"cells input fingerprint mismatch: {cells_fp['sha256']}")
    if poses_fp["sha256"] != EXPECTED_POSES_TREE_SHA256:
        raise RuntimeError(f"poses input fingerprint mismatch: {poses_fp['sha256']}")
    global_fp = {"cells":cells_fp,"poses":poses_fp}
    atomic_json(out / "provenance" / "INPUT_FINGERPRINT.json", global_fp)

    reuse = {}
    rerun = []
    if args.existing and args.existing.is_dir():
        for jid in REUSE_JOBS:
            vr = verify_reusable(args.existing, jid, global_fp, core_hash)
            reuse[jid] = vr
            if vr["verified"]:
                copy_verified_job(args.existing, out, jid)
            else:
                rerun.append(jid)
    else:
        for jid in REUSE_JOBS:
            reuse[jid] = {"job":jid,"verified":False,"failure":"existing result root not supplied"}
            rerun.append(jid)

    # Expected missing work. Keep the two primary-cell jobs together so their
    # per-track parsed-scene cache is reusable. Other scenes stay on track 1.
    tracks = {0:[make_job("template_analysis_primary","custom",base,out),
                 make_job("theta_90","core",base,out)],
              1:[make_job(f"multiscene_{s}","core",base,out) for s in MISSING_SCENES]}
    loads = {0:15.0, 1:9.0}
    for jid in sorted(rerun, key=lambda x: -make_job(x,"core",base,out)["cost"]):
        job = make_job(jid,"core",base,out)
        t = min(loads, key=lambda k:(loads[k],k))
        tracks[t].append(job); loads[t] += float(job["cost"])

    plan = {"schema_version":1,"executor_version":__version__,"created_utc":utc_now(),
            "base":str(base),"out":str(out),"existing":str(args.existing.resolve()) if args.existing else None,
            "core_root":str(core_root),"core_hash":core_hash,"executor_hash":ex_hash,
            "input_fingerprint":global_fp,"expected_reusable_jobs":list(REUSE_JOBS),
            "reuse_verification":reuse,"rerun_reusable_jobs":rerun,
            "tracks":{"0":tracks[0],"1":tracks[1]},"track_costs":loads,
            "science_rules":{"theta_reference_mode":"off for all sweep points; theta90 reference checked later as integrity evidence",
                             "skip_gate_allowed":False,"alpha_values":list(ALPHAS),
                             "bootstrap_unit":"source_cell","bootstrap_B":2000,
                             "permutation_B":10000,"tie_rule":"stable-first"}}
    atomic_json(out / "PLAN.json", plan)
    atomic_json(out / "REUSE_VERIFICATION.json", reuse)
    return plan


def _status_class(rc: int) -> str:
    return {0:"success",3:"gate_failure",4:"reference_mismatch"}.get(rc,"execution_failure")


def execute_job(plan: dict, spec: dict, track: int, force=False) -> dict:
    out = Path(plan["out"])
    jdir = out / "tracks" / f"track{track}" / "jobs" / spec["jid"]
    cache = out / "tracks" / f"track{track}" / "cache"
    jdir.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    marker = jdir / "_JOB_DONE.json"
    signature = hashlib.sha256(json.dumps({"spec":spec,"core_hash":plan["core_hash"],
                                           "executor_hash":plan["executor_hash"],
                                           "input_fingerprint":plan["input_fingerprint"]},
                                          sort_keys=True,default=str).encode()).hexdigest()
    if marker.is_file() and not force:
        try:
            old = read_json(marker)
            if old.get("completed") and old.get("signature") == signature and old.get("core_hash") == plan["core_hash"] and old.get("executor_hash") == plan["executor_hash"]:
                result = jdir / spec["result_file"]
                if old.get("ok") and result.is_file():
                    return {**old,"reused":True}
                if old.get("status") in {"gate_failure","reference_mismatch"}:
                    return {**old,"reused":True}
        except Exception:
            pass
    args = list(spec["args"])
    if spec["kind"] == "core":
        args += ["--out-dir",str(jdir),"--cache-dir",str(cache)]
    else:
        args += ["--out-dir",str(jdir),"--cache-dir",str(cache),
                 "--fingerprint-cache",str(out / "provenance" / "fingerprint_cache"),
                 "--global-fingerprint-json",str(out / "provenance" / "INPUT_FINGERPRINT.json"),
                 "--core-hash",plan["core_hash"],"--executor-hash",plan["executor_hash"]]
    if any(x == "--skip-gate" for x in args):
        raise RuntimeError("scientific plan contains forbidden --skip-gate")
    cmd = [sys.executable,"-u","-m",spec["module"]] + args
    log = jdir / "parallel.log"
    t0=time.time()
    with log.open("w",encoding="utf-8") as fh:
        fh.write("COMMAND: "+" ".join(cmd)+"\n\n"); fh.flush()
        rc=subprocess.call(cmd,stdout=fh,stderr=subprocess.STDOUT)
    status=_status_class(rc); completed=status in {"success","gate_failure","reference_mismatch"}
    result=jdir/spec["result_file"]
    rec={"job":spec["jid"],"track":track,"returncode":rc,"status":status,"ok":rc==0,
         "completed":completed,"core_hash":plan["core_hash"],"executor_hash":plan["executor_hash"],
         "input_fingerprint":plan["input_fingerprint"],"signature":signature,"command":cmd,
         "result":str(result),"log":str(log),"elapsed_s":round(time.time()-t0,2),"finished_utc":utc_now()}
    atomic_json(marker,rec)
    return rec


def run_track(args) -> int:
    plan=read_json(args.out/"PLAN.json")
    if int(args.track) not in (0,1): raise ValueError("track must be 0 or 1")
    results=[]
    for spec in plan["tracks"][str(args.track)]:
        rec=execute_job(plan,spec,args.track,args.force)
        results.append(rec)
        print(f"{rec['job']}: {rec['status']} rc={rec['returncode']}" + (" [reused]" if rec.get("reused") else ""),flush=True)
    atomic_json(args.out/"tracks"/f"track{args.track}"/"TRACK_STATUS.json",results)
    # Execution failures should make the notebook cell visibly nonzero; scientific
    # gate/reference failures remain recorded and are handled by final integrity.
    return 2 if any(r["status"]=="execution_failure" for r in results) else 0


def _canonical_source(plan: dict, jid: str) -> Path | None:
    out=Path(plan["out"])
    rv=(plan.get("reuse_verification") or {}).get(jid,{})
    if rv.get("verified"):
        p=out/"reused"/"jobs"/jid
        if p.is_dir(): return p
    for t in (0,1):
        p=out/"tracks"/f"track{t}"/"jobs"/jid
        m=p/"_JOB_DONE.json"
        if m.is_file():
            try:
                if read_json(m).get("ok") is True: return p
            except Exception: pass
    return None


def _job_result(plan, jid):
    src=_canonical_source(plan,jid)
    if src is None: return None,None
    name="template_analysis_results.json" if jid=="template_analysis_primary" else ("gt_validity_results.json" if jid.startswith("theta_") or jid.startswith("multiscene_") else "kpose_full_results.json")
    p=src/name
    return (read_json(p),src) if p.is_file() else (None,src)


def _make_theta_table(plan: dict, final: Path):
    rows=[]
    for t in (90,45,22.5,10):
        jid=f"theta_{float(t):g}"
        r,_=_job_result(plan,jid)
        if r is None: continue
        v=r["v_summary"]; h=r["headline"]
        rows.append({"theta_deg":t,"query_count":h["n_queries"],"min_V":v["min"],"p5_V":v["p5"],
                     "p25_V":v["p25"],"median_V":v["median"],"p75_V":v["p75"],"p95_V":v["p95"],
                     "max_V":v["max"],"mean_V":v["mean"],"sd_V":v["sd"],"V0":v["n_V0"],"V1":v["n_V1"],
                     "naive_ceiling":h["naive_ceiling_E_inv_V"],"recoverable_ceiling":h["recoverable_ceiling"],
                     "R":h["R_all"],"R_ci_lo":h["R_all_ci95"][0],"R_ci_hi":h["R_all_ci95"][1],
                     "oracle_recall_5m":"NA","oracle_recall_10m":"NA","oracle_recall_15m":"NA",
                     "oracle_applicability":"not_applicable_template_query",
                     "reference_status":r["reproduction_gate"]["status"],"reference_mode":r["reproduction_gate"]["mode"]})
    write_csv(final/"tables"/"theta_sweep_uniform.csv",rows)
    return rows


def _make_scene_table(plan: dict, final: Path):
    ids=[("2013_05_28_drive_0000_sync","theta_90")]+[(s,f"multiscene_{s}") for s in MISSING_SCENES]
    rows=[]
    for scene,jid in ids:
        r,src=_job_result(plan,jid)
        if r is None: continue
        ss=read_json(src/"load"/"scene_summary.json")
        v=r["v_summary"]; h=r["headline"]
        rows.append({"scene":scene,"cells":ss["n_cells"],"objects":ss["n_objects"],"queries":h["n_queries"],
                     "median_V":v["median"],"mean_V":v["mean"],"V0":v["n_V0"],"V1":v["n_V1"],
                     "naive_ceiling":h["naive_ceiling_E_inv_V"],"recoverable_ceiling":h["recoverable_ceiling"],
                     "R":h["R_all"],"R_ci_lo":h["R_all_ci95"][0],"R_ci_hi":h["R_all_ci95"][1]})
    if rows:
        numeric=["cells","objects","queries","median_V","mean_V","V0","V1","naive_ceiling","recoverable_ceiling","R"]
        macro={"scene":"MACRO_MEAN"}; rng={"scene":"RANGE"}
        for k in numeric:
            vals=np.asarray([float(x[k]) for x in rows]); macro[k]=float(vals.mean()); rng[k]=f"{vals.min():g}--{vals.max():g}"
        rows2=rows+[macro,rng]
    else: rows2=[]
    write_csv(final/"tables"/"multi_scene_ambiguity.csv",rows2)
    return rows


def _oracle_flat(result):
    rows=result["stage4_metric_ceiling"].get("oracle_recall",[])
    out={}
    for row in rows:
        k=str(int(float(row.get("threshold_m",row.get("k",0)))))
        if k!="0": out[k]=row
    return out


def _make_color_table(plan: dict, final: Path):
    rows=[]
    for label,jid in (("no_color","kpose_2013_05_28_drive_0000_sync"),("color","kpose_color")):
        r,_=_job_result(plan,jid)
        if r is None: continue
        s1=r["stage1_verification"]; v=r["stage2_ambiguity"]["v_summary"]; s3=r["stage3_validity"]; s4=r["stage4_metric_ceiling"]
        orows=_oracle_flat(r)
        row={"condition":label,"effect_interpretation":"descriptive",
             "verification_agreement":s1["agreement"],"n_descriptions":v["n_queries"],
             "median_V":v["median"],"mean_V":v["mean"],"V0":v["n_V0"],"V1":v["n_V1"],
             "R":s3["R"],"R_ci_lo":s3["R_ci95"][0],"R_ci_hi":s3["R_ci95"][1],
             "recoverable_ceiling":s3["recoverable_ceiling"],
             "region_area_median_m2":s4["region_area_m2"]["median"],"region_area_mean_m2":s4["region_area_m2"]["mean"],
             "region_area_p25_m2":s4["region_area_m2"]["p25"],"region_area_p75_m2":s4["region_area_m2"]["p75"],
             "primary_grid":r["config"].get("grid"),
             "grid_convergence_status":"reported" if s4.get("convergence") else "not_requested_or_empty",
             "convergence_points":len(s4.get("convergence",[]))}
        for k in (5,10,15):
            rr=orows.get(str(k),{})
            row[f"oracle_recall_{k}m"]=rr.get("oracle_recall")
            ci=rr.get("ci95",[None,None]); row[f"oracle_recall_{k}m_ci_lo"]=ci[0] if ci else None; row[f"oracle_recall_{k}m_ci_hi"]=ci[1] if ci else None
        rows.append(row)
    write_csv(final/"tables"/"color_ablation_direct.csv",rows)
    if len(rows)==2:
        diffs={"comparison":"color_minus_no_color","interpretation":"descriptive_only"}
        for k in ("median_V","mean_V","R","recoverable_ceiling","region_area_median_m2","region_area_mean_m2","oracle_recall_5m","oracle_recall_10m","oracle_recall_15m"):
            a,b=rows[0].get(k),rows[1].get(k)
            if isinstance(a,(int,float)) and isinstance(b,(int,float)): diffs[f"delta_{k}"]=float(b-a)
        write_csv(final/"tables"/"color_ablation_descriptive_differences.csv",[diffs])
    return rows


def run_integrity(plan: dict, final: Path) -> dict:
    checks=[]
    def add(name, ok, details=None, severity="required"):
        checks.append({"check":name,"status":"PASS" if ok else "MISMATCH","severity":severity,"details":details})
    test_path=final/"TEST_REPORT.json"
    if test_path.is_file():
        tr=read_json(test_path)
        add("required_test_suite_all_passed",tr.get("all_passed") is True and tr.get("core_hash")==plan["core_hash"],tr)
    else:
        add("required_test_suite_all_passed",False,"TEST_REPORT.json missing")
    # Exact direction basis and invalid-label policy are direct core properties.
    from kpose_diagnostics.kpose_predicate import direction_from_offset, normalize_direction_label
    offsets={"on-top":[.01,.01],"east":[.2,.02],"west":[-.2,.02],"north":[.02,.2],"south":[.02,-.2]}
    got={k:direction_from_offset(np.asarray(v)) for k,v in offsets.items()}
    add("five_way_kpose_mapping",got=={k:k for k in offsets},got)
    try: normalize_direction_label("definitely-invalid")
    except ValueError: invalid_ok=True
    else: invalid_ok=False
    add("invalid_direction_never_defaults_to_north",invalid_ok)
    # Accepted KPOSE results must have a perfect generator gate.
    for jid in ("kpose_2013_05_28_drive_0000_sync","kpose_color"):
        r,_=_job_result(plan,jid); ok=False; detail=None
        if r:
            s=r["stage1_verification"]; ok=(s.get("passed") is True and float(s.get("agreement",-1))==1.0 and r["config"].get("scientifically_valid") is True); detail=s
        add(f"{jid}:generator_predicate_agreement_1.0",ok,detail)
    # Mathematical theta identities. The manuscript correctly assigns dominant-axis to 45°, not 90°.
    d=np.asarray([[1,.2],[-1,.2],[.2,1],[.2,-1],[1,1],[-1,1],[1,-1],[-1,-1],[0,0]],dtype=float)
    half=(s_dir(d)>0)
    add("theta90_equals_support_of_Sdir",np.array_equal(satisfies(d,90),half))
    dx,dy=d[:,0],d[:,1]; ax,ay=np.abs(dx),np.abs(dy)
    dom=np.stack([(dy<0)&(ay>=ax),(dy>0)&(ay>=ax),(dx>0)&(ax>=ay),(dx<0)&(ax>=ay)],axis=1)
    add("theta45_equals_dominant_axis_cone",np.array_equal(satisfies(d,45),dom))
    add("theta90_equals_dominant_axis_rule",False,
        "Not a valid identity: v2.2.0/manuscript define theta=90 as half-plane support and theta=45 as dominant-axis. Kept visible rather than silently changing the predicate.",severity="specification_conflict_not_gate")
    # The theta=90 job is intentionally run with reference_mode=off to match
    # the already-completed theta points. We nevertheless verify the registered
    # v2.2.0 theta=90 reference here as an independent integrity check.
    t90r,t90src=_job_result(plan,"theta_90")
    if t90r:
        from kpose_diagnostics.cli.run_gt_validity import PUBLISHED, TOLERANCE
        pub=PUBLISHED.get(PRIMARY_SCENE,{})
        ss=read_json(t90src/"load"/"scene_summary.json")
        vs=t90r["v_summary"]; hh=t90r["headline"]
        tpl_count=None
        tv=t90src/"template_v.csv"
        if tv.is_file():
            import csv
            with tv.open(newline="",encoding="utf-8") as fh: tpl_count=sum(1 for _ in csv.DictReader(fh))
        measured={"n_cells":ss["n_cells"],"n_objects":ss["n_objects"],"n_queries":hh["n_queries"],
                  "n_templates_realised":tpl_count,"median_V":vs["median"],"mean_V":vs["mean"],
                  "n_V1":vs["n_V1"],"n_V0":vs["n_V0"],"naive_ceiling":hh["naive_ceiling_E_inv_V"]}
        oks=[]
        for k,want in pub.items():
            got=measured.get(k); tol=TOLERANCE.get(k,0)
            oks.append(got is not None and abs(float(got)-float(want))<=tol)
        add("theta90_registered_reference_consistency",all(oks),{"measured":measured,"registered":pub,"tolerance":TOLERANCE,
            "note":"independent check; run itself used reference_mode=off for sweep uniformity"})
    else:
        add("theta90_registered_reference_consistency",False,"theta_90 result missing")

    # Cross-output query/scene totals and article-declared references.
    scene_rows=[]
    ids=[(PRIMARY_SCENE,"theta_90")]+[(s,f"multiscene_{s}") for s in MISSING_SCENES]
    for scene,jid in ids:
        r,src=_job_result(plan,jid)
        if r is None:
            add(f"scene_result_present:{scene}",False); continue
        ss=read_json(src/"load"/"scene_summary.json"); ref=MANUSCRIPT_SCENE_REFERENCE[scene]
        measured={"cells":ss["n_cells"],"objects":ss["n_objects"],"queries":r["headline"]["n_queries"]}
        add(f"scene_totals:{scene}",all(measured[k]==ref[k] for k in measured),{"measured":measured,"manuscript":ref})
    # Per-query arrays must reproduce aggregate V/R exactly for every theta scene.
    for scene,jid in ids+[(PRIMARY_SCENE,"theta_45"),(PRIMARY_SCENE,"theta_22.5"),(PRIMARY_SCENE,"theta_10")]:
        r,src=_job_result(plan,jid)
        if not r: continue
        z=np.load(src/"per_query.npz"); v=z["v"]; gt=z["gt_valid"]
        ok=(len(v)==r["headline"]["n_queries"] and np.isclose(v.mean(),r["v_summary"]["mean"]) and int((v==0).sum())==r["v_summary"]["n_V0"] and int((v==1).sum())==r["v_summary"]["n_V1"] and np.isclose(gt.mean(),r["headline"]["R_all"]))
        add(f"per_query_vs_aggregate:{jid}",ok)
    # Template analysis arrays versus its aggregates and theta90 arrays.
    tr,tsrc=_job_result(plan,"template_analysis_primary"); t90,t90src=_job_result(plan,"theta_90")
    if tr:
        z=np.load(tsrc/"per_query_all_alpha.npz"); rows=tr["alpha_sweep"]
        ok=(z["T1"].shape[0]==len(ALPHAS) and z["T1"].shape[1]==tr["population"]["queries"] and all(np.isclose(z["T1"][i].mean(),rows[i]["T1"]) and np.isclose(z["T2"][i].mean(),rows[i]["T2"]) for i in range(len(rows))))
        add("template_per_query_vs_aggregate",ok)
        if t90:
            z90=np.load(t90src/"per_query.npz")
            add("template_analysis_equals_theta90_V",np.array_equal(z["V"],z90["v"]))
            add("template_analysis_equals_theta90_generating_validity",np.array_equal(z["generating_cell_valid"],z90["gt_valid"]))
    else: add("template_analysis_present",False)
    # Existing reused result verification remains part of the integrity record.
    for jid,vr in plan.get("reuse_verification",{}).items():
        if jid not in plan.get("rerun_reusable_jobs",[]): add(f"reuse_hash_embedding:{jid}",vr.get("verified") is True,vr)
    required_bad=[c for c in checks if c["severity"]=="required" and c["status"]!="PASS"]
    report={"generated_utc":utc_now(),"status":"PASS" if not required_bad else "MISMATCH",
            "required_failures":len(required_bad),"checks":checks,
            "theta_definition_note":"theta=90 is half-plane support; theta=45 is the dominant-axis cone. The contrary requested check is retained as a visible specification conflict, not used to alter science."}
    atomic_json(final/"SCIENTIFIC_INTEGRITY_CHECKS.json",report)
    write_csv(final/"tables"/"scientific_integrity_checks.csv",checks)
    return report


def collect_jobs(plan: dict, final: Path) -> list[dict]:
    final_jobs=final/"jobs"; final_jobs.mkdir(parents=True,exist_ok=True)
    jids=list(REUSE_JOBS)+["theta_90","template_analysis_primary"]+[f"multiscene_{s}" for s in MISSING_SCENES]
    rows=[]
    for jid in jids:
        src=_canonical_source(plan,jid)
        if src is None:
            rows.append({"job":jid,"status":"MISSING"}); continue
        dst=final_jobs/jid
        if dst.exists(): shutil.rmtree(dst)
        shutil.copytree(src,dst)
        result_name="template_analysis_results.json" if jid=="template_analysis_primary" else ("gt_validity_results.json" if jid.startswith("theta_") or jid.startswith("multiscene_") else "kpose_full_results.json")
        rows.append({"job":jid,"status":"PASS","source":str(src),"result":str(dst/result_name)})
    return rows


def merge_embedded(plan: dict, final: Path, job_rows: list[dict]) -> dict:
    jobs={}
    for row in job_rows:
        jid=row["job"]; jdir=final/"jobs"/jid
        if row["status"]!="PASS": jobs[jid]={"status":"missing"}; continue
        result_files=[p for p in jdir.glob("*_results.json") if p.name not in {"SCIENTIFIC_INTEGRITY_CHECKS.json"}]
        if jid=="template_analysis_primary": result_files=[jdir/"template_analysis_results.json"]
        result=read_json(result_files[0]) if result_files and result_files[0].is_file() else None
        logs={}
        for lp in (jdir/"run.log",jdir/"parallel.log"):
            if lp.is_file(): logs[lp.name]=lp.read_text(encoding="utf-8",errors="replace")
        marker=None
        for mp in (jdir/"_JOB_DONE.json",):
            if mp.is_file(): marker=read_json(mp); break
        jobs[jid]={"status":marker,"result":result,"logs":logs,
                   "files":[{"name":p.name,"size":p.stat().st_size,"sha256":sha256_file(p)} for p in sorted(jdir.iterdir()) if p.is_file()]}
    merged={"schema_version":1,"project_version":"article_evidence_v220","generated_utc":utc_now(),
            "core_version":"kpose_diagnostics/2.2.0","core_hash":plan["core_hash"],
            "executor_hash":plan["executor_hash"],"input_fingerprint":plan["input_fingerprint"],"jobs":jobs}
    atomic_json(final/"MERGED_RESULTS.json",merged)
    return merged


def finalize(args) -> int:
    plan=read_json(args.out/"PLAN.json")
    final=args.out/"FINAL_EVIDENCE"
    if final.exists(): shutil.rmtree(final)
    final.mkdir(parents=True)
    copy_atomic(args.out/"PLAN.json",final/"PLAN.json")
    copy_atomic(args.out/"REUSE_VERIFICATION.json",final/"REUSE_VERIFICATION.json")
    copy_atomic(args.out/"provenance"/"INPUT_FINGERPRINT.json",final/"INPUT_FINGERPRINT.json")
    # Preserve exact source archive used for the real scientific jobs.
    if args.core_archive and args.core_archive.is_file(): copy_atomic(args.core_archive,final/args.core_archive.name)
    if args.executor_archive and args.executor_archive.is_file(): copy_atomic(args.executor_archive,final/args.executor_archive.name)
    test_src=args.out/"TEST_REPORT.json"
    if test_src.is_file(): copy_atomic(test_src,final/"TEST_REPORT.json")
    if (args.out/"TEST_REPORT.txt").is_file(): copy_atomic(args.out/"TEST_REPORT.txt",final/"TEST_REPORT.txt")

    job_rows=collect_jobs(plan,final)
    # Consolidate manuscript-facing CSVs without removing the original per-job copies.
    tables=final/"tables"; tables.mkdir(parents=True,exist_ok=True)
    tdir=final/"jobs"/"template_analysis_primary"
    if tdir.is_dir():
        for fp in sorted(tdir.glob("*.csv")):
            copy_atomic(fp,tables/fp.name)
    for jid in ("theta_90","theta_45","theta_22.5","theta_10"):
        jdir=final/"jobs"/jid
        for name in ("strata.csv","template_v.csv"):
            if (jdir/name).is_file(): copy_atomic(jdir/name,tables/f"{jid}_{name}")
    _make_theta_table(plan,final); _make_scene_table(plan,final); _make_color_table(plan,final)
    integrity=run_integrity(plan,final)
    merged=merge_embedded(plan,final,job_rows)
    summary=["ARTICLE EVIDENCE v2.2.0","generated_utc: "+utc_now(),"core_hash: "+plan["core_hash"],
             "executor_hash: "+plan["executor_hash"],"integrity: "+integrity["status"],""]
    summary += [f"{r['job']:<48} {r['status']}" for r in job_rows]
    atomic_text(final/"RUN_SUMMARY.txt","\n".join(summary)+"\n")
    atomic_json(final/"EVIDENCE_READY_FOR_ARTICLE_AUDIT.json",
                {"ready":integrity["status"]=="PASS" and all(r["status"]=="PASS" for r in job_rows),
                 "article_audit_generated":False,
                 "reason":"User requested completion of code/results first. ARTICLE_* audit files are intentionally deferred until this evidence archive is returned for manuscript-number auditing.",
                 "generated_utc":utc_now()})
    manifest={"schema_version":1,"generated_utc":utc_now(),"core_hash":plan["core_hash"],
              "executor_hash":plan["executor_hash"],"input_fingerprint":plan["input_fingerprint"],
              "files":rel_manifest(final,{"MANIFEST.json"})}
    atomic_json(final/"MANIFEST.json",manifest)
    archive=args.out/f"{args.out.name}_FINAL_EVIDENCE_v2.2.0.zip"
    tmp=archive.with_suffix(".tmp.zip")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
        for fp in sorted(p for p in final.rglob("*") if p.is_file()):
            zf.write(fp,arcname=f"FINAL_EVIDENCE/{fp.relative_to(final).as_posix()}")
    os.replace(tmp,archive)
    atomic_json(args.out/"FINAL_ARCHIVE.json",{"path":str(archive),"sha256":sha256_file(archive),"size_bytes":archive.stat().st_size})
    print(archive)
    return 0 if integrity["status"]=="PASS" and all(r["status"]=="PASS" for r in job_rows) else 3


def run_tests(args) -> int:
    out=args.out; out.mkdir(parents=True,exist_ok=True)
    commands=[
      ("compileall",[sys.executable,"-m","compileall","-q","."] ,args.core_root),
      ("test_core",[sys.executable,"-m","kpose_diagnostics.tests.test_core"],args.core_root),
      ("smoke_test",[sys.executable,"-m","kpose_diagnostics.tests.smoke_test"],args.core_root),
      ("kpose_predicate_e2e",[sys.executable,"-m","kpose_diagnostics.tests.kpose_predicate_e2e"],args.core_root),
      ("regressions_v220",[sys.executable,"-m","unittest","-v","kpose_diagnostics.tests.test_regressions_v220"],args.core_root),
      ("executor_regressions",[sys.executable,"-m","unittest","-v","article_evidence.tests.test_executor"],Path(__file__).resolve().parents[1]),
    ]
    rows=[]; text=[]
    env=dict(os.environ)
    pp=[str(args.core_root),str(Path(__file__).resolve().parents[1])]
    if env.get("PYTHONPATH"): pp.append(env["PYTHONPATH"])
    env["PYTHONPATH"]=os.pathsep.join(pp)
    for name,cmd,cwd in commands:
        t=time.time(); p=subprocess.run(cmd,cwd=cwd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        rows.append({"name":name,"returncode":p.returncode,"passed":p.returncode==0,"elapsed_s":round(time.time()-t,2),"command":cmd})
        text.append(f"===== {name} rc={p.returncode} =====\n{p.stdout}\n")
    report={"generated_utc":utc_now(),"all_passed":all(r["passed"] for r in rows),"tests":rows,
            "core_hash":code_tree_hash(args.core_root),"expected_core_hash":EXPECTED_CORE_HASH}
    atomic_json(out/"TEST_REPORT.json",report); atomic_text(out/"TEST_REPORT.txt","\n".join(text))
    print(json.dumps(report,indent=2))
    return 0 if report["all_passed"] and report["core_hash"]==EXPECTED_CORE_HASH else 2


def status(args) -> int:
    if not (args.out/"PLAN.json").is_file(): print("PLAN.json missing"); return 2
    plan=read_json(args.out/"PLAN.json")
    for t in (0,1):
        print(f"Track {t}:")
        for spec in plan["tracks"][str(t)]:
            p=args.out/"tracks"/f"track{t}"/"jobs"/spec["jid"]/"_JOB_DONE.json"
            if p.is_file():
                r=read_json(p); print(f"  {spec['jid']:<48} {r['status']} rc={r['returncode']}")
            else: print(f"  {spec['jid']:<48} pending")
    print("Reuse:")
    for jid,v in plan["reuse_verification"].items(): print(f"  {jid:<48} {'VERIFIED' if v.get('verified') else 'RERUN SCHEDULED'}")
    return 0


def parser():
    p=argparse.ArgumentParser(); sp=p.add_subparsers(dest="command",required=True)
    common=argparse.ArgumentParser(add_help=False)
    common.add_argument("--out",type=Path,required=True)
    q=sp.add_parser("plan",parents=[common]); q.add_argument("--base",type=Path,required=True); q.add_argument("--existing",type=Path); q.add_argument("--core-root",type=Path,required=True)
    q=sp.add_parser("run-track",parents=[common]); q.add_argument("--track",type=int,required=True); q.add_argument("--force",action="store_true")
    q=sp.add_parser("status",parents=[common])
    q=sp.add_parser("tests",parents=[common]); q.add_argument("--core-root",type=Path,required=True)
    q=sp.add_parser("finalize",parents=[common]); q.add_argument("--core-archive",type=Path); q.add_argument("--executor-archive",type=Path)
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    if a.command=="plan": create_plan(a); return 0
    if a.command=="run-track": return run_track(a)
    if a.command=="status": return status(a)
    if a.command=="tests": return run_tests(a)
    if a.command=="finalize": return finalize(a)
    return 2

if __name__=="__main__": raise SystemExit(main())
