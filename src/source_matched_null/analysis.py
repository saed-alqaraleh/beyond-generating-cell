
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import core


DEFAULT_CONFIG = {
    "version": "source_membership_matched_semantic_null_v2_1",
    "base_seed": 20260905,
    "bootstrap_seed": 20260901,
    "chains": 4,
    "retained_per_chain": 250,
    "burn_edge_multiples": 160,
    "thin_edge_multiples": 16,
    "checkpoint_every": 5,
    "bootstrap_reps": 10000,
    "rhat_max": 1.10,
    "minimum_total_repetitions": 1000,
    "primary_subset": "V>1",
    "primary_query_count": 4794,
    "diagnostic_subset": "V>1 and source_supported",
    "diagnostic_query_count": 3780,
    "preserve_source_membership": True,
    "preserve_query_cardinality": True,
    "preserve_per_query_scene_cardinality": True,
    "preserve_candidate_column_degrees": True,
    "same_null_sets_for_both_models": True,
}


def ensure_runtime_deps():
    missing = []
    for mod, pkg in [
        ("pandas", "pandas"),
        ("pyarrow", "pyarrow"),
        ("matplotlib", "matplotlib"),
    ]:
        try:
            __import__(mod)
        except Exception:
            missing.append(pkg)
    if missing:
        print("Installing missing lightweight analysis dependencies:", ", ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


def code_hash(code_root: Path) -> str:
    parts = []
    for p in sorted(Path(code_root).rglob("*.py")):
        parts.append(p.relative_to(code_root).as_posix())
        parts.append(core.sha256_file(p))
    return core.sha256_text("\n".join(parts) + "\n")


def load_config(path: Path | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path is not None:
        user = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg.update(user)
    if cfg["chains"] < 4:
        raise RuntimeError("At least four independent chains are required")
    if cfg["retained_per_chain"] < 250:
        raise RuntimeError("At least 250 retained realizations per chain are required")
    if cfg["chains"] * cfg["retained_per_chain"] < 1000:
        raise RuntimeError("At least 1,000 total retained realizations are required")
    if cfg["bootstrap_reps"] != 10000:
        raise RuntimeError("The confirmatory clustered bootstrap must use exactly 10,000 replicates")
    return cfg


def write_preflight_tables(run_root: Path, data: dict, evidence_zip: Path, legacy_zip: Path):
    rows = [
        {"item": "parent_evidence_zip", "value": str(evidence_zip), "sha256": core.sha256_file(evidence_zip)},
        {"item": "color_valid_sets", "value": str(data["valid_color_sha256"]), "sha256": data["valid_color_sha256"]},
        {"item": "candidate_universe", "value": core.EXPECTED_CANDIDATE_COUNT, "sha256": core.EXPECTED_CANDIDATE_UNIVERSE_SHA256},
        {"item": "primary_queries_V_gt_1", "value": len(data["query_id"]), "sha256": ""},
        {"item": "source_supported_primary_queries", "value": int(data["source_supported"].sum()), "sha256": ""},
        {"item": "source_unsupported_primary_queries", "value": int((~data["source_supported"]).sum()), "sha256": ""},
        {"item": "legacy_unconditional_null_archive", "value": str(legacy_zip) if legacy_zip else "NOT_PROVIDED", "sha256": core.sha256_file(legacy_zip) if legacy_zip else ""},
    ]
    core.write_csv(run_root / "TABLE_input_integrity.csv", rows)


def assemble_results(
    *,
    prepared: dict,
    chain_dir: Path,
    cfg: dict,
    output_dir: Path,
    config_hash: str,
    input_hash: str,
):
    chain_npz = []
    chain_meta = []
    for cid in range(cfg["chains"]):
        j = chain_dir / f"chain_{cid:02d}" / f"CHAIN_{cid:02d}_FINAL.json"
        n = chain_dir / f"chain_{cid:02d}" / f"CHAIN_{cid:02d}_FINAL.npz"
        if not j.is_file() or not n.is_file():
            raise RuntimeError(f"Missing completed chain {cid}")
        meta = json.loads(j.read_text(encoding="utf-8"))
        if meta.get("status") != "PASS":
            raise RuntimeError(f"Chain {cid} is not PASS")
        if meta.get("config_hash") != config_hash or meta.get("input_hash") != input_hash:
            raise RuntimeError(f"Chain {cid} provenance mismatch")
        if meta.get("output_sha256") != core.sha256_file(n):
            raise RuntimeError(f"Chain {cid} output hash mismatch")
        chain_meta.append(meta)
        chain_npz.append(n)

    reps = cfg["retained_per_chain"]
    n = len(prepared["query_id"])
    ns = int(prepared["source_supported"].sum())
    rep_full_num = np.zeros((cfg["chains"], reps, 2, 3), dtype=np.int32)
    rep_supported_num = np.zeros_like(rep_full_num)
    query_hit_sums = np.zeros((n, 2, 3), dtype=np.uint32)
    overlap = np.zeros((cfg["chains"], reps), dtype=np.float32)

    for cid, p in enumerate(chain_npz):
        with np.load(p, allow_pickle=False) as z:
            rep_full_num[cid] = z["rep_full_num"]
            rep_supported_num[cid] = z["rep_supported_num"]
            query_hit_sums += z["query_hit_sums"].astype(np.uint32)
            overlap[cid] = z["overlap_seq"]

    total_reps = cfg["chains"] * reps
    if total_reps < cfg["minimum_total_repetitions"]:
        raise RuntimeError("Too few retained null realizations")

    rep_full = rep_full_num / n
    rep_supported = rep_supported_num / ns
    null_prob = query_hit_sums / total_reps

    diagnostics = core.chain_diagnostics(rep_full)
    for row in diagnostics:
        if not math.isfinite(row["split_Rhat"]) or row["split_Rhat"] > cfg["rhat_max"]:
            raise RuntimeError(
                f"Chain convergence gate failed for {row['model']}@{row['K']}: "
                f"split-Rhat={row['split_Rhat']}"
            )

    overlap_rhat = core.split_rhat(overlap)
    overlap_ess = float(sum(core.autocorr_ess_one_chain(c) for c in overlap))
    if not math.isfinite(overlap_rhat) or overlap_rhat > cfg["rhat_max"]:
        raise RuntimeError(
            f"Structural overlap convergence gate failed: split-Rhat={overlap_rhat}"
        )

    observed_source = prepared["observed_source"].astype(np.uint8)
    observed_semantic = prepared["observed_semantic"].astype(np.uint8)
    supported = prepared["source_supported"].astype(bool)

    # Mandatory full source-membership lower-bound checks over every repetition.
    exact_min_num = core.exact_supported_source_lower_bounds(observed_source, supported)
    for cid in range(cfg["chains"]):
        for r in range(reps):
            if np.any(rep_full_num[cid, r] < exact_min_num):
                raise RuntimeError("A retained null realization violated exact source-membership lower bounds")
            if np.any(rep_supported_num[cid, r] < exact_min_num):
                raise RuntimeError("Diagnostic supported-subset null fell below source Recall")
            for (model, k), bound in core.CONSERVATIVE_FULL_LOWER_BOUNDS.items():
                mi = core.MODEL_NAMES.index(model)
                ki = core.KS.index(k)
                if rep_full[cid, r, mi, ki] + 1e-15 < bound:
                    raise RuntimeError(f"Conservative lower-bound assertion failed for {model}@{k}")

    # Query-level null probabilities, retained for transparent bootstrap/audit.
    qprob_rows = []
    for i, qid in enumerate(prepared["query_id"].astype(str)):
        row = {
            "query_id": qid,
            "scene_code": int(prepared["query_scene_code"][i]),
            "source_group_id": str(prepared["source_group_id"][i]),
            "V": int(prepared["V"][i]),
            "source_supported": bool(supported[i]),
        }
        for mi, model in enumerate(core.MODEL_NAMES):
            for ki, k in enumerate(core.KS):
                row[f"{model}_source_hit_{k}"] = int(observed_source[i, mi, ki])
                row[f"{model}_semantic_hit_{k}"] = int(observed_semantic[i, mi, ki])
                row[f"{model}_corrected_null_probability_{k}"] = float(null_prob[i, mi, ki])
        qprob_rows.append(row)

    import pandas as pd
    qprob_df = pd.DataFrame(qprob_rows)
    qprob_path = output_dir / "QUERY_LEVEL_NULL_PROBABILITIES.parquet"
    tmp = qprob_path.with_name(qprob_path.name + f".tmp.{os.getpid()}")
    qprob_df.to_parquet(tmp, index=False, engine="pyarrow")
    os.replace(tmp, qprob_path)
    qprob_df.to_csv(output_dir / "QUERY_LEVEL_NULL_PROBABILITIES.csv.gz", index=False, compression="gzip")

    # Bootstrap six primary semantic-minus-corrected-null effects.
    point, boot = core.bootstrap_effects(
        scene_codes=prepared["query_scene_code"],
        source_group_ids=prepared["source_group_id"],
        observed_semantic=observed_semantic,
        null_prob=null_prob,
        reps=cfg["bootstrap_reps"],
        seed=cfg["bootstrap_seed"],
    )
    pw_lo, pw_hi = core.pointwise_intervals(boot)
    sim_lo, sim_hi, sim_q = core.simultaneous_intervals(point, boot)

    # Descriptive bootstrap sign fractions only.  These are NOT hypothesis-test
    # p-values because the ordinary cluster bootstrap is not null-centered.
    boot_nonpositive_fraction = [
        (1 + int((boot[:, j] <= 0).sum())) / (len(boot) + 1)
        for j in range(6)
    ]

    primary_rows = []
    simultaneous_rows = []
    diagnostic_rows = []
    null_repetition_rows = []

    primary_cluster_count = len(set(prepared["source_group_id"].astype(str).tolist()))
    diagnostic_cluster_count = len(
        set(prepared["source_group_id"][supported].astype(str).tolist())
    )
    chain_seeds = [
        core.derive_chain_seed(cfg["base_seed"], cid)
        for cid in range(cfg["chains"])
    ]

    flat_full = rep_full.reshape(total_reps, 2, 3)
    flat_supported = rep_supported.reshape(total_reps, 2, 3)
    observed_source_rate = observed_source.mean(axis=0)
    observed_sem_rate = observed_semantic.mean(axis=0)
    source_supported_source_rate = observed_source[supported].mean(axis=0)
    source_supported_sem_rate = observed_semantic[supported].mean(axis=0)

    flat_index = 0
    for mi, model in enumerate(core.MODEL_NAMES):
        for ki, k in enumerate(core.KS):
            null_values = flat_full[:, mi, ki]
            diag_null_values = flat_supported[:, mi, ki]
            null_mean = float(null_values.mean())
            sem = float(observed_sem_rate[mi, ki])
            src = float(observed_source_rate[mi, ki])
            effect = sem - null_mean
            mc_p = (1 + int((null_values >= sem).sum())) / (total_reps + 1)
            primary_rows.append({
                "condition": "source_membership_matched_degree_preserving_null",
                "subset": "V>1",
                "model": model,
                "K": k,
                "query_count": n,
                "source_cell_cluster_count": primary_cluster_count,
                "source_supported_query_count": ns,
                "source_recall_numerator": int(observed_source[:, mi, ki].sum()),
                "source_recall_denominator": n,
                "source_recall": src,
                "semantic_recall_numerator": int(observed_semantic[:, mi, ki].sum()),
                "semantic_recall_denominator": n,
                "semantic_recall": sem,
                "corrected_null_recall_mean": null_mean,
                "corrected_null_recall_mc_q025": float(np.quantile(null_values, 0.025)),
                "corrected_null_recall_mc_q975": float(np.quantile(null_values, 0.975)),
                "corrected_null_mcse_naive": float(null_values.std(ddof=1) / math.sqrt(total_reps)),
                "semantic_minus_corrected_null": effect,
                "effect_percentage_points": 100.0 * effect,
                "pointwise_ci_low": float(pw_lo[flat_index]),
                "pointwise_ci_high": float(pw_hi[flat_index]),
                "simultaneous_ci_low": float(sim_lo[flat_index]),
                "simultaneous_ci_high": float(sim_hi[flat_index]),
                "simultaneous_critical_absolute_deviation": sim_q,
                "mc_tail_p_semantic_vs_null": mc_p,
                "bootstrap_nonpositive_fraction_descriptive": boot_nonpositive_fraction[flat_index],
                "bootstrap_sign_note": "descriptive only; not a null-centered p-value",
                "null_repetitions": total_reps,
                "bootstrap_replicates": cfg["bootstrap_reps"],
                "bootstrap_seed": cfg["bootstrap_seed"],
                "null_base_seed": cfg["base_seed"],
                "chain_seeds": ";".join(str(x) for x in chain_seeds),
                "same_randomized_sets_for_both_models": True,
            })
            simultaneous_rows.append({
                "model": model,
                "K": k,
                "effect": effect,
                "pointwise_95_low": float(pw_lo[flat_index]),
                "pointwise_95_high": float(pw_hi[flat_index]),
                "familywise_simultaneous_95_low": float(sim_lo[flat_index]),
                "familywise_simultaneous_95_high": float(sim_hi[flat_index]),
                "familywise_interval_method": "joint max absolute centered bootstrap deviation over six primary effects",
                "bootstrap_replicates": cfg["bootstrap_reps"],
                "seed": cfg["bootstrap_seed"],
            })

            diagnostic_rows.append({
                "condition": "source_membership_matched_degree_preserving_null",
                "subset": "V>1_and_source_supported",
                "model": model,
                "K": k,
                "query_count": ns,
                "source_cell_cluster_count": diagnostic_cluster_count,
                "source_recall_numerator": int(observed_source[supported, mi, ki].sum()),
                "source_recall": float(source_supported_source_rate[mi, ki]),
                "semantic_recall_numerator": int(observed_semantic[supported, mi, ki].sum()),
                "semantic_recall": float(source_supported_sem_rate[mi, ki]),
                "corrected_null_recall_mean": float(diag_null_values.mean()),
                "corrected_null_recall_min_over_repetitions": float(diag_null_values.min()),
                "corrected_null_recall_max_over_repetitions": float(diag_null_values.max()),
                "mandatory_null_ge_source_recall": bool(
                    diag_null_values.min() + 1e-15 >= source_supported_source_rate[mi, ki]
                ),
                "null_repetitions": total_reps,
            })
            flat_index += 1

    for cid in range(cfg["chains"]):
        for r in range(reps):
            rr = {
                "chain_id": cid,
                "retained_index": r,
                "global_repetition_index": cid * reps + r,
                "free_edge_overlap_fraction": float(overlap[cid, r]),
            }
            for mi, model in enumerate(core.MODEL_NAMES):
                for ki, k in enumerate(core.KS):
                    rr[f"{model}_null_recall_{k}"] = float(rep_full[cid, r, mi, ki])
                    rr[f"{model}_supported_null_recall_{k}"] = float(
                        rep_supported[cid, r, mi, ki]
                    )
            null_repetition_rows.append(rr)

    core.write_csv(output_dir / "TABLE_source_membership_matched_null_primary.csv", primary_rows)
    # Compatibility name, but the condition column makes clear this is the corrected v2 primary.
    core.write_csv(output_dir / "TABLE_null_control_primary.csv", primary_rows)
    core.write_csv(output_dir / "TABLE_source_supported_diagnostic.csv", diagnostic_rows)
    core.write_csv(output_dir / "TABLE_null_repetitions.csv", null_repetition_rows)
    core.write_csv(output_dir / "TABLE_simultaneous_inference.csv", simultaneous_rows)
    core.write_csv(output_dir / "TABLE_chain_diagnostics.csv", diagnostics)
    core.write_csv(
        output_dir / "TABLE_bootstrap_sign_diagnostics.csv",
        [
            {
                "model": r["model"],
                "K": r["K"],
                "bootstrap_nonpositive_fraction_descriptive": r["bootstrap_nonpositive_fraction_descriptive"],
                "interpretation": "descriptive bootstrap sign fraction; not a formal p-value",
            }
            for r in primary_rows
        ],
    )

    # Scene robustness and prespecified V-stratified descriptive corrected-null effects.
    scene_rows = []
    for sc in sorted(set(prepared["query_scene_code"].tolist())):
        mm = prepared["query_scene_code"] == sc
        for mi, model in enumerate(core.MODEL_NAMES):
            for ki, k in enumerate(core.KS):
                scene_rows.append({
                    "scene_code": int(sc),
                    "model": model,
                    "K": k,
                    "query_count": int(mm.sum()),
                    "source_recall": float(observed_source[mm, mi, ki].mean()),
                    "semantic_recall": float(observed_semantic[mm, mi, ki].mean()),
                    "corrected_null_recall": float(null_prob[mm, mi, ki].mean()),
                    "semantic_minus_corrected_null": float(
                        observed_semantic[mm, mi, ki].mean() - null_prob[mm, mi, ki].mean()
                    ),
                })
    core.write_csv(output_dir / "TABLE_scene_robustness.csv", scene_rows)

    def vbin(v):
        v=int(v)
        if v == 2: return "V=2"
        if 3 <= v <= 5: return "V=3-5"
        if 6 <= v <= 10: return "V=6-10"
        if 11 <= v <= 20: return "V=11-20"
        return "V>20"
    vlabels = np.asarray([vbin(v) for v in prepared["V"]], dtype=object)
    vrows=[]
    for label in ("V=2","V=3-5","V=6-10","V=11-20","V>20"):
        mm=vlabels==label
        if not mm.any():
            continue
        for mi, model in enumerate(core.MODEL_NAMES):
            for ki, k in enumerate(core.KS):
                vrows.append({
                    "V_bin": label,
                    "model": model,
                    "K": k,
                    "query_count": int(mm.sum()),
                    "source_recall": float(observed_source[mm, mi, ki].mean()),
                    "semantic_recall": float(observed_semantic[mm, mi, ki].mean()),
                    "corrected_null_recall": float(null_prob[mm, mi, ki].mean()),
                    "semantic_minus_corrected_null": float(
                        observed_semantic[mm, mi, ki].mean() - null_prob[mm, mi, ki].mean()
                    ),
                })
    core.write_csv(output_dir / "TABLE_V_stratified_null_effects.csv", vrows)

    # Source-membership invariant report over every query x repetition.
    source_pair_checks = sum(int(x["source_membership_query_rep_checks"]) for x in chain_meta)
    invariant = {
        "status": "PASS",
        "primary_query_count": n,
        "source_supported_query_count": ns,
        "source_unsupported_query_count": n - ns,
        "retained_repetitions": total_reps,
        "source_membership_query_repetition_checks": source_pair_checks,
        "expected_source_membership_query_repetition_checks": n * total_reps,
        "source_membership_mismatches": sum(int(x["source_membership_mismatches"]) for x in chain_meta),
        "row_scene_degree_mismatches": sum(int(x["row_scene_degree_mismatches"]) for x in chain_meta),
        "candidate_column_degree_mismatches": sum(int(x["candidate_column_degree_mismatches"]) for x in chain_meta),
        "total_cardinality_mismatches": sum(int(x["total_cardinality_mismatches"]) for x in chain_meta),
        "source_supported_monotonic_violations": sum(int(x["source_supported_monotonic_violations"]) for x in chain_meta),
        "conservative_lower_bound_violations": sum(int(x["conservative_lower_bound_violations"]) for x in chain_meta),
        "assertion": "null_contains_source[i,r] == source_supported[i] for every primary query and retained realization",
        "candidate_column_degrees": "exactly preserved within each scene by constrained bipartite switches",
        "per_query_scene_cardinalities": "exactly preserved",
    }
    if source_pair_checks != n * total_reps:
        raise RuntimeError("Source-membership invariant check count incomplete")
    if any(
        invariant[k] != 0
        for k in (
            "source_membership_mismatches",
            "row_scene_degree_mismatches",
            "candidate_column_degree_mismatches",
            "total_cardinality_mismatches",
            "source_supported_monotonic_violations",
            "conservative_lower_bound_violations",
        )
    ):
        raise RuntimeError(f"Invariant report is not clean: {invariant}")
    core.atomic_json(output_dir / "SOURCE_MEMBERSHIP_INVARIANT_REPORT.json", invariant)
    core.atomic_json(output_dir / "CHAIN_DIAGNOSTICS.json", {
        "status": "PASS",
        "chains": chain_meta,
        "aggregate_metric_diagnostics": diagnostics,
        "structural_overlap_split_Rhat": overlap_rhat,
        "structural_overlap_estimated_ESS": overlap_ess,
        "rhat_gate": cfg["rhat_max"],
    })

    # Explicit lower-bound audit table.
    exact_min_num = core.exact_supported_source_lower_bounds(observed_source, supported)
    lb_rows = []
    for mi, model in enumerate(core.MODEL_NAMES):
        for ki, k in enumerate(core.KS):
            lb_rows.append({
                "model": model,
                "K": k,
                "full_query_count": n,
                "exact_supported_source_hit_minimum_numerator": int(exact_min_num[mi, ki]),
                "exact_supported_source_hit_minimum_rate": float(exact_min_num[mi, ki] / n),
                "minimum_observed_null_recall_over_1000_repetitions": float(
                    flat_full[:, mi, ki].min()
                ),
                "diagnostic_source_recall": float(source_supported_source_rate[mi, ki]),
                "minimum_diagnostic_null_recall": float(
                    flat_supported[:, mi, ki].min()
                ),
                "diagnostic_null_ge_source_recall_all_repetitions": bool(
                    flat_supported[:, mi, ki].min() + 1e-15
                    >= source_supported_source_rate[mi, ki]
                ),
                "user_requested_conservative_floor": core.CONSERVATIVE_FULL_LOWER_BOUNDS.get((model, k), ""),
                "user_requested_floor_pass": (
                    ""
                    if (model, k) not in core.CONSERVATIVE_FULL_LOWER_BOUNDS
                    else bool(
                        flat_full[:, mi, ki].min() + 1e-15
                        >= core.CONSERVATIVE_FULL_LOWER_BOUNDS[(model, k)]
                    )
                ),
            })
    core.write_csv(output_dir / "TABLE_source_membership_lower_bound_audit.csv", lb_rows)

    return {
        "primary_rows": primary_rows,
        "diagnostic_rows": diagnostic_rows,
        "simultaneous_rows": simultaneous_rows,
        "diagnostics": diagnostics,
        "invariant": invariant,
        "null_prob": null_prob,
        "all_six_simultaneous_exclude_zero": all(
            r["simultaneous_ci_low"] > 0 for r in primary_rows
        ),
        "query_probability_path": qprob_path,
    }


def make_figure(primary_rows: list[dict], diagnostic_rows: list[dict], out_dir: Path):
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    labels = [f"{r['model']}@{r['K']}" for r in primary_rows]
    x = np.arange(len(labels), dtype=float)
    sem = np.asarray([r["semantic_recall"] for r in primary_rows])
    src = np.asarray([r["source_recall"] for r in primary_rows])
    nul = np.asarray([r["corrected_null_recall_mean"] for r in primary_rows])
    eff = np.asarray([r["semantic_minus_corrected_null"] for r in primary_rows])
    lo = np.asarray([r["simultaneous_ci_low"] for r in primary_rows])
    hi = np.asarray([r["simultaneous_ci_high"] for r in primary_rows])

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    ax = axes[0]
    width = 0.24
    ax.bar(x - width, src, width=width, label="Source Recall")
    ax.bar(x, nul, width=width, label="Source-membership-matched null")
    ax.bar(x + width, sem, width=width, label="Semantic Recall")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.set_title("Observed Recall vs corrected matched null")
    ax.legend(fontsize=8)

    ax = axes[1]
    yerr = np.vstack([eff - lo, hi - eff])
    ax.errorbar(x, eff, yerr=yerr, fmt="o", capsize=4)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Semantic − corrected-null Recall")
    ax.set_title("Familywise simultaneous 95% intervals")
    fig.tight_layout()
    fig.savefig(out_dir / "FIG_source_membership_matched_semantic_null.pdf", bbox_inches="tight")
    fig.savefig(
        out_dir / "FIG_source_membership_matched_semantic_null.png",
        dpi=300,
        bbox_inches="tight",
    )
    # Compatibility filenames now point to the corrected source-membership-matched figure.
    fig.savefig(out_dir / "FIG_semantic_vs_matched_null.pdf", bbox_inches="tight")
    fig.savefig(
        out_dir / "FIG_semantic_vs_matched_null.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_article_handoff(result: dict, cfg: dict, out_dir: Path):
    rows = result["primary_rows"]
    diag = result["diagnostic_rows"]
    lines = [
        "# Source-membership-matched semantic-null control — manuscript handoff",
        "",
        "## Status",
        "",
        "**Primary control:** constrained degree-preserving null with exact source-membership matching.",
        "",
        f"- Primary ambiguity cohort: n={cfg['primary_query_count']:,} (`V>1`).",
        f"- Source-supported diagnostic cohort: n={cfg['diagnostic_query_count']:,}.",
        f"- Null realizations: {cfg['chains']} chains × {cfg['retained_per_chain']} retained = "
        f"{cfg['chains']*cfg['retained_per_chain']:,}.",
        f"- Cluster bootstrap: {cfg['bootstrap_reps']:,} replicates, seed {cfg['bootstrap_seed']}.",
        "",
        "## Exact primary values",
        "",
        "| Model | K | Source Recall | Semantic Recall | Corrected null Recall | Semantic − null | Simultaneous 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['K']} | {100*r['source_recall']:.2f}% | "
            f"{100*r['semantic_recall']:.2f}% | "
            f"{100*r['corrected_null_recall_mean']:.2f}% | "
            f"{100*r['semantic_minus_corrected_null']:+.2f} pp | "
            f"[{100*r['simultaneous_ci_low']:+.2f}, {100*r['simultaneous_ci_high']:+.2f}] pp |"
        )

    positive = result["all_six_simultaneous_exclude_zero"]
    lines.extend([
        "",
        "## Paper-ready Methods",
        "",
        (
            "For the 4,794 held-out queries admitting more than one generator-consistent "
            "cell, we constructed a source-membership-matched configuration null from the "
            "observed query–cell incidence matrix. Within each candidate scene, constrained "
            "bipartite edge switches preserved every query's scene-specific valid-set "
            "cardinality and every candidate cell's marginal inclusion frequency. The "
            "released source-cell position was fixed for every query: it was pinned when "
            "the true semantic set supported the source and structurally forbidden "
            "otherwise. Four independently seeded chains produced 250 retained realizations "
            "each (1,000 total) after burn-in and thinning; the same randomized sets were "
            "scored against the frozen Text2Pos and Text2Loc rankings. Semantic-minus-null "
            "effects were summarized with 10,000 paired, scene-stratified, source-cell-"
            "clustered bootstrap replicates. Familywise 95% intervals used the joint maximum "
            "absolute centered bootstrap deviation across the six model-by-K comparisons."
        ),
        "",
        "## Paper-ready Results",
        "",
    ])
    if positive:
        lines.append(
            "Across all six model-by-K comparisons, semantic Recall exceeded the corrected "
            "source-membership-matched null and every familywise simultaneous 95% interval "
            "excluded zero. Because the null preserves valid-set cardinality, per-query "
            "scene composition, candidate-cell marginal frequency, and the complete observed "
            "source-support vector, these gains cannot be attributed to randomizing away the "
            "released source cell or merely accepting more candidates."
        )
    else:
        lines.append(
            "The corrected source-membership-matched control does not support a universal "
            "positive semantic-minus-null effect across all six comparisons. The paper should "
            "report the individual model/K estimates and simultaneous intervals without using "
            "the prior unconditional null as primary evidence."
        )

    lines.extend([
        "",
        "## Figure caption",
        "",
        (
            "**Source-membership-matched semantic-null control.** The left panel compares "
            "exact-source Recall, corrected matched-null Recall, and observed semantic-set "
            "Recall for the `V>1` ambiguity cohort. The null fixes each query's observed "
            "source-membership status while preserving query cardinality, per-scene "
            "cardinality, and candidate-cell column degrees by constrained bipartite switches. "
            "The right panel shows semantic-minus-null effects with familywise simultaneous "
            "95% clustered-bootstrap intervals across the six primary comparisons."
        ),
        "",
        "## Table caption",
        "",
        (
            "**Corrected source-membership-matched null control on semantically ambiguous "
            "held-out queries (`V>1`, n=4,794).** Null Recall is averaged over 1,000 retained "
            "degree-preserving realizations from four independent chains. The source-supported "
            "diagnostic subset contains 3,780 queries. Simultaneous intervals are based on "
            "10,000 paired, scene-stratified, source-cell-clustered bootstrap replicates."
        ),
        "",
        "## Claims supported",
        "",
    ])
    if positive:
        lines.extend([
            "- The semantic advantage on ambiguous queries is not explained by set cardinality alone.",
            "- The result persists after matching the complete source-support vector.",
            "- The result also controls exact per-query scene composition and candidate-cell marginal frequency.",
        ])
    else:
        lines.extend([
            "- The corrected null provides the appropriate control, but the paper must condition claims on the model/K results whose simultaneous intervals exclude zero.",
        ])
    lines.extend([
        "",
        "## Claims that must not be made",
        "",
        "- Do not use the earlier unconditional configuration-model null as the primary matched control.",
        "- Do not claim that semantic scoring universally improves Recall over all 11,505 queries.",
        "- Do not describe the switch-chain null as uniformly sampled from all feasible incidence matrices unless a mixing theorem is established.",
        "- Do not call the cell-level valid-set analysis continuous localization-at-radius.",
        "",
        "## Limitation of the corrected null",
        "",
        (
            "The constrained switch chain exactly preserves the specified margins and source "
            "positions. Rejected switch proposals are retained as Markov-chain self-transitions, "
            "so burn-in and thinning are measured on the proposal clock rather than the embedded "
            "accepted-move jump chain. Finite Markov-chain sampling still does not prove coverage "
            "of the entire feasible constrained graph space; four-chain diagnostics, structural "
            "overlap, R-hat, and Monte Carlo error are therefore reported explicitly."
        ),
        "",
        "## Source-supported diagnostic",
        "",
        "| Model | K | Source Recall | Semantic Recall | Corrected null Recall | Null ≥ source in every repetition |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for r in diag:
        lines.append(
            f"| {r['model']} | {r['K']} | {100*r['source_recall']:.2f}% | "
            f"{100*r['semantic_recall']:.2f}% | {100*r['corrected_null_recall_mean']:.2f}% | "
            f"{'YES' if r['mandatory_null_ge_source_recall'] else 'NO'} |"
        )
    core.atomic_text(out_dir / "ARTICLE_HANDOFF.md", "\n".join(lines) + "\n")


def build_final_archive(
    *,
    run_root: Path,
    package_code_root: Path,
    project_root: Path,
    legacy_zip: Path,
    cfg: dict,
    config_hash: str,
    input_hash: str,
    result: dict,
):
    final_dir = project_root / "SOURCE_MEMBERSHIP_MATCHED_SEMANTIC_NULL_CONTROL_FINAL_v2_1"
    final_zip = project_root / "SOURCE_MEMBERSHIP_MATCHED_SEMANTIC_NULL_CONTROL_FINAL_v2_1.zip"
    if final_dir.exists() or final_zip.exists():
        raise FileExistsError(
            "A v2 final package already exists. Refusing to overwrite it; use audit-only "
            "or move the prior v2 package before creating a distinct rerun."
        )
    final_dir.mkdir(parents=True)

    # Copy scientific outputs and chain artifacts.
    for p in sorted(run_root.iterdir()):
        if p.name in {"work"}:
            continue
        dst = final_dir / p.name
        if p.is_dir():
            shutil.copytree(p, dst)
        elif p.is_file():
            shutil.copy2(p, dst)

    code_dst = final_dir / "code"
    shutil.copytree(package_code_root, code_dst)

    legacy_dir = final_dir / "LEGACY_UNCONSTRAINED_SECONDARY"
    legacy_info = core.verify_and_copy_legacy_secondary(legacy_zip, legacy_dir)
    core.atomic_json(final_dir / "LEGACY_SECONDARY_STATUS.json", legacy_info)

    core.atomic_json(final_dir / "CONFIGURATION_AND_SEEDS.json", {
        "status": "PASS",
        "configuration": cfg,
        "config_hash": config_hash,
        "input_hash": input_hash,
        "chain_seeds": [
            core.derive_chain_seed(cfg["base_seed"], cid)
            for cid in range(cfg["chains"])
        ],
    })

    env = [
        f"python={sys.version}",
        f"platform={platform.platform()}",
        f"cpu_count={os.cpu_count()}",
        f"numpy={np.__version__}",
    ]
    try:
        import pandas as pd
        env.append(f"pandas={pd.__version__}")
    except Exception:
        pass
    try:
        import pyarrow
        env.append(f"pyarrow={pyarrow.__version__}")
    except Exception:
        pass
    try:
        import matplotlib
        env.append(f"matplotlib={matplotlib.__version__}")
    except Exception:
        pass
    core.atomic_text(final_dir / "ENVIRONMENT_REPORT.txt", "\n".join(env) + "\n")
    # Explicit input-hash table for archival inspection.
    input_lock_path = run_root / "INPUT_LOCK.json"
    if input_lock_path.is_file():
        lock = json.loads(input_lock_path.read_text(encoding="utf-8"))
        core.write_csv(
            final_dir / "INPUT_HASHES.csv",
            [{"input": k, "sha256_or_value": v} for k, v in lock.items()],
        )

    audit = {
        "status": "PASS",
        "primary_query_count": cfg["primary_query_count"],
        "source_supported_primary_count": cfg["diagnostic_query_count"],
        "null_repetitions": cfg["chains"] * cfg["retained_per_chain"],
        "bootstrap_replicates": cfg["bootstrap_reps"],
        "all_source_membership_invariants_pass": True,
        "all_conservative_lower_bounds_pass": True,
        "all_six_simultaneous_intervals_exclude_zero": result["all_six_simultaneous_exclude_zero"],
        "primary_analysis": "source-membership-matched constrained degree-preserving null",
        "legacy_analysis_role": "secondary only",
        "config_hash": config_hash,
        "input_hash": input_hash,
    }
    core.atomic_json(final_dir / "ENGINEERING_AUDIT.json", audit)

    sci = [
        "# Scientific audit — source-membership-matched semantic-null control v2.1",
        "",
        "**STATUS: PASS**",
        "",
        "- Query cardinality is preserved exactly.",
        "- Per-query candidate-scene cardinalities are preserved exactly.",
        "- The complete source-membership vector is preserved in every retained realization.",
        "- Candidate-cell column degrees are preserved exactly within scenes by constrained bipartite switches.",
        "- Text2Pos/Text2Loc rankings are loaded from the frozen prediction files; no inference is rerun.",
        "- The same null realization is scored for both models.",
        f"- Four chains × {cfg['retained_per_chain']} retained = {cfg['chains']*cfg['retained_per_chain']} total null realizations.",
        f"- Cluster bootstrap replicates: {cfg['bootstrap_reps']}.",
        "- The older unconditional configuration-model control is retained as a labelled secondary analysis when available; it is not required for the primary corrected control.",
        "- Rejected switch proposals remain on the Markov-chain clock as self-transitions; retained states are separated by fixed proposal counts, not fixed accepted-move counts.",
        "",
        "The archive must not be interpreted as proving uniform sampling over all feasible constrained incidence matrices; finite switch-chain convergence is assessed empirically.",
    ]
    core.atomic_text(final_dir / "SCIENTIFIC_AUDIT.md", "\n".join(sci) + "\n")

    core.atomic_text(
        final_dir / "RUN_SUMMARY.txt",
        (
            "FINAL STATUS: PASS\n"
            "Primary control: source-membership-matched constrained degree-preserving semantic null\n"
            f"Primary queries: {cfg['primary_query_count']}\n"
            f"Source-supported diagnostic queries: {cfg['diagnostic_query_count']}\n"
            f"Null repetitions: {cfg['chains']*cfg['retained_per_chain']}\n"
            f"Bootstrap replicates: {cfg['bootstrap_reps']}\n"
            f"All six simultaneous intervals exclude zero: {result['all_six_simultaneous_exclude_zero']}\n"
        ),
    )

    # Build non-circular checksums: manifest covers every file except itself.
    base_rows = core.manifest_tree(
        final_dir, exclude={"MANIFEST.json", "SHA256SUMS.txt", "OUTPUT_HASHES.csv"}
    )
    core.write_csv(final_dir / "OUTPUT_HASHES.csv", base_rows)
    checksum_rows = core.manifest_tree(
        final_dir, exclude={"MANIFEST.json", "SHA256SUMS.txt"}
    )
    core.atomic_text(
        final_dir / "SHA256SUMS.txt",
        "\n".join(f"{r['sha256']}  {r['path']}" for r in checksum_rows) + "\n",
    )
    manifest_rows = core.manifest_tree(final_dir, exclude={"MANIFEST.json"})
    core.atomic_json(final_dir / "MANIFEST.json", {
        "status": "PASS",
        "schema_version": 2,
        "coverage": "all files in final directory except MANIFEST.json itself",
        "files": manifest_rows,
    })

    core.zip_tree(final_dir, final_zip)
    return final_dir, final_zip, core.sha256_file(final_zip)


def run_full(args):
    ensure_runtime_deps()
    cfg = load_config(args.config)
    project_root = Path(args.project_root)
    project_root.mkdir(parents=True, exist_ok=True)

    evidence_zip = Path(args.evidence_zip)
    legacy_zip = core.discover_legacy_null_archive(
        project_root, Path(args.legacy_null_zip) if args.legacy_null_zip else None
    )

    package_code_root = Path(__file__).resolve().parents[1]
    chash = code_hash(package_code_root)
    config_material = dict(cfg)
    config_material["code_hash"] = chash
    config_hash = core.sha256_text(json.dumps(config_material, sort_keys=True))

    run_root = project_root / "source_membership_matched_null_v2_1" / f"RUN_{config_hash[:12]}"
    run_root.mkdir(parents=True, exist_ok=True)
    work = run_root / "work"
    work.mkdir(parents=True, exist_ok=True)

    try:
        data = core.load_evidence_inputs(evidence_zip, work)
        input_lock = {
            "evidence_sha256": data["evidence_sha256"],
            "valid_color_sha256": data["valid_color_sha256"],
            "candidate_universe_sha256": core.EXPECTED_CANDIDATE_UNIVERSE_SHA256,
            "ranking_hashes": core.EXPECTED_PRED_SHA,
            "legacy_null_zip_sha256": core.sha256_file(legacy_zip) if legacy_zip else None,
        }
        input_hash = core.sha256_text(json.dumps(input_lock, sort_keys=True))
        prepared_path = work / "PREPARED_PRIMARY_INPUTS.npz"
        core.save_prepared_inputs(prepared_path, data, config_hash, input_hash)
        shutil.copy2(prepared_path, run_root / "PREPARED_PRIMARY_INPUTS.npz")
        core.atomic_json(run_root / "INPUT_LOCK.json", input_lock)
        core.atomic_json(run_root / "CONFIG.json", cfg)
        write_preflight_tables(run_root, data, evidence_zip, legacy_zip)

        workers = int(args.workers or min(cfg["chains"], max(1, (os.cpu_count() or 2) - 1)))
        workers = max(1, min(workers, cfg["chains"]))
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

        chain_dir = run_root / "chains"
        chain_dir.mkdir(exist_ok=True)
        futures = []
        print(
            f"Running {cfg['chains']} independent constrained chains "
            f"with {workers} worker(s)"
        )
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            for cid in range(cfg["chains"]):
                futures.append(
                    ex.submit(
                        core.run_chain,
                        prepared_path,
                        chain_dir / f"chain_{cid:02d}",
                        chain_id=cid,
                        base_seed=cfg["base_seed"],
                        burn_edge_multiples=cfg["burn_edge_multiples"],
                        thin_edge_multiples=cfg["thin_edge_multiples"],
                        retained_reps=cfg["retained_per_chain"],
                        checkpoint_every=cfg["checkpoint_every"],
                        config_hash=config_hash,
                        input_hash=input_hash,
                        smoke=False,
                    )
                )
            for fut in as_completed(futures):
                meta = fut.result()
                print(
                    f"CHAIN {meta['chain_id']} PASS; "
                    f"acceptance={meta['acceptance_rate']:.3f}; "
                    f"final overlap={meta['free_edge_overlap_final']:.4f}"
                )

        prepared = core.load_prepared_inputs(prepared_path)
        result = assemble_results(
            prepared=prepared,
            chain_dir=chain_dir,
            cfg=cfg,
            output_dir=run_root,
            config_hash=config_hash,
            input_hash=input_hash,
        )
        make_figure(result["primary_rows"], result["diagnostic_rows"], run_root)
        make_article_handoff(result, cfg, run_root)

        final_dir, final_zip, final_sha = build_final_archive(
            run_root=run_root,
            package_code_root=package_code_root,
            project_root=project_root,
            legacy_zip=legacy_zip,
            cfg=cfg,
            config_hash=config_hash,
            input_hash=input_hash,
            result=result,
        )
        print()
        print("=" * 96)
        print("FINAL STATUS: PASS")
        print("SOURCE-MEMBERSHIP-MATCHED SEMANTIC NULL CONTROL v2.1: PASS")
        print("=" * 96)
        print("Primary queries:", cfg["primary_query_count"])
        print("Source-supported diagnostic queries:", cfg["diagnostic_query_count"])
        print("Null repetitions:", cfg["chains"] * cfg["retained_per_chain"])
        print("Bootstrap replicates:", cfg["bootstrap_reps"])
        print(
            "All six familywise simultaneous intervals exclude zero:",
            result["all_six_simultaneous_exclude_zero"],
        )
        print("Final directory:", final_dir)
        print("Final ZIP:", final_zip)
        print("Final ZIP SHA256:", final_sha)
    except Exception as exc:
        core.atomic_json(run_root / "NOT_READY.json", {
            "status": "NOT_READY",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        print()
        print("=" * 96)
        print("FINAL STATUS: NOT_READY")
        print("No final v2 PASS archive was created.")
        print("Failure record:", run_root / "NOT_READY.json")
        print("=" * 96)
        raise


def run_smoke(args):
    ensure_runtime_deps()
    cfg = load_config(args.config)
    project_root = Path(args.project_root)
    evidence_zip = Path(args.evidence_zip)
    package_code_root = Path(__file__).resolve().parents[1]
    chash = code_hash(package_code_root)
    smoke_cfg = dict(cfg)
    smoke_cfg.update({
        "chains": 4,
        "retained_per_chain": 8,
        "burn_edge_multiples": 2,
        "thin_edge_multiples": 1,
        "checkpoint_every": 2,
    })
    config_hash = core.sha256_text(json.dumps(smoke_cfg, sort_keys=True))
    run_root = project_root / "source_membership_matched_null_v2_1" / "SMOKE"
    work = run_root / "work"
    work.mkdir(parents=True, exist_ok=True)
    data = core.load_evidence_inputs(evidence_zip, work, smoke_n=120)
    input_lock = {
        "evidence_sha256": data["evidence_sha256"],
        "valid_color_sha256": data["valid_color_sha256"],
        "smoke_n": len(data["query_id"]),
    }
    input_hash = core.sha256_text(json.dumps(input_lock, sort_keys=True))
    prepared = work / "PREPARED_SMOKE_INPUTS.npz"
    core.save_prepared_inputs(prepared, data, config_hash, input_hash)

    metas = []
    for cid in range(4):
        metas.append(
            core.run_chain(
                prepared,
                run_root / "chains" / f"chain_{cid:02d}",
                chain_id=cid,
                base_seed=smoke_cfg["base_seed"],
                burn_edge_multiples=smoke_cfg["burn_edge_multiples"],
                thin_edge_multiples=smoke_cfg["thin_edge_multiples"],
                retained_reps=smoke_cfg["retained_per_chain"],
                checkpoint_every=smoke_cfg["checkpoint_every"],
                config_hash=config_hash,
                input_hash=input_hash,
                smoke=True,
            )
        )
    if any(m["status"] != "PASS" for m in metas):
        raise RuntimeError("Smoke chain failure")
    print("SOURCE-MEMBERSHIP-MATCHED NULL SMOKE: PASS")


def run_audit_only(args):
    project_root = Path(args.project_root)
    final_dir = project_root / "SOURCE_MEMBERSHIP_MATCHED_SEMANTIC_NULL_CONTROL_FINAL_v2_1"
    if not final_dir.is_dir():
        raise FileNotFoundError(final_dir)
    manifest = json.loads((final_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    checked = 0
    for entry in manifest["files"]:
        p = final_dir / entry["path"]
        if not p.is_file():
            raise RuntimeError(f"Manifest file missing: {entry['path']}")
        if core.sha256_file(p) != entry["sha256"]:
            raise RuntimeError(f"Manifest hash mismatch: {entry['path']}")
        checked += 1
    inv = json.loads((final_dir / "SOURCE_MEMBERSHIP_INVARIANT_REPORT.json").read_text())
    if inv.get("status") != "PASS" or inv.get("source_membership_mismatches") != 0:
        raise RuntimeError("Source-membership invariant report failed")
    eng = json.loads((final_dir / "ENGINEERING_AUDIT.json").read_text())
    if eng.get("status") != "PASS":
        raise RuntimeError("Engineering audit not PASS")
    print(f"AUDIT-ONLY PASS: {checked} manifested files + invariant/audit gates")


def run_figure_only(args):
    ensure_runtime_deps()
    import pandas as pd
    project_root = Path(args.project_root)
    final_dir = project_root / "SOURCE_MEMBERSHIP_MATCHED_SEMANTIC_NULL_CONTROL_FINAL_v2_1"
    primary = pd.read_csv(final_dir / "TABLE_source_membership_matched_null_primary.csv").to_dict("records")
    diag = pd.read_csv(final_dir / "TABLE_source_supported_diagnostic.csv").to_dict("records")
    make_figure(primary, diag, final_dir)
    print("FIGURE-ONLY PASS")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full", "audit-only", "figure-only"], default="full")
    parser.add_argument(
        "--evidence-zip",
        default="/content/drive/MyDrive/or_fusion_verification/heldout_identifiability_v1/HELDOUT_IDENTIFIABILITY_FINAL_EVIDENCE_4c1794be64d2.zip",
    )
    parser.add_argument(
        "--project-root",
        default="/content/drive/MyDrive/or_fusion_verification/heldout_identifiability_v1",
    )
    parser.add_argument("--legacy-null-zip", default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)

    if args.mode == "full":
        run_full(args)
    elif args.mode == "smoke":
        run_smoke(args)
    elif args.mode == "audit-only":
        run_audit_only(args)
    elif args.mode == "figure-only":
        run_figure_only(args)
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()
