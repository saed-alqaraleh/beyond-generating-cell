
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_EVIDENCE_ZIP_SHA256 = "961934e5031459785663185c80dafd453451565bdd55c19d59de6930536446ac"
EXPECTED_VALID_COLOR_SHA256 = "fb062c7b743a39a368552ea85be6cbfecfa83136201fe9948b3af136f9561234"
EXPECTED_CANDIDATE_UNIVERSE_SHA256 = "df64a26bff45406e2fd4cfbcfb5b3e54691dfb6acac9bce7ed231014e13d9eb1"
EXPECTED_QUERY_COUNT = 11505
EXPECTED_CANDIDATE_COUNT = 4308
EXPECTED_PRIMARY_N = 4794
EXPECTED_PRIMARY_SUPPORTED_N = 3780
EXPECTED_PRIMARY_UNSUPPORTED_N = 1014
MODEL_NAMES = ("text2pos", "text2loc")
KS = (1, 5, 10)
SCENES = (
    "2013_05_28_drive_0003_sync",
    "2013_05_28_drive_0005_sync",
    "2013_05_28_drive_0009_sync",
)
EXPECTED_PRED_SHA = {
    "text2pos": "7d6cf45a44d89cb1a21c79aa79b09b24bba7d695fa7fb9c6f8e52bda477ff0fc",
    "text2loc": "20c4c667618f8f8dcb48b4458174bb6c7c660a9ab8ed1bf39f65d3ef4ba3fce3",
}
EXPECTED_ALL_SOURCE_HITS = {
    "text2pos": {1: 1372, 5: 3676, 10: 4901},
    "text2loc": {1: 3428, 5: 6917, 10: 8206},
}
EXPECTED_PRIMARY_HITS = {
    "text2pos": {
        "source": {1: 652, 5: 1671, 10: 2191},
        "semantic": {1: 1056, 5: 2319, 10: 2909},
    },
    "text2loc": {
        "source": {1: 1524, 5: 3064, 10: 3606},
        "semantic": {1: 2359, 5: 3756, 10: 4133},
    },
}
CONSERVATIVE_FULL_LOWER_BOUNDS = {
    ("text2pos", 5): 0.1370,
    ("text2pos", 10): 0.2455,
    ("text2loc", 1): 0.1064,
    ("text2loc", 5): 0.4276,
    ("text2loc", 10): 0.5407,
}


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            b = fh.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def json_default(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)


def write_csv(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_text(path, "")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            clean = {}
            for key in fields:
                v = row.get(key, "")
                if v is None:
                    v = ""
                if isinstance(v, float) and not math.isfinite(v):
                    v = ""
                clean[key] = v
            w.writerow(clean)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"Invalid JSONL {path}:{ln}: {exc}") from exc
    return rows


def query_id_of(row: dict) -> str | None:
    for key in ("query_uid", "query_id", "uid"):
        if row.get(key) is not None:
            return str(row[key])
    return None


def source_id_of(row: dict) -> str:
    for key in ("source_cell_id", "gt_cell_id", "ground_truth_cell_id", "target_cell_id"):
        if row.get(key) is not None:
            return str(row[key])
    raise KeyError(f"Cannot resolve source ID from keys: {sorted(row)}")


def ranked_ids_of(row: dict) -> list[str]:
    for key in ("retrieved_cell_ids", "ranked_cell_ids", "retrieval_cell_ids", "candidate_cell_ids"):
        value = row.get(key)
        if isinstance(value, list):
            out = []
            for x in value:
                if isinstance(x, dict):
                    found = None
                    for k in ("cell_id", "id", "candidate_id"):
                        if x.get(k) is not None:
                            found = x[k]
                            break
                    if found is None:
                        raise KeyError(f"Ranking item has no cell ID: {x}")
                    x = found
                out.append(str(x))
            return out
    raise KeyError(f"Cannot resolve ranked IDs from keys: {sorted(row)}")


def candidate_id_of(row: dict) -> str:
    for key in ("cell_id", "candidate_cell_id", "id"):
        if row.get(key) is not None:
            return str(row[key])
    raise KeyError(f"Cannot resolve candidate ID from keys: {sorted(row)}")


def scene_of_row(row: dict) -> str:
    for key in ("scene", "scene_id", "sequence", "drive"):
        if row.get(key) is not None:
            s = str(row[key])
            for full in SCENES:
                if s == full or s in full or full in s:
                    return full
    cid = str(row.get("cell_id") or row.get("source_cell_id") or "")
    for full in SCENES:
        short = full.split("_drive_")[1].split("_sync")[0]
        if cid.startswith(short + "_") or f"_{short}_" in cid:
            return full
    raise KeyError(f"Cannot resolve scene from row: {row}")


def verify_zip_manifest(zip_path: Path, expected_outer_sha: str | None = None) -> dict:
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    outer = sha256_file(zip_path)
    if expected_outer_sha and outer != expected_outer_sha:
        raise RuntimeError(
            f"Evidence ZIP SHA mismatch: {outer} != {expected_outer_sha}"
        )
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP integrity failure at {bad}")
        names = set(zf.namelist())
        if "MANIFEST.json" not in names:
            raise RuntimeError("Evidence ZIP has no MANIFEST.json")
        manifest = json.loads(zf.read("MANIFEST.json"))
        files = manifest.get("files")
        if not isinstance(files, list):
            raise RuntimeError("Unsupported evidence manifest schema")
        checked = 0
        for entry in files:
            rel = entry.get("path")
            expected = entry.get("sha256")
            if not rel or not expected:
                raise RuntimeError(f"Malformed manifest entry: {entry}")
            if rel not in names:
                raise RuntimeError(f"Manifested evidence file missing: {rel}")
            got = hashlib.sha256(zf.read(rel)).hexdigest()
            if got != expected:
                raise RuntimeError(
                    f"Evidence internal hash mismatch for {rel}: {got} != {expected}"
                )
            checked += 1
    return {
        "status": "PASS",
        "zip": str(zip_path),
        "outer_sha256": outer,
        "manifest_files_checked": checked,
    }


def safe_extract_zip(zip_path: Path, dest: Path) -> None:
    zip_path = Path(zip_path)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        root = dest.resolve()
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe ZIP member: {info.filename}")
        zf.extractall(dest)


def load_evidence_inputs(evidence_zip: Path, work_dir: Path, *, smoke_n: int | None = None) -> dict:
    """Load and strictly align the frozen held-out evidence and frozen rankings."""
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required") from exc
    try:
        import pyarrow  # noqa: F401
    except Exception as exc:
        raise RuntimeError("pyarrow is required to read the frozen Parquet audit files") from exc

    evidence_zip = Path(evidence_zip)
    work_dir = Path(work_dir)
    extract_dir = work_dir / "evidence_extracted"
    marker = extract_dir / ".EXTRACTED_SHA256"
    evidence_sha = sha256_file(evidence_zip)

    if evidence_sha != EXPECTED_EVIDENCE_ZIP_SHA256:
        raise RuntimeError(
            f"Wrong parent evidence archive: {evidence_sha} != "
            f"{EXPECTED_EVIDENCE_ZIP_SHA256}"
        )
    verify_zip_manifest(evidence_zip, EXPECTED_EVIDENCE_ZIP_SHA256)

    if not marker.is_file() or marker.read_text().strip() != evidence_sha:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        safe_extract_zip(evidence_zip, extract_dir)
        atomic_text(marker, evidence_sha + "\n")

    valid_path = extract_dir / "S04_enumerate_heldout_valid_sets" / "valid_sets_color.npz"
    if sha256_file(valid_path) != EXPECTED_VALID_COLOR_SHA256:
        raise RuntimeError("Frozen color valid-set archive hash changed")

    qaudit_path = extract_dir / "S06_compute_per_query_metrics" / "heldout_query_audit.parquet"
    input_manifest_path = extract_dir / "S01_input_inventory_and_hashes" / "INPUT_MANIFEST.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))

    with np.load(valid_path, allow_pickle=False) as z:
        query_ids = np.asarray(z["query_uid"]).astype(str)
        candidate_ids = np.asarray(z["candidate_cell_id"]).astype(str)
        packed = z["packed"].copy()
        V = z["V"].astype(np.int32)
        source_supported_saved = z["source_supported"].astype(bool)
        bit_length = int(np.asarray(z["bit_length"]).ravel()[0])
        candidate_universe_field = (
            str(np.asarray(z["candidate_universe_sha256"]).ravel()[0])
            if "candidate_universe_sha256" in z.files
            else EXPECTED_CANDIDATE_UNIVERSE_SHA256
        )

    if len(query_ids) != EXPECTED_QUERY_COUNT or len(set(query_ids.tolist())) != EXPECTED_QUERY_COUNT:
        raise RuntimeError("Frozen query count/uniqueness gate failed")
    if len(candidate_ids) != EXPECTED_CANDIDATE_COUNT or len(set(candidate_ids.tolist())) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("Frozen candidate count/uniqueness gate failed")
    if bit_length != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError(f"Unexpected valid-set bit length {bit_length}")
    if candidate_universe_field != EXPECTED_CANDIDATE_UNIVERSE_SHA256:
        raise RuntimeError("Candidate-universe SHA field changed")

    qdf = pd.read_parquet(qaudit_path)
    required_qcols = {
        "query_id", "scene_id", "source_cell_id", "source_group_id",
        "V_primary", "source_supported_primary",
    }
    missing = required_qcols - set(qdf.columns)
    if missing:
        raise RuntimeError(f"Frozen query audit missing columns: {sorted(missing)}")
    if qdf["query_id"].astype(str).tolist() != query_ids.tolist():
        raise RuntimeError("Query audit order differs from valid-set archive")
    if not np.array_equal(qdf["V_primary"].to_numpy(dtype=np.int32), V):
        raise RuntimeError("V_primary differs between query audit and valid-set archive")
    if not np.array_equal(
        qdf["source_supported_primary"].to_numpy(dtype=bool),
        source_supported_saved,
    ):
        raise RuntimeError("Source-support flag differs between query audit and valid-set archive")

    # External frozen canonical candidate metadata is required only for exact scene labels.
    cand_meta = input_manifest["canonical"]["candidates"]
    cand_path = Path(cand_meta["path"])
    if not cand_path.is_file():
        raise FileNotFoundError(
            f"Frozen canonical candidate file is missing: {cand_path}"
        )
    if sha256_file(cand_path) != cand_meta["sha256"]:
        raise RuntimeError("Frozen canonical candidate metadata hash changed")
    cand_rows = read_jsonl(cand_path)
    canon_candidate_ids = [candidate_id_of(r) for r in cand_rows]
    if canon_candidate_ids != candidate_ids.tolist():
        raise RuntimeError("Canonical candidate order differs from valid-set candidate order")
    cand_scenes = [scene_of_row(r) for r in cand_rows]
    scene_to_code = {s: i for i, s in enumerate(SCENES)}
    candidate_scene_code = np.asarray([scene_to_code[s] for s in cand_scenes], dtype=np.int8)

    candidate_index = {cid: i for i, cid in enumerate(candidate_ids.tolist())}
    source_ids = qdf["source_cell_id"].astype(str).tolist()
    source_idx = np.asarray(
        [candidate_index.get(x, -1) for x in source_ids],
        dtype=np.int32,
    )
    if (source_idx < 0).any():
        bad = np.flatnonzero(source_idx < 0)[:10].tolist()
        raise RuntimeError(f"Source cells outside candidate universe at rows {bad}")

    # Explicitly re-evaluate source membership from the frozen incidence matrix.
    valid_full = np.unpackbits(
        packed, axis=1, bitorder="big"
    )[:, :EXPECTED_CANDIDATE_COUNT].astype(bool)
    explicit_support = valid_full[np.arange(EXPECTED_QUERY_COUNT), source_idx]
    if not np.array_equal(explicit_support, source_supported_saved):
        raise RuntimeError("Stored source-support flag != explicit source membership")

    # Load the exact saved rankings. No inference and no ranking reconstruction.
    top10_all = np.empty((2, EXPECTED_QUERY_COUNT, 10), dtype=np.int32)
    ranking_alignment = {}
    for mi, model in enumerate(MODEL_NAMES):
        pred_info = input_manifest["predictions"][model]
        pred_path = Path(pred_info["path"])
        if not pred_path.is_file():
            raise FileNotFoundError(
                f"Frozen {model} predictions missing: {pred_path}"
            )
        observed_sha = sha256_file(pred_path)
        if observed_sha != EXPECTED_PRED_SHA[model] or observed_sha != pred_info["sha256"]:
            raise RuntimeError(f"{model} frozen prediction hash changed")
        rows = read_jsonl(pred_path)
        if len(rows) != EXPECTED_QUERY_COUNT:
            raise RuntimeError(f"{model} ranking row count changed")
        qid_presence = [query_id_of(r) is not None for r in rows]
        if any(qid_presence) and not all(qid_presence):
            raise RuntimeError(f"{model}: mixed presence/absence of query IDs")
        if all(qid_presence):
            observed_qids = [query_id_of(r) for r in rows]
            if observed_qids != query_ids.tolist():
                raise RuntimeError(f"{model}: frozen ranking query-ID order mismatch")
            ranking_alignment[model] = "explicit_query_id"
        else:
            # The standardized output was already validated in the parent experiment.
            # Without explicit IDs, require exact source-cell sequence as a second lock.
            observed_sources = [source_id_of(r) for r in rows]
            if observed_sources != source_ids:
                raise RuntimeError(
                    f"{model}: no query IDs and source-cell order does not match frozen audit"
                )
            ranking_alignment[model] = "exact_row_order_plus_source_id"
        for i, row in enumerate(rows):
            ranked = ranked_ids_of(row)
            if len(ranked) < 10:
                raise RuntimeError(f"{model} row {i}: fewer than 10 ranked candidates")
            top = ranked[:10]
            if len(set(top)) != 10:
                raise RuntimeError(f"{model} row {i}: duplicate candidate within top-10")
            try:
                top10_all[mi, i] = [candidate_index[c] for c in top]
            except KeyError as exc:
                raise RuntimeError(
                    f"{model} row {i}: ranking contains candidate outside frozen universe"
                ) from exc

        # Existing exact-source locks over all 11,505 queries.
        src_hit = top10_all[mi] == source_idx[:, None]
        observed = {
            1: int(src_hit[:, :1].any(axis=1).sum()),
            5: int(src_hit[:, :5].any(axis=1).sum()),
            10: int(src_hit[:, :10].any(axis=1).sum()),
        }
        if observed != EXPECTED_ALL_SOURCE_HITS[model]:
            raise RuntimeError(
                f"{model} all-query source Recall lock failed: "
                f"{observed} != {EXPECTED_ALL_SOURCE_HITS[model]}"
            )

    primary_mask = V > 1
    if int(primary_mask.sum()) != EXPECTED_PRIMARY_N:
        raise RuntimeError(f"Primary V>1 count {int(primary_mask.sum())} != {EXPECTED_PRIMARY_N}")
    support_primary = source_supported_saved[primary_mask]
    if int(support_primary.sum()) != EXPECTED_PRIMARY_SUPPORTED_N:
        raise RuntimeError(
            f"Source-supported ambiguous count {int(support_primary.sum())} "
            f"!= {EXPECTED_PRIMARY_SUPPORTED_N}"
        )
    if int((~support_primary).sum()) != EXPECTED_PRIMARY_UNSUPPORTED_N:
        raise RuntimeError("Unsupported ambiguous count changed")

    indices = np.flatnonzero(primary_mask)
    if smoke_n is not None:
        # Deterministic smoke subset spanning all scenes and both support states.
        pri_scene = qdf.loc[primary_mask, "scene_id"].astype(str).to_numpy()
        chosen = []
        for scene in SCENES:
            cand = indices[pri_scene == scene]
            chosen.extend(cand[:max(2, smoke_n // 6)].tolist())
        supported_idx = indices[source_supported_saved[indices]]
        unsupported_idx = indices[~source_supported_saved[indices]]
        chosen.extend(supported_idx[:max(2, smoke_n // 4)].tolist())
        chosen.extend(unsupported_idx[:max(2, smoke_n // 4)].tolist())
        chosen = sorted(set(chosen))
        if len(chosen) < smoke_n:
            for x in indices:
                if int(x) not in chosen:
                    chosen.append(int(x))
                    if len(chosen) >= smoke_n:
                        break
        indices = np.asarray(chosen[:smoke_n], dtype=np.int64)

    valid = valid_full[indices].copy()
    qdf_primary = qdf.iloc[indices].reset_index(drop=True)
    source_idx_primary = source_idx[indices].copy()
    source_supported = source_supported_saved[indices].copy()
    top10 = top10_all[:, indices, :].copy()
    V_primary = V[indices].copy()
    query_ids_primary = query_ids[indices].copy()

    if smoke_n is None:
        # Exact held-out H1 locks.
        for mi, model in enumerate(MODEL_NAMES):
            src_hits = top10[mi] == source_idx_primary[:, None]
            sem_hits = np.zeros((len(indices), 3), dtype=bool)
            for ki, k in enumerate(KS):
                sem_hits[:, ki] = np.asarray(
                    [
                        bool(valid[i, top10[mi, i, :k]].any())
                        for i in range(len(indices))
                    ],
                    dtype=bool,
                )
                src_n = int(src_hits[:, :k].any(axis=1).sum())
                sem_n = int(sem_hits[:, ki].sum())
                if src_n != EXPECTED_PRIMARY_HITS[model]["source"][k]:
                    raise RuntimeError(
                        f"{model}@{k} primary source lock {src_n} != "
                        f"{EXPECTED_PRIMARY_HITS[model]['source'][k]}"
                    )
                if sem_n != EXPECTED_PRIMARY_HITS[model]["semantic"][k]:
                    raise RuntimeError(
                        f"{model}@{k} primary semantic lock {sem_n} != "
                        f"{EXPECTED_PRIMARY_HITS[model]['semantic'][k]}"
                    )

    # Per-query scene composition.
    row_scene_counts = np.zeros((len(indices), len(SCENES)), dtype=np.int16)
    for si in range(len(SCENES)):
        row_scene_counts[:, si] = valid[:, candidate_scene_code == si].sum(axis=1)
    if not np.array_equal(row_scene_counts.sum(axis=1).astype(np.int32), V_primary):
        raise RuntimeError("Per-query scene composition does not sum to V_i")

    query_scene_code = np.asarray(
        [scene_to_code[str(x)] for x in qdf_primary["scene_id"].astype(str)],
        dtype=np.int8,
    )
    source_group_ids = qdf_primary["source_group_id"].astype(str).to_numpy()
    if len(set(query_ids_primary.tolist())) != len(query_ids_primary):
        raise RuntimeError("Primary query IDs are not unique")

    # Fixed observed source/semantic indicators.
    observed_source = np.zeros((len(indices), 2, 3), dtype=np.uint8)
    observed_semantic = np.zeros((len(indices), 2, 3), dtype=np.uint8)
    for mi in range(2):
        for ki, k in enumerate(KS):
            observed_source[:, mi, ki] = (
                top10[mi, :, :k] == source_idx_primary[:, None]
            ).any(axis=1)
            observed_semantic[:, mi, ki] = np.asarray(
                [bool(valid[i, top10[mi, i, :k]].any()) for i in range(len(indices))],
                dtype=np.uint8,
            )

    return {
        "evidence_sha256": evidence_sha,
        "valid_color_sha256": EXPECTED_VALID_COLOR_SHA256,
        "input_manifest": input_manifest,
        "evidence_extract_dir": extract_dir,
        "query_id": query_ids_primary,
        "query_scene_code": query_scene_code,
        "source_group_id": source_group_ids,
        "source_idx": source_idx_primary,
        "source_supported": source_supported,
        "V": V_primary,
        "candidate_id": candidate_ids,
        "candidate_scene_code": candidate_scene_code,
        "valid": valid,
        "row_scene_counts": row_scene_counts,
        "top10": top10,
        "observed_source": observed_source,
        "observed_semantic": observed_semantic,
        "ranking_alignment": ranking_alignment,
        "qdf_primary": qdf_primary,
        "smoke": smoke_n is not None,
    }


def save_prepared_inputs(path: Path, data: dict, config_hash: str, input_hash: str) -> None:
    valid = data["valid"]
    atomic_npz(
        path,
        query_id=np.asarray(data["query_id"], dtype="U180"),
        query_scene_code=data["query_scene_code"].astype(np.int8),
        source_group_id=np.asarray(data["source_group_id"], dtype="U220"),
        source_idx=data["source_idx"].astype(np.int32),
        source_supported=data["source_supported"].astype(np.bool_),
        V=data["V"].astype(np.int16),
        candidate_id=np.asarray(data["candidate_id"], dtype="U180"),
        candidate_scene_code=data["candidate_scene_code"].astype(np.int8),
        valid_packed=np.packbits(valid, axis=1, bitorder="big"),
        valid_bit_length=np.asarray([valid.shape[1]], dtype=np.int32),
        row_scene_counts=data["row_scene_counts"].astype(np.int16),
        top10=data["top10"].astype(np.int32),
        observed_source=data["observed_source"].astype(np.uint8),
        observed_semantic=data["observed_semantic"].astype(np.uint8),
        config_hash=np.asarray([config_hash], dtype="U64"),
        input_hash=np.asarray([input_hash], dtype="U64"),
    )


def load_prepared_inputs(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k].copy() for k in z.files}
    n_candidates = int(out["valid_bit_length"].ravel()[0])
    out["valid"] = np.unpackbits(
        out.pop("valid_packed"), axis=1, bitorder="big"
    )[:, :n_candidates].astype(bool)
    out["query_id"] = out["query_id"].astype(str)
    out["source_group_id"] = out["source_group_id"].astype(str)
    out["candidate_id"] = out["candidate_id"].astype(str)
    out["config_hash"] = str(out["config_hash"].ravel()[0])
    out["input_hash"] = str(out["input_hash"].ravel()[0])
    return out


@dataclass
class SwitchState:
    """Sparse constrained bipartite state.

    `adj[row]` stores all currently included global candidate indices.
    `edge_rows[scene]` / `edge_cols[scene]` store FREE edges only.
    The source position of every query is fixed: pinned if supported,
    structural zero if unsupported.
    """

    adj: list[set[int]]
    edge_rows: list[np.ndarray]
    edge_cols: list[np.ndarray]
    source_idx: np.ndarray
    source_supported: np.ndarray
    candidate_scene_code: np.ndarray
    initial_col_degrees: np.ndarray
    initial_row_scene_counts: np.ndarray
    initial_free_adj: list[set[int]]

    @classmethod
    def from_valid(
        cls,
        valid: np.ndarray,
        source_idx: np.ndarray,
        source_supported: np.ndarray,
        candidate_scene_code: np.ndarray,
        initial_row_scene_counts: np.ndarray,
    ) -> "SwitchState":
        n, c = valid.shape
        adj = [set(np.flatnonzero(valid[i]).astype(int).tolist()) for i in range(n)]
        initial_free_adj = []
        edge_rows = [[] for _ in SCENES]
        edge_cols = [[] for _ in SCENES]
        for i in range(n):
            free = set()
            for cand in adj[i]:
                if cand == int(source_idx[i]):
                    if not bool(source_supported[i]):
                        raise RuntimeError("Unsupported source edge unexpectedly present in initial graph")
                    continue
                free.add(cand)
                sc = int(candidate_scene_code[cand])
                edge_rows[sc].append(i)
                edge_cols[sc].append(cand)
            initial_free_adj.append(free)
        er = [np.asarray(x, dtype=np.int32) for x in edge_rows]
        ec = [np.asarray(x, dtype=np.int32) for x in edge_cols]
        return cls(
            adj=adj,
            edge_rows=er,
            edge_cols=ec,
            source_idx=source_idx.astype(np.int32, copy=True),
            source_supported=source_supported.astype(bool, copy=True),
            candidate_scene_code=candidate_scene_code.astype(np.int8, copy=True),
            initial_col_degrees=valid.sum(axis=0).astype(np.int32),
            initial_row_scene_counts=initial_row_scene_counts.astype(np.int16, copy=True),
            initial_free_adj=initial_free_adj,
        )

    @classmethod
    def from_edges(
        cls,
        edge_rows: list[np.ndarray],
        edge_cols: list[np.ndarray],
        source_idx: np.ndarray,
        source_supported: np.ndarray,
        candidate_scene_code: np.ndarray,
        initial_col_degrees: np.ndarray,
        initial_row_scene_counts: np.ndarray,
        initial_free_adj: list[set[int]],
    ) -> "SwitchState":
        n = len(source_idx)
        adj = [set() for _ in range(n)]
        for i in range(n):
            if bool(source_supported[i]):
                adj[i].add(int(source_idx[i]))
        for er, ec in zip(edge_rows, edge_cols):
            for r, c in zip(er.tolist(), ec.tolist()):
                adj[int(r)].add(int(c))
        return cls(
            adj=adj,
            edge_rows=[x.astype(np.int32, copy=True) for x in edge_rows],
            edge_cols=[x.astype(np.int32, copy=True) for x in edge_cols],
            source_idx=source_idx.astype(np.int32, copy=True),
            source_supported=source_supported.astype(bool, copy=True),
            candidate_scene_code=candidate_scene_code.astype(np.int8, copy=True),
            initial_col_degrees=initial_col_degrees.astype(np.int32, copy=True),
            initial_row_scene_counts=initial_row_scene_counts.astype(np.int16, copy=True),
            initial_free_adj=initial_free_adj,
        )

    @property
    def total_free_edges(self) -> int:
        return int(sum(len(x) for x in self.edge_rows))

    def attempt_swap(self, scene: int, rng: np.random.Generator) -> bool:
        er = self.edge_rows[scene]
        ec = self.edge_cols[scene]
        e = len(er)
        if e < 2:
            return False
        p1 = int(rng.integers(e))
        p2 = int(rng.integers(e - 1))
        if p2 >= p1:
            p2 += 1
        r1, c1 = int(er[p1]), int(ec[p1])
        r2, c2 = int(er[p2]), int(ec[p2])
        if r1 == r2 or c1 == c2:
            return False
        # Cross edges must be absent.
        if c2 in self.adj[r1] or c1 in self.adj[r2]:
            return False
        # Every query's source position is fixed to its observed membership.
        if c2 == int(self.source_idx[r1]) or c1 == int(self.source_idx[r2]):
            return False

        self.adj[r1].remove(c1)
        self.adj[r2].remove(c2)
        self.adj[r1].add(c2)
        self.adj[r2].add(c1)
        # Row endpoint stays fixed; candidate endpoints are exchanged.
        ec[p1] = c2
        ec[p2] = c1
        return True

    def run_proposals(
        self,
        target_proposals: int,
        rng: np.random.Generator,
        proposed_by_scene: np.ndarray,
        accepted_by_scene: np.ndarray,
    ) -> None:
        """Advance the Markov chain by a fixed number of proposal transitions.

        Rejected 2x2-switch proposals are genuine self-transitions and MUST
        remain on the chain clock.  Recording states after a fixed number of
        accepted moves would sample the embedded jump chain, whose stationary
        distribution is acceptance-rate weighted rather than the intended
        degree-preserving proposal-chain target when acceptance varies by state.
        """
        if target_proposals <= 0:
            return
        weights = np.asarray([len(x) for x in self.edge_rows], dtype=float)
        active = weights >= 2
        if not active.any():
            raise RuntimeError("No scene has at least two free edges; switch chain cannot move")
        weights[~active] = 0.0
        weights /= weights.sum()
        accepted_before = int(accepted_by_scene.sum())
        for _ in range(int(target_proposals)):
            scene = int(rng.choice(len(SCENES), p=weights))
            proposed_by_scene[scene] += 1
            if self.attempt_swap(scene, rng):
                accepted_by_scene[scene] += 1
        accepted_now = int(accepted_by_scene.sum()) - accepted_before
        if target_proposals >= 10000 and accepted_now == 0:
            raise RuntimeError(
                "Switch chain accepted no move across a long fixed-proposal block; "
                "constrained state may be non-movable"
            )

    def run_accepted_swaps(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "run_accepted_swaps is scientifically deprecated: accepted-move timing "
            "samples the jump chain. Use run_proposals so rejected proposals remain "
            "as self-transitions."
        )

    def invariant_report(self) -> dict:
        n = len(self.adj)
        source_membership = np.asarray(
            [int(self.source_idx[i]) in self.adj[i] for i in range(n)],
            dtype=bool,
        )
        source_mismatches = int(np.count_nonzero(source_membership != self.source_supported))

        row_scene = np.zeros_like(self.initial_row_scene_counts, dtype=np.int16)
        col_deg = np.zeros_like(self.initial_col_degrees, dtype=np.int32)
        edge_count = 0
        for i, cells in enumerate(self.adj):
            edge_count += len(cells)
            for c in cells:
                row_scene[i, int(self.candidate_scene_code[c])] += 1
                col_deg[c] += 1

        row_scene_mismatches = int(
            np.count_nonzero(row_scene != self.initial_row_scene_counts)
        )
        col_degree_mismatches = int(
            np.count_nonzero(col_deg != self.initial_col_degrees)
        )
        total_cardinality_mismatches = int(
            np.count_nonzero(
                row_scene.sum(axis=1) != self.initial_row_scene_counts.sum(axis=1)
            )
        )

        free_total = self.total_free_edges
        overlap = 0
        if free_total:
            for i, cells in enumerate(self.adj):
                # Source is never in initial_free_adj; intersection therefore measures free-edge overlap.
                overlap += len(cells.intersection(self.initial_free_adj[i]))
            overlap_fraction = overlap / free_total
        else:
            overlap_fraction = 1.0

        return {
            "source_membership_mismatches": source_mismatches,
            "row_scene_degree_cell_mismatches": row_scene_mismatches,
            "candidate_column_degree_mismatches": col_degree_mismatches,
            "total_cardinality_query_mismatches": total_cardinality_mismatches,
            "edge_count": edge_count,
            "free_edge_overlap_fraction_with_true_graph": float(overlap_fraction),
        }

    def edge_arrays_copy(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return (
            [x.copy() for x in self.edge_rows],
            [x.copy() for x in self.edge_cols],
        )


def evaluate_null_hits(adj: list[set[int]], top10: np.ndarray) -> np.ndarray:
    """Return [query, model, K] binary null hits for one retained state."""
    n = len(adj)
    hits = np.zeros((n, 2, 3), dtype=np.uint8)
    for i in range(n):
        cells = adj[i]
        for mi in range(2):
            first_hit_rank = 11
            for rank0, cand in enumerate(top10[mi, i].tolist()):
                if int(cand) in cells:
                    first_hit_rank = rank0 + 1
                    break
            hits[i, mi, 0] = first_hit_rank <= 1
            hits[i, mi, 1] = first_hit_rank <= 5
            hits[i, mi, 2] = first_hit_rank <= 10
    return hits


def exact_supported_source_lower_bounds(
    observed_source: np.ndarray, source_supported: np.ndarray
) -> np.ndarray:
    """Exact guaranteed minimum full-population null numerators, [model,K]."""
    return observed_source[source_supported].sum(axis=0).astype(np.int32)


def save_chain_checkpoint(
    path: Path,
    *,
    state: SwitchState,
    rng: np.random.Generator,
    burn_remaining: int,
    retained_done: int,
    rep_full_num: np.ndarray,
    rep_supported_num: np.ndarray,
    query_hit_sums: np.ndarray,
    overlap_seq: np.ndarray,
    proposed_by_scene: np.ndarray,
    accepted_by_scene: np.ndarray,
    config_hash: str,
    input_hash: str,
    chain_id: int,
    chain_seed: int,
) -> None:
    er, ec = state.edge_arrays_copy()
    arrays = {
        "burn_remaining": np.asarray([burn_remaining], dtype=np.int64),
        "retained_done": np.asarray([retained_done], dtype=np.int32),
        "rep_full_num": rep_full_num,
        "rep_supported_num": rep_supported_num,
        "query_hit_sums": query_hit_sums,
        "overlap_seq": overlap_seq,
        "proposed_by_scene": proposed_by_scene,
        "accepted_by_scene": accepted_by_scene,
        "config_hash": np.asarray([config_hash], dtype="U64"),
        "input_hash": np.asarray([input_hash], dtype="U64"),
        "chain_id": np.asarray([chain_id], dtype=np.int16),
        "chain_seed": np.asarray([chain_seed], dtype=np.uint64),
        "rng_state_json": np.asarray(
            [json.dumps(rng.bit_generator.state, sort_keys=True)], dtype="U4096"
        ),
    }
    for si in range(len(SCENES)):
        arrays[f"edge_rows_{si}"] = er[si]
        arrays[f"edge_cols_{si}"] = ec[si]
    atomic_npz(path, **arrays)


def load_chain_checkpoint(path: Path, *, config_hash: str, input_hash: str) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as z:
        if str(z["config_hash"].ravel()[0]) != config_hash:
            raise RuntimeError(f"Checkpoint configuration mismatch: {path}")
        if str(z["input_hash"].ravel()[0]) != input_hash:
            raise RuntimeError(f"Checkpoint input mismatch: {path}")
        out = {k: z[k].copy() for k in z.files}
    return out


def derive_chain_seed(base_seed: int, chain_id: int) -> int:
    ss = np.random.SeedSequence([int(base_seed), 0x534D4E32, int(chain_id)])
    words = ss.generate_state(2, dtype=np.uint64)
    return int(words[0] ^ (words[1] << np.uint64(1)))


def run_chain(
    prepared_path: Path,
    chain_dir: Path,
    *,
    chain_id: int,
    base_seed: int,
    burn_edge_multiples: int,
    thin_edge_multiples: int,
    retained_reps: int,
    checkpoint_every: int,
    config_hash: str,
    input_hash: str,
    smoke: bool = False,
) -> dict:
    """Run/resume one constrained source-membership-matched switch chain."""
    chain_dir = Path(chain_dir)
    chain_dir.mkdir(parents=True, exist_ok=True)
    final_npz = chain_dir / f"CHAIN_{chain_id:02d}_FINAL.npz"
    final_json = chain_dir / f"CHAIN_{chain_id:02d}_FINAL.json"
    checkpoint = chain_dir / f"CHAIN_{chain_id:02d}_CHECKPOINT.npz"

    if final_json.is_file() and final_npz.is_file():
        meta = json.loads(final_json.read_text(encoding="utf-8"))
        if (
            meta.get("status") == "PASS"
            and meta.get("config_hash") == config_hash
            and meta.get("input_hash") == input_hash
            and meta.get("output_sha256") == sha256_file(final_npz)
        ):
            print(f"[chain {chain_id}] REUSE final PASS")
            return meta

    d = load_prepared_inputs(prepared_path)
    valid = d["valid"]
    source_idx = d["source_idx"].astype(np.int32)
    source_supported = d["source_supported"].astype(bool)
    candidate_scene_code = d["candidate_scene_code"].astype(np.int8)
    row_scene_counts = d["row_scene_counts"].astype(np.int16)
    top10 = d["top10"].astype(np.int32)
    observed_source = d["observed_source"].astype(np.uint8)
    n = len(source_idx)
    supported_mask = source_supported
    supported_n = int(supported_mask.sum())

    initial_state = SwitchState.from_valid(
        valid, source_idx, source_supported, candidate_scene_code, row_scene_counts
    )
    initial_free_adj = initial_state.initial_free_adj
    initial_col_degrees = initial_state.initial_col_degrees.copy()

    chain_seed = derive_chain_seed(base_seed, chain_id)
    cp = load_chain_checkpoint(checkpoint, config_hash=config_hash, input_hash=input_hash)

    if cp is None:
        state = initial_state
        rng = np.random.default_rng(chain_seed)
        free_edges = state.total_free_edges
        if free_edges <= 0:
            raise RuntimeError("No free edges after source pinning")
        burn_total = max(1000 if smoke else 10000, burn_edge_multiples * free_edges)
        thin_proposals = max(100 if smoke else 1000, thin_edge_multiples * free_edges)
        burn_remaining = int(burn_total)
        retained_done = 0
        rep_full_num = np.zeros((retained_reps, 2, 3), dtype=np.int32)
        rep_supported_num = np.zeros((retained_reps, 2, 3), dtype=np.int32)
        query_hit_sums = np.zeros((n, 2, 3), dtype=np.uint16)
        overlap_seq = np.full(retained_reps, np.nan, dtype=np.float32)
        proposed_by_scene = np.zeros(len(SCENES), dtype=np.int64)
        accepted_by_scene = np.zeros(len(SCENES), dtype=np.int64)
    else:
        edge_rows = [cp[f"edge_rows_{si}"].astype(np.int32) for si in range(len(SCENES))]
        edge_cols = [cp[f"edge_cols_{si}"].astype(np.int32) for si in range(len(SCENES))]
        state = SwitchState.from_edges(
            edge_rows=edge_rows,
            edge_cols=edge_cols,
            source_idx=source_idx,
            source_supported=source_supported,
            candidate_scene_code=candidate_scene_code,
            initial_col_degrees=initial_col_degrees,
            initial_row_scene_counts=row_scene_counts,
            initial_free_adj=initial_free_adj,
        )
        rng = np.random.default_rng()
        rng.bit_generator.state = json.loads(str(cp["rng_state_json"].ravel()[0]))
        free_edges = state.total_free_edges
        thin_proposals = max(100 if smoke else 1000, thin_edge_multiples * free_edges)
        burn_remaining = int(cp["burn_remaining"].ravel()[0])
        retained_done = int(cp["retained_done"].ravel()[0])
        rep_full_num = cp["rep_full_num"].astype(np.int32)
        rep_supported_num = cp["rep_supported_num"].astype(np.int32)
        query_hit_sums = cp["query_hit_sums"].astype(np.uint16)
        overlap_seq = cp["overlap_seq"].astype(np.float32)
        proposed_by_scene = cp["proposed_by_scene"].astype(np.int64)
        accepted_by_scene = cp["accepted_by_scene"].astype(np.int64)
        print(
            f"[chain {chain_id}] RESUME burn_remaining={burn_remaining}, "
            f"retained={retained_done}/{retained_reps}"
        )

    exact_supported_min = exact_supported_source_lower_bounds(
        observed_source, source_supported
    )

    # Burn-in is checkpointed in deterministic proposal-transition chunks; rejections are self-loops.
    burn_chunk = max(1000, min(100000, max(1, state.total_free_edges * 5)))
    while burn_remaining > 0:
        todo = min(burn_remaining, burn_chunk)
        state.run_proposals(todo, rng, proposed_by_scene, accepted_by_scene)
        burn_remaining -= todo
        save_chain_checkpoint(
            checkpoint,
            state=state,
            rng=rng,
            burn_remaining=burn_remaining,
            retained_done=retained_done,
            rep_full_num=rep_full_num,
            rep_supported_num=rep_supported_num,
            query_hit_sums=query_hit_sums,
            overlap_seq=overlap_seq,
            proposed_by_scene=proposed_by_scene,
            accepted_by_scene=accepted_by_scene,
            config_hash=config_hash,
            input_hash=input_hash,
            chain_id=chain_id,
            chain_seed=chain_seed,
        )
        print(f"[chain {chain_id}] burn-in remaining proposals: {burn_remaining}")

    invariant_checks = 0
    source_pair_checks = 0
    source_mismatch_total = 0
    row_scene_mismatch_total = 0
    col_mismatch_total = 0
    cardinality_mismatch_total = 0
    supported_monotonic_violations = 0
    conservative_lower_bound_violations = 0

    # If resuming after some retained states, previous invariant counters are
    # reconstructed conservatively from the fact that checkpointed retained
    # states could only have been written after passing the gates.
    invariant_checks = retained_done
    source_pair_checks = retained_done * n

    started = time.perf_counter()
    for rep in range(retained_done, retained_reps):
        state.run_proposals(thin_proposals, rng, proposed_by_scene, accepted_by_scene)

        inv = state.invariant_report()
        invariant_checks += 1
        source_pair_checks += n
        source_mismatch_total += inv["source_membership_mismatches"]
        row_scene_mismatch_total += inv["row_scene_degree_cell_mismatches"]
        col_mismatch_total += inv["candidate_column_degree_mismatches"]
        cardinality_mismatch_total += inv["total_cardinality_query_mismatches"]
        if (
            inv["source_membership_mismatches"]
            or inv["row_scene_degree_cell_mismatches"]
            or inv["candidate_column_degree_mismatches"]
            or inv["total_cardinality_query_mismatches"]
        ):
            raise RuntimeError(
                f"Chain {chain_id}, rep {rep}: constrained-degree/source invariant failed: {inv}"
            )
        overlap_seq[rep] = inv["free_edge_overlap_fraction_with_true_graph"]

        hits = evaluate_null_hits(state.adj, top10)
        full_num = hits.sum(axis=0).astype(np.int32)
        supported_num = hits[supported_mask].sum(axis=0).astype(np.int32)
        rep_full_num[rep] = full_num
        rep_supported_num[rep] = supported_num
        query_hit_sums += hits.astype(np.uint16)

        # Exact source-supported monotonicity: every source hit among supported
        # queries must also be a null hit because the source is pinned in every set.
        if np.any(supported_num < exact_supported_min):
            supported_monotonic_violations += 1
            raise RuntimeError(
                f"Chain {chain_id}, rep {rep}: null Recall fell below exact "
                f"source-supported lower bound"
            )

        # Explicit conservative full-population bounds requested by the audit.
        for (model, k), bound in CONSERVATIVE_FULL_LOWER_BOUNDS.items():
            mi = MODEL_NAMES.index(model)
            ki = KS.index(k)
            rate = full_num[mi, ki] / n
            if rate + 1e-15 < bound:
                conservative_lower_bound_violations += 1
                raise RuntimeError(
                    f"Chain {chain_id}, rep {rep}: {model}@{k} null Recall "
                    f"{rate:.6f} < required minimum {bound:.6f}"
                )

        if (rep + 1) % checkpoint_every == 0 or rep + 1 == retained_reps:
            save_chain_checkpoint(
                checkpoint,
                state=state,
                rng=rng,
                burn_remaining=0,
                retained_done=rep + 1,
                rep_full_num=rep_full_num,
                rep_supported_num=rep_supported_num,
                query_hit_sums=query_hit_sums,
                overlap_seq=overlap_seq,
                proposed_by_scene=proposed_by_scene,
                accepted_by_scene=accepted_by_scene,
                config_hash=config_hash,
                input_hash=input_hash,
                chain_id=chain_id,
                chain_seed=chain_seed,
            )
            elapsed = time.perf_counter() - started
            print(
                f"[chain {chain_id}] retained {rep+1}/{retained_reps}; "
                f"elapsed {elapsed:.1f}s; overlap={float(overlap_seq[rep]):.4f}"
            )

    atomic_npz(
        final_npz,
        rep_full_num=rep_full_num,
        rep_supported_num=rep_supported_num,
        query_hit_sums=query_hit_sums,
        overlap_seq=overlap_seq,
        proposed_by_scene=proposed_by_scene,
        accepted_by_scene=accepted_by_scene,
        n_queries=np.asarray([n], dtype=np.int32),
        n_supported=np.asarray([supported_n], dtype=np.int32),
        chain_id=np.asarray([chain_id], dtype=np.int16),
        chain_seed=np.asarray([chain_seed], dtype=np.uint64),
        config_hash=np.asarray([config_hash], dtype="U64"),
        input_hash=np.asarray([input_hash], dtype="U64"),
    )

    proposed_total = int(proposed_by_scene.sum())
    accepted_total = int(accepted_by_scene.sum())
    meta = {
        "status": "PASS",
        "chain_id": chain_id,
        "chain_seed": chain_seed,
        "retained_repetitions": retained_reps,
        "query_count": n,
        "source_supported_query_count": supported_n,
        "free_edges": state.total_free_edges,
        "transition_clock": "fixed_proposals_including_rejected_self_loops",
        "burn_proposals_target": int(max(1000 if smoke else 10000, burn_edge_multiples * state.total_free_edges)),
        "thin_proposals_per_retained_state": int(max(100 if smoke else 1000, thin_edge_multiples * state.total_free_edges)),
        "proposed_swaps": proposed_total,
        "accepted_swaps": accepted_total,
        "acceptance_rate": accepted_total / proposed_total if proposed_total else None,
        "proposed_by_scene": proposed_by_scene.tolist(),
        "accepted_by_scene": accepted_by_scene.tolist(),
        "source_membership_query_rep_checks": source_pair_checks,
        "source_membership_mismatches": source_mismatch_total,
        "row_scene_degree_mismatches": row_scene_mismatch_total,
        "candidate_column_degree_mismatches": col_mismatch_total,
        "total_cardinality_mismatches": cardinality_mismatch_total,
        "source_supported_monotonic_violations": supported_monotonic_violations,
        "conservative_lower_bound_violations": conservative_lower_bound_violations,
        "free_edge_overlap_mean": float(np.nanmean(overlap_seq)),
        "free_edge_overlap_final": float(overlap_seq[-1]),
        "config_hash": config_hash,
        "input_hash": input_hash,
        "output": str(final_npz),
        "output_sha256": sha256_file(final_npz),
    }
    atomic_json(final_json, meta)
    return meta


def split_rhat(chains: np.ndarray) -> float:
    """Basic split-Rhat for shape [chain, draw]."""
    x = np.asarray(chains, dtype=float)
    if x.ndim != 2:
        raise ValueError("split_rhat expects [chain, draw]")
    m, n = x.shape
    half = n // 2
    if half < 2:
        return float("nan")
    y = np.concatenate([x[:, :half], x[:, -half:]], axis=0)
    m2, n2 = y.shape
    chain_means = y.mean(axis=1)
    chain_vars = y.var(axis=1, ddof=1)
    W = chain_vars.mean()
    B = n2 * chain_means.var(ddof=1)
    if W == 0:
        return 1.0 if B == 0 else float("inf")
    var_hat = ((n2 - 1) / n2) * W + B / n2
    return float(math.sqrt(max(var_hat / W, 0.0)))


def autocorr_ess_one_chain(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return float(n)
    x = x - x.mean()
    var = float(np.dot(x, x) / n)
    if var <= 0:
        return float(n)
    rho = []
    for lag in range(1, min(n - 1, 100)):
        ac = float(np.dot(x[:-lag], x[lag:]) / (n - lag) / var)
        rho.append(ac)
    s = 0.0
    for j in range(0, len(rho) - 1, 2):
        pair = rho[j] + rho[j + 1]
        if pair <= 0:
            break
        s += pair
    tau = max(1.0, 1.0 + 2.0 * s)
    return float(n / tau)


def chain_diagnostics(rep_full: np.ndarray) -> list[dict]:
    """rep_full shape [chains, draws, model, K] of proportions."""
    rows = []
    for mi, model in enumerate(MODEL_NAMES):
        for ki, k in enumerate(KS):
            x = rep_full[:, :, mi, ki]
            rhat = split_rhat(x)
            ess = float(sum(autocorr_ess_one_chain(c) for c in x))
            pooled = x.reshape(-1)
            rows.append({
                "model": model,
                "K": k,
                "split_Rhat": rhat,
                "estimated_ESS": ess,
                "pooled_mean": float(pooled.mean()),
                "pooled_sd": float(pooled.std(ddof=1)),
                "naive_MCSE": float(pooled.std(ddof=1) / math.sqrt(len(pooled))),
                "chain_means": [float(v) for v in x.mean(axis=1)],
                "chain_sds": [float(v) for v in x.std(axis=1, ddof=1)],
            })
    return rows


def holm_adjust(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adjusted_sorted[rank] = running
    adjusted = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted.tolist()


def bootstrap_effects(
    *,
    scene_codes: np.ndarray,
    source_group_ids: np.ndarray,
    observed_semantic: np.ndarray,
    null_prob: np.ndarray,
    reps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Paired scene-stratified source-cell-clustered bootstrap.

    Returns (point[6], boot[reps,6]) in model-major/K-major order.
    """
    n = len(scene_codes)
    d = observed_semantic.astype(float) - null_prob.astype(float)
    d6 = d.reshape(n, 6)
    point = d6.mean(axis=0)

    # Cluster sums and cluster sizes, stratified by scene.
    by_scene = {}
    for scene in sorted(set(scene_codes.tolist())):
        idx = np.flatnonzero(scene_codes == scene)
        cluster_map = defaultdict(list)
        for i in idx.tolist():
            cluster_map[str(source_group_ids[i])].append(i)
        sums = []
        sizes = []
        for key in sorted(cluster_map):
            ii = np.asarray(cluster_map[key], dtype=int)
            sums.append(d6[ii].sum(axis=0))
            sizes.append(len(ii))
        by_scene[int(scene)] = (
            np.asarray(sums, dtype=float),
            np.asarray(sizes, dtype=float),
        )

    rng = np.random.default_rng(seed)
    boot = np.empty((reps, 6), dtype=np.float64)
    for b in range(reps):
        total_sum = np.zeros(6, dtype=float)
        total_n = 0.0
        for scene in sorted(by_scene):
            sums, sizes = by_scene[scene]
            nc = len(sizes)
            picks = rng.integers(0, nc, size=nc)
            total_sum += sums[picks].sum(axis=0)
            total_n += sizes[picks].sum()
        boot[b] = total_sum / total_n
    return point, boot


def simultaneous_intervals(point: np.ndarray, boot: np.ndarray, alpha: float = 0.05):
    centered = boot - point[None, :]
    q = float(np.quantile(np.max(np.abs(centered), axis=1), 1.0 - alpha))
    lo = point - q
    hi = point + q
    return lo, hi, q


def pointwise_intervals(boot: np.ndarray, alpha: float = 0.05):
    return (
        np.quantile(boot, alpha / 2, axis=0),
        np.quantile(boot, 1 - alpha / 2, axis=0),
    )


def discover_legacy_null_archive(project_root: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    project_root = Path(project_root)
    preferred = project_root / "CARDINALITY_MATCHED_NULL_CONTROL_FINAL_v1.zip"
    if preferred.is_file():
        return preferred

    candidates = []
    for p in sorted(project_root.rglob("*.zip")):
        if "SOURCE_MEMBERSHIP_MATCHED_SEMANTIC_NULL_CONTROL_FINAL_v2" in p.name:
            continue
        try:
            with zipfile.ZipFile(p, "r") as zf:
                names = zf.namelist()
                if any(Path(n).name == "TABLE_null_control_primary.csv" for n in names):
                    candidates.append(p)
        except Exception:
            continue
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    # The legacy archive is secondary-only and must never block the corrected
    # primary control.  Ambiguous discovery therefore falls back to not bundling
    # a legacy archive; an explicit --legacy-null-zip can still select one.
    return None


def verify_and_copy_legacy_secondary(legacy_zip: Path, out_dir: Path) -> dict:
    if legacy_zip is None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_text(
            out_dir / "README_SECONDARY.md",
            "# Legacy unconditional null not bundled\n\n"
            "No prior unconditional null-control archive was supplied or discovered. "
            "This does not affect the corrected source-membership-matched primary control.\n",
        )
        return {
            "status": "NOT_PROVIDED",
            "legacy_zip": None,
            "legacy_zip_sha256": None,
            "analysis_role": "secondary_optional",
        }
    legacy_zip = Path(legacy_zip)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outer = sha256_file(legacy_zip)
    with zipfile.ZipFile(legacy_zip, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Legacy null ZIP integrity failure: {bad}")
        names = zf.namelist()
        target = None
        for name in names:
            if Path(name).name == "TABLE_null_control_primary.csv":
                target = name
                break
        if target is None:
            raise RuntimeError(
                "Legacy archive does not contain TABLE_null_control_primary.csv"
            )
        raw = zf.read(target)
        atomic_text(
            out_dir / "TABLE_legacy_unconditional_configuration_model_secondary.csv",
            raw.decode("utf-8"),
        )
    shutil.copy2(legacy_zip, out_dir / legacy_zip.name)
    note = (
        "# Legacy unconditional configuration-model null — secondary only\n\n"
        "This archive is retained solely as a secondary unconditional "
        "cardinality/scene/candidate-frequency-matched control. It does **not** "
        "preserve the query-specific source-membership vector and therefore must "
        "not be used as the primary matched-null evidence. The v2 source-membership-"
        "matched constrained-switch analysis is the primary control.\n"
    )
    atomic_text(out_dir / "README_SECONDARY.md", note)
    return {
        "status": "PASS",
        "legacy_zip": str(legacy_zip),
        "legacy_zip_sha256": outer,
        "analysis_role": "secondary_unconditional_does_not_preserve_source_membership",
    }


def manifest_tree(root: Path, exclude: set[str] | None = None) -> list[dict]:
    root = Path(root)
    exclude = exclude or set()
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name not in exclude:
            rows.append({
                "path": p.relative_to(root).as_posix(),
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })
    return rows


def zip_tree(root: Path, out_zip: Path) -> None:
    root = Path(root)
    out_zip = Path(out_zip)
    tmp = out_zip.with_name(out_zip.name + f".tmp.{os.getpid()}")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(root).as_posix())
    with zipfile.ZipFile(tmp, "r") as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"Final ZIP integrity failure: {bad}")
    os.replace(tmp, out_zip)
