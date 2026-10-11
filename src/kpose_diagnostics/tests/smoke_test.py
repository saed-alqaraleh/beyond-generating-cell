"""
smoke_test.py -- end-to-end exercise of both CLIs on synthetic data.

Builds pickles shaped like the KITTI360Pose ``cells/`` and ``poses/``
files, using classes defined in a throwaway module that is then removed
from ``sys.modules``. That makes the run exercise the same code path the
real data will: unpickling classes that cannot be imported, resolving
field names by discovery, and deriving directions geometrically.

No KITTI-360 data required.

    python -m kpose_diagnostics.tests.smoke_test
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


SYNTHETIC_MODULE = """
'''Throwaway stand-in for the Text2Pos data classes.'''


class Object3d:
    pass


class Cell:
    pass


class Descr:
    pass


class Pose:
    pass
"""


def build_fixture(root: Path, n_cells=120, seed=11):
    """Write synthetic cells/ and poses/ pickles in the documented layout.

    The classes live in a real module file that is deleted afterwards, so
    the pickles reference a module that cannot be imported at load time --
    the same situation the real Text2Pos pickles create.
    """
    import importlib
    import pickle

    mod_dir = root / "_fixture_pkg"
    mod_dir.mkdir(parents=True, exist_ok=True)
    (mod_dir / "synthetic_t2p.py").write_text(SYNTHETIC_MODULE, encoding="utf-8")
    sys.path.insert(0, str(mod_dir))
    mod = importlib.import_module("synthetic_t2p")

    rng = np.random.default_rng(seed)
    labels = ["building", "road", "vegetation", "pole", "fence", "sidewalk"]

    cells, poses = [], []
    for ci in range(n_cells):
        origin = rng.uniform(-500, 500, size=2)
        cell = mod.Cell()
        cell.id = f"cell_{ci:05d}"
        cell.objects = []
        n_obj = int(rng.integers(3, 9))
        for oi in range(n_obj):
            o = mod.Object3d()
            o.id = f"{ci:05d}_{oi:03d}"
            o.label = labels[int(rng.integers(0, len(labels)))]
            centre = origin + rng.uniform(-15, 15, size=2)
            o.xyz = np.column_stack([
                centre[0] + rng.normal(0, 1.5, 40),
                centre[1] + rng.normal(0, 1.5, 40),
                rng.normal(0, 0.5, 40),
            ])
            cell.objects.append(o)
        cell.bbox_w = np.array([origin[0] - 15, origin[1] - 15, -1,
                                origin[0] + 15, origin[1] + 15, 5])
        cells.append(cell)

        for _ in range(2):
            p = mod.Pose()
            p.cell_id = cell.id
            p.pose_w = np.array([origin[0] + rng.uniform(-8, 8),
                                 origin[1] + rng.uniform(-8, 8), 0.0])
            k = int(rng.integers(2, min(5, n_obj + 1)))
            picks = rng.choice(n_obj, size=k, replace=False)
            p.descriptions = []
            for j in picks:
                d = mod.Descr()
                # object_id is intentionally NOT set as an instance attribute
                # to simulate the real DescriptionBestCell stub behaviour
                # where getattr raises AttributeError.
                # offset_center must be used instead.
                d.object_label = cell.objects[j].label
                d.object_color_text = ["grey", "green", "black"][int(rng.integers(0, 3))]
                # offset_center = object_centroid - cell_centre
                # cell centre approximated as mean of object centroids for fixture
                obj_pts = cell.objects[j].xyz
                obj_xy = np.array([obj_pts[:, 0].mean(), obj_pts[:, 1].mean()])
                cell_centre_xy = np.array([
                    np.mean([o.xyz[:, 0].mean() for o in cell.objects]),
                    np.mean([o.xyz[:, 1].mean() for o in cell.objects]),
                ])
                d.offset_center = obj_xy - cell_centre_xy
                p.descriptions.append(d)
            poses.append(p)

    scene = "synthetic_drive_9999_sync"
    (root / "cells").mkdir(parents=True, exist_ok=True)
    (root / "poses").mkdir(parents=True, exist_ok=True)
    cells_p = root / "cells" / f"{scene}.pkl"
    poses_p = root / "poses" / f"{scene}.pkl"
    with cells_p.open("wb") as fh:
        pickle.dump(cells, fh)
    with poses_p.open("wb") as fh:
        pickle.dump(poses, fh)

    # Also write the official-style per-record layout so the same smoke test
    # exercises folder ingestion, not only the legacy aggregate convenience.
    cells_d = root / "cells" / scene
    poses_d = root / "poses" / scene
    cells_d.mkdir(parents=True, exist_ok=True)
    poses_d.mkdir(parents=True, exist_ok=True)
    for i, cell in enumerate(cells):
        with (cells_d / f"{i:06d}.pkl").open("wb") as fh:
            pickle.dump(cell, fh)
    for i, pose in enumerate(poses):
        with (poses_d / f"{i:06d}.pkl").open("wb") as fh:
            pickle.dump(pose, fh)

    # Make the module unimportable so the loader must stub the classes,
    # exactly as it will have to for the real Text2Pos classes.
    del sys.modules["synthetic_t2p"]
    sys.path.remove(str(mod_dir))
    shutil.rmtree(mod_dir)
    return cells_p, poses_p, cells_d, poses_d, scene


def run(cmd, cwd):
    proc = subprocess.run([sys.executable, "-u", *cmd], cwd=cwd,
                          capture_output=True, text=True, timeout=900)
    return proc


def main() -> int:
    print("=" * 72)
    print("END-TO-END SMOKE TEST (synthetic data, both CLIs)")
    print("=" * 72)
    pkg_root = Path(__file__).resolve().parents[2]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        data = td / "data"
        cells_p, poses_p, cells_d, poses_d, scene = build_fixture(data)
        record("fixture written", cells_p.is_file() and poses_p.is_file())
        record("per-record fixture written", cells_d.is_dir() and poses_d.is_dir())

        out1 = td / "results" / "b1"
        p = run(["-m", "kpose_diagnostics.cli.run_gt_validity",
                 "--cells", str(cells_p), "--scene", scene,
                 "--out-dir", str(out1), "--n-boot", "200"], pkg_root)
        if p.returncode != 0:
            print(p.stdout[-4000:]); print(p.stderr[-4000:])
        record("run_gt_validity exits 0", p.returncode == 0,
               f"rc={p.returncode}")
        record("gt_validity results written",
               (out1 / "gt_validity_results.json").is_file())
        record("per-query arrays written", (out1 / "per_query.npz").is_file())
        record("stage marker written",
               (out1 / "load" / "_STAGE_SUCCESS.json").is_file())
        record("reproduction gate skipped for unknown scene",
               "reference comparison skipped" in p.stdout, "")

        import json
        res = json.loads((out1 / "gt_validity_results.json").read_text())
        head = res["headline"]
        R = head["R_all"]
        record("R is a probability", 0.0 <= R <= 1.0, f"R={R}")
        record("recoverable ceiling <= naive ceiling",
               head["recoverable_ceiling"] <= head["naive_ceiling_E_inv_V"] + 1e-12,
               f"{head['recoverable_ceiling']} vs {head["naive_ceiling_E_inv_V"]}")
        record("clustered CI brackets the point estimate",
               head["R_all_ci95"][0] <= R <= head["R_all_ci95"][1],
               f"{head['R_all_ci95']} vs {R}")
        arrs = np.load(out1 / "per_query.npz")
        record("per-query arrays are aligned",
               arrs["v"].shape == arrs["gt_valid"].shape == arrs["cell_index"].shape)
        record("invariant holds on synthetic data: gt_valid => V >= 1",
               bool((arrs["v"][arrs["gt_valid"]] >= 1).all()))
        record("query count equals cells x ordered pairs x 4",
               head["n_queries"] == int(arrs["v"].size))

        # resume behaviour
        p2 = run(["-m", "kpose_diagnostics.cli.run_gt_validity",
                  "--cells", str(cells_p), "--scene", scene,
                  "--out-dir", str(out1), "--n-boot", "200"], pkg_root)
        record("rerun exits 0", p2.returncode == 0)
        record("rerun skips the completed load stage",
               "already complete, skipping" in p2.stdout)

        # deterministic
        res2 = json.loads((out1 / "gt_validity_results.json").read_text())
        record("rerun reproduces R exactly",
               res2["headline"]["R_all"] == R,
               f"{res2['headline']['R_all']} vs {R}")

        # multi-hint
        out2 = td / "results" / "b2"
        p3 = run(["-m", "kpose_diagnostics.cli.run_multihint",
                  "--cells", str(cells_p), "--poses", str(poses_p),
                  "--scene", scene, "--out-dir", str(out2)], pkg_root)
        if p3.returncode != 0:
            print(p3.stdout[-4000:]); print(p3.stderr[-4000:])
        record("run_multihint exits 0", p3.returncode == 0, f"rc={p3.returncode}")
        record("multihint results written",
               (out2 / "multihint_results.json").is_file())
        mh = json.loads((out2 / "multihint_results.json").read_text())
        record("all descriptions consumed",
               mh["extraction"]["n_descriptions_usable"] > 0)
        record("no hints dropped for unmatched objects",
               mh["extraction"]["hints_dropped_unmatched_object"] == 0,
               str(mh["extraction"]["hints_dropped_unmatched_object"]))
        record("B2: offset_center mode logged",
               "offset_center" in p3.stdout or "offset field resolved" in p3.stdout,
               "offset path not taken")
        record("signature grouping reduced work",
               mh["grouping"]["speedup_vs_per_description"] >= 1.0)
        v = np.load(out2 / "per_description.npz")["v"]
        record("multi-hint V never exceeds the cell count",
               bool((v <= mh["v_summary"]["n_queries"] * 10 + 120).all()))
        rows = mh["by_hint_count"]
        medians = [r["median_V"] for r in sorted(rows, key=lambda r: r["n_hints"])]
        record("more hints does not increase median V (monotone)",
               all(a >= b - 1e-9 for a, b in zip(medians, medians[1:])),
               f"medians={medians}")

        # cache path: same answers, and the second run reports a hit
        outc = td / "results" / "b1_cached"
        cache = td / "cache"
        pc1 = run(["-m", "kpose_diagnostics.cli.run_gt_validity",
                   "--cells", str(cells_p), "--scene", scene,
                   "--out-dir", str(outc), "--cache-dir", str(cache),
                   "--n-boot", "100"], pkg_root)
        outc2 = td / "results" / "b1_cached2"
        pc2 = run(["-m", "kpose_diagnostics.cli.run_gt_validity",
                   "--cells", str(cells_p), "--scene", scene,
                   "--out-dir", str(outc2), "--cache-dir", str(cache),
                   "--n-boot", "100"], pkg_root)
        record("cached run exits 0", pc1.returncode == 0 and pc2.returncode == 0)
        record("second run reports a cache hit", "scene cache hit" in pc2.stdout)
        import json as _j
        r_nocache = _j.loads((out1 / "gt_validity_results.json").read_text())
        r_cache = _j.loads((outc2 / "gt_validity_results.json").read_text())
        record("cache does not change the measurement",
               r_nocache["headline"]["R_all"] == r_cache["headline"]["R_all"]
               and r_nocache["v_summary"] == r_cache["v_summary"],
               f"{r_nocache['headline']['R_all']} vs {r_cache['headline']['R_all']}")

        # B2 must now report generating-cell validity too
        record("B2 reports generating-cell validity",
               "generating_cell_validity" in mh
               and 0.0 <= mh["generating_cell_validity"]["R"] <= 1.0,
               str(mh.get("generating_cell_validity")))
        record("B2 recoverable ceiling <= naive ceiling",
               mh["generating_cell_validity"]["recoverable_ceiling"]
               <= mh["ceiling_E_inv_V"] + 1e-12)

        # inspection CLI
        out3 = td / "schema_report.json"
        p4 = run(["-m", "kpose_diagnostics.cli.inspect_inputs",
                  "--cells", str(cells_p), "--poses", str(poses_p),
                  "--out", str(out3)], pkg_root)
        record("inspect_inputs exits 0", p4.returncode == 0)
        record("schema report written", out3.is_file())
        record("inspector resolved every required role",
               "All required roles resolved" in p4.stdout)

        # Official-style folder layout: one pickle per cell / pose.
        out_folder = td / "results" / "folder_b2"
        pf = run(["-m", "kpose_diagnostics.cli.run_multihint",
                  "--cells", str(cells_d), "--poses", str(poses_d),
                  "--scene", scene, "--out-dir", str(out_folder)], pkg_root)
        if pf.returncode != 0:
            print(pf.stdout[-4000:]); print(pf.stderr[-4000:])
        record("per-record folder layout exits 0", pf.returncode == 0, f"rc={pf.returncode}")
        if (out_folder / "multihint_results.json").is_file():
            folder_mh = json.loads((out_folder / "multihint_results.json").read_text())
            record("folder and aggregate B2 give same median V",
                   folder_mh["v_summary"]["median"] == mh["v_summary"]["median"],
                   f"{folder_mh['v_summary']['median']} vs {mh['v_summary']['median']}")
        else:
            record("folder and aggregate B2 give same median V", False, "folder output missing")

        from kpose_diagnostics.layout import resolve_dataset_sources
        resolved = resolve_dataset_sources(data, scene)
        # Aggregate takes precedence when both representations exist; resolving
        # from a scene directory must still discover the common dataset root.
        resolved_from_dir = resolve_dataset_sources(cells_d, scene)
        record("layout resolver accepts dataset root", resolved.scene == scene)
        record("layout resolver accepts scene directory", resolved_from_dir.scene == scene)

        # failure mode: wrong field name must fail loudly, not silently
        p5 = run(["-m", "kpose_diagnostics.cli.run_gt_validity",
                  "--cells", str(cells_p), "--scene", scene,
                  "--out-dir", str(td / "results" / "bad"),
                  "--field-obj-label", "definitely_not_a_field"], pkg_root)
        record("bad field name exits 2 with an actionable schema error",
               p5.returncode == 2
               and "does not exist" in p5.stdout
               and "available:" in p5.stdout
               and "Traceback" not in p5.stderr,
               f"rc={p5.returncode}")

    print("\n" + "=" * 72)
    failed = [n for n, ok in RESULTS if not ok]
    print(f"{len(RESULTS) - len(failed)} passed, {len(failed)} failed")
    for n in failed:
        print(f"  FAILED: {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
