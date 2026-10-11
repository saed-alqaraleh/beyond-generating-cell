"""Tiny end-to-end KPOSE fixture exercising official normalized semantics."""
from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from kpose_diagnostics.kpose_predicate import direction_from_offset


def main() -> int:
    td = Path(tempfile.mkdtemp())
    try:
        pkg = td / "pkg"; pkg.mkdir(parents=True)
        (pkg / "t2p_fy.py").write_text(
            "class Object3d: pass\nclass Cell: pass\nclass Pose: pass\nclass Descr: pass\n",
            encoding="utf-8")
        sys.path.insert(0, str(pkg)); import t2p_fy
        rng = np.random.default_rng(7)
        labels = ["building", "road", "vegetation", "pole", "fence", "sidewalk"]
        cells, poses = [], []
        for ci in range(40):
            c = t2p_fy.Cell(); c.id = f"cell_{ci:05d}"; c.objects = []; c.cell_size = 30.0
            lo = rng.uniform(-400, 400, 2)
            c.bbox_w = np.array([lo[0], lo[1], -2.0, lo[0]+30.0, lo[1]+30.0, 28.0])
            n = 7
            for j in range(n):
                o = t2p_fy.Object3d(); o.id = j
                o.label = labels[int(rng.integers(0, len(labels)))]
                ctr = rng.uniform(0.15, 0.85, 2)
                xy = np.clip(ctr + rng.normal(0, 0.035, (100, 2)), -0.1, 1.1)
                o.xyz = np.column_stack([xy, rng.normal(0.5, 0.05, 100)])
                o.rgb = np.tile(np.array([[0.45, 0.46, 0.43]]), (100, 1))
                c.objects.append(o)
            cells.append(c)

            pose_norm = rng.uniform(0.2, 0.8, 2)
            p = t2p_fy.Pose(); p.cell_id = c.id
            p.pose = np.array([pose_norm[0], pose_norm[1], 0.5])
            p.pose_w = np.array([lo[0]+30*pose_norm[0], lo[1]+30*pose_norm[1], 13.0])
            p.descriptions = []
            for j in rng.choice(n, size=6, replace=False):
                o = c.objects[j]
                d2 = ((o.xyz[:, :2] - pose_norm[None, :]) ** 2).sum(1)
                closest = o.xyz[np.argmin(d2), :2]
                offset = pose_norm - closest
                d = t2p_fy.Descr(); d.object_label = o.label
                d.direction = direction_from_offset(offset)
                d.object_color_text = "gray"; d.offset_closest = offset
                d.offset_center = pose_norm - o.xyz[:, :2].mean(0)
                p.descriptions.append(d)
            poses.append(p)

        (td / "cells").mkdir(); (td / "poses").mkdir()
        pickle.dump(cells, open(td / "cells" / "fx.pkl", "wb"))
        pickle.dump(poses, open(td / "poses" / "fx.pkl", "wb"))
        del sys.modules["t2p_fy"]; sys.path.remove(str(pkg)); shutil.rmtree(pkg)

        r = subprocess.run([
            sys.executable, "-u", "-m", "kpose_diagnostics.cli.run_kpose_ambiguity",
            "--cells", str(td / "cells" / "fx.pkl"),
            "--poses", str(td / "poses" / "fx.pkl"),
            "--scene", "fx", "--out-dir", str(td / "out"), "--grid", "6"],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-4000:]); print(r.stderr[-2000:])
            return 1
        result = json.loads((td / "out" / "kpose_ambiguity_results.json").read_text())
        agreement = result["verification"]["agreement"]
        print(f"agreement={agreement:.6f} exit={r.returncode}")
        if agreement != 1.0 or not result["verification"]["passed"]:
            return 1
        return 0
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
