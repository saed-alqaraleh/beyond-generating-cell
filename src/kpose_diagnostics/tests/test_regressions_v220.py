"""Regression tests for the v2.2.0 KPOSE reproducibility repair."""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kpose_diagnostics.kpose_predicate import (
    OUTCOMES,
    code_to_label,
    direction_from_offset,
    label_to_code,
    normalize_direction_label,
    verify_against_stored,
)
from kpose_diagnostics.cli.run_gt_validity import effective_reference_mode
from kpose_diagnostics.cli.run_parallel import merge, _classify_returncode
from kpose_diagnostics.loading import _color_text, _TEXT2POS_COLORS, _TEXT2POS_COLOR_NAMES


class DirectionMappingTests(unittest.TestCase):
    def test_all_five_directions(self):
        cases = {
            "on-top": np.array([0.01, 0.01]),
            "east": np.array([0.20, 0.01]),
            "west": np.array([-0.20, 0.01]),
            "north": np.array([0.01, 0.20]),
            "south": np.array([0.01, -0.20]),
        }
        self.assertEqual(set(cases), set(OUTCOMES))
        for want, offset in cases.items():
            self.assertEqual(direction_from_offset(offset), want)

    def test_code_label_roundtrip(self):
        for label in OUTCOMES:
            self.assertEqual(code_to_label(label_to_code(label)), label)

    def test_generator_tie_order_is_y_axis(self):
        # Official Text2Pos uses sequential if-statements; the y-axis tests run
        # last and overwrite x on exact |dx| == |dy| ties.
        self.assertEqual(direction_from_offset([0.2, 0.2]), "north")
        self.assertEqual(direction_from_offset([0.2, -0.2]), "south")

    def test_invalid_label_fails_loudly(self):
        for bad in (None, "", "n", "NORTH-EAST", 99):
            with self.assertRaises(ValueError):
                normalize_direction_label(bad)

    def test_official_color_centres_roundtrip(self):
        for rgb, name in zip(_TEXT2POS_COLORS, _TEXT2POS_COLOR_NAMES):
            self.assertEqual(_color_text(np.tile(rgb, (4, 1))), str(name))

    def test_confusion_does_not_collapse_to_north(self):
        offsets = {
            "on-top": [0.0, 0.0],
            "east": [0.2, 0.0],
            "west": [-0.2, 0.0],
            "north": [0.0, 0.2],
            "south": [0.0, -0.2],
        }
        report = verify_against_stored([(v, k) for k, v in offsets.items()])
        self.assertEqual(report.agreement, 1.0)
        self.assertEqual(report.n_compared, 5)
        for label in OUTCOMES:
            self.assertEqual(report.confusion.get((label, label)), 1)
        self.assertEqual(sum(n for (stored, got), n in report.confusion.items()
                             if got == "north" and stored != "north"), 0)


class GateFailureArtifactTests(unittest.TestCase):
    def test_failed_kpose_gate_writes_complete_diagnostic(self):
        with tempfile.TemporaryDirectory() as td0:
            td = Path(td0)
            cell = {
                "id": "cell_00000", "cell_size": 30.0,
                "bbox_w": np.array([0, 0, 0, 30, 30, 30], dtype=float),
                "objects": [{
                    "id": 0, "label": "building",
                    "xyz": np.array([[0.2, 0.2, 0.5], [0.21, 0.2, 0.5]], dtype=float),
                    "rgb": np.array([[0.4, 0.4, 0.4], [0.4, 0.4, 0.4]], dtype=float),
                }],
            }
            # offset says east but stored label says north -> guaranteed gate fail.
            pose = {
                "cell_id": "cell_00000", "pose_w": np.array([15, 15, 0], dtype=float),
                "descriptions": [{"object_label": "building", "direction": "north",
                                  "offset_closest": np.array([0.2, 0.0]),
                                  "object_color_text": "gray"}],
            }
            cells = td / "cells.pkl"; poses = td / "poses.pkl"; out = td / "out"
            with cells.open("wb") as fh: pickle.dump([cell], fh)
            with poses.open("wb") as fh: pickle.dump([pose], fh)
            r = subprocess.run([
                sys.executable, "-m", "kpose_diagnostics.cli.run_kpose_ambiguity",
                "--cells", str(cells), "--poses", str(poses), "--scene", "fx",
                "--out-dir", str(out), "--min-agreement", "0.95"],
                capture_output=True, text=True)
            self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
            diag_path = out / "verification_failure.json"
            self.assertTrue(diag_path.is_file())
            diag = json.loads(diag_path.read_text())
            required = {"return_code", "failure_reason", "gate_metrics",
                        "confusion_matrix", "command", "version",
                        "input_fingerprint", "code_hash"}
            self.assertTrue(required.issubset(diag), diag.keys())
            self.assertEqual(diag["return_code"], 3)
            self.assertEqual(diag["version"], "2.2.0")
            self.assertIn("north->east", diag["confusion_matrix"])
            self.assertFalse((out / "kpose_ambiguity_results.json").exists())

    def test_failed_full_kpose_gate_writes_complete_diagnostic(self):
        with tempfile.TemporaryDirectory() as td0:
            td = Path(td0)
            cell = {
                "id": "cell_00000", "cell_size": 30.0,
                "bbox_w": np.array([0, 0, 0, 30, 30, 30], dtype=float),
                "objects": [{
                    "id": 0, "label": "building",
                    "xyz": np.array([[0.2, 0.2, 0.5], [0.21, 0.2, 0.5]], dtype=float),
                    "rgb": np.array([[0.4, 0.4, 0.4], [0.4, 0.4, 0.4]], dtype=float),
                }],
            }
            pose = {
                "cell_id": "cell_00000", "pose_w": np.array([15, 15, 0], dtype=float),
                "descriptions": [{"object_label": "building", "direction": "north",
                                  "offset_closest": np.array([0.2, 0.0]),
                                  "object_color_text": "gray"}],
            }
            cells = td / "cells.pkl"; poses = td / "poses.pkl"; out = td / "out"
            with cells.open("wb") as fh: pickle.dump([cell], fh)
            with poses.open("wb") as fh: pickle.dump([pose], fh)
            r = subprocess.run([
                sys.executable, "-m", "kpose_diagnostics.cli.run_kpose_full",
                "--cells", str(cells), "--poses", str(poses), "--scene", "fx",
                "--out-dir", str(out), "--min-agreement", "0.95",
                "--grid-sweep", ""], capture_output=True, text=True)
            self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
            diag = json.loads((out / "verification_failure.json").read_text())
            required = {"return_code", "failure_reason", "gate_metrics",
                        "confusion_matrix", "command", "version",
                        "input_fingerprint", "code_hash"}
            self.assertTrue(required.issubset(diag), diag.keys())
            self.assertEqual(diag["return_code"], 3)
            self.assertFalse((out / "kpose_full_results.json").exists())


class ThetaAndMergeTests(unittest.TestCase):
    def test_theta_reference_policy(self):
        self.assertEqual(effective_reference_mode(90.0, "auto"), "strict")
        for theta in (45.0, 22.5, 10.0):
            self.assertEqual(effective_reference_mode(theta, "auto"), "off")
        self.assertEqual(_classify_returncode(0), "success")
        self.assertEqual(_classify_returncode(4), "expected_reference_mismatch")
        self.assertEqual(_classify_returncode(7), "execution_failure")

    def test_merge_survives_failed_job_and_missing_result(self):
        with tempfile.TemporaryDirectory() as td0:
            td = Path(td0); out = td / "results"; out.mkdir()
            jdir = out / "job_fail"; jdir.mkdir()
            plan = {
                "version": "2.2.0", "code_hash": "abc", "input_fingerprint": {"x": "y"},
                "jobs": [{"jid": "job_fail", "out": str(jdir),
                          "result_file": "missing.json"}],
            }
            (out / "PLAN.json").write_text(json.dumps(plan))
            (jdir / "_JOB_DONE.json").write_text(json.dumps({
                "job": "job_fail", "returncode": 3, "status": "gate_failure",
                "completed": True, "ok": False, "log": str(out / "logs" / "job_fail.log"),
                "result": str(jdir / "missing.json")}))
            (jdir / "verification_failure.json").write_text(json.dumps({"return_code": 3}))
            merged = merge(out)
            entry = merged["jobs"]["job_fail"]
            self.assertEqual(entry["status"]["status"], "gate_failure")
            self.assertEqual(entry["result_error"], "result file missing")
            self.assertIn("verification_failure.json", entry["diagnostics"])
            self.assertTrue((out / "MERGED_RESULTS.json").is_file())
            self.assertTrue((out / "RUN_SUMMARY.txt").is_file())
            self.assertTrue((out / "MANIFEST.json").is_file())
            self.assertTrue(Path(merged["archive"]).is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
