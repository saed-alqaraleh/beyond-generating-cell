from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from article_evidence import ALPHAS
from article_evidence.common import atomic_json, sha256_file
from article_evidence.template_analysis import build_population, build_score_matrices
from article_evidence.runner import verify_reusable
from kpose_diagnostics.ambiguity import compute_relation_queries
from kpose_diagnostics.loading import CellRecord, SceneData


def cell(cid, labels, xy):
    xyz=np.column_stack([np.asarray(xy,dtype=float),np.zeros(len(xy))])
    return CellRecord(cell_id=cid,center=np.asarray([0.,0.,0.]),
                      labels=np.asarray(labels,dtype=object),centroids=xyz,
                      object_ids=np.arange(len(labels)),colors=np.full(len(labels),None,dtype=object))


class ExecutorTests(unittest.TestCase):
    def test_alpha_grid_exact(self):
        self.assertEqual(ALPHAS,(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.98,0.99,1.0))

    def test_population_matches_core(self):
        scene=SceneData("toy",[
            cell("a",["road","building"],[[0,0],[1,0]]),
            cell("b",["road","building","pole"],[[0,0],[0,1],[1,1]])])
        tl,ti,qc,qt=build_population(scene)
        core=compute_relation_queries(scene,90,progress_every=0,log=lambda *_:None)
        self.assertTrue(np.array_equal(qc,core.query_cell_index))
        self.assertEqual(len(qc),4*(2+6))
        self.assertGreater(len(tl),0)

    def test_score_support_matches_core_V(self):
        scene=SceneData("toy",[
            cell("a",["a","b"],[[0,0],[1,0]]),
            cell("b",["a","b"],[[0,0],[0,-1]])])
        tl,ti,qc,qt=build_population(scene)
        with tempfile.TemporaryDirectory() as td:
            meta={"toy":1}
            obj,rel=build_score_matrices(scene,tl,ti,Path(td),meta,log=lambda *_:None)
            v=(rel>0).sum(1).astype(np.int32)[qt]
            core=compute_relation_queries(scene,90,progress_every=0,log=lambda *_:None)
            self.assertTrue(np.array_equal(v,core.query_v))
            self.assertTrue(np.all((obj==0)|(obj==.5)|(obj==1)))
            # Second invocation must reuse atomic chunks and remain identical.
            _,rel2=build_score_matrices(scene,tl,ti,Path(td),meta,log=lambda *_:None)
            self.assertTrue(np.array_equal(rel,rel2))

    def test_reuse_requires_manifest_and_embedding(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); jid="theta_45"; j=root/jid; j.mkdir()
            marker={"ok":True,"returncode":0,"version":"2.2.0","code_hash":"h","status":"success"}
            atomic_json(j/"_JOB_DONE.json",marker)
            result={"reproduction_gate":{"passed":True,"status":"not_applicable_theta"}}
            atomic_json(j/"gt_validity_results.json",result)
            np.savez_compressed(j/"per_query.npz",v=np.array([1]),gt_valid=np.array([True]))
            (j/"strata.csv").write_text("a\n1\n"); (j/"template_v.csv").write_text("a\n1\n"); (j/"run.log").write_text("ok\n")
            files=[]
            for fp in sorted(j.iterdir()):
                files.append({"path":f"{jid}/{fp.name}","size_bytes":fp.stat().st_size,"sha256":sha256_file(fp)})
            fp={"cells":{"sha256":"c"},"poses":{"sha256":"p"}}
            atomic_json(root/"MANIFEST.json",{"code_hash":"h","input_fingerprint":fp,"files":files})
            atomic_json(root/"MERGED_RESULTS.json",{"code_hash":"h","input_fingerprint":fp,
                         "jobs":{jid:{"status":marker,"result":result}}})
            vr=verify_reusable(root,jid,fp,"h")
            self.assertTrue(vr["verified"])
            (j/"run.log").write_text("tampered\n")
            vr2=verify_reusable(root,jid,fp,"h")
            self.assertFalse(vr2["verified"])

    def test_no_skip_gate_in_runner_source(self):
        import inspect, article_evidence.runner as r
        src=inspect.getsource(r.make_job)
        self.assertNotIn('"--skip-gate"',src)


if __name__=="__main__":
    unittest.main()
