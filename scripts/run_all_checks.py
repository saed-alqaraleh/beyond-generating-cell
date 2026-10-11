#!/usr/bin/env python3
"""Complete offline publication-quality verification; no raw dataset required.

Test categories:
  1. Frozen archive bytes/code/results integrity (SHA-256)
  2. Notebook/Python syntax audit
  3. Independent query-level statistical evidence audit
  4. Frozen main-manuscript numerical rows
  5. Source-matched null regression suite
  6. KPOSE regression suite
  7. Article-evidence regression suite
  8. Original historical code identity and original core/executor suites
  9. Full original synthetic/cache smoke tests and generator end-to-end gate

Never regenerates results, alters scientific data or bypasses upstream input locks.
"""
from __future__ import annotations
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PY=sys.executable

def check(label: str, argv: list[str], env: dict[str,str]) -> None:
    print(f"\n==== {label} ====", flush=True)
    rc=subprocess.run([PY,*argv],cwd=ROOT,env=env,check=False).returncode
    if rc:
        raise SystemExit(f"CHECK FAILED ({rc}): {label}")


def main() -> int:
    env=os.environ.copy()
    env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
    env['PYTHONDONTWRITEBYTECODE']='1'
    for label,argv in [
        ('Restore frozen score archive (SHA-256)', ['scripts/restore_large_archive.py']),
        ('Frozen data / original source SHA-256', ['scripts/verify_frozen_integrity.py']),
        ('Python and notebook static audit', ['scripts/audit_code_notebooks.py']),
        ('Independent query-level frozen evidence', ['scripts/verify_release.py']),
        ('Paper and numerical-table locks', ['scripts/check_manuscript_alignment.py']),
        ('Source-matched null regression tests', ['-m','source_matched_null.tests']),
        ('KPOSE source regression tests', ['-m','unittest','discover','-s','src/kpose_diagnostics/tests','-t','src','-v']),
        ('Article-evidence source regression tests', ['-m','unittest','article_evidence.tests.test_executor','-v']),
    ]:
        check(label,argv,env)
    with tempfile.TemporaryDirectory(prefix='bgc_original_') as directory:
        stage=['--stage',directory]
        check('Original historical core tests and source-tree lock', ['scripts/run_historical.py','core-tests',*stage],env)
        check('Original historical executor tests and source-tree lock', ['scripts/run_historical.py','executor-tests',*stage],env)
        core_path = str(Path(directory)/'verified_core'/'kpose_diagnostics_v2.2.0_repaired')
        executor_path = str(Path(directory)/'article_evidence_v220_runner')
        historical_env = env.copy()
        historical_env['PYTHONPATH'] = os.pathsep.join([core_path, executor_path, historical_env.get('PYTHONPATH','')])
        check('Original synthetic input/cache smoke (37 cases)', ['-m','kpose_diagnostics.tests.smoke_test'],historical_env)
        check('Original generator exact-direction end-to-end gate', ['-m','kpose_diagnostics.tests.kpose_predicate_e2e'],historical_env)
    print('\nALL VERIFICATION STAGES PASSED (frozen research results unchanged)',flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
