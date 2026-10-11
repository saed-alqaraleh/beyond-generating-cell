#!/usr/bin/env python3
"""Static syntax and notebook hygiene audit for the repository release.

This does not execute experiments and does not infer whether heavy datasets
or third-party checkpoints are available. All normal Python files are parsed;
all notebook cells are validated, and Python cells are compiled after skipping
Colab/IPython shell magics. Historical notebook magics remain documented.
"""
from __future__ import annotations
import ast
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def check_notebook(path: Path) -> tuple[int,int]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"),list):
        raise ValueError(f"Invalid v4 notebook: {path}")
    n=0
    legacy=0
    for ix,cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        n+=1
        source="".join(cell.get("source",[]))
        kept=[]
        for line in source.splitlines():
            if line.lstrip().startswith(("!","%","?")):
                legacy+=1
            else:
                kept.append(line)
        ast.parse("\n".join(kept),filename=f"{path.name}:cell{ix}")
        if cell.get("outputs"):
            raise ValueError(f"Notebook contains saved execution outputs: {path}:cell{ix}")
    return n,legacy


def main() -> int:
    pyfiles=sorted(list((ROOT/"src").rglob("*.py"))+list((ROOT/"scripts").rglob("*.py")))
    notebooks=sorted((ROOT/"notebooks").rglob("*.ipynb"))
    for p in pyfiles:
        ast.parse(p.read_text(encoding="utf-8"),filename=p.as_posix())
    cell_count=magics=0
    for p in notebooks:
        n,m=check_notebook(p)
        cell_count+=n;magics+=m
    print(json.dumps({"status":"PASS","python_files":len(pyfiles),
                      "notebooks":len(notebooks),"parsed_code_cells":cell_count,
                      "archived_ipython_magic_lines":magics},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
