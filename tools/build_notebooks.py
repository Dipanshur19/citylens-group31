"""
build_notebooks.py — converts the cell-marked .py files in notebooks/ into
ready-to-upload Kaggle .ipynb notebooks (one cell per "# %% CELL" block).

Run:  python tools/build_notebooks.py
Output: notebooks/ipynb/*.ipynb  (upload these straight into Kaggle)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "notebooks"
OUT_DIR = SRC_DIR / "ipynb"


def split_cells(text: str) -> list[str]:
    """Split a .py file on lines beginning with '# %%' into cell sources."""
    cells, current = [], []
    for line in text.splitlines():
        if line.startswith("# %%"):
            if current:
                cells.append("\n".join(current).strip("\n"))
            current = [line]
        else:
            current.append(line)
    if current:
        cells.append("\n".join(current).strip("\n"))
    return [c for c in cells if c.strip()]


def to_notebook(cells: list[str]) -> dict:
    nb_cells = []
    for src in cells:
        nb_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [l + "\n" for l in src.split("\n")],
        })
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    py_files = sorted(p for p in SRC_DIR.glob("*.py"))
    for py in py_files:
        cells = split_cells(py.read_text())
        nb = to_notebook(cells)
        out = OUT_DIR / f"{py.stem}.ipynb"
        out.write_text(json.dumps(nb, indent=1))
        print(f"built {out.name}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
