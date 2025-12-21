"""
CPU-only helper utilities.

Use these lightweight functions for tasks that don't benefit from GPU
acceleration (string munging, file metadata, simple JSON merges).
Keeping them here avoids accidental GPU execution for small jobs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


def read_lines(path: str | Path) -> List[str]:
    """Read a text file into a list of stripped lines (CPU-only)."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def merge_json_files(paths: Iterable[str | Path]) -> Dict:
    """
    Merge multiple JSON objects from disk into one dictionary.

    Later files override earlier keys. Intended for small config fragments,
    not large datasets.
    """
    merged: Dict = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            continue
        merged.update(data)
    return merged


def count_tokens(text: str, delimiter: str = " ") -> int:
    """Approximate token count by simple delimiter split (CPU-only)."""
    text = text.strip()
    if not text:
        return 0
    return len(text.split(delimiter))


def list_files(root: str | Path, exts: Iterable[str] = (".py",)) -> List[Path]:
    """
    List files under root matching extensions (CPU-only).

    Args:
        root: directory to walk
        exts: iterable of extensions to include
    """
    root_path = Path(root)
    matches: List[Path] = []
    for ext in exts:
        matches.extend(root_path.rglob(f"*{ext}"))
    return matches
