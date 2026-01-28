from __future__ import annotations
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from Liorhybrid.training import UniversalFileReader


@dataclass(frozen=True)
class LoadedInput:
    text: str
    source: str


def load_text_from_source(source: str, *, reader: Optional[UniversalFileReader] = None) -> LoadedInput:
    """Load arbitrary user input into plain text.

    Rules:
    - If `source` is a path to an existing file: read via `UniversalFileReader`.
    - Otherwise: treat `source` as literal text.

    This keeps inference ingestion consistent with training file ingestion.
    """

    src = (source or "").strip()
    if not src:
        return LoadedInput(text="", source="")

    path = Path(src)
    if path.exists() and path.is_file():
        if reader is None:
            reader = UniversalFileReader()

        try:
            text = reader.read_file(str(path))
        except AttributeError:
            # Backwards compatibility: some versions expose `read`.
            text = reader.read(str(path))

        return LoadedInput(text=text, source=str(path))

    return LoadedInput(text=src, source="inline")
