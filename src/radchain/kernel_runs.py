"""One place that knows where the L3 kernel sweep records live, and how to read.

WHY THIS EXISTS
---------------
The sweep used to be **716 individual JSON files** in `bench/rad_l3_kernel/`,
one per run, which was 49% of every tracked file in this repository. They were
read by globbing, and the sweep PARAMETERS were parsed out of the filenames.

Every record already carried a `tag` field equal to its own basename, so the
filenames held no information the records did not. Consolidating them into one
JSONL is therefore lossless, and `tag` still recovers everything a filename
encoded.

IT ALSO FIXES A WORSE BUG THAN FILE COUNT
-----------------------------------------
The old readers used paths relative to the CURRENT WORKING DIRECTORY -- literal
`"bench/rad_l3_kernel/"` strings. That works only when the process happens to be
started from the repository root, and fails silently everywhere else. This
module anchors on the package instead, so it does not care where it is run from.
"""
from __future__ import annotations

import json
from typing import Iterator

from radchain import _data

#: Anchored on the PACKAGE via `radchain._data`, which owns every data
#: location, so it works wherever the package is installed.
KERNEL_DIR = _data.L3
RUNS = _data.RUNS


def iter_runs(prefix: str = "") -> Iterator[dict]:
    """Every recorded run whose `tag` starts with `prefix`, in stable order.

    Order is the order written, which is sorted by tag, so a caller that takes
    "the first" gets the same record every time. The old glob-based readers
    depended on filesystem ordering for that, which was luck rather than a
    guarantee.
    """
    if not RUNS.is_file():
        raise FileNotFoundError(
            f"{RUNS} is missing. It replaced the 716 per-run JSON files that "
            f"used to sit in {KERNEL_DIR}; regenerate it with the G4CMP "
            f"sweeps (scripts/sweeps/) and ingest_kernel_runs.")
    with RUNS.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not prefix or str(rec.get("tag", "")).startswith(prefix):
                yield rec


def load_runs(prefix: str = "") -> list[dict]:
    return list(iter_runs(prefix))
