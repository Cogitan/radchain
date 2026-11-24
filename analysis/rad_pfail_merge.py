"""Merge chunked rad_pfail_physics outputs (results/rad_pfail_<tag>_partNN.json,
e.g. from runs with different --seed and --tag <tag>_partNN) into one
results/rad_pfail_<tag>.json that rad_apply_perase.perase_curve can read.
Parts must agree on d, shots, conv, echo, K, backend and phonon geometry.

    python analysis/rad_pfail_merge.py d11_conv0.5_echo [d5_conv0.5_echo ...]
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def merge(tag):
    parts = sorted(glob.glob(os.path.join(HERE, "results", f"rad_pfail_{tag}_part*.json")))
    if not parts:
        raise SystemExit(f"no parts for {tag}")
    rows, secs, head = [], 0.0, None
    for p in parts:
        r = json.load(open(p))
        keys = ("d", "shots", "conv", "echo", "p_base", "k_khz_per_keV",
                "backend", "phonon")
        head = head or {k: r.get(k) for k in keys}
        for k in keys:
            if r.get(k) != head[k]:
                raise SystemExit(f"{p}: {k} differs ({r[k]} vs {head[k]})")
        for e in r["events"]:
            e = dict(e)
            e["i"] = len(rows)
            e["part"] = os.path.basename(p)
            rows.append(e)
        secs += r.get("seconds", 0.0)
    out = dict(head, events=rows, seconds=secs, merged_from=[os.path.basename(p) for p in parts])
    path = os.path.join(HERE, "results", f"rad_pfail_{tag}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, path)
    print(f"{tag}: {len(parts)} parts, {len(rows)} events -> {path}")


if __name__ == "__main__":
    for t in sys.argv[1:]:
        merge(t)
