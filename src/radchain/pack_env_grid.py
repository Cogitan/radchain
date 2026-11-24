"""pack_env_grid.py -- turn run_env_grid.sh outputs into the shipped grid.

    python -m radchain.pack_env_grid <runs_dir>

`<runs_dir>` holds `<material>_<source>_<t>_<tag>.json` from run_env_grid.sh.
Writes `radchain/data/env_grid/<material>.npz` (grid points only) and
`bench/rad_env_grid_holdout/<material>.npz` (held-out points, never shipped and
never read by the interpolator).

Silicon at 0.3 / 0.5 / 0.725 mm is taken from `data/env/` unchanged, files
concatenated in the same sorted order the earlier reader used, so a request
at those thicknesses resamples exactly the same pool as the original reader.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

from radchain import _data
from radchain.env_grid import GRID_MM, HOLDOUT_MM, SOURCES

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOLDOUT_DIR = os.path.join(REPO, "bench", "rad_env_grid_holdout")
LEGACY_SI_MM = (0.3, 0.5, 0.725)


def _key(source: str, t: float) -> str:
    return "{}|{:g}".format(source, t)


def _collect(runs_dir: str) -> dict:
    """{(material, source, t): [(path, record), ...]} sorted by path."""
    out = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        name = os.path.basename(path)
        material = next(m for m in ("G4_ALUMINUM_OXIDE", "G4_Si")
                        if name.startswith(m + "_"))
        rest = name[len(material) + 1:]
        source, t, _tag = rest[:-len(".json")].rsplit("_", 2)
        with open(path) as fh:
            rec = json.load(fh)
        assert rec["source"] == source and abs(rec["thickness_mm"] - float(t)) < 1e-9
        assert rec.get("material", "G4_Si") == material, path
        out[(material, source, float(t))].append(rec)
    return out


def _pack(records: list[dict]) -> tuple[np.ndarray, int]:
    dep = np.concatenate([np.asarray(r["deposits_keV"], dtype=np.float32)
                          for r in records])
    return dep, int(sum(r["n_primaries"] for r in records))


def main(runs_dir: str) -> None:
    runs = _collect(runs_dir)
    for material in ("G4_Si", "G4_ALUMINUM_OXIDE"):
        for points, where in ((GRID_MM, os.path.join(str(_data.DATA), "env_grid")),
                              (HOLDOUT_MM, HOLDOUT_DIR)):
            arrays = {}
            for t in points:
                for source in SOURCES:
                    if material == "G4_Si" and t in LEGACY_SI_MM:
                        recs = []
                        for f in _data.env_files(source, t):
                            with open(f) as fh:
                                recs.append(json.load(fh))
                    else:
                        recs = runs.get((material, source, t), [])
                    if not recs:
                        raise SystemExit("missing {} {} {:g} mm".format(
                            material, source, t))
                    dep, n = _pack(recs)
                    arrays[_key(source, t) + "|deposits_keV"] = dep
                    arrays[_key(source, t) + "|n_primaries"] = np.array(n)
                    arrays[_key(source, t) + "|seeds"] = np.array(
                        [r["seed"] for r in recs])
            os.makedirs(where, exist_ok=True)
            out = os.path.join(where, material + ".npz")
            np.savez_compressed(out, **arrays)
            print("wrote", out, "{:.1f} MB".format(os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main(sys.argv[1])
