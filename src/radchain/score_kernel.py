"""score_kernel.py — G4CMP step dump -> transport kernel, then throw the dump away.

ONE QUASIPARTICLE BIRTH = ONE COOPER PAIR BROKEN, at the position where it was
born. That is the quantity the chain consumes: energy absorbed
in the pair-breaking film, as a function of lateral offset from the strike.

So the kernel is the radial histogram of first-appearance positions of
BogoliubovQP tracks, normalised per primary phonon. Bins are log-spaced from
30 um to 8 mm, because the interesting structure spans two decades and a linear
binning wastes resolution where the kernel is steep.

Run inside the job directory; see l3_kernel_job.sh.
"""
from __future__ import annotations

import json
import math
import sys

N_BINS = 24
R_MIN_MM = 0.03
R_MAX_MM = 8.0


def bin_edges():
    lo, hi = math.log10(R_MIN_MM), math.log10(R_MAX_MM)
    return [10 ** (lo + (hi - lo) * i / N_BINS) for i in range(N_BINS + 1)]


def score(path: str, x0: float, y0: float):
    """First row of each (event, track) is that quasiparticle's birth."""
    seen = set()
    births = []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if p[3] != "BogoliubovQP":
                continue
            key = (p[1], p[2])
            if key in seen:
                continue
            seen.add(key)
            births.append((float(p[4]), float(p[5]), float(p[6]),
                           float(p[9]), p[-1]))
    edges = bin_edges()
    counts = [0] * N_BINS
    energy = [0.0] * N_BINS
    volumes: dict = {}
    escaped = 0
    for x, y, z, e, vol in births:
        r = math.hypot(x - x0, y - y0)
        volumes[vol] = volumes.get(vol, 0) + 1
        if r < edges[0] or r >= edges[-1]:
            escaped += 1
            continue
        k = min(int(N_BINS * (math.log10(r) - math.log10(R_MIN_MM)) /
                    (math.log10(R_MAX_MM) - math.log10(R_MIN_MM))), N_BINS - 1)
        counts[k] += 1
        energy[k] += e
    return births, edges, counts, energy, volumes, escaped


def main() -> None:
    (src, out, tag, x, y, z, e_eV, n, seed, wall) = sys.argv[1:11]
    x0, y0 = float(x), float(y)
    births, edges, counts, energy, volumes, escaped = score(src, x0, y0)
    n_prim = int(n)
    json.dump({
        "tag": tag, "x_mm": x0, "y_mm": y0, "z_mm": float(z),
        "energy_eV": float(e_eV), "n_primaries": n_prim, "seed": int(seed),
        "wall_s": int(wall),
        "pair_breaks": len(births),
        "pair_breaks_per_primary": len(births) / n_prim,
        "outside_binning": escaped,
        "bin_edges_mm": edges,
        "counts": counts,
        "energy_eV_per_bin": energy,
        "birth_volumes": volumes,
    }, open(out, "w"))
    print(f"scored {tag}: {len(births)} pair breaks "
          f"({len(births)/n_prim:.1f} per primary), {escaped} outside binning")


if __name__ == "__main__":
    main()
