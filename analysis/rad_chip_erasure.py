"""Chip-scale correlated erasure: rates, p2, and Xu's lifetime with and without it.

Data: analysis/rad_data/chips/<layout>_<source>_<seed>.json (run_chip_sweep.sh).

A chip is ERASED when one event deposits >= E* in it. Per layout and E*:
  lam   single-chip erasure rate, per chip (/s)
  R1    rate of events erasing >= 1 chip, per array (/s)
  R2    rate of events erasing >= 2 chips, per array
  p2    R2 / R1
Xu et al. (PRL 129, 240502) for an [[n,1,d]] code over chips of this kind:
  Gamma_ind  = n lam [(n+1) lam tau]^(d-1) / (d-1)!
  Gamma_corr ~ n lam p2 (d = 2); n lam p3 + n lam p2 (n+1) lam tau (d = 3);
               p_k = P(>= k chips | >= 1), using the ARRAY's
               conditional (chips of one code are neighbours in a 3 x 3 array;
               stated as an approximation, it is an upper-side estimate when a
               code uses fewer than all 9 chips)
  suppression = lam / (Gamma_ind + Gamma_corr) against one monolithic chip.

    python analysis/rad_chip_erasure.py [layout ...]
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "rad_data", "chips")
E_STAR_KEV = (300.0, 500.0, 1000.0, 2000.0)
GAMMA_FLUX = (1.0, 1.7, 3.2)
MUON_FLUX = 1.0 / 60.0           # sea level, /cm2/s, horizontal plane
TAUS = (1e-4, 1e-3)
CODES = ((4, 2), (7, 3))


def load(layout: str) -> dict:
    acc = defaultdict(lambda: [[], 0, None, None])
    for f in sorted(glob.glob(os.path.join(DATA, f"{layout}_*.json"))):
        r = json.load(open(f))
        a = acc[r["source"]]
        a[0].extend(r["events"])
        a[1] += r["n_primaries"]
        a[2] = r["gen_area_cm2"]
        a[3] = r["n_chips"]
    if not acc:
        raise SystemExit("no deposit data at {}: copy the per-event Geant4 deposits from the data record (doi:10.5281/zenodo.24412856) to analysis/rad_data/; see README".format(DATA))
    return {s: (np.asarray(ev, dtype=float), n, area, nc)
            for s, (ev, n, area, nc) in acc.items()}


def counts(ev: np.ndarray, nc: int, e_star: float) -> np.ndarray:
    """Number of chips erased in each stored event."""
    if len(ev) == 0:
        return np.zeros(0, dtype=int)
    return (ev[:, :nc] >= e_star).sum(axis=1)


def rates(data: dict, gamma_src: str, gflux: float, e_star: float) -> dict:
    out = {"lam": 0.0, "R1": 0.0, "R2": 0.0, "R3": 0.0,
           "n_R2": 0, "n_R3": 0, "parts": {}}
    nc = None
    for src, flux in ((gamma_src, gflux), ("muon", MUON_FLUX)):
        if src not in data:
            continue
        ev, n, area, nc = data[src]
        k = counts(ev, nc, e_star)
        w = flux * area / n                       # events per second each
        part = {"lam": k.sum() * w / nc, "R1": (k >= 1).sum() * w,
                "R2": (k >= 2).sum() * w, "R3": (k >= 3).sum() * w,
                "n_R1": int((k >= 1).sum()), "n_R2": int((k >= 2).sum()),
                "n_R3": int((k >= 3).sum())}
        out["parts"][src] = part
        for key in ("lam", "R1", "R2", "R3"):
            out[key] += part[key]
        out["n_R2"] += part["n_R2"]
        out["n_R3"] += part["n_R3"]
    out["p2"] = out["R2"] / out["R1"] if out["R1"] else float("nan")
    out["p3"] = out["R3"] / out["R1"] if out["R1"] else float("nan")
    return out


def xu(lam: float, p2: float, p3: float, n: int, d: int, tau: float) -> dict:
    """d = 2: one event erasing >= 2 chips. d = 3: one event erasing >= 3,
    PLUS a 2-chip event with an independent erasure inside the window
    (n lam p2 x (n+1) lam tau). Higher mixed orders are dropped."""
    g_ind = n * lam * ((n + 1) * lam * tau) ** (d - 1) / math.factorial(d - 1)
    if d == 2:
        g_cor = n * lam * p2
    elif d == 3:
        g_cor = n * lam * p3 + n * lam * p2 * (n + 1) * lam * tau
    else:
        raise ValueError(d)
    tot = g_ind + g_cor
    return {"ind": g_ind, "corr": g_cor,
            "corr_over_ind": g_cor / g_ind if g_ind else float("inf"),
            "suppression_claimed": lam / g_ind if g_ind else float("inf"),
            "suppression_achieved": lam / tot if tot else float("inf")}


def report(layout: str) -> list[dict]:
    data = load(layout)
    if not data:
        print(f"[{layout}] no data")
        return []
    print(f"\n=== {layout} ===")
    for s, (ev, n, area, nc) in data.items():
        print(f"  {s:13s} primaries {n:.3g}  stored {len(ev)}  chips {nc}")
    rows = []
    for gsrc in [s for s in data if s.startswith("gamma")]:
        for gf in GAMMA_FLUX:
            for e in E_STAR_KEV:
                r = rates(data, gsrc, gf, e)
                if not r["R1"]:
                    continue
                mu = r["parts"].get("muon", {})
                line = (f"  {gsrc:13s} flux {gf:3.1f} E* {e / 1000:3.1f} MeV: "
                        f"lam {r['lam'] * 3600:7.2f}/h/chip  "
                        f"p2 {r['p2']:.2e} (n={r['n_R2']}, mu {mu.get('n_R2', 0)})  "
                        f"p3 {r['p3']:.1e} (n={r['n_R3']})")
                print(line)
                row = {"layout": layout, "gamma": gsrc, "gflux": gf,
                       "e_star_keV": e, **{k: v for k, v in r.items()
                                           if k != "parts"},
                       "muon_n_R2": mu.get("n_R2", 0), "xu": {}}
                for n_, d_ in CODES:
                    for tau in TAUS:
                        x = xu(r["lam"], r["p2"], r["p3"], n_, d_, tau)
                        row["xu"][f"[[{n_},1,{d_}]] tau={tau * 1e3:g}ms"] = x
                rows.append(row)
    return rows


def summary(rows: list[dict]) -> None:
    print("\nXu [[4,1,2]], tau 1 ms, gamma-lines, flux 1.7 (middle bracket):")
    for r in rows:
        if r["gamma"] != "gamma-lines" or r["gflux"] != 1.7:
            continue
        x = r["xu"]["[[4,1,2]] tau=1ms"]
        x3 = r["xu"]["[[7,1,3]] tau=1ms"]
        print(f"  {r['layout']} E* {r['e_star_keV'] / 1000:3.1f} MeV: "
              f"corr/ind {x['corr_over_ind']:9.1f}  suppression claimed "
              f"{x['suppression_claimed']:9.3g} achieved "
              f"{x['suppression_achieved']:9.3g} | [[7,1,3]] claimed "
              f"{x3['suppression_claimed']:9.3g} achieved "
              f"{x3['suppression_achieved']:9.3g}")


if __name__ == "__main__":
    layouts = sys.argv[1:] or ["A1", "B3", "V1", "S1", "A3"]
    rows = []
    for L in layouts:
        rows += report(L)
    summary(rows)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(rows, open(os.path.join(HERE, "results", "rad_chip_erasure.json"),
                         "w"), indent=1, default=float)
