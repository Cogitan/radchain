"""Code-block study: one Xu et al. [[4,1,2]] block of 5 chips (4 data + ancilla).

Data: analysis/rad_data/block/<layout>_<source>_<seed>.json (run_block_sweep.sh).

Per event, each of the 5 chips is lost with probability P(E_chip); the block
fails from that event with P(>= 2 of 5 lost) (Poisson-binomial). Rates:
  lam        per-chip loss rate (/s)
  G_corr     rate of single events losing >= 2 block chips (/s), computed
             directly, with no pair approximation
  G_ind      Xu's independent term n lam (n+1) lam tau, n = 4, tau = 1 ms
  achieved   lam / (G_ind + G_corr) against one monolithic chip
  claimed    lam / G_ind
Verdict (counts, E* = 0.5 MeV step): events with >= 2 chips, X1 vs XQ50,
exact Poisson intervals.

    python analysis/rad_code_block.py
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy.stats import chi2

import rad_apply_perase as ap

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "rad_data", "block")
LAYOUTS = ("X1", "XQ50", "XQ100")
MUON_FLUX = 1.0 / 60.0
GFLUX = 1.7
TAU = 1e-3
N = 4


def load(L, src):
    ev, n, area, nc = [], 0, None, 5
    for f in sorted(glob.glob(os.path.join(DATA, f"{L}_{src}_*.json"))):
        r = json.load(open(f))
        ev += [e[:5] for e in r["events"]]
        n += r["n_primaries"]
        area = r["gen_area_cm2"]
    if area is None:
        raise SystemExit("no deposit data at {}: copy the per-event Geant4 deposits from the data record (doi:10.5281/zenodo.24412856) to analysis/rad_data/; see README".format(DATA))
    return np.asarray(ev, float).reshape(-1, 5), n, area


def ci(k):
    lo = chi2.ppf(0.025, 2 * k) / 2 if k else 0.0
    return lo, chi2.ppf(0.975, 2 * k + 2) / 2


def block_rates(parts, fn):
    lam = g1 = g2 = 0.0
    for ev, n, area, flux in parts:
        if n == 0 or len(ev) == 0:
            continue
        P = fn(ev)
        al = ap.count_probs(P, kmax=2)
        w = flux * area / n
        lam += P.sum() * w / 5
        g1 += al[:, 1].sum() * w
        g2 += al[:, 2].sum() * w
    g_ind = N * lam * (N + 1) * lam * TAU
    return {"lam_per_h": lam * 3600, "G_corr_per_h": g2 * 3600,
            "p2_block": g2 / g1 if g1 else float("nan"),
            "claimed": lam / g_ind if g_ind else float("inf"),
            "achieved": lam / (g_ind + g2) if (g_ind + g2) else float("inf")}


def main():
    readings = {"flagged >=180 keV": lambda E: (E >= 180).astype(float)}
    for tag in ("d7_conv0.25_echo", "d7_conv0.5_echo", "d7_conv1.0_echo",
                "d7_conv0.5_noecho"):
        P, _ = ap.perase_curve(tag)
        readings[tag] = (lambda E, P=P: np.minimum(1.0, 2.0 * P(E)))
    out = {}
    print("COUNTS at E* = 0.5 MeV (muons): events with >= 2 of 5 chips")
    for L in LAYOUTS:
        ev, n, area = load(L, "muon")
        if n == 0:
            continue
        k = (ev >= 500).sum(axis=1)
        one, two = int((k >= 1).sum()), int((k >= 2).sum())
        lo, hi = ci(two)
        print(f"  {L:6s} muons {n:.2g}: {two}/{one} = {100 * two / one:.4f}% "
              f"[{100 * lo / one:.4f}-{100 * (hi if two else 3.0) / one:.4f}]")
        out[L] = {"counts_0p5": {"two": two, "one": one}}
    print("\nBLOCK under each erasure reading (muons + gamma-lines where run)")
    for name, fn in readings.items():
        line = f"  {name:18s}"
        ref = None
        for L in LAYOUTS:
            mu = load(L, "muon")
            g = load(L, "gamma-lines")
            parts = [(mu[0], mu[1], mu[2], MUON_FLUX)]
            if g[1]:
                parts.append((g[0], g[1], g[2], GFLUX))
            r = block_rates(parts, fn)
            out.setdefault(L, {})[name] = r
            if L == "X1":
                ref = r["G_corr_per_h"]
            line += (f" | {L}: G_corr {r['G_corr_per_h']:.2e}/h "
                     f"(x{ref / r['G_corr_per_h'] if r['G_corr_per_h'] else float('inf'):.0f}) "
                     f"achieved {r['achieved']:.3g} / claimed {r['claimed']:.3g}")
        print(line)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "rad_code_block.json"), "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
