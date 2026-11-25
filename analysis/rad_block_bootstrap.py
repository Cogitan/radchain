"""Bootstrap intervals on the block's G_corr reduction (X1 / XQ50) under each
erasure reading (mock-referee item: weighted readings had no intervals).

Resamples Geant4 events with replacement within each (layout, source) set and
recomputes G_corr with rad_code_block.block_rates, so the interval reflects how
few large-deposit events drive each weighted reading. Uses the pooled mono
data by default; --data DIR points at another directory (e.g. the spectrum
set assembled with the existing gamma files).

    python analysis/rad_block_bootstrap.py [--n 300] [--data DIR] [--seed 1]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import rad_apply_perase as ap
import rad_code_block as cb

HERE = os.path.dirname(os.path.abspath(__file__))


DISTANCE_TAGS = ("d5_conv0.5_echo", "d7_conv0.5_echo", "d11_conv0.5_echo",
                 "d7_conv1.0_echo", "d11_conv1.0_echo", "d7_conv0.5_noecho",
                 "d11_conv0.5_noecho")


def readings(distance=False):
    r = {} if distance else {"flagged >=180 keV": lambda E: (E >= 180).astype(float)}
    for tag in (DISTANCE_TAGS if distance else
                ("d7_conv0.25_echo", "d7_conv0.5_echo", "d7_conv1.0_echo",
                 "d7_conv0.5_noecho")):
        P, _ = ap.perase_curve(tag)
        r[tag] = (lambda E, P=P: np.minimum(1.0, 2.0 * P(E)))
    return r


def gcorr(parts, fn):
    return cb.block_rates(parts, fn)["G_corr_per_h"]


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--n", type=int, default=300)
    ap_.add_argument("--data", default=None)
    ap_.add_argument("--seed", type=int, default=1)
    ap_.add_argument("--distance", action="store_true",
                     help="bootstrap the inner-distance readings (d5/d7/d11) instead")
    a = ap_.parse_args()
    if a.data:
        cb.DATA = a.data
    rng = np.random.default_rng(a.seed)
    sets = {}
    for L in ("X1", "XQ50"):
        mu = cb.load(L, "muon")
        g = cb.load(L, "gamma-lines")
        sets[L] = [(mu[0], mu[1], mu[2], cb.MUON_FLUX)] + (
            [(g[0], g[1], g[2], cb.GFLUX)] if g[1] else [])
    out = {}
    for name, fn in readings(a.distance).items():
        # precompute per-event contribution once: G_corr is a sum over events,
        # so a bootstrap resample is a weighted sum of per-event terms.
        contrib = {}
        for L, parts in sets.items():
            terms = []
            for ev, n, area, flux in parts:
                if n == 0 or len(ev) == 0:
                    continue
                P = fn(ev)
                al = ap.count_probs(P, kmax=2)
                terms.append((al[:, 2] * flux * area / n * 3600.0))
            contrib[L] = terms
        central = (sum(t.sum() for t in contrib["X1"])
                   / sum(t.sum() for t in contrib["XQ50"]))
        reps = []
        for _ in range(a.n):
            g = {}
            for L, terms in contrib.items():
                g[L] = sum(t[rng.integers(0, len(t), len(t))].sum() for t in terms)
            reps.append(g["X1"] / g["XQ50"] if g["XQ50"] > 0 else np.inf)
        lo, hi = np.percentile(reps, [2.5, 97.5])
        out[name] = {"central": central, "lo95": lo, "hi95": hi}
        print(f"{name:20s} reduction {central:6.1f}x  95% bootstrap {lo:6.1f}-{hi:6.1f}x")
    tag = "distance" if a.distance else ("spectrum" if a.data else "pooled")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", f"rad_block_bootstrap_{tag}.json"), "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
