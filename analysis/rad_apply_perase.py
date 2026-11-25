"""Apply the physics erasure curve P_erase(E) to the correlated-erasure studies.

P_erase(E) comes from analysis/rad_pfail_physics.py (results/rad_pfail_<tag>.json):
binned, UNCLIPPED mean of (f_burst - f_base)/(1 - f_base) over strike positions;
negative or unresolved bins -> 0; made monotone; log-log interpolation; above
the last bin, extrapolated on the last two bins' slope, capped at 1.

LOSS = 2 x FLIP (added 2026-10-16). The Stim runs measure the probability the
Z-memory logical FLIPS. A fully scrambled logical flips only half the time, and
a Z-basis memory sees X-type logical errors but not Z-type, which a depolarizing
burst causes at a similar rate. So the probability the chip's logical state is
LOST is ~2 x P_flip in both limits: P_loss = min(1, 2 P_flip). The no-echo run,
which saturates at P_flip ~ 0.5, is what exposed it.

Two readings of "a chip is erased" (they differ by orders of magnitude, and the
E* proxy sat between them):
  CORRUPTED  the inner d=7 code actually failed: each chip independently with
             P_erase(E_chip). Conditional independence given the deposits holds
             because the two chips' noise realisations are independent.
  FLAGGED    a burst is detected and the outer code writes the chip off:
             E_chip >= 180 keV (Willow's quoted detection threshold).

Outputs per layout: lam (per chip, /h), p2 = P(>= 2 erased | >= 1), p3, and
Xu's [[4,1,2]] / [[7,1,3]] suppression claimed vs achieved (tau 1 ms).

    python analysis/rad_apply_perase.py [--tag d7_conv0.5_echo]
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

import rad_chip_erasure as ce

HERE = os.path.dirname(os.path.abspath(__file__))
E_FLAG_KEV = 180.0
TAU = 1e-3
LAYOUTS = ("A1", "A3", "B3", "V1", "S1")


def perase_curve(tag: str):
    r = json.load(open(os.path.join(HERE, "results", f"rad_pfail_{tag}.json")))
    ev = r["events"]
    E = np.array([e["e_keV"] for e in ev])
    fb = np.array([e["f_burst"] if e["f_burst"] is not None else 0.0 for e in ev])
    f0 = np.array([e["f_base"] if e["f_base"] is not None else 0.0 for e in ev])
    d = np.where(f0 < 1, (fb - f0) / np.clip(1 - f0, 1e-9, None), 0.0)
    edges = np.geomspace(30, 5000, 9)
    xc, yc = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (E >= lo) & (E < hi)
        if m.any():
            xc.append(math.sqrt(lo * hi))
            yc.append(max(0.0, float(d[m].mean())))
    xc, yc = np.array(xc), np.maximum.accumulate(np.array(yc))
    pos = yc > 0
    lx, ly = np.log(xc[pos]), np.log(yc[pos])
    slope = (ly[-1] - ly[-2]) / (lx[-1] - lx[-2]) if len(lx) >= 2 else 1.0
    x_first = xc[pos][0]

    def P(e_keV):
        e = np.asarray(e_keV, float)
        out = np.zeros_like(e)
        le = np.log(np.clip(e, 1e-9, None))
        inside = (e >= x_first) & (e <= xc[-1])
        out[inside] = np.exp(np.interp(le[inside], lx, ly))
        above = e > xc[-1]
        out[above] = np.exp(ly[-1] + slope * (le[above] - lx[-1]))
        # below the first resolved bin: linear ramp to 0 at half that energy
        ramp = (e >= x_first / 2) & (e < x_first)
        out[ramp] = yc[pos][0] * (e[ramp] - x_first / 2) / (x_first / 2)
        return np.clip(out, 0.0, 1.0)

    return P, {"centres_keV": xc.tolist(), "p": yc.tolist(), "slope_hi": slope}


def count_probs(P: np.ndarray, kmax: int = 3) -> np.ndarray:
    """Poisson-binomial P(at least k erased) for k = 1..kmax, per event row."""
    n = P.shape[0]
    dist = np.zeros((n, kmax + 1))
    dist[:, 0] = 1.0
    for j in range(P.shape[1]):
        p = P[:, j][:, None]
        shifted = np.zeros_like(dist)
        shifted[:, 1:] = dist[:, :-1]
        shifted[:, kmax] += dist[:, kmax]          # absorb >= kmax
        dist = dist * (1 - p) + shifted * p
        dist[:, kmax] = dist[:, kmax]              # (kept as ">= kmax")
    at_least = np.zeros((n, kmax + 1))
    for k in range(1, kmax + 1):
        at_least[:, k] = 1.0 - dist[:, :k].sum(axis=1)
    return at_least


def rates(data, gamma_src, gflux, chip_prob):
    """lam per chip (/s), R1, R2, R3 (/s) with chip_prob(E array) -> probs."""
    out = {"lam": 0.0, "R1": 0.0, "R2": 0.0, "R3": 0.0}
    for src, flux in ((gamma_src, gflux), ("muon", ce.MUON_FLUX)):
        if src not in data:
            continue
        ev, n, area, nc = data[src]
        if len(ev) == 0:
            continue
        P = chip_prob(ev[:, :nc])
        w = flux * area / n
        al = count_probs(P)
        out["lam"] += P.sum() * w / nc
        out["R1"] += al[:, 1].sum() * w
        out["R2"] += al[:, 2].sum() * w
        out["R3"] += al[:, 3].sum() * w
    out["p2"] = out["R2"] / out["R1"] if out["R1"] else float("nan")
    out["p3"] = out["R3"] / out["R1"] if out["R1"] else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="d7_conv0.5_echo")
    ap.add_argument("--gflux", type=float, default=1.7)
    a = ap.parse_args()
    P, curve = perase_curve(a.tag)
    print(f"P_erase({a.tag}): " + ", ".join(
        f"{c / 1000:.2f} MeV -> {p:.4f}" for c, p in zip(curve["centres_keV"],
                                                         curve["p"])))
    readings = {
        "CORRUPTED": lambda E: np.minimum(1.0, 2.0 * P(E)),
        "FLAGGED": lambda E: (E >= E_FLAG_KEV).astype(float),
    }
    rows = []
    for L in LAYOUTS:
        data = ce.load(L)
        if not data:
            continue
        for gsrc in ("gamma-lines", "gamma-scatter"):
            for name, fn in readings.items():
                r = rates(data, gsrc, a.gflux, fn)
                x2 = ce.xu(r["lam"], r["p2"], r["p3"], 4, 2, TAU)
                x3 = ce.xu(r["lam"], r["p2"], r["p3"], 7, 3, TAU)
                rows.append({"layout": L, "gamma": gsrc, "reading": name,
                             **r, "xu412": x2, "xu713": x3})
                if gsrc == "gamma-lines":
                    print(f"{L} {name:9s}: lam {r['lam'] * 3600:8.3f}/h/chip  "
                          f"p2 {100 * r['p2']:.3f}%  p3 {100 * r['p3']:.4f}%  "
                          f"[[4,1,2]] claimed {x2['suppression_claimed']:.3g} "
                          f"achieved {x2['suppression_achieved']:.3g} | "
                          f"[[7,1,3]] claimed {x3['suppression_claimed']:.3g} "
                          f"achieved {x3['suppression_achieved']:.3g}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump({"tag": a.tag, "curve": curve, "rows": rows},
              open(os.path.join(HERE, "results", f"rad_apply_perase_{a.tag}.json"),
                   "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
