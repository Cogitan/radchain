"""Analytic upper bound on two-chip erasures from air-shower bundles.

Single-primary Geant4 cannot make two DIFFERENT particles from one air shower
hit two chips. At cm-to-m separations the shower density is effectively uniform
over both chips, so this term does not fall with separation and is the
candidate floor of the separation rule.

Sea-level extensive-air-shower density spectrum (ALL charged particles):
    N(> Delta) = B Delta^-gamma  per hour, Delta in particles per m^2
Sources, each checked against the publisher's abstract (2026-10-19):
    Broadbent, Kellermann, Hakeem, Proc. Phys. Soc. A 63, 864 (1950),
      doi 10.1088/0370-1298/63/8/312: "N(x) = B x^-gamma per hr", "in the
      density range 5-500 particles/m2", "gamma = 1.425 +/- 0.022, B = 620".
    Norman, Proc. Phys. Soc. A 69, 804 (1956), doi 10.1088/0370-1298/69/11/303:
      "nu(rho) = 540 rho^-1.39+/-0.04 h^-1, for rho < 500 particles m^-2";
      "For rho > 500 the exponent increases quickly, reaching 2.2 at rho = 1000".
    (The earlier Cocconi 1946 values B = 700, gamma = 1.47 could not be traced
    to the paper and were dropped; they lay inside the range below.)
The bound is dominated by densities near 1/a = 1e4 /m^2, ABOVE every measured
range, so the high-density behaviour matters. Three spectra are carried:
two single power laws (Broadbent, Norman; the loose envelope, contradicted
above 500 /m^2 by Norman's own data) and Norman with his measured steepening,
the local exponent rising linearly in ln(rho) from 1.39 at 500 to 2.2 at
1000 /m^2 and HELD at 2.2 above (conservative: no further steepening).
Rate of showers putting >= 1 particle in EACH of two chips of area a:
    R2_hit = int dN/dDelta (1 - e^{-Delta a})^2 dDelta
    single power law: = gamma B a^gamma I(gamma),
                      I(gamma) = int_0^inf u^{-gamma-1} (1-e^{-u})^2 du
    steepened: computed numerically (r2_hit_numeric).
A hit erases a chip with probability q (taken from our own muon data at the
same threshold: per-chip muon erasure rate / per-chip muon hit rate), so
    R2_erase <= R2_hit q^2.

CONSERVATIVE in three stated ways (all push the bound UP):
  - all shower particles (mostly electrons) counted, with no absorption by the
    building, the fridge or the sample box;
  - above 1000 /m^2 the exponent is held at 2.2 (single-law envelopes: held at
    1.39-1.425), although measured spectra keep steepening;
  - q for electrons is taken equal to q for muons (shower electrons that survive
    the fridge are MIP-like; low-energy ones would stop before the chip).

    python bench/rad_bundle_bound.py
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from scipy.integrate import quad

import rad_chip_separation as cs

HERE = os.path.dirname(os.path.abspath(__file__))
A_M2 = 1e-4                    # 1 cm2 chip
MUON_FLUX_CM2_S = 1.0 / 60.0
#: name -> (B, gamma below the break, steepening (d_break, d_top, gamma_top) or None)
SPECTRA = {
    "Broadbent 1950 single law": (620.0, 1.425, None),
    "Norman 1956 single law": (540.0, 1.39, None),
    "Norman 1956 steepened": (540.0, 1.39, (500.0, 1000.0, 2.2)),
}


def local_gamma(D, g1, steep):
    if steep is None or D <= steep[0]:
        return g1
    d0, d1, g2 = steep
    if D >= d1:
        return g2
    t = (math.log(D) - math.log(d0)) / (math.log(d1) - math.log(d0))
    return g1 + t * (g2 - g1)


def r2_hit_numeric(B, g1, steep, a=A_M2, n=200_000):
    """R2_hit for a spectrum given by its local exponent, normalised as
    N(>1 /m^2) = B (identical to B D^-g1 below the break)."""
    lnD = np.linspace(math.log(1e-3), math.log(1e9), n)
    D = np.exp(lnD)
    gs = np.array([local_gamma(d, g1, steep) for d in D])
    dl = lnD[1] - lnD[0]
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (gs[1:] + gs[:-1]) * dl)])
    cum -= cum[np.searchsorted(lnD, 0.0)]
    f = B * np.exp(-cum) * gs * (1 - np.exp(-D * a)) ** 2   # (-dN/dlnD) x hit^2
    return float(np.sum(0.5 * (f[1:] + f[:-1])) * dl)


def spectrum_r2(spec, a=A_M2):
    B, g1, steep = spec
    return r2_hit_per_h(B, g1, a) if steep is None else r2_hit_numeric(B, g1, steep, a)


def I(gamma):
    f = lambda u: u ** (-gamma - 1) * (1 - math.exp(-u)) ** 2
    a, _ = quad(f, 0, 1, limit=200)
    b, _ = quad(f, 1, np.inf, limit=200)
    return a + b


def r2_hit_per_h(B, gamma, a=A_M2):
    return gamma * B * a ** gamma * I(gamma)


def muon_q(thr_keV, layout="Q50"):
    """Per-hit erasure probability for a MIP-like track, from our Geant4 muons:
    per-chip erasure count / expected per-chip hits (flux x area, horizontal
    plane; edge hits ignored, which makes q slightly HIGH, i.e. conservative)."""
    import glob
    ev, n, area = [], 0, None
    for f in sorted(glob.glob(os.path.join(cs.data_dir(False),
                                           f"{layout}_muon_*.json"))):
        r = json.load(open(f))
        ev += [e[:2] for e in r["events"]]
        n += r["n_primaries"]
        area = r["gen_area_cm2"]
    if area is None:
        raise SystemExit("no deposit data at {}: copy the per-event Geant4 deposits from the data record (doi:10.5281/zenodo.24412856) to analysis/rad_data/; see README".format(cs.data_dir(False)))
    ev = np.asarray(ev, float)
    erasures_per_chip = (ev >= thr_keV).sum(axis=0).mean()
    hits_per_chip = n * (1.0 / area)          # 1 cm2 chip, uniform disk
    return erasures_per_chip / hits_per_chip, ev, n, area


def main():
    out = {}
    for thr in (180.0, 300.0, 500.0, 1000.0):
        q, ev, n, area = muon_q(thr)
        # Geant4 single-primary two-chip rate at Q50 for comparison (/h, muons)
        both = int(((ev[:, 0] >= thr) & (ev[:, 1] >= thr)).sum())
        w = MUON_FLUX_CM2_S * area / n * 3600.0
        g4 = both * w
        line = f"E* {thr / 1000:.2f} MeV: q = {q:.3f}; Geant4 Q50 two-chip {g4:.2e}/h ({both} ev)"
        for name, spec in SPECTRA.items():
            r2 = spectrum_r2(spec) * q * q
            line += f"\n    {name}: bundle bound {r2:.2e}/h ({r2 / g4 if g4 else float('inf'):.2g} of G4)"
        print(line)
        out[str(thr)] = {"q": q, "g4_Q50_per_h": g4, "g4_events": both,
                         "bounds": {k: spectrum_r2(v) * q * q
                                    for k, v in SPECTRA.items()}}
    for name, spec in SPECTRA.items():
        print(f"{name}: R2_hit = {spectrum_r2(spec):.2e}/h "
              f"for two 1 cm2 chips (any separation << shower scale)")
    json.dump(out, open(os.path.join(HERE, "results", "rad_bundle_bound.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
