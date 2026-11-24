"""device2_li2025.py — the second device, and the first test we could fail.

WHY A SECOND DEVICE
-------------------
A model calibrated on one device (Google's gap-engineered Willow,
arXiv:2506.18228) and never tested on another is a curve fit, and several
parameter combinations fit Willow equally well.

DEVICE 2: Li et al., Nat. Commun. 2025 (arXiv:2402.04245)
  63 qubits, SAPPHIRE substrate 15 x 15 x 0.43 mm (not silicon)
  Ta capacitor pads (gap 690 ueV) with small Al films (gap 180 ueV, 87 um^2
  per qubit at 36 nm = 3.1 um^3) -- the Al acts as an ACCIDENTAL QP trap
  No deliberate gap engineering, muon and gamma contributions separated by
  coincidence with two muon detectors and by a 1 cm Pb shield

Everything about this device differs from Willow: substrate material, thickness,
area, pad metal, trap regime. The only thing shared is our chain.

WHAT IS MEASURED, AND WHAT WE PREDICT
-------------------------------------
They count the SAME bursts two ways, with different sensitivity:

    MQSCPJ (charge-parity jumps, sensitive)   all 12.7 +/- 0.4 s, muon 67 +/- 3 s
    MQSBF  (simultaneous bit flips, blunt)    all 386 +/- 19 s, muon 1500 +/- 173 s

and they measure the incident muon rate independently: 0.40 +/- 0.02 per minute
per cm^2 by coincidence, over a 2.25 cm^2 chip.

So each observable is a CUT on the deposited-energy spectrum, and the two cuts
sit at different energies. The test:

  1. The blunt cut is fixed by muons alone: 2.4 of 54 muons per hour produce a
     bit-flip burst, so its threshold is the 95.6th percentile of the MEASURED
     muon deposition spectrum in sapphire.
  2. The sensitive cut is then fixed by the GAMMA ratio, which is a different
     particle and a different spectrum.
  3. **The prediction:** at that sensitive threshold, what fraction of muons
     should be detected? Measured, it is 99.4% (53.7 of 54.0 per hour). Nothing
     in the earlier steps knew about that number.

A chain carrying the wrong energy dependence fails this: the gamma spectrum and
the muon spectrum in sapphire have different shapes, so a threshold inferred
from one does not automatically reproduce the other.

PRE-REGISTERED PASS BAND: predicted muon parity efficiency within 0.90 to 1.00.
Below 0.90 the chain is wrong; a value above 1.0 is impossible and would mean
the inferred threshold is below every muon deposit, which is a soft fail (the
test could not discriminate).
"""
from __future__ import annotations

import glob
import json

import numpy as np

from radchain import _data

#: Device 2's cache. Was the cwd-relative "bench/rad_env2/", in shipped code.
DATA = str(_data.ENV2) + "/"

# ------------------------------------------------------------------ MEASURED
CHIP_MM = 15.0
THICK_MM = 0.43
AREA_CM2 = (CHIP_MM / 10.0) ** 2
T_CPJ_ALL_S, T_CPJ_ALL_ERR = 12.7, 0.4
T_CPJ_MUON_S, T_CPJ_MUON_ERR = 67.0, 3.0
T_BF_ALL_S, T_BF_ALL_ERR = 386.0, 19.0
T_BF_MUON_S, T_BF_MUON_ERR = 1500.0, 173.0
MUON_RATE_PER_MIN_CM2, MUON_RATE_ERR = 0.40, 0.02
PASS_BAND = (0.90, 1.00)

#: Low-gap Al per qubit on this device, from their Table S1: 87 um^2 x 36 nm.
AL_VOLUME_UM3 = 87.0 * 0.036
TAU_TRAP_CPJ_S = 36e-6          #: measured tail, parity
TAU_TRAP_BF_S = 21e-6           #: measured tail, bit flip


def deposits(source: str) -> np.ndarray:
    v: list = []
    for f in glob.glob(f"{DATA}{source}_*.json"):
        v += json.load(open(f))["deposits_keV"]
    return np.sort(np.array(v, dtype=float))


def per_hour(t_s: float) -> float:
    return 3600.0 / t_s


def incident_muons_per_hour() -> float:
    return MUON_RATE_PER_MIN_CM2 * AREA_CM2 * 60.0


def threshold_for_efficiency(spectrum: np.ndarray, eff: float,
                             n_primaries: int) -> float:
    """Energy cut whose upper tail holds `eff` of INCIDENT particles.

    For muons every primary crosses the die, so incident = simulated.
    """
    k = eff * n_primaries
    if k >= len(spectrum):
        return 0.0
    return float(spectrum[len(spectrum) - int(round(k))])


def fraction_above(spectrum: np.ndarray, e_keV: float) -> float:
    return float((spectrum >= e_keV).sum())


def solve_sensitive_threshold(gamma: np.ndarray, ratio: float,
                              e_blunt: float) -> float:
    """Find E such that P_gamma(>= e_blunt) / P_gamma(>= E) equals `ratio`."""
    n_blunt = fraction_above(gamma, e_blunt)
    target = int(round(n_blunt / ratio))          # required count above E
    if target > len(gamma):
        return 0.0                                # ratio unreachable
    return float(gamma[len(gamma) - target])


def _main() -> None:
    mu = deposits("muon")
    n_mu = json.load(open(glob.glob(f"{DATA}muon_*.json")[0]))["n_primaries"]
    print("=" * 76)
    print("DEVICE 2 — Li et al. 2025, 63 qubits on SAPPHIRE 15x15x0.43 mm")
    print("=" * 76)
    print(f"  muon deposits in sapphire: median {np.median(mu):.0f} keV, "
          f"1st pct {np.percentile(mu, 1):.0f}, 99th {np.percentile(mu, 99):.0f}")

    inc = incident_muons_per_hour()
    eff_cpj = per_hour(T_CPJ_MUON_S) / inc
    eff_bf = per_hour(T_BF_MUON_S) / inc
    print(f"  incident muons {inc:.1f}/hr; measured muon bursts "
          f"parity {per_hour(T_CPJ_MUON_S):.1f}/hr ({eff_cpj:.1%}), "
          f"bit-flip {per_hour(T_BF_MUON_S):.2f}/hr ({eff_bf:.1%})\n")

    e_blunt = threshold_for_efficiency(mu, eff_bf, n_mu)
    print(f"  STEP 1  blunt (bit-flip) threshold from muons: {e_blunt:,.0f} keV")

    g_cpj = per_hour(T_CPJ_ALL_S) - per_hour(T_CPJ_MUON_S)
    g_bf = per_hour(T_BF_ALL_S) - per_hour(T_BF_MUON_S)
    ratio = g_bf / g_cpj
    print(f"  measured gamma bursts: parity {g_cpj:.1f}/hr, bit-flip {g_bf:.2f}/hr"
          f"  -> ratio {ratio:.2%}")

    for src in ("gamma-lines", "gamma-flat", "gamma-scatter"):
        gam = deposits(src)
        e_sens = solve_sensitive_threshold(gam, ratio, e_blunt)
        print(f"\n  STEP 2  [{src}] sensitive threshold from the gamma ratio: "
              f"{e_sens:,.0f} keV")
        pred = fraction_above(mu, e_sens) / n_mu
        lo, hi = PASS_BAND
        if e_sens <= 0:
            best = fraction_above(gam, e_blunt) / len(gam)
            verdict = (f"FAIL: the ratio is unreachable. This spectrum's floor "
                       f"is {best:.2%}, above the measured {ratio:.2%}")
        else:
            verdict = "PASS" if lo <= pred <= hi else "FAIL"
        print(f"  STEP 3  PREDICTED muon parity efficiency {pred:.1%} "
              f"vs measured {eff_cpj:.1%}   -> {verdict}")

    print(f"\n  Trap regime: Al volume per qubit {AL_VOLUME_UM3:.1f} um^3.")
    print(f"  Measured exponential tails {TAU_TRAP_CPJ_S*1e6:.0f} us (parity), "
          f"{TAU_TRAP_BF_S*1e6:.0f} us (bit flip): TRAPPED regime, the opposite "
          f"of Willow's\n  pure-recombination power law. Same chain, both regimes.")


if __name__ == "__main__":
    _main()
