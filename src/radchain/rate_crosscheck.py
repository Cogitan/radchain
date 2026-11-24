"""rate_crosscheck.py — close the last weak observable: the burst RATE.

THE GAP THIS FILLS
------------------
Rate has been a "weak pass" since 2026-09-16: a single detection threshold was
SOLVED FOR to match Google's 50.7 bursts/hr, which tests nothing. Two things
changed:

  1. We now have a measured environmental gamma spectrum, including the
     scattered continuum that device 2 forced on us.
  2. We have two devices in what is plausibly a similar terrestrial gamma
     environment.

That allows a rate test with NO free parameter left on the second device:

    Device 2 (sapphire, 55 keV threshold)  ->  implied incident gamma flux
    Apply that SAME flux to device 1 (Si, 180 keV threshold, their number)
    ->  PREDICTED gamma burst rate on Willow, to compare with 41.4/hr measured

And the muon side needs no flux inference at all, because the sea-level muon
flux is a textbook constant (~1 per cm^2 per minute):

    P(muon deposit >= 180 keV in Si) x flux x area  ->  muon burst rate

NORMALISATION
-------------
Primaries are generated on a sphere of radius R enclosing the die with
cosine-law inward directions, which is the standard construction for an
isotropic field. The fluence inside is then N/(pi R^2), so a measured burst rate
maps to a flux as

    rate = flux x (pi R^2) x p_detect_per_primary

with p_detect_per_primary the fraction of GENERATED primaries (not of those that
hit) depositing above threshold. Misses are therefore accounted for.

WHAT A DISAGREEMENT WOULD MEAN
------------------------------
The two fridges are in different buildings, so their
gamma environments genuinely differ; a factor of a few is expected and is not a
failure of the chain. An order of magnitude would be.
"""
from __future__ import annotations

import glob
import json
import math

from radchain import _data
import numpy as np

# ------------------------------------------------------------------ MEASURED
#: Willow, arXiv:2506.18228: 50.7 bursts/hr total, and the muon share from the
#: coincidence measurement on device 2 (18.4%) is the only split available.
D1_RATE_ALL_PER_HR = 3600.0 / 71.0
D1_MUON_SHARE = 0.184
D1_THRESHOLD_KEV = 180.0
D1_AREA_CM2 = 0.95
D1_DIE_MM = 10.0

#: Li et al. 2025: gamma-induced parity bursts, and the threshold derived in
#: device2_li2025.py from their own two-observable ratio.
D2_GAMMA_RATE_PER_HR = 3600.0 / 12.7 - 3600.0 / 67.0
D2_THRESHOLD_KEV = 55.0
D2_DIE_MM = 15.0
D2_THICK_MM = 0.43

#: Sea-level muon flux, textbook.
MUON_FLUX_PER_CM2_PER_MIN = 1.0


def _load(pattern: str):
    dep, n = [], 0
    for f in glob.glob(pattern):
        d = json.load(open(f))
        dep += d["deposits_keV"]
        n += d["n_primaries"]
    if not n:
        raise FileNotFoundError(pattern)
    return np.array(dep, dtype=float), n


def sphere_area_cm2(die_mm: float, thick_mm: float) -> float:
    """pi R^2 for the generation sphere used by g4_env_spectrum, in cm^2."""
    r_mm = 1.05 * math.sqrt(2 * (die_mm / 2) ** 2 + (thick_mm / 2) ** 2)
    return math.pi * (r_mm / 10.0) ** 2


def detect_fraction(pattern: str, threshold_keV: float) -> float:
    dep, n = _load(pattern)
    return float((dep >= threshold_keV).sum()) / n


def implied_gamma_flux_per_cm2_per_hr(rate_per_hr: float, pattern: str,
                                      threshold_keV: float, die_mm: float,
                                      thick_mm: float) -> float:
    p = detect_fraction(pattern, threshold_keV)
    return rate_per_hr / (sphere_area_cm2(die_mm, thick_mm) * p)


#: Thicknesses with a cached deposition spectrum for device 1, in mm.
CACHED_THICKNESS_MM = (0.3, 0.5, 0.725)


def channel_ratios(thickness_mm: float, spectrum: str = "gamma-lines") -> dict:
    """Predicted/measured for BOTH channels at one assumed die thickness.

    Willow's die thickness is not published, and both channels depend on it
    steeply, so a point estimate of either is a statement about an assumption.
    """
    p_mu = detect_fraction(str(_data.ENV / "muon_{:g}.json".format(thickness_mm)),
                           D1_THRESHOLD_KEV)
    muon = p_mu * MUON_FLUX_PER_CM2_PER_MIN * 60.0 * D1_AREA_CM2
    flux = implied_gamma_flux_per_cm2_per_hr(
        D2_GAMMA_RATE_PER_HR,
        str(_data.ENV2 / "{}_{:g}_*.json".format(spectrum, D2_THICK_MM)),
        D2_THRESHOLD_KEV, D2_DIE_MM, D2_THICK_MM)
    p_g = detect_fraction(
        str(_data.ENV / "{}_{:g}_*.json".format(spectrum, thickness_mm)),
        D1_THRESHOLD_KEV)
    gamma = flux * sphere_area_cm2(D1_DIE_MM, thickness_mm) * p_g
    meas_mu = D1_RATE_ALL_PER_HR * D1_MUON_SHARE
    meas_g = D1_RATE_ALL_PER_HR * (1 - D1_MUON_SHARE)
    return {"thickness_mm": thickness_mm, "spectrum": spectrum,
            "p_detect_muon": p_mu, "p_detect_gamma": p_g,
            "muon_per_hr": muon, "muon_ratio": muon / meas_mu,
            "gamma_per_hr": gamma, "gamma_ratio": gamma / meas_g,
            "implied_gamma_flux_per_cm2_per_s": flux / 3600.0}


def joint_thickness_constraint(spectrum: str = "gamma-lines") -> dict:
    """Can one die thickness satisfy BOTH channels? Measured 2026-09-30: no.

    THE DEFECT THIS EXPOSES. The gamma channel was recorded as "0.29x of
    measured, the weakest observable" and the muon channel as "9.5 vs 9.3,
    nothing fitted". Both were point estimates at an ASSUMED 0.3 mm, and the
    assumption was never tested against both channels at once:

        thickness   muon ratio   gamma ratio
        0.300 mm       1.03x        0.29x
        0.500 mm       3.86x        0.70x
        0.725 mm       6.05x        1.23x

    Muon can only be right on a THIN die; gamma only on a THICK one. There is
    no crossing point. So the honest reading is not "gamma is 3.4x low" but
    "the two channels disagree, and a single number hid it".

    WHICH CHANNEL WINS, and why it matters. The muon channel has NO free
    parameter: the sea-level flux is a textbook constant and the die area is
    known, so it pins the thickness at about 0.3 mm rather than agreeing there
    by luck. Given that, the gamma shortfall is a FLUX difference, and its
    size is a prediction the chain now makes explicitly: Google's site would
    need about 3.4x the gamma flux inferred at Li et al.'s. Terrestrial gamma
    varies by more than that between buildings, so this is consistent -- but
    it is an inference about someone else's building, and it must be presented as
    one and not as a validated rate.
    """
    rows = [channel_ratios(t, spectrum) for t in CACHED_THICKNESS_MM]
    mu = [r["muon_ratio"] for r in rows]
    ga = [r["gamma_ratio"] for r in rows]
    # a channel is satisfiable where its ratio brackets 1.0 across the cache
    mu_ok = min(mu) <= 1.0 <= max(mu)
    ga_ok = min(ga) <= 1.0 <= max(ga)
    # compatible only if some thickness puts BOTH within a factor of 1.5
    both = [r["thickness_mm"] for r in rows
            if 1 / 1.5 <= r["muon_ratio"] <= 1.5
            and 1 / 1.5 <= r["gamma_ratio"] <= 1.5]
    thin = rows[0]
    return {"spectrum": spectrum, "rows": rows,
            "muon_satisfiable": mu_ok, "gamma_satisfiable": ga_ok,
            "compatible_thicknesses_mm": both,
            "consistent": bool(both),
            "thickness_pinned_by": ("muon: sea-level flux is a textbook "
                                    "constant and the die area is known, so "
                                    "this channel has no free parameter"),
            "implied_gamma_flux_scaling": 1.0 / thin["gamma_ratio"],
            "note": ("no single thickness satisfies both channels; the muon "
                     "channel pins the thickness and the gamma shortfall is "
                     "therefore a site flux difference, not a model error")}


def _main() -> None:
    print("=" * 76)
    print("RATE CROSS-CHECK — device 2 fixes the gamma flux, device 1 is predicted")
    print("=" * 76)

    # ---- muons on device 1, no free parameter at all
    for t in ("0.3", "0.5"):
        p = detect_fraction(str(_data.ENV / f"muon_{t}.json"), D1_THRESHOLD_KEV)
        pred = p * MUON_FLUX_PER_CM2_PER_MIN * 60.0 * D1_AREA_CM2
        meas = D1_RATE_ALL_PER_HR * D1_MUON_SHARE
        print(f"  MUONS  Si {t} mm: P(dep >= 180 keV) = {p:.1%} -> "
              f"{pred:5.1f} bursts/hr  vs measured {meas:.1f}/hr "
              f"({pred/meas:.2f}x)")

    # ---- gammas: flux from device 2, rate predicted on device 1
    print()
    for src in ("gamma-scatter", "gamma-lines"):
        flux = implied_gamma_flux_per_cm2_per_hr(
            D2_GAMMA_RATE_PER_HR, str(_data.ENV2 / f"{src}_*.json"),
            D2_THRESHOLD_KEV, D2_DIE_MM, D2_THICK_MM)
        print(f"  [{src}] device 2 implies an incident gamma flux of "
              f"{flux/3600.0:.2f} per cm^2 per s")
        for t in ("0.3", "0.5"):
            p1 = detect_fraction(str(_data.ENV / f"{src}_{t}_*.json"),
                                 D1_THRESHOLD_KEV)
            pred = flux * sphere_area_cm2(D1_DIE_MM, float(t)) * p1
            meas = D1_RATE_ALL_PER_HR * (1 - D1_MUON_SHARE)
            print(f"     -> device 1, Si {t} mm: predicted {pred:6.1f} gamma "
                  f"bursts/hr vs measured {meas:.1f}/hr  ({pred/meas:.2f}x)")
        print()


if __name__ == "__main__":
    _main()
