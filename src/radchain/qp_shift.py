"""qp_shift.py -- kHz of qubit shift per keV absorbed in its cell, per device.

WHY
---
Re-fitting `k` for every chip so that the median burst peaks at exactly
2 MHz, Willow's measured value (arXiv:2506.18228, Fig. 8), forces Willow's
peak shift onto every chip, whatever its frequency, gap, film coverage or
pitch. This module scales Willow's `k` by the physics that differs.

THE PHYSICS, AND WHAT IT CANNOT PIN
-----------------------------------
    dfq / fq = -a * x_qp                                    Google Eq. (1)

    a = 1/4 + (1/4pi) [ sqrt(2 Delta / (dDelta - h fq))
                      + sqrt(2 Delta / (dDelta + h fq)) ]   Google Eq. (17),
                                                            gap-engineered,
                                                            dDelta > h fq
    a = 1/2 + (1/2pi) sqrt(2 Delta / h fq)                  Catelani et al.,
                                                            PRB 84, 064517
                                                            (2011) Eq. (73) at
                                                            phi_0 = 0, omega_p
                                                            = omega_10: a
                                                            single symmetric
                                                            junction

    x_qp = n_qp / n_cp,  n_qp = eta E / Delta per film volume,  n_cp = 2 N0 Delta

A cell's absorbed energy is spread over the film in that cell, coverage x
A_cell x t_film, so

    k  proportional to  a fq / (Delta^2 * coverage * A_cell)  x  eta / t_film

The last factor is NOT published for Willow (film thickness, pair-breaking
efficiency, and how much of the film is the low-gap side), so an absolute,
unfitted k cannot be tested on the one device graded. It is borrowed as ONE
scalar: Willow's fitted k (K_WILLOW_KHZ_PER_KEV). Everything else is a physics
ratio to Willow:

    k_dev = K_WILLOW * [a_dev fq_dev / (a_W fq_W)] * (Delta_W / Delta_dev)^2
                     * (cov_W A_W) / (cov_dev A_dev)

On Willow every ratio is 1, so its graded answer is unchanged by construction.
That is a regression property, not a validation; whether the ratios transfer
to another chip is a separate test.

WHAT IS REFUSED
---------------
0 < dDelta < h fq: QPs populate both sides of the junction in proportions
this module does not model. And within 1 GHz of dDelta = h fq, Eq. (17)
diverges (a resonance it does not resolve). The 1 GHz margin is a stated
choice, not a derived one. Both refuse in words; neither is clamped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- WILLOW
#: Google's own parameters for Eq. (17), arXiv:2506.18228: Delta/h = 52 GHz,
#: dDelta/h = 12 GHz, fq ~= 6 GHz, giving their a ~= 0.77.
WILLOW_FQ_GHZ = 6.0
WILLOW_GAP_GHZ = 52.0
WILLOW_GAP_DIFF_GHZ = 12.0
#: Willow's 3 sigma_f burst threshold (their Appendix C, ~200 kHz).
WILLOW_THRESHOLD_KHZ = 200.0
WILLOW_CELL_MM = 1.15

#: k fitted so Willow's median peak shift is 2 MHz, with the shipped
#: `g4cmp-kernel` phonon backend (stock geometry, 0.381 mm Si): 60-site grid at
#: 1.15 mm on a 10.2 mm die, gamma-lines spectrum at 0.3 mm, mean of 8 seeds x
#: 3000 strikes: 44.89, seed-to-seed sd 0.64, standard error 0.23.
#:
#: TRANSPORT-SPECIFIC. `k` absorbs the transport's overall normalisation, so a
#: `k` fitted with one backend or geometry must not be used with another. Refit
#: it with `qubit_footprint.fit_k` on your backend's hits
#: on the no-trap reference, never on a trapped design. The qubit and gap ratios
#: in `k_for_device` are transport-independent; the film-area ratio assumes the
#: absorbed energy scales with film area, which the G4CMP kernels (one film
#: pattern per family) do not test.
K_WILLOW_KHZ_PER_KEV = 44.89

#: Eq. (17) diverges at dDelta = h fq. Inside this margin it is refused.
RESONANCE_MARGIN_GHZ = 1.0


@dataclass(frozen=True)
class ShiftCoefficient:
    a: float | None
    regime: str
    refused: str | None = None


def a_coefficient(fq_ghz: float, gap_ghz: float,
                  gap_diff_ghz: float) -> ShiftCoefficient:
    """dfq/fq = -a x_qp. Frequencies in GHz (energies over h)."""
    if fq_ghz <= 0 or gap_ghz <= 0 or gap_diff_ghz < 0:
        return ShiftCoefficient(None, "invalid", (
            "qubit frequency and gap must be positive and the gap difference "
            "non-negative"))
    if gap_diff_ghz == 0:
        a = 0.5 + np.sqrt(2.0 * gap_ghz / fq_ghz) / (2.0 * np.pi)
        return ShiftCoefficient(float(a), "symmetric junction, Catelani "
                                "2011 Eq. 73")
    if gap_diff_ghz - fq_ghz < RESONANCE_MARGIN_GHZ:
        if gap_diff_ghz > fq_ghz - RESONANCE_MARGIN_GHZ:
            why = ("the gap difference ({:g} GHz) is within {:g} GHz of the "
                   "qubit frequency ({:g} GHz), where Google's Eq. 17 "
                   "diverges; this resonance is not modelled".format(
                       gap_diff_ghz, RESONANCE_MARGIN_GHZ, fq_ghz))
        else:
            why = ("a gap difference ({:g} GHz) below the qubit frequency "
                   "({:g} GHz) puts quasiparticles on both sides of the "
                   "junction in proportions this model does not compute. "
                   "Modelled: 0 (no gap engineering) or more than {:g} GHz "
                   "above the qubit frequency".format(
                       gap_diff_ghz, fq_ghz, RESONANCE_MARGIN_GHZ))
        return ShiftCoefficient(None, "not modelled", why)
    a = 0.25 + (np.sqrt(2.0 * gap_ghz / (gap_diff_ghz - fq_ghz)) +
                np.sqrt(2.0 * gap_ghz / (gap_diff_ghz + fq_ghz))) / (4.0 * np.pi)
    return ShiftCoefficient(float(a), "gap-engineered, Google Eq. 17")


def k_for_device(*, fq_ghz: float = WILLOW_FQ_GHZ,
                 gap_ghz: float = WILLOW_GAP_GHZ,
                 gap_diff_ghz: float = WILLOW_GAP_DIFF_GHZ,
                 f_film: float | None = None,
                 ref_f_film: float | None = None,
                 cell_mm: float = WILLOW_CELL_MM) -> dict:
    """k (kHz per keV absorbed in a cell) and every factor that made it.

    The film-area factor scales k by (ref_f_film * Willow cell area) /
    (f_film * cell area). Willow's film coverage is not published, so the
    coverage ratio is applied only when you give both ``f_film`` (this device)
    and ``ref_f_film`` (Willow's, as you estimate it); otherwise only the cell
    areas enter.

    Returns {"refused": reason} when `a` cannot be computed.
    """
    c = a_coefficient(fq_ghz, gap_ghz, gap_diff_ghz)
    if c.a is None:
        return {"refused": c.refused}
    w = a_coefficient(WILLOW_FQ_GHZ, WILLOW_GAP_GHZ, WILLOW_GAP_DIFF_GHZ)
    factors = {
        "qubit": (c.a * fq_ghz) / (w.a * WILLOW_FQ_GHZ),
        "gap": (WILLOW_GAP_GHZ / gap_ghz) ** 2,
        "film_area": (WILLOW_CELL_MM ** 2 / cell_mm ** 2) *
                     ((ref_f_film / f_film)
                      if f_film is not None and ref_f_film is not None
                      else 1.0),
    }
    k = K_WILLOW_KHZ_PER_KEV * float(np.prod(list(factors.values())))
    return {"k_khz_per_keV": k, "a": c.a, "a_regime": c.regime,
            "a_willow": w.a, "factors_vs_willow": factors,
            "k_willow_khz_per_keV": K_WILLOW_KHZ_PER_KEV}
