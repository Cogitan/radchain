"""qp_recovery.py — L3 quasiparticle recovery. The deciding observable.

SUPERSEDED FOR NO-TRAP DEVICES, 2026-09-17. Use `recovery_google2025.py` when
comparing to arXiv:2506.18228 or any device without deployed traps: there the
decay is pure recombination at a MEASURED r = 1/(88 ns), not the trapping
exponential with tau_0 = 438 ns assumed here. This module remains correct for
the TRAPPED regime (a device with traps, where the tail is exponential) and is
kept for that case and for the crossover analysis.

WHY THIS EXISTS
---------------
Rate and footprint alone are inconclusive on Willow: both observables
(rate, footprint) are consistent with a phonon-mediated mechanism and both are
weakly discriminating: a rate can be matched by tuning an energy threshold, a
footprint by tuning trap coverage. The third observable carries the signal.

    Willow          ~400 us recovery   (369 +/- 6 us, exponential fit)
    Google 2021     tens of ms         ATTRIBUTED to high-energy impacts

A 62x separation on one axis.

THE PHYSICS, AND THE DISCRIMINATOR HIDDEN IN IT
-----------------------------------------------
Excess quasiparticles decay by two channels with different ORDERS, and that is
what makes the decay SHAPE informative rather than just its timescale:

    dx/dt = -x^2 / tau_0   -   x / tau_trap
            ^^^^^^^^^^^^       ^^^^^^^^^^^^
            recombination      trapping
            two QPs meet       one QP falls into a low-gap region
            QUADRATIC          LINEAR
            -> POWER LAW       -> EXPONENTIAL

**CORRECTED 2026-09-16, and the correction is the result.** I first wrote that
an exponential decay is itself evidence of trapping dominance. That is wrong for
the TAIL. Recombination at high density is fast (tau_0/x ~ 4 us at x=0.1), so it
burns the excess down within microseconds and the tail is trapping-dominated
REGARDLESS of how large the burst was. Measured in this module:

    x0      early tau (0-60us)   early r2   late tau   late r2
    1e-4        342.5 us          1.0000      369 us    1.0000
    1e-3        213.2 us          0.9998      368 us    1.0000
    1e-2         62.7 us          0.9892      366 us    1.0000
    1e-1         26.5 us          0.9131      366 us    1.0000

Two consequences, both stronger than the original claim:

1. **The late tail is a clean, burst-size-independent measurement of tau_trap.**
   Willow's 369 +/- 6 us IS its trapping time. That is a materials-stack
   property.
2. **The initial QP density lives only in the EARLY decay**, where recombination
   leaves a faster, less exponential signature. So burst SIZE is readable from
   recovery data, but only from roughly the first tens of microseconds.

THE CROSSOVER DENSITY
---------------------
The two channels are equal when x = tau_0 / tau_trap. Above that density
recombination wins and the decay is a power law; below it trapping wins and the
decay is exponential. So an observed exponential is a CEILING on the
quasiparticle density during the observation window, which can then be checked
against the density a given deposition would create.

SCOPE, stated rather than buried
--------------------------------
* Spatially uniform. Real bursts are not, and diffusion is not modelled: the
  trapping term stands in for "QP leaves the active region", whether by falling
  into a gap trap or diffusing away.
* `tau_0` is the material's characteristic electron-phonon time. 438 ns for
  aluminium is the standard literature value and is used as-is.
* Phonon re-absorption (a recombination phonon breaking another pair) is not
  modelled. It lengthens the effective recombination time, so ignoring it is
  conservative in the direction of favouring recombination.

This is a simplified rate model: its job is to test whether the answer is
SENSITIVE to the mechanism, not to produce a precise number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Characteristic electron-phonon time. Aluminium, standard LITERATURE value.
#: WARNING 2026-09-16: 5x off a real device. Google MEASURED r = 88 +/- 12 ns on
#: gap-engineered Willow (arXiv:2506.18228). Prefer TAU_RECOMB_WILLOW_S for any
#: comparison to that device. Kept only so the error stays visible.
TAU_0_AL_S = 438e-9

#: MEASURED recombination time on gap-engineered Willow, arXiv:2506.18228.
TAU_RECOMB_WILLOW_S = 88e-9

#: Cooper pair density in aluminium, per cubic micron. Standard figure.
N_CP_AL_PER_UM3 = 4.0e6

#: Pair-breaking threshold in aluminium, 2*Delta, in eV.
TWO_DELTA_AL_EV = 3.6e-4

#: Fraction of deposited energy that ends up breaking pairs.
PAIR_BREAKING_FRACTION = 0.57


@dataclass(frozen=True)
class Decay:
    t_s: np.ndarray
    x: np.ndarray
    tau_0_s: float
    tau_trap_s: float
    x0: float

    @property
    def crossover_density(self) -> float:
        """Density where recombination and trapping rates are equal."""
        return self.tau_0_s / self.tau_trap_s

    @property
    def recombination_dominated_at_start(self) -> bool:
        return self.x0 > self.crossover_density


def solve(x0: float,
          tau_trap_s: float,
          tau_0_s: float = TAU_0_AL_S,
          t_end_s: float = 3e-3,
          n_steps: int = 20000) -> Decay:
    """Integrate dx/dt = -x^2/tau_0 - x/tau_trap.

    Closed form exists, but integrating keeps the two channels visible and
    makes it trivial to add diffusion or a time-dependent source later.
    """
    t = np.linspace(0.0, t_end_s, n_steps)
    dt = t[1] - t[0]
    x = np.empty(n_steps)
    x[0] = x0
    for i in range(1, n_steps):
        xi = x[i - 1]
        dx = -(xi * xi) / tau_0_s - xi / tau_trap_s
        xi = xi + dx * dt
        x[i] = max(xi, 0.0)
    return Decay(t, x, tau_0_s, tau_trap_s, x0)


def fit_exponential_tail(d: Decay, lo_frac: float = 0.4,
                         hi_frac: float = 0.9) -> tuple[float, float]:
    """Fit exp(-t/tau) over a window of the decay. Returns (tau_s, r2).

    A LOW r2 is the informative outcome: it says the decay is not exponential,
    which means recombination is still dominating and the observed 369 +/- 6 us
    exponential cannot be that regime.
    """
    n = len(d.t_s)
    sl = slice(int(lo_frac * n), int(hi_frac * n))
    t, x = d.t_s[sl], d.x[sl]
    good = x > 0
    t, x = t[good], x[good]
    if len(t) < 10:
        return float("nan"), 0.0
    y = np.log(x)
    a, b = np.polyfit(t, y, 1)
    pred = a * t + b
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return (-1.0 / a if a < 0 else float("inf")), r2


def qp_from_deposit(energy_keV: float, volume_um3: float) -> float:
    """Normalised QP density x_qp from a deposition spread over a film volume.

    Two quasiparticles per broken pair.
    """
    pairs = PAIR_BREAKING_FRACTION * (energy_keV * 1e3) / TWO_DELTA_AL_EV
    n_qp = 2.0 * pairs
    return n_qp / (N_CP_AL_PER_UM3 * volume_um3)


def implied_tau_trap(observed_tau_s: float) -> float:
    """If the observed decay is exponential, the trapping time IS the observed
    time constant. Stated as a function so the assumption is explicit."""
    return observed_tau_s


def density_ceiling(observed_tau_s: float,
                    tau_0_s: float = TAU_0_AL_S) -> float:
    """Upper bound on x_qp DURING THE FITTED WINDOW, not at the peak.

    CORRECTED: an exponential TAIL does not bound the initial density, because
    recombination drains any peak below this crossover within microseconds.
    It only bounds the density in the window the exponential was fitted over.
    """
    return tau_0_s / observed_tau_s


def early_to_late_ratio(x0: float, tau_trap_s: float,
                        early_frac: float = 0.02) -> float:
    """How much faster the early decay runs than the tail. The burst-size probe."""
    d = solve(x0, tau_trap_s=tau_trap_s, t_end_s=3e-3, n_steps=60000)
    te, _ = fit_exponential_tail(d, 0.0, early_frac)
    tl, _ = fit_exponential_tail(d, 0.4, 0.9)
    return tl / te if te > 0 else float("inf")


def predict_early_decay(energy_keV: float, volume_um3: float,
                        tau_trap_s: float) -> dict:
    """The falsifiable prediction.

    Given a hypothesised deposition and the film volume it spreads over, what
    should the FIRST tens of microseconds of recovery look like? If a device's
    early-time data disagrees, the hypothesised deposition is wrong.
    """
    x0 = qp_from_deposit(energy_keV, volume_um3)
    d = solve(x0, tau_trap_s=tau_trap_s, t_end_s=3e-3, n_steps=60000)
    te, re = fit_exponential_tail(d, 0.0, 0.02)
    tl, rl = fit_exponential_tail(d, 0.4, 0.9)
    return {"energy_keV": energy_keV, "volume_um3": volume_um3, "x0": x0,
            "early_tau_s": te, "early_r2": re,
            "late_tau_s": tl, "late_r2": rl,
            "speedup": tl / te if te > 0 else float("inf")}
