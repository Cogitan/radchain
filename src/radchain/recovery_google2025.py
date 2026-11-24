"""recovery_google2025.py — recovery on the MEASURED recombination rate.

WHAT WAS WRONG
--------------
`qp_recovery.py` assumed a trapping-dominated exponential with the literature
tau_0 = 438 ns. Google's gap-engineered Willow has NO traps and a measured
recombination rate r = 1/(88 +/- 12 ns) (arXiv:2506.18228). Wrong regime, and a
5x wrong constant. This module is the corrected layer.

THEIR MODEL, QUOTED IN SYMBOLS
------------------------------
    dfq/fq = -a * xqp             Eq. (1), a ~= 0.77, fq ~= 6 GHz
    dxqp/dt = -r * xqp^2          Eq. (2), r = 1/(88 +/- 12 ns)
    dfq(t) = dfq(0)/(1 + t/trec)  Eq. (3), trec = (1/r) * a*fq/dfq(0)

So recovery is a power law, not an exponential, and trec is not a constant of
the device: it is set by how hard the burst hit. A big burst recovers FASTER in
absolute terms, which is the opposite of the trapping picture we had.

THE ONE GENUINELY INDEPENDENT TEST IN HERE
------------------------------------------
Everything above is their algebra; feeding their numbers back through it proves
nothing. The test that is ours:

    The footprint layer fits ONE constant k (kHz of shift per keV absorbed in a
    qubit cell). Physics predicts that constant, given the volume of low-gap
    aluminium the quasiparticles occupy near the junction:

        broken pairs = 0.57 * E / (2*Delta)      (2*Delta = 0.36 meV)
        xqp          = 2 * pairs / (n_cp * V)    (n_cp = 4e6 per um^3)
        dfq          = a * fq * xqp

    Invert k to get the implied V. If the footprint calibration is fitting
    physics, V lands in the range a transmon's low-gap aluminium actually
    occupies (roughly 1e3 to 1e6 um^3: pads of order 0.1-0.5 mm across and
    100 nm thick, plus ground plane nearby). If it lands at 1 um^3 or 1e12, the
    footprint calibration matched two numbers by accident.

Google's own anchor for the same quantity: xqp(0) ~= 3.2e-4 at the epicentre of
the burst in their Fig. 3(b).

NOT MODELLED, AND WHY
---------------------
The T1 error burst (35 +/- 15 us) is a different observable: it needs QPs above
the gap difference d(Delta)/h = 12 GHz, so it is about the QP energy
distribution cooling, not the density decaying. Their Fig. 4 separates the two
timescales explicitly. Predicting it needs a cooling model this chain does not
have, and claiming it from a density model would be the same error as before.
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ MEASURED
R_RECOMB_S = 88e-9            #: 1/r, arXiv:2506.18228 (fit spread +/- 12 ns)
R_RECOMB_ERR_S = 12e-9
A_COEFF = 0.77                #: dfq/fq = -a xqp, their theoretical estimate
F_Q_HZ = 6.0e9                #: qubit frequency, 5.9-6.5 GHz band
XQP_EPICENTRE_FIG3B = 3.2e-4  #: their extracted epicentre density
MEDIAN_PEAK_SHIFT_HZ = 2.0e6

# --------------------------------------------------------------- MATERIALS
TWO_DELTA_AL_EV = 3.6e-4
N_CP_AL_PER_UM3 = 4.0e6
PAIR_BREAKING_FRACTION = 0.57

#: Plausible low-gap aluminium volume per qubit, um^3. A transmon pad pair of
#: order 0.1-0.5 mm across at 100 nm thickness is 1e3-2.5e4 um^3; the nearby
#: ground plane can add an order of magnitude. Stated as a RANGE to test
#: against, fixed before the comparison.
V_AL_PLAUSIBLE_UM3 = (1.0e3, 1.0e6)


def xqp_from_shift(df_hz: float, a: float = A_COEFF, fq: float = F_Q_HZ) -> float:
    """Eq. (1) inverted: a frequency shift is a QP density."""
    return df_hz / (a * fq)


def shift_from_xqp(xqp: float, a: float = A_COEFF, fq: float = F_Q_HZ) -> float:
    return a * fq * xqp


def xqp_from_energy(e_keV: float, volume_um3: float) -> float:
    """Absorbed energy -> QP density in a low-gap aluminium volume."""
    pairs = PAIR_BREAKING_FRACTION * (e_keV * 1e3) / TWO_DELTA_AL_EV
    return 2.0 * pairs / (N_CP_AL_PER_UM3 * volume_um3)


def decay(t_s: np.ndarray, xqp0: float, r_s: float = R_RECOMB_S) -> np.ndarray:
    """Eq. (2) solved: pure recombination, no traps."""
    return xqp0 / (1.0 + t_s * xqp0 / r_s)


def t_rec_s(df0_hz: float, r_s: float = R_RECOMB_S, a: float = A_COEFF,
            fq: float = F_Q_HZ) -> float:
    """Eq. (3): the recovery time a burst of this peak shift should show."""
    return r_s * a * fq / df0_hz


def burst_duration_s(df0_hz: float, threshold_hz: float = 2.0e5,
                     r_s: float = R_RECOMB_S, a: float = A_COEFF,
                     fq: float = F_Q_HZ) -> float:
    """How long a burst stays visible: time for the shift to fall to threshold.

    From Eq. (3): duration = trec * (df0/threshold - 1) = (r*a*fq/threshold)
    - (r*a*fq/df0). For df0 >> threshold this SATURATES at r*a*fq/threshold,
    about 2 ms, independent of how hard the burst hit. No free parameter, and
    testable against their statement that bursts found without the matched
    filter have "durations in the interval 1-2 ms".
    """
    return r_s * a * fq * (1.0 / threshold_hz - 1.0 / df0_hz)


def implied_al_volume_um3(k_khz_per_keV: float, a: float = A_COEFF,
                          fq: float = F_Q_HZ,
                          pair_breaking_fraction: float = 1.0) -> float:
    """Volume of low-gap Al implied by the footprint layer's fitted constant.

    k [kHz/keV] = dfq[kHz] / E[keV], and dfq = a*fq*xqp with
    xqp = 2*f*E[eV]/(2Delta * n_cp * V). Solve for V.

    ``f`` is the fraction of the cell energy E that ends in broken pairs. With
    the G4CMP-kernel backend, E is already counted as quasiparticle births x
    Delta, so f = 1 (the default). Pass PAIR_BREAKING_FRACTION (0.57) only when
    E is phonon energy absorbed by the film rather than energy at the gap.
    """
    pairs_per_keV = pair_breaking_fraction * 1e3 / TWO_DELTA_AL_EV
    return (a * fq * 2.0 * pairs_per_keV) / (N_CP_AL_PER_UM3 * k_khz_per_keV * 1e3)


def exponential_misfit(xqp0: float = 4.3e-4, t_end_s: float = 1e-3) -> float:
    """r^2 of an exponential fit to the recombination decay. Low = power law.

    This is the shape discriminator the old module had backwards for the
    no-trap case: with no trapping channel there is no exponential tail at all.
    """
    t = np.linspace(1e-9, t_end_s, 4000)
    x = decay(t, xqp0)
    y = np.log(x)
    p = np.polyfit(t, y, 1)
    res = y - np.polyval(p, t)
    return float(1.0 - np.sum(res**2) / np.sum((y - y.mean()) ** 2))


def _main() -> None:
    print("=" * 74)
    print("RECOVERY vs arXiv:2506.18228 — measured r = 1/(88 +/- 12 ns), no traps")
    print("=" * 74)

    x_med = xqp_from_shift(MEDIAN_PEAK_SHIFT_HZ)
    print(f"  median peak shift 2 MHz  -> xqp(0) = {x_med:.2e}")
    print(f"  their Fig. 3(b) epicentre   xqp(0) = {XQP_EPICENTRE_FIG3B:.2e}")
    print(f"  recovery time for that burst: {t_rec_s(MEDIAN_PEAK_SHIFT_HZ)*1e6:.0f} us")
    print(f"  shift falls below the 200 kHz size threshold after "
          f"{t_rec_s(MEDIAN_PEAK_SHIFT_HZ)*(MEDIAN_PEAK_SHIFT_HZ/2e5 - 1)*1e3:.2f} ms")
    print(f"  exponential fit r^2 to the decay: {exponential_misfit():.3f} "
          f"(power law, not exponential)\n")

    print("  BURST DURATION -- no free parameter (Eq. 3, their constants)")
    for df0 in (0.5e6, 1e6, 2e6, 5e6, 20e6):
        print(f"  {df0/1e6:9.1f} MHz peak -> visible for {burst_duration_s(df0)*1e3:5.2f} ms")
    print(f"  saturates at {burst_duration_s(1e12)*1e3:.2f} ms for a hard hit; "
          f"measured 1-2 ms\n")

    print("  THE INDEPENDENT CHECK — footprint's fitted k, as a volume")
    print(f"  {'k kHz/keV':>10s} {'implied low-gap Al volume':>28s}   verdict")
    from radchain import phonon as ph
    from radchain import qubit_footprint as qf
    rng = np.random.default_rng(7)
    # Measured gamma lines + 18.4% muons among detected events. The deposit
    # pool is the cached 0.5 mm one; the transport thickness is the G4CMP one.
    pool = qf.detected_mixed_pool(40_000, rng)
    backend = ph.get_backend()
    for t in (0.381, 0.6):
        hits = backend.film_hits(seed=0, thickness_mm=t)
        g = qf.grade(qf.Geometry(thickness_mm=t), None,
                     n_events=2000, hits=hits, deposits_keV=pool)
        v = implied_al_volume_um3(g.k_khz_per_keV)
        lo, hi = V_AL_PLAUSIBLE_UM3
        ok = "PLAUSIBLE" if lo <= v <= hi else "OUT OF RANGE"
        print(f"  {g.k_khz_per_keV:10.1f} {v:24.3g} um^3   {ok} "
              f"(median {g.median_size:.0f}/60)")


if __name__ == "__main__":
    _main()
