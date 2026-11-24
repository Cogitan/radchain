"""Tests for the corrected recovery layer.

The old model assumed a trapping exponential with a literature constant on a
device with no traps and a measured rate. These pin the corrected relations to
arXiv:2506.18228's own equations, so a future edit cannot quietly drift back.
"""
from __future__ import annotations

import numpy as np
import pytest

from radchain import recovery_google2025 as R


def test_decay_solves_their_equation_2():
    """dx/dt = -r x^2 must hold for the closed form, not just look like it."""
    t = np.linspace(0, 1e-3, 100_001)
    x = R.decay(t, 4.3e-4)
    dxdt = np.gradient(x, t)
    rhs = -(x**2) / R.R_RECOMB_S
    assert np.allclose(dxdt[1:-1], rhs[1:-1], rtol=1e-4)


def test_trec_is_their_equation_3():
    df0 = 2.0e6
    t = R.t_rec_s(df0)
    # At t = trec the shift must be exactly half its initial value.
    x0 = R.xqp_from_shift(df0)
    assert R.shift_from_xqp(float(R.decay(np.array([t]), x0)[0])) == pytest.approx(df0 / 2)


def test_median_burst_density_is_near_their_measured_epicentre():
    """2 MHz through their Eq. (1) must land near the 3.2e-4 they extracted."""
    x = R.xqp_from_shift(R.MEDIAN_PEAK_SHIFT_HZ)
    assert 0.5 < x / R.XQP_EPICENTRE_FIG3B < 2.0


def test_duration_saturates_in_the_measured_window():
    """The no-free-parameter prediction: any hard burst lasts about 2 ms,
    matching their 1-2 ms for bursts seen without the matched filter."""
    hard = R.burst_duration_s(1e12)
    assert 1e-3 <= hard <= 2.5e-3
    assert R.burst_duration_s(20e6) > R.burst_duration_s(1e6)
    assert R.burst_duration_s(20e6) < hard


def test_volume_inversion_round_trips():
    v = 3.0e4
    e_keV = 100.0
    x = R.xqp_from_energy(e_keV, v)
    k = R.shift_from_xqp(x) / 1e3 / e_keV          # kHz per keV
    # xqp_from_energy treats E as deposit energy, so it carries the 0.57.
    assert R.implied_al_volume_um3(
        k, pair_breaking_fraction=R.PAIR_BREAKING_FRACTION) == pytest.approx(v, rel=1e-9)


def test_the_no_trap_decay_is_not_exponential():
    assert R.exponential_misfit() < 0.99
