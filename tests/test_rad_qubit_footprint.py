"""Tests for the per-qubit footprint.

These pin the mechanics the result depends on, not the result itself.
"""
from __future__ import annotations

import numpy as np
import pytest

from radchain import qubit_footprint as qf




def test_grid_is_sixty_qubits_at_the_published_pitch():
    xy = qf.qubit_grid()
    assert len(xy) == 60
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    assert d.min(axis=1).max() == pytest.approx(qf.PITCH_MM)




def test_k_reproduces_the_median_peak_exactly():
    rng = np.random.default_rng(0)
    E = rng.gamma(2.0, 3.0, size=(501, 60))
    k = qf.fit_k(E)
    assert np.median(k * E.max(axis=1)) == pytest.approx(qf.MEDIAN_PEAK_SHIFT_KHZ)


def test_size_shrinks_as_the_threshold_rises():
    rng = np.random.default_rng(1)
    E = rng.exponential(1.0, size=(200, 60))
    s_lo = qf.sizes(E, 100.0, threshold_khz=50.0)
    s_hi = qf.sizes(E, 100.0, threshold_khz=300.0)
    assert (s_hi <= s_lo).all() and s_hi.sum() < s_lo.sum()


def test_pass_bands_are_the_preregistered_ones():
    """Changing these after seeing a run is the failure this test exists for."""
    assert qf.MEDIAN_SIZE_BAND == (11, 19)
    lo, hi = qf.LARGE_SHARE_BAND
    assert lo == pytest.approx(0.0289, abs=5e-4)
    assert hi == pytest.approx(0.0886, abs=5e-4)
    assert qf.LARGE_SHARE == pytest.approx(71 / 1320)
