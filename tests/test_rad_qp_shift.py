"""Per-device frequency-shift constant k: Willow's k times physics ratios.

These pin the transcription of the physics and the direction of every factor.
None of this is a validation on a second device.
"""
from __future__ import annotations

import numpy as np
import pytest

from radchain import qp_shift


def test_a_reproduces_googles_077():
    """Their Eq. 17 at their numbers: Delta/h 52 GHz, dDelta/h 12, fq 6."""
    c = qp_shift.a_coefficient(6.0, 52.0, 12.0)
    assert c.a == pytest.approx(0.77, abs=0.005)


def test_symmetric_junction_is_catelani_eq_73():
    """a = 1/2 + (1/2pi) sqrt(2 Delta / h fq) at phi_0 = 0."""
    c = qp_shift.a_coefficient(6.0, 52.0, 0.0)
    assert c.a == pytest.approx(0.5 + np.sqrt(2 * 52 / 6) / (2 * np.pi))
    assert "Catelani" in c.regime


@pytest.mark.parametrize("gap_diff", [3.0, 5.5, 6.0, 6.9])
def test_unmodelled_gap_differences_are_refused_in_words(gap_diff):
    c = qp_shift.a_coefficient(6.0, 52.0, gap_diff)
    assert c.a is None and c.refused
    assert "not" in c.refused


def test_willow_is_every_ratio_one():
    d = qp_shift.k_for_device()
    assert all(v == pytest.approx(1.0) for v in d["factors_vs_willow"].values())
    assert d["k_khz_per_keV"] == qp_shift.K_WILLOW_KHZ_PER_KEV


@pytest.mark.parametrize("kw, direction", [
    ({"f_film": 0.6, "ref_f_film": 0.3}, -1),  # same energy, twice the film
    ({"cell_mm": 0.8}, +1),         # smaller cell, same density per keV
    ({"gap_ghz": 45.0}, +1),        # lower gap: more QPs per keV, fewer pairs
    ({"gap_diff_ghz": 0.0}, +1),    # QPs reach both sides of the junction
    ({"fq_ghz": 5.0}, -1),          # dfq scales with fq
])
def test_every_factor_moves_k_the_physical_way(kw, direction):
    k = qp_shift.k_for_device(**kw)["k_khz_per_keV"]
    assert np.sign(k - qp_shift.K_WILLOW_KHZ_PER_KEV) == direction


