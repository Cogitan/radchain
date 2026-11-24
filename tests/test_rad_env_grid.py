"""Any substrate thickness, both substrates.

Until 2026-10-09 the thickness snapped to the nearest of 0.3 / 0.5 / 0.725 mm
and every spectrum was silicon, so a sapphire chip got silicon deposition. The
interpolation itself is graded on held-out Geant4 runs by
bench/rad_env_grid_holdout.py; these pin the mechanics.
"""
from __future__ import annotations

import numpy as np
import pytest

from radchain import env_grid, rate_crosscheck as rc

pytestmark = pytest.mark.skipif(
    not (env_grid.available("G4_Si") and env_grid.available("G4_ALUMINUM_OXIDE")),
    reason="env_grid not packed")


def test_the_muon_rate_rises_smoothly_through_an_interpolated_point():
    p = [env_grid.spectrum("silicon", "muon", t).detect_fraction(180.0)
         for t in (0.3, 0.35, 0.4)]
    assert p[0] < p[1] < p[2]


def test_the_muon_peak_moves_with_thickness_rather_than_splitting():
    """Quantile interpolation: the median deposit at 0.35 mm lies between the
    two neighbours' medians, as a single peak."""
    med = [np.median(env_grid.spectrum("silicon", "muon", t).deposits_keV)
           for t in (0.3, 0.35, 0.4)]
    assert med[0] < med[1] < med[2]


@pytest.mark.parametrize("t", [0.049, 2.01])
def test_outside_the_grid_is_refused_in_words(t):
    with pytest.raises(ValueError, match="does not extrapolate"):
        env_grid.spectrum("silicon", "muon", t)


def test_weights_are_per_primary_probabilities():
    for t in (0.3, 0.35, 1.1):
        s = env_grid.spectrum("sapphire", "gamma-lines", t)
        assert 0.0 < s.weights.sum() < 1.0
        assert s.detect_fraction(0.0) == pytest.approx(s.weights.sum())


def test_the_fast_quantiles_are_numpys():
    """_quantiles replaced np.quantile (1.2 s per muon interpolation, up to 8
    per evaluation) with sort-once linear interpolation. Same numbers."""
    rng = np.random.default_rng(0)
    x = rng.gamma(2.0, 50.0, size=5000)
    u = (np.arange(3000) + 0.5) / 3000
    assert np.allclose(env_grid._quantiles(x, u), np.quantile(x, u),
                       rtol=0, atol=1e-10)


def test_cached_spectra_cannot_be_modified():
    s = env_grid.spectrum("silicon", "muon", 0.35)
    with pytest.raises(ValueError):
        s.deposits_keV[0] = 0.0
    assert env_grid.spectrum("silicon", "muon", 0.35) is s
