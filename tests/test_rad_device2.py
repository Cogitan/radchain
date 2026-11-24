"""Device 2 (Li et al. 2025) cross-check. Pins the measured inputs and the
solver logic that produced the verdict, so a later edit cannot turn an
unreachable ratio into a silent pass (it did exactly that once)."""
from __future__ import annotations

import numpy as np
import pytest

from radchain import device2_li2025 as D


def test_measured_constants_match_the_paper():
    assert (D.T_CPJ_ALL_S, D.T_CPJ_MUON_S) == (12.7, 67.0)
    assert (D.T_BF_ALL_S, D.T_BF_MUON_S) == (386.0, 1500.0)
    assert D.AREA_CM2 == pytest.approx(2.25)
    assert D.incident_muons_per_hour() == pytest.approx(54.0)
    assert D.AL_VOLUME_UM3 == pytest.approx(3.132)


def test_gamma_ratio_is_the_one_measured():
    g_cpj = D.per_hour(D.T_CPJ_ALL_S) - D.per_hour(D.T_CPJ_MUON_S)
    g_bf = D.per_hour(D.T_BF_ALL_S) - D.per_hour(D.T_BF_MUON_S)
    assert g_bf / g_cpj == pytest.approx(0.0301, abs=5e-4)


def test_unreachable_ratio_returns_zero_not_a_threshold():
    """A spectrum whose floor is above the target must not yield a threshold."""
    spec = np.sort(np.array([100.0, 200.0, 300.0, 900.0]))
    assert D.solve_sensitive_threshold(spec, ratio=0.01, e_blunt=691.0) == 0.0


def test_solver_inverts_a_reachable_ratio():
    spec = np.sort(np.concatenate([np.full(97, 50.0), np.full(3, 800.0)]))
    e = D.solve_sensitive_threshold(spec, ratio=0.03, e_blunt=691.0)
    assert e == 50.0
    assert D.fraction_above(spec, 691.0) / D.fraction_above(spec, e) == \
        pytest.approx(0.03)


def test_threshold_for_efficiency_is_a_percentile():
    spec = np.sort(np.arange(1, 1001, dtype=float))
    assert D.threshold_for_efficiency(spec, 0.10, 1000) == pytest.approx(901.0)
    assert D.threshold_for_efficiency(spec, 1.0, 1000) == 0.0


