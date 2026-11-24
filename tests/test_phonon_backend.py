"""The phonon backend interface and the shipped G4CMP-kernel backend."""
import json

import numpy as np
import pytest

from radchain import _data
from radchain import phonon as ph
from radchain import qubit_footprint as qf


@pytest.fixture(scope="module")
def backend():
    return ph.get_backend()


def _records():
    with open(_data.RUNS) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_default_backend_is_g4cmp_kernel(backend):
    assert backend.name == "g4cmp-kernel"
    assert "g4cmp-kernel" in ph.available_backends()


def test_the_trap_grid_and_stock_families_load_as_455_configs(backend):
    # 450 trap-grid configurations + 5 stock (no-trap) thicknesses.
    assert len(backend.kernels) == 455
    stock = [c for c in backend.kernels if c.trap_coverage == 0]
    assert sorted(c.thickness_mm for c in stock) == [0.1, 0.2, 0.381, 0.6, 1.0]


def test_only_g_and_kt_tags_are_pooled(backend):
    recs = _records()
    loaded = sum(k.n_runs for k in backend.kernels.values())
    families = sum(r["tag"].startswith(("g_t", "kt_t")) for r in recs)
    assert loaded == families
    assert loaded < len(recs)          # placement / sensitivity runs ignored


def test_normalisation_is_births_times_gap_over_primary_energy(backend):
    recs = [r for r in _records() if r["tag"].startswith("kt_t0.381_")]
    births = sum(sum(r["counts"]) + r["outside_binning"] for r in recs)
    e_in = sum(r["energy_eV"] * r["n_primaries"] for r in recs)
    k = backend.kernel(thickness_mm=0.381)
    assert k.film_fraction == pytest.approx(births * ph.DELTA_FILM_EV / e_in)
    for k in backend.kernels.values():
        assert 0.0 < k.film_fraction < 1.0


def test_births_outside_the_binning_are_kept_as_an_inner_bin(backend):
    recs = [r for r in _records() if r["tag"].startswith("kt_t0.1_")]
    outside = sum(r["outside_binning"] for r in recs)
    assert outside > 0
    k = backend.kernel(thickness_mm=0.1)
    assert k.bin_edges_mm[0] == 0.0
    e_in = sum(r["energy_eV"] * r["n_primaries"] for r in recs)
    assert k.energy_fraction[0] == pytest.approx(outside * ph.DELTA_FILM_EV / e_in)
    hx, hy = ph.sample_hits(k, 200_000, seed=1)
    r = np.hypot(hx, hy)
    inner = r < k.bin_edges_mm[1]
    assert inner.mean() == pytest.approx(
        k.energy_fraction[0] / k.film_fraction, abs=0.005)


def test_runs_pooled_under_one_tag_must_share_strike_and_energy(tmp_path):
    recs = [r for r in _records() if r["tag"].startswith("kt_t0.381_")][:2]
    recs[1] = dict(recs[1], z_mm=recs[1]["z_mm"] + 0.1)
    p = tmp_path / "runs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    with pytest.raises(ValueError, match="z_mm"):
        ph.load_kernels(p)


def test_off_grid_request_refuses_and_names_neighbours(backend):
    with pytest.raises(KeyError, match="Nearest simulated"):
        backend.film_hits(n_phonons=1000, thickness_mm=0.45)
    hx, _ = backend.film_hits(n_phonons=1000, thickness_mm=0.45, nearest=True)
    assert hx.size > 0


def test_hits_are_reproducible_and_follow_the_kernel(backend):
    a = backend.film_hits(n_phonons=50_000, seed=3, thickness_mm=0.381)
    b = backend.film_hits(n_phonons=50_000, seed=3, thickness_mm=0.381)
    assert np.array_equal(a[0], b[0])
    k = backend.kernel(thickness_mm=0.381)
    assert abs(a[0].size / 50_000 - k.film_fraction) < 0.01
    r = np.hypot(*a)
    assert r.max() <= k.bin_edges_mm[-1]


def test_no_event_gives_its_qubits_more_than_its_deposit(backend):
    hits = backend.film_hits(n_phonons=50_000, thickness_mm=0.381)
    rng = np.random.default_rng(0)
    pool = qf.detected_mixed_pool(5000, rng)
    E, es = qf.layout_event_energies(
        qf.make_layout(n_qubits=60, pitch_mm=1.15, die_w_mm=10.2,
                       die_h_mm=10.2)[0],
        n_events=300, seed=2, deposits_keV=pool, hits=hits, n_phonons=50_000,
        return_deposits=True)
    assert (E >= 0).all()
    assert (E.sum(axis=1) <= es * (1 + 1e-12)).all()


def test_the_detected_pool_has_the_stated_muon_share():
    rng = np.random.default_rng(0)
    pool = qf.detected_mixed_pool(10_000, rng, muon_share=0.184)
    assert (pool >= qf.DETECTION_CUT_KEV).all()
    assert pool.size == 10_000


def test_layout_path_equals_grid_path_on_willow(backend):
    hits = backend.film_hits(n_phonons=50_000, thickness_mm=0.381)
    pool = qf.detected_mixed_pool(5000, np.random.default_rng(0))
    geo = qf.Geometry()
    E1 = qf.event_cell_energies(geo, None, n_events=100, n_phonons=50_000,
                                seed=4, hits=hits, deposits_keV=pool)
    lay = qf.Layout(sites=qf.qubit_grid(geo.pitch_mm),
                    die_w_mm=2 * geo.die_half_mm, die_h_mm=2 * geo.die_half_mm,
                    cell_mm=geo.pitch_mm, description="willow grid")
    E2 = qf.layout_event_energies(lay, n_events=100, seed=4,
                                  deposits_keV=pool, hits=hits,
                                  n_phonons=50_000)
    assert np.allclose(E1, E2)


def test_a_registered_backend_is_used_by_name():
    class Point:
        name = "point"

        def film_hits(self, *, n_phonons, seed=0, **geometry):
            return np.zeros(n_phonons), np.zeros(n_phonons)

    ph.register_backend("point", Point)
    try:
        geo = qf.Geometry(backend="point", phonon={})
        E = qf.event_cell_energies(geo, 2.0, n_events=20, n_phonons=100)
        # Every hit lands at the strike, so at most one qubit per event.
        assert ((E > 0).sum(axis=1) <= 1).all()
    finally:
        ph._REGISTRY.pop("point", None)
        ph._INSTANCES.pop("point", None)
