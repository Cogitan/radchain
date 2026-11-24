"""qubit_footprint.py — footprint as a COUNT OF QUBITS, with a per-qubit threshold.

WHY A QUBIT COUNT
-----------------
Google's gap-engineered Willow array (arXiv:2506.18228) has a median burst of
15 of 60 qubits, 25%. A die-area footprint metric overcounts against that,
because it has no threshold. Google's burst size is defined per qubit, quoted
from their Appendix C:

    "We define the burst size as the number of qubits with |dfq(0)| > 3 sigma_f
     ~ 200 kHz"

A qubit far from the strike still absorbs phonons, but too few to shift it past
200 kHz. The old metric has no threshold, so it counts that qubit. This module
does the qubit-level count.

THE MODEL, AND WHY IT HAS ONE FREE CONSTANT
-------------------------------------------
1. Transport: a phonon backend's `film_hits` (`radchain.phonon`; the default
   is drawn from recorded G4CMP kernels), one draw per geometry; each strike is
   the same hit cloud translated to a uniform random position and clipped to the
   die.
2. Energy per qubit: a strike depositing E keV gives qubit q the energy
   E_q = E * (hits in q's cell) / n_phonons.
3. Response: the frequency shift is linear in QP density near the junction
   (their Eq. 1, before recombination acts), so df_q = k * E_q. `k` (kHz per
   keV absorbed in a cell) folds together pair-breaking efficiency, the
   sensitive area of a qubit inside its cell, and dfq/x_qp. It is NOT
   predicted.
4. Detection: only strikes depositing >= 180 keV count. Their Appendix C: the
   rate "is consistent with the integrated event rate for all energy depositions
   >= 0.18 MeV". A quoted cut, not a fit.

`k` is fixed by ONE observable, the median peak shift (2 MHz, their Fig. 8).
Because df is linear in k, the median peak is linear in k and the fit is exact
and closed-form. Then TWO observables are predictions nobody tuned:

    median burst size                 15 of 60 qubits (25%)
    share of bursts over half array   rate 1/(22 min) over 1/(71 s) = 5.4%

That is the test. A model with one constant that gets the peak right by
construction and the size distribution wrong is a model that is missing physics,
and this file must say so rather than sweep geometry until it passes.

WHAT IS UNKNOWN, AND THEREFORE SWEPT, NOT CHOSEN
-----------------------------------------------
  substrate thickness      not published
  top film coverage        not published
  deposition spectrum      above 180 keV, ~82% gammas from the environment; we
                           have no measured spectrum for their fridge, so a
                           power law dN/dE ~ E^-gamma is swept
  qubit layout             60 qubits taken as an 8x8 grid minus corners at the
                           published 1.15 mm mean spacing; Willow's real lattice
                           is not reproduced

With the shipped G4CMP-kernel backend, the transport is G4CMP's stock example
geometry (8 mm die, copper housing) or the plain-film trap grid (4 mm die).
Neither has Willow's bump-bonded carrier chip, which is a phonon sink through
the bumps and would shorten the reach of a strike.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from radchain import phonon as ph

# ------------------------------------------------------------------ MEASURED
# arXiv:2506.18228, Appendix C and Fig. 8. 265 bursts in 5.2 h, 60 qubits.

N_QUBITS = 60
PITCH_MM = 1.15
SIZE_THRESHOLD_KHZ = 200.0
MEDIAN_PEAK_SHIFT_KHZ = 2000.0
MEDIAN_SIZE_QUBITS = 15
DETECTION_CUT_KEV = 180.0
N_BURSTS = 265
HOURS = 5.2
RATE_ALL_PER_S = 1.0 / 71.0
RATE_LARGE_PER_S = 1.0 / (22 * 60.0)
LARGE_SHARE = RATE_LARGE_PER_S / RATE_ALL_PER_S          # 0.0538

# ------------------------------------------------------------ PASS CRITERIA
# Fixed BEFORE the first run (2026-09-16). Not to be widened after seeing it.
#
# Large-burst share: ~14 large bursts (5.2 h / 22 min) among 265. Poisson 95%
# interval on 14 counts is [7.65, 23.49], so the share band is [2.9%, 8.9%].
LARGE_SHARE_BAND = (7.65 / N_BURSTS, 23.49 / N_BURSTS)
# Median size: the paper gives no interval. +/- 4 qubits (+/- 27%) is a
# tolerance we chose, not a confidence interval, and is reported as such.
MEDIAN_SIZE_BAND = (MEDIAN_SIZE_QUBITS - 4, MEDIAN_SIZE_QUBITS + 4)


def qubit_grid(pitch_mm: float = PITCH_MM) -> np.ndarray:
    """60 qubit sites: 8x8 at `pitch_mm`, corners removed, centred on 0."""
    ij = [(i, j) for i in range(8) for j in range(8)
          if (i, j) not in {(0, 0), (0, 7), (7, 0), (7, 7)}]
    xy = (np.array(ij, dtype=float) - 3.5) * pitch_mm
    assert len(xy) == N_QUBITS
    return xy


@dataclass
class Geometry:
    """Willow-class qubit grid. ``phonon`` is passed to the phonon backend.

    With the default G4CMP backend it takes ``thickness_mm`` and, optionally,
    ``trap_pitch_mm``, ``trap_coverage``, ``trap_absorption`` and ``nearest``.
    If ``phonon`` is None, ``{"thickness_mm": thickness_mm}`` is used.
    """
    thickness_mm: float = 0.381
    margin_mm: float = 0.5
    pitch_mm: float = PITCH_MM
    phonon: dict | None = None
    backend: str | None = None

    def phonon_params(self) -> dict:
        return dict(self.phonon) if self.phonon is not None else             {"thickness_mm": self.thickness_mm}

    @property
    def die_half_mm(self) -> float:
        return 4 * self.pitch_mm + self.margin_mm


def sample_energies(n: int, gamma: float, rng: np.random.Generator,
                    e_min: float = DETECTION_CUT_KEV,
                    e_max: float = 5000.0) -> np.ndarray:
    """Deposits above the detection cut, dN/dE ~ E^-gamma on [e_min, e_max]."""
    u = rng.random(n)
    if abs(gamma - 1.0) < 1e-12:
        return e_min * (e_max / e_min) ** u
    a = 1.0 - gamma
    return (e_min**a + u * (e_max**a - e_min**a)) ** (1.0 / a)


def cell_energies(hx: np.ndarray, hy: np.ndarray, n_phonons: int,
                  x0: float, y0: float, e_keV: float, geo: Geometry,
                  sites: np.ndarray) -> np.ndarray:
    """Energy (keV) absorbed in each qubit's cell for one strike at (x0, y0).

    A cell is the pitch-sized square centred on the qubit. Hits outside the die
    are lost to the edge (the ray left the die and cannot return).
    """
    x = hx + x0
    y = hy + y0
    h = geo.die_half_mm
    on_die = (np.abs(x) <= h) & (np.abs(y) <= h)
    x, y = x[on_die], y[on_die]
    p = geo.pitch_mm
    # Grid index of the cell each hit falls in (sites are at (i-3.5)*p).
    i = np.floor(x / p + 4.0).astype(int)
    j = np.floor(y / p + 4.0).astype(int)
    ok = (i >= 0) & (i < 8) & (j >= 0) & (j < 8)
    counts = np.zeros((8, 8))
    np.add.at(counts, (i[ok], j[ok]), 1.0)
    si = np.rint(sites[:, 0] / p + 3.5).astype(int)
    sj = np.rint(sites[:, 1] / p + 3.5).astype(int)
    return e_keV * counts[si, sj] / n_phonons


@dataclass
class Grade:
    geo: Geometry
    gamma: float | None
    n_events: int
    k_khz_per_keV: float
    median_size: float
    large_share: float

    @property
    def size_pass(self) -> bool:
        lo, hi = MEDIAN_SIZE_BAND
        return lo <= self.median_size <= hi

    @property
    def large_pass(self) -> bool:
        lo, hi = LARGE_SHARE_BAND
        return lo <= self.large_share <= hi

    @property
    def passes(self) -> bool:
        return self.size_pass and self.large_pass


def event_cell_energies(geo: Geometry, gamma: float | None, n_events: int = 2000,
                        n_phonons: int = 200_000, seed: int = 0,
                        hits: tuple | None = None,
                        deposits_keV: np.ndarray | None = None) -> np.ndarray:
    """(n_events, 60) array of per-qubit absorbed energy for detected strikes."""
    if hits is None:
        hits = ph.get_backend(geo.backend).film_hits(
            n_phonons=n_phonons, seed=seed, **geo.phonon_params())
    hx, hy = hits
    rng = np.random.default_rng(seed + 1)
    sites = qubit_grid(geo.pitch_mm)
    h = geo.die_half_mm
    xs = rng.uniform(-h, h, n_events)
    ys = rng.uniform(-h, h, n_events)
    if deposits_keV is not None:
        # MEASURED spectrum: resample the detected (>= cut) deposits.
        pool = np.asarray(deposits_keV, dtype=float)
        pool = pool[pool >= DETECTION_CUT_KEV]
        es = rng.choice(pool, size=n_events, replace=True)
    else:
        es = sample_energies(n_events, gamma, rng)
    return np.stack([cell_energies(hx, hy, n_phonons, xs[e], ys[e], es[e],
                                   geo, sites) for e in range(n_events)])


def detected_mixed_pool(n: int, rng: np.random.Generator,
                        thickness_mm: float = 0.5,
                        muon_share: float = 0.184) -> np.ndarray:
    """Measured gamma-line deposits plus muons, mixed AMONG DETECTED events.

    Both pools are cut at DETECTION_CUT_KEV BEFORE mixing, so ``muon_share``
    (default rate_crosscheck.D1_MUON_SHARE) is the muon share of detected
    events. Mixing raw deposits and cutting afterwards would raise the muon
    share to ~32%, because muons pass the cut twice as often.
    """
    import json

    from radchain import _data

    def detected(source):
        v = []
        for f in _data.env_files(source, thickness_mm):
            with open(f) as fh:
                v += json.load(fh)["deposits_keV"]
        v = np.asarray(v, dtype=float)
        return v[v >= DETECTION_CUT_KEV]

    g, mu = detected("gamma-lines"), detected("muon")
    n_mu = int(round(muon_share * n))
    return np.concatenate([rng.choice(g, n - n_mu), rng.choice(mu, n_mu)])


def fit_k(E: np.ndarray, median_peak_khz: float = MEDIAN_PEAK_SHIFT_KHZ) -> float:
    """Exact: median_e(k * max_q E) = k * median_e(max_q E)."""
    return median_peak_khz / float(np.median(E.max(axis=1)))


def sizes(E: np.ndarray, k: float,
          threshold_khz: float = SIZE_THRESHOLD_KHZ) -> np.ndarray:
    return (k * E > threshold_khz).sum(axis=1)


def grade(geo: Geometry, gamma: float | None, n_events: int = 2000,
          n_phonons: int = 200_000, seed: int = 0,
          hits: tuple | None = None,
          deposits_keV: np.ndarray | None = None) -> Grade:
    E = event_cell_energies(geo, gamma, n_events, n_phonons, seed, hits,
                            deposits_keV)
    k = fit_k(E)
    s = sizes(E, k)
    return Grade(geo=geo, gamma=gamma, n_events=n_events, k_khz_per_keV=k,
                 median_size=float(np.median(s)),
                 large_share=float(np.mean(s > N_QUBITS / 2)))


# ------------------------------------------------------------- ANY LAYOUT
# Any qubit layout, not just Willow's 60 sites rescaled by pitch (which would
# silently give a 17- or 400-qubit chip Willow's answer).
#
# The rule is the one above, generalised: a qubit absorbs the phonons landing
# in its own cell, a square of side `cell_mm` centred on it, and where squares
# overlap a hit goes to the nearer qubit (Chebyshev distance, so on a square
# grid with cell = pitch the cells are exactly the grid squares and the old
# rule is reproduced). `cell_mm` defaults to the median nearest-neighbour
# spacing, which is the pitch on a grid. Like `k`, the cell is a modelling
# choice, not a measured sensitive area.

#: Raster resolution: pixels per cell side, tried finest first, and the pixel
#: budget that decides how fine we can afford. On a regular grid with
#: cell = spacing the pixel edges fall on the cell edges, so the raster is
#: exact there; on an irregular layout the boundary error is ~1/(2K) of a cell.
RASTER_K = (32, 16, 8, 4)
#: 4 Mpx costs ~1.1 s to label (0.27 s per Mpx, measured 2026-10-09 on the
#: bench machine); 10,000 qubits fit at 16 px per cell.
RASTER_MAX_PIXELS = 4_000_000
RASTER_S_PER_PIXEL = 0.27e-6
MAX_QUBITS = 10_000


@dataclass(frozen=True)
class Layout:
    """Qubit sites in mm with the origin at the DIE CENTRE, and the die."""
    sites: np.ndarray
    die_w_mm: float
    die_h_mm: float
    cell_mm: float
    description: str

    @property
    def n(self) -> int:
        return len(self.sites)

    @property
    def spacing_mm(self) -> float | None:
        return nearest_neighbour_mm(self.sites)


def grid_sites(n: int, pitch_mm: float) -> tuple[np.ndarray, str]:
    """`n` sites on a square grid at `pitch_mm`, filled from the centre out.

    The smallest s x s grid that holds `n`, keeping the `n` sites nearest its
    centre. For n = 60 that is 8x8 minus the four corners, which is exactly
    `qubit_grid`, so Willow's graded layout is a special case, not a branch.
    """
    s = int(np.ceil(np.sqrt(n)))
    ij = np.array([(i, j) for i in range(s) for j in range(s)], dtype=float)
    xy = (ij - (s - 1) / 2.0) * pitch_mm
    d2 = np.round((xy ** 2).sum(axis=1) / pitch_mm ** 2, 9)
    keep = np.lexsort((ij[:, 0], ij[:, 1], d2))[:n]
    dropped = s * s - n
    desc = "{0}x{0} grid".format(s)
    if dropped == 4 and s > 2:
        desc += " minus corners"
    elif dropped:
        desc += " minus {} outer sites".format(dropped)
    return xy[np.sort(keep)], "{}, {} sites".format(desc, n)


def nearest_neighbour_mm(sites: np.ndarray) -> float | None:
    """Median nearest-neighbour distance, or None for a single site."""
    if len(sites) < 2:
        return None
    from scipy.spatial import cKDTree
    d, _ = cKDTree(sites).query(sites, k=2)
    return float(np.median(d[:, 1]))


def make_layout(*, n_qubits: int | None, pitch_mm: float,
                die_w_mm: float, die_h_mm: float,
                qubit_xy_mm: list | None = None,
                cell_mm: float | None = None) -> tuple[Layout | None, list[str]]:
    """Build the layout the arguments describe, or say in words why it can't be.

    `qubit_xy_mm` is measured from the die's LOWER-LEFT corner, the way a
    layout file is drawn. Without it, `n_qubits` sites go on a centred grid at
    `pitch_mm`. Returns (layout, problems); a non-empty `problems` means the
    layout is rejected, never clamped or repaired.
    """
    problems: list[str] = []
    if qubit_xy_mm is not None:
        sites = np.asarray(qubit_xy_mm, dtype=float).reshape(-1, 2)
        if n_qubits is not None and n_qubits != len(sites):
            problems.append(
                "n_qubits={} but qubit_xy_mm lists {} positions; pass one or "
                "make them agree".format(n_qubits, len(sites)))
        sites = sites - np.array([die_w_mm / 2.0, die_h_mm / 2.0])
        desc = "custom, {} sites".format(len(sites))
    else:
        sites, desc = grid_sites(int(n_qubits), pitch_mm)
    if len(sites) == 0:
        return None, problems + ["the layout has no qubits"]
    if len(sites) > MAX_QUBITS:
        problems.append("{} qubits is over the {} the footprint will "
                        "simulate".format(len(sites), MAX_QUBITS))
    off = (np.abs(sites[:, 0]) > die_w_mm / 2.0 + 1e-9) | \
          (np.abs(sites[:, 1]) > die_h_mm / 2.0 + 1e-9)
    if off.any():
        corner = sites[off][:3] + np.array([die_w_mm / 2.0, die_h_mm / 2.0])
        need = ""
        if qubit_xy_mm is None:
            w, h = 2.0 * np.abs(sites).max(axis=0)
            need = (" This grid of {} at {:g} mm needs a die of at least "
                    "{:.3g} x {:.3g} mm.".format(len(sites), pitch_mm, w, h))
        problems.append(
            "{} of {} qubits sit off the {:g} x {:g} mm die (first, from the "
            "lower-left corner: {}). A qubit off the die cannot absorb phonons "
            "from it; enlarge the die or move the qubits.{}".format(
                int(off.sum()), len(sites), die_w_mm, die_h_mm,
                ", ".join("({:.3g}, {:.3g})".format(*p) for p in corner),
                need))
    nn = nearest_neighbour_mm(sites)
    if nn is not None and nn < 1e-6:
        problems.append("two or more qubits share a position; each needs its "
                        "own site")
    if cell_mm is None:
        cell_mm = nn if nn is not None else pitch_mm
    if not cell_mm or cell_mm <= 0:
        problems.append("qubit_cell_mm must be positive")
    if problems:
        return None, problems
    return Layout(sites=sites, die_w_mm=float(die_w_mm),
                  die_h_mm=float(die_h_mm), cell_mm=float(cell_mm),
                  description=desc), []


@dataclass(frozen=True)
class Raster:
    """Pixel -> qubit index (-1 for no qubit), covering every qubit's cell."""
    x0: float
    y0: float
    px: float
    labels: np.ndarray


def label_raster(layout: Layout) -> Raster:
    """Assign each pixel to the qubit whose cell holds its centre.

    Raises ValueError if even the coarsest resolution is over the pixel
    budget; callers should turn that into an error with the reason.
    """
    from scipy.spatial import cKDTree
    c = layout.cell_mm
    lo = layout.sites.min(axis=0) - c / 2.0
    span = layout.sites.max(axis=0) + c / 2.0 - lo
    for K in RASTER_K:
        px = c / K
        nx, ny = (int(np.ceil(s / px - 1e-9)) for s in span)
        if nx * ny <= RASTER_MAX_PIXELS:
            break
    else:
        raise ValueError(
            "the qubit cells cover {:.3g} x {:.3g} mm at a {:.3g} mm cell, "
            "over {:,} pixels even at {} per cell; use a larger "
            "qubit_cell_mm".format(span[0], span[1], c, RASTER_MAX_PIXELS,
                                   RASTER_K[-1]))
    gx = lo[0] + (np.arange(nx) + 0.5) * px
    gy = lo[1] + (np.arange(ny) + 0.5) * px
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    _, idx = cKDTree(layout.sites).query(
        np.column_stack([X.ravel(), Y.ravel()]), p=np.inf,
        distance_upper_bound=c / 2.0)
    labels = np.where(idx < layout.n, idx, -1).astype(np.int32)
    return Raster(x0=float(lo[0]), y0=float(lo[1]), px=float(px),
                  labels=labels.reshape(nx, ny))


def layout_cell_energies(hx: np.ndarray, hy: np.ndarray, n_phonons: int,
                         x0: float, y0: float, e_keV: float, layout: Layout,
                         raster: Raster) -> np.ndarray:
    """`cell_energies` for any layout: keV absorbed by each qubit's cell."""
    x = hx + x0
    y = hy + y0
    on_die = (np.abs(x) <= layout.die_w_mm / 2.0) & \
             (np.abs(y) <= layout.die_h_mm / 2.0)
    x, y = x[on_die], y[on_die]
    i = np.floor((x - raster.x0) / raster.px).astype(np.int64)
    j = np.floor((y - raster.y0) / raster.px).astype(np.int64)
    nx, ny = raster.labels.shape
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    lab = raster.labels[i[ok], j[ok]]
    counts = np.bincount(lab[lab >= 0], minlength=layout.n)
    return e_keV * counts / n_phonons


def layout_event_energies(layout: Layout, *, n_events: int, seed: int,
                          deposits_keV: np.ndarray,
                          phonon: dict | None = None,
                          backend: str | None = None,
                          n_phonons: int = 200_000,
                          hits: tuple | None = None,
                          deposit_probs: np.ndarray | None = None,
                          return_deposits: bool = False):
    """(n_events, n_qubits) absorbed energy, strikes uniform over the die.

    Draws in the same order as `event_cell_energies`, so on Willow's grid and
    die it returns the same array. Give either ``hits`` or ``phonon`` (the
    backend's geometry keywords, e.g. ``{"thickness_mm": 0.381}``).
    """
    if hits is None:
        if phonon is None:
            raise ValueError("give hits= or phonon= (backend geometry keywords)")
        hits = ph.get_backend(backend).film_hits(n_phonons=n_phonons, seed=seed,
                                                 **phonon)
    hx, hy = hits
    raster = label_raster(layout)
    rng = np.random.default_rng(seed + 1)
    xs = rng.uniform(-layout.die_w_mm / 2.0, layout.die_w_mm / 2.0, n_events)
    ys = rng.uniform(-layout.die_h_mm / 2.0, layout.die_h_mm / 2.0, n_events)
    pool = np.asarray(deposits_keV, dtype=float)
    if deposit_probs is None:
        pool = pool[pool >= DETECTION_CUT_KEV]
        es = rng.choice(pool, size=n_events, replace=True)
    else:
        # An interpolated spectrum (env_grid): deposits carry weights. Only
        # used off the Geant4 grid, so on it the draws are unchanged.
        keep = pool >= DETECTION_CUT_KEV
        p = np.asarray(deposit_probs, dtype=float)[keep]
        es = rng.choice(pool[keep], size=n_events, replace=True, p=p / p.sum())
    E = np.stack([layout_cell_energies(hx, hy, n_phonons, xs[e], ys[e],
                                       es[e], layout, raster)
                  for e in range(n_events)])
    # The per-event deposits, for energy-dependent responses. Same draws
    # either way.
    return (E, es) if return_deposits else E


def _main() -> None:
    print("=" * 78)
    print("QUBIT FOOTPRINT vs arXiv:2506.18228 (Willow, gap-engineered, no traps)")
    print("phonon backend: G4CMP kernels, stock example geometry (no back-side traps)")
    print("=" * 78)
    print(f"  fitted on: median peak shift {MEDIAN_PEAK_SHIFT_KHZ:.0f} kHz  (k, exact)")
    print(f"  predicted: median size {MEDIAN_SIZE_QUBITS}/60, pass band "
          f"{MEDIAN_SIZE_BAND}")
    print(f"             share > 30 qubits {LARGE_SHARE:.1%}, pass band "
          f"[{LARGE_SHARE_BAND[0]:.1%}, {LARGE_SHARE_BAND[1]:.1%}]\n")
    print(f"  {'thick':>6s} {'gamma':>5s} | "
          f"{'k kHz/keV':>9s} {'med size':>8s} {'>half':>6s} | fails")
    n_ph = 200_000
    backend = ph.get_backend()
    for t in (0.2, 0.381, 0.6, 1.0):
        hits = backend.film_hits(n_phonons=n_ph, seed=0, thickness_mm=t)
        for g in (1.5, 2.0, 3.0):
            r = grade(Geometry(thickness_mm=t), g, n_phonons=n_ph, hits=hits)
            v = ("PASS" if r.passes else  # else: which band(s) it FAILS
                 "size" if not r.size_pass and r.large_pass else
                 "large" if r.size_pass else "both")
            print(f"  {t:6.3f} {g:5.1f} | "
                  f"{r.k_khz_per_keV:9.1f} {r.median_size:5.0f}/60 "
                  f"{r.large_share:6.1%} | {v}")
        print()


if __name__ == "__main__":
    _main()
