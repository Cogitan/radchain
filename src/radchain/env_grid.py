"""env_grid.py -- the deposition spectrum at ANY substrate thickness, per material.

WHAT THIS IS FOR
----------------
Snapping the thickness to the nearest of 0.3 / 0.5 / 0.725 mm, or reading
silicon spectra for a sapphire chip, both give wrong deposition. Running
Geant4 on demand is too slow for a sweep. Instead a grid is generated
(`run_env_grid.sh`), packed into `data/env_grid/<material>.npz`
(`pack_env_grid.py`), and interpolated here.

ONE REPRESENTATION FOR EVERY THICKNESS
--------------------------------------
A `Spectrum` is a pool of deposits, each with a weight PER PRIMARY. At a grid
point every weight is 1 / n_primaries, which is exactly the old
"count over n" -- the rate and the footprint read the same numbers they did.
Between grid points t0 < t < t1, with w = (t - t0) / (t1 - t0):

  gammas   MIXTURE: both pools, weights (1-w)/n0 and w/n1. Linear in t of the
           per-primary deposit density. Gamma lines sit at fixed energies and
           must not move with thickness, which a mixture respects.
  muons    QUANTILE (displacement) interpolation: Q_t(u) = (1-w) Q0(u) + w Q1(u).
           The muon deposit is a peak whose position scales with path length;
           a mixture would make it two peaks. Weights carry the interpolated
           per-primary interaction probability.

Both choices were fixed before the held-out test ran. Outside the grid: refused,
not extrapolated.

NOT IN HERE: the die's lateral size. Every grid run used a 10 mm die; the rate
scales the gamma channel by the generation sphere of the requested die and the
muon channel by its area, as before.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from radchain import _data

#: Thicknesses generated and shipped, mm, both materials.
GRID_MM = (0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.725,
           0.85, 1.0, 1.25, 1.5, 2.0)
#: Generated for the held-out test only. Never shipped, never interpolated from.
HOLDOUT_MM = (0.35, 0.675, 1.1)
SOURCES = ("muon", "gamma-lines", "gamma-flat")
MATERIALS = {"silicon": "G4_Si", "sapphire": "G4_ALUMINUM_OXIDE"}
QUANTILE_SOURCES = ("muon",)


@dataclass(frozen=True)
class Spectrum:
    material: str
    source: str
    thickness_mm: float
    deposits_keV: np.ndarray
    #: per-primary weight of each deposit; sums to the interaction probability
    weights: np.ndarray
    #: True only at a grid point, where weights are uniform 1/n
    exact: bool
    method: str
    neighbours_mm: tuple

    def detect_fraction(self, threshold_keV: float) -> float:
        """Probability per primary of a deposit >= threshold."""
        return float(self.weights[self.deposits_keV >= threshold_keV].sum())

    def pool_above(self, cut_keV: float) -> tuple[np.ndarray, np.ndarray | None]:
        """Deposits >= cut and their sampling probabilities. None at a grid
        point, so the caller's uniform resampling (and its random stream) is
        exactly what it was before interpolation existed."""
        m = self.deposits_keV >= cut_keV
        d = self.deposits_keV[m].astype(float)
        if self.exact:
            return d, None
        w = self.weights[m]
        return d, w / w.sum()

    def describe(self) -> str:
        if self.exact:
            return "Geant4 run at {:g} mm ({})".format(self.thickness_mm,
                                                       self.material)
        return ("{} interpolation between Geant4 runs at {:g} and {:g} mm "
                "({})".format(self.method, *self.neighbours_mm, self.material))


def _path(material: str):
    return _data.DATA / "env_grid" / (material + ".npz")


@lru_cache(maxsize=4)
def _load(material: str) -> dict:
    with np.load(_path(material)) as z:
        return {k: z[k] for k in z.files}


def _raw(material: str, source: str, t: float) -> tuple[np.ndarray, int]:
    z = _load(material)
    key = "{}|{:g}".format(source, t)
    # Stored as float32 to keep the shipped grid small; g4_env_spectrum rounds
    # every deposit to 0.001 keV, so rounding back restores the run exactly.
    return (np.round(z[key + "|deposits_keV"].astype(np.float64), 3),
            int(z[key + "|n_primaries"]))


def available(material: str) -> bool:
    return _path(material).exists()


def _quantiles(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """np.quantile(x, u) with the default linear method, sorting ONCE.

    np.quantile partitions the array for every requested quantile, which for
    60,000 muon deposits at 60,000 quantiles cost ~1.2 s per call and up to 8
    calls in one footprint evaluation.
    Same arithmetic, same numbers: pinned by tests/test_rad_env_grid.py.
    """
    xs = np.sort(x)
    pos = u * (len(xs) - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def spectrum(substrate: str, source: str, thickness_mm: float) -> Spectrum:
    """The deposition spectrum for one substrate, source and thickness.

    Cached: one evaluation asks for the same few spectra up to eight times.
    Callers must not modify the returned arrays.
    """
    return _spectrum(substrate, source, round(float(thickness_mm), 9))


def _frozen(s: Spectrum) -> Spectrum:
    """Cached spectra are shared between calls, so their arrays are read-only:
    a caller that tried to modify one would raise instead of corrupting the
    next call."""
    s.deposits_keV.setflags(write=False)
    s.weights.setflags(write=False)
    return s


@lru_cache(maxsize=256)
def _spectrum(substrate: str, source: str, thickness_mm: float) -> Spectrum:
    """The deposition spectrum for one substrate, source and thickness.

    Raises ValueError, in words, for an unknown substrate or source, or a
    thickness outside the grid.
    """
    if substrate not in MATERIALS:
        raise ValueError("substrate must be one of {}, got {!r}".format(
            tuple(MATERIALS), substrate))
    if source not in SOURCES:
        raise ValueError("source must be one of {}, got {!r}".format(
            SOURCES, source))
    material = MATERIALS[substrate]
    lo, hi = GRID_MM[0], GRID_MM[-1]
    if not lo - 1e-9 <= thickness_mm <= hi + 1e-9:
        raise ValueError(
            "no deposition spectrum for {:g} mm: Geant4 runs cover {:g} to "
            "{:g} mm and the chain does not extrapolate beyond them".format(
                thickness_mm, lo, hi))
    for t in GRID_MM:
        if abs(t - thickness_mm) < 1e-9:
            d, n = _raw(material, source, t)
            return _frozen(Spectrum(substrate, source, t, d,
                                    np.full(len(d), 1.0 / n), True, "exact",
                                    (t,)))
    i = int(np.searchsorted(GRID_MM, thickness_mm))
    t0, t1 = GRID_MM[i - 1], GRID_MM[i]
    w = (thickness_mm - t0) / (t1 - t0)
    d0, n0 = _raw(material, source, t0)
    d1, n1 = _raw(material, source, t1)
    if source in QUANTILE_SOURCES:
        m = min(len(d0), len(d1))
        u = (np.arange(m) + 0.5) / m
        d = (1 - w) * _quantiles(d0, u) + w * _quantiles(d1, u)
        p_int = (1 - w) * len(d0) / n0 + w * len(d1) / n1
        weights = np.full(m, p_int / m)
        method = "quantile"
    else:
        d = np.concatenate([d0, d1])
        weights = np.concatenate([np.full(len(d0), (1 - w) / n0),
                                  np.full(len(d1), w / n1)])
        method = "mixture"
    return _frozen(Spectrum(substrate, source, float(thickness_mm), d,
                            weights, False, method, (t0, t1)))


#: Primary photon energies each gamma source fires, as in g4_env_spectrum.py.
#: Needed to put a MEASURED site flux onto a spectrum shape 
SCATTER_KEV = (30.0, 2600.0)    # dN/dE ~ 1/E
FLAT_KEV = (200.0, 3000.0)      # uniform


def primary_fraction_above(source: str, e_min_keV: float | None) -> float:
    """Share of a gamma source's primaries with energy >= e_min_keV.

    A site's measured flux above e_min is assigned to this share of the
    source's photons, so the source's total flux is measured / fraction.
    """
    e = 0.0 if e_min_keV is None else float(e_min_keV)
    if source == "gamma-lines":
        from radchain.geant4.g4_env_spectrum import line_table
        return float(sum(w for en, w in line_table() if en >= e))
    if source == "gamma-flat":
        lo, hi = FLAT_KEV
        return float(np.clip((hi - max(e, lo)) / (hi - lo), 0.0, 1.0))
    if source == "gamma-scatter":
        lo, hi = SCATTER_KEV
        if e >= hi:
            return 0.0
        return float(np.log(hi / max(e, lo)) / np.log(hi / lo))
    raise ValueError("not a gamma source: {!r}".format(source))
