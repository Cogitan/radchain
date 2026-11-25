"""Phonon transport backends: where a strike's pair-breaking energy lands.

The chain needs one thing from phonon transport: for a strike at the origin,
where in the qubit film the deposited energy ends up breaking Cooper pairs.
Every downstream stage (`qubit_footprint`, `analysis/rad_pfail_physics.py`)
consumes that as a cloud of hit positions ``(hx, hy)`` normalised by
``n_phonons``, so that a strike depositing ``E`` keV gives a region the energy

    E_region = E * (hits inside region) / n_phonons

A backend is anything that produces that cloud. This module defines the
interface and ships one backend:

``g4cmp-kernel`` (default)
    Built from recorded G4CMP runs (``data/l3_kernel/runs.jsonl``, produced by
    ``g4cmp/l3_kernel_job.sh`` + ``score_kernel.py``). Each run is the radial
    histogram of quasiparticle births around a strike, for one substrate and
    back-side trap configuration. The backend pools every seed of the requested
    configuration and draws ``n_phonons`` outcomes: with probability equal to
    the fraction of primary phonon energy spent breaking pairs in radial bin
    ``b``, the outcome is a hit at a radius log-uniform inside ``b`` and a
    uniform azimuth; otherwise the phonon is lost (trapped, escaped, or
    downconverted). Nothing is fitted.

    The energy fraction is counted, not read from the per-bin energy column:
    each quasiparticle birth is charged the film gap ``DELTA_FILM_EV`` (Al,
    180 ueV), the minimum a broken pair costs per quasiparticle. The record's
    ``energy_eV_per_bin`` column sums to more than the primary energy, so it
    is not used here. Downstream stages fit or take an overall response
    coefficient, so this normalisation affects absolute energies only.

To add a backend, either call :func:`register_backend` or expose a factory
under the ``radchain.phonon_backends`` entry-point group. A backend is any
object with a ``name`` attribute and a method

    film_hits(*, n_phonons: int, seed: int = 0, **geometry) -> (hx, hy)

returning hit positions in mm for a strike at the origin of an unbounded die.

LIMITS OF THE G4CMP BACKEND, STATED UP FRONT
--------------------------------------------
- Only configurations that were actually simulated are available. There is no
  interpolation between grid points: ask for a point that is not on the grid
  and you get an error listing the nearest ones (or pass ``nearest=True``).
  Run G4CMP at your own geometry with ``g4cmp/l3_kernel_job.sh`` and add it
  with ``python -m radchain.ingest_kernel_runs``.
- The trap grid was simulated on a 4 mm die, so its kernels stop near 2 mm
  from the strike; the no-trap kernels come from an 8 mm die with a housing.
  See :class:`KernelConfig`. For large dies the far tail is missing.
- The scorer bins radii from 30 um to 8 mm and records births outside that
  range only as a count (``outside_binning``). On the simulated dies none can
  lie beyond 8 mm, so they are all within 30 um of the strike; they are kept
  as an inner bin, drawn uniformly over a 30 um disc. They are 0.04-3.7% of
  births in the no-trap kernels and up to ~18% in thin trapped configurations.
- Only the ``g_*`` (trap grid) and ``kt_*`` (stock) tag families are loaded.
  Runs with other tags are ignored. Runs pooled under one tag must share strike
  position and primary energy (checked). Die size, housing and film pattern are
  NOT recorded in a run, so a run you add must use the same settings as its
  family (see the sweep scripts), or a new tag family.
- Strikes are placed anywhere on the target die, but the stock kernels already
  include reflection at the simulated 8 mm die's edge; for strikes near the
  edge of a larger die the outer geometry is approximate.
- The record is axisymmetric about the strike: anisotropic phonon focusing in
  the crystal is averaged over azimuth.
- The kernel is measured for a strike at one depth and one primary phonon
  energy (``energy_eV`` in each record). A handful of runs vary depth and
  energy (tags ``s3_*``, ``s4_*``) if you need to check sensitivity.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from importlib import metadata
from typing import Callable, Protocol

import numpy as np

from radchain import _data

ENTRY_POINT_GROUP = "radchain.phonon_backends"
DEFAULT_BACKEND = "g4cmp-kernel"

#: Superconducting gap of the film the G4CMP example breaks pairs in (Al).
DELTA_FILM_EV = 180e-6


class PhononBackend(Protocol):
    name: str

    def film_hits(self, *, n_phonons: int, seed: int = 0,
                  **geometry) -> tuple[np.ndarray, np.ndarray]:
        ...


_REGISTRY: dict[str, Callable[[], PhononBackend]] = {}
_INSTANCES: dict[str, PhononBackend] = {}


def register_backend(name: str, factory: Callable[[], PhononBackend]) -> None:
    """Make a backend available to :func:`get_backend` under ``name``."""
    _REGISTRY[name] = factory
    _INSTANCES.pop(name, None)


def available_backends() -> list[str]:
    names = set(_REGISTRY)
    for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
        names.add(ep.name)
    return sorted(names)


def get_backend(name: str | PhononBackend | None = None) -> PhononBackend:
    """Return a backend instance by name (or pass an instance straight through)."""
    if name is None:
        name = DEFAULT_BACKEND
    if not isinstance(name, str):
        return name
    if name in _INSTANCES:
        return _INSTANCES[name]
    factory = _REGISTRY.get(name)
    if factory is None:
        for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
            if ep.name == name:
                factory = ep.load()
                break
    if factory is None:
        raise KeyError(f"no phonon backend {name!r}; available: "
                       f"{available_backends()}")
    inst = factory()
    _INSTANCES[name] = inst
    return inst


# ------------------------------------------------------------ G4CMP backend

#: Largest half-diagonal of a die in the record (8 mm stock die), mm.
_MAX_DIE_HALF_DIAGONAL_MM = 4.0 * math.sqrt(2.0)

_GRID_TAG = re.compile(
    r"^g_t(?P<t>[\d.]+)_p(?P<p>[\d.]+)_c(?P<c>[\d.]+)_a(?P<a>[\d.]+)_\d+$")
_NOTRAP_TAG = re.compile(r"^kt_t(?P<t>[\d.]+)_\d+$")


@dataclass(frozen=True)
class KernelConfig:
    """One simulated configuration: substrate thickness and back-side traps.

    Two families were simulated, and they differ in more than the traps:

    - ``trap_coverage > 0`` (tags ``g_*``): plain Al film on a 4 mm die, no
      copper housing, back-side trap islands at the given pitch, coverage and
      absorption. A 4 mm die means the kernel is truncated past ~2 mm from the
      strike: phonons reflect at the die edge instead of travelling on.
    - ``trap_coverage == 0`` (tags ``kt_*``): the stock G4CMP quasiparticle
      example, an 8 mm die with its resonator pattern and copper housing, no
      back-side traps. ``trap_pitch_mm`` and ``trap_absorption`` are 0.
    """
    thickness_mm: float
    trap_pitch_mm: float
    trap_coverage: float
    trap_absorption: float


@dataclass(frozen=True)
class Kernel:
    config: KernelConfig
    n_runs: int
    n_primaries: int
    bin_edges_mm: np.ndarray
    #: Fraction of primary phonon energy ending in a pair break, per radial bin.
    energy_fraction: np.ndarray

    @property
    def film_fraction(self) -> float:
        return float(self.energy_fraction.sum())


def _config_of(tag: str) -> KernelConfig | None:
    m = _GRID_TAG.match(tag)
    if m:
        return KernelConfig(float(m["t"]), float(m["p"]), float(m["c"]),
                            float(m["a"]))
    m = _NOTRAP_TAG.match(tag)
    if m:
        return KernelConfig(float(m["t"]), 0.0, 0.0, 0.0)
    return None


def load_kernels(path=None) -> dict[KernelConfig, Kernel]:
    """Pool the recorded G4CMP runs by configuration."""
    path = _data.RUNS if path is None else path
    acc: dict[KernelConfig, list[dict]] = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            cfg = _config_of(r["tag"])
            if cfg is not None:
                acc[cfg].append(r)
    out = {}
    for cfg, runs in acc.items():
        r0 = runs[0]
        edges = np.asarray(r0["bin_edges_mm"], dtype=float)
        # Births outside the binning go into an inner bin [0, edges[0]]: on the
        # simulated dies (4 mm and 8 mm) no birth can lie beyond edges[-1] =
        # 8 mm from a centred strike, so every one of them is within 30 um.
        if edges[-1] < _MAX_DIE_HALF_DIAGONAL_MM:
            raise ValueError(f"{r0['tag']}: binning stops inside the die; "
                             "outside_binning is not all inner")
        edges = np.concatenate([[0.0], edges])
        e_abs = np.zeros(len(edges) - 1)
        e_in = 0.0
        n_prim = 0
        for r in runs:
            if not np.allclose(r["bin_edges_mm"], edges[1:]):
                raise ValueError(f"{r['tag']}: bin edges differ within config")
            for key in ("x_mm", "y_mm", "z_mm", "energy_eV"):
                if not np.isclose(r[key], r0[key]):
                    raise ValueError(
                        f"{r['tag']}: {key} {r[key]} differs from {r0['tag']} "
                        f"({r0[key]}); runs pooled under one tag must share "
                        "strike position and primary energy")
            e_abs[0] += r.get("outside_binning", 0) * DELTA_FILM_EV
            e_abs[1:] += np.asarray(r["counts"], dtype=float) * DELTA_FILM_EV
            e_in += r["energy_eV"] * r["n_primaries"]
            n_prim += r["n_primaries"]
        out[cfg] = Kernel(cfg, len(runs), n_prim, edges, e_abs / e_in)
    return out


class G4CMPKernelBackend:
    """Phonon hits drawn from recorded G4CMP quasiparticle-birth kernels."""

    name = "g4cmp-kernel"

    def __init__(self, path=None):
        self.kernels = load_kernels(path)

    def configs(self) -> list[KernelConfig]:
        return sorted(self.kernels, key=lambda c: (c.thickness_mm, c.trap_coverage,
                                                   c.trap_pitch_mm,
                                                   c.trap_absorption))

    def kernel(self, *, thickness_mm: float, trap_pitch_mm: float = 0.0,
               trap_coverage: float = 0.0, trap_absorption: float = 0.0,
               nearest: bool = False) -> Kernel:
        want = KernelConfig(float(thickness_mm), float(trap_pitch_mm),
                            float(trap_coverage), float(trap_absorption))
        if want.trap_coverage == 0.0:
            want = KernelConfig(want.thickness_mm, 0.0, 0.0, 0.0)
        for cfg, k in self.kernels.items():
            if all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
                   for a, b in zip(_tuple(cfg), _tuple(want))):
                return k
        ranked = sorted(self.kernels, key=lambda c: _distance(c, want))
        if nearest:
            return self.kernels[ranked[0]]
        raise KeyError(
            f"no G4CMP record for {want}. Nearest simulated: "
            + "; ".join(str(c) for c in ranked[:3])
            + ". Pass nearest=True to use the closest, or simulate this "
              "configuration (see g4cmp/README.md).")

    def film_hits(self, *, n_phonons: int = 200_000, seed: int = 0,
                  thickness_mm: float, trap_pitch_mm: float = 0.0,
                  trap_coverage: float = 0.0, trap_absorption: float = 0.0,
                  nearest: bool = False) -> tuple[np.ndarray, np.ndarray]:
        k = self.kernel(thickness_mm=thickness_mm, trap_pitch_mm=trap_pitch_mm,
                        trap_coverage=trap_coverage,
                        trap_absorption=trap_absorption, nearest=nearest)
        return sample_hits(k, n_phonons, seed)


def sample_hits(kernel: Kernel, n_phonons: int,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n_phonons`` outcomes from a kernel; return the hit positions."""
    rng = np.random.default_rng(seed)
    p = kernel.energy_fraction
    lost = max(0.0, 1.0 - float(p.sum()))
    probs = np.append(p, lost)
    counts = rng.multinomial(n_phonons, probs / probs.sum())[:-1]
    lo_e = kernel.bin_edges_mm[:-1]
    hi_e = kernel.bin_edges_mm[1:]
    b = np.repeat(np.arange(len(counts)), counts)
    u = rng.random(b.size)
    r = np.empty(b.size)
    inner = lo_e[b] == 0.0
    # Inner bin: uniform over the disc. Others: log-uniform in radius.
    r[inner] = hi_e[b[inner]] * np.sqrt(u[inner])
    lo, hi = np.log(lo_e[b[~inner]]), np.log(hi_e[b[~inner]])
    r[~inner] = np.exp(lo + (hi - lo) * u[~inner])
    phi = rng.uniform(0.0, 2.0 * np.pi, b.size)
    return r * np.cos(phi), r * np.sin(phi)


def _tuple(c: KernelConfig) -> tuple:
    return (c.thickness_mm, c.trap_pitch_mm, c.trap_coverage, c.trap_absorption)


def _distance(c: KernelConfig, w: KernelConfig) -> float:
    def lg(x):
        return math.log(x) if x > 0 else -10.0
    return (abs(lg(c.thickness_mm) - lg(w.thickness_mm))
            + abs(lg(c.trap_pitch_mm) - lg(w.trap_pitch_mm))
            + abs(c.trap_coverage - w.trap_coverage) * 4
            + abs(c.trap_absorption - w.trap_absorption))


register_backend(G4CMPKernelBackend.name, G4CMPKernelBackend)
