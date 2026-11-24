"""g4_species_spectrum.py — deposited energy BY PARTICLE SPECIES, for attribution.

WHY
---
Pinckney et al. (arXiv:2603.13460 v2) attribute the ~1/hour correlated error
floor to deposits above 1 MeV and describe that part of the spectrum as
"primarily hadronic", citing Fowler et al. (arXiv:2404.10866), who report on
>2 MeV deposits at Boulder, 1650 m. This measures the species split for a
Willow-like die at sea level, flat and vertical.

`g4_env_spectrum.py` already covers environmental gammas (isotropic, so the
die orientation does not matter for them) and a 4 GeV top-face muon. This adds
the species and geometry that one could not represent:

SOURCES
-------
muon      mono-energetic mu- (energy is an argument; run at 1 and 4 GeV as a
          bracket, since a minimum-ionising track in sub-mm Si barely depends
          on it). Zenith law I(theta) ~ cos^2 per solid angle (PDG cosmic-ray
          review, Sec. 30.3.1), downward-going only.
neutron   JESD89A sea-level (New York City) spectrum, 1 MeV to 10 GeV, as the
          Gordon et al. (2004) fit in `jesd89_flux`. Zenith law cos^n with n an
          argument (2 by default; 0 = isotropic over the upper hemisphere, a
          bracket). Below ~1.35 MeV an elastic Si recoil cannot reach 180 keV
          (max recoil fraction 4A/(A+1)^2 = 0.133), so 1 MeV is a safe floor.
proton    SAME energy and angular sampling as the neutron source. The sea-level
          proton spectrum is NOT sourced here; this gives the per-primary
          response only, and the analysis applies a proton/neutron ratio as a
          stated, bracketed assumption. Never quote a proton RATE from this
          file alone.

GEOMETRY AND NORMALISATION
--------------------------
Directional sources are generated the standard way for a flux I(Omega): sample
a direction from I, then a start point uniform on a disk of radius R normal to
that direction, centred on the die, backed off by R. Then, per primary,

    rate = pi R^2 * Integral I dOmega * (events / N).

`orient="vertical"` mounts the die with its normal horizontal (sky zenith
along die +x); `"flat"` has the normal vertical. That is the axis on which Loer
et al. (arXiv:2403.01032, Table 6) see a 5x difference above 1 MeV.

Each recorded event carries its energy and the energy-weighted centroid in the
die plane, so the footprint chain can be applied later with real strike
positions. Events below `min_keV` are counted but not stored.

Runs in WSL against ~/rad/.venv (g4cmp/README.md). One G4RunManager per process.
"""
from __future__ import annotations

import bisect
import json
import math
import random
import time

#: Sea-level horizontal-detector muon flux, PDG: "I ~ 1 cm^-2 min^-1".
MUON_J_HORIZONTAL_PER_CM2_S = 1.0 / 60.0


def jesd89_flux(e_mev: float) -> float:
    """JESD89A / Gordon et al. 2004 sea-level neutron differential flux.

    n cm^-2 s^-1 MeV^-1 at New York City. Checked numerically against the
    published integrals (13 /cm^2/h above 10 MeV) by `check_jesd89()`; if that
    check fails the fit is mis-remembered and nothing downstream is usable.
    """
    L = math.log(e_mev)
    return (1.006e-6 * math.exp(-0.35 * L * L + 2.1451 * L)
            + 1.011e-3 * math.exp(-0.4106 * L * L - 0.667 * L))


def _log_grid(lo: float, hi: float, n: int = 4000):
    return [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]


def jesd89_integral_per_cm2_h(e_lo: float, e_hi: float = 1.0e4) -> float:
    g = _log_grid(e_lo, e_hi)
    s = 0.0
    for a, b in zip(g, g[1:]):
        s += 0.5 * (jesd89_flux(a) + jesd89_flux(b)) * (b - a)
    return s * 3600.0


def check_jesd89() -> dict:
    return {"above_1MeV_per_cm2_h": jesd89_integral_per_cm2_h(1.0),
            "above_10MeV_per_cm2_h": jesd89_integral_per_cm2_h(10.0),
            "published_above_10MeV": 13.0}


def _energy_sampler(e_lo: float = 1.0, e_hi: float = 1.0e4):
    g = _log_grid(e_lo, e_hi)
    cum, acc = [0.0], 0.0
    for a, b in zip(g, g[1:]):
        acc += 0.5 * (jesd89_flux(a) + jesd89_flux(b)) * (b - a)
        cum.append(acc)
    cum = [c / acc for c in cum]

    def sample(u: float) -> float:
        i = max(1, bisect.bisect_left(cum, u))
        f = (u - cum[i - 1]) / max(cum[i] - cum[i - 1], 1e-300)
        return g[i - 1] * (g[i] / g[i - 1]) ** f
    return sample


def angular_factor(n: float) -> float:
    """Integral I dOmega over the upper hemisphere / horizontal flux J_h,
    for I ~ cos^n theta: (n + 2) / (n + 1)."""
    return (n + 2.0) / (n + 1.0)


def run(source: str, n_primaries: int, thickness_mm: float,
        orient: str = "flat", xy_mm: float = 10.0, seed: int = 1,
        energy_mev: float = 4000.0, zenith_n: float = 2.0,
        min_keV: float = 50.0, physics: str = "QGSP_BIC_HP",
        material: str = "G4_Si") -> dict:
    import geant4_pybind as g4
    from geant4_pybind import (G4RunManagerFactory, G4RunManagerType,
                               G4UImanager, G4ParticleGun, G4ParticleTable,
                               G4ThreeVector, mm, keV, MeV,
                               G4VUserPrimaryGeneratorAction,
                               G4VUserActionInitialization,
                               G4UserSteppingAction, G4UserEventAction)
    import g4_l2_bench as b

    if source not in ("muon", "neutron", "proton"):
        raise ValueError(f"unknown source {source!r}")
    if orient not in ("flat", "vertical"):
        raise ValueError(f"unknown orient {orient!r}")
    n_ang = 2.0 if source == "muon" else zenith_n

    Slab, _, _, _ = b._g4_classes()
    rng = random.Random(seed)
    esample = _energy_sampler() if source != "muon" else None
    R = 1.05 * math.sqrt(2 * (xy_mm / 2) ** 2 + (thickness_mm / 2) ** 2)

    def sky_to_die(v):
        # flat: sky zenith = die +z.  vertical: sky zenith = die +x.
        x, y, z = v
        return (x, y, z) if orient == "flat" else (z, y, -x)

    class Gun(G4VUserPrimaryGeneratorAction):
        def __init__(self):
            super().__init__()
            self.gun = G4ParticleGun(1)
            name = {"muon": "mu-", "neutron": "neutron",
                    "proton": "proton"}[source]
            self.gun.SetParticleDefinition(
                G4ParticleTable.GetParticleTable().FindParticle(name))

        def GeneratePrimaries(self, event):
            ct = rng.random() ** (1.0 / (n_ang + 1.0))   # I ~ cos^n per sr
            st = math.sqrt(max(0.0, 1.0 - ct * ct))
            a = rng.uniform(0.0, 2.0 * math.pi)
            d = sky_to_die((st * math.cos(a), st * math.sin(a), -ct))
            # orthonormal basis normal to d
            t = (1.0, 0.0, 0.0) if abs(d[0]) < 0.9 else (0.0, 1.0, 0.0)
            u = (d[1] * t[2] - d[2] * t[1], d[2] * t[0] - d[0] * t[2],
                 d[0] * t[1] - d[1] * t[0])
            un = math.sqrt(sum(c * c for c in u))
            u = tuple(c / un for c in u)
            v = (d[1] * u[2] - d[2] * u[1], d[2] * u[0] - d[0] * u[2],
                 d[0] * u[1] - d[1] * u[0])
            r = R * math.sqrt(rng.random())
            ph = rng.uniform(0.0, 2.0 * math.pi)
            p = tuple(r * (math.cos(ph) * u[i] + math.sin(ph) * v[i]) - R * d[i]
                      for i in range(3))
            e = energy_mev if source == "muon" else esample(rng.random())
            self.gun.SetParticleEnergy(e * MeV)
            self.gun.SetParticlePosition(G4ThreeVector(*[c * mm for c in p]))
            self.gun.SetParticleMomentumDirection(G4ThreeVector(*d))
            self.gun.GeneratePrimaryVertex(event)

    events: list = []
    counts = {"interacting": 0}

    class Scorer(G4UserSteppingAction):
        def __init__(self):
            super().__init__()
            self.e = self.ex = self.ey = 0.0

        def UserSteppingAction(self, step):
            vol = step.GetPreStepPoint().GetTouchable().GetVolume()
            if vol is not None and vol.GetName() == "Die":
                de = step.GetTotalEnergyDeposit() / keV
                if de > 0.0:
                    p = step.GetPostStepPoint().GetPosition()
                    self.e += de
                    self.ex += de * p.x / mm     # pybind: properties
                    self.ey += de * p.y / mm

    class Ev(G4UserEventAction):
        def __init__(self, sc):
            super().__init__()
            self.sc = sc

        def BeginOfEventAction(self, event):
            self.sc.e = self.sc.ex = self.sc.ey = 0.0

        def EndOfEventAction(self, event):
            e = self.sc.e
            if e > 0.0:
                counts["interacting"] += 1
                if e >= min_keV:
                    events.append([round(e, 3), round(self.sc.ex / e, 3),
                                   round(self.sc.ey / e, 3)])

    rm = G4RunManagerFactory.CreateRunManager(G4RunManagerType.Serial)
    rm.SetUserInitialization(Slab(xy_mm, thickness_mm, material))
    rm.SetUserInitialization(getattr(g4, physics)())

    class Actions(G4VUserActionInitialization):
        def Build(self):
            self.SetUserAction(Gun())
            sc = Scorer()
            self.SetUserAction(sc)
            self.SetUserAction(Ev(sc))

    rm.SetUserInitialization(Actions())
    rm.Initialize()
    ui = G4UImanager.GetUIpointer()
    for cmd in ("/run/verbose 0", "/event/verbose 0", "/tracking/verbose 0",
                "/run/printProgress 0", f"/random/setSeeds {seed} {seed + 7}"):
        ui.ApplyCommand(cmd)

    t0 = time.perf_counter()
    rm.BeamOn(n_primaries)
    sec = time.perf_counter() - t0

    if source == "muon":
        j_h = MUON_J_HORIZONTAL_PER_CM2_S
        j_note = "PDG horizontal-detector muon flux, 1 /cm^2/min"
    else:
        j_h = jesd89_integral_per_cm2_h(1.0) / 3600.0
        j_note = ("JESD89A neutron flux above 1 MeV, treated as horizontal-"
                  "detector flux; for protons this is a PER-NEUTRON response "
                  "and must be scaled by an assumed p/n ratio")
    omega_int = j_h * angular_factor(n_ang)          # /cm^2/s
    gen_area_cm2 = math.pi * (R / 10.0) ** 2
    return {"source": source, "orient": orient, "thickness_mm": thickness_mm,
            "xy_mm": xy_mm, "material": material, "physics": physics,
            "energy_mev": energy_mev if source == "muon" else "jesd89",
            "zenith_n": n_ang, "n_primaries": n_primaries, "seed": seed,
            "seconds": sec, "min_keV": min_keV,
            "interacting": counts["interacting"],
            "rate_per_primary_s": gen_area_cm2 * omega_int,
            "rate_note": j_note, "gen_area_cm2": gen_area_cm2,
            "events": events}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("n", type=int)
    ap.add_argument("thickness_mm", type=float)
    ap.add_argument("orient")
    ap.add_argument("seed", type=int)
    ap.add_argument("out")
    ap.add_argument("--energy-mev", type=float, default=4000.0)
    ap.add_argument("--zenith-n", type=float, default=2.0)
    ap.add_argument("--xy-mm", type=float, default=10.0)
    ap.add_argument("--physics", default="QGSP_BIC_HP")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        print("JESD89 CHECK", check_jesd89(), flush=True)
    r = run(a.source, a.n, a.thickness_mm, orient=a.orient, seed=a.seed,
            energy_mev=a.energy_mev, zenith_n=a.zenith_n, xy_mm=a.xy_mm,
            physics=a.physics)
    json.dump(r, open(a.out, "w"))
    ev = r["events"]
    k1 = sum(1 for e in ev if e[0] >= 1000.0)
    k018 = sum(1 for e in ev if e[0] >= 180.0)
    rp = r["rate_per_primary_s"] * 3600.0
    print(f"DONE {a.source} {a.orient} t={a.thickness_mm} n={a.n} "
          f"{r['seconds']:.1f}s {r['seconds'] / a.n * 1e6:.1f} us/primary "
          f"interacting={r['interacting']} >=180keV={k018} >=1MeV={k1} "
          f"rate>=1MeV={rp * k1 / a.n:.3g}/h", flush=True)
