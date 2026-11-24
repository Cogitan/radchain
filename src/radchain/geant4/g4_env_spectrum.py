"""g4_env_spectrum.py — deposited-energy spectrum from the ENVIRONMENT, measured.

WHY
---
`qubit_footprint.py` passes Google's footprint only if deposits above 180 keV
fall off as steeply as E^-3, and fails at E^-2. That slope was swept because we had no spectrum. The only
spectra on disk (`~/rad/l3/spec_gamma1.json`, `spec_gamma26.json`) are
monoenergetic gammas fired straight down, which is a benchmark, not an
environment. This measures the real thing.

SOURCES
-------
gamma-lines   isotropic flux of the primordial gamma lines (K-40, U-238 chain,
              Th-232 chain), weighted by UNSCEAR 2000 world-average specific
              activity (K-40 420, U-238 33, Th-232 45 Bq/kg) times emission
              probability per decay. Unscattered lines. What Google's fridge,
              lab and building actually emit is NOT known; this is the textbook
              environment, stated as such.
gamma-flat    isotropic flux, energy uniform on 0.2 to 3 MeV. A bracket for the
              scattered continuum: if the answer depends on which of these two
              is used, the spectrum is not pinned by physics we have.
muon          4 GeV mu-, sea-level zenith distribution I ~ cos^2, so flux
              through a horizontal plane ~ cos^3. Uniform over the die.

Isotropic flux is generated the standard way: a start point uniform on a sphere
enclosing the die, direction cosine-weighted inward. Most primaries miss; the
miss rate is part of the measured cost.

Runs in WSL (see g4cmp/README.md). One G4RunManager per process.
"""
from __future__ import annotations

import json
import math
import random
import time

#: (energy keV, emission probability per decay), per chain. NNDC values.
LINES = {
    "K-40": (420.0, [(1460.8, 0.1066)]),
    "U-238": (33.0, [(295.2, 0.184), (351.9, 0.356), (609.3, 0.455),
                     (1120.3, 0.149), (1764.5, 0.153)]),
    "Th-232": (45.0, [(238.6, 0.436), (583.2, 0.305), (911.2, 0.258),
                      (969.0, 0.158), (2614.5, 0.359)]),
}


def line_table():
    """Flattened (energy_keV, weight) with weights normalised to 1."""
    rows = [(e, act * p) for act, lines in LINES.values() for e, p in lines]
    tot = sum(w for _, w in rows)
    return [(e, w / tot) for e, w in rows]


def run(source: str, n_primaries: int, thickness_mm: float,
        xy_mm: float = 10.0, seed: int = 1,
        material: str = "G4_Si", map_bin_mm: float = 0.0,
        map_min_keV: float = 180.0) -> dict:
    from geant4_pybind import (G4RunManagerFactory, G4RunManagerType,
                               FTFP_BERT, G4UImanager, G4ParticleGun,
                               G4ParticleTable, G4ThreeVector, mm, keV, MeV,
                               G4VUserPrimaryGeneratorAction,
                               G4VUserActionInitialization,
                               G4UserSteppingAction, G4UserEventAction)
    import g4_l2_bench as b

    Slab, _, _, _ = b._g4_classes()
    rng = random.Random(seed)
    table = line_table()
    cum, acc = [], 0.0
    for e, w in table:
        acc += w
        cum.append((acc, e))

    half_diag = math.sqrt(2 * (xy_mm / 2) ** 2 + (thickness_mm / 2) ** 2)
    R = half_diag * 1.05

    class Gun(G4VUserPrimaryGeneratorAction):
        def __init__(self):
            super().__init__()
            self.gun = G4ParticleGun(1)
            name = "mu-" if source == "muon" else "gamma"
            self.gun.SetParticleDefinition(
                G4ParticleTable.GetParticleTable().FindParticle(name))

        def _isotropic(self):
            # start uniform on sphere, direction cosine-weighted inward
            z = rng.uniform(-1, 1)
            ph = rng.uniform(0, 2 * math.pi)
            s = math.sqrt(1 - z * z)
            n = (s * math.cos(ph), s * math.sin(ph), z)      # outward normal
            pos = tuple(R * c for c in n)
            ct = math.sqrt(rng.random())                     # cosine law
            st = math.sqrt(1 - ct * ct)
            a = rng.uniform(0, 2 * math.pi)
            # orthonormal basis around -n
            w = (-n[0], -n[1], -n[2])
            t = (1.0, 0.0, 0.0) if abs(w[0]) < 0.9 else (0.0, 1.0, 0.0)
            u = (w[1] * t[2] - w[2] * t[1], w[2] * t[0] - w[0] * t[2],
                 w[0] * t[1] - w[1] * t[0])
            un = math.sqrt(sum(c * c for c in u))
            u = tuple(c / un for c in u)
            v = (w[1] * u[2] - w[2] * u[1], w[2] * u[0] - w[0] * u[2],
                 w[0] * u[1] - w[1] * u[0])
            d = tuple(ct * w[i] + st * (math.cos(a) * u[i] + math.sin(a) * v[i])
                      for i in range(3))
            return pos, d

        def _muon(self):
            # flux through horizontal plane ~ cos^3: cos(theta) = U^(1/4)
            ct = rng.random() ** 0.25
            st = math.sqrt(1 - ct * ct)
            a = rng.uniform(0, 2 * math.pi)
            d = (st * math.cos(a), st * math.sin(a), -ct)
            x = rng.uniform(-xy_mm / 2, xy_mm / 2)
            y = rng.uniform(-xy_mm / 2, xy_mm / 2)
            back = 20.0
            pos = (x - back * d[0], y - back * d[1], -back * d[2])
            return pos, d

        def GeneratePrimaries(self, event):
            if source == "muon":
                pos, d = self._muon()
                self.gun.SetParticleEnergy(4000.0 * MeV)
            else:
                pos, d = self._isotropic()
                if source == "gamma-lines":
                    r = rng.random()
                    e = next(e for c, e in cum if r <= c)
                elif source == "gamma-scatter":
                    # Degraded/scattered field: dN/dE ~ 1/E from 30 keV to
                    # 2.6 MeV. Buildings scatter primordial lines down into a
                    # continuum; neither the lines nor a 0.2-3 MeV flat source
                    # has that low-energy population.
                    lo, hi = 30.0, 2600.0
                    e = lo * (hi / lo) ** rng.random()
                else:
                    e = rng.uniform(200.0, 3000.0)
                self.gun.SetParticleEnergy(e * keV)
            self.gun.SetParticlePosition(G4ThreeVector(*[c * mm for c in pos]))
            self.gun.SetParticleMomentumDirection(G4ThreeVector(*d))
            self.gun.GeneratePrimaryVertex(event)

    deposits: list = []
    maps: list = []    # [index into deposits, [i, j, keV], ...] for E >= map_min_keV

    class Scorer(G4UserSteppingAction):
        def __init__(self):
            super().__init__()
            self.e = 0.0
            self.m = {}

        def UserSteppingAction(self, step):
            vol = step.GetPreStepPoint().GetTouchable().GetVolume()
            if vol is not None and vol.GetName() == "Die":
                de = step.GetTotalEnergyDeposit() / keV
                self.e += de
                if map_bin_mm > 0 and de > 0.0:
                    # spread along the step (see g4_cryostat_spectrum._spread)
                    from g4_cryostat_spectrum import _spread
                    q = step.GetPostStepPoint().GetPosition()
                    q0 = step.GetPreStepPoint().GetPosition()
                    _spread(self.m, q0.x / mm, q0.y / mm, q.x / mm, q.y / mm,
                            de, map_bin_mm)

    class Ev(G4UserEventAction):
        def __init__(self, sc):
            super().__init__()
            self.sc = sc

        def BeginOfEventAction(self, event):
            self.sc.e = 0.0
            self.sc.m = {}

        def EndOfEventAction(self, event):
            if self.sc.e > 0.0:
                if map_bin_mm > 0 and self.sc.e >= map_min_keV:
                    maps.append([len(deposits)] + [
                        [i, j, round(v, 2)] for (i, j), v in self.sc.m.items()])
                deposits.append(round(self.sc.e, 3))

    rm = G4RunManagerFactory.CreateRunManager(G4RunManagerType.Serial)
    rm.SetUserInitialization(Slab(xy_mm, thickness_mm, material))
    rm.SetUserInitialization(FTFP_BERT())

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
    return {"source": source, "thickness_mm": thickness_mm, "xy_mm": xy_mm,
            "material": material,
            "n_primaries": n_primaries, "seed": seed, "seconds": sec,
            "deposits_keV": deposits,
            "map_bin_mm": map_bin_mm, "map_min_keV": map_min_keV,
            "maps": maps}


def tail_slope(deposits, e_min: float = 180.0, e_max: float = 5000.0):
    """MLE of gamma for dN/dE ~ E^-gamma above e_min (untruncated form), and n."""
    v = [e for e in deposits if e_min <= e <= e_max]
    if len(v) < 5:
        return float("nan"), len(v)
    s = sum(math.log(e / e_min) for e in v)
    return 1.0 + len(v) / s, len(v)


if __name__ == "__main__":
    import sys
    src, n, t, seed, out = (sys.argv[1], int(sys.argv[2]), float(sys.argv[3]),
                            int(sys.argv[4]), sys.argv[5])
    xy = float(sys.argv[6]) if len(sys.argv) > 6 else 10.0
    mat = sys.argv[7] if len(sys.argv) > 7 else "G4_Si"
    mb = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0
    r = run(src, n, t, seed=seed, xy_mm=xy, material=mat, map_bin_mm=mb)
    json.dump(r, open(out, "w"))
    g, k = tail_slope(r["deposits_keV"])
    print(f"DONE {src} t={t} n={n} {r['seconds']:.1f}s "
          f"{r['seconds'] / n * 1e6:.1f} us/primary  interacting={len(r['deposits_keV'])} "
          f">=180keV={k} slope={g:.2f}", flush=True)
