"""g4_kid_stack.py: reproduce the KID muon-tagger coincidence (arXiv:2512.10679).

Second experimental anchor for the RAD Geant4 chain: measured top-bottom
coincidence 0.192 +/- 0.009 /s (muon-induced), their Geant4 0.195 +/- 0.012.

GEOMETRY (+z up, mm), from the paper's full text, unknowns bracketed by flags:
  three Si wafers, 45 x 45 x 0.525 (20.25 cm2), stacked T / C / B with `gap`
  between wafers (paper: "approximately 4.5 mm spacing between each layer")
  centre wafer in a Cu box. The paper's "5 mm-thick copper box" cannot be a
  5 mm lid: the wafers are only ~4.5 mm apart. Read as 5 mm SIDE walls, with
  lid and floor plates `lid` mm thick inside the gaps (bracketed 0.5-2 mm),
  0.5 mm clear of every wafer. An assumption, stated as one.
  top and bottom wafers in open Cu holders: square frames `frame` mm thick and
  4 mm wide around the wafer edge (thickness not given; bracketed)
  pot of three nested box shells: Al (inner), Cu, Cryophy (outer, modelled as
  Ni80Fe20), each `pot_wall` mm (not given; bracketed), on a Cu MXC plate
  fridge shells and plates as g4_cryostat_spectrum (muon runs, --cryostat)
SOURCES: muon (cos^2 zenith, 4 GeV, horizontal disk covering the stack plus
  margin; rate = P x flux x pi R^2) and gamma-lines / gamma-scatter
  (isotropic sphere; rate = P x flux x pi R^2).
OUTPUT: per event with any wafer >= min_keV: [E_T, E_C, E_B] keV.
"""
from __future__ import annotations

import json
import math
import random
import time

from g4_env_spectrum import line_table

W = 45.0
T_W = 0.525


def run(source: str, n: int, gap: float = 4.5, frame: float = 3.0,
        lid: float = 1.0,
        pot_wall: float = 1.0, seed: int = 1, cryostat: bool = False,
        r_muon_mm: float = 90.0, muon_mev: float = 4000.0,
        min_keV: float = 1.0, check_overlaps: bool = False) -> dict:
    import geant4_pybind as g4
    from geant4_pybind import (G4VUserDetectorConstruction, G4NistManager,
                               G4Box, G4SubtractionSolid, G4LogicalVolume,
                               G4PVPlacement, G4RunManagerFactory,
                               G4RunManagerType, G4UImanager, G4ParticleGun,
                               G4ParticleTable, G4ThreeVector, G4Material,
                               mm, keV, MeV, g, cm3,
                               G4VUserPrimaryGeneratorAction,
                               G4VUserActionInitialization,
                               G4UserSteppingAction, G4UserEventAction)

    # Keep every geometry object alive for the whole run. geant4_pybind lets
    # Python garbage-collect a solid or volume that Geant4 still points to;
    # rebinding names in a loop (five sample boxes) freed earlier boxes and
    # produced random segfaults mid-run (2026-10-16, XQ50/XQ100). Geant4 owns
    # nothing created here, so this list is what keeps them.
    _keep = []

    def _kept(cls):
        def make(*a, **kw):
            o = cls(*a, **kw)
            _keep.append(o)
            return o
        return make

    G4Box = _kept(G4Box)
    G4SubtractionSolid = _kept(G4SubtractionSolid)
    G4LogicalVolume = _kept(G4LogicalVolume)
    G4PVPlacement = _kept(G4PVPlacement)
    rng = random.Random(seed)
    table = line_table()
    cum, acc = [], 0.0
    for e, w in table:
        acc += w
        cum.append((acc, e))
    pitch = T_W + gap
    zs = {"T": pitch, "C": 0.0, "B": -pitch}
    box_in_h = W / 2 + 1.0
    box_in_z = T_W / 2 + 0.5
    if lid + 1.0 > gap:
        raise ValueError("lid plus clearances do not fit in the gap")
    pot_in_h = W / 2 + 4.0 + frame + 8.0
    pot_in_z = pitch + T_W / 2 + frame + 8.0
    R_SPHERE = math.sqrt(2 * (pot_in_h + 3 * pot_wall + 20) ** 2 +
                         (pot_in_z + 3 * pot_wall + 20) ** 2) * 1.02

    # World sized from the muon start distance. A primary starting OUTSIDE
    # the world is dropped silently, and in the 5-box + fridge geometry that
    # path segfaulted non-deterministically (bisected 2026-10-16: a 1500 mm
    # world ran clean 3/3 where 800 mm crashed 5/5). Never let a start fall
    # outside: the world half-size must exceed r_muon + back-off + margin.
    WORLD_H = max(800.0, r_muon_mm + 700.0 + 100.0)
    class Det(G4VUserDetectorConstruction):
        def Construct(self):
            nist = G4NistManager.Instance()
            vac = nist.FindOrBuildMaterial("G4_Galactic")
            si = nist.FindOrBuildMaterial("G4_Si")
            cu = nist.FindOrBuildMaterial("G4_Cu")
            al = nist.FindOrBuildMaterial("G4_Al")
            permalloy = G4Material("Cryophy", 8.7 * g / cm3, 2)
            permalloy.AddElement(nist.FindOrBuildElement("Ni"), 0.8)
            permalloy.AddElement(nist.FindOrBuildElement("Fe"), 0.2)
            self.keep = [permalloy]
            wl = G4LogicalVolume(G4Box("World", WORLD_H * mm, WORLD_H * mm,
                                       WORLD_H * mm),
                                 vac, "World")
            wp = G4PVPlacement(None, G4ThreeVector(), wl, "World", None,
                               False, 0)

            def shell(name, mat, hx, hz, t, z0=0.0):
                o = G4Box(name + "O", (hx + t) * mm, (hx + t) * mm, (hz + t) * mm)
                i_ = G4Box(name + "I", hx * mm, hx * mm, hz * mm)
                lv = G4LogicalVolume(G4SubtractionSolid(name, o, i_), mat, name)
                G4PVPlacement(None, G4ThreeVector(0, 0, z0 * mm), lv, name, wl,
                              False, 0)

            for name, z in zs.items():
                lv = G4LogicalVolume(G4Box(name, W / 2 * mm, W / 2 * mm,
                                           T_W / 2 * mm), si, name)
                G4PVPlacement(None, G4ThreeVector(0, 0, z * mm), lv, name, wl,
                              False, 0)
            # side walls 5 mm, lid and floor `lid` mm
            o = G4Box("CuBoxO", (box_in_h + 5.0) * mm, (box_in_h + 5.0) * mm,
                      (box_in_z + lid) * mm)
            i_ = G4Box("CuBoxI", box_in_h * mm, box_in_h * mm, box_in_z * mm)
            lvb = G4LogicalVolume(G4SubtractionSolid("CuBox", o, i_), cu,
                                  "CuBox")
            G4PVPlacement(None, G4ThreeVector(), lvb, "CuBox", wl, False, 0)
            for name in ("T", "B"):        # open holders: square frames
                z = zs[name]
                o = G4Box("F" + name + "O", (W / 2 + 4.0) * mm,
                          (W / 2 + 4.0) * mm, frame / 2 * mm)
                i_ = G4Box("F" + name + "I", (W / 2 - 0.0) * mm,
                           (W / 2 - 0.0) * mm, (frame / 2 + 1) * mm)
                lv = G4LogicalVolume(G4SubtractionSolid("Holder" + name, o, i_),
                                     cu, "Holder" + name)
                zf = z + (T_W / 2 + frame / 2) * (1 if name == "T" else -1)
                G4PVPlacement(None, G4ThreeVector(0, 0, zf * mm), lv,
                              "Holder" + name, wl, False, 0)
            h, zz = pot_in_h, pot_in_z
            for name, mat in (("PotAl", al), ("PotCu", cu),
                              ("PotCryo", permalloy)):
                shell(name, mat, h, zz, pot_wall)
                h += pot_wall
                zz += pot_wall
            # MXC plate under the pot
            mxc = G4LogicalVolume(G4Box("MXC", 150 * mm, 150 * mm, 4 * mm),
                                  cu, "MXC")
            G4PVPlacement(None, G4ThreeVector(0, 0, -(zz + 4.5) * mm), mxc,
                          "MXC", wl, False, 0)
            if cryostat:
                from g4_cryostat_spectrum import SHELLS, PLATES, PLATE_GAP_MM
                ox = 160.0 - 8.0
                oz = 80.0 - 6.0
                for name, mat, hxy, hz, w in SHELLS:
                    shell(name, nist.FindOrBuildMaterial(mat), hxy + ox,
                          hz + oz, w)
                ztop = SHELLS[-1][3] + oz + SHELLS[-1][4] + 5.0
                for name, mat, w in PLATES:
                    lv = G4LogicalVolume(G4Box(name, 220 * mm, 220 * mm,
                                               w / 2 * mm),
                                         nist.FindOrBuildMaterial(mat), name)
                    G4PVPlacement(None, G4ThreeVector(0, 0, (ztop + w / 2) * mm),
                                  lv, name, wl, False, 0)
                    ztop += w + PLATE_GAP_MM
            return wp

    def _iso():
        zz = rng.uniform(-1, 1)
        ph = rng.uniform(0, 2 * math.pi)
        s = math.sqrt(1 - zz * zz)
        nv = (s * math.cos(ph), s * math.sin(ph), zz)
        pos = tuple(R_SPHERE * c for c in nv)
        ct = math.sqrt(rng.random())
        st = math.sqrt(1 - ct * ct)
        a = rng.uniform(0, 2 * math.pi)
        w = (-nv[0], -nv[1], -nv[2])
        t = (1.0, 0.0, 0.0) if abs(w[0]) < 0.9 else (0.0, 1.0, 0.0)
        u = (w[1] * t[2] - w[2] * t[1], w[2] * t[0] - w[0] * t[2],
             w[0] * t[1] - w[1] * t[0])
        un = math.sqrt(sum(c * c for c in u))
        u = tuple(c / un for c in u)
        v = (w[1] * u[2] - w[2] * u[1], w[2] * u[0] - w[0] * u[2],
             w[0] * u[1] - w[1] * u[0])
        return pos, tuple(ct * w[i] + st * (math.cos(a) * u[i] +
                                           math.sin(a) * v[i]) for i in range(3))

    def _mu():
        ct = rng.random() ** 0.25
        st = math.sqrt(1 - ct * ct)
        a = rng.uniform(0, 2 * math.pi)
        d = (st * math.cos(a), st * math.sin(a), -ct)
        r = r_muon_mm * math.sqrt(rng.random())
        b = rng.uniform(0, 2 * math.pi)
        x, y = r * math.cos(b), r * math.sin(b)
        return (x - 700 * d[0], y - 700 * d[1], -700 * d[2]), d

    class Gun(G4VUserPrimaryGeneratorAction):
        def __init__(self):
            super().__init__()
            self.gun = G4ParticleGun(1)
            self.gun.SetParticleDefinition(G4ParticleTable.GetParticleTable()
                                           .FindParticle("mu-" if source == "muon"
                                                         else "gamma"))

        def GeneratePrimaries(self, ev):
            if source == "muon":
                pos, d = _mu()
                self.gun.SetParticleEnergy(muon_mev * MeV)
            else:
                pos, d = _iso()
                if source == "gamma-lines":
                    r = rng.random()
                    e = next(e for c, e in cum if r <= c)
                else:
                    e = 30.0 * (2600.0 / 30.0) ** rng.random()
                self.gun.SetParticleEnergy(e * keV)
            self.gun.SetParticlePosition(G4ThreeVector(*[c * mm for c in pos]))
            self.gun.SetParticleMomentumDirection(G4ThreeVector(*d))
            self.gun.GeneratePrimaryVertex(ev)

    idx = {"T": 0, "C": 1, "B": 2}
    events: list = []

    class Sc(G4UserSteppingAction):
        def __init__(self):
            super().__init__()
            self.e = [0.0, 0.0, 0.0]

        def UserSteppingAction(self, step):
            v = step.GetPreStepPoint().GetTouchable().GetVolume()
            if v is not None:
                i = idx.get(v.GetName())
                if i is not None:
                    self.e[i] += step.GetTotalEnergyDeposit() / keV

    class Ev(G4UserEventAction):
        def __init__(self, sc):
            super().__init__()
            self.sc = sc

        def BeginOfEventAction(self, e):
            self.sc.e = [0.0, 0.0, 0.0]

        def EndOfEventAction(self, e):
            if max(self.sc.e) >= min_keV:
                events.append([round(v, 1) for v in self.sc.e])

    if cryostat and source != "muon":
        raise ValueError("cryostat is for muon runs only")
    rm = G4RunManagerFactory.CreateRunManager(G4RunManagerType.Serial)
    rm.SetUserInitialization(Det())
    rm.SetUserInitialization(g4.FTFP_BERT())

    class Act(G4VUserActionInitialization):
        def Build(self):
            self.SetUserAction(Gun())
            sc = Sc()
            self.SetUserAction(sc)
            self.SetUserAction(Ev(sc))

    rm.SetUserInitialization(Act())
    rm.Initialize()
    ui = G4UImanager.GetUIpointer()
    for c in ("/run/verbose 0", "/event/verbose 0", "/tracking/verbose 0",
              "/run/printProgress 0", f"/random/setSeeds {seed} {seed + 7}"):
        ui.ApplyCommand(c)
    if check_overlaps:
        ui.ApplyCommand("/geometry/test/run")
    t0 = time.perf_counter()
    rm.BeamOn(n)
    sec = time.perf_counter() - t0
    area = math.pi * ((r_muon_mm if source == "muon" else R_SPHERE) / 10) ** 2
    return {"source": source, "n_primaries": n, "seed": seed, "gap_mm": gap,
            "frame_mm": frame, "lid_mm": lid, "pot_wall_mm": pot_wall,
            "cryostat": cryostat,
            "r_muon_mm": r_muon_mm, "gen_area_cm2": area, "min_keV": min_keV,
            "seconds": sec, "events": events}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("n", type=int)
    p.add_argument("out")
    p.add_argument("--gap", type=float, default=4.5)
    p.add_argument("--frame", type=float, default=3.0)
    p.add_argument("--lid", type=float, default=1.0)
    p.add_argument("--pot-wall", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cryostat", action="store_true")
    p.add_argument("--r-muon", type=float, default=90.0)
    p.add_argument("--check-overlaps", action="store_true")
    a = p.parse_args()
    r = run(a.source, a.n, gap=a.gap, frame=a.frame, lid=a.lid,
            pot_wall=a.pot_wall,
            seed=a.seed, cryostat=a.cryostat, r_muon_mm=a.r_muon,
            check_overlaps=a.check_overlaps)
    json.dump(r, open(a.out, "w"))
    ev = r["events"]
    tb = sum(1 for e in ev if e[0] >= 1 and e[2] >= 1)
    print(f"DONE {a.source} n={a.n} {r['seconds']:.1f}s events={len(ev)} T&B={tb}",
          flush=True)
