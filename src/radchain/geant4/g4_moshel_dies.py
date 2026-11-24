"""g4_moshel_dies.py: per-die deposits in Moshel et al.'s two-die device.

WHY
---
Moshel et al. (arXiv:2505.16272, APL 2025) measured 10 of 352 MKID bursts
(2.8%) on both of two Si dies 0.2 mm apart in one copper box and attribute
them to one particle crossing both dies. Our paper's contribution (ii) says
cross-chip coincidences come from shared material, tested only in Wu's
geometry. Target and verdict rules were fixed before the runs.

GEOMETRY (+z up, millimetres, cavity floor at z = 0, cavity centre x = y = 0)
-------------------------------------------------------------------------
- two Si dies L x W x T side by side along x, edge gap `gap`, centred at
  y = Y_PAIR, bottoms at z = h_ped, each on four Cu pedestals 1 x 1 x h_ped
  at its corners (paper: "supported at its corners by four OFHC copper
  pedestals"; photo Fig. 1)
- a second Si chip 20.5 x 8.1 x T at y = Y_CHIP2 on its own pedestals
  (the other slot in Fig. 1(a)); passive
- PCB slab 0.5 mm on the floor with a 1 mm-margin cutout around each chip
  (FR4-like: SiO2 at 1.85 g/cm3; the Ag plating is not modelled)
- OFHC Cu box: cavity CAV_XY x CAV_XY x cav_h, walls 5 mm, floor 5 mm,
  lid 3 mm, built from six plates (no boolean solids)
- Cu MXC plate 50 x 50 x 8 mm under the box (as in g4_wu_modules)
- optional refrigerator shells and plates for muon runs (as g4_wu_modules)

SOURCES (identical conventions to g4_wu_modules, so rates compare)
-------------------------------------------------------------------
gamma-lines, gamma-scatter: isotropic from a sphere R_SPHERE_MM about the
cavity centre, cosine-weighted inward. rate = P x flux x pi R^2.
muon: 4 GeV mu-, cos^2 zenith, uniform on a horizontal disk of radius
r_muon through the die mid-plane, centred on the pair.
rate = P x flux (/cm2/s, horizontal plane) x pi r_muon^2.

OUTPUT
------
`events`: one row per primary that deposited >= min_keV in either die:
[E_L, E_R, prim_L, prim_R, hit_L, hit_R, hit_Box, hit_Ped, hit_PCB,
 hit_Plate, hit_Fridge, hit_Chip2] (keV to 0.1; hit_* = the PRIMARY track
entered that volume).

Runs in WSL (g4cmp/README.md). One G4RunManager per process.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
import random
import time

from g4_env_spectrum import line_table

R_SPHERE_MM = 45.0
CAV_XY = 36.0
WALL, FLOOR, LID = 5.0, 5.0, 3.0
PLATE = (50.0, 50.0, 8.0)
PED = 1.0                       # pedestal footprint
Y_PAIR, Y_CHIP2 = -6.6, 5.2     # photo, Fig. 1(a)
CHIP2 = (20.5, 8.1)
T_PCB = 0.5
SKIN = 1.0      # inner Cu skin of the box kept in the fine-cut region
COLUMNS = ["E_L", "E_R", "prim_L", "prim_R", "hit_L", "hit_R", "hit_Box",
           "hit_Ped", "hit_PCB", "hit_Plate", "hit_Fridge", "hit_Chip2"]


def layout(L, W, T, gap, h_ped):
    """{name: (centre, half sizes)} for dies, chip 2 and pedestals."""
    zc = h_ped + T / 2
    vols = {
        "DieL": ((-(gap / 2 + L / 2), Y_PAIR, zc), (L / 2, W / 2, T / 2)),
        "DieR": ((+(gap / 2 + L / 2), Y_PAIR, zc), (L / 2, W / 2, T / 2)),
        "Chip2": ((0.0, Y_CHIP2, zc), (CHIP2[0] / 2, CHIP2[1] / 2, T / 2)),
    }
    peds = []
    for name in ("DieL", "DieR", "Chip2"):
        (x, y, _), (hx, hy, _) = vols[name]
        for sx in (-1, 1):
            for sy in (-1, 1):
                peds.append((x + sx * (hx - PED / 2), y + sy * (hy - PED / 2),
                             h_ped / 2))
    return vols, peds


def run(source: str, n_primaries: int, L: float = 13.5, W: float = 10.0,
        T: float = 0.5, gap: float = 0.2, h_ped: float = 0.5,
        cav_h: float = 8.0, seed: int = 1, muon_mev: float = 4000.0,
        min_keV: float = 1.0, physics: str = "FTFP_BERT",
        cryostat: bool = False, r_muon_mm: float = 60.0,
        cut_mm: float | None = 0.001, cut_mode: str = "region",
        fridge_off: float = 20.0, check_overlaps: bool = False,
        dump_couples: bool = False, entry: bool = False) -> dict:
    """cut_mode 'region': cut_mm in the device region only, Geant4 default
    0.7 mm elsewhere. 'global': cut_mm everywhere (slow with --cryostat)."""
    import geant4_pybind as g4
    from geant4_pybind import (G4VUserDetectorConstruction, G4NistManager,
                               G4Box, G4LogicalVolume, G4PVPlacement,
                               G4RunManagerFactory, G4RunManagerType,
                               G4UImanager, G4ParticleGun, G4ParticleTable,
                               G4ThreeVector, mm, keV, MeV, g, cm3,
                               G4VUserPrimaryGeneratorAction,
                               G4VUserActionInitialization,
                               G4UserSteppingAction, G4UserEventAction,
                               G4Region)

    # Keep every geometry object alive (see g4_wu_modules: geant4_pybind lets
    # Python free volumes Geant4 still uses; random mid-run segfaults).
    _keep = []

    def _kept(cls):
        def make(*a, **kw):
            o = cls(*a, **kw)
            _keep.append(o)
            return o
        return make

    G4Box = _kept(G4Box)
    G4LogicalVolume = _kept(G4LogicalVolume)
    G4PVPlacement = _kept(G4PVPlacement)

    if cryostat and source != "muon":
        raise ValueError("cryostat geometry is for muon runs only")
    rng = random.Random(seed)
    table = line_table()
    cum, acc = [], 0.0
    for e, w in table:
        acc += w
        cum.append((acc, e))
    vols, peds = layout(L, W, T, gap, h_ped)
    z_mid = h_ped + T / 2
    hc = CAV_XY / 2
    # World from the muon start distance: a primary born outside the world
    # is dropped silently (and segfaulted once; g4_wu_modules 2026-10-16).
    WORLD_H = max(500.0, r_muon_mm + 400.0 + 100.0)

    fridge_names: set[str] = set()

    class Det(G4VUserDetectorConstruction):
        def Construct(self):
            nist = G4NistManager.Instance()
            vac = nist.FindOrBuildMaterial("G4_Galactic")
            si = nist.FindOrBuildMaterial("G4_Si")
            cu = nist.FindOrBuildMaterial("G4_Cu")
            pcb = nist.BuildMaterialWithNewDensity(
                "PCB_FR4ish", "G4_SILICON_DIOXIDE", 1.85 * g / cm3)
            world_l = G4LogicalVolume(
                G4Box("World", WORLD_H * mm, WORLD_H * mm, WORLD_H * mm),
                vac, "World")
            world_p = G4PVPlacement(None, G4ThreeVector(), world_l, "World",
                                    None, False, 0)

            fine_lvs = []

            def slab(name, mat, c, h, logical_name=None, fine=False):
                lv = G4LogicalVolume(
                    G4Box(name, h[0] * mm, h[1] * mm, h[2] * mm), mat,
                    logical_name or name)
                G4PVPlacement(None, G4ThreeVector(*[v * mm for v in c]), lv,
                              logical_name or name, world_l, False, 0)
                if fine:
                    fine_lvs.append(lv)

            # Box: floor, lid, two full-length x walls, two y walls between.
            # Each plate is split into an inner SKIN (facing the cavity,
            # in the fine-cut region) and the outer remainder.
            ob = hc + WALL
            sk = SKIN
            slab("BoxFloorS", cu, (0, 0, -sk / 2), (hc, hc, sk / 2), "Box",
                 fine=True)
            slab("BoxFloor", cu, (0, 0, -(FLOOR + sk) / 2),
                 (ob, ob, (FLOOR - sk) / 2), "Box")
            # floor ring beside the skin, same z as the skin
            for s in (-1, 1):
                slab("BoxFloorRx", cu, (0, s * (hc + WALL / 2), -sk / 2),
                     (ob, WALL / 2, sk / 2), "Box")
                slab("BoxFloorRy", cu, (s * (hc + WALL / 2), 0, -sk / 2),
                     (WALL / 2, hc, sk / 2), "Box")
            slab("BoxLidS", cu, (0, 0, cav_h + sk / 2), (hc, hc, sk / 2),
                 "Box", fine=True)
            slab("BoxLid", cu, (0, 0, cav_h + (LID + sk) / 2),
                 (ob, ob, (LID - sk) / 2), "Box")
            for s in (-1, 1):
                slab("BoxLidRx", cu, (0, s * (hc + WALL / 2), cav_h + sk / 2),
                     (ob, WALL / 2, sk / 2), "Box")
                slab("BoxLidRy", cu, (s * (hc + WALL / 2), 0, cav_h + sk / 2),
                     (WALL / 2, hc, sk / 2), "Box")
                # walls: skin (fine) next to the cavity, rest outside
                slab("BoxWxS", cu, (0, s * (hc + sk / 2), cav_h / 2),
                     (hc, sk / 2, cav_h / 2), "Box", fine=True)
                slab("BoxWx", cu, (0, s * (hc + (WALL + sk) / 2), cav_h / 2),
                     (ob, (WALL - sk) / 2, cav_h / 2), "Box")
                slab("BoxWxC", cu, (s * (hc + sk / 2), s * (hc + sk / 2),
                                    cav_h / 2), (sk / 2, sk / 2, cav_h / 2),
                     "Box")
                slab("BoxWxC2", cu, (-s * (hc + sk / 2), s * (hc + sk / 2),
                                     cav_h / 2), (sk / 2, sk / 2, cav_h / 2),
                     "Box")
                slab("BoxWyS", cu, (s * (hc + sk / 2), 0, cav_h / 2),
                     (sk / 2, hc, cav_h / 2), "Box", fine=True)
                slab("BoxWy", cu, (s * (hc + (WALL + sk) / 2), 0, cav_h / 2),
                     ((WALL - sk) / 2, hc + sk, cav_h / 2), "Box")
            px, py, pz = PLATE
            slab("Plate", cu, (0, 0, -FLOOR - pz / 2), (px / 2, py / 2, pz / 2))
            # PCB: the cavity floor minus a 1 mm-margin hole around each chip,
            # built as strips along y (holes span the full x of each chip).
            holes = []
            for name in ("DieL", "Chip2"):
                (x, y, _), (hx, hy, _) = vols[name]
                if name == "DieL":   # one hole for the pair
                    hx_pair = gap / 2 + 2 * vols["DieL"][1][0]
                    holes.append((0.0, y, hx_pair + 1, hy + 1))
                else:
                    holes.append((x, y, hx + 1, hy + 1))
            holes.sort(key=lambda h: h[1])
            ylo = -hc
            k = 0
            for (x, y, hx, hy) in holes:
                if y - hy > ylo:                   # full-width strip below
                    slab(f"PCB{k}", pcb, (0, (ylo + y - hy) / 2, T_PCB / 2),
                         (hc, (y - hy - ylo) / 2, T_PCB / 2), "PCB", True)
                    k += 1
                for s in (-1, 1):                  # side strips beside hole
                    x0, x1 = (-hc, x - hx) if s < 0 else (x + hx, hc)
                    if x1 > x0:
                        slab(f"PCB{k}", pcb, ((x0 + x1) / 2, y, T_PCB / 2),
                             ((x1 - x0) / 2, hy, T_PCB / 2), "PCB", True)
                        k += 1
                ylo = y + hy
            if hc > ylo:
                slab(f"PCB{k}", pcb, (0, (ylo + hc) / 2, T_PCB / 2),
                     (hc, (hc - ylo) / 2, T_PCB / 2), "PCB", True)
            for i, (x, y, z) in enumerate(peds):
                slab(f"Ped{i}", cu, (x, y, z), (PED / 2, PED / 2, h_ped / 2),
                     "Ped", True)
            for name, (c, h) in vols.items():
                slab(name, si, c, h, fine=True)
            # Fine production cut only in the device region (dies, chip 2,
            # pedestals, PCB, inner box skin). A global 1 um cut made muon
            # runs 31 ms/primary because every fridge shell tracks keV
            # electrons that cannot reach the dies (smoke test 2026-11-16).
            if cut_mm is not None and cut_mode == "region":
                # geant4_pybind has no G4ProductionCuts; the cut is set
                # below with /run/setCutForRegion, which creates them.
                reg = G4Region("Device")
                for lv in fine_lvs:
                    reg.AddRootLogicalVolume(lv)
                _keep.append(reg)
            if cryostat:
                from g4_cryostat_spectrum import SHELLS, PLATES, PLATE_GAP_MM
                from geant4_pybind import G4SubtractionSolid
                off = fridge_off
                for name, mat, hxy, hz, w in SHELLS:
                    hxy, hz = hxy + off, hz + off
                    o = G4Box(name + "O", (hxy + w) * mm, (hxy + w) * mm,
                              (hz + w) * mm)
                    i_ = G4Box(name + "I", hxy * mm, hxy * mm, hz * mm)
                    sol = G4SubtractionSolid(name, o, i_)
                    _keep.append(sol)
                    lv = G4LogicalVolume(sol, nist.FindOrBuildMaterial(mat),
                                         name)
                    G4PVPlacement(None, G4ThreeVector(0, 0, cav_h / 2 * mm),
                                  lv, name, world_l, False, 0)
                    fridge_names.add(name)
                ztop = cav_h / 2 + SHELLS[-1][3] + off + SHELLS[-1][4] + 5.0
                for name, mat, w in PLATES:
                    lv = G4LogicalVolume(
                        G4Box(name, 100 * mm, 100 * mm, w / 2 * mm),
                        nist.FindOrBuildMaterial(mat), name)
                    G4PVPlacement(None,
                                  G4ThreeVector(0, 0, (ztop + w / 2) * mm),
                                  lv, name, world_l, False, 0)
                    fridge_names.add(name)
                    ztop += w + PLATE_GAP_MM
            return world_p

    def _isotropic():
        zz = rng.uniform(-1, 1)
        ph = rng.uniform(0, 2 * math.pi)
        s = math.sqrt(1 - zz * zz)
        n = (s * math.cos(ph), s * math.sin(ph), zz)
        pos = (R_SPHERE_MM * n[0], R_SPHERE_MM * n[1],
               cav_h / 2 + R_SPHERE_MM * n[2])
        ct = math.sqrt(rng.random())
        st = math.sqrt(1 - ct * ct)
        a = rng.uniform(0, 2 * math.pi)
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

    def _muon():
        ct = rng.random() ** 0.25
        st = math.sqrt(1 - ct * ct)
        a = rng.uniform(0, 2 * math.pi)
        d = (st * math.cos(a), st * math.sin(a), -ct)
        r = r_muon_mm * math.sqrt(rng.random())
        b = rng.uniform(0, 2 * math.pi)
        x, y = r * math.cos(b), Y_PAIR + r * math.sin(b)
        back = 400.0
        return (x - back * d[0], y - back * d[1], z_mid - back * d[2]), d

    class Gun(G4VUserPrimaryGeneratorAction):
        def __init__(self):
            super().__init__()
            self.gun = G4ParticleGun(1)
            self.gun.SetParticleDefinition(
                G4ParticleTable.GetParticleTable().FindParticle(
                    "mu-" if source == "muon" else "gamma"))

        def GeneratePrimaries(self, event):
            if source == "muon":
                pos, d = _muon()
                self.gun.SetParticleEnergy(muon_mev * MeV)
            else:
                pos, d = _isotropic()
                if source == "gamma-lines":
                    r = rng.random()
                    e = next(e for c, e in cum if r <= c)
                elif source == "gamma-scatter":
                    e = 30.0 * (2600.0 / 30.0) ** rng.random()
                else:
                    raise ValueError(source)
                self.gun.SetParticleEnergy(e * keV)
            self.gun.SetParticlePosition(G4ThreeVector(*[c * mm for c in pos]))
            self.gun.SetParticleMomentumDirection(G4ThreeVector(*d))
            self.gun.GeneratePrimaryVertex(event)

    events: list = []
    flag_of = {"DieL": 0, "DieR": 1, "Box": 2, "Ped": 3, "PCB": 4,
               "Plate": 5, "Chip2": 7}

    class Scorer(G4UserSteppingAction):
        def __init__(self):
            super().__init__()
            self.reset()

        def reset(self):
            self.e = [0.0, 0.0]
            self.prim = [0.0, 0.0]
            self.hit = [0] * 8
            self.entry = [defaultdict(float), defaultdict(float)]

        def UserSteppingAction(self, step):
            vol = step.GetPreStepPoint().GetTouchable().GetVolume()
            if vol is None:
                return
            name = vol.GetLogicalVolume().GetName()
            primary = step.GetTrack().GetTrackID() == 1
            if name == "DieL" or name == "DieR":
                i = 0 if name == "DieL" else 1
                de = step.GetTotalEnergyDeposit() / keV
                self.e[i] += de
                if primary:
                    self.prim[i] += de
                # entry diagnostic: a non-primary track crossing INTO a die;
                # record its birth volume and kinetic energy (keV)
                pre = step.GetPreStepPoint()
                if (entry and not primary and pre.GetStepStatus()
                        == g4.G4StepStatus.fGeomBoundary):
                    vv = step.GetTrack().GetLogicalVolumeAtVertex()
                    vn = vv.GetName() if vv is not None else "?"
                    if vn in fridge_names:
                        vn = "Fridge"
                    self.entry[i][vn] += pre.GetKineticEnergy() / keV
            if primary:
                j = flag_of.get(name)
                if j is None and name in fridge_names:
                    j = 6
                if j is not None:
                    self.hit[j] = 1

    class Ev(G4UserEventAction):
        def __init__(self, sc):
            super().__init__()
            self.sc = sc

        def BeginOfEventAction(self, event):
            self.sc.reset()

        def EndOfEventAction(self, event):
            if max(self.sc.e) >= min_keV:
                row = ([round(v, 1) for v in self.sc.e]
                       + [round(v, 1) for v in self.sc.prim] + self.sc.hit)
                if entry and min(self.sc.e) >= min_keV:
                    row.append([{k: round(v, 1) for k, v in d.items()}
                                for d in self.sc.entry])
                events.append(row)

    rm = G4RunManagerFactory.CreateRunManager(G4RunManagerType.Serial)
    rm.SetUserInitialization(Det())
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
    # 1 um cut: the default 0.7 mm suppresses the knock-on electrons that
    # cross gaps between chips (g4_wu_modules, 2026-10-12).
    if cut_mm is not None and cut_mode == "global":
        ui.ApplyCommand(f"/run/setCut {cut_mm} mm")
    elif cut_mm is not None:
        ui.ApplyCommand(f"/run/setCutForRegion Device {cut_mm} mm")
    if check_overlaps:
        ui.ApplyCommand("/geometry/test/run")
    t0 = time.perf_counter()
    rm.BeamOn(n_primaries)
    sec = time.perf_counter() - t0
    if dump_couples:
        ui.ApplyCommand("/run/dumpCouples")
    gen_area_cm2 = math.pi * ((r_muon_mm if source == "muon"
                               else R_SPHERE_MM) / 10.0) ** 2
    return {"source": source, "n_primaries": n_primaries, "seed": seed,
            "L_mm": L, "W_mm": W, "T_mm": T, "gap_mm": gap,
            "h_ped_mm": h_ped, "cav_h_mm": cav_h,
            "muon_mev": muon_mev if source == "muon" else None,
            "physics": physics, "gen_area_cm2": gen_area_cm2,
            "min_keV": min_keV, "cryostat": cryostat,
            "r_muon_mm": r_muon_mm, "fridge_off": fridge_off,
            "cut_mm": cut_mm if cut_mm is not None else 0.7,
            "cut_mode": cut_mode if cut_mm is not None else "default",
            "skin_mm": SKIN,
            "columns": COLUMNS + (["entry_L_R"] if entry else []),
            "entry": entry, "seconds": sec, "events": events}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("n", type=int)
    p.add_argument("out")
    p.add_argument("--L", type=float, default=13.5)
    p.add_argument("--W", type=float, default=10.0)
    p.add_argument("--T", type=float, default=0.5)
    p.add_argument("--gap", type=float, default=0.2)
    p.add_argument("--h-ped", type=float, default=0.5)
    p.add_argument("--cav-h", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--muon-mev", type=float, default=4000.0)
    p.add_argument("--physics", default="FTFP_BERT")
    p.add_argument("--cryostat", action="store_true")
    p.add_argument("--r-muon", type=float, default=60.0)
    p.add_argument("--fridge-off", type=float, default=20.0)
    p.add_argument("--cut", type=float, default=0.001,
                   help="production cut in mm (default 1 um; 0.7 = Geant4)")
    p.add_argument("--cut-mode", choices=["region", "global"],
                   default="region")
    p.add_argument("--check-overlaps", action="store_true")
    p.add_argument("--dump-couples", action="store_true")
    p.add_argument("--entry", action="store_true",
                   help="for events with both dies >= min_keV, append per die "
                        "{birth volume: entering kinetic energy keV} of "
                        "non-primary tracks crossing into the die")
    a = p.parse_args()
    r = run(a.source, a.n, L=a.L, W=a.W, T=a.T, gap=a.gap, h_ped=a.h_ped,
            cav_h=a.cav_h, seed=a.seed, muon_mev=a.muon_mev,
            physics=a.physics, cryostat=a.cryostat, r_muon_mm=a.r_muon,
            cut_mm=a.cut, cut_mode=a.cut_mode, fridge_off=a.fridge_off,
            check_overlaps=a.check_overlaps, dump_couples=a.dump_couples,
            entry=a.entry)
    json.dump(r, open(a.out, "w"))
    ev = r["events"]
    both = sum(1 for e in ev if e[0] > 0 and e[1] > 0)
    direct = sum(1 for e in ev if e[2] > 0 and e[3] > 0)
    print(f"DONE {a.source} n={a.n} {r['seconds']:.1f}s "
          f"{r['seconds'] / a.n * 1e6:.1f} us/primary events={len(ev)} "
          f"L&R>0={both} direct={direct}", flush=True)
