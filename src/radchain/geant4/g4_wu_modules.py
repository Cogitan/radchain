"""g4_wu_modules.py: per-module deposits in Wu et al.'s two-module device.

WHY
---
Reproduce the cross-module coincidence rate of Wu et al. (arXiv:2505.15919,
PR Applied 24, 044022, 2025): ~2% of events in one flip-chip module coincide
with an event in the other. Targets and verdict rules were fixed before the runs.

GEOMETRY (+z up, millimetres)
-----------------------------
Wu PRX 14, 041030 (2024) App. A and Wu 2025 Sec. II A:
- motherboard (MB): sapphire 20 x 20 x t_mb, bottom on the enclosure floor
- two daughterboards (DBA, DBB): sapphire 3 (x) x 7 (y) x t_db, side by side,
  long axes parallel to y, edge gap `gap` in x, 6 um above the MB top
  (SU-8 studs; the studs and adhesive are NOT modelled: 6 um of polymer is
  negligible for ionizing transport, and phonon transmission is out of scope)
- Al enclosure, `wall` thick, cavity 26 x 26 x 6 mm (dimensions not
  published; bracket the wall)
- Cu mixing-chamber plate 50 x 50 x 8 mm under the enclosure (Loer et al.
  MXC plate thickness)
Substrate thickness and gap are NOT in the papers: brackets in the prereg.

SOURCES
-------
gamma-lines    primordial lines (UNSCEAR activities, as g4_env_spectrum)
gamma-scatter  dN/dE ~ 1/E, 30 keV - 2.6 MeV (degraded continuum)
  Both isotropic from a sphere of radius R_SPHERE_MM, cosine-weighted inward.
  Rate convention (same as rate_crosscheck): rate = P x flux x pi R^2.
muon           mu-, `muon_mev` (default 4 GeV), sea-level zenith I ~ cos^2,
               uniform on a horizontal disk of radius R_MUON_MM through the
               device mid-plane, so flux through it ~ cos^3.
               Rate = P x muon flux (/cm2/s) x pi R_MUON^2.

OUTPUT
------
`events`: one row per primary that deposited anything in any of the three
substrates: [E_DBA keV, E_DBB keV, E_MB keV], 0.1 keV resolution.

Runs in WSL (g4cmp/README.md). One G4RunManager per process.
"""
from __future__ import annotations

import json
import math
import random
import time

from g4_env_spectrum import line_table

R_SPHERE_MM = 45.0
R_MUON_MM = 30.0
CAVITY = (26.0, 26.0, 6.0)          # inner x, y, z of the Al enclosure
PLATE = (50.0, 50.0, 8.0)           # Cu MXC plate
STUD_MM = 0.006                     # SU-8 stud height


def layout(t_db: float, t_mb: float, gap: float):
    """Centres and half sizes of every volume, cavity floor at z = 0."""
    mb = ((0.0, 0.0, t_mb / 2), (10.0, 10.0, t_mb / 2))
    zdb = t_mb + STUD_MM + t_db / 2
    xoff = 1.5 + gap / 2
    dba = ((-xoff, 0.0, zdb), (1.5, 3.5, t_db / 2))
    dbb = ((+xoff, 0.0, zdb), (1.5, 3.5, t_db / 2))
    return {"MB": mb, "DBA": dba, "DBB": dbb}


def run(source: str, n_primaries: int, t_db: float = 0.5, t_mb: float = 0.5,
        gap: float = 2.0, wall: float = 5.0, seed: int = 1,
        muon_mev: float = 4000.0, min_keV: float = 1.0,
        physics: str = "FTFP_BERT", range_kill: bool = False,
        diag: bool = False, cryostat: bool = False,
        r_muon_mm: float = R_MUON_MM, cut_mm: float | None = None,
        orient: str | None = None, fridge_off: float = 20.0,
        check_overlaps: bool = False) -> dict:
    """`orient` (None = legacy placement, identical to before 2026-10-13):
    'flat', 'vy' or 'vx' put the whole package (enclosure, plate, substrates)
    in a mother volume rotated about its own centre, so the fridge stays
    aligned with the sky:
      flat  as built (device plane horizontal)
      vy    rotated 90 deg about y: module planes vertical, module A ABOVE B
      vx    rotated 90 deg about x: module planes vertical, A BESIDE B
    Use the same `fridge_off` for every orientation being compared (28 mm
    clears the rotated plate; the legacy value 20 does not)."""
    import geant4_pybind as g4
    from geant4_pybind import (G4VUserDetectorConstruction, G4NistManager,
                               G4Box, G4SubtractionSolid, G4LogicalVolume,
                               G4PVPlacement, G4RunManagerFactory,
                               G4RotationMatrix,
                               G4RunManagerType, G4UImanager, G4ParticleGun,
                               G4ParticleTable, G4ThreeVector, mm, keV, MeV,
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
    lay = layout(t_db, t_mb, gap)
    z_mid = lay["DBA"][0][2]          # device mid-plane for the muon disk
    cx, cy, cz = CAVITY

    # World sized from the muon start distance. A primary starting OUTSIDE
    # the world is dropped silently, and in the 5-box + fridge geometry that
    # path segfaulted non-deterministically (bisected 2026-10-16: a 1500 mm
    # world ran clean 3/3 where 800 mm crashed 5/5). Never let a start fall
    # outside: the world half-size must exceed r_muon + back-off + margin.
    WORLD_H = max(500.0, r_muon_mm + 400.0 + 100.0)
    class Det(G4VUserDetectorConstruction):
        def Construct(self):
            nist = G4NistManager.Instance()
            vac = nist.FindOrBuildMaterial("G4_Galactic")
            sap = nist.FindOrBuildMaterial("G4_ALUMINUM_OXIDE")
            al = nist.FindOrBuildMaterial("G4_Al")
            cu = nist.FindOrBuildMaterial("G4_Cu")
            world_s = G4Box("World", WORLD_H * mm, WORLD_H * mm, WORLD_H * mm)
            world_l = G4LogicalVolume(world_s, vac, "World")
            world_p = G4PVPlacement(None, G4ThreeVector(), world_l, "World",
                                    None, False, 0)
            # Package mother: with `orient`, everything below goes into a
            # vacuum box centred on the package, rotated as a unit.
            if orient is None:
                pkg_l, dz = world_l, 0.0
            else:
                z_lo, z_hi = -wall - PLATE[2], cz + wall
                zc = (z_lo + z_hi) / 2
                pkg_l = G4LogicalVolume(
                    G4Box("Pkg", (PLATE[0] / 2 + 1) * mm,
                          (PLATE[1] / 2 + 1) * mm, ((z_hi - z_lo) / 2 + 1) * mm),
                    vac, "Pkg")
                self.rot = None
                if orient in ("vy", "vx"):
                    self.rot = G4RotationMatrix()
                    if orient == "vy":
                        self.rot.rotateY(math.pi / 2)
                    else:
                        self.rot.rotateX(math.pi / 2)
                elif orient != "flat":
                    raise ValueError(orient)
                G4PVPlacement(self.rot, G4ThreeVector(0, 0, zc * mm), pkg_l,
                              "Pkg", world_l, False, 0)
                dz = -zc
            # Al enclosure as outer box minus cavity; cavity floor at z = 0.
            outer = G4Box("BoxOuter", (cx / 2 + wall) * mm,
                          (cy / 2 + wall) * mm, (cz / 2 + wall) * mm)
            inner = G4Box("BoxInner", cx / 2 * mm, cy / 2 * mm, cz / 2 * mm)
            shell = G4SubtractionSolid("Enclosure", outer, inner)
            shell_l = G4LogicalVolume(shell, al, "Enclosure")
            G4PVPlacement(None, G4ThreeVector(0, 0, (cz / 2 + dz) * mm), shell_l,
                          "Enclosure", pkg_l, False, 0)
            # Cu plate directly under the enclosure.
            px, py, pz = PLATE
            plate_l = G4LogicalVolume(
                G4Box("Plate", px / 2 * mm, py / 2 * mm, pz / 2 * mm), cu,
                "Plate")
            G4PVPlacement(None, G4ThreeVector(0, 0, (-wall - pz / 2 + dz) * mm),
                          plate_l, "Plate", pkg_l, False, 0)
            if cryostat:
                # Fridge material OUTSIDE the enclosure, for muon-induced
                # secondaries only (the gamma field is defined at the
                # enclosure, so gamma runs never use this). Wall thicknesses
                # and materials from g4_cryostat_spectrum (Loer et al.
                # arXiv:2403.01032 Table 1), box shells moved out to clear
                # the enclosure; plates as 200 mm slabs above.
                from g4_cryostat_spectrum import SHELLS, PLATES, PLATE_GAP_MM
                off = fridge_off                 # clear the package
                for name, mat, hxy, hz, w in SHELLS:
                    hxy, hz = hxy + off, hz + off
                    o = G4Box(name + "O", (hxy + w) * mm, (hxy + w) * mm,
                              (hz + w) * mm)
                    i_ = G4Box(name + "I", hxy * mm, hxy * mm, hz * mm)
                    lv = G4LogicalVolume(G4SubtractionSolid(name, o, i_),
                                         nist.FindOrBuildMaterial(mat), name)
                    G4PVPlacement(None, G4ThreeVector(0, 0, cz / 2 * mm), lv,
                                  name, world_l, False, 0)
                ztop = cz / 2 + SHELLS[-1][3] + off + SHELLS[-1][4] + 5.0
                for name, mat, w in PLATES:
                    lv = G4LogicalVolume(
                        G4Box(name, 100 * mm, 100 * mm, w / 2 * mm),
                        nist.FindOrBuildMaterial(mat), name)
                    G4PVPlacement(None, G4ThreeVector(0, 0, (ztop + w / 2) * mm),
                                  lv, name, world_l, False, 0)
                    ztop += w + PLATE_GAP_MM
            # Substrates live in the world (the cavity is world vacuum).
            for name, ((x, y, z), (hx, hy, hz)) in lay.items():
                lv = G4LogicalVolume(G4Box(name, hx * mm, hy * mm, hz * mm),
                                     sap, name)
                G4PVPlacement(None, G4ThreeVector(x * mm, y * mm,
                                                  (z + dz) * mm),
                              lv, name, pkg_l, False, 0)
            return world_p

    def _isotropic():
        zz = rng.uniform(-1, 1)
        ph = rng.uniform(0, 2 * math.pi)
        s = math.sqrt(1 - zz * zz)
        n = (s * math.cos(ph), s * math.sin(ph), zz)
        pos = (R_SPHERE_MM * n[0], R_SPHERE_MM * n[1], cz / 2 + R_SPHERE_MM * n[2])
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
        x, y = r * math.cos(b), r * math.sin(b)
        back = 400.0
        zd = z_mid if orient is None else (-wall - PLATE[2] + cz + wall) / 2
        return (x - back * d[0], y - back * d[1], zd - back * d[2]), d

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

    # Range cut on electrons born in the metal. An e+/- whose CSDA range is
    # shorter than its distance to the cavity cannot reach a substrate, so it
    # is killed at birth. Katz-Penfold range-energy relation (Al-like, used
    # for Cu too, which over-estimates range in Cu by density and so only
    # under-kills). Bremsstrahlung from killed electrons is lost; at these
    # energies the radiative yield in Al/Cu is < ~2%. Stated, not hidden.
    half_cav = (cx / 2, cy / 2, cz / 2)
    cav_c = (0.0, 0.0, cz / 2)
    killed = [0]

    def _dist_to_cavity_mm(p):
        d2 = 0.0
        for k in range(3):
            o = abs(p[k] - cav_c[k]) - half_cav[k]
            if o > 0:
                d2 += o * o
        return math.sqrt(d2)

    def _range_mm(e_mev, rho):
        if e_mev <= 0:
            return 0.0
        n = 1.265 - 0.0954 * math.log(e_mev)
        return 0.412 * e_mev ** n / rho * 10.0     # g/cm2 -> mm

    rho_of = {"Enclosure": 2.70, "Plate": 8.96}

    class Stack(g4.G4UserStackingAction):
        def ClassifyNewTrack(self, track):
            pdg = track.GetDefinition().GetPDGEncoding()
            if abs(pdg) == 11:
                vol = track.GetVolume()
                name = vol.GetName() if vol is not None else ""
                rho = rho_of.get(name)
                if rho is not None:
                    p = track.GetPosition()
                    pos = (p.x / mm, p.y / mm, p.z / mm)
                    if _range_mm(track.GetKineticEnergy() / MeV, rho) < \
                            _dist_to_cavity_mm(pos):
                        killed[0] += 1
                        return g4.G4ClassificationOfNewTrack.fKill
            return g4.G4ClassificationOfNewTrack.fUrgent

    events: list = []
    idx = {"DBA": 0, "DBB": 1, "MB": 2}

    # diag: per event also record which volumes the PRIMARY itself entered
    # (DBA, DBB, MB, Enclosure, Plate) and, for coincidences, how much of each
    # module's deposit came from the primary versus its secondaries.
    diag_names = {"DBA": 0, "DBB": 1, "MB": 2, "Enclosure": 3, "Plate": 4}

    class Scorer(G4UserSteppingAction):
        def __init__(self):
            super().__init__()
            self.e = [0.0, 0.0, 0.0]
            self.prim = [0.0, 0.0, 0.0]
            self.hit = [0, 0, 0, 0, 0]

        def UserSteppingAction(self, step):
            vol = step.GetPreStepPoint().GetTouchable().GetVolume()
            if vol is None:
                return
            name = vol.GetName()
            i = idx.get(name)
            if i is not None:
                de = step.GetTotalEnergyDeposit() / keV
                self.e[i] += de
                if diag and step.GetTrack().GetTrackID() == 1:
                    self.prim[i] += de
            if diag and step.GetTrack().GetTrackID() == 1:
                j = diag_names.get(name)
                if j is not None:
                    self.hit[j] = 1

    class Ev(G4UserEventAction):
        def __init__(self, sc):
            super().__init__()
            self.sc = sc

        def BeginOfEventAction(self, event):
            self.sc.e = [0.0, 0.0, 0.0]
            self.sc.prim = [0.0, 0.0, 0.0]
            self.sc.hit = [0, 0, 0, 0, 0]

        def EndOfEventAction(self, event):
            if max(self.sc.e) >= min_keV:
                row = [round(v, 1) for v in self.sc.e]
                if diag:
                    row += [round(v, 1) for v in self.sc.prim] + self.sc.hit
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
            if range_kill:
                self.SetUserAction(Stack())

    rm.SetUserInitialization(Actions())
    rm.Initialize()
    ui = G4UImanager.GetUIpointer()
    for cmd in ("/run/verbose 0", "/event/verbose 0", "/tracking/verbose 0",
                "/run/printProgress 0", f"/random/setSeeds {seed} {seed + 7}"):
        ui.ApplyCommand(cmd)
    # Production cut. Geant4's default 0.7 mm means knock-on electrons below
    # roughly 0.3-0.6 MeV in sapphire are NOT created as tracks: their energy
    # is deposited along the parent. That suppresses exactly the low-energy
    # secondaries that could cross the gap between modules, so it is a knob
    # to test, not a default to trust (added 2026-10-12).
    if cut_mm is not None:
        ui.ApplyCommand(f"/run/setCut {cut_mm} mm")
    if check_overlaps:
        ui.ApplyCommand("/geometry/test/run")
    t0 = time.perf_counter()
    rm.BeamOn(n_primaries)
    sec = time.perf_counter() - t0
    if cryostat and source != "muon":
        raise ValueError("cryostat geometry is for muon runs only")
    gen_area_cm2 = math.pi * ((r_muon_mm if source == "muon"
                               else R_SPHERE_MM) / 10.0) ** 2
    return {"source": source, "n_primaries": n_primaries, "seed": seed,
            "t_db_mm": t_db, "t_mb_mm": t_mb, "gap_mm": gap, "wall_mm": wall,
            "muon_mev": muon_mev if source == "muon" else None,
            "physics": physics, "gen_area_cm2": gen_area_cm2,
            "min_keV": min_keV, "range_kill": range_kill,
            "killed_electrons": killed[0], "diag": diag,
            "cryostat": cryostat, "r_muon_mm": r_muon_mm,
            "orient": orient, "fridge_off": fridge_off,
            "cut_mm": cut_mm if cut_mm is not None else 0.7,
            "diag_columns": (["E_DBA", "E_DBB", "E_MB", "prim_DBA",
                              "prim_DBB", "prim_MB", "hit_DBA", "hit_DBB",
                              "hit_MB", "hit_Enclosure", "hit_Plate"]
                             if diag else None),
            "seconds": sec, "events": events}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("n", type=int)
    p.add_argument("out")
    p.add_argument("--t-db", type=float, default=0.5)
    p.add_argument("--t-mb", type=float, default=0.5)
    p.add_argument("--gap", type=float, default=2.0)
    p.add_argument("--wall", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--muon-mev", type=float, default=4000.0)
    p.add_argument("--physics", default="FTFP_BERT")
    p.add_argument("--range-kill", action="store_true",
                   help="kill e+- in metal that cannot reach the cavity; "
                        "measured 2026-10-12 to buy nothing (16 vs 18 "
                        "us/primary), off by default")
    p.add_argument("--diag", action="store_true",
                   help="also record primary-track deposits and volumes hit")
    p.add_argument("--cryostat", action="store_true",
                   help="add fridge shells and plates (muon runs only)")
    p.add_argument("--r-muon", type=float, default=R_MUON_MM)
    p.add_argument("--orient", choices=["flat", "vy", "vx"], default=None)
    p.add_argument("--fridge-off", type=float, default=20.0)
    p.add_argument("--check-overlaps", action="store_true")
    p.add_argument("--cut", type=float, default=None,
                   help="production range cut in mm (Geant4 default 0.7)")
    a = p.parse_args()
    r = run(a.source, a.n, t_db=a.t_db, t_mb=a.t_mb, gap=a.gap, wall=a.wall,
            seed=a.seed, muon_mev=a.muon_mev, physics=a.physics,
            range_kill=a.range_kill, diag=a.diag, cryostat=a.cryostat,
            r_muon_mm=a.r_muon, cut_mm=a.cut, orient=a.orient,
            fridge_off=a.fridge_off, check_overlaps=a.check_overlaps)
    json.dump(r, open(a.out, "w"))
    ev = r["events"]
    both = sum(1 for e in ev if e[0] > 0 and e[1] > 0)
    print(f"DONE {a.source} n={a.n} {r['seconds']:.1f}s "
          f"{r['seconds'] / a.n * 1e6:.1f} us/primary events={len(ev)} "
          f"A&B>0={both}", flush=True)
