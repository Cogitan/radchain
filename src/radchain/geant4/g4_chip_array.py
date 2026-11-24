"""g4_chip_array.py: per-chip deposits for multi-chip layouts (correlated erasure).

WHY
---
Distributed QEC across chips (Xu et al., PRL 129, 240502, 2022) assumes each
particle erases at most one chip. This scores, event by event, the energy
every chip receives, so the rate of events erasing >= 2 or >= 3 chips can be
read off for a chosen erasure threshold E*. The method was validated against
Wu et al. 2025 (g4_wu_modules.py).

GEOMETRY (+z up, mm). Chips 10 x 10 x 0.5 Si, 10 um above their carrier
(flip-chip; bumps are not modelled, they matter for phonons, not particles).
Layouts (see `layout`):
  A1  3 x 3 chips, 1 mm gaps, one shared 0.5 mm Si carrier
  A3  same, 3 mm gaps
  B3  each chip on its own 12 x 12 x 0.5 Si carrier, 3 mm between carriers
  V1  A1 with the whole package rotated 90 deg about y (array plane vertical)
  S1  two chips face to face, 1 mm apart, bottom chip on a carrier
The package (chips, carriers, a 5 mm Al enclosure, an 8 mm Cu plate under it)
sits in a vacuum mother volume `Dev`, rotated for V1, so the fridge (muon
runs only, `--cryostat`) stays aligned with the sky.

SOURCES as g4_wu_modules: gamma-lines, gamma-scatter (isotropic from a sphere
enclosing Dev; rate = P x flux x pi R^2) and muon (cos^2 zenith, 4 GeV,
uniform on a horizontal disk through the package centre; rate = P x flux x
pi R^2).

OUTPUT: per event with max chip deposit >= min_keV, [E_chip0 ... E_chipN-1,
E_carriers] in keV (0.1 keV).
"""
from __future__ import annotations

import json
import math
import random
import time

from g4_env_spectrum import line_table

CHIP = 10.0
T_CHIP = 0.5
T_CARRIER = 0.5
BUMP = 0.010
WALL = 5.0
PLATE_T = 8.0


def layout(lid: str):
    """(chips, carriers): lists of (name, centre xyz, half xyz)."""
    h = CHIP / 2
    chips, carriers = [], []
    zc = T_CARRIER / 2
    zchip = T_CARRIER + BUMP + T_CHIP / 2
    if lid in ("A1", "A3", "V1", "B3"):
        gap = {"A1": 1.0, "V1": 1.0, "A3": 3.0, "B3": 5.0}[lid]
        pitch = CHIP + gap
        k = 0
        for iy in (-1, 0, 1):
            for ix in (-1, 0, 1):
                x, y = ix * pitch, iy * pitch
                chips.append((f"C{k}", (x, y, zchip), (h, h, T_CHIP / 2)))
                if lid == "B3":
                    carriers.append((f"K{k}", (x, y, zc), (6.0, 6.0, T_CARRIER / 2)))
                k += 1
        if lid != "B3":
            half = (3 * CHIP + 2 * gap) / 2 + 2.0
            carriers.append(("K0", (0.0, 0.0, zc), (half, half, T_CARRIER / 2)))
    elif lid[:1] in ("P", "Q") and lid[1:].isdigit():
        # Pair study: two chips, each
        # on its OWN carrier, edge gap g mm. P = one shared Al box, Q = one
        # box per chip.
        g = float(lid[1:])
        for k, sx in enumerate((-1, 1)):
            x = sx * (h + g / 2)
            chips.append((f"C{k}", (x, 0.0, zchip), (h, h, T_CHIP / 2)))
            # 12 x 12 carriers, narrowed at tight gaps so the two never overlap
            # (at g = 1 mm they meet edge to edge)
            ch = min(6.0, h + g / 2)
            carriers.append((f"K{k}", (x, 0.0, zc), (ch, 6.0, T_CARRIER / 2)))
    elif lid[:1] == "X" and (lid == "X1" or lid[1:3] == "Q5" or lid[1:3] == "Q1"):
        # Code-block study: five chips in a
        # plus pattern. X1 = one shared carrier, 1 mm gaps. XQ<g> = own
        # carrier, own box, edge gap g mm.
        if lid == "X1":
            p = CHIP + 1.0
        else:
            p = CHIP + float(lid[2:])
        pts = [(0.0, 0.0), (p, 0.0), (-p, 0.0), (0.0, p), (0.0, -p)]
        for k, (x, y) in enumerate(pts):
            chips.append((f"C{k}", (x, y, zchip), (h, h, T_CHIP / 2)))
            if lid != "X1":
                carriers.append((f"K{k}", (x, y, zc), (6.0, 6.0, T_CARRIER / 2)))
        if lid == "X1":
            half = (3 * CHIP + 2.0) / 2 + 2.0
            carriers.append(("K0", (0.0, 0.0, zc), (half, half, T_CARRIER / 2)))
    elif lid == "S1":
        carriers.append(("K0", (0.0, 0.0, zc), (h + 2, h + 2, T_CARRIER / 2)))
        chips.append(("C0", (0.0, 0.0, zchip), (h, h, T_CHIP / 2)))
        z1 = T_CARRIER + BUMP + T_CHIP + 1.0 + T_CHIP / 2
        chips.append(("C1", (0.0, 0.0, z1), (h, h, T_CHIP / 2)))
    else:
        raise ValueError(lid)
    return chips, carriers


PAIR_BOX_HALF = 10.0          # cavity half width per chip box (carrier 6 + 4)
#: Pair layouts use ONE fridge size for every separation, so the fridge does
#: not move as the chips do: inner Cu box half-width 150 mm, height half 60 mm.
PAIR_FRIDGE_OX = 150.0 - 8.0
PAIR_FRIDGE_OZ = 60.0 - 6.0


def boxes(lid: str, chips, carriers):
    """Sample-box cavities as (cx, cy, half). None = one shared box (legacy)."""
    if lid[:1] == "Q" or lid[:2] == "XQ":
        return [(c[1][0], c[1][1], PAIR_BOX_HALF) for c in chips]
    return None


MU_MASS_MEV = 105.658
#: Chirkin's parameters for the curvature-corrected zenith angle (Guan et al.
#: 2015, Eq. 2 and Table 1, arXiv:1509.06176).
_P = (0.102573, -0.068287, 0.958633, 0.0407253, 0.817285)


def guan_flux(e_gev: float, cos_t: float) -> float:
    """Sea-level muon differential intensity dI/(dE dOmega), cm^-2 s^-1 sr^-1
    GeV^-1: Guan, Chu, Cao, Luk, Yang, arXiv:1509.06176, Eq. (3), a modified
    Gaisser formula fitted to world data down to ~1 GeV, with cos(theta*)
    from their Eq. (2). E is the muon energy in GeV."""
    p1, p2, p3, p4, p5 = _P
    c = max(cos_t, 1e-6)
    cs = math.sqrt((c * c + p1 * p1 + p2 * c ** p3 + p4 * c ** p5)
                   / (1 + p1 * p1 + p2 + p4))
    return (0.14 * (e_gev * (1 + 3.64 / (e_gev * cs ** 1.29))) ** -2.7
            * (1 / (1 + 1.1 * e_gev * cs / 115) + 0.054 / (1 + 1.1 * e_gev * cs / 850)))


def guan_sampler(rng, e_min_gev: float = 0.2, e_max_gev: float = 1000.0,
                 n_c: int = 200, n_e: int = 400):
    """Return a function () -> (cos_theta, total energy in MeV) sampling the
    muon flux THROUGH A HORIZONTAL PLANE (intensity x cos theta), on a grid in
    cos theta and ln E, uniform within cells. Total energy E >= 0.2 GeV
    (kinetic >= ~94 MeV); lower-energy muons stop in the building and fridge."""
    cs_edges = [i / n_c for i in range(n_c + 1)]
    le = [math.log(e_min_gev) + (math.log(e_max_gev) - math.log(e_min_gev)) * j / n_e
          for j in range(n_e + 1)]
    cells, cum, tot = [], [], 0.0
    for i in range(n_c):
        c = 0.5 * (cs_edges[i] + cs_edges[i + 1])
        for j in range(n_e):
            e = math.exp(0.5 * (le[j] + le[j + 1]))
            # weight = I(E, c) * c (horizontal plane) * dc * dE, dE = E dlnE
            w = guan_flux(e, c) * c * e
            tot += w
            cells.append((i, j))
            cum.append(tot)
    cum = [x / tot for x in cum]
    import bisect

    def sample():
        k = bisect.bisect_left(cum, rng.random())
        i, j = cells[min(k, len(cells) - 1)]
        c = cs_edges[i] + rng.random() / n_c
        e = math.exp(le[j] + rng.random() * (le[j + 1] - le[j]))
        return c, e * 1000.0
    return sample


def run(source: str, n_primaries: int, lid: str = "A1", seed: int = 1,
        muon_mev: float = 4000.0, min_keV: float = 100.0,
        cryostat: bool = False, r_muon_mm: float = 60.0,
        cut_mm: float | None = None, physics: str = "FTFP_BERT",
        check_overlaps: bool = False,
        min_carrier_keV: float | None = None,
        fridge_scale: float = 1.0, muon_spectrum: str = "mono",
        zenith_n: float = 2.0) -> dict:
    import geant4_pybind as g4
    from geant4_pybind import (G4VUserDetectorConstruction, G4NistManager,
                               G4Box, G4SubtractionSolid, G4LogicalVolume,
                               G4PVPlacement, G4RunManagerFactory,
                               G4RunManagerType, G4UImanager, G4ParticleGun,
                               G4ParticleTable, G4ThreeVector, mm, keV, MeV,
                               G4RotationMatrix,
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
    chips, carriers = layout(lid)
    n_chips = len(chips)
    # cavity around everything, 4 mm margin, 10 mm tall, floor at z = 0
    ext = max(max(abs(c[1][0]) + c[2][0], abs(c[1][1]) + c[2][1])
              for c in chips + carriers)
    cav_h = ext + 4.0                       # cavity half width
    cav_z = 10.0                            # cavity height
    bxs = boxes(lid, chips, carriers)
    if bxs is None:
        plate_h = cav_h + WALL + 10.0
    else:
        plate_h = max(abs(b[0]) + b[2] for b in bxs) + WALL + 10.0
    pair = (lid[:1] in ("P", "Q") and lid[1:].isdigit()) or lid[:1] == "X"
    # Dev mother: holds enclosure + plate; local origin at the cavity floor
    dev_hxy = plate_h
    dev_zlo = -WALL - PLATE_T
    dev_zhi = cav_z + WALL
    dev_hz = (dev_zhi - dev_zlo) / 2
    dev_zc = (dev_zhi + dev_zlo) / 2        # centre of Dev in local z
    rotate = lid == "V1"
    # world-frame half extents of Dev (rotation about y swaps x and z)
    wx, wy, wz = ((dev_hz, dev_hxy, dev_hxy) if rotate
                  else (dev_hxy, dev_hxy, dev_hz))
    R_SPHERE = math.sqrt(wx * wx + wy * wy + wz * wz) * 1.02

    # World sized from the muon start distance. A primary starting OUTSIDE
    # the world is dropped silently, and in the 5-box + fridge geometry that
    # path segfaulted non-deterministically (bisected 2026-10-16: a 1500 mm
    # world ran clean 3/3 where 800 mm crashed 5/5). Never let a start fall
    # outside: the world half-size must exceed r_muon + back-off + margin.
    BACK_MM = 700.0
    WORLD_H = max(800.0, r_muon_mm + BACK_MM + 100.0)
    class Det(G4VUserDetectorConstruction):
        def Construct(self):
            nist = G4NistManager.Instance()
            vac = nist.FindOrBuildMaterial("G4_Galactic")
            si = nist.FindOrBuildMaterial("G4_Si")
            al = nist.FindOrBuildMaterial("G4_Al")
            cu = nist.FindOrBuildMaterial("G4_Cu")
            world_l = G4LogicalVolume(G4Box("World", WORLD_H * mm,
                                            WORLD_H * mm, WORLD_H * mm),
                                      vac, "World")
            world_p = G4PVPlacement(None, G4ThreeVector(), world_l, "World",
                                    None, False, 0)
            dev_l = G4LogicalVolume(G4Box("Dev", dev_hxy * mm, dev_hxy * mm,
                                          dev_hz * mm), vac, "Dev")
            self.rot = None
            if rotate:
                self.rot = G4RotationMatrix()
                self.rot.rotateY(math.pi / 2)
            # Dev placed so the package centre sits at the world origin
            G4PVPlacement(self.rot, G4ThreeVector(0, 0, 0), dev_l, "Dev",
                          world_l, False, 0)
            off = -dev_zc            # local z -> Dev-centred z
            for bi, (bx, by, bh) in enumerate(
                    bxs if bxs is not None else [(0.0, 0.0, cav_h)]):
                outer = G4Box(f"EncO{bi}", (bh + WALL) * mm, (bh + WALL) * mm,
                              (cav_z / 2 + WALL) * mm)
                inner = G4Box(f"EncI{bi}", bh * mm, bh * mm, cav_z / 2 * mm)
                enc_l = G4LogicalVolume(G4SubtractionSolid(
                    "Enclosure", outer, inner), al, "Enclosure")
                G4PVPlacement(None, G4ThreeVector(
                    bx * mm, by * mm, (cav_z / 2 + off) * mm),
                    enc_l, "Enclosure", dev_l, False, bi)
            plate_l = G4LogicalVolume(G4Box("Plate", plate_h * mm,
                                            plate_h * mm, PLATE_T / 2 * mm),
                                      cu, "Plate")
            G4PVPlacement(None, G4ThreeVector(
                0, 0, (-WALL - PLATE_T / 2 + off) * mm), plate_l, "Plate",
                dev_l, False, 0)
            for name, (x, y, z), (hx, hy, hz) in chips + carriers:
                lv = G4LogicalVolume(G4Box(name, hx * mm, hy * mm, hz * mm),
                                     si, name)
                G4PVPlacement(None, G4ThreeVector(x * mm, y * mm,
                                                  (z + off) * mm),
                              lv, name, dev_l, False, 0)
            if cryostat:
                from g4_cryostat_spectrum import SHELLS, PLATES, PLATE_GAP_MM
                if pair:
                    ox, oz = PAIR_FRIDGE_OX, PAIR_FRIDGE_OZ
                    assert max(wx, wy) < 8.0 + ox and wz < 6.0 + oz,                         "package does not fit the fixed pair fridge"
                else:
                    ox = max(wx, wy) - 8.0 + 3.0
                    oz = wz - 6.0 + 3.0
                extra = 0.0
                for name, mat, hxy, hz, w0 in SHELLS:
                    # fridge_scale thickens every wall (floor test). Each shell is
                    # pushed out by the extra thickness of the shells inside
                    # it, so the original gaps are kept and nothing overlaps.
                    w = w0 * fridge_scale
                    a, b = hxy + ox + extra, hz + oz + extra
                    extra += w - w0
                    o = G4Box(name + "O", (a + w) * mm, (a + w) * mm,
                              (b + w) * mm)
                    i_ = G4Box(name + "I", a * mm, a * mm, b * mm)
                    lv = G4LogicalVolume(G4SubtractionSolid(name, o, i_),
                                         nist.FindOrBuildMaterial(mat), name)
                    G4PVPlacement(None, G4ThreeVector(), lv, name, world_l,
                                  False, 0)
                ztop = SHELLS[-1][3] + oz + extra + SHELLS[-1][4] * fridge_scale + 5.0
                for name, mat, w in PLATES:
                    w = w * fridge_scale
                    lv = G4LogicalVolume(G4Box(name, 200 * mm, 200 * mm,
                                               w / 2 * mm),
                                         nist.FindOrBuildMaterial(mat), name)
                    G4PVPlacement(None, G4ThreeVector(0, 0, (ztop + w / 2) * mm),
                                  lv, name, world_l, False, 0)
                    ztop += w + PLATE_GAP_MM
            return world_p

    def _isotropic():
        zz = rng.uniform(-1, 1)
        ph = rng.uniform(0, 2 * math.pi)
        s = math.sqrt(1 - zz * zz)
        n = (s * math.cos(ph), s * math.sin(ph), zz)
        pos = tuple(R_SPHERE * c for c in n)
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

    spectrum = guan_sampler(rng) if muon_spectrum == "guan" else None
    last_e = [muon_mev]
    hadron = source in ("neutron", "proton")
    if hadron:
        # JESD89A sea-level neutron spectrum, 1 MeV-10 GeV (Gordon et al.
        # 2004), shared with g4_species_spectrum. Protons use the same shape;
        # their normalisation is a bracketed p/n ratio applied in analysis
        # (PDG 2022 Sec. 30.3.3: ~1/3 of sea-level nucleons above 1 GeV/c are
        # neutrons). Zenith law I ~ cos^n per sr; through a horizontal plane
        # the flux is ~ cos^(n+1), sampled as ct = u^(1/(n+2)).
        from g4_species_spectrum import _energy_sampler
        hadron_e = _energy_sampler(1.0, 1.0e4)

    def _muon():
        if hadron:
            ct = rng.random() ** (1.0 / (zenith_n + 2.0))
            last_e[0] = hadron_e(rng.random())                # kinetic, MeV
        elif spectrum is None:
            ct = rng.random() ** 0.25
        else:
            ct, e_tot = spectrum()
            last_e[0] = max(e_tot - MU_MASS_MEV, 1.0)      # kinetic energy
        st = math.sqrt(1 - ct * ct)
        a = rng.uniform(0, 2 * math.pi)
        d = (st * math.cos(a), st * math.sin(a), -ct)
        r = r_muon_mm * math.sqrt(rng.random())
        b = rng.uniform(0, 2 * math.pi)
        x, y = r * math.cos(b), r * math.sin(b)
        back = BACK_MM
        return (x - back * d[0], y - back * d[1], -back * d[2]), d

    class Gun(G4VUserPrimaryGeneratorAction):
        def __init__(self):
            super().__init__()
            self.gun = G4ParticleGun(1)
            self.gun.SetParticleDefinition(
                G4ParticleTable.GetParticleTable().FindParticle(
                    {"muon": "mu-", "neutron": "neutron",
                     "proton": "proton"}.get(source, "gamma")))

        def GeneratePrimaries(self, event):
            if source in ("muon", "neutron", "proton"):
                pos, d = _muon()
                self.gun.SetParticleEnergy(last_e[0] * MeV)
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

    idx = {name: i for i, (name, _, _) in enumerate(chips)}
    carrier_names = {name for name, _, _ in carriers}
    events: list = []

    class Scorer(G4UserSteppingAction):
        def __init__(self):
            super().__init__()
            self.e = [0.0] * (n_chips + 1)

        def UserSteppingAction(self, step):
            vol = step.GetPreStepPoint().GetTouchable().GetVolume()
            if vol is None:
                return
            name = vol.GetName()
            i = idx.get(name)
            if i is None:
                if name in carrier_names:
                    i = n_chips
                else:
                    return
            self.e[i] += step.GetTotalEnergyDeposit() / keV

    class Ev(G4UserEventAction):
        def __init__(self, sc):
            super().__init__()
            self.sc = sc

        def BeginOfEventAction(self, event):
            self.sc.e = [0.0] * (n_chips + 1)

        def EndOfEventAction(self, event):
            # Carrier-only hits matter for phonon coupling through a shared
            # carrier, so they can be kept above their own floor.
            if max(self.sc.e[:n_chips]) >= min_keV or (
                    min_carrier_keV is not None and
                    self.sc.e[n_chips] >= min_carrier_keV):
                events.append([round(v, 1) for v in self.sc.e])

    if cryostat and source not in ("muon", "neutron", "proton"):
        raise ValueError("cryostat geometry is for muon and hadron runs only")
    rm = G4RunManagerFactory.CreateRunManager(G4RunManagerType.Serial)
    det = Det()
    rm.SetUserInitialization(det)
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
    if cut_mm is not None:
        ui.ApplyCommand(f"/run/setCut {cut_mm} mm")
    if check_overlaps:
        ui.ApplyCommand("/geometry/test/run")
    t0 = time.perf_counter()
    rm.BeamOn(n_primaries)
    sec = time.perf_counter() - t0
    directional = source in ("muon", "neutron", "proton")
    gen_area = math.pi * ((r_muon_mm if directional else R_SPHERE) / 10) ** 2
    return {"source": source, "layout": lid, "n_chips": n_chips,
            "n_primaries": n_primaries, "seed": seed,
            "muon_mev": muon_mev if source == "muon" else None,
            "muon_spectrum": muon_spectrum if source == "muon" else None,
            "zenith_n": zenith_n if source in ("neutron", "proton") else None,
            "hadron_spectrum": ("JESD89A 1 MeV-10 GeV (shape; proton normalisation "
                                "applied in analysis)") if source in ("neutron", "proton") else None,
            "cryostat": cryostat, "r_muon_mm": r_muon_mm,
            "r_sphere_mm": R_SPHERE, "gen_area_cm2": gen_area,
            "cut_mm": cut_mm if cut_mm is not None else 0.7,
            "min_keV": min_keV, "min_carrier_keV": min_carrier_keV,
            "fridge_scale": fridge_scale,
            "physics": physics, "seconds": sec,
            "columns": [n for n, _, _ in chips] + ["carriers"],
            "events": events}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("n", type=int)
    p.add_argument("out")
    p.add_argument("--layout", default="A1")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--muon-mev", type=float, default=4000.0)
    p.add_argument("--min-kev", type=float, default=100.0)
    p.add_argument("--cryostat", action="store_true")
    p.add_argument("--r-muon", type=float, default=60.0)
    p.add_argument("--cut", type=float, default=None)
    p.add_argument("--check-overlaps", action="store_true")
    p.add_argument("--min-carrier-kev", type=float, default=None)
    p.add_argument("--fridge-scale", type=float, default=1.0)
    p.add_argument("--muon-spectrum", choices=("mono", "guan"), default="mono",
                   help="mono: all muons at --muon-mev; guan: sea-level energy "
                        "and angle spectrum of Guan et al. 2015 (arXiv:1509.06176)")
    p.add_argument("--zenith-n", type=float, default=2.0,
                   help="hadron sources: I ~ cos^n per sr (0 = isotropic upper hemisphere)")
    p.add_argument("--physics", default=None,
                   help="Geant4 physics list; default FTFP_BERT, or QGSP_BIC_HP "
                        "for neutron/proton sources (needs the HP data sets)")
    a = p.parse_args()
    phys = a.physics or ("QGSP_BIC_HP" if a.source in ("neutron", "proton") else "FTFP_BERT")
    r = run(a.source, a.n, lid=a.layout, seed=a.seed, muon_mev=a.muon_mev,
            min_keV=a.min_kev, cryostat=a.cryostat, r_muon_mm=a.r_muon,
            cut_mm=a.cut, check_overlaps=a.check_overlaps,
            min_carrier_keV=a.min_carrier_kev, fridge_scale=a.fridge_scale,
            muon_spectrum=a.muon_spectrum, zenith_n=a.zenith_n, physics=phys)
    json.dump(r, open(a.out, "w"))
    ev = r["events"]
    nc = r["n_chips"]
    multi = sum(1 for e in ev if sum(1 for v in e[:nc] if v >= 500) >= 2)
    print(f"DONE {a.source} {a.layout} n={a.n} {r['seconds']:.1f}s "
          f"{r['seconds'] / a.n * 1e6:.1f} us/primary events={len(ev)} "
          f">=2 chips at 0.5 MeV: {multi}", flush=True)
