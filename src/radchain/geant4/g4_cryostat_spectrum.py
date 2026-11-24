"""g4_cryostat_spectrum.py — species-resolved deposits with the FRIDGE around the die.

WHY
---
`g4_species_spectrum.py` fires sea-level particles at a bare die. That is fine
for muons but wrong for hadrons: NIST PSTAR ranges through a Bluefors-like
cryostat stop sky protons below ~40 MeV (side cans) to ~210 MeV (top plates),
and the slow protons that DO reach a die are made inside the fridge by neutron
interactions in the innermost copper. Fowler et al. (arXiv:2404.10866, Supp.
A2) see the same thing: their cryostat cuts the >2 MeV hadronic spectrum ~3x,
"mainly by reducing the flux of lower-energy protons".

GEOMETRY (sky frame, +z up; die centred at the origin)
--------------------------------------------------------
Thicknesses from Loer et al., arXiv:2403.01032, Table 1 and p3 (a fridge
"based approximately on" a Bluefors LD-400). Nested box shells, inside out:

    Cu sample box      5 mm walls          (Loer: 0.5 cm Cu box)
    Cu base plate      6.35 mm, under box  (Loer: 6.35 mm Cu plate)
    Al inner shield    15.9 mm             (Loer: 1.59 cm Al standing in for
                                            "copper, aluminum, and mumetal")
    4 K can            Al 1.5 mm
    still can          Cu 0.5 mm
    50 K can           Al 1.0 mm
    vacuum can (OVC)   Al 3.2 mm
    plates above, as wide slabs: MXC Cu 8, cold Cu 6, still Cu 9, 4 K Cu 10,
    50 K Al 12, top flange stainless 12 mm (Loer Table 1)
    optional building ceiling: concrete slab (Fowler used 18 cm)

APPROXIMATIONS, stated so they are not mistaken for findings
-------------------------------------------------------------
- Shells are boxes at radii of 1-6 cm, not cylinders at 10-30 cm. For
  secondaries what matters is areal density along the path and that the shell
  surrounds the die; solid-angle details of a real fridge are not represented.
- Plates are slabs 20 cm wide placed just above the OVC instead of tens of cm
  above the sample. A uniform slab wide compared with its distance gives the
  same flux below it at any distance, so moving it closer is acceptable as
  long as the generation disk does not under-cover it. Each event records the
  primary's impact radius on the generation disk so that coverage can be
  CHECKED (contribution per unit area should fall off before R_GEN).
- Neutrons are killed below `kill_n_mev` (1 MeV default): below ~1.35 MeV an
  elastic Si recoil cannot reach 180 keV, and HP transport of slow neutrons in
  copper is where the CPU goes. Capture gammas from thermalised neutrons are
  therefore NOT included. Stated, not hidden.
- Protons use the JESD89 NEUTRON energy/angle sampling and are scored per
  primary; each event stores the primary energy so the analysis can apply a
  sourced, bracketed p/n(E). Never quote a proton rate from this file alone.

Each event: [E_dep keV, u mm, v mm (in-die centroid), primary MeV, r_gen mm,
importance weight]. Rates MUST use the weights (see TIERS).
"""
from __future__ import annotations

import json
import math
import random
import time

import g4_species_spectrum as gs

R_GEN_MM = 80.0        # generation disk radius
#: Importance sampling of the impact radius on the generation disk. The die's
#: projection always lies within ~7 mm of the disk centre, so direct hits come
#: from r < 10 mm; secondaries from the inner Cu box from r < ~30 mm. Each
#: tier is (radius mm, probability), sampled uniform-in-area within the tier
#: disk; every event carries weight w = f(r)/q(r) against the true uniform
#: density on R_GEN, so the estimate is unbiased. Uniform sampling cost >1000
#: core-hours for this matrix; this cuts it ~30x.
TIERS = [(10.0, 0.5), (30.0, 0.25), (R_GEN_MM, 0.25)]


def tier_weight(r_mm: float) -> float:
    f = 1.0 / (math.pi * R_GEN_MM ** 2)
    q = sum(p / (math.pi * R * R) for R, p in TIERS if r_mm < R)
    return f / q
SLAB_HALF_MM = 100.0   # half-width of the plate slabs
BACKOFF_MM = 450.0     # start distance behind the disk

# (name, material, inner half xy, inner half z, wall mm), inside out
SHELLS = [
    ("CuBox", "G4_Cu", 8.0, 6.0, 5.0),
    ("AlShield", "G4_Al", 20.0, 25.0, 15.9),
    ("Can4K", "G4_Al", 40.0, 45.0, 1.5),
    ("CanStill", "G4_Cu", 45.0, 50.0, 0.5),
    ("Can50K", "G4_Al", 50.0, 55.0, 1.0),
    ("OVC", "G4_Al", 55.0, 60.0, 3.2),
]
# bottom to top, mm
PLATES = [("PlateMXC", "G4_Cu", 8.0), ("PlateCold", "G4_Cu", 6.0),
          ("PlateStill", "G4_Cu", 9.0), ("Plate4K", "G4_Cu", 10.0),
          ("Plate50K", "G4_Al", 12.0),
          ("PlateTop", "G4_STAINLESS-STEEL", 12.0)]
PLATE_Z0_MM = 70.0
PLATE_GAP_MM = 2.0


def _spread(m: dict, a0: float, b0: float, a1: float, b1: float,
            de: float, bin_mm: float) -> None:
    """Add `de` to map `m`, uniformly along the segment (a0,b0)->(a1,b1),
    sampled every half bin."""
    dist = math.hypot(a1 - a0, b1 - b0)
    n = max(1, int(math.ceil(dist / (0.5 * bin_mm))))
    w = de / n
    for i in range(n):
        f = (i + 0.5) / n
        k = (math.floor((a0 + f * (a1 - a0)) / bin_mm),
             math.floor((b0 + f * (b1 - b0)) / bin_mm))
        m[k] = m.get(k, 0.0) + w


def run(source: str, n_primaries: int, thickness_mm: float,
        orient: str = "flat", xy_mm: float = 10.0, seed: int = 1,
        energy_mev: float = 4000.0, zenith_n: float = 2.0,
        ceiling_cm: float = 0.0, kill_n_mev: float = 1.0,
        min_keV: float = 50.0, physics: str = "QGSP_BIC_HP",
        map_bin_mm: float = 0.0, map_min_keV: float = 180.0) -> dict:
    import geant4_pybind as g4
    from geant4_pybind import (G4VUserDetectorConstruction, G4NistManager,
                               G4Box, G4LogicalVolume, G4PVPlacement,
                               G4RotationMatrix, G4RunManagerFactory,
                               G4RunManagerType, G4UImanager, G4ParticleGun,
                               G4ParticleTable, G4ThreeVector, mm, keV, MeV,
                               deg, G4VUserPrimaryGeneratorAction,
                               G4VUserActionInitialization,
                               G4UserSteppingAction, G4UserEventAction,
                               G4TrackStatus)

    if source not in ("muon", "neutron", "proton"):
        raise ValueError(source)
    if orient not in ("flat", "vertical"):
        raise ValueError(orient)
    n_ang = 2.0 if source == "muon" else zenith_n
    rng = random.Random(seed)
    esample = gs._energy_sampler() if source != "muon" else None
    keep = []   # keep pybind objects alive

    class Geo(G4VUserDetectorConstruction):
        def Construct(self):
            nist = G4NistManager.Instance()
            vac = nist.FindOrBuildMaterial("G4_Galactic")
            W = BACKOFF_MM + R_GEN_MM + 50.0
            world_l = G4LogicalVolume(G4Box("World", W * mm, W * mm, W * mm),
                                      vac, "World")
            world_p = G4PVPlacement(None, G4ThreeVector(), world_l, "World",
                                    None, False, 0)
            keep.extend([world_l, world_p])
            mother = world_l
            # outermost shell first
            for name, mat, hxy, hz, t in reversed(SHELLS):
                outer = G4LogicalVolume(
                    G4Box(name, (hxy + t) * mm, (hxy + t) * mm, (hz + t) * mm),
                    nist.FindOrBuildMaterial(mat), name)
                pv = G4PVPlacement(None, G4ThreeVector(), outer, name, mother,
                                   False, 0)
                cav = G4LogicalVolume(G4Box(name + "Cav", hxy * mm, hxy * mm,
                                            hz * mm), vac, name + "Cav")
                pv2 = G4PVPlacement(None, G4ThreeVector(), cav, name + "Cav",
                                    outer, False, 0)
                keep.extend([outer, pv, cav, pv2])
                if name == "AlShield":
                    # Cu base plate under the sample box, inside the Al shield
                    box = SHELLS[0]
                    zt = -(box[3] + box[4])
                    plate = G4LogicalVolume(
                        G4Box("BasePlate", hxy * mm, hxy * mm, 6.35 / 2 * mm),
                        nist.FindOrBuildMaterial("G4_Cu"), "BasePlate")
                    pvp = G4PVPlacement(None, G4ThreeVector(0, 0, (zt - 6.35 / 2) * mm),
                                        plate, "BasePlate", cav, False, 0)
                    keep.extend([plate, pvp])
                mother = cav
            die = G4LogicalVolume(
                G4Box("Die", xy_mm / 2 * mm, xy_mm / 2 * mm,
                      thickness_mm / 2 * mm),
                nist.FindOrBuildMaterial("G4_Si"), "Die")
            rot = None
            if orient == "vertical":
                rot = G4RotationMatrix()
                rot.rotateY(90 * deg)
                keep.append(rot)
            pvd = G4PVPlacement(rot, G4ThreeVector(), die, "Die", mother,
                                False, 0)
            keep.extend([die, pvd])
            z = PLATE_Z0_MM
            slabs = list(PLATES)
            if ceiling_cm > 0:
                slabs.append(("Ceiling", "G4_CONCRETE", ceiling_cm * 10.0))
            for name, mat, t in slabs:
                sl = G4LogicalVolume(
                    G4Box(name, SLAB_HALF_MM * mm, SLAB_HALF_MM * mm, t / 2 * mm),
                    nist.FindOrBuildMaterial(mat), name)
                pvs = G4PVPlacement(None, G4ThreeVector(0, 0, (z + t / 2) * mm),
                                    sl, name, world_l, False, 0)
                keep.extend([sl, pvs])
                z += t + PLATE_GAP_MM
            return world_p

    class Gun(G4VUserPrimaryGeneratorAction):
        def __init__(self):
            super().__init__()
            self.gun = G4ParticleGun(1)
            nm = {"muon": "mu-", "neutron": "neutron", "proton": "proton"}[source]
            self.gun.SetParticleDefinition(
                G4ParticleTable.GetParticleTable().FindParticle(nm))
            self.e = self.r = 0.0

        def GeneratePrimaries(self, event):
            ct = rng.random() ** (1.0 / (n_ang + 1.0))
            st = math.sqrt(max(0.0, 1.0 - ct * ct))
            a = rng.uniform(0.0, 2.0 * math.pi)
            d = (st * math.cos(a), st * math.sin(a), -ct)
            t = (1.0, 0.0, 0.0) if abs(d[0]) < 0.9 else (0.0, 1.0, 0.0)
            u = (d[1] * t[2] - d[2] * t[1], d[2] * t[0] - d[0] * t[2],
                 d[0] * t[1] - d[1] * t[0])
            un = math.sqrt(sum(c * c for c in u))
            u = tuple(c / un for c in u)
            v = (d[1] * u[2] - d[2] * u[1], d[2] * u[0] - d[0] * u[2],
                 d[0] * u[1] - d[1] * u[0])
            x, acc = rng.random(), 0.0
            for Rt, pt in TIERS:
                acc += pt
                if x <= acc:
                    break
            r = Rt * math.sqrt(rng.random())
            ph = rng.uniform(0.0, 2.0 * math.pi)
            p = tuple(r * (math.cos(ph) * u[i] + math.sin(ph) * v[i])
                      - BACKOFF_MM * d[i] for i in range(3))
            e = energy_mev if source == "muon" else esample(rng.random())
            self.e, self.r = e, r
            self.gun.SetParticleEnergy(e * MeV)
            self.gun.SetParticlePosition(G4ThreeVector(*[c * mm for c in p]))
            self.gun.SetParticleMomentumDirection(G4ThreeVector(*d))
            self.gun.GeneratePrimaryVertex(event)

    events: list = []
    counts = {"interacting": 0, "n_killed": 0}
    kill_e = kill_n_mev * MeV
    kill = G4TrackStatus.fStopAndKill

    maps: list = []    # parallel to the events with E >= map_min_keV

    class Step(G4UserSteppingAction):
        def __init__(self):
            super().__init__()
            self.e = self.eu = self.ev = 0.0
            self.m = {}

        def UserSteppingAction(self, step):
            trk = step.GetTrack()
            if (trk.GetDefinition().GetPDGEncoding() == 2112
                    and trk.GetKineticEnergy() < kill_e):
                trk.SetTrackStatus(kill)
                counts["n_killed"] += 1
                return
            vol = step.GetPreStepPoint().GetTouchable().GetVolume()
            if vol is not None and vol.GetName() == "Die":
                de = step.GetTotalEnergyDeposit() / keV
                if de > 0.0:
                    q = step.GetPostStepPoint().GetPosition()
                    if orient == "flat":
                        a, b = q.x / mm, q.y / mm
                    else:
                        a, b = q.z / mm, q.y / mm
                    self.e += de
                    self.eu += de * a
                    self.ev += de * b
                    if map_bin_mm > 0:
                        # A charged track often crosses the thin die in ONE
                        # step; spread its loss along the segment, or a 6 mm
                        # muon track lands in a single bin.
                        q0 = step.GetPreStepPoint().GetPosition()
                        if orient == "flat":
                            a0, b0 = q0.x / mm, q0.y / mm
                        else:
                            a0, b0 = q0.z / mm, q0.y / mm
                        _spread(self.m, a0, b0, a, b, de, map_bin_mm)

    class Ev(G4UserEventAction):
        def __init__(self, sc, gun):
            super().__init__()
            self.sc, self.gun = sc, gun

        def BeginOfEventAction(self, event):
            self.sc.e = self.sc.eu = self.sc.ev = 0.0
            self.sc.m = {}

        def EndOfEventAction(self, event):
            e = self.sc.e
            if e > 0.0:
                counts["interacting"] += 1
                if map_bin_mm > 0 and e >= map_min_keV:
                    maps.append([len(events)] + [
                        [i, j, round(v, 2)] for (i, j), v in self.sc.m.items()])
                if e >= min_keV:
                    events.append([round(e, 3), round(self.sc.eu / e, 3),
                                   round(self.sc.ev / e, 3),
                                   round(self.gun.e, 3), round(self.gun.r, 2),
                                   tier_weight(self.gun.r)])

    rm = G4RunManagerFactory.CreateRunManager(G4RunManagerType.Serial)
    rm.SetUserInitialization(Geo())
    rm.SetUserInitialization(getattr(g4, physics)())

    class Actions(G4VUserActionInitialization):
        def Build(self):
            gun = Gun()
            self.SetUserAction(gun)
            sc = Step()
            self.SetUserAction(sc)
            self.SetUserAction(Ev(sc, gun))

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
        j_h = gs.MUON_J_HORIZONTAL_PER_CM2_S
    else:
        j_h = gs.jesd89_integral_per_cm2_h(1.0) / 3600.0
    omega_int = j_h * gs.angular_factor(n_ang)
    gen_area_cm2 = math.pi * (R_GEN_MM / 10.0) ** 2
    return {"source": source, "orient": orient, "thickness_mm": thickness_mm,
            "xy_mm": xy_mm, "physics": physics, "geometry": "cryostat-v1",
            "ceiling_cm": ceiling_cm, "kill_n_mev": kill_n_mev,
            "energy_mev": energy_mev if source == "muon" else "jesd89",
            "zenith_n": n_ang, "n_primaries": n_primaries, "seed": seed,
            "seconds": sec, "min_keV": min_keV, "r_gen_mm": R_GEN_MM,
            "interacting": counts["interacting"],
            "neutrons_killed": counts["n_killed"],
            "rate_per_primary_s": gen_area_cm2 * omega_int,
            "gen_area_cm2": gen_area_cm2,
            "tiers": TIERS,
            "rate_note": "rate = rate_per_primary_s * sum(weight) / n_primaries",
            "event_fields": ["E_keV", "u_mm", "v_mm", "primary_MeV", "r_gen_mm",
                             "weight"],
            "map_bin_mm": map_bin_mm, "map_min_keV": map_min_keV,
            "maps_note": ("each map = [event index, [i, j, keV], ...]; bin (i, j) "
                          "covers [i, i+1) x [j, j+1) * map_bin_mm in the die "
                          "plane (u, v)"),
            "maps": maps,
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
    ap.add_argument("--ceiling-cm", type=float, default=0.0)
    ap.add_argument("--kill-n-mev", type=float, default=1.0)
    ap.add_argument("--physics", default="QGSP_BIC_HP")
    ap.add_argument("--map-bin-mm", type=float, default=0.0)
    a = ap.parse_args()
    r = run(a.source, a.n, a.thickness_mm, orient=a.orient, seed=a.seed,
            energy_mev=a.energy_mev, zenith_n=a.zenith_n,
            ceiling_cm=a.ceiling_cm, kill_n_mev=a.kill_n_mev,
            physics=a.physics, map_bin_mm=a.map_bin_mm)
    json.dump(r, open(a.out, "w"))
    ev = r["events"]
    k1 = sum(1 for e in ev if e[0] >= 1000.0)
    k018 = sum(1 for e in ev if e[0] >= 180.0)
    w1 = sum(e[5] for e in ev if e[0] >= 1000.0)
    rp = r["rate_per_primary_s"] * 3600.0 * w1 / max(k1, 1) if k1 else 0.0
    print(f"DONE cryo {a.source} {a.orient} t={a.thickness_mm} "
          f"ceil={a.ceiling_cm} n={a.n} {r['seconds']:.1f}s "
          f"{r['seconds'] / a.n * 1e6:.1f} us/primary "
          f"interacting={r['interacting']} >=180keV={k018} >=1MeV={k1} "
          f"rate>=1MeV={rp * k1 / a.n:.3g}/h", flush=True)
