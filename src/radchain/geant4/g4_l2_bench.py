"""g4_l2_bench.py — L2 transport: cost measurement, and the first deposition map.

Two jobs:

1.  **Measure what L2 actually costs** per primary, rather than assume it is
    cheap next to L3 (phonon transport).

2.  **Produce the deposition map.** The L1/L2 interface is the
    deposition map: where energy was left in the substrate, and how much, plus
    the substrate geometry and material. Building it against a live solver is
    the only way to find out whether the contract as written is usable.

THE L2/L3 BOUNDARY IS ENFORCED HERE BY CONSTRUCTION. Nothing in this file knows what is
patterned on top of the substrate: no gap, no film stack, no quasiparticles, no
traps, no superconductivity. A Geant4 deposition calculation in silicon does not
care.

RUNS IN WSL, NOT ON WINDOWS. `geant4_pybind` is installed in the WSL venv at
``~/rad/.venv`` and needs X11 shared libraries extracted locally (see
``radchain/g4cmp/README.md``). This module is import-safe on Windows: the Geant4
import is inside the functions, so the repo's test suite can import it and skip.

    wsl -e bash -lc "cd ~/rad && LD_LIBRARY_PATH=... .venv/bin/python g4_l2_bench.py"
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict

# --------------------------------------------------------------------------- #
# The deposition map
# --------------------------------------------------------------------------- #

@dataclass
class DepositionEvent:
    """One energy deposit in the substrate. Positions in mm, energy in keV."""
    x_mm: float
    y_mm: float
    z_mm: float
    e_keV: float


@dataclass
class DepositionMap:
    """The L1/L2 interface artifact. THE BOUNDARY: nothing above L2 appears
    here, and nothing that reads this may need to know the device is
    superconducting.
    """
    substrate_material: str
    substrate_x_mm: float
    substrate_y_mm: float
    substrate_thickness_mm: float
    primary: str
    primary_energy_MeV: float
    n_primaries: int
    events: list = field(default_factory=list)
    #: Deposited energy per PRIMARY, keV. This is the attribution observable:
    #: a burst threshold is a cut on this distribution, not on the total.
    per_primary_keV: list = field(default_factory=list)
    total_edep_keV: float = 0.0
    #: Wall-clock seconds for the transport run. MEASURED, never declared.
    seconds: float = 0.0

    @property
    def seconds_per_primary(self) -> float:
        return self.seconds / self.n_primaries if self.n_primaries else 0.0

    def to_json(self, path: str) -> str:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh)
        return path


# --------------------------------------------------------------------------- #
# Geant4 pieces. Imported lazily so this file is importable without Geant4.
# --------------------------------------------------------------------------- #

def _g4_classes():
    from geant4_pybind import (G4VUserDetectorConstruction, G4NistManager,
                               G4Box, G4LogicalVolume, G4PVPlacement,
                               G4VUserPrimaryGeneratorAction, G4ParticleGun,
                               G4ParticleTable, G4ThreeVector,
                               G4UserSteppingAction, G4UserEventAction,
                               mm, MeV, keV)

    class Slab(G4VUserDetectorConstruction):
        """A bare silicon (or sapphire) die on nothing. No circuit, by design."""

        def __init__(self, xy_mm, thickness_mm, material):
            super().__init__()
            self.xy_mm = xy_mm
            self.thickness_mm = thickness_mm
            self.material = material

        def Construct(self):
            nist = G4NistManager.Instance()
            sub = nist.FindOrBuildMaterial(self.material)
            vac = nist.FindOrBuildMaterial("G4_Galactic")

            # World generously larger so primaries start outside the die.
            wx = self.xy_mm * 4 * mm
            wz = max(self.thickness_mm * 40, 40.0) * mm
            world_s = G4Box("World", wx / 2, wx / 2, wz / 2)
            world_l = G4LogicalVolume(world_s, vac, "World")
            world_p = G4PVPlacement(None, G4ThreeVector(), world_l, "World",
                                    None, False, 0)

            die_s = G4Box("Die", self.xy_mm / 2 * mm, self.xy_mm / 2 * mm,
                          self.thickness_mm / 2 * mm)
            die_l = G4LogicalVolume(die_s, sub, "Die")
            G4PVPlacement(None, G4ThreeVector(), die_l, "Die", world_l,
                          False, 0)
            return world_p

    class Gun(G4VUserPrimaryGeneratorAction):
        """One primary, fired at the die along -z from above."""

        def __init__(self, particle, energy_MeV, z_start_mm):
            super().__init__()
            self.gun = G4ParticleGun(1)
            pt = G4ParticleTable.GetParticleTable().FindParticle(particle)
            self.gun.SetParticleDefinition(pt)
            self.gun.SetParticleEnergy(energy_MeV * MeV)
            self.gun.SetParticleMomentumDirection(G4ThreeVector(0, 0, -1))
            self.z_start_mm = z_start_mm

        def GeneratePrimaries(self, event):
            self.gun.SetParticlePosition(
                G4ThreeVector(0, 0, self.z_start_mm * mm))
            self.gun.GeneratePrimaryVertex(event)

    class EventCounter(G4UserEventAction):
        """Snapshot deposited energy per primary. Attribution needs the
        DISTRIBUTION, because an observed burst rate is a cut on it."""

        def __init__(self, dep_map, scorer):
            super().__init__()
            self.map = dep_map
            self.scorer = scorer

        def BeginOfEventAction(self, event):
            self.scorer.event_keV = 0.0

        def EndOfEventAction(self, event):
            self.map.per_primary_keV.append(self.scorer.event_keV)

    class Scorer(G4UserSteppingAction):
        """Accumulate the deposition map. Records position and energy only.

        This is the whole of what crosses the L2/L3 boundary. If a future
        change makes this want to know about films or gaps, the boundary has
        been broken.
        """

        def __init__(self, dep_map, keep_events=True):
            super().__init__()
            self.map = dep_map
            self.keep_events = keep_events
            self.event_keV = 0.0

        def UserSteppingAction(self, step):
            vol = step.GetPreStepPoint().GetTouchable().GetVolume()
            if vol is None or vol.GetName() != "Die":
                return
            e = step.GetTotalEnergyDeposit() / keV
            if e <= 0.0:
                return
            self.map.total_edep_keV += e
            self.event_keV += e
            if self.keep_events:
                p = step.GetPreStepPoint().GetPosition()
                self.map.events.append(
                    DepositionEvent(p.x / mm, p.y / mm, p.z / mm, e))

    return Slab, Gun, Scorer, EventCounter


def run(particle: str = "mu-",
        energy_MeV: float = 4000.0,
        n_primaries: int = 100,
        xy_mm: float = 10.0,
        thickness_mm: float = 0.3,
        material: str = "G4_Si",
        keep_events: bool = True,
        verbose: bool = False) -> DepositionMap:
    """Transport `n_primaries` through the die and return the deposition map.

    Defaults: a 10 x 10 mm die, 300 um of silicon, and a 4 GeV muon, which is
    roughly the mean energy of the sea-level atmospheric muon flux.
    """
    from geant4_pybind import (G4RunManagerFactory, G4RunManagerType,
                               FTFP_BERT, G4UImanager)

    Slab, Gun, Scorer, EventCounter = _g4_classes()

    dep = DepositionMap(
        substrate_material=material, substrate_x_mm=xy_mm,
        substrate_y_mm=xy_mm, substrate_thickness_mm=thickness_mm,
        primary=particle, primary_energy_MeV=energy_MeV,
        n_primaries=n_primaries)

    rm = G4RunManagerFactory.CreateRunManager(G4RunManagerType.Serial)
    rm.SetUserInitialization(Slab(xy_mm, thickness_mm, material))
    rm.SetUserInitialization(FTFP_BERT())

    from geant4_pybind import G4VUserActionInitialization

    class Actions(G4VUserActionInitialization):
        def Build(self):
            self.SetUserAction(Gun(particle, energy_MeV,
                                   z_start_mm=thickness_mm * 10 + 5.0))
            sc = Scorer(dep, keep_events)
            self.SetUserAction(sc)
            self.SetUserAction(EventCounter(dep, sc))

    rm.SetUserInitialization(Actions())
    rm.Initialize()

    ui = G4UImanager.GetUIpointer()
    if not verbose:
        for cmd in ("/run/verbose 0", "/event/verbose 0", "/tracking/verbose 0",
                    "/run/printProgress 0"):
            ui.ApplyCommand(cmd)

    t0 = time.perf_counter()
    rm.BeamOn(n_primaries)
    dep.seconds = time.perf_counter() - t0
    return dep


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="L2 transport cost measurement")
    ap.add_argument("--particle", default="mu-")
    ap.add_argument("--energy-MeV", type=float, default=4000.0)
    ap.add_argument("-n", "--n-primaries", type=int, default=100)
    ap.add_argument("--thickness-mm", type=float, default=0.3)
    ap.add_argument("--xy-mm", type=float, default=10.0)
    ap.add_argument("--material", default="G4_Si")
    ap.add_argument("--no-events", action="store_true",
                    help="do not retain the per-step map (cost of scoring)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dep = run(particle=args.particle, energy_MeV=args.energy_MeV,
              n_primaries=args.n_primaries, xy_mm=args.xy_mm,
              thickness_mm=args.thickness_mm, material=args.material,
              keep_events=not args.no_events)

    print()
    print("=" * 70)
    print("L2 TRANSPORT — MEASURED")
    print("=" * 70)
    print(f"  die              {dep.substrate_x_mm} x {dep.substrate_y_mm} mm, "
          f"{dep.substrate_thickness_mm} mm {dep.substrate_material}")
    print(f"  primary          {dep.primary} @ {dep.primary_energy_MeV} MeV")
    print(f"  primaries        {dep.n_primaries}")
    print(f"  deposits         {len(dep.events)}")
    print(f"  total E dep      {dep.total_edep_keV:.1f} keV")
    print(f"  mean per primary {dep.total_edep_keV / dep.n_primaries:.1f} keV")
    print()
    print(f"  WALL TIME        {dep.seconds:.3f} s")
    print(f"  per primary      {dep.seconds_per_primary * 1e3:.3f} ms")
    print()
    if args.out:
        print("  wrote", dep.to_json(args.out))


if __name__ == "__main__":
    _main()
