"""One place that knows where the chain's packaged data lives.

The cached Geant4 deposition spectra and the G4CMP kernel record live INSIDE
the package (``radchain/data``), and every path is anchored on the package
directory, never on the working directory or a repository root. Both of those
were tried and both failed, in ways that read as physics: a missing spectrum
looks exactly like a legitimate "no data" error.
"""
from __future__ import annotations

import glob
from pathlib import Path

#: This package's own directory. NOT the repository root, and NOT the working
#: directory: both were tried and both failed, in ways that read as physics.
DATA = Path(__file__).resolve().parent / "data"

#: Device 1's cached Geant4 deposition spectra, per source and thickness.
ENV = DATA / "env"

#: Device 2's OWN cache at its own thickness. Kept separate because the gamma
#: flux must be inferred from device 2's response, never by substituting a
#: device-1 thickness into device 2's rate.
ENV2 = DATA / "env2"

#: The consolidated G4CMP kernel record (`runs.jsonl`).
L3 = DATA / "l3_kernel"

RUNS = L3 / "runs.jsonl"


def env_files(source: str, thickness_mm: float) -> list[str]:
    """Device 1 spectra for one source at one thickness.

    Sharded sources are `<src>_<t>_<n>.json`; the muon runs are a single
    `<src>_<t>.json`. Both spellings are matched, because matching only the
    sharded one once silently dropped the muon channel.
    """
    stem = "{}_{:g}".format(source, thickness_mm)
    return sorted(glob.glob(str(ENV / f"{stem}_*.json")) +
                  glob.glob(str(ENV / f"{stem}.json")))


def env2_files(source: str, thickness_mm: float) -> list[str]:
    """Device 2 spectra for one source at one thickness."""
    return sorted(glob.glob(str(ENV2 / "{}_{:g}_*.json".format(
        source, thickness_mm))))
