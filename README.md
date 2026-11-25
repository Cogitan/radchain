# radchain: from a particle strike to logical errors in superconducting qubit chips

`radchain` is an open, end-to-end simulation of how ionising radiation turns into
correlated errors on a superconducting quantum processor:

```
 environment / cosmic rays          Geant4             energy deposited per event
            │                                          per die (keV, position)
            ▼
 deposit ──► phonon transport       G4CMP              where pair-breaking lands
            │                                          around the strike
            ▼
 quasiparticles ──► qubit shift     analytic           per-qubit frequency shift,
            │       and recovery    (published eqs.)   recovery over time
            ▼
 burst errors ──► decoder           Stim + PyMatching  logical failure per event,
                                                       erasure / code-block analysis
```

Every stage is physics you can run and change. Nothing in the default chain is
fitted to a target except where it says so, and every place it is says what it
was fitted to.

## What is in here

| Stage | Code | Needs |
|---|---|---|
| Radiation transport: cryostat, die stacks, chip arrays, carriers, muon/gamma/species sources | `src/radchain/geant4/g4_*.py` | `geant4_pybind` |
| Cached deposition spectra (Si, sapphire; several thicknesses) and thickness interpolation | `src/radchain/data/env*`, `env_grid.py` | none |
| Phonon transport: G4CMP build, run-time-configurable trap geometry patch, job script, kernel scorer | `g4cmp/`, `score_kernel.py`, `ingest_kernel_runs.py` | Geant4 11.4.1 + G4CMP |
| 3,821 recorded G4CMP runs: 3,681 in the 455 configurations the backend loads (450 trap configurations, thickness × back-side trap pitch × coverage × absorption, plus 5 no-trap thicknesses), and 140 placement and sensitivity runs | `src/radchain/data/l3_kernel/runs.jsonl` | none |
| Phonon backend interface + the G4CMP-kernel backend | `src/radchain/phonon.py` | none |
| Per-qubit absorbed energy on any layout | `qubit_footprint.py` | none |
| Quasiparticle (QP) frequency shift per device (Eq. 17 of arXiv:2506.18228; Catelani et al., PRB 84, 064517 (2011)) | `qp_shift.py` | none |
| Recovery: recombination (no traps) and trapping regimes | `recovery_google2025.py`, `qp_recovery.py` | none |
| Burst rate cross-checks, second-device test (Li et al., Nat. Commun. 16, 4677 (2025), arXiv:2402.04245) | `rate_crosscheck.py`, `device2_li2025.py` | none |
| Logical failure per event: surface-code memory under the burst, decoded | `analysis/rad_pfail_physics.py` | `stim`, `pymatching` |
| Cross-chip erasure, code blocks, erasure policies, bootstrap intervals | `analysis/rad_*.py` | per-event deposit data and `rad_pfail_*` results (below) |
| The sweep launchers that produced the data | `scripts/sweeps/` | WSL/Linux, Geant4 |

## Quick start

```bash
pip install "radchain[analysis]"
```

or, to run the tests, the analysis scripts and the sweeps, from a clone:

```bash
git clone https://github.com/Cogitan/radchain && cd radchain
pip install -e ".[analysis,test]"
pytest
python -m radchain.qubit_footprint          # Willow-class footprint on G4CMP kernels (~5 min)
python analysis/rad_pfail_physics.py --d 5 --events 50 --shots 500 --workers 4   # ~6 min
```

In the footprint table, the last column names the fixed band(s) a row fails
("size", "large", "both"), or PASS. The bands are set in `qubit_footprint.py`; the
record of when they were fixed is not part of this release.

Phonon hits for any simulated configuration:

```python
from radchain import phonon
b = phonon.get_backend()                     # "g4cmp-kernel"
hx, hy = b.film_hits(n_phonons=200_000, thickness_mm=0.381)
hx, hy = b.film_hits(n_phonons=200_000, thickness_mm=0.6,
                     trap_pitch_mm=0.5, trap_coverage=0.25, trap_absorption=1.0)
```

To simulate a geometry that is not in the record, build G4CMP with the patch
(`g4cmp/README.md`), run `g4cmp/l3_kernel_job.sh` (by default with
`g4cmp/base.mac`, the G4CMP macro the shipped record used; see the script header), and
add the result with `python -m radchain.ingest_kernel_runs --src <dir>`. That
writes into the package's own `data/l3_kernel/runs.jsonl`, so use an editable
install. The backend loads only the `g_*` (trap grid) and `kt_*` (no-trap) tag
families. A run you add must use the same G4CMP settings as its family (die
size, housing, film pattern; see `scripts/sweeps/run_trap_grid*.sh` and
`run_kernel_thickness.sh`), because those settings are not recorded in a run.

## Phonon backends

Downstream stages only need, for a strike at the origin, the positions where
pair-breaking energy lands (`phonon.py` documents the contract). The shipped
backend draws them from the recorded G4CMP kernels (binned in radius and
averaged over azimuth), with no interpolation, and it refuses configurations
that were not simulated. In the shipped record G4CMP took 0.4-3 s per primary
phonon with back-side traps and up to about 2 minutes without them. The
record's `pair_breaks` fields count quasiparticle births, two per broken pair;
the backend charges each one the film gap.

Other transport models can be registered with `phonon.register_backend(...)` or
the `radchain.phonon_backends` entry-point group, and every stage then uses them
by name. Cogitan's own transport model plugs in through this same interface;
it is not part of this repository.

## Known limits, stated before you find them

- **The trap grid was simulated on a 4 mm die** with a plain film and no
  housing, so those kernels stop near 2 mm from the strike. The no-trap kernels
  use G4CMP's example geometry (8 mm die with its housing). All five were run
  on the patched binary; at 0.381 mm its default settings give the example's
  own thickness.
  Neither is a production qubit die.
- **The no-trap kernels do not reproduce Willow's burst-size distribution**
  (arXiv:2506.18228). `python -m radchain.qubit_footprint` grades four of the
  five thicknesses and shows median sizes of 4 to 20 of 60 qubits across
  thickness and spectrum, against 15 measured, and
  no row passes both fixed bands. The example geometry differs from
  Willow's (no carrier chip, a different housing). The grade is printed rather
  than tuned until it passes.
- **The response coefficient is tied to the transport that fitted it.**
  `qp_shift.K_WILLOW_KHZ_PER_KEV` (44.89 kHz/keV) is fitted to Willow's 2 MHz
  median peak with the shipped backend on the no-trap 0.381 mm kernel, using
  Geant4 gamma-line deposits in a 0.3 mm die (8 seeds of 3000 strikes; see
  `qp_shift.py`). It is the device's response and stays fixed when you change
  the design (traps, thickness). Refit it only when you change the transport
  model: pass `qubit_footprint.fit_k` the per-event absorbed energies from that
  model's no-trap reference, never from a trapped design, or you scale the traps'
  effect back out. `rad_pfail_physics.py --fit-k-reference` also refits it, but
  with its own deposit mix (gamma lines plus 18.4% muons, 0.5 mm die), so it does
  not return 44.89 on the shipped backend.
- **Willow's response does not transfer as-is to other devices.** In Cogitan's
  own tests, made with its own transport model and not part of this release, a
  Willow-calibrated response over-predicted the frequency shifts on Pinckney et
  al.'s JJ-Only array (arXiv:2603.13460, Fig. 13(b): below 100 kHz at deposits
  of 57 to 699 keV) by 16 to 80 times, depending on energy and statistic. The cause is unresolved, and nothing here shows the
  shipped backend transfers either.
- The quasiparticle-birth kernel is axisymmetric (azimuth-averaged) and taken at
  one strike depth and primary phonon energy. A few runs (`s3_*`, `s4_*` tags)
  probe sensitivity.
- Births within 30 um of the strike are recorded by the scorer only as a count;
  the backend places them uniformly within 30 um. They are 0.04-3.7% of births
  in the no-trap kernels (the most at 0.1 mm) and up to ~18% in thin trapped
  ones.
- The no-trap kernels include reflection at the simulated 8 mm die's edge, so
  for strikes near the edge of a larger die the outer geometry is approximate.
- Some docstrings mention pre-registered tests and earlier file locations from
  the research history. Those records are not part of this release; the paper
  below covers the published results.

## Data not in git

The per-event Geant4 deposit files used by the erasure and code-block analysis
(about 1.3 GB, `bench/rad_data/` in the data record of the paper below) are not
in git. Copy them to `analysis/rad_data/` for the analysis scripts. The sweep
scripts take `RADCHAIN_REPO` (this repository), `RADCHAIN_DATA` (where to write
deposits) and `RADCHAIN_PY` (a Python with `geant4_pybind`).

`rad_apply_perase.py` and the scripts built on it (`rad_code_block`,
`rad_erasure_policy`, `rad_policy_threshold`, `rad_block_bootstrap`) also read
`analysis/results/rad_pfail_<tag>.json` for tags `d7_conv0.25_echo`,
`d7_conv0.5_echo`, `d7_conv1.0_echo` and `d7_conv0.5_noecho` (plus d5/d11 for
`--distance`). Produce them with `rad_pfail_physics.py --d 7 --conv ... [--noecho]`
(hours each at the default 150 events x 1000 shots), or copy the paper's from
`bench/results/` in the data record. The paper's were made with Cogitan's own
transport model, which is not in this repository, so they will not match runs
on the shipped backend.

## Papers and data

- D. Henderson, *Correlated radiation erasures limit cross-chip erasure codes in
  modular superconducting processors: a simulation study* (2026).
  doi:10.5281/zenodo.24412996
- Data and code: doi:10.5281/zenodo.24412856

## Status

1.0.0. The interfaces above (`phonon.get_backend`, `film_hits`,
`register_backend`, the `radchain.phonon_backends` entry point) are kept stable
within 1.x. The limits listed above are open questions, not bugs scheduled for a
fix.

## License

- Code: Apache-2.0 (`LICENSE`).
- `g4cmp/l3trap.patch` and `g4cmp/base.mac` modify files from G4CMP's
  `examples/quasiparticle` and are distributed under GPL-3.0, the license of
  G4CMP (`g4cmp/NOTICE`, full text in `g4cmp/COPYING`). Neither is in the Python
  package. The rest of this repository reads G4CMP's output files and does not
  link against it.
- Data in `src/radchain/data/`: CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/); credit "Cogitan, radchain"
  (`NOTICE`).

G4CMP: M.H. Kelsey et al., NIM A 1055, 168473 (2023).
