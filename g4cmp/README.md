# The G4CMP reference solver: what it is, and how to rebuild it

G4CMP is the reference phonon and quasiparticle solver for this chain. This
file describes the patch that makes the trap sweeps possible and how to build
it.

## Upstream

| | |
|---|---|
| G4CMP | `g4cmp-V10-05-00`, installed to `$HOME/opt/g4cmp-1141` |
| Geant4 | **11.4.1 only** |
| example | `examples/quasiparticle`, from `$HOME/opt/src/G4CMP-1141` |

**Geant4 11.4.2 segfaults before tracking a particle.**
`G4CMPConfigManager::setPhysicsModelID()` guards on
`#elif G4VERSION_NUMBER < 1142`, so 11.4.2 is the first release to take the
`#else` branch and it does not work: `G4Exception TRACK0982, Process/model ID
<-1> is invalid`, core dump at the first track. The example's own README
specifies 11.4.1. G4CMP's top-level README (Geant4 10 through 10.7) predates
its `ChangeHistory`, which shows development against Geant4 11; check
`ChangeHistory` when choosing a version.

## l3trap.patch

The example's geometry is compile-time. `l3trap.patch` makes the parts RAD
needs to sweep readable from the environment at RUN time, which is what turned
a recompile-per-configuration study into a sweepable grid (the shipped record
holds 450 trap configurations).

Two files, four hunks:

- `include/QuasiparticleDetectorParameters.hh`: the parameter block becomes
  `const` values computed by getenv-backed inline accessors.
- `src/QuasiparticleDetectorConstruction.cc`: back-side trap islands and the
  copper housing are built according to those values.

| variable | effect |
|---|---|
| `RAD_CHIP_MM` | die lateral size |
| `RAD_THICK_MM` | substrate thickness |
| `RAD_PLAIN` | plain film instead of the resonator pattern |
| `RAD_TRAP_PITCH_MM` | island spacing (0 = stock geometry, bit-for-bit) |
| `RAD_TRAP_FRAC` | fraction of the back face covered |
| `RAD_TRAP_ABS` | phonon absorption probability per encounter |
| `RAD_TRAP_THICK_UM` | island thickness |
| `RAD_NO_HOUSING` | omit the copper housing |

**What depends on this patch.** The placement arms, the whole trap grid and the no-trap thickness series (`RAD_THICK_MM`; all five thicknesses were run on the patched binary) in `src/radchain/data/l3_kernel/runs.jsonl`,
which the `g4cmp-kernel` phonon backend draws from.

### Apply and build

```bash
# VERIFIED clean-room rebuild, 2026-09-29. Every line matters; see below.
D=$HOME/rad/l3trap
cp -r $HOME/opt/src/G4CMP-1141/examples/quasiparticle "$D"
cd "$D"
patch -p1 --dry-run < <repo>/g4cmp/l3trap.patch   # check first
patch -p1          < <repo>/g4cmp/l3trap.patch

export G4CMPINSTALL=$HOME/opt/g4cmp-1141/share/G4CMP   # NOT the install prefix
source $HOME/opt/geant4-11.4.1/bin/geant4.sh           # required for Geant4_DIR

mkdir -p build && cd build && cmake .. && make -j8
```

**The two things that make this fail, both found by doing it:**

1. **`G4CMPINSTALL` is the `share/G4CMP` subdirectory, not the install
   prefix.** The example's `CMakeLists.txt` does
   `include($ENV{G4CMPINSTALL}/FindGeant4.cmake)` on line 14 and
   `set(CMAKE_PREFIX_PATH $ENV{G4CMPINSTALL}/../..)` on line 7, so it has to be
   the directory holding `FindGeant4.cmake` with the install prefix two levels
   up. Pointing it at `$HOME/opt/g4cmp-1141` fails at line 14. Earlier notes
   said "set `G4CMPINSTALL` to the 1141 install", which is two levels off.
2. **`geant4.sh` must be sourced.** `CMAKE_PREFIX_PATH` from step 1 points into
   the G4CMP prefix, which contains no Geant4, so `find_package(Geant4)` fails
   with "Add the installation prefix of Geant4 to CMAKE_PREFIX_PATH". Sourcing
   it is what supplies `Geant4_DIR`, and `Geant4_USE_FILE` on line 23 is empty
   without it. Passing `-DG4CMP_DIR` and `-DGeant4_DIR` on the command line is
   NOT sufficient on its own, because line 14 still reads the env var.

`G4CMPINSTALL` is load-bearing. A build against `$HOME/opt/g4cmp` links a
`libG4cmp.so` that crashes at event start, and if both prefixes exist it is
easy to get wrong. `l3_kernel_job.sh` now refuses
rather than running such a binary.

### Verify the build is the one you think it is

```bash
strings build/libg4cmpQuasiparticle.so | grep -c RAD_TRAP_PITCH_MM   # want >= 1
ldd  build/g4cmpQuasiparticle | grep libG4cmp                        # want -1141
RAD_TRAP_PITCH_MM=0 ./build/g4cmpQuasiparticle <macro>               # bit-for-bit stock
```

The stock example's `libg4cmpQuasiparticle.so` has **zero** occurrences of
`RAD_TRAP_PITCH_MM`. That is the check `l3_kernel_job.sh` runs, because a
binary without the knobs does not complain: it returns stock geometry, so a
sweep over pitch produces identical kernels and reads like a physics result.

## Guards in `l3_kernel_job.sh`

| exit | condition |
|---|---|
| 3 | a `RAD_*` geometry variable is set but the detector library has no knobs |
| 4 | `libG4cmp.so` resolves outside the `-1141` install |
| 5 | no G4CMP macro at `RAD_BASE_MAC` (default: `g4cmp/base.mac`) |

The ldd check runs **after** the script exports `LD_LIBRARY_PATH`, so it tests
what will actually be loaded rather than what the binary was linked against.
An env override that fixes a bad link is fine; a bad link with no override is
not.

`RAD_EXE` overrides the binary, defaulting to
`$HOME/rad/l3trap/build/g4cmpQuasiparticle`.

## Other traps, recorded so they are not rediscovered

- **`G4LATTICEDATA` must be exported by hand** or the binary aborts with
  `Unable to open Si/config.txt`. `g4cmp_env.sh` is not reliable here, and this
  was hit on both 09-15 and 09-19.
- **`pkill -f <path>` kills its own shell** when the pattern appears in the
  shell's own command line. It killed two running jobs.
- **`ulimit -v` is not a usable guard for Geant4.** It reserves a large virtual
  address space with a small resident set, and a 3 GB cap killed every job at
  event start. Control memory with parallelism; `RAD_JOB_KB` exists for the
  no-housing case where phonons live long enough that 22 concurrent jobs
  OOM-killed a sweep. Keep parallelism low enough for your RAM.
- **Cost depends enormously on whether the geometry has a phonon sink**: 0.66 s
  per primary with back-side traps, 58.88 s with housing and no traps (both in
  the shipped record). With no sink at all, an earlier test that is not in the
  record ran past 1200 s per primary. Quote the configuration with any cost or speedup figure.
