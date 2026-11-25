#!/bin/bash
# l3_kernel_job.sh: one G4CMP transport-kernel job, scored and cleaned up.
#
# Runs N primary phonons from a point in the substrate, scores every
# quasiparticle birth (two per broken Cooper pair) into a radial histogram around
# the injection point, writes JSON, and DELETES the raw step file. The raw file
# is ~50 MB per primary phonon, so keeping them is not an option at sweep scale.
#
# Usage: l3_kernel_job.sh <tag> <x_mm> <y_mm> <z_mm> <energy_eV> <n_primaries> <seed>
#
# Environment (defaults in brackets):
#   RAD_ROOT       work dir: jobs/, out/, base.mac          [$HOME/rad/kernel]
#   RAD_BASE_MAC   G4CMP macro the job macro is built from  [$RAD_ROOT/base.mac if it
#                  exists, else base.mac next to this script]
#   RAD_EXE        patched g4cmpQuasiparticle binary        [$HOME/rad/l3trap/build/...]
#   G4CMP_PREFIX   G4CMP install prefix                     [$HOME/opt/g4cmp-1141]
#   GEANT4_PREFIX  Geant4 11.4.1 install prefix             [$HOME/opt/geant4-11.4.1]
#   RAD_PYTHON     python with radchain installed           [python3]
#   RAD_JOB_KB     optional ulimit -v per job, kB (see g4cmp/README.md)
#
# base.mac is a macro from G4CMP's examples/quasiparticle/G4Macros; the one the
# shipped kernel record used is g4cmp/base.mac in this repository. Its
# /gps/energy, /gps/pos/centre, /gps/number and /run/beamOn lines are
# replaced per job, so only its physics and output settings matter. It must
# enable the quasiparticle step-information output the scorer reads.
#
# G4LATTICEDATA must be set explicitly; g4cmp_env.sh was not reliable.
set -e
TAG=$1; X=$2; Y=$3; Z=$4; E=$5; N=$6; SEED=$7
ROOT=${RAD_ROOT:-$HOME/rad/kernel}
BASE_MAC=${RAD_BASE_MAC:-$ROOT/base.mac}
[ -n "${RAD_BASE_MAC:-}" ] || [ -f "$BASE_MAC" ] || BASE_MAC="$(cd "$(dirname "$0")" && pwd)/base.mac"
G4CMP_PREFIX=${G4CMP_PREFIX:-$HOME/opt/g4cmp-1141}
GEANT4_PREFIX=${GEANT4_PREFIX:-$HOME/opt/geant4-11.4.1}
RAD_PYTHON=${RAD_PYTHON:-python3}
# THE PATCHED BINARY, not the stock example. The stock example's
# libg4cmpQuasiparticle.so has ZERO of the RAD_TRAP_* knobs, so pointing at it
# runs the STOCK GEOMETRY SILENTLY and every trap sweep collapses to one
# configuration. Build it from g4cmp/l3trap.patch: see g4cmp/README.md.
EXE=${RAD_EXE:-$HOME/rad/l3trap/build/g4cmpQuasiparticle}
export G4LATTICEDATA=$G4CMP_PREFIX/share/G4CMP/CrystalMaps
export LD_LIBRARY_PATH=$G4CMP_PREFIX/lib:$LD_LIBRARY_PATH
source "$GEANT4_PREFIX/bin/geant4.sh"

# ---------------------------------------------------------------- preflight
# Two traps that each cost a wrong conclusion. Both are cheap to check and
# neither announces itself at runtime, so check them before burning core-hours.
#
# 1. A binary whose detector library lacks the knobs ignores every RAD_*
#    geometry variable (traps, thickness, die size, film, housing) and returns
#    stock geometry. A sweep over pitch then produces
#    identical kernels and looks like a physics result.
# 2. G4CMP built against $HOME/opt/g4cmp (not -1141) crashes at event start,
#    and g4cmp_env.sh puts its own lib path first, so a stale libG4cmp.so can
#    be loaded ahead of the intended one (recorded 09-15 and again 09-19).
SO="$(dirname "$EXE")/libg4cmpQuasiparticle.so"
if [ -n "${RAD_TRAP_PITCH_MM:-}${RAD_TRAP_FRAC:-}${RAD_TRAP_ABS:-}${RAD_TRAP_THICK_UM:-}${RAD_THICK_MM:-}${RAD_CHIP_MM:-}${RAD_PLAIN:-}${RAD_NO_HOUSING:-}" ]; then
  if ! strings "$SO" 2>/dev/null | grep -q RAD_TRAP_PITCH_MM; then
    echo "FATAL: RAD_* geometry variables set but $SO has no knobs." >&2
    echo "       This binary would run STOCK geometry and ignore them." >&2
    echo "       Apply g4cmp/l3trap.patch and rebuild." >&2
    exit 3
  fi
fi
RESOLVED=$(ldd "$EXE" 2>/dev/null | awk '/libG4cmp\.so/ {print $3}')
case "$RESOLVED" in
  *g4cmp-1141*) : ;;
  "") echo "WARN: could not resolve libG4cmp.so for $EXE" >&2 ;;
  *) echo "FATAL: $EXE resolves libG4cmp.so to $RESOLVED, not the -1141 install." >&2
     echo "       That build segfaults at event start. Set G4CMPINSTALL to" >&2
     echo "       \$HOME/opt/g4cmp-1141/share/G4CMP and rebuild (g4cmp/README.md)." >&2
     exit 4 ;;
esac

[ -f "$BASE_MAC" ] || { echo "FATAL: no G4CMP macro at $BASE_MAC (set RAD_BASE_MAC)" >&2; exit 5; }
DIR=$ROOT/jobs/$TAG
rm -rf "$DIR"; mkdir -p "$DIR" "$ROOT/out"
cd "$DIR"

grep -v "^/gps/pos/centre\|^/run/beamOn\|^/gps/number\|^/gps/energy" \
    "$BASE_MAC" > job.mac
cat >> job.mac <<EOF
/gps/energy $E eV
/gps/pos/centre $X $Y $Z mm
/gps/number 1
/random/setSeeds $SEED $((SEED + 7))
/run/beamOn $N
EOF

# Cap per-job address space when asked. Without the copper housing phonons
# live much longer and 22 concurrent jobs OOM-killed a whole sweep (09-19); a
# job over the cap then dies alone instead of taking the sweep with it.
# Do NOT cap with `ulimit -v` by default: Geant4 reserves a large virtual
# address space even when resident use is small, and a 3 GB cap killed every
# job at event start (09-19). Control memory with parallelism instead.
[ -n "${RAD_JOB_KB:-}" ] && ulimit -v "$RAD_JOB_KB"

T0=$(date +%s)
"$EXE" job.mac > run.out 2> run.err
WALL=$(( $(date +%s) - T0 ))

"$RAD_PYTHON" -m radchain.score_kernel QuasiparticleStepInformationFile.txt \
        "$ROOT/out/$TAG.json" "$TAG" "$X" "$Y" "$Z" "$E" "$N" "$SEED" "$WALL"
rm -f QuasiparticleStepInformationFile.txt
echo "DONE $TAG n=$N wall=${WALL}s"
