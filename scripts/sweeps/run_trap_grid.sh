#!/bin/bash
# run_trap_grid.sh — G4CMP quasiparticle-birth kernels over a grid of
# back-side trap geometries (plain 4 mm Si die, no housing).
#
#   thickness   0.2, 0.381, 0.6 mm       sets the phonon's lateral step
#   pitch       0.25, 0.5, 1.0, 2.0 mm   island spacing
#   coverage    0.10, 0.25, 0.50         trap area fraction
#   absorption  0.5, 1.0                 per-encounter trap absorption
#
# 72 configurations.
#
# Usage: run_trap_grid.sh <thickness> [jobs_per_config] [primaries_per_job]
set -u
ROOT=$HOME/rad/kernel
cd "$ROOT"
source "$HOME/opt/geant4-11.4.1/bin/geant4.sh" > /dev/null 2>&1
export RAD_NO_HOUSING=1 RAD_CHIP_MM=4.0 RAD_PLAIN=1
T=${1:?thickness in mm}
JOBS=${2:-12}
PRIM=${3:-5}
export RAD_THICK_MM=$T
# Inject at mid-substrate: the chip top sits at z = 5.0 mm.
Z=$(python3 -c "print(round(5.0 - $T / 2, 4))")

LIST=$ROOT/trap_list_$T.txt
: > "$LIST"
for PITCH in 0.25 0.5 1.0 2.0; do
  for COV in 0.10 0.25 0.50; do
    for ABS in 0.5 1.0; do
      TAG="g_t${T}_p${PITCH}_c${COV}_a${ABS}"
      for s in $(seq 1 "$JOBS"); do
        echo "$TAG|$PITCH|$COV|$ABS|$TAG#$s $Z $PRIM $((s * 1009))" >> "$LIST"
      done
    done
  done
done

run_one() {
  IFS='|' read -r _tag pitch cov abs rest <<< "$1"
  set -- $rest
  export RAD_TRAP_PITCH_MM=$pitch RAD_TRAP_FRAC=$cov RAD_TRAP_ABS=$abs
  "${RADCHAIN_REPO:?set RADCHAIN_REPO}/g4cmp/l3_kernel_job.sh" "${1/\#/_}" 0.0 0.0 "$2" 0.0369 "$3" "$4" > /dev/null 2>&1
}
export -f run_one
export ROOT

xargs -P "${RAD_PAR:-16}" -I{} bash -c 'run_one "$@"' _ {} < "$LIST"
echo "thickness $T done: $(ls out/g_t${T}_*.json 2>/dev/null | wc -l) jobs"
