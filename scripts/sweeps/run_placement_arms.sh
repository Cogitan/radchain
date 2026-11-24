#!/bin/bash
# run_placement_arms.sh — does trap PLACEMENT matter at fixed coverage?
#
# Plain 4 mm silicon die, ground plane on top, no copper housing, back-side
# copper trap islands. All arms share coverage and absorption; only the PITCH
# changes.
#
#   B  0.5 mm pitch, 25% coverage   64 islands of 0.25 mm
#   C  2.0 mm pitch, 25% coverage    4 islands of 1.0 mm
#   E  1.0 mm pitch, 25% coverage   16 islands of 0.5 mm   (trend point)
#   D  0.5 mm pitch, 100% coverage  uniform back absorber  (reference)
#
# The no-trap arm is deliberately absent: with no back-side sink a phonon
# recycles through the film (quasiparticles radiate phonons back into the
# substrate) and one primary costs over 600 s instead of 4 s.
set -e
ROOT=$HOME/rad/kernel
cd "$ROOT"
source $HOME/opt/geant4-11.4.1/bin/geant4.sh > /dev/null 2>&1
export RAD_NO_HOUSING=1 RAD_CHIP_MM=4.0 RAD_PLAIN=1 RAD_TRAP_ABS=1.0
N_JOBS=${1:-30}
N_PRIM=${2:-5}

for ARM in B C E D; do
  case $ARM in
    B) export RAD_TRAP_PITCH_MM=0.5 RAD_TRAP_FRAC=0.25 ;;
    C) export RAD_TRAP_PITCH_MM=2.0 RAD_TRAP_FRAC=0.25 ;;
    E) export RAD_TRAP_PITCH_MM=1.0 RAD_TRAP_FRAC=0.25 ;;
    D) export RAD_TRAP_PITCH_MM=0.5 RAD_TRAP_FRAC=1.00 ;;
  esac
  rm -f list_$ARM.txt
  for s in $(seq 1 "$N_JOBS"); do
    echo "p${ARM}_$s 0.0 0.0 4.81 0.0369 $N_PRIM $((s * 1009))" >> list_$ARM.txt
  done
  xargs -P "${RAD_PAR:-16}" -L 1 ./l3_kernel_job.sh < list_$ARM.txt
  echo "ARM $ARM COMPLETE: $(ls out/p${ARM}_*.json 2>/dev/null | wc -l) jobs"
done
echo PLACEMENT_ARMS_DONE
