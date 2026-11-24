#!/bin/bash
# run_kernel_thickness.sh -- the G4CMP transport kernel at more substrate
# thicknesses (0.1 to 1.0 mm).
#
# Same geometry as the original s1_centre gate (stock 8 mm chip, patterned top,
# copper housing, centre strike at mid-depth, 36.9 meV primaries); the ONLY
# knob is RAD_THICK_MM. 0.381 is re-run as a control on the patched binary.
#
#   bash run_kernel_thickness.sh smoke         # one job, one primary, 0.2 mm
#   bash run_kernel_thickness.sh full [P] [J]  # 5 thicknesses x J jobs, P workers
#
# Resumable: jobs with an output file are skipped. Outputs: ~/rad/kernel/out/kt_*.json
set -u
MODE=${1:-smoke}
P=${2:-16}
JOBS=${3:-20}
ROOT=$HOME/rad/kernel
cd "$ROOT"
# Original-gate geometry: none of the trap or plain-geometry knobs.
unset RAD_PLAIN RAD_NO_HOUSING RAD_CHIP_MM RAD_TRAP_PITCH_MM RAD_TRAP_FRAC \
      RAD_TRAP_ABS RAD_TRAP_THICK_UM

run_one() {  # thickness job_index primaries
  local t=$1 j=$2 n=$3
  local tag="kt_t${t}_${j}"
  [ -s "$ROOT/out/$tag.json" ] && { echo "SKIP $tag"; return 0; }
  local z
  z=$(python3 -c "print(round(5.0 - $t / 2, 4))")   # chip top at z = 5.0 mm
  RAD_THICK_MM=$t "${RADCHAIN_REPO:?set RADCHAIN_REPO}/g4cmp/l3_kernel_job.sh" "$tag" 0.0 0.0 "$z" 0.0369 "$n" \
      $((j * 1009 + 7)) 2>&1 | tail -1
}
export -f run_one
export ROOT

if [ "$MODE" = smoke ]; then
  run_one 0.2 0 1
  exit 0
fi

for t in 0.381 0.1 0.2 0.6 1.0; do
  for j in $(seq 1 "$JOBS"); do echo "$t $j 5"; done
done | xargs -P "$P" -L 1 bash -c 'run_one "$@"' _
echo "ALL DONE $(date -Is)"
