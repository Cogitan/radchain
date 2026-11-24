#!/bin/bash
# run_trap_grid_wide.sh — the wider G4CMP trap grid.
#
# `run_trap_grid.sh` covered a 72-configuration box:
#
#     thickness   0.2, 0.381, 0.6 mm
#     pitch       0.25, 0.5, 1.0, 2.0 mm
#     coverage    0.10, 0.25, 0.50
#     absorption  0.5, 1.0
#
# The original grid left two specific holes:
# absorption had only two levels, and nothing was run below 0.25 mm pitch or
# above 0.50 coverage.
#
# This widens every axis and adds a third absorption level:
#
#     thickness   0.1, 0.2, 0.381, 0.6, 1.0 mm      (5)
#     pitch       0.1, 0.25, 0.5, 1.0, 2.0, 4.0 mm  (6)
#     coverage    0.05, 0.10, 0.25, 0.50, 0.75      (5)
#     absorption  0.2, 0.5, 1.0                     (3)
#
# 450 configurations. At 8 jobs x 5 primaries that is 18,000 primary phonons.
#
# COST, MEASURED BEFORE LAUNCHING rather than assumed. Per-primary cost from
# `wall_s` on 5-primary probe jobs:
#
#     thickness 0.1 mm, coverage 0.05, absorption 0.2   1.8 s   <- worst
#     thickness 0.2 mm, coverage 0.10, absorption 0.5   0.4 s
#     thickness 1.0 mm, coverage 0.05, absorption 0.2   0.8 s
#
# So about 5 core-hours, and there is no runaway corner. The intuition that a
# weak sink would blow up is WRONG here: cost falls with thickness, because a
# phonon in a thick die travels further laterally per traversal and reaches the
# 4 mm edge sooner. **Thin is the expensive axis, not weak.** A 1-primary probe
# reads 14 s at 0.2 mm only because it pays the full process startup; read
# `wall_s` from a multi-primary job instead.
#
# Usage: run_trap_grid_wide.sh <thickness> [jobs_per_config] [primaries_per_job]
#   RAD_PAR   parallel jobs (default 12; lower it if jobs are OOM-killed)
set -u
ROOT=$HOME/rad/kernel
cd "$ROOT"
source "$HOME/opt/geant4-11.4.1/bin/geant4.sh" > /dev/null 2>&1
export RAD_NO_HOUSING=1 RAD_CHIP_MM=4.0 RAD_PLAIN=1
T=${1:?thickness in mm}
JOBS=${2:-8}
PRIM=${3:-5}
export RAD_THICK_MM=$T
# Inject at mid-substrate: the chip top sits at z = 5.0 mm.
Z=$(python3 -c "print(round(5.0 - $T / 2, 4))")

LIST=$ROOT/trap_wide_list_$T.txt
: > "$LIST"
for PITCH in 0.1 0.25 0.5 1.0 2.0 4.0; do
  for COV in 0.05 0.10 0.25 0.50 0.75; do
    for ABS in 0.2 0.5 1.0; do
      TAG="g_t${T}_p${PITCH}_c${COV}_a${ABS}"
      # Tags follow run_trap_grid.sh exactly, so radchain.phonon and
      # kernel_runs.iter_runs("g_t") pick these up with no change.
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

xargs -P "${RAD_PAR:-12}" -I{} bash -c 'run_one "$@"' _ {} < "$LIST"
echo "thickness $T done: $(ls out/g_t${T}_*.json 2>/dev/null | wc -l) job files"
