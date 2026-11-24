#!/usr/bin/env bash
# Chip-scale correlated erasure.
# Runs in WSL, <= 8 workers. Writes ~/rad/chips/ALLDONE when finished; wait on
# that file, never on `pgrep -f` (it matches the waiting shell itself).
#
#   bash run_chip_sweep.sh [P]
set -u
P=${1:-8}
REPO=${RADCHAIN_REPO:?set RADCHAIN_REPO to the repo root}/src/radchain/geant4
WORK=$HOME/rad/chips
mkdir -p "$WORK/out" "$WORK/log"
rm -f "$WORK/ALLDONE"
cp "$REPO"/{g4_chip_array.py,g4_env_spectrum.py,g4_l2_bench.py,g4_cryostat_spectrum.py,g4_species_spectrum.py} "$WORK/"
cd "$WORK"
export LD_LIBRARY_PATH=$HOME/rad/sys/root/usr/lib/x86_64-linux-gnu
PY="${RADCHAIN_PY:-$HOME/rad/.venv/bin/python}"
export PY

job() {  # layout source n seed [extra...]
  local L=$1 src=$2 n=$3 s=$4; shift 4
  local out="out/${L}_${src}_${s}.json"
  [ -s "$out" ] && { echo "SKIP $out"; return 0; }
  echo Y | "$PY" g4_chip_array.py "$src" "$n" "$out" --layout "$L" --seed "$s" "$@" \
     > "log/${L}_${src}_${s}.log" 2>&1
  tail -1 "log/${L}_${src}_${s}.log"
}
export -f job

list() {
  for L in A1 B3 V1 S1 A3; do
    for s in 1 2 3 4 5 6 7 8; do
      echo "$L gamma-lines 50000000 $((s+10))"
      echo "$L gamma-scatter 50000000 $((s+20))"
      echo "$L muon 1500000 $((s+30)) --cryostat --r-muon 80"
    done
  done
}
list | xargs -P "$P" -L 1 bash -c 'job "$@"' _
mkdir -p ${RADCHAIN_DATA:?set RADCHAIN_DATA to the deposit-data directory}/chips
cp out/[ABVS]*_*.json ${RADCHAIN_DATA:?set RADCHAIN_DATA to the deposit-data directory}/chips/ 2>/dev/null
touch "$WORK/ALLDONE"
echo ALLDONE
