#!/usr/bin/env bash
# XQ50 statistics extension. WSL, <= 8 workers.
# 32 x 6M muons, seeds 501-532. Writes ~/rad/xq50more/XQ50DONE; wait on that
# file, never on pgrep -f.
set -u
P=${1:-8}
REPO=${RADCHAIN_REPO:?set RADCHAIN_REPO to the repo root}/src/radchain/geant4
WORK=$HOME/rad/xq50more
mkdir -p "$WORK/out" "$WORK/log"
rm -f "$WORK/XQ50DONE"
cp "$REPO"/{g4_chip_array.py,g4_env_spectrum.py,g4_l2_bench.py,g4_cryostat_spectrum.py,g4_species_spectrum.py} "$WORK/"
cd "$WORK"
export LD_LIBRARY_PATH=$HOME/rad/sys/root/usr/lib/x86_64-linux-gnu
PY="${RADCHAIN_PY:-$HOME/rad/.venv/bin/python}"
export PY
job() {  # seed
  local s=$1
  local out="out/XQ50_muon_${s}.json"
  [ -s "$out" ] && return 0
  echo Y | "$PY" g4_chip_array.py muon 6000000 "$out" --layout XQ50 --seed "$s" \
     --cryostat --r-muon 130 > "log/XQ50_muon_${s}.log" 2>&1
  tail -1 "log/XQ50_muon_${s}.log"
  cp "$out" ${RADCHAIN_DATA:?set RADCHAIN_DATA to the deposit-data directory}/block/
}
export -f job
mkdir -p ${RADCHAIN_DATA:?set RADCHAIN_DATA to the deposit-data directory}/block
seq 501 532 | xargs -P "$P" -L 1 bash -c 'job "$@"' _
touch "$WORK/XQ50DONE"
echo XQ50DONE
