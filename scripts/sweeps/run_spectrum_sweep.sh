#!/usr/bin/env bash
# Muon-spectrum robustness. WSL, <= 8 workers.
# X1 8 x 2M (seeds 601-608), XQ50 32 x 6M (seeds 701-732), --muon-spectrum guan.
# Writes ~/rad/spec/SPECDONE; wait on that file, never on pgrep -f.
set -u
P=${1:-8}
REPO=${RADCHAIN_REPO:?set RADCHAIN_REPO to the repo root}/src/radchain/geant4
WORK=$HOME/rad/spec
DEST=${RADCHAIN_DATA:?set RADCHAIN_DATA to the deposit-data directory}/block_spectrum
mkdir -p "$WORK/out" "$WORK/log" "$DEST"
rm -f "$WORK/SPECDONE"
cp "$REPO"/{g4_chip_array.py,g4_env_spectrum.py,g4_l2_bench.py,g4_cryostat_spectrum.py,g4_species_spectrum.py} "$WORK/"
cd "$WORK"
export LD_LIBRARY_PATH=$HOME/rad/sys/root/usr/lib/x86_64-linux-gnu
PY="${RADCHAIN_PY:-$HOME/rad/.venv/bin/python}"
export PY DEST
job() {  # layout n seed rmuon
  local L=$1 n=$2 s=$3 r=$4
  local out="out/${L}_muon_${s}.json"
  [ -s "$out" ] && return 0
  echo Y | "$PY" g4_chip_array.py muon "$n" "$out" --layout "$L" --seed "$s" \
     --cryostat --r-muon "$r" --muon-spectrum guan > "log/${L}_${s}.log" 2>&1
  tail -1 "log/${L}_${s}.log"
  cp "$out" "$DEST/"
}
export -f job
{
  for s in $(seq 701 732); do echo "XQ50 6000000 $s 130"; done
  for s in $(seq 601 608); do echo "X1 2000000 $s 82"; done
} | xargs -P "$P" -L 1 bash -c 'job "$@"' _
touch "$WORK/SPECDONE"
echo SPECDONE
