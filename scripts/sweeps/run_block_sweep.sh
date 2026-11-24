#!/usr/bin/env bash
# Code-block study. WSL, <= 8 workers.
# Writes ~/rad/block/BLOCKDONE; wait on that file, never on pgrep -f.
# Muon disk covers the whole block plus 60 mm.
set -u
P=${1:-8}
REPO=${RADCHAIN_REPO:?set RADCHAIN_REPO to the repo root}/src/radchain/geant4
WORK=$HOME/rad/block
mkdir -p "$WORK/out" "$WORK/log"
rm -f "$WORK/BLOCKDONE"
cp "$REPO"/{g4_chip_array.py,g4_env_spectrum.py,g4_l2_bench.py,g4_cryostat_spectrum.py,g4_species_spectrum.py} "$WORK/"
cd "$WORK"
export LD_LIBRARY_PATH=$HOME/rad/sys/root/usr/lib/x86_64-linux-gnu
PY="${RADCHAIN_PY:-$HOME/rad/.venv/bin/python}"
export PY
job() {  # layout source n seed rmuon
  local L=$1 src=$2 n=$3 s=$4 r=$5
  local out="out/${L}_${src}_${s}.json"
  [ -s "$out" ] && { echo "SKIP $out"; return 0; }
  local extra=""
  [ "$src" = muon ] && extra="--cryostat --r-muon $r"
  echo Y | "$PY" g4_chip_array.py "$src" "$n" "$out" --layout "$L" --seed "$s" $extra \
     > "log/${L}_${src}_${s}.log" 2>&1
  tail -1 "log/${L}_${src}_${s}.log"
}
export -f job
list() {
  for s in 1 2 3 4 5 6 7 8; do echo "X1 muon 2000000 $((s+100)) 82"; done
  for s in 1 2 3 4 5 6 7 8; do echo "XQ50 muon 6000000 $((s+200)) 130"; done
  for s in 1 2 3 4 5 6 7 8; do echo "XQ100 muon 12000000 $((s+300)) 180"; done
  for L in X1 XQ50; do for s in 1 2 3 4; do
    echo "$L gamma-lines 50000000 $((s+400)) 0"
    echo "$L gamma-scatter 50000000 $((s+410)) 0"
  done; done
}
list | xargs -P "$P" -L 1 bash -c 'job "$@"' _
mkdir -p ${RADCHAIN_DATA:?set RADCHAIN_DATA to the deposit-data directory}/block
cp out/X*_*.json ${RADCHAIN_DATA:?set RADCHAIN_DATA to the deposit-data directory}/block/ 2>/dev/null
touch "$WORK/BLOCKDONE"
echo BLOCKDONE
