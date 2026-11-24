#!/usr/bin/env bash
# Chip-separation study. WSL, <= 8 workers.
# Writes ~/rad/pairs/PAIRDONE at the end; wait on that file, never on pgrep -f.
# Muons are generated over a disk covering BOTH chips and the space between
# them (+60 mm), so showers from muons passing midway are not missed.
set -u
P=${1:-8}
REPO=${RADCHAIN_REPO:?set RADCHAIN_REPO to the repo root}/src/radchain/geant4
WORK=$HOME/rad/pairs
mkdir -p "$WORK/out" "$WORK/log"
rm -f "$WORK/PAIRDONE"
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

rmu() { python3 -c "g=float('$1'[1:]); print(round(5+g/2+5+60,1))"; }

list() {
  # most informative first: reference, then the decisive separations
  for L in P1 Q100 P10 Q50 Q20 P20 P5 Q200; do
    n=3000000; [ "$L" = Q100 ] && n=4000000; [ "$L" = Q200 ] && n=5000000
    r=$(rmu $L)
    for s in 1 2 3 4 5 6 7 8; do echo "$L muon $n $((s+100)) $r"; done
  done
  for L in P1 Q50; do
    for s in 1 2 3 4; do
      echo "$L gamma-lines 50000000 $((s+200)) 0"
      echo "$L gamma-scatter 50000000 $((s+210)) 0"
    done
  done
}
list | xargs -P "$P" -L 1 bash -c 'job "$@"' _
mkdir -p ${RADCHAIN_DATA:?set RADCHAIN_DATA to the deposit-data directory}/pairs
cp out/[PQ]*_*.json ${RADCHAIN_DATA:?set RADCHAIN_DATA to the deposit-data directory}/pairs/ 2>/dev/null
touch "$WORK/PAIRDONE"
echo PAIRDONE
