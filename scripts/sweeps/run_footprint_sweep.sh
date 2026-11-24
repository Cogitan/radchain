#!/usr/bin/env bash
# Deposit-MAP sweep for the footprint test of the "burst size is monotone in
# deposited energy" assumption.
# Same matrix as the in-cryostat sweep (minus the ceiling) plus environmental
# gammas, every event >= 180 keV saved with a 0.1 mm die-plane energy map.
#   bash run_footprint_sweep.sh smoke
#   bash run_footprint_sweep.sh full [P]
set -u
MODE=${1:-smoke}
P=${2:-20}
BIN=0.1
REPO=${RADCHAIN_REPO:?set RADCHAIN_REPO to the repo root}/src/radchain/geant4
WORK=$HOME/rad/fp
mkdir -p "$WORK/out" "$WORK/log"
cp "$REPO/g4_cryostat_spectrum.py" "$REPO/g4_species_spectrum.py" \
   "$REPO/g4_env_spectrum.py" "$REPO/g4_l2_bench.py" "$WORK/"
cd "$WORK"
export LD_LIBRARY_PATH=$HOME/rad/sys/root/usr/lib/x86_64-linux-gnu
G4D=$HOME/opt/geant4/share/Geant4/data
[ -z "${G4NEUTRONHPDATA:-}" ] && [ -d "$G4D/G4NDL4.7.1" ] && export G4NEUTRONHPDATA=$G4D/G4NDL4.7.1
PY="${RADCHAIN_PY:-$HOME/rad/.venv/bin/python}"

cjob() {  # cryostat: source n t orient seed tag [extra...]
  local src=$1 n=$2 t=$3 o=$4 s=$5 tag=$6; shift 6
  local out="out/${tag}_${o}_${t}_${s}.json"
  [ -s "$out" ] && { echo "SKIP $out"; return 0; }
  echo Y | "$PY" g4_cryostat_spectrum.py "$src" "$n" "$t" "$o" "$s" "$out" \
     --map-bin-mm "$BIN" "$@" > "log/${tag}_${o}_${t}_${s}.log" 2>&1
  tail -1 "log/${tag}_${o}_${t}_${s}.log"
}
gjob() {  # env gamma: src n t seed
  local src=$1 n=$2 t=$3 s=$4
  local out="out/g-${src}_flat_${t}_${s}.json"
  [ -s "$out" ] && { echo "SKIP $out"; return 0; }
  echo Y | "$PY" g4_env_spectrum.py "$src" "$n" "$t" "$s" "$out" 10.0 G4_Si "$BIN" \
     > "log/g-${src}_${t}_${s}.log" 2>&1
  tail -1 "log/g-${src}_${t}_${s}.log"
}
job() { local kind=$1; shift; "$kind" "$@"; }
export -f cjob gjob job
export PY BIN

if [ "$MODE" = smoke ]; then
  cjob muon 3000 0.3 vertical 1 smoke-mu --energy-mev 4000
  cjob neutron 20000 0.3 flat 1 smoke-n
  gjob gamma-flat 200000 0.3 1
  exit 0
fi

{
for t in 0.3 0.5; do
  for o in flat vertical; do
    for s in $(seq 1 10); do echo "cjob neutron 200000 $t $o $s fn2 --zenith-n 2"; done
    for s in $(seq 1 5);  do echo "cjob proton 200000 $t $o $s fp2 --zenith-n 2"; done
    for s in 1 2;         do echo "cjob muon 100000 $t $o $s fmu4000 --energy-mev 4000"; done
  done
  for src in gamma-lines gamma-flat gamma-scatter; do
    for s in 1 2 3 4; do echo "gjob $src 1000000 $t $s"; done
  done
done
} > joblist.txt
echo "START $(date -Is) jobs=$(wc -l < joblist.txt) P=$P"
xargs -P "$P" -L 1 bash -c 'job "$@"' _ < joblist.txt
# Research output: kept out of the package and out of git because of its size.
DATA=${RADCHAIN_DATA:?set RADCHAIN_DATA to the deposit-data directory}
mkdir -p "$DATA/footprint"
cp out/*.json "$DATA/footprint/" && echo "COPIED $(ls out/*.json | wc -l) files"
echo "SWEEP_DONE $(date -Is)"
