#!/usr/bin/env bash
# Species-resolved L2 sweep for the hadronic-floor attribution
#. Runs in WSL.
#
#   bash run_species_sweep.sh smoke     # tiny runs, prints timings
#   bash run_species_sweep.sh full [P]  # whole matrix, P workers (default 20)
#
# Outputs land in ~/rad/species/out and are copied to the repo at the end.
set -u
MODE=${1:-smoke}
P=${2:-20}
REPO=${RADCHAIN_REPO:?set RADCHAIN_REPO to the repo root}/src/radchain/geant4
WORK=$HOME/rad/species
mkdir -p "$WORK/out" "$WORK/log"
cp "$REPO/g4_species_spectrum.py" "$REPO/g4_l2_bench.py" "$WORK/"
cd "$WORK"
export LD_LIBRARY_PATH=$HOME/rad/sys/root/usr/lib/x86_64-linux-gnu
# High-precision neutron data from the source build, in case the wheel's own
# dataset set lacks it.
G4D=$HOME/opt/geant4/share/Geant4/data
[ -z "${G4NEUTRONHPDATA:-}" ] && [ -d "$G4D/G4NDL4.7.1" ] && export G4NEUTRONHPDATA=$G4D/G4NDL4.7.1
PY="${RADCHAIN_PY:-$HOME/rad/.venv/bin/python}"

job() {  # source n t orient seed tag [extra args...]
  local src=$1 n=$2 t=$3 o=$4 s=$5 tag=$6; shift 6
  local out="out/${tag}_${o}_${t}_${s}.json"
  [ -s "$out" ] && { echo "SKIP $out"; return 0; }
  echo Y | "$PY" g4_species_spectrum.py "$src" "$n" "$t" "$o" "$s" "$out" "$@" \
     > "log/${tag}_${o}_${t}_${s}.log" 2>&1
  tail -1 "log/${tag}_${o}_${t}_${s}.log"
}
export -f job
export PY

if [ "$MODE" = smoke ]; then
  job muon    2000  0.3 flat     1 smoke-mu4  --energy-mev 4000
  job muon    2000  0.3 vertical 1 smoke-mu4v --energy-mev 4000
  job neutron 20000 0.3 flat     1 smoke-n    --check
  job proton  5000  0.3 flat     1 smoke-p
  exit 0
fi

# FULL MATRIX. Shard sizes are set from the smoke timings; see the findings.
{
for t in 0.3 0.5 0.725; do
  for o in flat vertical; do
    for s in 1 2; do
      echo "muon 500000 $t $o $s mu1000 --energy-mev 1000"
      echo "muon 500000 $t $o $s mu4000 --energy-mev 4000"
    done
    for s in $(seq 1 10); do
      echo "neutron 1000000 $t $o $s n2 --zenith-n 2"
    done
    for s in $(seq 1 4); do
      echo "neutron 1000000 $t $o $s n0 --zenith-n 0"
      echo "proton 500000 $t $o $s p2 --zenith-n 2"
    done
  done
done
} > joblist.txt
echo "START $(date -Is) jobs=$(wc -l < joblist.txt) P=$P"
xargs -P "$P" -L 1 bash -c 'job "$@"' _ < joblist.txt
# Research output: kept out of the package and out of git because of its size.
DATA=${RADCHAIN_DATA:?set RADCHAIN_DATA to the deposit-data directory}
mkdir -p "$DATA/species"
cp out/*.json "$DATA/species/" && echo "COPIED $(ls out/*.json | wc -l) files"
echo "SWEEP_DONE $(date -Is)"
