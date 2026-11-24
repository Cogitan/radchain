#!/usr/bin/env bash
# In-cryostat species sweep.
#   bash run_cryostat_sweep.sh smoke
#   bash run_cryostat_sweep.sh full [P]
set -u
MODE=${1:-smoke}
P=${2:-20}
REPO=${RADCHAIN_REPO:?set RADCHAIN_REPO to the repo root}/src/radchain/geant4
WORK=$HOME/rad/cryo
mkdir -p "$WORK/out" "$WORK/log"
cp "$REPO/g4_cryostat_spectrum.py" "$REPO/g4_species_spectrum.py" \
   "$REPO/g4_l2_bench.py" "$WORK/"
cd "$WORK"
export LD_LIBRARY_PATH=$HOME/rad/sys/root/usr/lib/x86_64-linux-gnu
G4D=$HOME/opt/geant4/share/Geant4/data
[ -z "${G4NEUTRONHPDATA:-}" ] && [ -d "$G4D/G4NDL4.7.1" ] && export G4NEUTRONHPDATA=$G4D/G4NDL4.7.1
PY="${RADCHAIN_PY:-$HOME/rad/.venv/bin/python}"

job() {  # source n t orient seed tag [extra...]
  local src=$1 n=$2 t=$3 o=$4 s=$5 tag=$6; shift 6
  local out="out/${tag}_${o}_${t}_${s}.json"
  [ -s "$out" ] && { echo "SKIP $out"; return 0; }
  echo Y | "$PY" g4_cryostat_spectrum.py "$src" "$n" "$t" "$o" "$s" "$out" "$@" \
     > "log/${tag}_${o}_${t}_${s}.log" 2>&1
  tail -1 "log/${tag}_${o}_${t}_${s}.log"
}
export -f job
export PY

if [ "$MODE" = smoke ]; then
  job neutron 30000 0.3 flat     1 smoke-n
  job proton  10000 0.3 flat     1 smoke-p
  job muon    2000 0.3 vertical 1 smoke-mu --energy-mev 4000
  job neutron 1000 0.3 flat     1 smoke-nc --ceiling-cm 18
  exit 0
fi

# FULL MATRIX. Sized from the 2026-10-02 smoke (importance-sampled): neutron
# 1.4 ms and ~2.3e-4 events >=1 MeV per primary, proton 0.74 ms / 8e-4, muon
# 3.7 ms / 3e-3, neutron+ceiling 12 ms. Targets >=300 events >=1 MeV per
# config; ~12 core-hours total; 271 MB peak RSS per worker.
N_N=${N_N:-200000}; S_N=${S_N:-10}
N_P=${N_P:-200000}; S_P=${S_P:-5}
N_M=${N_M:-100000}; S_M=${S_M:-2}
{
for t in 0.3 0.5; do
  for o in flat vertical; do
    for s in $(seq 1 "$S_N"); do echo "neutron $N_N $t $o $s cn2 --zenith-n 2"; done
    for s in $(seq 1 "$S_P"); do echo "proton $N_P $t $o $s cp2 --zenith-n 2"; done
  done
  for o in flat vertical; do
    for s in $(seq 1 "$S_M"); do echo "muon $N_M $t $o $s cmu4000 --energy-mev 4000"; done
  done
done
for s in $(seq 1 "$S_N"); do echo "neutron $N_N 0.3 flat $s cn2ceil18 --zenith-n 2 --ceiling-cm 18"; done
} > joblist.txt
echo "START $(date -Is) jobs=$(wc -l < joblist.txt) P=$P"
xargs -P "$P" -L 1 bash -c 'job "$@"' _ < joblist.txt
# Research output: kept out of the package and out of git because of its size.
DATA=${RADCHAIN_DATA:?set RADCHAIN_DATA to the deposit-data directory}
mkdir -p "$DATA/cryo"
cp out/*.json "$DATA/cryo/" && echo "COPIED $(ls out/*.json | wc -l) files"
echo "SWEEP_DONE $(date -Is)"
