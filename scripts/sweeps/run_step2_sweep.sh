#!/usr/bin/env bash
# Heavier-fridge floor test and Q200 on the fixed code.
# WSL, 8 workers.
# Writes ~/rad/step2/STEP2DONE; wait on that file, never on pgrep -f.
set -u
P=${1:-8}
REPO=${RADCHAIN_REPO:?set RADCHAIN_REPO to the repo root}/src/radchain/geant4
WORK=$HOME/rad/step2
mkdir -p "$WORK/out" "$WORK/log"
rm -f "$WORK/STEP2DONE"
cp "$REPO"/{g4_chip_array.py,g4_env_spectrum.py,g4_l2_bench.py,g4_cryostat_spectrum.py,g4_species_spectrum.py} "$WORK/"
cd "$WORK"
export LD_LIBRARY_PATH=$HOME/rad/sys/root/usr/lib/x86_64-linux-gnu
PY="${RADCHAIN_PY:-$HOME/rad/.venv/bin/python}"
export PY
job() {  # tag layout n seed rmuon [extra...]
  local tag=$1 L=$2 n=$3 s=$4 r=$5; shift 5
  local out="out/${tag}_${L}_muon_${s}.json"
  [ -s "$out" ] && return 0
  echo Y | "$PY" g4_chip_array.py muon "$n" "$out" --layout "$L" --seed "$s" \
     --cryostat --r-muon "$r" "$@" > "log/${tag}_${L}_${s}.log" 2>&1
  tail -1 "log/${tag}_${L}_${s}.log"
}
export -f job
{
  for s in $(seq 1 8); do echo "heavy P1 3000000 $((s+700)) 75.5 --fridge-scale 2"; done
  for s in $(seq 1 8); do echo "heavy Q50 7500000 $((s+710)) 100 --fridge-scale 2"; done
  for s in $(seq 1 8); do echo "fixed Q200 16000000 $((s+720)) 175"; done
} | xargs -P "$P" -L 1 bash -c 'job "$@"' _
mkdir -p ${RADCHAIN_DATA:?set RADCHAIN_DATA to the deposit-data directory}/step2
cp out/*.json ${RADCHAIN_DATA:?set RADCHAIN_DATA to the deposit-data directory}/step2/ 2>/dev/null
touch "$WORK/STEP2DONE"
echo STEP2DONE
