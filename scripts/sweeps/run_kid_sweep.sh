#!/usr/bin/env bash
# KID-tagger reproduction.
# Runs with 3 workers alongside a 5-worker sweep (never more than 8 in WSL).
# Writes ~/rad/kid/KIDDONE.
set -u
# (runs alongside the block rerun: 3 workers here + 5 there)
REPO=${RADCHAIN_REPO:?set RADCHAIN_REPO to the repo root}/src/radchain/geant4
WORK=$HOME/rad/kid
mkdir -p "$WORK/out" "$WORK/log"
rm -f "$WORK/KIDDONE"
cp "$REPO"/{g4_kid_stack.py,g4_env_spectrum.py,g4_l2_bench.py,g4_cryostat_spectrum.py,g4_species_spectrum.py} "$WORK/"
cd "$WORK"
export LD_LIBRARY_PATH=$HOME/rad/sys/root/usr/lib/x86_64-linux-gnu
PY="${RADCHAIN_PY:-$HOME/rad/.venv/bin/python}"
export PY
job() {  # tag source n seed [extra...]
  local tag=$1 src=$2 n=$3 s=$4; shift 4
  local out="out/${tag}_${src}_${s}.json"
  [ -s "$out" ] && return 0
  echo Y | "$PY" g4_kid_stack.py "$src" "$n" "$out" --seed "$s" "$@" > "log/${tag}_${src}_${s}.log" 2>&1
  tail -1 "log/${tag}_${src}_${s}.log"
}
export -f job
{
  # nominal, with fridge (muons): 4 x 400k
  for s in 1 2 3 4; do echo "nom muon 400000 $((s+10)) --cryostat"; done
  # geometry brackets (muons, fridge)
  for s in 1 2; do
    echo "gap40 muon 400000 $((s+20)) --cryostat --gap 4.0"
    echo "gap50 muon 400000 $((s+30)) --cryostat --gap 5.0"
    echo "lid05 muon 400000 $((s+40)) --cryostat --lid 0.5"
    echo "lid20 muon 400000 $((s+50)) --cryostat --lid 2.0"
    echo "nofridge muon 400000 $((s+60))"
  done
  # gammas for R2/R3 (no fridge; field defined at the pot)
  for s in 1 2; do
    echo "nom gamma-lines 20000000 $((s+70))"
    echo "nom gamma-scatter 20000000 $((s+80))"
  done
} | xargs -P 3 -L 1 bash -c 'job "$@"' _
mkdir -p ${RADCHAIN_DATA:?set RADCHAIN_DATA to the deposit-data directory}/kid
cp out/*.json ${RADCHAIN_DATA:?set RADCHAIN_DATA to the deposit-data directory}/kid/
touch "$WORK/KIDDONE"
echo KIDDONE
