#!/usr/bin/env bash
# Wu et al. 2025 cross-module reproduction.
# Runs in WSL. Keep workers low enough for your RAM.
#
#   bash run_wu_sweep.sh base       # baseline geometry, all sources
#   bash run_wu_sweep.sh brackets   # gap, thickness, wall, muon energy brackets
#   bash run_wu_sweep.sh final      # measured 2.2 mm gap, fridge for muons
set -u
MODE=${1:-base}
P=${2:-8}
REPO=${RADCHAIN_REPO:?set RADCHAIN_REPO to the repo root}/src/radchain/geant4
WORK=$HOME/rad/wu
mkdir -p "$WORK/out" "$WORK/log"
cp "$REPO/g4_wu_modules.py" "$REPO/g4_env_spectrum.py" "$REPO/g4_l2_bench.py"    "$REPO/g4_cryostat_spectrum.py" "$REPO/g4_species_spectrum.py" "$WORK/"
cd "$WORK"
export LD_LIBRARY_PATH=$HOME/rad/sys/root/usr/lib/x86_64-linux-gnu
PY="${RADCHAIN_PY:-$HOME/rad/.venv/bin/python}"
export PY

job() {  # tag source n seed [extra args...]
  local tag=$1 src=$2 n=$3 s=$4; shift 4
  local out="out/${tag}_${src}_${s}.json"
  [ -s "$out" ] && { echo "SKIP $out"; return 0; }
  echo Y | "$PY" g4_wu_modules.py "$src" "$n" "$out" --seed "$s" "$@" \
     > "log/${tag}_${src}_${s}.log" 2>&1
  tail -1 "log/${tag}_${src}_${s}.log"
}
export -f job

list() {
  if [ "$MODE" = final ]; then
    # measured gap (PRX Fig. 1c pads + stud marks: 2.2 +/- 0.15 mm), both
    # thickness brackets, muons WITH fridge material (r-muon 60 mm)
    for t in 0.5 0.65; do
      tg=fin${t/./}
      for s in 1 2 3 4; do
        echo "$tg gamma-lines 20000000 $((s+500)) --gap 2.2 --t-db $t --t-mb $t"
        echo "$tg gamma-scatter 20000000 $((s+510)) --gap 2.2 --t-db $t --t-mb $t"
      done
      for s in 1 2 3 4 5 6 7 8; do
        echo "$tg muon 3000000 $((s+520)) --gap 2.2 --t-db $t --t-mb $t --cryostat --r-muon 60"
      done
    done
  elif [ "$MODE" = base ]; then
    for s in $(seq 1 12); do
      echo "base gamma-lines 20000000 $s"
      echo "base gamma-scatter 20000000 $((s+100))"
    done
    for s in 1 2 3 4; do echo "base muon 2000000 $((s+200))"; done
  else
    for s in 1 2 3 4; do
      for src in gamma-lines gamma-scatter; do
        echo "gap1 $src 20000000 $((s+300)) --gap 1.0"
        echo "gap3 $src 20000000 $((s+310)) --gap 3.0"
        echo "t043 $src 20000000 $((s+320)) --t-db 0.43 --t-mb 0.43"
        echo "t065 $src 20000000 $((s+330)) --t-db 0.65 --t-mb 0.65"
        echo "wall2 $src 20000000 $((s+340)) --wall 2.0"
        echo "wall10 $src 20000000 $((s+350)) --wall 10.0"
      done
      echo "gap1 muon 2000000 $((s+400)) --gap 1.0"
      echo "gap3 muon 2000000 $((s+410)) --gap 3.0"
      echo "t043 muon 2000000 $((s+420)) --t-db 0.43 --t-mb 0.43"
      echo "t065 muon 2000000 $((s+430)) --t-db 0.65 --t-mb 0.65"
      echo "mu1gev muon 2000000 $((s+440)) --muon-mev 1000"
    done
  fi
}
list | xargs -P "$P" -L 1 bash -c 'job "$@"' _
mkdir -p "${RADCHAIN_DATA:?set RADCHAIN_DATA}/wu"
cp out/*.json "${RADCHAIN_DATA:?set RADCHAIN_DATA}/wu/" 2>/dev/null
echo ALLDONE
