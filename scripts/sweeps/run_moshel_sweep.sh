#!/usr/bin/env bash
# Moshel et al. two-die coincidence.
# Runs in WSL, nice 19, P workers (default 4; keep it low enough for your RAM).
# Restart-safe: a job whose output exists is skipped.
#
#   bash run_moshel_sweep.sh [P] [main|entry]
# entry: nominal geometry again with --entry (birth volume of every track
# entering a die in coincident events). Post hoc diagnostic, added after the
# main sweep showed direct crossing is a minority (2026-11-16).
# Cost measured 2026-11-16 (region cut 1 um): gamma ~33 us, muon+fridge
# ~541 us per primary. Whole sweep ~3.5 core-h.
set -u
P=${1:-4}
MODE=${2:-main}
REPO=${RADCHAIN_REPO:?set RADCHAIN_REPO to the repo root}/src/radchain/geant4
WORK=$HOME/rad/moshel
mkdir -p "$WORK/out" "$WORK/log"
cp "$REPO/g4_moshel_dies.py" "$REPO/g4_env_spectrum.py" \
   "$REPO/g4_cryostat_spectrum.py" "$WORK/"
cp "$REPO"/g4_l2_bench.py "$REPO"/g4_species_spectrum.py "$WORK/" 2>/dev/null
cd "$WORK"
export LD_LIBRARY_PATH=$HOME/rad/sys/root/usr/lib/x86_64-linux-gnu
PY="${RADCHAIN_PY:-$HOME/rad/.venv/bin/python}"
export PY

job() {  # tag source n seed [extra args...]
  local tag=$1 src=$2 n=$3 s=$4; shift 4
  local out="out/${tag}_${src}_${s}.json"
  [ -s "$out" ] && { echo "SKIP $out"; return 0; }
  echo Y | nice -n 19 "$PY" g4_moshel_dies.py "$src" "$n" "$out.tmp" \
     --seed "$s" "$@" > "log/${tag}_${src}_${s}.log" 2>&1 \
     && mv "$out.tmp" "$out"
  echo "$tag $src $s: $(tail -1 "log/${tag}_${src}_${s}.log")"
}
export -f job

list() {
  if [ "$MODE" = entry ]; then
    for s in 1 2 3 4 5 6 7 8; do
      echo "nomE muon 500000 $((s+600)) --cryostat --entry"
    done
    for s in 1 2; do
      echo "nomE gamma-lines 10000000 $((s+610)) --entry"
      echo "nomE gamma-scatter 10000000 $((s+620)) --entry"
    done
    return
  fi
  # nominal: 13.5 x 10 x 0.5 mm dies, 0.5 mm pedestals, 8 mm cavity
  for s in 1 2 3 4 5 6 7 8; do
    echo "nom muon 500000 $((s+100)) --cryostat"
  done
  for s in 1 2 3 4; do
    echo "nom gamma-lines 10000000 $((s+110))"
    echo "nom gamma-scatter 10000000 $((s+120))"
  done
  # brackets
  for s in 1 2 3 4; do
    echo "sml muon 500000 $((s+200)) --cryostat --L 10.5 --W 9.2"
    echo "t675 muon 500000 $((s+300)) --cryostat --T 0.675"
    echo "ped1 muon 500000 $((s+400)) --cryostat --h-ped 1.0"
    echo "cav12 muon 500000 $((s+500)) --cryostat --cav-h 12"
  done
  for s in 1 2; do
    echo "sml gamma-lines 10000000 $((s+210)) --L 10.5 --W 9.2"
    echo "sml gamma-scatter 10000000 $((s+220)) --L 10.5 --W 9.2"
    echo "t675 gamma-lines 10000000 $((s+310)) --T 0.675"
    echo "t675 gamma-scatter 10000000 $((s+320)) --T 0.675"
  done
}
list | xargs -P "$P" -L 1 bash -c 'job "$@"' _
mkdir -p "${RADCHAIN_DATA:?set RADCHAIN_DATA}/moshel"
cp out/*.json "${RADCHAIN_DATA:?set RADCHAIN_DATA}/moshel/" 2>/dev/null
echo ALLDONE
