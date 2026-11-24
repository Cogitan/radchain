#!/usr/bin/env bash
# Deposition-spectrum grid over substrate thickness and material, for reading
# any thickness by interpolation (env_grid.py). Runs in WSL.
#
#   bash run_env_grid.sh smoke      # one tiny run per material, prints timing
#   bash run_env_grid.sh full [P]   # grid + held-out points, P workers (12)
#
# Resumable: a job whose output exists is skipped. Outputs land in
# ~/rad/envgrid/out; pack_env_grid.py turns them into the shipped grid.
set -u
MODE=${1:-smoke}
P=${2:-12}
REPO=${RADCHAIN_REPO:?set RADCHAIN_REPO to the repo root}/src/radchain/geant4
WORK=$HOME/rad/envgrid
mkdir -p "$WORK/out" "$WORK/log"
cp "$REPO/g4_env_spectrum.py" "$REPO/g4_l2_bench.py" "$WORK/"
cd "$WORK"
export LD_LIBRARY_PATH=$HOME/rad/sys/root/usr/lib/x86_64-linux-gnu
PY="${RADCHAIN_PY:-$HOME/rad/.venv/bin/python}"
export PY

job() {  # src n t seed material tag
  local src=$1 n=$2 t=$3 s=$4 m=$5 tag=$6
  local out="out/${m}_${src}_${t}_${tag}.json"
  [ -s "$out" ] && { echo "SKIP $out"; return 0; }
  echo Y | "$PY" g4_env_spectrum.py "$src" "$n" "$t" "$s" "$out.tmp" 10.0 "$m" \
     > "log/${m}_${src}_${t}_${tag}.log" 2>&1 && mv "$out.tmp" "$out"
  echo "$m $(tail -1 "log/${m}_${src}_${t}_${tag}.log")"
}
export -f job

if [ "$MODE" = smoke ]; then
  job gamma-lines 20000 0.3 7 G4_Si smoke
  job gamma-lines 20000 0.3 7 G4_ALUMINUM_OXIDE smoke
  job muon 2000 1.0 7 G4_ALUMINUM_OXIDE smoke
  exit 0
fi

GRID="0.05 0.075 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.725 0.85 1.0 1.25 1.5 2.0"
HOLD="0.35 0.675 1.1"
{
  mi=0
  for m in G4_Si G4_ALUMINUM_OXIDE; do
    mi=$((mi + 1)); ti=0
    for t in $GRID $HOLD; do
      ti=$((ti + 1))
      # Silicon at the three original thicknesses already exists in data/env and
      # is reused unchanged, so Willow's graded inputs read the same data.
      if [ "$m" = G4_Si ] && { [ "$t" = 0.3 ] || [ "$t" = 0.5 ] || [ "$t" = 0.725 ]; }; then
        continue
      fi
      for k in 1 2 3 4; do
        echo "gamma-lines 1000000 $t $((100000*mi + 1000*ti + 10*k + 1)) $m s$k"
        echo "gamma-flat 1000000 $t $((100000*mi + 1000*ti + 10*k + 2)) $m s$k"
      done
      echo "muon 60000 $t $((100000*mi + 1000*ti + 3)) $m s1"
    done
  done
} | xargs -P "$P" -L 1 bash -c 'job "$@"' _
echo "ALL DONE $(date -Is)"
