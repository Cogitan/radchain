#!/usr/bin/env bash
# Smoke test for g4_moshel_dies.py (minutes). Runs in WSL.
set -u
REPO=${RADCHAIN_REPO:?set RADCHAIN_REPO to the repo root}/src/radchain/geant4
WORK=$HOME/rad/moshel
mkdir -p "$WORK/out" "$WORK/log"
cp "$REPO/g4_moshel_dies.py" "$REPO/g4_env_spectrum.py" "$REPO/g4_cryostat_spectrum.py" "$WORK/"
cp "$REPO"/g4_l2_bench.py "$REPO"/g4_species_spectrum.py "$WORK/" 2>/dev/null
cd "$WORK"
export LD_LIBRARY_PATH=$HOME/rad/sys/root/usr/lib/x86_64-linux-gnu
PY="${RADCHAIN_PY:-$HOME/rad/.venv/bin/python}"
echo Y | timeout 900 "$PY" g4_moshel_dies.py gamma-lines 200000 out/smoke_gl2.json --seed 1 --check-overlaps --dump-couples > log/smoke_gl2.log 2>&1
grep -c "OK!" log/smoke_gl2.log; grep -i -E "overlap.*(fail|detect)|DONE|Error|Traceback" log/smoke_gl2.log | head
echo Y | timeout 900 "$PY" g4_moshel_dies.py muon 20000 out/smoke_mu2.json --seed 2 --cryostat > log/smoke_mu2.log 2>&1
grep -i -E "DONE|Error|Traceback" log/smoke_mu2.log | tail -5
grep -E "Material :|Range cuts|Region\(s\)" log/smoke_gl2.log | head -40
