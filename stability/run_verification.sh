#!/usr/bin/env bash
# Final verification on the fixed stack: V1 (long conversation), V2 (session
# rotation, question during warm-up), V3 (abort / parallel / long answer), soak.
# Usage: run_verification.sh RUN_DIR [SOAK_MINUTES]
set -u
RUN="$1"; SOAK="${2:-30}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../whisper-proj/.venv/bin/python"
for sc in v1 v2 v3; do
  echo "$(date -u +%T) start $sc"
  "$PY" -u -W ignore "$HERE/driver.py" --protocol fixed --scenario "$HERE/scenarios/$sc.json" \
    --out "$RUN/driver.jsonl" --proxy-log "$RUN/proxy.jsonl" > "$RUN/driver_${sc}_stdout.log" 2>&1
  echo "$(date -u +%T) end $sc rc=$?"
done
echo "$(date -u +%T) start soak ${SOAK}m"
"$PY" -u -W ignore "$HERE/driver.py" --protocol fixed --scenario "$HERE/scenarios/s5.json" \
  --soak-minutes "$SOAK" --seed 7 --out "$RUN/driver.jsonl" --proxy-log "$RUN/proxy.jsonl" \
  > "$RUN/driver_soak_stdout.log" 2>&1
echo "$(date -u +%T) end soak rc=$?"
