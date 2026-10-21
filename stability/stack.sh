#!/usr/bin/env bash
# Start the Wilga stack for a stability run, every service logging into RUN_DIR.
# Same commands as ../start.sh, plus: logging proxy in front of the RPi Ollama,
# backend pointed at it via OLLAMA_URL env (the .env file stays untouched),
# and uvicorn --log-config so app-level INFO logs (Chat done, keep-warm) are printed (UTC).
# Usage: stack.sh RUN_DIR      (Ctrl+C / SIGTERM stops everything)
set -u
RUN="$(cd "$1" && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
RPI="${RPI:-192.168.1.173}"
export TZ=UTC
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; exit 0; }
trap cleanup INT TERM

if ! curl -sf http://localhost:11434/api/tags >/dev/null; then
  ollama serve >> "$RUN/mac_ollama.log" 2>&1 &
  PIDS+=($!)
  for i in $(seq 1 25); do curl -sf http://localhost:11434/api/tags >/dev/null && break; sleep 1; done
fi
echo "$(date -u +%FT%T) warming teacher" >> "$RUN/stack.log"
curl -s --max-time 300 http://localhost:11434/api/chat -d '{"model":"qwen2.5","messages":[{"role":"user","content":"OK"}],"stream":false,"keep_alive":"30m"}' >/dev/null

( cd "$HERE" && UPSTREAM="http://$RPI:11434" PROXY_LOG="$RUN/proxy.jsonl" \
  "$ROOT/whisper-proj/.venv/bin/uvicorn" ollama_proxy:app --host 127.0.0.1 --port 11500 \
  --log-config "$HERE/logging.json" ) >> "$RUN/proxy_stdout.log" 2>&1 &
PIDS+=($!)

( cd "$ROOT/context_service" && .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8001 \
  --log-config "$HERE/logging.json" ) >> "$RUN/context_service.log" 2>&1 &
PIDS+=($!)
sleep 2

# Fixed-backend settings (same values as backend/.env.example); process env wins
# over the local .env, so the run does not depend on what that file still holds.
FIXED_ENV=()
if [ "${STACK_FIXED:-0}" = "1" ]; then
  FIXED_ENV=(OLLAMA_KEEP_ALIVE=-1 OLLAMA_READ_TIMEOUT_SECONDS=180 STUDENT_KEEP_WARM_INTERVAL_SECONDS=60
    CHAT_HISTORY_MAX_MESSAGES=20 STUDENT_CTX_WINDOW=1024 STUDENT_BUDGET_MARGIN=32
    STUDENT_NUM_PREDICT_MAX=128 STUDENT_NUM_PREDICT_MIN=64 STUDENT_HISTORY_TRIM_RATIO=0.5 STUDENT_HISTORY_KEEP_MAX_TOKENS=120
    STUDENT_TEMPERATURE=0.2 STUDENT_TOP_P=0.9 STUDENT_REPEAT_PENALTY=1.05 STUDENT_NUM_CTX=2048)
fi

( cd "$ROOT/whisper-proj/backend" && env ${FIXED_ENV[@]+"${FIXED_ENV[@]}"} OLLAMA_URL="http://127.0.0.1:11500/api/chat" \
  "$ROOT/whisper-proj/.venv/bin/uvicorn" app.main:app --host 127.0.0.1 --port 8000 \
  --log-config "$HERE/logging.json" ) >> "$RUN/backend.log" 2>&1 &
PIDS+=($!)

echo "$(date -u +%FT%T) stack started pids=${PIDS[*]}" >> "$RUN/stack.log"
wait
