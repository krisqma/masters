#!/usr/bin/env bash
# Background telemetry for a stability run. Usage: monitor.sh RUN_DIR
# Read-only towards the RPi: ping, :9000/metrics, /api/ps, journalctl -f.
set -u
RUN="$1"
RPI="${RPI:-192.168.1.173}"
mkdir -p "$RUN"
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; exit 0; }
trap cleanup INT TERM

# Wi-Fi reachability (1 Hz) with wall-clock timestamps.
( ping -i 1 "$RPI" 2>&1 | while IFS= read -r line; do
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%S)" "$line"; done ) >> "$RUN/ping.log" &
PIDS+=($!)

# RPi temperature / RAM / throttling every 5 s.
( while true; do
    m="$(curl -s -m 3 "http://$RPI:9000/metrics" || echo '{"error":"unreachable"}')"
    printf '{"ts":"%s","m":%s}\n' "$(date -u +%Y-%m-%dT%H:%M:%S)" "${m:-null}"
    sleep 5
  done ) >> "$RUN/rpi_metrics.jsonl" &
PIDS+=($!)

# Loaded models / context / expiry every 10 s.
( while true; do
    p="$(curl -s -m 3 "http://$RPI:11434/api/ps" || echo '{"error":"unreachable"}')"
    printf '{"ts":"%s","ps":%s}\n' "$(date -u +%Y-%m-%dT%H:%M:%S)" "${p:-null}"
    sleep 10
  done ) >> "$RUN/rpi_ps.jsonl" &
PIDS+=($!)

# Ollama journal (model loads, truncation, errors), reconnecting if SSH drops.
( while true; do
    ssh -o BatchMode=yes -o ServerAliveInterval=10 "krisqma@$RPI" \
      'journalctl -u ollama -f -n 0 -o short-iso' 2>&1
    echo "$(date -u +%Y-%m-%dT%H:%M:%S) [monitor] journal ssh dropped, reconnecting"
    sleep 3
  done ) >> "$RUN/rpi_ollama_journal.log" &
PIDS+=($!)

# vcgencmd throttle bits every 30 s (metrics endpoint only gives a boolean).
( while true; do
    t="$(ssh -o BatchMode=yes "krisqma@$RPI" 'vcgencmd measure_temp; vcgencmd get_throttled; vcgencmd measure_clock arm' 2>&1 | tr '\n' ' ')"
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%S)" "$t"
    sleep 30
  done ) >> "$RUN/rpi_vcgencmd.log" &
PIDS+=($!)

wait
