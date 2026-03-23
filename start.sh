#!/usr/bin/env bash
# =============================================================================
#  Wilga — skrypt uruchomieniowy (Mac)
#
#  Startuje 3 serwisy:
#    1. Context Service  (port 8001) — Teacher + dane z czujników
#    2. Whisper Backend   (port 8000) — STT + chat proxy do Ucznia
#    3. Frontend          (port 5173) — React dev server
#
#  Wymagania:
#    - Ollama musi działać w tle (brew services start ollama)
#    - Python 3.9+
#    - Node.js 18+
#
#  Użycie:
#    chmod +x start.sh
#    ./start.sh
#
#  Ctrl+C zatrzymuje wszystkie serwisy.
# =============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTEXT_DIR="$ROOT_DIR/context_service"
WHISPER_DIR="$ROOT_DIR/whisper-proj"
BACKEND_DIR="$WHISPER_DIR/backend"

# Kolory
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${GREEN}[WILGA]${NC} $1"; }
warn() { echo -e "${YELLOW}[WILGA]${NC} $1"; }
err()  { echo -e "${RED}[WILGA]${NC} $1"; }

# PID-y procesów w tle
PIDS=()

cleanup() {
    echo ""
    log "Zatrzymuję serwisy..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    log "Gotowe. Do zobaczenia!"
    exit 0
}
trap cleanup SIGINT SIGTERM

# ---------------------------------------------------------------------------
# 1. Sprawdź Ollamę
# ---------------------------------------------------------------------------
log "Sprawdzam Ollamę..."
if ! curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    err "Ollama nie odpowiada na localhost:11434."
    err "Uruchom ją: brew services start ollama"
    exit 1
fi
log "Ollama działa."

# ---------------------------------------------------------------------------
# 2. Przygotuj venv dla Context Service (jeśli nie istnieje)
# ---------------------------------------------------------------------------
if [ ! -d "$CONTEXT_DIR/.venv" ]; then
    log "Tworzę venv dla Context Service..."
    python3 -m venv "$CONTEXT_DIR/.venv"
    "$CONTEXT_DIR/.venv/bin/pip" install --quiet --upgrade pip
    "$CONTEXT_DIR/.venv/bin/pip" install --quiet -r "$CONTEXT_DIR/requirements.txt"
    log "Venv Context Service gotowy."
else
    log "Venv Context Service istnieje."
fi

# ---------------------------------------------------------------------------
# 3. Sprawdź venv dla Whisper Backend
# ---------------------------------------------------------------------------
if [ ! -d "$WHISPER_DIR/.venv" ]; then
    err "Brak venv w $WHISPER_DIR/.venv — uruchom najpierw:"
    err "  cd $WHISPER_DIR && python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt"
    exit 1
fi

# ---------------------------------------------------------------------------
# 4. Startuj Context Service (port 8001)
# ---------------------------------------------------------------------------
log "Startuję ${BLUE}Context Service${NC} na porcie 8001..."
(
    cd "$CONTEXT_DIR"
    .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8001 --log-level info
) &
PIDS+=($!)

# Daj mu chwilę na start
sleep 2

# ---------------------------------------------------------------------------
# 5. Startuj Whisper Backend (port 8000)
# ---------------------------------------------------------------------------
log "Startuję ${BLUE}Whisper Backend${NC} na porcie 8000..."
(
    cd "$BACKEND_DIR"
    "$WHISPER_DIR/.venv/bin/uvicorn" app.main:app --host 127.0.0.1 --port 8000 --log-level info
) &
PIDS+=($!)

# ---------------------------------------------------------------------------
# 6. Startuj Frontend (port 5173)
# ---------------------------------------------------------------------------
if [ -f "$WHISPER_DIR/package.json" ]; then
    log "Startuję ${BLUE}Frontend${NC} na porcie 5173..."
    (
        cd "$WHISPER_DIR"
        npm run dev 2>&1
    ) &
    PIDS+=($!)
else
    warn "Brak package.json w $WHISPER_DIR — pomijam frontend."
fi

# ---------------------------------------------------------------------------
# 7. Gotowe
# ---------------------------------------------------------------------------
echo ""
log "========================================="
log "  Wilga uruchomiona!"
log ""
log "  Context Service:  http://127.0.0.1:8001"
log "  Whisper Backend:  http://127.0.0.1:8000"
log "  Frontend:         http://127.0.0.1:5173"
log ""
log "  Ctrl+C aby zatrzymać wszystko"
log "========================================="
echo ""

# Czekaj na wszystkie procesy
wait
