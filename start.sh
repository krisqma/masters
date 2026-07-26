#!/usr/bin/env bash
# =============================================================================
#  Wilga — skrypt uruchomieniowy (Mac)
#
#  Startuje 4 procesy:
#    1. Lokalna Ollama    (port 11434) — Teacher model
#    2. Context Service   (port 8001)  — Teacher + dane z czujników
#    3. Whisper Backend   (port 8000)  — STT + chat proxy do Ucznia
#    4. Frontend          (port 5173)  — React dev server
#
#  Wymagania:
#    - Python 3.9+
#    - Node.js 18+
#    - Ollama CLI zainstalowana lokalnie
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

read_env_value() {
    local file="$1"
    local key="$2"
    if [ ! -f "$file" ]; then
        return 1
    fi

    local line
    line="$(grep -E "^${key}=" "$file" | tail -n 1 || true)"
    if [ -z "$line" ]; then
        return 1
    fi

    line="${line#*=}"
    line="${line%\"}"
    line="${line#\"}"
    line="${line%\'}"
    line="${line#\'}"
    printf '%s\n' "$line"
}

check_ollama_url() {
    local label="$1"
    local chat_url="$2"
    local base_url="${chat_url%/api/chat}"
    local tags_url="${base_url%/}/api/tags"

    log "Sprawdzam Ollamę dla ${label}: ${tags_url}"
    if ! curl -sf "$tags_url" > /dev/null 2>&1; then
        err "Ollama dla ${label} nie odpowiada pod ${tags_url}."
        exit 1
    fi
}

is_local_ollama_url() {
    local chat_url="$1"
    case "$chat_url" in
        http://localhost:*|http://127.0.0.1:*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

wait_for_ollama() {
    local label="$1"
    local chat_url="$2"
    local base_url="${chat_url%/api/chat}"
    local tags_url="${base_url%/}/api/tags"
    local attempts="${3:-20}"

    for ((i=1; i<=attempts; i+=1)); do
        if curl -sf "$tags_url" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done

    err "Ollama dla ${label} nie wystartowała pod ${tags_url}."
    exit 1
}

ensure_ollama_model() {
    local model="$1"
    if ! ollama list | awk 'NR>1 {print $1}' | grep -Eq "^${model}(:|$)"; then
        err "Brak modelu Ollama '${model}' lokalnie."
        err "Pobierz go najpierw: ollama pull ${model}"
        exit 1
    fi
}

warmup_ollama_model() {
    local label="$1"
    local chat_url="$2"
    local model="$3"
    local keep_alive="${4:-30m}"

    log "Rozgrzewam model ${label} '${model}' (keep_alive=${keep_alive})..."
    # Cold load on RPi can take a few minutes for ~3GB GGUF.
    if ! curl -sf --connect-timeout 10 --max-time 300 "$chat_url" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"OK\"}],\"stream\":false,\"keep_alive\":\"${keep_alive}\"}" \
        > /dev/null 2>&1; then
        err "Nie udało się załadować modelu ${label} '${model}' przez ${chat_url}."
        exit 1
    fi
}

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
# 1. Uruchom / sprawdź Ollamę zgodnie z konfiguracją serwisów
# ---------------------------------------------------------------------------
TEACHER_OLLAMA_URL="$(read_env_value "$CONTEXT_DIR/.env" "TEACHER_OLLAMA_URL" || printf '%s' 'http://localhost:11434/api/chat')"
TEACHER_OLLAMA_MODEL="$(read_env_value "$CONTEXT_DIR/.env" "TEACHER_OLLAMA_MODEL" || printf '%s' 'qwen2.5')"
STUDENT_OLLAMA_URL="$(read_env_value "$BACKEND_DIR/.env" "OLLAMA_URL" || printf '%s' 'http://192.168.1.173:11434/api/chat')"
STUDENT_OLLAMA_MODEL="$(read_env_value "$BACKEND_DIR/.env" "OLLAMA_MODEL" || printf '%s' 'gemma3:4b')"

if is_local_ollama_url "$TEACHER_OLLAMA_URL"; then
    if ! command -v ollama > /dev/null 2>&1; then
        err "Brak polecenia 'ollama' w PATH."
        exit 1
    fi

    if curl -sf "${TEACHER_OLLAMA_URL%/api/chat}/api/tags" > /dev/null 2>&1; then
        log "Lokalna Ollama już działa."
    else
        log "Uruchamiam lokalną Ollamę dla Context Service..."
        ollama serve > /tmp/wilga-ollama.log 2>&1 &
        PIDS+=($!)
        wait_for_ollama "Context Service" "$TEACHER_OLLAMA_URL" 25
        log "Lokalna Ollama wystartowała."
    fi

    ensure_ollama_model "$TEACHER_OLLAMA_MODEL"
    warmup_ollama_model "nauczyciela" "$TEACHER_OLLAMA_URL" "$TEACHER_OLLAMA_MODEL" "30m"
else
    check_ollama_url "Context Service" "$TEACHER_OLLAMA_URL"
fi

check_ollama_url "Whisper Backend" "$STUDENT_OLLAMA_URL"
# Student prefix-aware warmup + keep-warm loop live in Whisper Backend startup
# (same static_prefix as production chat). Avoid a second dumb "OK" warmup here.
log "Ollama Studenta dostępna (${STUDENT_OLLAMA_MODEL}); keep-warm wystartuje z backendem."

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
