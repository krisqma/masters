# Whisper + Ollama Streaming + edge-tts (React + FastAPI)

Aplikacja głosowa z wake word (`hej wilguś`), lokalnym Whisper, strumieniowanym LLM
(`gemma3:4b` na RPi) i syntezą mowy przez `edge-tts`.

## Architektura

1. Frontend (React) nagrywa audio i wysyła je do `POST /api/transcribe`.
2. Backend (FastAPI) transkrybuje audio lokalnie przez `faster-whisper`.
3. Frontend odcina frazę wake word, dodaje wiadomość użytkownika i woła `POST /api/chat`.
4. Backend streamuje odpowiedź z Ollama (`stream: true`, `keep_alive=30m`) jako NDJSON.
5. Frontend dokleja chunki do UI i równolegle syntezuje pełne zdania przez `POST /api/tts`.
6. Context Service dostarcza stały prefix reguł + dynamiczny insight Nauczyciela (prefix cache).

## Zmienne środowiskowe

### Frontend (`.env.local`)

```env
VITE_API_URL=http://127.0.0.1:8000
```

### Backend (`backend/.env`)

```env
WHISPER_MODEL_SIZE=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_LANGUAGE=pl
WHISPER_BEAM_SIZE=5
WHISPER_VAD_FILTER=true
MAX_UPLOAD_MB=25
CORS_ORIGINS=http://127.0.0.1:5173,http://localhost:5173
OLLAMA_URL=http://192.168.1.173:11434/api/chat
OLLAMA_MODEL=gemma3:4b
OLLAMA_KEEP_ALIVE=30m
OLLAMA_READ_TIMEOUT_SECONDS=120
CONTEXT_SERVICE_URL=http://127.0.0.1:8001
TTS_VOICE=pl-PL-ZofiaNeural
TTS_RATE=+0%
TTS_MAX_CHARS=1000
```

## Uruchomienie

Najprościej z katalogu głównego repo:

```bash
./start.sh
```

Albo ręcznie:

### 1. Backend

```bash
cd /Users/krzysztof/Documents/studia/magisterka/whisper-proj
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Swagger: `http://127.0.0.1:8000/docs`

### 2. Frontend

```bash
cd /Users/krzysztof/Documents/studia/magisterka/whisper-proj
npm install
cp .env.example .env.local
npm run dev
```

## Endpointy

- `GET /api/health`
- `POST /api/transcribe` (`multipart/form-data`, pole `file`)
- `POST /api/chat` (`application/json`, body: `{"message":"..."}`) -> streaming NDJSON (`chunk`, `error`, `done`)
- `POST /api/tts` (`application/json`, body: `{"text":"..."}`) -> `audio/mpeg` (edge-tts)

Smoke TTS:

```bash
curl -sS -X POST http://127.0.0.1:8000/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Cześć, tu Wilga."}' \
  --output /tmp/wilga-tts.mp3
```

## Uwaga

- Backend wymaga `ffmpeg` dostępnego w systemie.
- Pierwsze uruchomienie Whisper może pobrać model i potrwać dłużej.
- `edge-tts` wymaga sieci wychodzącej do Microsoft (głosy neuralne).
- Wake word korzysta z Web Speech API (Chrome/Edge).
