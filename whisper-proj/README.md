# Whisper + Ollama Streaming (React + FastAPI)

Aplikacja voice-to-text z wake word (`hej wilga`) i strumieniowaną odpowiedzią LLM.

## Architektura

1. Frontend (React) nagrywa audio i wysyła je do `POST /api/transcribe`.
2. Backend (FastAPI) transkrybuje audio lokalnie przez `faster-whisper`.
3. Frontend dodaje wiadomość użytkownika i wywołuje `POST /api/chat`.
4. Backend streamuje odpowiedź z Ollama (`stream: true`) i przekazuje chunki do frontendu.
5. Frontend dokleja chunki na żywo do wiadomości `wilga`.

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
OLLAMA_MODEL=wilgus-pl
```

## Uruchomienie

### 1. Backend

```bash
cd /Users/krzysztof/Documents/studia/magisterka/whisper-proj
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Swagger:

- `http://127.0.0.1:8000/docs`

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
- `POST /api/chat` (`application/json`, body: `{"message":"..."}`) -> streaming text

## Uwaga

- Backend wymaga `ffmpeg` dostępnego w systemie.
- Pierwsze uruchomienie Whisper może pobrać model i potrwać dłużej.
