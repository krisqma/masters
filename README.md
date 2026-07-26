# Wilga

Wilga to lokalny asystent głosowy po polsku dla scenariusza smart-home. Projekt łączy:

- frontend React do nagrywania głosu i podglądu rozmowy,
- backend FastAPI z lokalną transkrypcją `faster-whisper`,
- osobny `Context Service`, który analizuje dane z czujników i buduje kontekst dla modelu,
- dwa modele Ollama:
  - `Teacher` analizujący stan mieszkania,
  - `Student` odpowiadający użytkownikowi.

Repo zawiera kompletny przepływ od mikrofonu do odpowiedzi modelu, z dokładaniem kontekstu z danych IoT.

## Co robi aplikacja

1. Użytkownik mówi `hej wilga` albo uruchamia nagrywanie przyciskiem.
2. Frontend nagrywa audio i kończy nagranie po chwili ciszy.
3. `whisper-proj/backend` transkrybuje mowę lokalnie przez `faster-whisper`.
4. Backend pobiera aktualny kontekst z `context_service`.
5. Backend wysyła pytanie do modelu Ollama `Student` i streamuje odpowiedź do UI.
6. `context_service` cyklicznie bierze jeden snapshot z CSV z czujnikami, wysyła go do modelu `Teacher` i buduje z tego `system prompt` dla `Student`.

## Struktura repo

```text
.
├── context_service/        # Teacher + analiza danych z czujników
├── db_sampler/             # CSV z danymi smart-home
├── whisper-proj/           # frontend React + backend Whisper/chat
├── architecture.png        # diagram architektury
├── benchmark_spec.pdf      # materiał pomocniczy do benchmarku/specyfikacji
└── start.sh                # uruchomienie całości lokalnie (macOS)
```

## Architektura

### `context_service`

- FastAPI na porcie `8001`
- ładuje CSV `db_sampler/sensor_data_2025-10-31_2026-01-27.csv`
- symuluje kolejne odczyty czujników, przesuwając kursor po jednym wierszu
- co `CONTEXT_REFRESH_INTERVAL` sekund wysyła snapshot do modelu `Teacher`
- wystawia:
  - `GET /api/health`
  - `GET /api/context`

### `whisper-proj/backend`

- FastAPI na porcie `8000`
- `POST /api/transcribe`:
  - przyjmuje audio,
  - transkrybuje je przez `faster-whisper`
- `POST /api/chat`:
  - pobiera kontekst z `context_service`,
  - otwiera stream do Ollama `Student`,
  - zwraca frontendowi NDJSON ze zdarzeniami `chunk`, `error`, `done`

### `whisper-proj`

- React + Vite na porcie `5173`
- obsługuje:
  - wake word `hej wilga` przez Web Speech API,
  - ręczne nagrywanie,
  - auto-stop po ciszy,
  - konsolowy podgląd rozmowy na żywo

## Wymagania

- macOS dla `start.sh`
- Python `3.9+`
- Node.js `18+`
- `npm`
- `ffmpeg` dostępny w systemie
- zainstalowana Ollama CLI
- dostęp do modelu `Teacher` w Ollamie
- dostęp do modelu `Student` w Ollamie

Domyślny układ z kodu:

- `Teacher` działa lokalnie na Macu: `http://localhost:11434/api/chat`
- `Student` działa pod osobnym adresem Ollama, domyślnie w sieci lokalnej: `http://192.168.1.173:11434/api/chat`

## Szybki start

### 1. Przygotuj pliki konfiguracyjne

```bash
cp context_service/.env.example context_service/.env
cp whisper-proj/.env.example whisper-proj/.env.local
cp whisper-proj/backend/.env.example whisper-proj/backend/.env
```

Najważniejsze pola:

- `context_service/.env`
  - `TEACHER_OLLAMA_URL`
  - `TEACHER_OLLAMA_MODEL`
  - `CONTEXT_REFRESH_INTERVAL`
  - `SENSOR_DATA_PATH`
- `whisper-proj/backend/.env`
  - `WHISPER_MODEL_SIZE`
  - `OLLAMA_URL`
  - `OLLAMA_MODEL`
  - `CONTEXT_SERVICE_URL`
- `whisper-proj/.env.local`
  - `VITE_API_URL`

### 2. Przygotuj Python venv dla Whisper Backend

`start.sh` sam tworzy venv tylko dla `context_service`. Dla `whisper-proj` trzeba to zrobić wcześniej:

```bash
cd whisper-proj
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r backend/requirements.txt
cd ..
```

### 3. Zainstaluj frontend

```bash
cd whisper-proj
npm install
cd ..
```

### 4. Upewnij się, że modele Ollama istnieją

Skrypt startowy sprawdza dostępność obu endpointów Ollama, a dla lokalnego `Teacher` dodatkowo sprawdza obecność modelu. Dla domyślnej konfiguracji `Teacher` potrzebny jest lokalnie co najmniej:

```bash
ollama pull qwen2.5
```

Dla `Student` model i endpoint muszą być dostępne zgodnie z `whisper-proj/backend/.env`.

### 5. Uruchom całość

```bash
chmod +x start.sh
./start.sh
```

Po starcie:

- frontend: `http://127.0.0.1:5173`
- Whisper Backend: `http://127.0.0.1:8000`
- Context Service: `http://127.0.0.1:8001`

Przy pierwszym wejściu do frontendu trzeba zezwolić przeglądarce na użycie mikrofonu. Zatrzymanie całego stacku: `Ctrl+C`.

## Uruchamianie ręczne

Przydaje się przy debugowaniu poszczególnych elementów.

### Context Service

```bash
cd context_service
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8001 --log-level info
```

### Whisper Backend

```bash
cd whisper-proj/backend
../.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-level info
```

### Frontend

```bash
cd whisper-proj
npm run dev
```

## Kontekst z czujników

`context_service` nie korzysta z live feedu, tylko z datasetu CSV i symuluje przebieg czasu:

- źródło: `db_sampler/sensor_data_2025-10-31_2026-01-27.csv`
- jeden refresh = jeden kolejny wiersz CSV
- po dojściu do końca pliku symulacja zawija się do początku
- interesujące metryki obejmują m.in. temperaturę, wilgotność, moc, energię, obecność, kontakt i oświetlenie

Teacher generuje krótkie podsumowanie stanu mieszkania po polsku, a z tego budowany jest `system prompt` dla asystenta.
Do momentu pierwszego odświeżenia endpoint `/api/context` może zwracać status `pending`.

## API

### Context Service

- `GET /`
- `GET /api/health`
- `GET /api/context`

Przykładowa odpowiedź gotowego kontekstu:

```json
{
  "status": "ready",
  "source_timestamp": "2025-11-01T12:00:00Z",
  "insight": "...",
  "system_prompt": "..."
}
```

### Whisper Backend

- `GET /`
- `GET /api/health`
- `POST /api/transcribe`
- `POST /api/chat`
- Swagger: `http://127.0.0.1:8000/docs`

`POST /api/chat` zwraca strumień NDJSON, np.:

```json
{"type":"chunk","content":"Cześć"}
{"type":"chunk","content":"! Jak mogę pomóc?"}
{"type":"done"}
```

## Ograniczenia i uwagi

- Wake word działa tylko tam, gdzie przeglądarka wspiera Web Speech API. Bez tego aplikacja przełącza się na tryb ręczny.
- Pierwsze uruchomienie `faster-whisper` może potrwać dłużej.
- Backend Whisper wymaga systemowego `ffmpeg`.
- `start.sh` jest napisany pod macOS.
- W repo nie ma jeszcze widocznego zestawu testów automatycznych, więc główna weryfikacja jest runtime'owa.

## Przydatne pliki

- `start.sh` - najszybsza droga do uruchomienia całego stacku
- `architecture.png` - diagram poglądowy
- `whisper-proj/README.md` - README skupione na samym frontendzie i backendzie Whisper
- `whisper-proj/roadmap.md` - opis założeń i flow
