# Roadmap: Voice-to-Text (Whisper) + LLM (Ollama na RPi) ze Streamowaniem

## 1. Architektura i Przepływ Danych (Data Flow)
Aplikacja składa się z trzech głównych węzłów:
1. **Frontend (React):** Odpowiada za UI, nasłuchiwanie "hej wilguś", nagrywanie audio i renderowanie odpowiedzi.
2. **Backend (Python / FastAPI):** Działa jako orkiestrator. Obsługuje lokalny model Whisper do transkrypcji oraz działa jako proxy dla zewnętrznego LLM.
3. **LLM (Raspberry Pi / Ollama):** Serwer w sieci lokalnej (`192.168.1.173:11434`) obsługujący model `gemma3:4b` (Q4_K_M, `keep_alive=30m`).

**Flow działania:**
1. Użytkownik mówi -> React nagrywa audio i wysyła je (`multipart/form-data`) na endpoint backendu `/api/transcribe`.
2. FastAPI przepuszcza audio przez lokalny model Whisper i zwraca gotowy tekst (transkrypcję) do Reacta.
3. React od razu wyświetla transkrypcję w UI (jako wiadomość użytkownika).
4. React natychmiast wysyła ten sam tekst na drugi endpoint backendu: `/api/chat` (POST z JSONem).
5. FastAPI otwiera połączenie z Ollamą na RPi, wysyłając prompt z flagą `"stream": true`.
6. FastAPI odbiera fragmenty odpowiedzi z RPi i natychmiast przesyła je dalej do Reacta jako typowany strumień NDJSON.
7. React na bieżąco dokleja nowe słowa do UI, tworząc płynny efekt pisania.
8. Równolegle frontend tnie odpowiedź na zdania i woła `POST /api/tts` (`edge-tts`); odtwarza MP3 w kolejce ze statusem `SPEAKING` (wake word wyłączony).

## 2. Zmienne Środowiskowe
- **Frontend (`.env.local`):**
  - `VITE_API_URL=http://127.0.0.1:8000` (Adres lokalnego backendu w Pythonie)
- **Backend (`.env`):**
  - `OLLAMA_URL=http://192.168.1.173:11434/api/chat`
  - `OLLAMA_MODEL=gemma3:4b`
  - `OLLAMA_KEEP_ALIVE=30m`

## 3. Zadania dla Backendu (Python / FastAPI)
1. **Endpoint `/api/transcribe` (POST):**
   - Przyjmuje plik audio.
   - Używa `faster-whisper` (lub innej lokalnej biblioteki Whisper), aby zmienić mowę na tekst.
   - Zwraca JSON: `{"text": "Rozpoznany tekst z mowy"}`.
2. **Endpoint `/api/chat` (POST):**
   - Przyjmuje JSON: `{"message": "Rozpoznany tekst z mowy"}`.
   - Używa asynchronicznego klienta HTTP (np. `httpx` w Pythonie), aby wysłać zapytanie do `OLLAMA_URL`.
   - Parametry dla Ollamy: `{"model": "gemma3:4b", "messages": [{"role": "system", "content": "<stałe reguły>"}, {"role": "user", "content": "<kontekst + pytanie>"}], "stream": true, "keep_alive": "30m"}`.
   - Zwraca do frontendu `StreamingResponse` (z `fastapi.responses`), iterując po chunkach danych przychodzących z Ollamy i wysyłając je jako strumień tekstu.

## 4. Zadania dla Frontendu (React / TypeScript)
1. **Zarządzanie stanem wiadomości:** - Utworzenie tablicy wiadomości w formacie `[{ role: 'user' | 'wilga', content: string }]`.
2. **Komunikacja z Backendem:**
   - Utworzenie funkcji wysyłającej `Blob` z audio do `/api/transcribe`.
   - Po otrzymaniu transkrypcji: dodanie wiadomości `role: 'user'` do stanu i wywołanie strumieniowania.
3. **Odbieranie Strumienia (Streaming):**
   - Utworzenie funkcji uderzającej do `/api/chat` używając natywnego API `fetch`.
   - Czytanie strumienia odpowiedzi za pomocą `response.body.getReader()`.
   - W miarę dekodowania kolejnych chunków (`TextDecoder`), aktualizowanie w czasie rzeczywistym ostatniej wiadomości (`role: 'wilga'`) w stanie aplikacji.
4. **Rozszerzenie UI:**
   - Dodanie auto-scrollowania w `ConsoleWindow`, aby użytkownik zawsze widział najnowsze, pojawiające się słowa.
   - Odpowiednie blokowanie nasłuchiwania "hej wilguś", gdy model aktualnie generuje i streamuje odpowiedź (aby aplikacja nie przerwała mu w połowie słowa).
