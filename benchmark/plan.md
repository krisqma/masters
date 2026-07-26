# Plan dla Claude Code — Mac (Benchmark Master)

## Twoja rola
Jesteś agentem działającym na Macu. Twoim zadaniem jest zbudowanie
kompletnego pipeline'u benchmarku od zera w katalogu `benchmark/`
w istniejącym monorepo magisterki.

**Zakładaj że RPi jest gotowe:** Ollama działa na `mill-56-rpi.local:11434`,
metrics_server działa na `mill-56-rpi.local:9000`.

---

## Struktura katalogów do stworzenia

```
benchmark/
├── .env                    # klucz OpenRouter + adresy
├── requirements.txt        # zależności Pythona
├── generate_golden.py      # jednorazowe generowanie golden datasetu
├── run_benchmark.py        # główny orchestrator
├── evaluate.py             # moduł ewaluacji przez Haiku
├── report.py               # generator report.html
├── golden.jsonl            # (generowany przez generate_golden.py)
├── results/                # (tworzony automatycznie)
│   └── *.jsonl
└── report.html             # (generowany przez report.py)
```

---

## Krok 0 — Przygotuj środowisko

```bash
cd <root_monorepo>
mkdir -p benchmark/results
cd benchmark

python3 -m venv .venv
.venv/bin/pip install \
  optuna \
  httpx \
  openai \
  python-dotenv \
  tqdm \
  jinja2
```

Stwórz plik `.env`:
```
OPENROUTER_API_KEY=<klucz_od_uzytkownika>
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

RPI_OLLAMA_URL=http://mill-56-rpi.local:11434
RPI_METRICS_URL=http://mill-56-rpi.local:9000

OPUS_MODEL=anthropic/claude-opus-4
HAIKU_MODEL=anthropic/claude-haiku-4-5
```

---

## Krok 1 — generate_golden.py

Skrypt jednorazowy. Wywołuje Claude Opus przez OpenRouter i generuje
`golden.jsonl` z 20 parami testowymi.

### Wymagania:

**Używaj OpenAI-compatible klienta** (`openai` library) z base_url OpenRouter.

**Podaj Opusowi następujący system prompt:**
```
Jesteś ekspertem od ewaluacji systemów RAG dla asystentów smart-home.
Generujesz złoty dataset testowy w formacie JSONL.
Każdy rekord to jeden obiekt JSON na jednej linii.
Odpowiadaj WYŁĄCZNIE surowym JSONL — zero komentarzy, zero markdown.
```

**Podaj Opusowi następujący user prompt** (dokładnie ten tekst, nie skracaj):
```
Wygeneruj dokładnie 20 par testowych dla benchmarku LLM asystenta smart-home po polsku.

Kontekst systemu: Asystent otrzymuje system_prompt z aktualnym stanem mieszkania
(czujniki temperatury, wilgotności, obecności, oświetlenia, mocy, energii)
i odpowiada na pytanie użytkownika krótko, po polsku, na podstawie TYLKO tego co jest w kontekście.

Rozkład 20 przypadków:
- 8x factual_status: pytanie o konkretny fakt ze stanu mieszkania
- 4x missing_data: dane brakują w kontekście — asystent MUSI odmówić
- 3x conflicting_data: sprzeczne dane w kontekście — asystent MUSI wskazać sprzeczność
- 2x stale_data: timestamp starszy niż 2 godziny — asystent MUSI zasygnalizować brak pewności
- 3x paraphrase: te same pytania co w factual_status ale inaczej sformułowane (podaj case_id oryginału w polu paraphrase_of)

Każdy rekord JSON musi zawierać dokładnie te pola:
{
  "case_id": "unikalny_snake_case_id",
  "category": "factual_status|missing_data|conflicting_data|stale_data|paraphrase",
  "system_prompt": "Jesteś asystentem smart-home. [realistyczny stan mieszkania z czujnikami, timestamp UTC]",
  "question": "pytanie użytkownika po polsku",
  "golden_answer": "idealna odpowiedź asystenta — krótka, po polsku, bez halucynacji",
  "paraphrase_of": null
}

Wymagania jakościowe:
- system_prompt musi zawierać realistyczne wartości (temp 18-24°C, wilgotność 30-65%, moc 0-3000W)
- golden_answer dla factual_status: 1-2 zdania, podaj konkretną wartość z kontekstu
- golden_answer dla missing_data: "Nie mam danych o [X] w aktualnym kontekście."
- golden_answer dla conflicting_data: "Kontekst zawiera sprzeczne dane dotyczące [X]."
- golden_answer dla stale_data: "Dane mogą być nieaktualne — ostatni odczyt pochodzi sprzed [N] godzin."
- golden_answer dla paraphrase: identyczna jak oryginał
- Użyj różnych pomieszczeń: salon, sypialnia, kuchnia, łazienka, przedpokój
- Użyj różnych metryk: temperatura, wilgotność, oświetlenie, obecność, moc, energia, kontakt (okno/drzwi)
```

**Parsowanie odpowiedzi:**
- Opus może zwrócić JSONL lub JSON array — obsłuż oba przypadki
- Waliduj każdy rekord: sprawdź że ma wszystkie wymagane pola
- Zapisz do `golden.jsonl` — jeden JSON per linia
- Wydrukuj podsumowanie: ile rekordów per kategoria

**Uruchomienie:**
```bash
.venv/bin/python generate_golden.py
```

Zaraportuj output i pokaż pierwsze 3 rekordy z wygenerowanego pliku.

---

## Krok 2 — evaluate.py

Moduł (nie skrypt) importowany przez orchestrator. Ewaluuje pojedynczą odpowiedź modelu.

### Interfejs publiczny:

```python
def evaluate_response(
    system_prompt: str,      # kontekst z golden datasetu
    question: str,           # pytanie użytkownika
    golden_answer: str,      # oczekiwana odpowiedź
    model_answer: str,       # odpowiedź testowanego modelu
    category: str,           # kategoria przypadku
) -> dict:
    """
    Zwraca:
    {
        "faithfulness": float,       # 0.0-1.0
        "answer_relevancy": float,   # 0.0-1.0
        "conciseness": float,        # 0.0-1.0
        "polish_quality": float,     # 0.0-1.0
        "hallucination_flag": bool,
        "haiku_raw": str             # surowa odpowiedź Haiku do debugowania
    }
    """
```

### Implementacja:

**Używaj OpenAI-compatible klienta** z OpenRouter, model: `HAIKU_MODEL` z `.env`.

**Wywołuj Haiku JEDNYM requestem** z prośbą o wszystkie metryki naraz (nie 5 osobnych callów).

**System prompt dla Haiku:**
```
Jesteś precyzyjnym ewaluatorem odpowiedzi asystenta smart-home.
Oceniasz odpowiedź według podanych kryteriów i zwracasz WYŁĄCZNIE JSON.
Zero komentarzy, zero markdown, tylko surowy obiekt JSON.
```

**User prompt dla Haiku** (konstruuj dynamicznie):
```
Oceń odpowiedź asystenta smart-home.

KONTEKST SYSTEMU (jedyne źródło prawdy):
{system_prompt}

PYTANIE UŻYTKOWNIKA:
{question}

WZORCOWA ODPOWIEDŹ:
{golden_answer}

ODPOWIEDŹ DO OCENY:
{model_answer}

Zwróć JSON z dokładnie tymi polami:
{{
  "faithfulness": <float 0.0-1.0, czy każde twierdzenie jest poparte kontekstem>,
  "answer_relevancy": <float 0.0-1.0, czy odpowiedź adresuje pytanie>,
  "conciseness": <float 0.0-1.0, czy odpowiedź jest zwięzła bez zbędnych informacji>,
  "polish_quality": <float 0.0-1.0, naturalność i poprawność języka polskiego>,
  "hallucination_flag": <true jeśli odpowiedź zawiera fakty spoza kontekstu, false inaczej>,
  "reasoning": "<krótkie uzasadnienie ocen, max 2 zdania>"
}}

Kryteria oceny:
- faithfulness=1.0: każde twierdzenie faktyczne wprost wynika z kontekstu
- faithfulness=0.0: odpowiedź zawiera fakty których nie ma w kontekście
- answer_relevancy=1.0: odpowiedź bezpośrednio i kompletnie odpowiada na pytanie
- conciseness=1.0: odpowiedź jest krótka i nie zawiera zbędnych informacji
- polish_quality=1.0: naturalna polszczyzna, poprawna gramatyka, brak anglicyzmów
- hallucination_flag=true: TYLKO gdy są konkretne fakty spoza kontekstu (nie ogólne zwroty)
```

**Parsowanie:** użyj try/except, jeśli JSON nie parsuje się poprawnie — zaloguj błąd i zwróć domyślne wartości (wszystkie 0.0, hallucination_flag=True jako "bezpieczny default").

**Retry logic:** maksymalnie 3 próby z wykładniczym backoffem (1s, 2s, 4s).

---

## Krok 3 — run_benchmark.py

Główny orchestrator. Uruchamiany na Macu, zarządza wszystkim.

### Stałe konfiguracyjne (na górze pliku):

```python
MODELS = [
    "gemma2:2b",
    "gemma3:4b",
    "llama3.2:3b",
    "qwen2.5:3b",
    "phi3.5",
]

N_TRIALS = 30          # triali Optuny per model
LATENCY_CAP_MS = 20000 # próg latency w ms
REQUEST_TIMEOUT = 60   # timeout pojedynczego requestu do Ollamy w sekundach

# Wagi composite score
W_FAITHFULNESS = 0.40
W_RELEVANCY = 0.30
W_CONCISENESS = 0.20
W_LATENCY = 0.10
```

### Przestrzeń parametrów Optuny (per trial):

```python
temperature = trial.suggest_float("temperature", 0.1, 0.8)
num_ctx     = trial.suggest_categorical("num_ctx", [1024, 2048, 4096])
num_predict = trial.suggest_int("num_predict", 48, 160)
top_p       = trial.suggest_float("top_p", 0.80, 0.99)
repeat_penalty = trial.suggest_float("repeat_penalty", 1.0, 1.15)
```

### Composite score (funkcja):

```python
def composite_score(faithfulness, answer_relevancy, conciseness, e2e_ms, hallucination_flag):
    if hallucination_flag:
        return 0.0
    normalized_latency = min(e2e_ms / LATENCY_CAP_MS, 1.0)
    return (
        W_FAITHFULNESS * faithfulness +
        W_RELEVANCY    * answer_relevancy +
        W_CONCISENESS  * conciseness +
        W_LATENCY      * (1.0 - normalized_latency)
    )
```

### Główna pętla — pseudokod:

```
dla każdego MODEL:
    1. Sprawdź że model jest na RPi (GET /api/tags)
    2. Stwórz Optuna study (direction="maximize", sampler=TPESampler)
    3. Zdefiniuj objective(trial):
        a. Pobierz parametry z trial.suggest_*
        b. trial_scores = []
        c. dla każdego case w golden.jsonl (20 cases):
            - zmierz t_start = time.time()
            - wyślij POST do Ollama /api/chat z:
                messages=[system: case.system_prompt, user: case.question]
                options={temperature, num_ctx, num_predict, top_p, repeat_penalty}
                stream=false
            - zmierz e2e_ms = (time.time() - t_start) * 1000
            - pobierz metryki z /metrics (RAM, temp, throttling)
            - wywołaj evaluate.evaluate_response(...)
            - oblicz score = composite_score(...)
            - zapisz rekord do results/{model}_{timestamp}.jsonl
            - trial_scores.append(score)
        d. return mean(trial_scores)
    4. study.optimize(objective, n_trials=N_TRIALS)
    5. Wydrukuj best trial dla modelu
    6. (NIE usuwaj modelu po trialu — orchestrator nie zarządza modelami na RPi)

po wszystkich modelach:
    wywołaj report.generate_report("results/", "report.html")
```

### Format rekordu JSONL (jeden per case per trial):

```python
{
    "run_id": "2026-04-25_benchmark_01",
    "model": "gemma2:2b",
    "trial": 14,
    "temperature": 0.3,
    "num_ctx": 1024,
    "num_predict": 96,
    "top_p": 0.92,
    "repeat_penalty": 1.05,
    "case_id": "status_temp_01",
    "category": "factual_status",
    "question": "...",
    "golden_answer": "...",
    "model_answer": "...",
    "ttft_ms": null,          # null — niemierzalne bez streamingu
    "e2e_ms": 2140,
    "ram_mb": 1840.5,
    "cpu_temp_c": 67.2,
    "throttling": false,
    "faithfulness": 0.95,
    "answer_relevancy": 0.88,
    "conciseness": 0.91,
    "polish_quality": 0.87,
    "hallucination_flag": false,
    "composite_score": 0.72
}
```

### Ważne szczegóły implementacyjne:
- Zapisuj każdy rekord do JSONL **od razu po obliczeniu** (nie czekaj na koniec trialu)
  — żeby nie stracić danych przy ewentualnym crashu
- Dodaj `tqdm` progress bar: `[MODEL] Trial X/30 | Case Y/20`
- Obsłuż timeout Ollamy: jeśli request trwa > `REQUEST_TIMEOUT`s → zapisz rekord z e2e_ms=REQUEST_TIMEOUT*1000, wszystkie metryki jakości=0.0, hallucination_flag=True
- Jeden plik JSONL per model: `results/gemma2_2b_20260425_120000.jsonl`

---

## Krok 4 — report.py

Generator statycznego `report.html`. Wczytuje wszystkie JSONL z `results/`,
agreguje dane i generuje dashboard.

### Wymagania:

**Biblioteki:** tylko Python stdlib + wbudowanie danych jako JSON w `<script>` tag.
Wizualizacje przez **Chart.js** z CDN (`https://cdn.jsdelivr.net/npm/chart.js`).

**Sekcje dashboardu:**

1. **Overview** — tabela rankingowa modeli:
   - kolumny: Model, Avg Composite Score, Avg Faithfulness, Avg Latency (ms), Best Trial Score
   - posortowana malejąco po Avg Composite Score
   - podświetl najlepszy model w każdej kolumnie

2. **Latency** — box plot per model (E2E ms):
   - pokaż: min, p25, median, p75, max
   - pozioma linia na 20000ms (próg)

3. **Quality** — grouped bar chart:
   - metryki: faithfulness, answer_relevancy, conciseness, polish_quality
   - grupy = modele
   - średnie per model per metryka

4. **Trade-off** — scatter plot:
   - oś X: avg e2e_ms
   - oś Y: avg faithfulness
   - jeden punkt per model, podpisany nazwą modelu

5. **Parameters** — tabela top 3 konfiguracji per model:
   - kolumny: Model, Temperature, num_ctx, num_predict, Composite Score
   - najlepsze 3 trialy per model

6. **Drilldown** — filtrowana tabela wszystkich rekordów:
   - filtry: model, category, hallucination_flag
   - kolumny: case_id, category, model_answer (skrócona), composite_score, hallucination_flag

**Implementacja HTML:**
- Jeden plik, zero zewnętrznych plików CSS/JS poza Chart.js CDN
- Wszystkie dane wbudowane jako `const DATA = {...}` w tagu `<script>`
- Czytelny, responsywny layout (można użyć inline CSS)
- Tytuł strony: "Wilga Benchmark Report — {timestamp}"

---

## Krok 5 — Weryfikacja pipeline'u (dry run)

Zanim puścisz pełny benchmark, zrób dry run:

```bash
# Wygeneruj golden dataset
.venv/bin/python generate_golden.py

# Sprawdź że golden.jsonl ma 20 rekordów
wc -l golden.jsonl

# Zrób test 1 modelu, 2 trialy, żeby sprawdzić że pipeline działa
# (dodaj flagę --dry-run do run_benchmark.py która ustawia N_TRIALS=2 i bierze tylko 3 cases)
.venv/bin/python run_benchmark.py --dry-run

# Sprawdź wynikowy JSONL
head -5 results/*.jsonl | python3 -m json.tool

# Wygeneruj raport z dry run danych
.venv/bin/python report.py
```

Zaraportuj:
- Czy golden.jsonl ma 20 rekordów z właściwym rozkładem kategorii?
- Czy dry run przeszedł bez błędów?
- Czy report.html się otwiera i pokazuje dane?

---

## Krok 6 — Pełny benchmark

Jeśli dry run przeszedł poprawnie:

```bash
.venv/bin/python run_benchmark.py
```

Benchmark trwa ~7 godzin. Nie musisz go pilnować.
Po zakończeniu wygeneruj finalny raport:

```bash
.venv/bin/python report.py
open report.html
```

---

## Czego NIE rób
- Nie modyfikuj niczego na RPi
- Nie usuwaj modeli z RPi między trialami
- Nie używaj `anthropic` SDK — używaj `openai` library z OpenRouter base_url
- Nie generuj golden datasetu więcej niż raz (chyba że wyraźnie poproszone)
- Nie commituj `.env` do gita (dodaj do `.gitignore`)
