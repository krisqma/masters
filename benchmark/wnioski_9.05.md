# Co zmieniłem w benchmarku — notatka

Notatka po sesji 9 maja 2026, między pierwszym pełnym runem (5 triali per model) a finalnym runem (30+30+10+10 triali).

---

## Co było — pierwszy run, `--trials 5`

Komenda:

```bash
python run_benchmark.py --trials 5
```

Wynik: 5 modeli × 5 triali × 20 promptów = 500 rekordów.

### Trzy problemy, które wyszły dopiero przy analizie

**1. `LATENCY_CAP_MS = 20000` — latency nie różnicowało nic.**

Wzór composite (sekcja 9 spec):

```
score = 0.40*faith + 0.30*rel + 0.20*conc + 0.10*(1 - normalized_latency)
gdzie normalized_latency = min(e2e_ms / 20000, 1.0)
```

Ale na RPi żaden model nie schodzi poniżej 15s. Mediany E2E:

- gemma2:2b — 21s
- gemma3:4b — 32s
- phi3.5 — 52s

Wszystkie powyżej 20s → wszystkie dostawały `normalized_latency = 1.0` → wkład latency = `0.10 * 0 = 0`. Optuna myślała, że optymalizuje latency, ale ten składnik był martwy.

**2. TTFT we wszystkich 500 rekordach `null`.**

`_call_ollama` używał `stream: False` — Ollama buforowała całą odpowiedź i zwracała ją jednym JSON. Nie ma sposobu zmierzyć kiedy pojawił się pierwszy token, jeśli widzisz tylko gotową odpowiedź.

A TTFT to pierwsza metryka wydajności w spec (sekcja 3.1, źródło NVIDIA GenAI-Perf — standard branżowy LLM inference). Brak tej metryki w finalnych wynikach to poważna dziura metodologiczna.

**3. Wszystkie modele nazwane "phi3.5" itd. miały wszystkie metryki sprzętowe = 0 albo null.**

Pierwsza wersja `metrics_server.py` nie była uruchomiona / nie była podpięta. Nie wiedziałem czy radiator wystarcza, bo `cpu_temp_c=0` dla wszystkich rekordów. Naprawione w trakcie sesji — drugi run pokazał że throttlingu nie ma (max 73.6°C dla phi3.5, próg RPi to 80°C).

---

## Co zmieniłem

### Zmiana 1: `LATENCY_CAP_MS = 20000` → rozbicie na E2E + TTFT z wyższymi capami

W kodzie:

```python
E2E_NORM_CAP_MS = 90_000     # było 20_000 (jako jeden cap)
TTFT_NORM_CAP_MS = 45_000    # nowe
W_LATENCY_E2E  = W_LATENCY / 2.0   # 0.05
W_LATENCY_TTFT = W_LATENCY / 2.0   # 0.05
```

Nowy wzór composite:

```python
norm_e2e  = min(e2e_ms  / 90_000, 1.0)
norm_ttft = min(ttft_ms / 45_000, 1.0)
latency_bonus = 0.05 * (1 - norm_e2e) + 0.05 * (1 - norm_ttft)
score = 0.40*faith + 0.30*rel + 0.20*conc + latency_bonus
```

**Dlaczego cap E2E = 90s a nie 60s (timeout):** żeby timeouty nie spłaszczały się z bardzo wolnymi ale udanymi odpowiedziami. Jeśli `cap = timeout`, to E2E=59s i E2E=60s wpadają w ten sam punkt skali (1.0 i 1.0) i Optuna ich nie odróżnia.

**Dlaczego TTFT osobno z wagą 0.05:** TTFT i E2E mierzą różne rzeczy. TTFT = czas do pierwszego tokena = faktyczna responsywność dla użytkownika (bo TTS może zacząć syntezować zanim model skończy generować). E2E = długość całej odpowiedzi. Dla asystenta głosowego w smart-home obie metryki są ważne, ale w różnym sensie. Rozdzielenie wagi 0.10 po połowie odzwierciedla że żadna z nich nie powinna dominować.

**Dlaczego cap TTFT = 45s:** połowa cap E2E. Na CPU-only inference TTFT to typowo 1/3 do 1/2 E2E.

### Zmiana 2: streaming + pomiar TTFT

Funkcja `_call_ollama` dostała:

- `stream: True` w payloadzie do Ollamy
- `httpx.stream(...)` zamiast `httpx.post(...)` — żeby czytać chunki NDJSON na bieżąco
- pomiar `ttft_ms` w momencie pierwszego niepustego chunka treści
- nowa sygnatura: zwraca `(answer, e2e_ms, ttft_ms)` zamiast `(answer, e2e_ms)`

W `_run_trial` `ttft_ms` trafia do rekordu JSONL zamiast hardcoded `None`.

### Zmiana 3: `--purge-models` zamiast `rm optuna.db`

Po zmianie wzoru composite stare triale w bazie SQLite są bezużyteczne (liczone na zepsutej funkcji celu). Trzeba zacząć z czystym kontem, ale globalne `rm optuna.db` jest zbyt brutalne — usuwa wszystko, też studia z innych eksperymentów.

`--purge-models gemma3:4b gemma2:2b ...` usuwa tylko studia o nazwach zaczynających się od `{model}_*`, zostawiając resztę.

### Zmiana 4: `--model-trials MODEL=N` — różne liczby triali per model

Dodane bo w obecnym runie chcę:

- gemma3:4b — 30 triali (najlepszy model, warto wycisnąć max z Optuny)
- gemma2:2b — 30 triali (drugi profil: szybszy ale gorszy jakościowo, też jest co optymalizować)
- llama3.2:3b — 10 triali (baseline porównawczy, nie ma co marnować budżetu)
- qwen2.5:3b — 10 triali (jw.)
- phi3.5 — pominięty zupełnie (49% halucynacji w pierwszym runie, nie jest kandydatem)

Pierwotnie `--trials` był jeden globalny dla wszystkich modeli. Teraz można per-model.

### Zmiana 5: `--run-id` — wspólny sufiks `study_name`

Wcześniej `study_name` zawierał `datetime.now()` generowany w konstruktorze runnera. Każde wywołanie skryptu = inny `study_name`, więc nie dało się zgrupować runów per "sesja eksperymentalna" w `optuna-dashboard`.

Teraz `--run-id` pozwala podać wspólny ID dla wielu modeli w jednej komendzie:

```bash
RUN_ID="$(date -u +%Y-%m-%d_benchmark_%H%M)"
python run_benchmark.py --run-id "$RUN_ID" --models "..." --model-trials ...
```

---

## Komenda finalnego runu

```bash
# Wyczyść stare studia tych 4 modeli
python run_benchmark.py --purge-models gemma3:4b gemma2:2b llama3.2:3b qwen2.5:3b

# Wspólny RUN_ID dla całej sesji
RUN_ID="$(date -u +%Y-%m-%d_benchmark_%H%M)"

# Run
python run_benchmark.py --run-id "$RUN_ID" \
  --models "gemma3:4b,gemma2:2b,llama3.2:3b,qwen2.5:3b" \
  --model-trials gemma3:4b=30 \
  --model-trials gemma2:2b=30 \
  --model-trials llama3.2:3b=10 \
  --model-trials qwen2.5:3b=10
```

Łącznie 80 triali × 20 promptów = 1600 ewaluacji. Szacowany czas: ~11–12 godzin. Najlepiej zostawić na noc.

---

## Co zostało do zrobienia poza tym runem

- Po zakończeniu runu **nie zmieniać wag composite retroaktywnie** żeby któryś model wyszedł lepiej. To dyskwalifikuje wyniki metodologicznie. Wagi do analizy końcowej (rekomendacja produkcyjna w pracy) dobierać osobno na surowych metrykach faith/rel/conc/latency, nie przez modyfikację `composite_score` w kodzie.
- W rozdziale metodologicznym pracy opisać rozróżnienie: **composite optymalizacyjny** (wagi ustalone przed runem, dla Optuny) vs **composite analityczny** (wagi dobrane post-hoc, dla finalnej rekomendacji). To są dwie różne rzeczy i warto je rozdzielić wprost.
- phi3.5 wykluczony — w pracy uzasadnić jednym akapitem (49% halucynacji, 25% timeoutów, polish_quality 0.44 w pierwszym runie).
- Rozważyć dodanie polskiego modelu (Bielik?) do kolejnego runu — żaden z obecnych nie jest trenowany specjalnie na polskim, a w spec sekcja 5 wymienia "polski wariant" jako jedną z opcji.
