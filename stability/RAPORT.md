# Wilga: diagnostyka stabilności (20.10.2026)

Cel: wywołać zgłaszany objaw („pierwsze 3-4 odpowiedzi w 6-10 s, potem 40 s, timeouty, zawieszenie”) i wyjaśnić go z logów. **Bez napraw**: kod aplikacji, `.env` i RPi są nietknięte.

## TL;DR

Objaw jest w pełni powtarzalny i ma jedną główną przyczynę oraz kilka wyzwalaczy.

**Przyczyna:** gemma3 ma w części warstw okno przesuwne (sliding window) o długości **1024 tokenów**. Gdy rozmowa w jedynym slocie KV Ollamy przekroczy 1024 tokeny (prompt + wygenerowane), **każda** rozbieżność nowego promptu z zawartością cache, nawet o 1 token, wymusza **przeliczenie całego promptu od zera**. Nieskeszowany prefill na RPi 5 to tylko **~7,2 tok/s**, więc 1000-1200 tokenów to 140-170 s. To więcej niż timeout frontendu (90 s) i backendu (120 s). Żądanie jest przerywane (`500 | 2m0s` w Ollamie), praca przepada, a kolejna tura zaczyna od nowa. Sesja nie wychodzi z tego stanu.

**Wyzwalacze rozbieżności** (każdy z nich odtworzony w teście):
1. Okno historii 6 wiadomości zaczyna się przesuwać w 5. turze.
2. `.strip()` odpowiedzi w historii (gdy model zakończył `\n\n`).
3. Ping keep-warm `system + "ping"` co 10 min.
4. Ping rozgrzewający z faktami kończy się innym tekstem niż czat.
5. Nieudane tury zostawiają w historii samo pytanie.
6. Rotacja sesji.

**Dlaczego „pierwsze 3-4 działają”:** dopóki slot ma ≤ 1024 tokenów, cache cofa się poprawnie i tury kosztują 2-6 s prefillu. Próg przekracza się po kilku turach albo **od razu**, gdy fakty są duże (mediana faktów w zbiorze danych ≈ 1236 znaków, czyli ~1100 tokenów w slocie już przy 1. pytaniu).

## Jak testowano

- Prawdziwy stack (Context Service, Whisper Backend, Teacher na Macu, Uczeń `gemma3:4b` na RPi) uruchomiony z [stack.sh](stack.sh). Backend przekierowany zmienną środowiskową na **logujące proxy** [ollama_proxy.py](ollama_proxy.py), które przy każdym żądaniu do RPi zapisuje m.in. `prompt_eval_duration` (= faktyczny prefill), `eval_count` i kształt promptu.
- [driver.py](driver.py) symuluje frontend 1:1: historia 6 wiadomości, timeout 90 s, polling `/api/health` (2/30 s), rotacja po 600 s bezczynności.
- Pytania pochodzą z **prawdziwych rozmów** z 26.07 i 30.07 (transkrypty w Cursorze), [questions.json](questions.json).
- Telemetria: ping Wi-Fi 1 Hz, temperatura i throttling RPi, `/api/ps`, journal Ollamy ([monitor.sh](monitor.sh)).
- Pomiary bezpośrednie na RPi: [calibrate.py](calibrate.py) i eksperyment progu okna [swa_test.py](swa_test.py).
- Runy: `runs/20261020T1659Z` (kalibracja, SWA, S0, S1 z artefaktem proxy, patrz uwagi) oraz `runs/20261020T1749Z` (S1, S2, S4 z poprawionym proxy). W każdym są `analysis.md` (tabele per tura i per żądanie) oraz `timeline.png`.

## Problemy

### P1. Przekroczenie okna 1024 tokenów, pełne przeliczenie promptu (przyczyna główna)

**Warunek:** łączna długość rozmowy w slocie > 1024 tokenów **i** nowy prompt różni się od cache gdziekolwiek przed jego końcem.

**Dowód, eksperyment kontrolowany** (`runs/20261020T1659Z/swa_test.jsonl`). Rozmowa o zadanej długości, a potem ping ze wspólnym prefiksem 596 tokenów:

| Długość rozmowy w slocie | Prefill pingu (596 tok) |
|---|---|
| 901, 934, 975, 1013, 1024 tok | 0,25-0,88 s (cache użyty) |
| 1042, 1052, 1137 tok | **70,7-70,9 s (pełne przeliczenie)** |

Próg leży w przedziale (1024, 1042], zgodnie z oknem przesuwnym gemma3. Czyste dopisanie bez rozbieżności działa także powyżej progu: run 2, tura S1-4, slot 1077 tok, prefill 2,1 s.

**Koszt:**
- Prefill nieskeszowany ~7,2 tok/s (kalibracja: 295 tok w 40,5 s; 444 tok w 62,5 s), generacja ~4 tok/s.
- Pełne przeliczenie 1100-1300 tokenów trwa 150-180 s, czyli dłużej niż timeouty 90/120 s.

**Dowód w realnym przebiegu** (run 2, S1, replay rozmowy z 30.07):

| Tura | Slot przed turą | Prefill | E2E |
|---|---|---|---|
| 1-4 | 830 / 931 / 977 / 1077 (dopisanie) | 12,7 / 3,7 / 5,6 / 2,1 s | 21 / 10 / 24 / 10 s |
| 5 (okno historii się przesuwa) | 1116 | przerwane po 120 s | **timeout 90 s** |
| 6, 7 | | przerwane po 120 s | **timeout 90 s** |

To ten sam kształt co w logach z 30.07 (10,6 / 7,9 / 9,9 s, a potem 34 / 31 / 63 s i `2m0s`).

### P2. Wyzwalacze rozbieżności promptu (każdy z osobna wystarcza powyżej progu)

| Wyzwalacz | Gdzie w kodzie | Dowód |
|---|---|---|
| Okno historii 6 wiadomości przesuwa się od 5. tury | `CHAT_HISTORY_MAX_MESSAGES=6`, `chatService.ts` + `_normalize_history` | run 2 S1-5: pierwsza tura z przesunięciem, od razu timeout |
| `.strip()` odpowiedzi w historii | `_normalize_history` (`item.content.strip()`), `buildChatHistory` (`trim()`) | run 1 S1-4: odpowiedź #12 kończyła się `\n\n`, następna tura przeliczała się 173 s; tury z odpowiedziami bez białych znaków na końcu przechodziły w 2-6 s |
| Ping keep-warm `system + "ping"` co 600 s nadpisuje jedyny slot | `student_warmup.py::_keep_warm_once(include_facts=False)` | kalibracja: to samo pytanie po pingu ma prefill 40,5 s zamiast 0,37 s; run 2 S2a-after-ping: pierwszy token po 77 s |
| Ping rozgrzewający fakty kończy się „Potwierdź jednym słowem…”, a czat „Powyższy kontekst domu…” | `student_warmup.py` vs `main.py::_build_chat_messages` | run 2 sesja S2: rozgrzewanie (#12, 1029 tok) przekroczyło próg, pierwsze pytanie sesji (#13) przerwane po 120 s |
| Nieudana tura zostawia w historii samo pytanie, a każda kolejna tura zmienia środek promptu | `App.tsx` (wiadomość usera dodawana przed odpowiedzią) | run 2 S2a-1..3: trzy timeouty z rzędu |

### P3. Kontekst domu jest za duży względem okna, więc próg pada już przy 1. pytaniu

- Prefiks statyczny = **596 tokenów** (1886 znaków). Dynamiczny kontekst w całym CSV (n=1008 próbek): min 173, **mediana 1236**, p90 1766, max 2739 znaków.
- Przy medianie slot ma ~1100 tokenów jeszcze przed historią. Findings od Teachera dokładają kolejne ~500 znaków (sesja S2: 577 → 1136 znaków).
- Pierwsze tury S1 były szybkie tylko dlatego, że pierwszy wiersz CSV ma wyjątkowo małe fakty (577 znaków). W S4 fakty miały 457 znaków i wszystko chodziło płynnie (6-18 s), dopóki jedna długa odpowiedź nie przepchnęła slotu za próg.

### P4. Timeouty 90 s / 120 s i ich skutki

- Frontend przerywa po **90 s**, backend czeka na nagłówki Ollamy do **120 s** (`OLLAMA_READ_TIMEOUT_SECONDS`). Ollama wysyła nagłówki dopiero po prefillu, więc przez cały prefill backend „wisi” w `client.send()`.
- Po 120 s backend zrywa połączenie, a Ollama przerywa z **`500 | 2m0s`**. To dokładnie wpis z journala z 31.07 07:35, po którym RPi zostało twardo zrestartowane. Cały policzony prefill przepada.
- Między 90 a 120 s użytkownik ma już błąd, ale lock backendu i RPi są nadal zajęte. Pytanie zadane w tym oknie czeka w kolejce.
- **Kolejka:** w runie 1 proxy nie przekazywało zerwania połączenia (artefakt, poprawione), więc osierocone żądania trwały 173 → 226 → 247 → 291 → 340 → 383 s, a `in_flight` dochodził do 2. Pokazuje to, co dzieje się, gdy klient nie przerwie żądania: RPi potrzebowało ~5 min, żeby się odkorkować. Taki sam efekt daje każdy klient bez timeoutu (np. `curl`, testy ręczne).

### P5. Keep-warm i status `warming`/`error` blokują użytkownika

- Każdy ping interwałowy ustawia status `warming`, a czat w tym czasie zwraca **503** („Model się rozgrzewa”). Run 2 S2b: pytanie w trakcie pingu dostało 503 po 13 ms. Frontend przy statusie `warm` odpytuje co 30 s, więc UI pokazuje gotowość, a użytkownik dostaje błąd.
- Powyżej progu sam ping trwa **~71 s** (pełne przeliczenie 596 tok) i tyle trwa `warming`. W S2b+1 użytkownik czekał 52 s na odblokowanie plus 90 s timeoutu, razem 142 s.
- Ping, który stoi w kolejce i potem przelicza wszystko, trwał 121,5 s, czyli dłużej niż 120 s. Keep-warm skończył się błędem z **pustym komunikatem** i statusem `error` (run 1, 17:47:09), choć RPi działało.
- Rozgrzewanie nowej sesji (`session/new`) trwało 61-131 s (statyczny ping 70 s po slocie >1024 plus ping faktów 24-60 s). Przez ten czas UI jest zablokowane, a efekt i tak przepada przy pierwszym pytaniu (P2).

### P6. Brak parametrów generacji z pracy w payloadzie

- `main.py::_build_ollama_payload` nie wysyła `options`, więc nie ma `num_predict 128`, `temperature 0.2`, `num_ctx 2048` (journal: `KvSize:4096`), `top_p`, `repeat_penalty`.
- Bez `num_predict`: S4c (prośba o szczegółowy opis) generowała do timeoutu 90 s, a urwana długa odpowiedź weszła do historii i przepchnęła slot za próg. Następne 2 tury to timeouty.
- Bez `temperature`: odpowiedzi są losowe między runami (run 1 S1-4 timeout vs run 2 S1-4 OK przy tej samej rozmowie), więc to, czy odpowiedź kończy się `\n\n`, też jest losowe.

### P7. Brak obserwowalności

- Backend nie konfiguruje `logging`. Przy uruchomieniu przez `start.sh` linie `Chat done ...` i `Student keep-warm ...` (poziom INFO) **nigdy się nie wypisują**. Dlatego nie było historycznych logów. W teście włączyłem je przez `--log-config` bez zmian w kodzie.
- Tury zakończone timeoutem nie zostawiają w backendzie żadnego śladu: brak `Chat done`, brak linii access logu. Jedynym świadectwem jest `500 | 2m0s` w journalu RPi.

## Hipotezy obalone lub niepotwierdzone

| Hipoteza | Wynik |
|---|---|
| H5 termika | max 75,2°C, zero throttlingu (`0x0` w 180 próbkach), pod obciążeniem zegar 2,4 GHz (1,5-1,7 GHz tylko w bezczynności, przy 45-59°C). Obalona. |
| H7 Wi-Fi | 1,4% utraconych pingów, RTT p95 87 ms, zero błędów połączenia w proxy. W tych testach nie jest przyczyną. Historycznie (03.08) były dłuższe zaniki sieci obsłużone przez `wifi-recover`. |
| H8 obcinanie kontekstu 4096 | max prompt w testach 1234 tok, 0 wpisów o obcinaniu. Obalona. |
| H6 wyciek locka po zerwaniu strumienia | przerwanie w trakcie strumienia (S4a) zwalnia lock i slot; następna tura 7,3 s. Równoległe pytania (S4b) są poprawnie szeregowane (6,5 s i 18 s). Obalona. |

## Zgodność z historycznymi logami (journal RPi, 30-31.07, 23.08)

- Ping faktów zawsze ~32,6 s, w teście 32,8 s: prefill ~230 tok przy ~7 tok/s.
- Seria 10,6 / 7,9 / 9,9 s, potem 34 / 31 / 63 s (30.07 21:01): ten sam kształt co S1.
- 47 s i 66 s zaraz po pingach o 21:20:28 i 21:30:30: P2, ping keep-warm.
- `2m0s` / `500 | 2m0s`: P4.

## Otwarte pytania (poza zakresem tej serii)

- Prefill ~7 tok/s przy generacji ~4 tok/s to nietypowo mały stosunek (zwykle prefill jest wielokrotnie szybszy). Warto osobno sprawdzić, skąd się bierze: flash attention na CPU, rozmiar batcha, wątki.
- Twarde restarty RPi 30-31.07 (boot kończy się bez logów zamknięcia) nie wyjaśniają się tymi testami. Najprawdopodobniej to ręczne odłączenie zasilania przy „zawieszeniu” opisanym w P1/P4, ale to wymaga potwierdzenia.

## Uwagi metodologiczne

- `prompt_eval_count` w Ollamie 0.24 raportuje cały prompt także przy trafieniu w cache. Wskaźnikiem chybienia jest `prompt_eval_duration` (kolumna „prefill”).
- Run 1, S1 od tury 4: czasy osieroconych żądań (173-383 s) są zawyżone przez proxy, które nie przekazywało zerwania połączenia przed nagłówkami. Poprawione i zweryfikowane (Ollama `500` po 5,0 s przy zerwaniu klienta), run 2 jest wierny produkcji.
- Driver nie odtwarza TTS ani STT przeglądarki (jedna tura STT przez `/api/transcribe` w S0 przeszła poprawnie). Scenariusze S3 (rotacja po ciszy) i S5 (soak 30 min) pominięto, bo mechanizmy były już jednoznaczne: rotację i rozgrzewanie zaobserwowano w S0, S2 i S4.
