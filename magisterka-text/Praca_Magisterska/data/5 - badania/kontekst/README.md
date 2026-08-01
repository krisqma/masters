# Wilga Benchmark — po co to jest i co się zmienia

To jest benchmark pod Twoją **Wilgę**: te same pytania i „złote” odpowiedzi (`golden.jsonl`), modele z **Ollamy na malince**, a Optuna szuka sensownych ustawień (`temperature`, kontekst, długość odpowiedzi itd.). Każda próba trafia do pliku JSONL — potem z tego robisz wykresy i tekst w pracy.

Poniżej jest opis **jak to działało / działa w myśli**, **co chcemy poprawić** i **dlaczego** — normalnym językiem. Na końcu jest krótka wersja „do wklejenia” w rozdział metodyki, jeśli zechcesz brzmieć bardziej naukowo.

---

## Co ogólnie liczymy (to się raczej nie zmienia)

Dla każdego pytania dostajesz kilka liczb od ewaluatora (czy odpowiedź trzyma się faktów, czy odpowiada na pytanie, czy nie jest za długa) plus **czas** od wysłania requestu do pełnej odpowiedzi. Z tego składamy **jedną liczbę** — **composite score** — żeby Optuna miała prostą zasadę: **większa = lepiej**.

Jeśli model **halucynuje** (wg Twojej logiki w `evaluate`), cała ta próba dostaje **zero** punktów. Idea jest prosta: w domowym asystencie wolę wolniejszą odpowiedź niż taką, która wymyśla stany czujników. Optuna nie powinna „oszukiwać” i szukać setupu, który jest szybki, ale niebezpieczny merytorycznie.

---

## 1. Opóźnienie w wyniku — jak było, co jest nie tak, co robimy

**Jak było:**  
Czas odpowiedzi (`e2e_ms`) wrzucamy do wzoru tak, żeby **krótsze** odpowiedzi **dokładały** trochę punktów do composite. Jest też **timeout** HTTP (np. 60 s): jak Ollama nie zdąży, ucinamy, traktujemy jak porażkę i **wpisujemy czas równy temu limitowi**.

**Na czym polega problem:**  
Gdy **górny skal** do normalizacji pełnego czasu (`E2E_NORM_CAP_MS` w kodzie; dawniej wizualnie zbliżone do `LATENCY_CAP_MS`) ustawisz **na tę samą wartość** co timeout w milisekundach, to **każdy** timeout wypada dokładnie na „maksymalnie wolno” w tej skali. W praktyce kawałek wzoru odpowiedzialny za „nagrodę za szybkość” **nie rozróżnia** różnych timeoutów — wszystkie lądują w tym samym roku. To **nie jest** to samo co „latency = 0 wszędzie” w sensie buga w kodzie, ale **psuje sygnał dla Optuny**: TPE widzi dużo punktów przyklejonych do jednej ściany, zamiast sensownego gradientu.

**Dlaczego to ważne:**  
Timeout to **sztuczny sufit** („przestaliśmy czekać”), a nie „tak długo model realnie liczył”. Mieszanie tych dwóch pojęć w **jednej** liczbie bez oddzielenia skali powoduje, że **optymalizator głupieje** albo wręcz ignoruje subtelności czasowe tam, gdzie dużo przejść kończy się na limicie.

**Co zmieniamy:**  
Trzymać **timeout** jako twardy limit po stronie klienta, ale **cap do normalizacji** w funkcji celu ustawić **inaczej** — zwykle **wyżej** niż `timeout × 1000 ms`, albo w ogóle myśleć o tym jako o **osobnej pokrętle**: „jak karzemy wolność w punkacji”, a nie „to samo co po ile czekamy”. W pracy możesz to opisać jednym zdaniem: *rozdzieliliśmy limit żądania od skali kary za opóźnienie w funkcji celu*.

---

## 2. TTFT (czas do pierwszego tokenu) — jak było, co zmieniamy, dlaczego

**Jak było:**  
Ollama wołana była w trybie **bez streamingu** — dostajesz całą odpowiedź **dopiero na końcu**. W logach pole `ttft_ms` było **`null`**, bo **fizycznie nie było co zmierzyć**.

**Problem:**  
Dla asystenta głosowego **bardzo** liczy się, kiedy user **pierwszy raz** „coś usłyszy” / zobaczy pierwszy fragment — to inna psychofizyka niż „ile trwa cały monolog”. Sam `e2e_ms` **zlewa** „długo się kręcił zanim cokolwiek powiedział” z „potem długo gadał”.

**Co zmieniamy:**  
Włączamy **streaming** (`stream: true`), czytamy zdarzenia z API i mierzymy:

- **`ttft_ms`** — od wysłania pytania do **pierwszego sensownego kawałka** odpowiedzi modelu,
- **`e2e_ms`** — nadal **do końca** całej odpowiedzi.

**Dlaczego:**  
Masz **bogatsze dane** do pracy (wykresy, porównania, zdania w dyskusji). Możesz też — jeśli zechcesz — **wpleść TTFT do composite** (np. druga mała waga obok pełnego czasu), żeby Optuna **jawnie** preferowała konfiguracje „szybko zaczyna odpowiadać”, nie tylko „szybko się kończy całość”. Nawet bez zmiany wag **sam fakt, że TTFT jest w JSONL**, zmienia jakość opisu eksperymentu.

---

## 3. Baza Optuny — jak było, czemu „wyrzucić stare triale”, dlaczego

**Jak działa:**  
Każdy model ma swoją **study** w SQLite (`optuna.db`). Optuna zbiera próby i **uczy się** na całej historii, którą już zebrała (TPE szuka „gdzie są dobre punkty” na podstawie **wcześniejszych** wyników).

**Co poszło nie tak w kontekście projektu:**  
Jeśli **zmienisz wzór** na composite **albo** sposób mierzenia czasu (cap, streaming, TTFT), to **stare triale mają przyklejone stare „punkty”**, które **nie odpowiadają nowej zasadzie gry**. Dla Optuny to jak mieszanie ocen z dwóch różnych kluczy odpowiedzi — nadal próbuje się uczyć, ale na **śmieciowych etykietach**.

**Co robimy:**  
Dla modeli, które były optymalizowane pod **starą** metrykę (u Ciebie mowa była m.in. o **gemma3:4b** i **llama3.2:3b**), **kasujemy** odpowiednie studia z bazy **zanim** odpalisz nową rundę — żeby TPE startowało na **czysto**. Całego pliku `optuna.db` nie musisz usuwać, jeśli inne modele mają sensowną historię pod **tę samą** metrykę co dalej używasz.

**Dlaczego po polsku:**  
„Bo inaczej Optuna pamięta stare kłamstwa o tym, które ustawienia są dobre.”

---

## 4. Różna liczba triali na model — po co

**Jak mogło wyglądać wcześniej:**  
Jeden przełącznik „zrób **N** triali dla każdego modelu” — wygodne, ale nierówne pod kątem **czasu na malince** i **tego, ile zyskujesz**.

**Co zmieniamy w strategii:**  
Np. dla modelu, który i tak jest najlepszy — **więcej** triali (np. **30**), żeby **dobrze** sprawdzić plateau. Dla modelu wyraźnie słabszego / referencyjnego — **mniej** triali (np. **10**), żeby mieć baseline bez przepalania dni na RPi.

**Dlaczego:**  
To nie jest „ściemnianie wyników”, tylko **świadomy design**: inny budżet tam, gdzie się coś wygrać, i mniejszy tam, gdzie i tak widać, że daleko do czoła. W pracy warto napisać wprost: *różne budżety HPO wg oczekiwanego zysku informacyjnego*.

---

## 5. Jeden dodatkowy model „po polsku”

**Sytuacja:**  
Porównujesz głównie modele „ogólne”, wielojęzyczne. To OK na tło rynku, ale **Wilga gada po polsku** — czytelnik zapyta: *a co z modelem bardziej pod PL?*

**Co robimy:**  
Na RPi sprawdzasz `ollama list`, wybierasz **jeden** sensowny, **lekki** model z dobrym PL (albo podajesz go ręcznie przez zmienną / flagę), i dokładasz go do porównania.

**Dlaczego:**  
Żeby porównanie było **uczciwsze wobec tematu pracy** (asystent po polsku), a nie tylko „losowe popularne checkpointy”.

---

## Pliki, które warto znać

| Co | Po co |
|----|--------|
| `golden.jsonl` | Stały zestaw pytań i odpowiedzi referencyjnych |
| `results/*.jsonl` | Każda próba — parametry + metryki + composite |
| `optuna.db` | Historia Optuny (dashboard, reproducibility) |
| `RPI_OLLAMA_URL`, `RPI_METRICS_URL` | Gdzie stoi Ollama i skąd ściągasz temperaturę RAM itd. |

W kodzie: **`REQUEST_TIMEOUT`** = twardy limit HTTP (sekundy); **`E2E_NORM_CAP_MS`** oraz **`TTFT_NORM_CAP_MS`** = jak skalowane są czasy w funkcji celu (`run_benchmark.py`).

---

## Komendy — przykładowy run (bez odpalania za Ciebie)

Z katalogu `benchmark`, **po** ustawieniu `export RPI_OLLAMA_URL=...` i `export RPI_METRICS_URL=...` (na RPi lub tunelu):

```bash
# Jednorazowo: usuń stare studia Optuny dla wybranych modeli (zła metryka w historii).
python run_benchmark.py --purge-models gemma3:4b llama3.2:3b

# Opcjonalnie dopnij jeden polski checkpoint z Ollamy (bez edycji listy MODELS w kodzie):
export POLISH_OLLAMA_MODEL="twoj-tag-z-ollama-list"

RUN_ID="$(date -u +%Y-%m-%d_benchmark_%H%M)"

python run_benchmark.py --run-id "$RUN_ID" \
  --models "gemma3:4b,llama3.2:3b" \
  --model-trials gemma3:4b=30 \
  --model-trials llama3.2:3b=10
```

`EXTRA_OLLAMA_MODELS=bielik:7b,foo:tag` dopina wiele nazw przez przecinki. Jeśli **nie** podasz `--models`, lecą **`MODELS` z kodu + ENV**.

---


## Jedno zdanie „na rozdział w pracy” (sztywniej)

Optymalizacja hiperparametrów prowadzona jest przez **Optunę (TPE)** przy **jednej funkcji skalarnej** łączącej metryki jakości z penalizacją opóźnienia; **timeout HTTP** został logicznie oddzielony od **normalizacji czasu w celu**, a **TTFT** jest mierzony przez **API streamingu**, aby oddzielić opóźnienie pierwszej odpowiedzi od czasu pełnej generacji. Po zmianie definicji celu lub pomiaru czasu **czyszczona jest historia studiów** dla dotkniętych modeli, aby uniknąć wprowadzenia optymalizatora w błąd **nieaktualnymi** wynikami wcześniejszych triali.
