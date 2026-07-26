import json
import time
import logging
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
    return _client


_SYSTEM_PROMPT = (
    "Jesteś precyzyjnym ewaluatorem odpowiedzi asystenta smart-home.\n"
    "Oceniasz odpowiedź według podanych kryteriów i zwracasz WYŁĄCZNIE JSON.\n"
    "Zero komentarzy, zero markdown, tylko surowy obiekt JSON."
)

_USER_TEMPLATE = """\
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
- hallucination_flag=true: TYLKO gdy są konkretne fakty spoza kontekstu (nie ogólne zwroty)\
"""

_FALLBACK = {
    "faithfulness": 0.0,
    "answer_relevancy": 0.0,
    "conciseness": 0.0,
    "polish_quality": 0.0,
    "hallucination_flag": True,
    "haiku_raw": "",
}


def evaluate_response(
    system_prompt: str,
    question: str,
    golden_answer: str,
    model_answer: str,
    category: str,
) -> dict:
    client = _get_client()
    model = os.environ["HAIKU_MODEL"]
    user_prompt = _USER_TEMPLATE.format(
        system_prompt=system_prompt,
        question=question,
        golden_answer=golden_answer,
        model_answer=model_answer,
    )

    raw = ""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            raw = resp.choices[0].message.content or ""
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("```", 2)[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
            return {
                "faithfulness": float(parsed["faithfulness"]),
                "answer_relevancy": float(parsed["answer_relevancy"]),
                "conciseness": float(parsed["conciseness"]),
                "polish_quality": float(parsed["polish_quality"]),
                "hallucination_flag": bool(parsed["hallucination_flag"]),
                "haiku_raw": raw,
            }
        except Exception as exc:
            logger.warning("evaluate attempt %d failed: %s", attempt + 1, exc)
            if attempt < 2:
                time.sleep(2 ** attempt)

    fallback = dict(_FALLBACK)
    fallback["haiku_raw"] = raw
    return fallback
