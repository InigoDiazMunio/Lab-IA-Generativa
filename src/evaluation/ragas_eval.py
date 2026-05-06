"""
Evaluación del sistema RAG con métricas estilo RAGAS.

Implementación propia que llama a Ollama de forma síncrona para evitar
los problemas de timeout/async del framework RAGAS con modelos locales.

Métricas implementadas:
- Faithfulness:       ¿La respuesta se basa fielmente en el contexto recuperado?
- Answer Relevancy:   ¿La respuesta es relevante para la pregunta?
- Context Precision:  ¿Los fragmentos recuperados son relevantes para la pregunta?
- Context Recall:     ¿Se recuperó todo el contexto necesario?
"""

import json
import math
import re
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"


def _call_ollama(prompt: str, retries: int = 2) -> str:
    """Llama a Ollama de forma síncrona. Reintenta si falla."""
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "num_predict": 256,
                    },
                },
                timeout=180,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            if attempt == retries:
                print(f"    [WARN] Ollama falló tras {retries + 1} intentos: {e}")
                return ""
            print(f"    [RETRY] Intento {attempt + 1} falló: {e}")
    return ""


def _parse_score(response: str) -> float:
    """
    Extrae un score numérico (0.0-1.0) de la respuesta del LLM.
    Intenta parsear JSON, y si no, busca un número.
    """
    text = response.strip()

    # Intentar parsear JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("score", "verdict", "rating", "value"):
                if key in data:
                    val = data[key]
                    if isinstance(val, (int, float)):
                        return max(0.0, min(float(val), 1.0))
                    if isinstance(val, str):
                        try:
                            return max(0.0, min(float(val), 1.0))
                        except ValueError:
                            pass
    except json.JSONDecodeError:
        pass

    # Buscar patrón "score": 0.X en texto no-JSON
    match = re.search(r'"?score"?\s*[:=]\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        return max(0.0, min(float(match.group(1)), 1.0))

    # Buscar X/10 o X/1
    match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+)", text)
    if match:
        num, denom = float(match.group(1)), float(match.group(2))
        if denom > 0:
            return max(0.0, min(num / denom, 1.0))

    # Buscar un decimal entre 0 y 1
    numbers = re.findall(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
    if numbers:
        return float(numbers[0])

    return float("nan")


def _eval_faithfulness(question: str, answer: str, contexts: list) -> float:
    """¿La respuesta se basa fielmente en el contexto?"""
    context_text = "\n---\n".join(contexts[:3])

    prompt = f"""You are an impartial judge. Evaluate FAITHFULNESS: whether the answer is grounded in the provided context.

1.0 = every claim in the answer can be found in the context
0.0 = the answer invents information not in the context

Question: {question}

Context:
{context_text}

Answer: {answer}

Reply ONLY with: {{"score": <number between 0.0 and 1.0>}}"""

    return _parse_score(_call_ollama(prompt))


def _eval_answer_relevancy(question: str, answer: str) -> float:
    """¿La respuesta es relevante para la pregunta?"""
    prompt = f"""You are an impartial judge. Evaluate ANSWER RELEVANCY: how well the answer addresses the question.

1.0 = the answer directly and completely addresses the question
0.0 = the answer is completely irrelevant

Question: {question}

Answer: {answer}

Reply ONLY with: {{"score": <number between 0.0 and 1.0>}}"""

    return _parse_score(_call_ollama(prompt))


def _eval_context_precision(question: str, contexts: list, reference: str) -> float:
    """¿Los fragmentos recuperados son relevantes?"""
    context_text = "\n---\n".join(contexts[:3])

    prompt = f"""You are an impartial judge. Evaluate CONTEXT PRECISION: whether the retrieved fragments are relevant to the question.

1.0 = all fragments are highly relevant
0.0 = none of the fragments are relevant

Question: {question}
Expected answer: {reference}

Retrieved fragments:
{context_text}

Reply ONLY with: {{"score": <number between 0.0 and 1.0>}}"""

    return _parse_score(_call_ollama(prompt))


def _eval_context_recall(question: str, contexts: list, reference: str) -> float:
    """¿Se recuperó todo el contexto necesario?"""
    context_text = "\n---\n".join(contexts[:3])

    prompt = f"""You are an impartial judge. Evaluate CONTEXT RECALL: whether the retrieved context contains all information needed to produce the expected answer.

1.0 = the context contains all needed information
0.0 = the context is missing critical information

Question: {question}
Expected answer: {reference}

Retrieved context:
{context_text}

Reply ONLY with: {{"score": <number between 0.0 and 1.0>}}"""

    return _parse_score(_call_ollama(prompt))


def run_ragas_eval(
    questions: list,
    answers: list,
    contexts: list,
    references: list = None,
    llm=None,
    embeddings=None,
):
    """
    Ejecuta la evaluación estilo RAGAS de forma síncrona con Ollama.

    Los parámetros llm y embeddings se ignoran — se usa Ollama directamente
    para evitar problemas de timeout y async.

    Returns:
        dict con 'columns', 'rows' y 'averages'.
    """

    n = len(questions)

    # Preparar references
    if references is None or len(references) != n:
        references = [""] * n

    filled_refs = [
        ref if ref and ref.strip() else ans
        for ref, ans in zip(references, answers)
    ]

    rows = []
    totals = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
    }

    for i in range(n):
        q = questions[i]
        a = answers[i]
        ctx = contexts[i] if i < len(contexts) else [""]
        ref = filled_refs[i]

        print(f"  [{i+1}/{n}] {q[:60]}...")

        row = {"question": q, "answer": a}

        # Faithfulness
        score = _eval_faithfulness(q, a, ctx)
        row["faithfulness"] = score if not math.isnan(score) else None
        if not math.isnan(score):
            totals["faithfulness"].append(score)

        # Answer relevancy
        score = _eval_answer_relevancy(q, a)
        row["answer_relevancy"] = score if not math.isnan(score) else None
        if not math.isnan(score):
            totals["answer_relevancy"].append(score)

        # Context precision
        score = _eval_context_precision(q, ctx, ref)
        row["context_precision"] = score if not math.isnan(score) else None
        if not math.isnan(score):
            totals["context_precision"].append(score)

        # Context recall
        score = _eval_context_recall(q, ctx, ref)
        row["context_recall"] = score if not math.isnan(score) else None
        if not math.isnan(score):
            totals["context_recall"].append(score)

        # Mostrar scores de esta pregunta
        scores_str = " | ".join(
            f"{m}: {row.get(m, 'n/a')}"
            for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            if row.get(m) is not None
        )
        print(f"    → {scores_str}")

        rows.append(row)

    # Promedios
    averages = {}
    for metric, values in totals.items():
        if values:
            averages[metric] = round(sum(values) / len(values), 4)

    if averages:
        print(f"\n  Promedios finales:")
        for m, v in averages.items():
            print(f"    {m}: {v:.4f}")

    return {
        "columns": ["question", "answer", "faithfulness", "answer_relevancy",
                     "context_precision", "context_recall"],
        "rows": rows,
        "averages": averages,
    }


def format_ragas_result(ragas_result) -> dict:
    """Formatea el resultado. Si ya es dict, lo devuelve directamente."""
    if isinstance(ragas_result, dict):
        return ragas_result
    return {"result": str(ragas_result)}