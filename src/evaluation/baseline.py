from generation.llm import generate_answer


def build_baseline_prompt(query: str) -> str:
    return f"""
Responde a la siguiente pregunta usando únicamente tu conocimiento general.
Si no estás seguro, indícalo de forma explícita.

Pregunta:
{query}

Respuesta:
"""


def answer_without_rag(query: str) -> str:
    prompt = build_baseline_prompt(query)
    return generate_answer(prompt)