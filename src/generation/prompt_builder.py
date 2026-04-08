def build_baseline_prompt(query: str) -> str:
    return f"""Pregunta: {query}
Respuesta breve en español:"""


def build_rag_prompt(query: str, retrieved_docs: list) -> str:
    context_parts = []

    for doc in retrieved_docs:
        content = doc.page_content.strip()
        context_parts.append(content)

    context = "\n\n".join(context_parts[:3])

    return f"""Contexto:
{context}

Pregunta: {query}
Respuesta breve en español usando solo el contexto:"""