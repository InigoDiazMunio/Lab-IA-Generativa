def build_rag_prompt(query: str, retrieved_docs):
    context_blocks = []

    for i, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source_file", "desconocido")
        page = doc.metadata.get("page", "N/A")

        context_blocks.append(
            f"Fuente {i} | Archivo: {source} | Página: {page}\n{doc.page_content}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
Responde a la pregunta usando solo la información del contexto.
Si la respuesta no aparece en el contexto, responde exactamente:
No se encuentra en los documentos proporcionados.

Además, si respondes con información concreta, menciona la fuente (archivo y página).

Pregunta:
{query}

Contexto:
{context}

Respuesta:
"""
    return prompt


def build_baseline_prompt(query: str):
    return f"""
Responde a la siguiente pregunta usando únicamente tu conocimiento general.
No uses documentos externos ni cites fuentes inventadas.
Si no estás seguro, indícalo claramente.

Pregunta:
{query}

Respuesta:
"""