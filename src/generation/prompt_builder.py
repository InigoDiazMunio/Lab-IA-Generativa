def build_baseline_prompt(query: str) -> str:
    """
    Prompt para el baseline: el modelo responde SIN ningún documento de contexto.
    """
    return f"""Eres un asistente universitario experto. Responde la siguiente pregunta
de forma clara y concisa en español. Si no estás seguro de la respuesta, indícalo
explícitamente en lugar de inventar información.

Pregunta: {query}

Respuesta:"""


def build_rag_prompt(query: str, retrieved_docs: list) -> str:
    """
    Prompt para el sistema RAG con contexto numerado.
    Los fragmentos pueden estar en inglés — se indica al modelo que los traduzca.
    """
    context_parts = []

    for i, doc in enumerate(retrieved_docs, start=1):
        source_file = doc.metadata.get("source_file", doc.metadata.get("source", "desconocido"))
        page = doc.metadata.get("page", "?")
        content = doc.page_content.strip().replace("[Contenido visual] ", "")
        context_parts.append(
            f"[Fragmento {i} — {source_file}, pág. {page}]\n{content}"
        )

    context = "\n\n".join(context_parts)

    return f"""Eres un asistente universitario. A continuación tienes fragmentos extraídos de documentos académicos.
Usa ÚNICAMENTE esa información para responder en español.
Los fragmentos pueden estar en inglés — tradúcelos y úsalos para responder.
Si la respuesta no se encuentra en los fragmentos, di que no dispones de esa información.

--- CONTEXTO ---
{context}
--- FIN DEL CONTEXTO ---

Pregunta: {query}

Responde en español de forma clara y concisa, citando el número de fragmento cuando uses información de él (ej: "Según el fragmento 2, ..."):"""


def build_multimodal_rag_prompt(query: str, retrieved_docs: list, image_captions: list) -> str:
    """
    Versión multimodal: combina contexto textual + descripciones de imágenes.
    """
    text_context = build_rag_prompt(query, retrieved_docs)

    if not image_captions:
        return text_context

    captions_text = "\n".join(
        f"- [Imagen pág. {c.get('page', '?')} de {c.get('pdf', 'desconocido')}]: {c['caption']}"
        for c in image_captions
    )

    return text_context + f"""

--- CONTENIDO VISUAL RELEVANTE ---
{captions_text}
--- FIN DEL CONTENIDO VISUAL ---

Recuerda también el contenido visual si es relevante para la respuesta:"""