def build_prompt(query: str, retrieved_docs):
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

Pregunta:
{query}

Contexto:
{context}

Respuesta:
"""
    return prompt