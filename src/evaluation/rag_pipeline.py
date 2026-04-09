"""
Pipeline RAG con recuperación híbrida adaptativa y deduplicación.

Mejoras:
- Deduplicación de chunks antes del reranking
- Reranking multilingüe adaptativo
- Captions solo para queries visuales explícitas
"""

from src.retrieval.retriever import retrieve_context
from src.retrieval.ranking import rank_retrieved_docs
from src.generation.prompt_builder import build_rag_prompt, build_multimodal_rag_prompt
from src.generation.llm import generate_answer


def _deduplicate(docs):
    """Elimina chunks duplicados manteniendo el orden de relevancia."""
    seen = set()
    unique = []
    for doc in docs:
        key = (
            doc.metadata.get("source_file", ""),
            doc.metadata.get("page", ""),
            doc.page_content[:80]
        )
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def answer_with_rag(query: str, vector_store, k: int = 4) -> tuple:
    """Pipeline RAG con deduplicación y reranking multilingüe."""
    candidates = retrieve_context(vector_store, query, k=k * 3, translate=False)
    candidates = _deduplicate(candidates)
    retrieved_docs = rank_retrieved_docs(candidates, query=query, top_k=k)

    prompt = build_rag_prompt(query, retrieved_docs)
    answer = generate_answer(prompt).strip()

    sources = []
    for doc in retrieved_docs:
        sources.append({
            "source_file": doc.metadata.get("source_file", doc.metadata.get("source", "desconocido")),
            "page": doc.metadata.get("page", "N/A"),
            "content_preview": doc.page_content[:300],
            "type": doc.metadata.get("type", "texto")
        })

    return answer, sources


def answer_with_rag_multimodal(
    query: str,
    vector_store,
    image_captions: list = None,
    k: int = 4
) -> tuple:
    """
    Pipeline RAG multimodal con deduplicación y reranking adaptativo.
    Los captions compiten en igualdad con el texto.
    El reranker multilingüe decide cuáles son más relevantes.
    """
    candidates = retrieve_context(vector_store, query, k=k * 3, translate=False)
    candidates = _deduplicate(candidates)
    retrieved_docs = rank_retrieved_docs(candidates, query=query, top_k=k)

    # Captions externos solo para queries visuales explícitas
    relevant_captions = []
    visual_keywords = ["diagrama", "figura", "tabla", "imagen", "gráfico",
                       "esquema", "foto", "ilustración", "ejemplo visual"]
    if any(kw in query.lower() for kw in visual_keywords) and image_captions:
        relevant_captions = image_captions[:2]

    if relevant_captions:
        prompt = build_multimodal_rag_prompt(query, retrieved_docs, relevant_captions)
    else:
        prompt = build_rag_prompt(query, retrieved_docs)

    answer = generate_answer(prompt).strip()

    sources = []
    for doc in retrieved_docs:
        sources.append({
            "source_file": doc.metadata.get("source_file", doc.metadata.get("source", "desconocido")),
            "page": doc.metadata.get("page", "N/A"),
            "content_preview": doc.page_content[:300],
            "type": doc.metadata.get("type", "texto")
        })

    return answer, sources, retrieved_docs