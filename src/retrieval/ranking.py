"""
Reranking mejorado con cross-encoder multilingüe.

Cambio respecto a la versión anterior:
- Modelo cambiado de ms-marco-MiniLM-L-6-v2 (solo inglés)
  a mmarco-mMiniLMv2-L12-H384-v1 (multilingüe)
- Permite puntuar correctamente pares query-español / contexto-inglés
  y query-español / caption-español sin penalizar por idioma
"""

from typing import List

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
            print("[INFO] Cross-encoder multilingüe cargado correctamente.")
        except Exception as e:
            print(f"[AVISO] No se pudo cargar el cross-encoder multilingüe: {e}")
    return _reranker


def rank_retrieved_docs(retrieved_docs, query: str, top_k: int = 4) -> List:
    """
    Reordena los documentos usando un cross-encoder multilingüe con diversidad de fuente.

    Diversidad: ningún paper puede ocupar más de max_per_source slots en el top-k.
    Esto evita que un paper muy largo (surveys, RAG surveys) monopolice los resultados
    y desplace fragmentos relevantes de papers más especializados.
    """
    if not retrieved_docs:
        return []

    try:
        reranker = _get_reranker()

        if reranker is None:
            return retrieved_docs[:top_k]

        pairs = [(query, doc.page_content) for doc in retrieved_docs]
        scores = reranker.predict(pairs)

        scored_docs = sorted(
            zip(scores, retrieved_docs),
            key=lambda x: x[0],
            reverse=True
        )

        # Diversidad de fuente: máximo top_k//2 fragmentos del mismo paper
        max_per_source = max(2, top_k // 2)
        result: List = []
        overflow: List = []
        source_count: dict = {}

        for score, doc in scored_docs:
            src = doc.metadata.get("source_file", doc.metadata.get("source", ""))
            count = source_count.get(src, 0)
            if count < max_per_source:
                result.append(doc)
                source_count[src] = count + 1
            else:
                overflow.append(doc)
            if len(result) == top_k:
                break

        # Si no se llenaron top_k slots con diversidad, completar con los overflow
        for doc in overflow:
            if len(result) >= top_k:
                break
            result.append(doc)

        return result

    except Exception as e:
        print(f"[AVISO] Error en el reranking: {e}. Usando orden original.")
        return retrieved_docs[:top_k]


def rank_retrieved_docs_simple(retrieved_docs, top_k: int = 4) -> List:
    return retrieved_docs[:top_k]