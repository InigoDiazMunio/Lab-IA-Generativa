"""
Retriever mejorado con traducción de query al inglés.

Problema: los papers están en inglés pero las preguntas se formulan en español.
El modelo de embeddings multilingüe reduce este gap pero no lo elimina del todo.
Traducir la query al inglés antes de buscar mejora el solapamiento semántico
con los chunks y por tanto la calidad de lo recuperado.

Implementación: usamos Helsinki-NLP/opus-mt-es-en, un modelo de traducción
ligero (~300MB) que ya viene disponible con transformers (ya instalado).
Si el modelo no está disponible o falla, cae back a la query original.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def _get_translator():
    """
    Carga el modelo de traducción es->en una sola vez y lo cachea.
    lru_cache evita recargar el modelo en cada llamada.
    """
    try:
        from transformers import pipeline
        translator = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-es-en",
            max_length=512,
        )
        print("[INFO] Modelo de traducción es->en cargado correctamente.")
        return translator
    except Exception as e:
        print(f"[AVISO] No se pudo cargar el modelo de traducción: {e}")
        return None


def translate_query(query: str) -> str:
    """
    Traduce la query del español al inglés para mejorar la recuperación
    en índices construidos con texto en inglés.

    Si la traducción falla por cualquier motivo, devuelve la query original
    sin romper el flujo del sistema.
    """
    translator = _get_translator()

    if translator is None:
        return query

    try:
        result = translator(query, max_length=512)
        translated = result[0]["translation_text"].strip()
        return translated
    except Exception as e:
        print(f"[AVISO] Error al traducir query: {e}. Usando original.")
        return query


def retrieve_context(vector_store, query: str, k: int = 4, translate: bool = False):
    """
    Recupera los k chunks mas relevantes del vector store.

    Args:
        vector_store:  Indice FAISS
        query:         Pregunta del usuario (puede estar en español)
        k:             Numero de chunks a recuperar
        translate:     Si True, traduce la query al inglés antes de buscar

    Returns:
        Lista de LangChain Documents
    """
    search_query = translate_query(query) if translate else query
    return vector_store.similarity_search(search_query, k=k)