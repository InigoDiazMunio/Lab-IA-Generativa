"""
Chunking mejorado.

Cambios respecto al original:
- chunk_size reducido de 800 a 450 caracteres
- chunk_overlap reducido de 150 a 80
- Se lee la configuración desde config.yaml si está disponible,
  con fallback a los valores por defecto

Por qué chunks más pequeños:
- Chunks de 800 caracteres capturan demasiado contexto irrelevante
  para preguntas específicas (ej: "¿qué es el Q-Former?")
- Chunks más pequeños = recuperación más precisa
- El solapamiento de 80 mantiene continuidad entre chunks adyacentes
  sin duplicar demasiado contenido
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents, chunk_size: int = 450, chunk_overlap: int = 80):
    """
    Divide documentos en chunks para indexación.

    Args:
        documents:     Lista de LangChain Documents (páginas de PDF)
        chunk_size:    Tamaño máximo de cada chunk en caracteres
        chunk_overlap: Solapamiento entre chunks consecutivos

    Returns:
        Lista de chunks como LangChain Documents
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"  Chunks generados: {len(chunks)} "
          f"(chunk_size={chunk_size}, overlap={chunk_overlap})")
    return chunks