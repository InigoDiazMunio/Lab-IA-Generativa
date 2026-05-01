"""
Capa de compatibilidad del vector store.

Antes el proyecto usaba FAISS desde este archivo. Ahora mantiene los mismos
nombres de funciones, pero por debajo usa ChromaDB persistente. Así no hay que
cambiar todo el pipeline RAG, la interfaz web ni los evaluadores.
"""

import os
import shutil
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.embeddings.embedder import get_embedder


DEFAULT_COLLECTION_NAME = "rag_universitario"


def _safe_remove_path(path: str) -> None:
    """Borra una ruta aunque sea carpeta o archivo corrupto/creado por error."""
    p = Path(path)
    if not p.exists():
        return

    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()


def build_vector_store(chunks: List[Document], persist_path: str, reset: bool = True):
    """
    Crea una base de datos vectorial persistente con ChromaDB.

    Args:
        chunks: documentos/chunks de LangChain.
        persist_path: carpeta donde se guardará ChromaDB.
        reset: si True, elimina el índice anterior antes de reconstruirlo.
    """
    if not chunks:
        raise ValueError("No hay chunks para indexar. Revisa data/raw y la carga de PDFs.")

    if reset:
        _safe_remove_path(persist_path)

    os.makedirs(persist_path, exist_ok=True)

    embeddings = get_embedder()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_path,
        collection_name=DEFAULT_COLLECTION_NAME,
    )

    return vector_store


def load_vector_store(persist_path: str):
    """
    Carga la base vectorial ChromaDB persistente.
    """
    p = Path(persist_path)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(
            f"No existe la base vectorial en '{persist_path}'. Ejecuta primero: python -m src.build_index"
        )

    embeddings = get_embedder()

    return Chroma(
        persist_directory=persist_path,
        embedding_function=embeddings,
        collection_name=DEFAULT_COLLECTION_NAME,
    )
