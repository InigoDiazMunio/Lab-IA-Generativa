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


def add_documents(chunks: List[Document], persist_path: str) -> dict:
    """
    Añade documentos a una base vectorial existente de forma incremental.
    Si la BD no existe, la crea automáticamente.
    Detecta duplicados comparando source_file + page + inicio del contenido.

    Args:
        chunks: nuevos chunks a añadir.
        persist_path: ruta de la BD ChromaDB existente.

    Returns:
        dict con estadísticas: total_recibidos, duplicados, añadidos.
    """
    if not chunks:
        return {"total_recibidos": 0, "duplicados": 0, "añadidos": 0}

    os.makedirs(persist_path, exist_ok=True)
    embeddings = get_embedder()

    p = Path(persist_path)
    db_exists = p.exists() and (p / "chroma.sqlite3").exists()

    if db_exists:
        vector_store = Chroma(
            persist_directory=persist_path,
            embedding_function=embeddings,
            collection_name=DEFAULT_COLLECTION_NAME,
        )
    else:
        # Primera vez: crear desde cero con los chunks
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_path,
            collection_name=DEFAULT_COLLECTION_NAME,
        )
        return {
            "total_recibidos": len(chunks),
            "duplicados": 0,
            "añadidos": len(chunks),
        }

    # --- Deduplicación contra lo que ya está indexado ---
    existing_sources = list_indexed_sources(persist_path)
    existing_keys = set()

    for src in existing_sources:
        existing_keys.add((src["source_file"], src.get("page", "")))

    new_chunks = []
    duplicados = 0

    for chunk in chunks:
        key = (
            chunk.metadata.get("source_file", chunk.metadata.get("source", "")),
            str(chunk.metadata.get("page", "")),
        )
        if key in existing_keys:
            duplicados += 1
        else:
            new_chunks.append(chunk)

    if new_chunks:
        vector_store.add_documents(new_chunks)

    return {
        "total_recibidos": len(chunks),
        "duplicados": duplicados,
        "añadidos": len(new_chunks),
    }


def list_indexed_sources(persist_path: str) -> List[dict]:
    """
    Devuelve la lista de fuentes (source_file + páginas) ya indexadas en la BD.
    Útil para mostrar qué documentos están en el índice y detectar duplicados.
    """
    p = Path(persist_path)
    if not p.exists() or not (p / "chroma.sqlite3").exists():
        return []

    embeddings = get_embedder()

    vector_store = Chroma(
        persist_directory=persist_path,
        embedding_function=embeddings,
        collection_name=DEFAULT_COLLECTION_NAME,
    )

    collection = vector_store._collection
    result = collection.get(include=["metadatas"])

    sources = {}
    for meta in result.get("metadatas", []):
        source_file = meta.get("source_file", meta.get("source", "desconocido"))
        page = meta.get("page", "?")
        doc_type = meta.get("type", "texto")

        if source_file not in sources:
            sources[source_file] = {
                "source_file": source_file,
                "pages": set(),
                "types": set(),
                "chunk_count": 0,
            }

        sources[source_file]["pages"].add(str(page))
        sources[source_file]["types"].add(doc_type)
        sources[source_file]["chunk_count"] += 1

    result_list = []
    for info in sources.values():
        result_list.append({
            "source_file": info["source_file"],
            "pages": sorted(info["pages"], key=lambda x: int(x) if x.isdigit() else 0),
            "types": sorted(info["types"]),
            "chunk_count": info["chunk_count"],
        })

    return sorted(result_list, key=lambda x: x["source_file"])


def delete_source(source_file: str, persist_path: str) -> int:
    """
    Elimina todos los chunks de un source_file concreto de la BD.

    Args:
        source_file: nombre del archivo a eliminar (ej: 'paper 1.pdf').
        persist_path: ruta de la BD ChromaDB.

    Returns:
        Número de chunks eliminados.
    """
    p = Path(persist_path)
    if not p.exists() or not (p / "chroma.sqlite3").exists():
        return 0

    embeddings = get_embedder()

    vector_store = Chroma(
        persist_directory=persist_path,
        embedding_function=embeddings,
        collection_name=DEFAULT_COLLECTION_NAME,
    )

    collection = vector_store._collection
    result = collection.get(
        where={"source_file": source_file},
        include=[],
    )

    ids_to_delete = result.get("ids", [])

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)

    return len(ids_to_delete)