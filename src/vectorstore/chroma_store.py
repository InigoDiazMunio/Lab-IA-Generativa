"""
Funciones específicas de ChromaDB.

Este archivo se mantiene por compatibilidad con src/build_index.py.
Internamente delega en src.embeddings.vector_store para que todo el proyecto
use una única implementación de base vectorial.
"""

import os

from langchain_chroma import Chroma
from src.embeddings.embedder import get_embedding_model


CHROMA_PATH = "data/embeddings/chroma_db"


def create_vectorstore(chunks, reset: bool = False):
    embedding_model = get_embedding_model()

    if os.path.exists(CHROMA_PATH):
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_model
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=CHROMA_PATH
        )

    return vectorstore


def load_vectorstore():
    embedding_model = get_embedding_model()

    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )


def get_retriever(k: int = 4):
    vectorstore = load_vectorstore()

    return vectorstore.as_retriever(
        search_kwargs={"k": k}
    )