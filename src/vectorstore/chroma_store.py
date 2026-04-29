import os
import shutil

from langchain_chroma import Chroma
from src.embeddings.embedder import get_embedding_model


CHROMA_PATH = "vectorstore/chroma_db"


def create_vectorstore(chunks, reset: bool = True):
    """
    Crea la base de datos vectorial persistente con ChromaDB.
    Si reset=True, borra el índice anterior antes de crearlo.
    """
    if reset and os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embedding_model = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )

    return vectorstore


def load_vectorstore():
    """
    Carga la base de datos vectorial ya existente.
    """
    embedding_model = get_embedding_model()

    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )


def get_retriever(k: int = 4):
    """
    Devuelve el retriever para consultar el índice.
    """
    vectorstore = load_vectorstore()

    return vectorstore.as_retriever(
        search_kwargs={"k": k}
    )