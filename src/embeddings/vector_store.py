from langchain_community.vectorstores import FAISS
from src.embeddings.embedder import get_embedder


def build_vector_store(chunks, persist_path: str):
    embeddings = get_embedder()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(persist_path)
    return vector_store


def load_vector_store(persist_path: str):
    embeddings = get_embedder()
    return FAISS.load_local(
        persist_path,
        embeddings,
        allow_dangerous_deserialization=True
    )