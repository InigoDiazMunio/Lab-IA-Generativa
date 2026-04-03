from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def build_vector_store(chunks, persist_path: str):
    embeddings = get_embedding_model()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(persist_path)
    return vector_store


def load_vector_store(persist_path: str):
    embeddings = get_embedding_model()
    return FAISS.load_local(
        persist_path,
        embeddings,
        allow_dangerous_deserialization=True
    )