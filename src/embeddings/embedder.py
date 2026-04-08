from langchain_community.embeddings import HuggingFaceEmbeddings


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL):
    return HuggingFaceEmbeddings(model_name=model_name)