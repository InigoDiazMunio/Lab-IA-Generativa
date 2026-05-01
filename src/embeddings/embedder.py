from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_embedding_model():
    """
    Devuelve el modelo de embeddings usado por ChromaDB.
    Se usa un modelo multilingüe porque las preguntas pueden estar en español
    y muchos documentos están en inglés.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


# Alias para mantener compatibilidad con el código antiguo del proyecto.
def get_embedder():
    return get_embedding_model()
