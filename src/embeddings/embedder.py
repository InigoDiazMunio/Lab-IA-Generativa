from langchain_huggingface import HuggingFaceEmbeddings


def get_embedder():
    """
    Usa un modelo multilingüe en lugar de all-MiniLM-L6-v2 (solo inglés).
    paraphrase-multilingual-mpnet-base-v2 funciona bien en español e inglés,
    que es lo habitual en papers académicos con preguntas en español.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  
    )
