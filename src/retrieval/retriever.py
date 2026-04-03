def retrieve_context(vector_store, query: str, k: int = 4):
    return vector_store.similarity_search(query, k=k)