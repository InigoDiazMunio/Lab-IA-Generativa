from src.retrieval.retriever import retrieve_context
from src.generation.prompt_builder import build_rag_prompt
from src.generation.llm import generate_answer


def answer_with_rag(query: str, vector_store, k: int = 3):
    retrieved_docs = retrieve_context(vector_store, query, k=k)
    prompt = build_rag_prompt(query, retrieved_docs)
    answer = generate_answer(prompt).strip()

    sources = []
    for doc in retrieved_docs:
        sources.append({
            "source_file": doc.metadata.get("source_file", doc.metadata.get("source", "desconocido")),
            "page": doc.metadata.get("page", "N/A"),
            "content_preview": doc.page_content[:300]
        })

    return answer, sources