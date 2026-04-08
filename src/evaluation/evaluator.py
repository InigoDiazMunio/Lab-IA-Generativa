import json
from pathlib import Path

from embeddings.vector_store import load_vector_store
from retrieval.retriever import retrieve_context
from generation.prompt_builder import build_prompt
from generation.llm import generate_answer
from evaluation.baseline import answer_without_rag


VECTOR_STORE_PATH = "data/embeddings/faiss_index"


def load_questions(path: str = "src/evaluation/questions.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def answer_with_rag(query: str, vector_store):
    retrieved_docs = retrieve_context(vector_store, query, k=4)
    prompt = build_prompt(query, retrieved_docs)
    answer = generate_answer(prompt)

    sources = []
    for doc in retrieved_docs:
        sources.append({
            "source_file": doc.metadata.get("source_file", "desconocido"),
            "page": doc.metadata.get("page", "N/A")
        })

    return answer, sources


def run_evaluation():
    questions = load_questions()
    vector_store = load_vector_store(VECTOR_STORE_PATH)

    results = []

    for item in questions:
        query = item["question"]

        rag_answer, rag_sources = answer_with_rag(query, vector_store)
        baseline_answer = answer_without_rag(query)

        results.append({
            "id": item["id"],
            "question": query,
            "reference_answer": item.get("reference_answer", ""),
            "rag_answer": rag_answer,
            "rag_sources": rag_sources,
            "baseline_answer": baseline_answer
        })

    output_path = Path("experiments")
    output_path.mkdir(exist_ok=True)

    with open(output_path / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Evaluación terminada. Resultados guardados en experiments/evaluation_results.json")


if __name__ == "__main__":
    run_evaluation()