import json
from pathlib import Path

from src.embeddings.vector_store import load_vector_store
from src.evaluation.baseline import answer_without_rag
from src.evaluation.rag_pipeline import answer_with_rag


VECTOR_STORE_PATH = "data/embeddings/faiss_index"
QUESTIONS_PATH = "src/evaluation/questions.json"
OUTPUT_PATH = "experiments/evaluation_results.json"


def load_questions(path: str = QUESTIONS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation():
    questions = load_questions()
    vector_store = load_vector_store(VECTOR_STORE_PATH)

    results = []

    for item in questions:
        query = item["question"]

        print(f"Evaluando pregunta {item['id']}: {query}")

        rag_answer, rag_sources = answer_with_rag(query, vector_store)
        baseline_answer = answer_without_rag(query)

        results.append({
            "id": item["id"],
            "question": query,
            "category": item.get("category", ""),
            "rag_answer": rag_answer,
            "rag_sources": rag_sources,
            "baseline_answer": baseline_answer
        })

    Path("experiments").mkdir(exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResultados guardados en: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_evaluation()