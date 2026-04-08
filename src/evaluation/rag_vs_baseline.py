import json
from pathlib import Path

from src.embeddings.vector_store import load_vector_store
from src.evaluation.baseline import answer_without_rag
from src.evaluation.rag_pipeline import answer_with_rag
from src.evaluation.metrics import has_sources, answer_is_empty, mentions_no_info, simple_answer_length


VECTOR_STORE_PATH = "data/embeddings/faiss_index"
QUESTIONS_PATH = "src/evaluation/questions.json"
EXPERIMENTS_DIR = "experiments"


def load_questions(path: str = QUESTIONS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_comparison():
    questions = load_questions()
    vector_store = load_vector_store(VECTOR_STORE_PATH)

    rag_results = []
    baseline_results = []
    comparison_results = []

    for item in questions:
        qid = item["id"]
        question = item["question"]
        category = item.get("category", "")

        print(f"Procesando pregunta {qid}: {question}")

        rag_answer, rag_sources = answer_with_rag(question, vector_store)
        baseline_answer = answer_without_rag(question)

        rag_item = {
            "id": qid,
            "question": question,
            "category": category,
            "answer": rag_answer,
            "sources": rag_sources,
            "metrics": {
                "has_sources": has_sources(rag_sources),
                "is_empty": answer_is_empty(rag_answer),
                "mentions_no_info": mentions_no_info(rag_answer),
                "answer_length": simple_answer_length(rag_answer)
            }
        }

        baseline_item = {
            "id": qid,
            "question": question,
            "category": category,
            "answer": baseline_answer,
            "metrics": {
                "is_empty": answer_is_empty(baseline_answer),
                "mentions_no_info": mentions_no_info(baseline_answer),
                "answer_length": simple_answer_length(baseline_answer)
            }
        }

        comparison_item = {
            "id": qid,
            "question": question,
            "category": category,
            "rag_answer": rag_answer,
            "rag_sources": rag_sources,
            "baseline_answer": baseline_answer
        }

        rag_results.append(rag_item)
        baseline_results.append(baseline_item)
        comparison_results.append(comparison_item)

    save_json(rag_results, f"{EXPERIMENTS_DIR}/rag_results.json")
    save_json(baseline_results, f"{EXPERIMENTS_DIR}/baseline_results.json")
    save_json(comparison_results, f"{EXPERIMENTS_DIR}/rag_vs_baseline_results.json")

    print("\nResultados guardados en experiments/")


if __name__ == "__main__":
    run_comparison()