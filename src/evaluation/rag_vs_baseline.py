"""
Comparación RAG vs Baseline con métricas reales.

Mejoras en esta versión:
- Incluye ROUGE contra reference_answer en el resumen estadístico
- Muestra las métricas por categoría de pregunta
- Lee config desde config.yaml
"""

import json
from pathlib import Path

import yaml

from src.embeddings.vector_store import load_vector_store
from src.evaluation.baseline import answer_without_rag
from src.evaluation.rag_pipeline import answer_with_rag_multimodal
from src.evaluation.metrics import compute_all_metrics


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_questions(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def compute_summary(results: list, system: str) -> dict:
    """Calcula promedios de todas las métricas numéricas disponibles."""
    numeric_keys = [
        "answer_length_words", "faithfulness",
        "rouge1_vs_context", "rougeL_vs_context",
        "rouge1_f1", "rouge2_f1", "rougeL_f1",
        "is_empty", "mentions_no_info"
    ]
    summary = {"system": system, "n_questions": len(results)}

    for key in numeric_keys:
        values = [r["metrics"].get(key) for r in results
                  if "metrics" in r and r["metrics"].get(key) is not None]
        if values:
            summary[f"avg_{key}"] = round(sum(values) / len(values), 4)

    return summary


def run_comparison():
    config = load_config()

    vector_store_path = config["paths"]["vector_store"]
    questions_path    = config["paths"]["questions"]
    experiments_dir   = config["paths"]["experiments"]

    questions    = load_questions(questions_path)
    vector_store = load_vector_store(vector_store_path)

    rag_results      = []
    baseline_results = []

    for item in questions:
        qid       = item["id"]
        question  = item["question"]
        category  = item.get("category", "")
        reference = item.get("reference_answer", None)

        print(f"\n[{qid}] {question}")

        # ── RAG ──────────────────────────────────────────────────────────────
        rag_answer, rag_sources, retrieved_docs = answer_with_rag_multimodal(
            question, vector_store, k=4
        )
        print(f"  RAG → {rag_answer[:100]}...")

        rag_metrics = compute_all_metrics(
            answer=rag_answer,
            rag_sources=rag_sources,
            retrieved_docs=retrieved_docs,
            reference_answer=reference,
        )

        rag_results.append({
            "id": qid, "question": question, "category": category,
            "answer": rag_answer, "sources": rag_sources,
            "metrics": rag_metrics,
        })

        # ── Baseline ─────────────────────────────────────────────────────────
        baseline_answer = answer_without_rag(question)
        print(f"  Baseline → {baseline_answer[:100]}...")

        baseline_metrics = compute_all_metrics(
            answer=baseline_answer,
            rag_sources=None,
            retrieved_docs=None,
            reference_answer=reference,
        )

        baseline_results.append({
            "id": qid, "question": question, "category": category,
            "answer": baseline_answer, "metrics": baseline_metrics,
        })

    # ── Guardar resultados ────────────────────────────────────────────────────
    save_json(rag_results,      f"{experiments_dir}/rag_results.json")
    save_json(baseline_results, f"{experiments_dir}/baseline_results.json")

    # ── Resumen estadístico ───────────────────────────────────────────────────
    rag_summary      = compute_summary(rag_results,      "RAG")
    baseline_summary = compute_summary(baseline_results, "Baseline")
    summary          = [rag_summary, baseline_summary]
    save_json(summary, f"{experiments_dir}/summary.json")

    print("\n" + "=" * 60)
    print("RESUMEN DE MÉTRICAS")
    print("=" * 60)
    for s in summary:
        print(f"\n{s['system']} ({s['n_questions']} preguntas):")
        for k, v in s.items():
            if k not in ("system", "n_questions"):
                print(f"  {k}: {v}")

    print(f"\nResultados guardados en: {experiments_dir}/")


if __name__ == "__main__":
    run_comparison()