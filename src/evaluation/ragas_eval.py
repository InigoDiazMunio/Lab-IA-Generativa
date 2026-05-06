"""
Evaluación del sistema RAG con RAGAS (v0.4+).

Métricas implementadas:
- Faithfulness:       ¿La respuesta se basa fielmente en el contexto recuperado?
- Answer Relevancy:   ¿La respuesta es relevante para la pregunta?
- Context Precision:  ¿Los fragmentos recuperados son precisos (relevantes para la pregunta)?
- Context Recall:     ¿Se recuperó todo el contexto necesario? (requiere reference_answer)

RAGAS usa un LLM como juez para evaluar estas métricas. En nuestro caso
usamos Ollama (local) a través de LangchainLLMWrapper.
"""

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


def run_ragas_eval(
    questions: list,
    answers: list,
    contexts: list,
    references: list = None,
    llm=None,
    embeddings=None,
):
    """
    Ejecuta la evaluación RAGAS sobre un conjunto de resultados RAG.

    Args:
        questions:   Lista de preguntas.
        answers:     Lista de respuestas generadas por el RAG.
        contexts:    Lista de listas de contextos recuperados (strings).
        references:  Lista de respuestas de referencia (ground truth).
                     Necesarias para context_recall.
        llm:         LLM envuelto con LangchainLLMWrapper para el juez.
        embeddings:  Embeddings envueltos con LangchainEmbeddingsWrapper
                     (para answer_relevancy). Si no se pasa, se omite esa métrica.

    Returns:
        Resultado de ragas.evaluate con las métricas calculadas.
    """

    # Construir el dataset en formato RAGAS
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }

    # Seleccionar métricas disponibles
    metrics = [faithfulness, context_precision]

    # answer_relevancy necesita embeddings
    if embeddings is not None:
        metrics.append(answer_relevancy)

    # context_recall necesita ground truth
    has_references = (
        references is not None
        and len(references) == len(questions)
        and any(r and r.strip() for r in references)
    )

    if has_references:
        data["ground_truth"] = references
        metrics.append(context_recall)

    dataset = Dataset.from_dict(data)

    # Preparar kwargs para evaluate
    eval_kwargs = {
        "metrics": metrics,
    }

    if llm is not None:
        eval_kwargs["llm"] = llm

    if embeddings is not None:
        eval_kwargs["embeddings"] = embeddings

    result = evaluate(dataset, **eval_kwargs)

    return result


def format_ragas_result(ragas_result) -> dict:
    """
    Convierte el resultado de RAGAS a un dict serializable para guardar en JSON.

    Returns:
        dict con 'columns', 'rows' (por pregunta) y 'averages' (promedios globales).
    """
    ragas_dict = {}

    if hasattr(ragas_result, "to_pandas"):
        ragas_df = ragas_result.to_pandas()

        # Convertir NaN a None para JSON
        ragas_df = ragas_df.where(ragas_df.notna(), None)

        ragas_dict["columns"] = list(ragas_df.columns)
        ragas_dict["rows"] = ragas_df.to_dict(orient="records")

        # Calcular promedios de las métricas numéricas
        metric_cols = [
            c for c in ragas_df.columns
            if c not in ("question", "answer", "contexts", "ground_truth")
        ]

        averages = {}
        for col in metric_cols:
            values = [
                v for v in ragas_df[col]
                if v is not None and isinstance(v, (int, float))
            ]
            if values:
                averages[col] = round(sum(values) / len(values), 4)

        ragas_dict["averages"] = averages

    else:
        ragas_dict = {"result": str(ragas_result)}

    return ragas_dict