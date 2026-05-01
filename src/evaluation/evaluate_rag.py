import json
import re
from statistics import mean

from rouge_score import rouge_scorer
from src.vectorstore.chroma_store import get_retriever
from src.main import answer_question  # cambia esto si tu función se llama distinto


DATASET_PATH = "data/eval_dataset.json"
K = 3


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\sáéíóúñü]", "", text)
    return text


def exact_match(prediction, reference):
    return int(normalize_text(prediction) == normalize_text(reference))


def rouge_scores(prediction, reference):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )
    scores = scorer.score(reference, prediction)

    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def is_relevant(doc, relevant_sources):
    """
    Comprueba si un documento recuperado pertenece a las fuentes esperadas.
    relevant_sources puede contener nombres de archivo o páginas.
    """
    metadata = doc.metadata or {}

    source = str(metadata.get("source", "")).lower()
    page = str(metadata.get("page", "")).lower()

    for rel in relevant_sources:
        rel = str(rel).lower()

        if rel in source or rel == page:
            return True

    return False


def precision_at_k(retrieved_docs, relevant_sources, k):
    retrieved_docs = retrieved_docs[:k]

    if not retrieved_docs:
        return 0

    relevant_retrieved = sum(
        is_relevant(doc, relevant_sources)
        for doc in retrieved_docs
    )

    return relevant_retrieved / len(retrieved_docs)


def recall_at_k(retrieved_docs, relevant_sources, k):
    if not relevant_sources:
        return 0

    retrieved_docs = retrieved_docs[:k]

    found_relevant = set()

    for doc in retrieved_docs:
        metadata = doc.metadata or {}
        source = str(metadata.get("source", "")).lower()
        page = str(metadata.get("page", "")).lower()

        for rel in relevant_sources:
            rel_norm = str(rel).lower()
            if rel_norm in source or rel_norm == page:
                found_relevant.add(rel_norm)

    return len(found_relevant) / len(relevant_sources)


def simple_faithfulness(answer, contexts):
    """
    Métrica aproximada:
    mide qué porcentaje de palabras importantes de la respuesta aparecen en el contexto.
    No sustituye a RAGAS, pero sirve como métrica automática local.
    """
    answer_words = set(normalize_text(answer).split())
    context_words = set(normalize_text(" ".join(contexts)).split())

    stopwords = {
        "el", "la", "los", "las", "un", "una", "unos", "unas",
        "de", "del", "que", "en", "y", "o", "a", "con", "por",
        "para", "es", "son", "se", "al", "lo", "como"
    }

    answer_words = {
        w for w in answer_words
        if len(w) > 3 and w not in stopwords
    }

    if not answer_words:
        return 0

    supported_words = answer_words.intersection(context_words)

    return len(supported_words) / len(answer_words)


def simple_answer_relevance(question, answer):
    """
    Métrica aproximada:
    mide solapamiento léxico entre pregunta y respuesta.
    """
    question_words = set(normalize_text(question).split())
    answer_words = set(normalize_text(answer).split())

    if not question_words:
        return 0

    overlap = question_words.intersection(answer_words)

    return len(overlap) / len(question_words)


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_rag():
    dataset = load_dataset()
    retriever = get_retriever(k=K)

    results = []

    for sample in dataset:
        question = sample["question"]
        reference = sample["reference_answer"]
        relevant_sources = sample.get("relevant_sources", [])

        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]

        answer = answer_question(question)

        rouge = rouge_scores(answer, reference)

        result = {
            "question": question,
            "answer": answer,
            "reference_answer": reference,

            "precision_at_k": precision_at_k(docs, relevant_sources, K),
            "recall_at_k": recall_at_k(docs, relevant_sources, K),

            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],

            "exact_match": exact_match(answer, reference),

            "faithfulness_simple": simple_faithfulness(answer, contexts),
            "answer_relevance_simple": simple_answer_relevance(question, answer),
        }

        results.append(result)

    return results


def print_summary(results):
    metrics = [
        "precision_at_k",
        "recall_at_k",
        "rouge1",
        "rouge2",
        "rougeL",
        "exact_match",
        "faithfulness_simple",
        "answer_relevance_simple",
    ]

    print("\nRESULTADOS MEDIOS")
    print("=================")

    for metric in metrics:
        values = [r[metric] for r in results]
        print(f"{metric}: {mean(values):.4f}")


def save_results(results):
    output_path = "data/evaluation_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nResultados guardados en: {output_path}")


if __name__ == "__main__":
    results = evaluate_rag()
    print_summary(results)
    save_results(results)