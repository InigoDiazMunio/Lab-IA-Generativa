"""
Métricas de evaluación para el sistema RAG.

Mejoras respecto a la versión anterior:
1. Umbral de faithfulness bajado de 0.25 a 0.10 para compensar el gap
   idiomático (respuestas en español, contexto en inglés)
2. ROUGE se calcula también contra reference_answer cuando existe,
   no solo contra el contexto recuperado
3. Se añade rouge1_f1 y rougeL_f1 vs referencia como métricas principales
4. El aviso de rouge-score no instalado se muestra solo una vez
"""

from typing import Optional

_ROUGE_IMPORT_WARNING_SHOWN = False


# ── Métricas básicas ─────────────────────────────────────────────────────────

def has_sources(rag_sources) -> int:
    return 1 if rag_sources and len(rag_sources) > 0 else 0


def answer_is_empty(answer: str) -> int:
    return 1 if not answer or not answer.strip() else 0


def mentions_no_info(answer: str) -> int:
    lowered = answer.lower()
    patterns = [
        "no se encuentra en los documentos",
        "no dispongo de",
        "no aparece en el contexto",
        "no tengo información",
        "no estoy seguro",
        "no puedo responder",
        "fuera de los documentos",
        "no se menciona",
        "no hay información",
    ]
    return 1 if any(p in lowered for p in patterns) else 0


def simple_answer_length(answer: str) -> int:
    return len(answer.split()) if answer else 0


# ── ROUGE ────────────────────────────────────────────────────────────────────

def compute_rouge(hypothesis: str, reference: str) -> dict:
    """
    Calcula ROUGE-1, ROUGE-2 y ROUGE-L entre hypothesis y reference.

    En el contexto RAG:
    - hypothesis = respuesta generada
    - reference  = respuesta de referencia O contexto recuperado

    Nota: con respuestas en español y contexto en inglés el ROUGE léxico
    será bajo estructuralmente. Bajamos el umbral de faithfulness a 0.10.
    """
    global _ROUGE_IMPORT_WARNING_SHOWN

    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=False
        )
        scores = scorer.score(reference, hypothesis)

        return {
            "rouge1_f1": round(scores["rouge1"].fmeasure, 4),
            "rouge2_f1": round(scores["rouge2"].fmeasure, 4),
            "rougeL_f1": round(scores["rougeL"].fmeasure, 4),
        }

    except ImportError:
        if not _ROUGE_IMPORT_WARNING_SHOWN:
            print("[AVISO] rouge-score no instalado. Ejecuta: pip install rouge-score")
            _ROUGE_IMPORT_WARNING_SHOWN = True
        return {"rouge1_f1": 0.0, "rouge2_f1": 0.0, "rougeL_f1": 0.0}

    except Exception as e:
        print(f"[AVISO] Error calculando ROUGE: {e}")
        return {"rouge1_f1": 0.0, "rouge2_f1": 0.0, "rougeL_f1": 0.0}


# ── Faithfulness ─────────────────────────────────────────────────────────────

def compute_faithfulness(answer: str, retrieved_docs: list) -> float:
    """
    Estima qué tan fiel es la respuesta al contexto recuperado.

    Umbral bajado a 0.10 (antes 0.25) para compensar el gap idiomático:
    las respuestas están en español pero los chunks de los papers en inglés,
    lo que reduce el solapamiento léxico aunque el contenido sea correcto.

    Returns:
        Float entre 0.0 y 1.0
    """
    if not answer or not retrieved_docs:
        return 0.0

    full_context = " ".join(doc.page_content for doc in retrieved_docs)
    sentences = [s.strip() for s in answer.split(".") if len(s.strip().split()) >= 4]

    if not sentences:
        return 0.0

    supported = 0
    for sentence in sentences:
        rouge = compute_rouge(sentence, full_context)
        if rouge["rouge1_f1"] > 0.10:
            supported += 1

    return round(supported / len(sentences), 4)


# ── Función principal ─────────────────────────────────────────────────────────

def compute_all_metrics(
    answer: str,
    rag_sources: Optional[list] = None,
    retrieved_docs: Optional[list] = None,
    reference_answer: Optional[str] = None,
) -> dict:
    """
    Calcula todas las métricas disponibles para una respuesta.

    Métricas incluidas:
    - has_sources:          si el RAG recuperó fuentes
    - is_empty:             si la respuesta está vacía
    - mentions_no_info:     si el sistema dice que no tiene información
    - answer_length_words:  longitud en palabras
    - faithfulness:         proporción de frases apoyadas en el contexto
    - rouge1/rougeL_vs_context:    solapamiento con chunks recuperados
    - rouge1/rouge2/rougeL_f1:     solapamiento con respuesta de referencia
                                    (solo si reference_answer está disponible)
    """
    metrics = {
        "has_sources": has_sources(rag_sources) if rag_sources is not None else 0,
        "is_empty": answer_is_empty(answer),
        "mentions_no_info": mentions_no_info(answer),
        "answer_length_words": simple_answer_length(answer),
    }

    if retrieved_docs:
        metrics["faithfulness"] = compute_faithfulness(answer, retrieved_docs)

    if reference_answer:
        rouge_ref = compute_rouge(answer, reference_answer)
        metrics["rouge1_f1"] = rouge_ref["rouge1_f1"]
        metrics["rouge2_f1"] = rouge_ref["rouge2_f1"]
        metrics["rougeL_f1"] = rouge_ref["rougeL_f1"]

    if retrieved_docs:
        full_context = " ".join(doc.page_content for doc in retrieved_docs)
        rouge_ctx = compute_rouge(answer, full_context)
        metrics["rouge1_vs_context"] = rouge_ctx["rouge1_f1"]
        metrics["rougeL_vs_context"] = rouge_ctx["rougeL_f1"]

    return metrics