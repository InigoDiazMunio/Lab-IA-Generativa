def has_sources(rag_sources) -> int:
    return 1 if rag_sources and len(rag_sources) > 0 else 0


def answer_is_empty(answer: str) -> int:
    return 1 if not answer or not answer.strip() else 0


def mentions_no_info(answer: str) -> int:
    lowered = answer.lower()
    patterns = [
        "no se encuentra en los documentos proporcionados",
        "no estoy seguro",
        "no dispongo de suficiente información",
        "no aparece en el contexto"
    ]
    return 1 if any(p in lowered for p in patterns) else 0


def simple_answer_length(answer: str) -> int:
    return len(answer.split()) if answer else 0