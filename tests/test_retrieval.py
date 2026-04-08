from src.generation.prompt_builder import build_baseline_prompt, build_rag_prompt


class DummyDoc:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def test_build_baseline_prompt():
    query = "¿Qué es RAG?"
    prompt = build_baseline_prompt(query)

    assert "¿Qué es RAG?" in prompt
    assert "conocimiento general" in prompt.lower()


def test_build_rag_prompt():
    query = "¿Qué es RAG?"
    docs = [
        DummyDoc(
            page_content="RAG combina recuperación y generación.",
            metadata={"source_file": "apuntes.pdf", "page": 2}
        )
    ]

    prompt = build_rag_prompt(query, docs)

    assert "¿Qué es RAG?" in prompt
    assert "apuntes.pdf" in prompt
    assert "Página: 2" in prompt