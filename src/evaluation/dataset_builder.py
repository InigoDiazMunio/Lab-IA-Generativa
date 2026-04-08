import json
from pathlib import Path


DEFAULT_QUESTIONS = [
    {
        "id": 1,
        "question": "¿Qué es Retrieval-Augmented Generation?",
        "category": "teoria"
    },
    {
        "id": 2,
        "question": "¿Para qué sirven los embeddings en un sistema RAG?",
        "category": "teoria"
    },
    {
        "id": 3,
        "question": "¿Qué papel tiene una base de datos vectorial en este sistema?",
        "category": "teoria"
    },
    {
        "id": 4,
        "question": "¿Qué diferencia hay entre un sistema RAG textual y uno multimodal?",
        "category": "comparacion"
    },
    {
        "id": 5,
        "question": "¿Por qué el contenido visual puede ser problemático para un RAG puramente textual?",
        "category": "multimodalidad"
    }
]


def build_default_dataset(output_path: str = "src/evaluation/questions.json"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_QUESTIONS, f, indent=2, ensure_ascii=False)

    print(f"Dataset de preguntas guardado en {output_path}")


if __name__ == "__main__":
    build_default_dataset()