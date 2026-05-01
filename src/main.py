"""
Menú principal actualizado.

Cambios:
- Lee config.yaml.
- Construye índice texto o multimodal.
- Añade answer_question() para evaluación automática.
- Usa ChromaDB/vector store desde config.
"""

import sys
import os
import logging
import warnings
from pathlib import Path

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

warnings.filterwarnings("ignore", message=".*symlinks.*")
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", category=UserWarning)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()

RAW_DATA_PATH = CONFIG["paths"]["raw_data"]
VECTOR_STORE_PATH = CONFIG["paths"]["vector_store"]


# ==============================
# FUNCIÓN PARA EVALUACIÓN
# ==============================
def answer_question(question: str) -> str:
    """
    Función puente para evaluate_rag.py.
    Devuelve solo la respuesta generada por el sistema RAG.
    """
    from src.embeddings.vector_store import load_vector_store
    from src.evaluation.rag_pipeline import answer_with_rag_multimodal

    vector_store = load_vector_store(VECTOR_STORE_PATH)
    answer, _, _ = answer_with_rag_multimodal(question, vector_store, k=4)

    return answer


# ==============================
# INDEXACIÓN
# ==============================
def build_index(multimodal: bool = False):
    from src.ingestion.pdf_loader import load_pdfs_from_folder
    from src.ingestion.chunking import split_documents

    print("Cargando PDFs...")
    docs = load_pdfs_from_folder(RAW_DATA_PATH)

    if not docs:
        print("No hay documentos para indexar.")
        return

    print(f"Páginas cargadas: {len(docs)}")
    print("Dividiendo en chunks...")
    chunks = split_documents(docs)
    print(f"Chunks generados: {len(chunks)}")

    if multimodal:
        from src.multimodal.indexer import build_combined_index
        print("Construyendo índice combinado (texto + imágenes)...")
        build_combined_index(chunks, RAW_DATA_PATH, VECTOR_STORE_PATH)
    else:
        from src.embeddings.vector_store import build_vector_store
        print("Construyendo índice vectorial (solo texto)...")
        build_vector_store(chunks, VECTOR_STORE_PATH)

    print("Índice creado correctamente.")


# ==============================
# RAG
# ==============================
def ask_question_with_rag():
    from src.embeddings.vector_store import load_vector_store
    from src.evaluation.rag_pipeline import answer_with_rag_multimodal

    print("Cargando índice vectorial...")
    vector_store = load_vector_store(VECTOR_STORE_PATH)

    while True:
        query = input("\nEscribe tu pregunta ('salir' para terminar): ").strip()
        if query.lower() == "salir":
            break

        answer, sources, _ = answer_with_rag_multimodal(query, vector_store, k=4)

        print("\n--- RESPUESTA RAG ---")
        print(answer)

        print("\n--- FUENTES RECUPERADAS ---")

        seen = set()
        unique_sources = []

        for s in sources:
            key = (s.get("source_file"), s.get("page"), s.get("type", "texto"))
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        for i, src in enumerate(unique_sources, start=1):
            print(f"{i}. {src.get('source_file')} | Pág. {src.get('page')}")


# ==============================
# BASELINE
# ==============================
def ask_question_baseline():
    from src.evaluation.baseline import answer_without_rag

    while True:
        query = input("\nEscribe tu pregunta para el baseline ('salir' para terminar): ").strip()
        if query.lower() == "salir":
            break

        answer = answer_without_rag(query)
        print("\n--- RESPUESTA BASELINE ---")
        print(answer)


# ==============================
# DATASET
# ==============================
def build_questions_dataset():
    from src.evaluation.dataset_builder import build_default_dataset
    build_default_dataset()


# ==============================
# COMPARACIÓN
# ==============================
def run_comparison():
    from src.evaluation.rag_vs_baseline import run_comparison as _run
    _run()


# ==============================
# EVALUACIÓN AVANZADA
# ==============================
def run_advanced_evaluation():
    from src.evaluation.evaluate_rag import evaluate_rag, print_summary, save_results

    results = evaluate_rag()
    print_summary(results)
    save_results(results)


# ==============================
# PIPELINE COMPLETO
# ==============================
def run_all():
    print("\n[1/4] Construyendo índice...")
    build_index(multimodal=False)

    print("\n[2/4] Generando dataset de preguntas...")
    build_questions_dataset()

    print("\n[3/4] Ejecutando comparación RAG vs baseline...")
    run_comparison()

    print("\n[4/4] Ejecutando evaluación avanzada...")
    run_advanced_evaluation()

    print("\nProceso completo terminado.")
    print("Revisa experiments/ y data/evaluation_results.json.")


# ==============================
# MENÚ
# ==============================
if __name__ == "__main__":
    while True:
        print("\n" + "=" * 50)
        print("LAB IA GENERATIVA — MENÚ PRINCIPAL")
        print("=" * 50)
        print("1. Construir índice (solo texto)")
        print("2. Construir índice (texto + imágenes con LLaVA)")
        print("3. Hacer preguntas con RAG")
        print("4. Hacer preguntas con baseline")
        print("5. Crear dataset de preguntas")
        print("6. Ejecutar comparación RAG vs baseline")
        print("7. Ejecutar evaluación avanzada")
        print("8. Ejecutar todo")
        print("0. Salir")

        option = input("Selecciona una opción: ").strip()

        if option == "1":
            build_index(multimodal=False)
        elif option == "2":
            build_index(multimodal=True)
        elif option == "3":
            ask_question_with_rag()
        elif option == "4":
            ask_question_baseline()
        elif option == "5":
            build_questions_dataset()
        elif option == "6":
            run_comparison()
        elif option == "7":
            run_advanced_evaluation()
        elif option == "8":
            run_all()
        elif option == "0":
            print("Saliendo...")
            break
        else:
            print("Opción no válida.")