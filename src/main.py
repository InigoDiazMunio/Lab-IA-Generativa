RAW_DATA_PATH = "data/raw"
VECTOR_STORE_PATH = "data/embeddings/faiss_index"


def build_index():
    from src.ingestion.pdf_loader import load_pdfs_from_folder
    from src.ingestion.chunking import split_documents
    from src.embeddings.vector_store import build_vector_store

    print("Cargando PDFs...")
    docs = load_pdfs_from_folder(RAW_DATA_PATH)

    if not docs:
        print("No hay documentos para indexar.")
        return

    print(f"Páginas cargadas: {len(docs)}")

    print("Dividiendo en chunks...")
    chunks = split_documents(docs)
    print(f"Chunks generados: {len(chunks)}")

    print("Construyendo índice vectorial...")
    build_vector_store(chunks, VECTOR_STORE_PATH)

    print("Índice creado correctamente.")


def ask_question_with_rag():
    from src.embeddings.vector_store import load_vector_store
    from src.evaluation.rag_pipeline import answer_with_rag

    print("Cargando índice vectorial...")
    vector_store = load_vector_store(VECTOR_STORE_PATH)

    while True:
        query = input("\nEscribe tu pregunta ('salir' para terminar): ").strip()

        if query.lower() == "salir":
            break

        answer, sources = answer_with_rag(query, vector_store)

        print("\n--- RESPUESTA RAG ---")
        print(answer)

        print("\n--- FUENTES RECUPERADAS ---")
        for i, src in enumerate(sources, start=1):
            print(f"{i}. Archivo: {src['source_file']} | Página: {src['page']}")


def ask_question_baseline():
    from src.evaluation.baseline import answer_without_rag

    while True:
        query = input("\nEscribe tu pregunta para el baseline ('salir' para terminar): ").strip()

        if query.lower() == "salir":
            break

        answer = answer_without_rag(query)

        print("\n--- RESPUESTA BASELINE ---")
        print(answer)


def build_questions_dataset():
    from src.evaluation.dataset_builder import build_default_dataset
    build_default_dataset()


def run_comparison():
    from src.evaluation.rag_vs_baseline import run_comparison as run_rag_vs_baseline
    run_rag_vs_baseline()


def run_all():
    print("\n[1/3] Construyendo índice...")
    build_index()

    print("\n[2/3] Generando dataset de preguntas...")
    build_questions_dataset()

    print("\n[3/3] Ejecutando comparación RAG vs baseline...")
    run_comparison()

    print("\nProceso completo terminado.")
    print("Revisa la carpeta experiments/ para ver los resultados.")


if __name__ == "__main__":
    while True:
        print("\n" + "=" * 50)
        print("LAB IA GENERATIVA - MENÚ PRINCIPAL")
        print("=" * 50)
        print("1. Construir índice")
        print("2. Hacer preguntas con RAG")
        print("3. Hacer preguntas con baseline")
        print("4. Crear dataset de preguntas")
        print("5. Ejecutar comparación RAG vs baseline")
        print("6. Ejecutar todo")
        print("0. Salir")

        option = input("Selecciona una opción: ").strip()

        if option == "1":
            build_index()
        elif option == "2":
            ask_question_with_rag()
        elif option == "3":
            ask_question_baseline()
        elif option == "4":
            build_questions_dataset()
        elif option == "5":
            run_comparison()
        elif option == "6":
            run_all()
        elif option == "0":
            print("Saliendo...")
            break
        else:
            print("Opción no válida.")