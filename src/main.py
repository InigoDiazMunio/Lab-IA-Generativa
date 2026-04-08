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


def ask_question():
    from src.embeddings.vector_store import load_vector_store
    from src.evaluation.rag_pipeline import answer_with_rag

    print("Cargando índice vectorial...")
    vector_store = load_vector_store(VECTOR_STORE_PATH)

    while True:
        query = input("\nEscribe tu pregunta ('salir' para terminar): ").strip()

        if query.lower() == "salir":
            break

        answer, sources = answer_with_rag(query, vector_store)

        print("\n--- RESPUESTA ---")
        print(answer)

        print("\n--- FUENTES RECUPERADAS ---")
        for i, src in enumerate(sources, start=1):
            print(f"{i}. Archivo: {src['source_file']} | Página: {src['page']}")


if __name__ == "__main__":
    print("1. Construir índice")
    print("2. Hacer preguntas con RAG")
    option = input("Selecciona una opción: ").strip()

    if option == "1":
        build_index()
    elif option == "2":
        ask_question()
    else:
        print("Opción no válida.")