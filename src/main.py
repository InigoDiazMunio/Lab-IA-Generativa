from ingestion.pdf_loader import load_pdfs_from_folder
from ingestion.chunking import split_documents
from embeddings.vector_store import build_vector_store, load_vector_store
from retrieval.retriever import retrieve_context
from generation.prompt_builder import build_prompt
from generation.llm import generate_answer


RAW_DATA_PATH = "data/raw"
VECTOR_STORE_PATH = "data/embeddings/faiss_index"


def build_index():
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
    print("Cargando índice vectorial...")
    vector_store = load_vector_store(VECTOR_STORE_PATH)

    while True:
        query = input("\nEscribe tu pregunta ('salir' para terminar): ").strip()

        if query.lower() == "salir":
            break

        retrieved_docs = retrieve_context(vector_store, query, k=4)
        prompt = build_prompt(query, retrieved_docs)
        answer = generate_answer(prompt)

        print("\n--- RESPUESTA ---")
        print(answer)

        print("\n--- FUENTES RECUPERADAS ---")
        for i, doc in enumerate(retrieved_docs, start=1):
            print(
                f"{i}. Archivo: {doc.metadata.get('source_file', 'desconocido')} | "
                f"Página: {doc.metadata.get('page', 'N/A')}"
            )


if __name__ == "__main__":
    print("1. Construir índice")
    print("2. Hacer preguntas")
    option = input("Selecciona una opción: ").strip()

    if option == "1":
        build_index()
    elif option == "2":
        ask_question()
    else:
        print("Opción no válida.")