from pathlib import Path
from langchain_core.documents import Document

from src.multimodal.image_extraction import extract_images_from_pdf
from src.multimodal.captioning import generate_basic_caption


def build_caption_documents(raw_data_path: str, images_output_path: str = "data/processed/images") -> list:
    raw_folder = Path(raw_data_path)
    pdf_files = list(raw_folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No se encontraron PDFs en {raw_data_path}")
        return []

    caption_docs = []

    for pdf_path in pdf_files:
        print(f"Procesando imágenes de: {pdf_path.name}")

        try:
            extracted = extract_images_from_pdf(str(pdf_path), images_output_path)
        except Exception as e:
            print(f"  [ERROR] No se pudieron extraer imágenes de {pdf_path.name}: {e}")
            continue

        print(f"  {len(extracted)} imágenes encontradas")

        for item in extracted:
            image_path = item["image_path"]
            page = item["page"]

            caption = generate_basic_caption(image_path)

            if not caption or "No se pudo generar" in caption:
                continue

            doc = Document(
                page_content=f"[Contenido visual] {caption}",
                metadata={
                    "source_file": pdf_path.name,
                    "source": str(pdf_path),
                    "page": page,
                    "type": "image_caption",
                    "image_path": image_path,
                },
            )
            caption_docs.append(doc)
            print(f"  Pág. {page}: {caption[:80]}...")

    print(f"\nTotal de documentos visuales generados: {len(caption_docs)}")
    return caption_docs


def build_combined_index(text_chunks: list, raw_data_path: str, vector_store_path: str):
    """
    Construye un índice ChromaDB combinando chunks de texto + captions.
    """
    from src.embeddings.vector_store import build_vector_store

    print("\n[1/2] Generando captions de imágenes...")
    caption_docs = build_caption_documents(raw_data_path)

    all_docs = text_chunks + caption_docs
    print(f"\n[2/2] Indexando {len(text_chunks)} chunks de texto + {len(caption_docs)} captions...")

    vector_store = build_vector_store(all_docs, vector_store_path, reset=True)

    print(f"Índice ChromaDB combinado guardado en: {vector_store_path}")
    return vector_store
