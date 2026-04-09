"""
Indexación multimodal: integra captions de imágenes en el índice FAISS.

La estrategia es la más sencilla y robusta para un prototipo:
1. Extraer imágenes de cada PDF
2. Generar una descripción (caption) con LLaVA
3. Crear un Document de LangChain con el caption como contenido y
   metadatos que enlazan con el PDF y la página de origen
4. Añadir esos documentos al índice junto con los chunks de texto

Ventaja: no necesita embeddings multimodales (CLIP, etc.), solo el
mismo modelo de texto ya usado → menos complejidad, menos dependencias.
"""

from pathlib import Path
from langchain_core.documents import Document

from src.multimodal.image_extraction import extract_images_from_pdf
from src.multimodal.captioning import generate_basic_caption


def build_caption_documents(raw_data_path: str, images_output_path: str = "data/processed/images") -> list:
    """
    Extrae imágenes de todos los PDFs de la carpeta y genera documentos
    de texto con sus captions, listos para indexar en FAISS.

    Args:
        raw_data_path:      Carpeta con los PDFs (ej: "data/raw")
        images_output_path: Dónde guardar las imágenes extraídas

    Returns:
        Lista de LangChain Documents (uno por imagen con caption)
    """
    raw_folder = Path(raw_data_path)
    pdf_files  = list(raw_folder.glob("*.pdf"))

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
            page       = item["page"]

            caption = generate_basic_caption(image_path)

            if not caption or "No se pudo generar" in caption:
                continue  # saltar imágenes sin descripción útil

            # Crear documento con el caption como contenido
            doc = Document(
                page_content=f"[Contenido visual] {caption}",
                metadata={
                    "source_file": pdf_path.name,
                    "source":      str(pdf_path),
                    "page":        page,
                    "type":        "image_caption",
                    "image_path":  image_path,
                }
            )
            caption_docs.append(doc)
            print(f"  Pág. {page}: {caption[:80]}...")

    print(f"\nTotal de documentos visuales generados: {len(caption_docs)}")
    return caption_docs


def build_combined_index(text_chunks: list, raw_data_path: str, vector_store_path: str):
    """
    Construye un índice FAISS combinando chunks de texto + captions de imágenes.

    Args:
        text_chunks:        Chunks de texto ya divididos (de la ingesta normal)
        raw_data_path:      Carpeta con PDFs (para extraer imágenes)
        vector_store_path:  Dónde guardar el índice FAISS
    """
    from src.embeddings.embedder import get_embedder
    from langchain_community.vectorstores import FAISS

    print("\n[1/2] Generando captions de imágenes...")
    caption_docs = build_caption_documents(raw_data_path)

    all_docs = text_chunks + caption_docs
    print(f"\n[2/2] Indexando {len(text_chunks)} chunks de texto + {len(caption_docs)} captions...")

    embeddings    = get_embedder()
    vector_store  = FAISS.from_documents(all_docs, embeddings)
    vector_store.save_local(vector_store_path)

    print(f"Índice combinado guardado en: {vector_store_path}")
    return vector_store
