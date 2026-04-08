from src.ingestion.pdf_loader import load_pdfs_from_folder


def extract_text_documents(folder_path: str):
    """
    Devuelve los documentos cargados desde PDFs.
    Cada documento corresponde a una página con su metadata.
    """
    return load_pdfs_from_folder(folder_path)