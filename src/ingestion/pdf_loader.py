from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_pdfs_from_folder(folder_path: str):
    folder = Path(folder_path)
    all_docs = []

    if not folder.exists():
        raise FileNotFoundError(f"No existe la carpeta: {folder_path}")

    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No se encontraron PDFs en {folder_path}")
        return []

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        for doc in docs:
            doc.metadata["source_file"] = pdf_file.name

        all_docs.extend(docs)

    return all_docs