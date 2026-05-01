from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.vectorstore.chroma_store import create_vectorstore


DATA_PATH = "data/raw"


def load_documents():
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    return splitter.split_documents(documents)


def main():
    print("Cargando documentos...")
    documents = load_documents()

    print(f"Documentos cargados: {len(documents)}")

    print("Dividiendo documentos en chunks...")
    chunks = split_documents(documents)

    print(f"Chunks generados: {len(chunks)}")

    print("Creando base de datos vectorial...")
    create_vectorstore(chunks, reset=False)

    print("Base de datos vectorial creada correctamente.")


if __name__ == "__main__":
    main()