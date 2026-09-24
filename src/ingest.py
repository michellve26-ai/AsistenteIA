"""
Carga de manuales en PDF y división en fragmentos (chunking).
"""
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def load_pdfs(pdf_dir: str = config.PDF_DIR) -> list:
    """Carga todos los PDF de un directorio y devuelve la lista de Documents (uno por página)."""
    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(
            f"No existe la carpeta '{pdf_dir}'. Crea 'data/pdfs' y coloca allí los manuales en PDF."
        )

    pdf_files = sorted(f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No se encontraron archivos .pdf en '{pdf_dir}'.")

    documents = []
    for pdf_file in pdf_files:
        path = os.path.join(pdf_dir, pdf_file)
        loader = PyPDFLoader(path)
        pages = loader.load()
        documents.extend(pages)
        print(f"  {pdf_file}: {len(pages)} páginas cargadas")

    print(f"Total de páginas cargadas: {len(documents)}")
    return documents


def split_documents(documents: list) -> list:
    """Divide los documentos en fragmentos (chunks) para indexarlos."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"Fragmentos generados: {len(chunks)} (a partir de {len(documents)} páginas)")
    return chunks
