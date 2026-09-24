"""
Embeddings locales (Sentence Transformers) y base vectorial (ChromaDB).
"""
import os
import shutil

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from . import config


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks: list, embeddings: HuggingFaceEmbeddings, rebuild: bool = True) -> Chroma:
    """Crea (o recrea) la base vectorial a partir de los fragmentos."""
    if rebuild and os.path.exists(config.PERSIST_DIR):
        shutil.rmtree(config.PERSIST_DIR)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.PERSIST_DIR,
        collection_name=config.COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"[OK] Base vectorial creada: {vector_store._collection.count()} fragmentos indexados.")
    return vector_store


def load_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Carga una base vectorial ya existente en disco, sin reprocesar los PDFs."""
    if not os.path.exists(config.PERSIST_DIR):
        raise FileNotFoundError(
            f"No existe '{config.PERSIST_DIR}'. Ejecuta primero la indexación (ver README)."
        )
    return Chroma(
        persist_directory=config.PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=config.COLLECTION_NAME,
    )


def index_exists() -> bool:
    return os.path.exists(config.PERSIST_DIR) and len(os.listdir(config.PERSIST_DIR)) > 0
