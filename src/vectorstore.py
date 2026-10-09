"""
Embeddings locales ligeros con FastEmbed y base vectorial ChromaDB.

Esta implementación evita Sentence Transformers / PyTorch para reducir
el consumo de memoria en entornos con recursos limitados como Render.
"""

import os
import shutil
import time

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

from . import config



def get_embeddings() -> FastEmbedEmbeddings:
    """Inicializa el modelo de embeddings y registra su tiempo de carga."""
    inicio = time.perf_counter()

    print(
        f"[EMBEDDINGS] Iniciando modelo: {config.EMBEDDING_MODEL}",
        flush=True,
    )

    try:
        embeddings = FastEmbedEmbeddings(
            model_name=config.EMBEDDING_MODEL
        )

        duracion = time.perf_counter() - inicio
        print(
            f"[EMBEDDINGS] Modelo cargado en {duracion:.2f} segundos",
            flush=True,
        )

        return embeddings

    except Exception as error:
        print(
            f"[EMBEDDINGS] Error durante la carga: {error}",
            flush=True,
        )
        raise


def build_vectorstore(
    chunks: list,
    embeddings: FastEmbedEmbeddings,
    rebuild: bool = True,
) -> Chroma:
    """
    Crea o recrea la base vectorial a partir de los fragmentos.
    """
    if rebuild and os.path.exists(config.PERSIST_DIR):
        shutil.rmtree(config.PERSIST_DIR)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.PERSIST_DIR,
        collection_name=config.COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(
        "[OK] Base vectorial creada: "
        f"{vector_store._collection.count()} fragmentos indexados."
    )

    return vector_store


def load_vectorstore(
    embeddings: FastEmbedEmbeddings,
) -> Chroma:
    """
    Carga una base vectorial existente sin volver a procesar los PDFs.
    """
    if not os.path.exists(config.PERSIST_DIR):
        raise FileNotFoundError(
            f"No existe '{config.PERSIST_DIR}'. "
            "Ejecuta primero la indexación."
        )

    return Chroma(
        persist_directory=config.PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=config.COLLECTION_NAME,
    )


def index_exists() -> bool:
    """
    Verifica si existe una base vectorial persistida.
    """
    return (
        os.path.exists(config.PERSIST_DIR)
        and len(os.listdir(config.PERSIST_DIR)) > 0
    )
