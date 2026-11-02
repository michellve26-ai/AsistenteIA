"""
Backend — Asistente de Soporte Técnico (RAG)
Ejecutar con: uvicorn api:app --reload
Abre luego: http://127.0.0.1:8000
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import config
from src.ingest import load_pdfs, split_documents
from src.vectorstore import (
    get_embeddings,
    build_vectorstore,
    load_vectorstore,
    index_exists,
)
from src.rag_chain import get_llm, get_prompt_template, rag_pipeline


app = FastAPI(title=config.ASSISTANT_NAME)

BASE_DIR = Path(__file__).resolve().parent
FAQS_PATH = BASE_DIR / "data" / "faqs.json"


# Recursos pesados: se cargan una sola vez (lazy)
# y se reutilizan entre peticiones.
_embeddings = None
_llm = None
_prompt_template = None
_vector_store = None


def _lazy_embeddings():
    global _embeddings

    if _embeddings is None:
        _embeddings = get_embeddings()

    return _embeddings


def _lazy_llm():
    global _llm

    if _llm is None:
        _llm = get_llm()

    return _llm


def _lazy_prompt_template():
    global _prompt_template

    if _prompt_template is None:
        _prompt_template = get_prompt_template()

    return _prompt_template


def _lazy_vectorstore():
    global _vector_store

    if _vector_store is None:
        _vector_store = load_vectorstore(_lazy_embeddings())

    return _vector_store


class ChatRequest(BaseModel):
    question: str
    k: int = config.TOP_K


# =========================================================
# DETECCIÓN DE SALUDOS
# =========================================================

def es_saludo(texto: str) -> bool:
    """
    Detecta saludos simples.

    Los saludos se responden directamente y no pasan
    por ChromaDB ni por el pipeline RAG.
    """

    texto = texto.lower().strip()

    # Quitamos signos comunes al final.
    texto = texto.rstrip("!?.,;:")

    saludos = {
        "hola",
        "holaa",
        "holaaa",
        "buenas",
        "buen día",
        "buen dia",
        "buenos días",
        "buenos dias",
        "buenas tardes",
        "buenas noches",
        "hey",
        "hello",
        "hi",
        "qué tal",
        "que tal",
        "cómo estás",
        "como estas",
        "cómo estas",
        "como estás",
    }

    return texto in saludos


@app.get("/api/status")
def status():
    return {
        "indexed": index_exists(),
        "assistant_name": config.ASSISTANT_NAME,
        "empresa": config.EMPRESA,
    }


@app.get("/api/faqs")
def get_faqs():
    if not FAQS_PATH.exists():
        return []

    return json.loads(
        FAQS_PATH.read_text(
            encoding="utf-8"
        )
    )


@app.post("/api/index")
def reindex():
    """
    Reprocesa los PDF de data/pdfs
    y reconstruye la base vectorial.
    """

    global _vector_store

    try:
        documents = load_pdfs()
        chunks = split_documents(documents)

        build_vectorstore(
            chunks,
            _lazy_embeddings(),
            rebuild=True,
        )

        # Se recarga en la próxima consulta.
        _vector_store = None

        return {
            "ok": True,
            "chunks_indexados": len(chunks),
        }

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


@app.post("/api/chat")
def chat(req: ChatRequest):

    # =====================================================
    # 1. Validar pregunta
    # =====================================================

    pregunta = req.question.strip()

    if not pregunta:
        raise HTTPException(
            status_code=400,
            detail="La pregunta no puede estar vacía.",
        )


    # =====================================================
    # 2. RESPONDER SALUDOS
    # =====================================================

    if es_saludo(pregunta):
        return {
            "respuesta": (
                "¡Hola! 👋 Soy Astrea, tu asistente virtual de soporte. "
                "Estoy aquí para ayudarte con tus dudas sobre OficinaPro. "
                "¿En qué puedo ayudarte hoy?"
            ),
            "fuentes": [],
        }


    # =====================================================
    # 3. Verificar base vectorial
    # =====================================================

    if not index_exists():
        raise HTTPException(
            status_code=400,
            detail=(
                "Aún no hay manuales indexados. "
                "Indexa los PDF antes de usar el chat."
            ),
        )


    # =====================================================
    # 4. Ejecutar RAG
    # =====================================================

    resultado = rag_pipeline(
        _lazy_vectorstore(),
        _lazy_llm(),
        _lazy_prompt_template(),
        pregunta,
        k=req.k,
    )


    # =====================================================
    # 5. Fuentes
    # =====================================================

    fuentes = [
        {
            "archivo": Path(
                d.metadata.get("source", "?")
            ).name,

            "pagina": d.metadata.get(
                "page",
                "?"
            ),

            "fragmento": d.page_content[:300],
        }

        for d in resultado["fragmentos"]
    ]


    return {
        "respuesta": resultado["respuesta"],
        "fuentes": fuentes,
    }


# =========================================================
# INTERFAZ ESTÁTICA
# =========================================================

# IMPORTANTE:
# Debe mantenerse AL FINAL para no tapar /api/*.

app.mount(
    "/",
    StaticFiles(
        directory=BASE_DIR / "static",
        html=True,
    ),
    name="static",
)
