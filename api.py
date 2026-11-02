```python
"""
Backend — Asistente de Soporte Técnico (RAG)

Ejecutar con:
    uvicorn api:app --reload

Abrir luego:
    http://127.0.0.1:8000
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
from src.rag_chain import (
    get_llm,
    get_prompt_template,
    rag_pipeline,
)


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

app = FastAPI(title=config.ASSISTANT_NAME)

BASE_DIR = Path(__file__).resolve().parent
FAQS_PATH = BASE_DIR / "data" / "faqs.json"


# =========================================================
# RECURSOS PESADOS
# Se cargan una sola vez (lazy) y se reutilizan.
# =========================================================

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
        _vector_store = load_vectorstore(
            _lazy_embeddings()
        )

    return _vector_store


# =========================================================
# MODELOS
# =========================================================

class ChatRequest(BaseModel):
    question: str
    k: int = config.TOP_K


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def es_saludo(texto: str) -> bool:
    """
    Detecta saludos simples para responder directamente
    sin consultar ChromaDB ni ejecutar el RAG.
    """

    texto = texto.lower().strip()

    saludos = [
        "hola",
        "holaa",
        "holaaa",
        "buenas",
        "buenos dias",
        "buenos días",
        "buen dia",
        "buen día",
        "buenas tardes",
        "buenas noches",
        "hey",
        "hello",
        "hi",
        "que tal",
        "qué tal",
        "como estas",
        "cómo estás",
        "como estás",
        "cómo estas",
    ]

    return any(
        texto == saludo
        or texto.startswith(saludo + " ")
        or texto.startswith(saludo + ",")
        or texto.startswith(saludo + "!")
        for saludo in saludos
    )


# =========================================================
# ESTADO DEL SISTEMA
# =========================================================

@app.get("/api/status")
def status():
    return {
        "indexed": index_exists(),
        "assistant_name": config.ASSISTANT_NAME,
        "empresa": config.EMPRESA,
    }


# =========================================================
# FAQS
# =========================================================

@app.get("/api/faqs")
def get_faqs():

    if not FAQS_PATH.exists():
        return []

    return json.loads(
        FAQS_PATH.read_text(
            encoding="utf-8"
        )
    )


# =========================================================
# INDEXACIÓN DE DOCUMENTOS
# =========================================================

@app.post("/api/index")
def reindex():
    """
    Reprocesa los PDF almacenados en data/pdfs
    y reconstruye la base vectorial.
    """

    global _vector_store

    try:

        documents = load_pdfs()

        chunks = split_documents(
            documents
        )

        build_vectorstore(
            chunks,
            _lazy_embeddings(),
            rebuild=True,
        )

        # Se fuerza a recargar la base vectorial
        # en la próxima consulta.
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


# =========================================================
# CHAT
# =========================================================

@app.post("/api/chat")
def chat(req: ChatRequest):

    # -----------------------------------------------------
    # 1. Validar que la pregunta no esté vacía
    # -----------------------------------------------------

    if not req.question.strip():

        raise HTTPException(
            status_code=400,
            detail="La pregunta no puede estar vacía.",
        )


    # -----------------------------------------------------
    # 2. Detectar saludos
    #
    # Los saludos NO pasan por ChromaDB ni por el RAG.
    # Esto evita búsquedas innecesarias en los manuales.
    # -----------------------------------------------------

    if es_saludo(req.question):

        return {
            "respuesta": (
                "¡Hola! 👋 Soy AstreaS, tu asistente de soporte técnico. "
                "Estoy aquí para ayudarte con tus dudas. "
                "¿En qué puedo ayudarte hoy?"
            ),
            "fuentes": [],
        }


    # -----------------------------------------------------
    # 3. Verificar que exista la base vectorial
    #
    # Esta validación se realiza después del saludo
    # para que AstreaS pueda saludar aunque todavía
    # no existan documentos indexados.
    # -----------------------------------------------------

    if not index_exists():

        raise HTTPException(
            status_code=400,
            detail=(
                "Aún no hay manuales indexados. "
                "Indexa los PDF antes de usar el chat."
            ),
        )


    # -----------------------------------------------------
    # 4. Ejecutar pipeline RAG
    # -----------------------------------------------------

    resultado = rag_pipeline(
        _lazy_vectorstore(),
        _lazy_llm(),
        _lazy_prompt_template(),
        req.question,
        k=req.k,
    )


    # -----------------------------------------------------
    # 5. Preparar las fuentes utilizadas
    # -----------------------------------------------------

    fuentes = [
        {
            "archivo": Path(
                d.metadata.get(
                    "source",
                    "?",
                )
            ).name,

            "pagina": d.metadata.get(
                "page",
                "?",
            ),

            "fragmento": d.page_content[:300],
        }

        for d in resultado["fragmentos"]
    ]


    # -----------------------------------------------------
    # 6. Respuesta final
    # -----------------------------------------------------

    return {
        "respuesta": resultado["respuesta"],
        "fuentes": fuentes,
    }


# =========================================================
# INTERFAZ ESTÁTICA
# =========================================================

# Sirve:
# - index.html
# - style.css
# - app.js
#
# IMPORTANTE:
# Debe montarse AL FINAL para no tapar las rutas /api/*.

app.mount(
    "/",
    StaticFiles(
        directory=BASE_DIR / "static",
        html=True,
    ),
    name="static",
)
```
