"""
Backend — Asistente de Soporte Técnico (RAG)

Ejecutar con:
    uvicorn api:app --reload

Abrir luego:
    http://127.0.0.1:8000
"""

import json
import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

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
from src.index_state import (
    documentos_cambiaron,
    guardar_estado_actual,
    obtener_estado_actual,
)
from src.extra_docs import (
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_MB,
    CombinedVectorStore,
    ExtraDocs,
    ExtraDocsError,
    token_valido,
)


# =========================================================
# LOGGING
# =========================================================

logger = logging.getLogger(__name__)


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

app = FastAPI(
    title=config.ASSISTANT_NAME
)

BASE_DIR = Path(__file__).resolve().parent

FAQS_PATH = (
    BASE_DIR
    / "data"
    / "faqs.json"
)


# =========================================================
# RECURSOS PESADOS
# =========================================================
#
# Estos recursos se cargan de forma lazy:
#
# - embeddings
# - modelo LLM
# - prompt
# - base vectorial
#
# Esto evita cargarlos nuevamente en cada petición.
# =========================================================

_embeddings = None
_llm = None
_prompt_template = None
_vector_store = None


def _lazy_embeddings():
    """
    Carga los embeddings una sola vez.
    """

    global _embeddings

    if _embeddings is None:
        _embeddings = get_embeddings()

    return _embeddings


def _lazy_llm():
    """
    Carga el modelo LLM una sola vez.
    """

    global _llm

    if _llm is None:
        _llm = get_llm()

    return _llm


def _lazy_prompt_template():
    """
    Carga la plantilla principal del RAG una sola vez.
    """

    global _prompt_template

    if _prompt_template is None:
        _prompt_template = get_prompt_template()

    return _prompt_template


def _lazy_vectorstore():
    """
    Carga la base vectorial una sola vez.

    Si se realiza una nueva indexación,
    _vector_store se reinicia a None
    para forzar su recarga.
    """

    global _vector_store

    if _vector_store is None:
        _vector_store = load_vectorstore(
            _lazy_embeddings()
        )

    return _vector_store


# =========================================================
# MANUALES EXTRA (subida manual de PDF desde el chat)
# =========================================================
#
# Función adicional, aislada de la indexación automática:
# usa su propia base Chroma y su propia carpeta (ver src/extra_docs.py).

extra_docs = ExtraDocs(
    embeddings_provider=_lazy_embeddings
)


# =========================================================
# INDEXACIÓN AUTOMÁTICA AL INICIAR
# =========================================================

@app.on_event("startup")
def auto_reindexar_si_es_necesario():
    """
    Verifica automáticamente si los documentos PDF
    cambiaron desde la última indexación.

    Reindexa únicamente cuando:

    - no existe un índice vectorial, o
    - se agregó un PDF, o
    - se modificó un PDF, o
    - se eliminó un PDF.

    Si los documentos no cambiaron, reutiliza el índice
    existente y no recalcula embeddings.

    Una falla durante esta comprobación no impide
    que la aplicación pueda iniciar.
    """

    global _vector_store

    try:

        # -------------------------------------------------
        # OBTENER LOS PDF ACTUALES
        # -------------------------------------------------

        estado_actual = obtener_estado_actual()


        # -------------------------------------------------
        # SEGURIDAD:
        #
        # Si no existe ningún PDF actualmente,
        # no reconstruimos automáticamente la base.
        #
        # Esto evita eliminar o reemplazar accidentalmente
        # un índice existente por una colección vacía.
        # -------------------------------------------------

        if not estado_actual:

            logger.warning(
                "Autoindexación omitida: "
                "no se encontraron documentos PDF."
            )

            return


        # -------------------------------------------------
        # DETERMINAR SI ES NECESARIO REINDEXAR
        # -------------------------------------------------

        necesita_indexacion = (
            not index_exists()
            or documentos_cambiaron()
        )


        if not necesita_indexacion:

            logger.info(
                "Base vectorial actualizada. "
                "No es necesario reindexar."
            )

            return


        logger.info(
            "Se detectaron cambios en los documentos. "
            "Iniciando reindexación automática."
        )


        # -------------------------------------------------
        # CARGAR DOCUMENTOS
        # -------------------------------------------------

        documents = load_pdfs()


        # -------------------------------------------------
        # CREAR CHUNKS
        # -------------------------------------------------

        chunks = split_documents(
            documents
        )


        # -------------------------------------------------
        # SEGURIDAD:
        #
        # Nunca reconstruimos ChromaDB con cero chunks.
        # -------------------------------------------------

        if not chunks:

            logger.warning(
                "Autoindexación cancelada: "
                "no se generaron fragmentos."
            )

            return


        # -------------------------------------------------
        # RECONSTRUIR CHROMADB
        # -------------------------------------------------

        build_vectorstore(
            chunks,
            _lazy_embeddings(),
            rebuild=True,
        )


        # -------------------------------------------------
        # FORZAR RECARGA DEL VECTORSTORE
        # -------------------------------------------------

        _vector_store = None


        # -------------------------------------------------
        # GUARDAR ESTADO SOLO DESPUÉS DE ÉXITO
        # -------------------------------------------------

        guardar_estado_actual()


        logger.info(
            "Reindexación automática completada "
            "correctamente. Chunks indexados: %s",
            len(chunks),
        )


    except Exception:

        # -------------------------------------------------
        # IMPORTANTE:
        #
        # Un problema durante la autoindexación no debe
        # impedir que FastAPI / AstreaS inicie.
        #
        # El error queda registrado en los logs de Render.
        # -------------------------------------------------

        logger.exception(
            "No fue posible realizar la "
            "reindexación automática."
        )


# =========================================================
# MODELOS DE DATOS
# =========================================================


class ChatHistoryMessage(BaseModel):
    """
    Representa un mensaje anterior de la conversación.

    Solo se aceptan mensajes del usuario o del asistente.
    """

    role: Literal[
        "user",
        "assistant",
    ]

    content: str


class ChatRequest(BaseModel):
    """
    Datos recibidos por POST /api/chat.

    question:
        Pregunta actual del usuario.

    k:
        Cantidad de fragmentos que recuperará el RAG.

    historial:
        Mensajes anteriores utilizados para contextualizar
        preguntas de seguimiento.
    """

    question: str

    k: int = config.TOP_K

    historial: list[
        ChatHistoryMessage
    ] = Field(
        default_factory=list
    )


# =========================================================
# DETECCIÓN DE SALUDOS
# =========================================================


def es_saludo(
    texto: str,
) -> bool:
    """
    Detecta saludos simples.

    Los saludos se responden directamente y no pasan por:

    - Tool Calling
    - ChromaDB
    - RAG

    Esto evita búsquedas innecesarias.
    """

    texto = str(
        texto
    ).lower().strip()

    texto = texto.rstrip(
        "!?.,;:"
    )

    saludos = {
        "hola",
        "holaa",
        "holaaa",
        "holaaaa",
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


# =========================================================
# ESTADO DEL SISTEMA
# =========================================================


@app.get(
    "/api/status"
)
def status():
    """
    Devuelve información básica del estado del asistente.
    """

    return {
        "indexed": index_exists(),
        "assistant_name": config.ASSISTANT_NAME,
        "empresa": config.EMPRESA,
    }


# =========================================================
# PREGUNTAS FRECUENTES
# =========================================================


@app.get(
    "/api/faqs"
)
def get_faqs():
    """
    Devuelve el contenido de data/faqs.json.
    """

    if not FAQS_PATH.exists():
        return []

    try:

        contenido = FAQS_PATH.read_text(
            encoding="utf-8"
        )

        datos = json.loads(
            contenido
        )

        if not isinstance(
            datos,
            list,
        ):
            return []

        return datos

    except (
        OSError,
        json.JSONDecodeError,
    ):

        return []


# =========================================================
# INDEXACIÓN
# =========================================================


@app.post(
    "/api/index"
)
def reindex():
    """
    Reprocesa los PDF disponibles y reconstruye
    la base vectorial.

    Flujo:

    PDFs
      ↓
    load_pdfs()
      ↓
    split_documents()
      ↓
    embeddings
      ↓
    ChromaDB
      ↓
    guardar estado de documentos
    """

    global _vector_store

    try:

        documents = load_pdfs()

        chunks = split_documents(
            documents
        )


        # -------------------------------------------------
        # SEGURIDAD
        #
        # Evita reconstruir la base con cero fragmentos.
        # -------------------------------------------------

        if not chunks:

            raise HTTPException(
                status_code=400,
                detail=(
                    "No se encontraron fragmentos "
                    "válidos para indexar."
                ),
            )


        build_vectorstore(
            chunks,
            _lazy_embeddings(),
            rebuild=True,
        )


        # Forzar que la base vectorial
        # se vuelva a cargar después.
        _vector_store = None


        # -------------------------------------------------
        # GUARDAR ESTADO
        #
        # Solo se ejecuta después de que ChromaDB
        # fue reconstruido correctamente.
        # -------------------------------------------------

        guardar_estado_actual()


        return {
            "ok": True,
            "chunks_indexados": len(
                chunks
            ),
        }


    except HTTPException:

        raise


    except FileNotFoundError as error:

        raise HTTPException(
            status_code=400,
            detail=str(
                error
            ),
        )


    except Exception:

        logger.exception(
            "Error durante la indexación manual."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "No fue posible reconstruir "
                "la base de conocimiento."
            ),
        )


# =========================================================
# CHAT
# =========================================================


@app.post(
    "/api/chat"
)
def chat(
    req: ChatRequest,
):
    """
    Endpoint principal del asistente.

    Flujo:

    Pregunta
       ↓
    Validación
       ↓
    ¿Saludo?
       ├── Sí → respuesta directa
       │
       └── No
            ↓
         historial
            ↓
         rag_pipeline()
            ↓
       contextualización
            ↓
        Tool Calling
           /     \
        Tool      RAG
         ↓         ↓
     respuesta  ChromaDB
                  ↓
               respuesta
    """

    # =====================================================
    # 1. VALIDAR PREGUNTA
    # =====================================================

    pregunta = str(
        req.question
    ).strip()

    if not pregunta:

        raise HTTPException(
            status_code=400,
            detail=(
                "La pregunta no puede estar vacía."
            ),
        )


    # =====================================================
    # 2. RESPONDER SALUDOS
    # =====================================================

    if es_saludo(
        pregunta
    ):

        return {
            "respuesta": "Hola 👋 ¿En qué puedo ayudarte?",
            "fuentes": [],
            "tipo": "saludo",
            "tools_usadas": [],
            "iteraciones_tools": 0,
        }


    # =====================================================
    # 3. PREPARAR HISTORIAL
    # =====================================================

    historial = [
        mensaje.model_dump()
        for mensaje in req.historial
    ]


    # =====================================================
    # 4. VERIFICAR BASE VECTORIAL
    # =====================================================
    #
    # Mantenemos esta validación porque
    # el pipeline principal utiliza ChromaDB como fallback.
    #
    # Las tools operativas también pueden consultar
    # información del sistema, pero la base documental
    # sigue siendo el núcleo del asistente.
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
    # 5. EJECUTAR PIPELINE
    # =====================================================

    try:

        resultado = rag_pipeline(
            CombinedVectorStore(
                _lazy_vectorstore(),
                extra_docs,
            ),
            _lazy_llm(),
            _lazy_prompt_template(),
            pregunta,
            k=req.k,
            historial=historial,
        )

    except Exception:

        logger.exception(
            "Error procesando consulta del chat."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "No fue posible procesar "
                "la consulta en este momento."
            ),
        )


    # =====================================================
    # 6. PREPARAR FUENTES
    # =====================================================
    #
    # Cuando responde una tool:
    #
    # fragmentos = []
    #
    # Por lo tanto no se mostrarán fuentes documentales.
    #
    # Cuando responde el RAG:
    #
    # fragmentos contiene los documentos recuperados.
    # =====================================================

    fuentes = []

    for documento in resultado.get(
        "fragmentos",
        [],
    ):

        metadata = getattr(
            documento,
            "metadata",
            {},
        ) or {}

        page_content = str(
            getattr(
                documento,
                "page_content",
                "",
            )
        )

        fuentes.append(
            {
                "archivo": Path(
                    metadata.get(
                        "source",
                        "?",
                    )
                ).name,

                "pagina": metadata.get(
                    "page",
                    "?",
                ),

                "fragmento": page_content[
                    :300
                ],
            }
        )


    # =====================================================
    # 7. RESPUESTA FINAL
    # =====================================================

    return {
        "respuesta": resultado.get(
            "respuesta",
            (
                "No fue posible generar "
                "una respuesta."
            ),
        ),

        "fuentes": fuentes,

        "tipo": resultado.get(
            "tipo",
            "rag",
        ),

        "tools_usadas": resultado.get(
            "tools_usadas",
            [],
        ),

        "iteraciones_tools": resultado.get(
            "iteraciones_tools",
            0,
        ),
    }


# =========================================================
# MANUALES EXTRA (subida manual de PDF)
# =========================================================


def _exigir_clave(token: str | None) -> None:
    """Si UPLOAD_TOKEN está definido, exige la clave."""

    if not token_valido(token):
        raise HTTPException(
            status_code=401,
            detail="Se requiere la clave para modificar los manuales extra.",
        )


@app.get("/api/extra-docs")
def listar_extra_docs():
    """Manuales extra subidos manualmente."""

    return {
        "documentos": extra_docs.listar(),
        "max_mb": MAX_UPLOAD_MB,
    }


@app.post("/api/extra-docs")
async def subir_extra_doc(
    request: Request,
    nombre: str,
    x_upload_token: str | None = Header(default=None),
):
    """
    Recibe un PDF (cuerpo de la petición = bytes del archivo) y lo agrega
    como referencia extra, sin tocar la indexación automática.
    """

    _exigir_clave(x_upload_token)

    limite = (
        f"El PDF supera el máximo permitido ({MAX_UPLOAD_MB} MB)."
    )

    declarado = request.headers.get("content-length", "")

    if declarado.isdigit() and int(declarado) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=limite)

    recibido = bytearray()

    async for bloque in request.stream():
        recibido.extend(bloque)

        if len(recibido) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=limite)

    if not recibido:
        raise HTTPException(
            status_code=400,
            detail="No se recibió ningún archivo.",
        )

    try:
        resultado = await run_in_threadpool(
            extra_docs.agregar_pdf,
            nombre,
            bytes(recibido),
        )

    except ExtraDocsError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception:
        logger.exception("Error indexando un manual extra.")

        raise HTTPException(
            status_code=500,
            detail="No fue posible indexar el documento.",
        )

    return {"ok": True, **resultado}


@app.delete("/api/extra-docs/{nombre}")
def eliminar_extra_doc(
    nombre: str,
    x_upload_token: str | None = Header(default=None),
):
    """Quita un manual extra (no afecta los PDF de data/pdfs)."""

    _exigir_clave(x_upload_token)

    quitados = extra_docs.eliminar(nombre)

    if quitados is None:
        raise HTTPException(
            status_code=404,
            detail="No existe ese manual extra.",
        )

    return {"ok": True, "fragmentos_quitados": quitados}


# =========================================================
# INTERFAZ ESTÁTICA
# =========================================================

# IMPORTANTE:
#
# StaticFiles debe mantenerse AL FINAL.
#
# Si se monta antes de /api/* podría interceptar
# las rutas del backend.

app.mount(
    "/",
    StaticFiles(
        directory=BASE_DIR / "static",
        html=True,
    ),
    name="static",
)
