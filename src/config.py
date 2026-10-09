
"""
Configuración central del asistente RAG de soporte técnico.

Contiene los parámetros utilizados para:
- Generación de respuestas con Groq.
- Embeddings locales con FastEmbed.
- Procesamiento de documentos.
- Almacenamiento vectorial con ChromaDB.
- Recuperación de información.
- Personalización del asistente.
"""

import os

from dotenv import load_dotenv


load_dotenv()


# ---------------------------------------------------------
# MODELO DE LENGUAJE - GROQ
# ---------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)

LLM_TEMPERATURE = 0.0


# ---------------------------------------------------------
# EMBEDDINGS - FASTEMBED
# ---------------------------------------------------------

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# DOCUMENTOS
# ---------------------------------------------------------

PDF_DIR = os.getenv(
    "PDF_DIR",
    "data/pdfs",
)


# ---------------------------------------------------------
# FRAGMENTACIÓN DE DOCUMENTOS
# ---------------------------------------------------------

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ---------------------------------------------------------
# BASE VECTORIAL - CHROMADB
# ---------------------------------------------------------

PERSIST_DIR = os.getenv(
    "PERSIST_DIR",
    "chroma_db_minilm_l6",
)

COLLECTION_NAME = "manuales_soporte"


# ---------------------------------------------------------
# RECUPERACIÓN DE INFORMACIÓN
# ---------------------------------------------------------

TOP_K = 5


# ---------------------------------------------------------
# PERSONALIZACIÓN DEL ASISTENTE
# ---------------------------------------------------------

ASSISTANT_NAME = "Asistente de Soporte Técnico"

EMPRESA = "OficinaPro.co"


# ---------------------------------------------------------
# VALIDACIÓN DE CONFIGURACIÓN
# ---------------------------------------------------------

if not GROQ_API_KEY:
    print(
        "[ADVERTENCIA] GROQ_API_KEY no está definida. "
        "Configura la variable de entorno en GitHub Actions, "
        "Render o en el archivo .env.",
        flush=True,
    )
