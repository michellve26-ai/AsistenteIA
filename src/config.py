"""
Configuración central del asistente RAG de soporte técnico.
Todos los parámetros ajustables del sistema viven aquí.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Groq ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = 0.0

# --- Embeddings (locales, corren en CPU) ---
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# --- Documentos ---
PDF_DIR = os.getenv("PDF_DIR", "data/pdfs")

# --- Chunking ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Vector store (ChromaDB) ---
PERSIST_DIR = os.getenv("PERSIST_DIR", "chroma_db")
COLLECTION_NAME = "manuales_soporte"

# --- Recuperación ---
TOP_K = 5

# --- Personalización de la interfaz ---
ASSISTANT_NAME = "Asistente de Soporte Técnico"
EMPRESA = "Finanzas Corp"  # cámbialo por el nombre real de la empresa

if not GROQ_API_KEY:
    print("[!] GROQ_API_KEY no está definida. Crea un archivo .env con tu clave (ver .env.example).")
