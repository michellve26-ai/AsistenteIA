"""
Tools específicas para AstreaS.

Estas herramientas complementan el sistema RAG y permiten
ejecutar funciones relacionadas con:

- estado de la base de conocimiento,
- manuales disponibles,
- información general de AstreaS,
- categorías de preguntas frecuentes,
- búsqueda de preguntas frecuentes.

Las consultas documentales siguen siendo atendidas
por el pipeline RAG con ChromaDB.
"""

import json
from pathlib import Path

from langchain_core.tools import tool

from src import config
from src.vectorstore import index_exists


# =========================================================
# RUTAS DEL PROYECTO
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
FAQS_PATH = BASE_DIR / "data" / "faqs.json"
PDF_DIR = BASE_DIR / "data" / "pdfs"


# =========================================================
# TOOL 1 - ESTADO DE LA BASE DE CONOCIMIENTO
# =========================================================

@tool("estado_base_conocimiento")
def estado_base_conocimiento() -> str:
    """
    Consulta si AstreaS tiene una base de conocimiento indexada.

    Usa esta herramienta cuando el usuario pregunte si los
    manuales están cargados, si la base está disponible o si
    AstreaS tiene documentos listos para consultar.

    Returns:
        Mensaje con el estado actual de la base de conocimiento.
    """

    if index_exists():
        return (
            "La base de conocimiento de AstreaS está disponible "
            "y existen documentos indexados para realizar consultas."
        )

    return (
        "La base de conocimiento de AstreaS no está indexada "
        "actualmente."
    )


# =========================================================
# TOOL 2 - LISTAR MANUALES DISPONIBLES
# =========================================================

@tool("listar_manuales")
def listar_manuales() -> str:
    """
    Lista los archivos PDF disponibles para AstreaS.

    Usa esta herramienta cuando el usuario pregunte qué manuales,
    documentos o archivos están disponibles para consulta.

    Returns:
        Lista de nombres de los archivos PDF disponibles.
    """

    if not PDF_DIR.exists():
        return "No se encontró la carpeta de manuales."

    archivos = sorted(PDF_DIR.glob("*.pdf"))

    if not archivos:
        return "No hay manuales PDF disponibles actualmente."

    nombres = [archivo.name for archivo in archivos]

    return (
        "Manuales disponibles en AstreaS:\n- "
        + "\n- ".join(nombres)
    )


# =========================================================
# TOOL 3 - INFORMACIÓN GENERAL DE ASTREAS
# =========================================================

@tool("informacion_astreas")
def informacion_astreas() -> str:
    """
    Devuelve información general sobre AstreaS.

    Usa esta herramienta cuando el usuario pregunte qué es AstreaS,
    para qué sirve, cómo funciona o qué tipo de asistente es.

    Returns:
        Descripción general del asistente.
    """

    return (
        f"{config.ASSISTANT_NAME} es un asistente virtual de soporte técnico. "
        "Utiliza una arquitectura RAG para recuperar información "
        "desde documentos y manuales previamente indexados en una "
        "base vectorial. "
        "Además, puede utilizar herramientas específicas para "
        "consultar información del sistema y apoyar determinadas "
        "acciones."
    )


# =========================================================
# TOOL 4 - LISTAR CATEGORÍAS DE FAQ
# =========================================================

@tool("listar_categorias_faq")
def listar_categorias_faq() -> str:
    """
    Lista las categorías disponibles en las preguntas frecuentes.

    Usa esta herramienta cuando el usuario quiera saber qué tipos
    de ayuda o categorías de soluciones rápidas existen.

    Returns:
        Lista de categorías disponibles en el archivo de FAQ.
    """

    if not FAQS_PATH.exists():
        return "No se encontró el archivo de preguntas frecuentes."

    try:
        faqs = json.loads(
            FAQS_PATH.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError):
        return "No fue posible leer las preguntas frecuentes."

    categorias = sorted(
        {
            faq.get("category")
            for faq in faqs
            if faq.get("category")
        }
    )

    if not categorias:
        return "No hay categorías de preguntas frecuentes disponibles."

    return (
        "Categorías disponibles:\n- "
        + "\n- ".join(categorias)
    )


# =========================================================
# TOOL 5 - BUSCAR FAQ
# =========================================================

@tool("buscar_faq")
def buscar_faq(consulta: str) -> str:
    """
    Busca coincidencias dentro de las preguntas frecuentes de AstreaS.

    Usa esta herramienta cuando el usuario pregunte por un caso común,
    una solución rápida o un tema que pueda estar registrado en las FAQ.

    Args:
        consulta: Texto o tema que desea buscar el usuario.

    Returns:
        Coincidencias encontradas dentro del archivo de preguntas frecuentes.
    """

    if not consulta or not consulta.strip():
        return "Debes indicar un tema para buscar en las preguntas frecuentes."

    if not FAQS_PATH.exists():
        return "No se encontró el archivo de preguntas frecuentes."

    try:
        faqs = json.loads(
            FAQS_PATH.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError):
        return "No fue posible leer las preguntas frecuentes."

    consulta_normalizada = consulta.lower().strip()

    resultados = []

    for faq in faqs:
        titulo = str(
            faq.get("title", "")
        ).lower()

        resumen = str(
            faq.get("summary", "")
        ).lower()

        categoria = str(
            faq.get("category", "")
        ).lower()

        if (
            consulta_normalizada in titulo
            or consulta_normalizada in resumen
            or consulta_normalizada in categoria
        ):
            resultados.append(faq)

    if not resultados:
        return (
            f"No se encontraron preguntas frecuentes relacionadas "
            f"con '{consulta}'."
        )

    respuestas = []

    for faq in resultados[:5]:

        titulo = faq.get(
            "title",
            "Sin título",
        )

        categoria = faq.get(
            "category",
            "Sin categoría",
        )

        resumen = faq.get(
            "summary",
            "Sin resumen",
        )

        respuestas.append(
            f"Título: {titulo}\n"
            f"Categoría: {categoria}\n"
            f"Resumen: {resumen}"
        )

    return "\n\n".join(respuestas)


# =========================================================
# TOOL 6 - CONTAR MANUALES DISPONIBLES
# =========================================================

@tool("contar_manuales")
def contar_manuales() -> str:
    """
    Cuenta cuántos archivos PDF tiene disponibles AstreaS.

    Usa esta herramienta cuando el usuario pregunte cuántos
    manuales o documentos están disponibles.

    Returns:
        Cantidad de archivos PDF encontrados.
    """

    if not PDF_DIR.exists():
        return "Actualmente no se encontró la carpeta de manuales."

    archivos = list(PDF_DIR.glob("*.pdf"))

    cantidad = len(archivos)

    if cantidad == 0:
        return "Actualmente no hay manuales PDF disponibles."

    if cantidad == 1:
        return "AstreaS tiene 1 manual PDF disponible."

    return f"AstreaS tiene {cantidad} manuales PDF disponibles."


# =========================================================
# REGISTRO DE TOOLS DISPONIBLES
# =========================================================

TOOLS = [
    estado_base_conocimiento,
    listar_manuales,
    informacion_astreas,
    listar_categorias_faq,
    buscar_faq,
    contar_manuales,
]
