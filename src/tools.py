"""
Herramientas de AstreaS.

Este módulo define las tools que el modelo de lenguaje puede
invocar cuando una consulta necesita ejecutar una operación
sobre recursos internos del asistente.

Las tools complementan el RAG.

- El RAG recupera conocimiento semántico desde ChromaDB.
- Las tools ejecutan funciones concretas y determinísticas.

Este archivo no modifica rutas FastAPI ni sustituye
el pipeline RAG existente.
"""

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from src import config
from src.vectorstore import index_exists


# =========================================================
# RUTAS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_DIR = BASE_DIR / config.PDF_DIR
FAQS_PATH = BASE_DIR / "data" / "faqs.json"


# =========================================================
# FUNCIONES INTERNAS
# =========================================================

def _cargar_faqs() -> list[dict[str, Any]]:
    """
    Lee y valida el archivo de preguntas frecuentes.

    Esta función es interna y no está expuesta directamente
    como tool al modelo.
    """

    if not FAQS_PATH.exists():
        return []

    try:
        contenido = FAQS_PATH.read_text(
            encoding="utf-8"
        )

        datos = json.loads(contenido)

        if not isinstance(datos, list):
            return []

        return datos

    except (OSError, json.JSONDecodeError):
        return []


def _normalizar(texto: str) -> str:
    """
    Normaliza texto para búsquedas simples.
    """

    return str(texto).lower().strip()


# =========================================================
# TOOL 1
# ESTADO DE LA BASE DE CONOCIMIENTO
# =========================================================

@tool("consultar_estado_base")
def consultar_estado_base() -> str:
    """
    Consulta el estado actual de la base de conocimiento de AstreaS.

    Utiliza esta herramienta cuando el usuario pregunte si AstreaS
    tiene documentos indexados, si la base de conocimiento está
    disponible o si el sistema está listo para realizar búsquedas
    documentales.

    Returns:
        Estado actual de la base de conocimiento.
    """

    disponible = index_exists()

    if disponible:
        return (
            "La base de conocimiento está disponible "
            "y existe un índice vectorial listo para consultas."
        )

    return (
        "La base de conocimiento no tiene actualmente "
        "un índice vectorial disponible."
    )


# =========================================================
# TOOL 2
# LISTAR DOCUMENTOS DISPONIBLES
# =========================================================

@tool("listar_documentos")
def listar_documentos() -> str:
    """
    Consulta los documentos PDF que están físicamente disponibles
    en la carpeta documental de AstreaS.

    Utiliza esta herramienta cuando el usuario pregunte qué
    documentos, archivos o manuales tiene disponibles AstreaS.

    Returns:
        Lista de documentos PDF disponibles.
    """

    if not PDF_DIR.exists():
        return (
            "No se encontró el directorio configurado "
            "para los documentos."
        )

    documentos = sorted(
        archivo
        for archivo in PDF_DIR.glob("*.pdf")
        if archivo.is_file()
    )

    if not documentos:
        return (
            "Actualmente no hay documentos PDF "
            "disponibles."
        )

    nombres = [
        documento.name
        for documento in documentos
    ]

    resultado = (
        f"Se encontraron {len(nombres)} "
        f"documento(s) PDF disponible(s):\n"
    )

    resultado += "\n".join(
        f"- {nombre}"
        for nombre in nombres
    )

    return resultado


# =========================================================
# TOOL 3
# BUSCAR SOLUCIÓN RÁPIDA
# =========================================================

@tool("buscar_solucion_rapida")
def buscar_solucion_rapida(consulta: str) -> str:
    """
    Busca un caso dentro de las soluciones rápidas registradas
    en AstreaS.

    Utiliza esta herramienta cuando el usuario pregunte por una
    solución rápida, un caso frecuente o quiera saber si existe
    una FAQ relacionada con determinado problema.

    Esta herramienta no sustituye la búsqueda documental RAG.

    Args:
        consulta: Tema, problema o palabras que desea buscar
        el usuario.

    Returns:
        Hasta cinco soluciones rápidas relacionadas.
    """

    consulta = _normalizar(consulta)

    if not consulta:
        return (
            "No se indicó ningún término "
            "para realizar la búsqueda."
        )

    faqs = _cargar_faqs()

    if not faqs:
        return (
            "No hay soluciones rápidas disponibles "
            "para consultar."
        )

    palabras_consulta = set(
        consulta.split()
    )

    coincidencias = []

    for faq in faqs:

        titulo = _normalizar(
            faq.get("title", "")
        )

        resumen = _normalizar(
            faq.get("summary", "")
        )

        categoria = _normalizar(
            faq.get("category", "")
        )

        contenido = (
            f"{titulo} "
            f"{resumen} "
            f"{categoria}"
        )

        # Coincidencia exacta
        if consulta in contenido:
            puntuacion = 100

        else:
            palabras_contenido = set(
                contenido.split()
            )

            puntuacion = len(
                palabras_consulta
                & palabras_contenido
            )

        if puntuacion > 0:
            coincidencias.append(
                (
                    puntuacion,
                    faq,
                )
            )


    if not coincidencias:
        return (
            "No se encontraron soluciones rápidas "
            f"relacionadas con '{consulta}'."
        )


    coincidencias.sort(
        key=lambda item: item[0],
        reverse=True,
    )


    resultados = []

    for _, faq in coincidencias[:5]:

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

        resultados.append(
            f"Título: {titulo}\n"
            f"Categoría: {categoria}\n"
            f"Resumen: {resumen}"
        )


    return "\n\n---\n\n".join(
        resultados
    )


# =========================================================
# REGISTRO CENTRAL DE TOOLS
# =========================================================

TOOLS = [
    consultar_estado_base,
    listar_documentos,
    buscar_solucion_rapida,
]


TOOLS_BY_NAME = {
    herramienta.name: herramienta
    for herramienta in TOOLS
}


# =========================================================
# EJECUTOR DE TOOLS
# =========================================================

def ejecutar_tools(
    llm,
    pregunta: str,
) -> dict | None:
    """
    Permite que el LLM determine si necesita utilizar
    alguna herramienta registrada.

    Flujo:

    1. Se enlazan las tools al LLM mediante bind_tools().
    2. Se envía la pregunta.
    3. El modelo puede solicitar una o varias tools.
    4. Python ejecuta cada tool solicitada.
    5. Los resultados se devuelven mediante ToolMessage.
    6. El LLM genera la respuesta final.

    Si el modelo decide que no necesita ninguna herramienta,
    devuelve None para que la consulta continúe por el RAG.
    """

    llm_con_tools = llm.bind_tools(
        TOOLS
    )

    mensajes = [
        HumanMessage(
            content=pregunta
        )
    ]


    # -----------------------------------------------------
    # 1. El modelo decide si necesita una tool
    # -----------------------------------------------------

    respuesta_modelo = (
        llm_con_tools.invoke(
            mensajes
        )
    )


    # -----------------------------------------------------
    # 2. Si no pidió tools, continuar con RAG
    # -----------------------------------------------------

    if not respuesta_modelo.tool_calls:
        return None


    # La solicitud de tool debe conservarse
    # dentro del historial.
    mensajes.append(
        respuesta_modelo
    )


    herramientas_usadas = []


    # -----------------------------------------------------
    # 3. Ejecutar las tools solicitadas
    # -----------------------------------------------------

    for tool_call in respuesta_modelo.tool_calls:

        nombre = tool_call.get(
            "name"
        )

        argumentos = tool_call.get(
            "args",
            {},
        )

        herramienta = TOOLS_BY_NAME.get(
            nombre
        )


        if herramienta is None:

            resultado = (
                f"La herramienta '{nombre}' "
                "no está registrada."
            )

        else:

            try:

                resultado = herramienta.invoke(
                    argumentos
                )

            except Exception as error:

                resultado = (
                    "No fue posible ejecutar "
                    f"la herramienta '{nombre}': "
                    f"{error}"
                )


        herramientas_usadas.append(
            nombre
        )


        # -------------------------------------------------
        # 4. Devolver resultado al modelo
        # -------------------------------------------------

        mensajes.append(
            ToolMessage(
                content=str(resultado),
                tool_call_id=tool_call["id"],
            )
        )


    # -----------------------------------------------------
    # 5. El LLM redacta la respuesta final
    # -----------------------------------------------------

    respuesta_final = (
        llm_con_tools.invoke(
            mensajes
        )
    )


    return {
        "respuesta": respuesta_final.content,
        "tools_usadas": herramientas_usadas,
    }
