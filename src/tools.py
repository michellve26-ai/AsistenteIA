"""
Herramientas de AstreaS.

Este módulo define las tools que el modelo de lenguaje puede
invocar cuando una consulta necesita ejecutar una operación
sobre recursos internos del asistente.

Las tools complementan el sistema RAG:

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
# RUTAS DEL PROYECTO
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# config.PDF_DIR ya está definido en src/config.py
# con el valor por defecto "data/pdfs".
PDF_DIR = BASE_DIR / config.PDF_DIR

FAQS_PATH = BASE_DIR / "data" / "faqs.json"


# =========================================================
# FUNCIONES INTERNAS
# =========================================================

def _cargar_faqs() -> list[dict[str, Any]]:
    """
    Lee y valida el archivo de preguntas frecuentes.

    Esta función es interna.
    No se expone directamente como tool al modelo.
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

        return [
            item
            for item in datos
            if isinstance(item, dict)
        ]

    except (OSError, json.JSONDecodeError):
        return []


def _normalizar(texto: str) -> str:
    """
    Normaliza texto para realizar búsquedas simples.
    """

    return str(texto).lower().strip()


# =========================================================
# TOOL 1
# CONSULTAR ESTADO DE LA BASE DE CONOCIMIENTO
# =========================================================

@tool("consultar_estado_base")
def consultar_estado_base() -> str:
    """
    Consulta si AstreaS tiene disponible una base de conocimiento
    vectorial previamente indexada.

    Usa esta herramienta cuando el usuario pregunte si existen
    documentos indexados, si la base de conocimiento está disponible
    o si el sistema está listo para realizar búsquedas documentales.

    Returns:
        Estado actual de la base de conocimiento.
    """

    if index_exists():
        return (
            "La base de conocimiento está disponible y existe "
            "un índice vectorial listo para realizar consultas."
        )

    return (
        "Actualmente no existe un índice vectorial disponible "
        "en la base de conocimiento."
    )


# =========================================================
# TOOL 2
# LISTAR DOCUMENTOS DISPONIBLES
# =========================================================

@tool("listar_documentos")
def listar_documentos() -> str:
    """
    Consulta los documentos PDF disponibles actualmente
    en la carpeta documental de AstreaS.

    Usa esta herramienta cuando el usuario pregunte qué documentos,
    manuales o archivos PDF están disponibles para consulta.

    Returns:
        Lista de documentos PDF disponibles.
    """

    if not PDF_DIR.exists():
        return (
            "No se encontró el directorio configurado "
            "para almacenar los documentos."
        )

    if not PDF_DIR.is_dir():
        return (
            "La ruta configurada para los documentos "
            "no corresponde a un directorio válido."
        )

    try:
        documentos = sorted(
            archivo
            for archivo in PDF_DIR.glob("*.pdf")
            if archivo.is_file()
        )

    except OSError:
        return (
            "No fue posible consultar actualmente "
            "el directorio de documentos."
        )

    if not documentos:
        return (
            "Actualmente no hay documentos PDF "
            "disponibles para consulta."
        )

    nombres = [
        documento.name
        for documento in documentos
    ]

    listado = "\n".join(
        f"- {nombre}"
        for nombre in nombres
    )

    return (
        f"Se encontraron {len(nombres)} "
        f"documento(s) PDF disponible(s):\n"
        f"{listado}"
    )


# =========================================================
# TOOL 3
# BUSCAR SOLUCIONES RÁPIDAS
# =========================================================

@tool("buscar_solucion_rapida")
def buscar_solucion_rapida(
    consulta: str,
) -> str:
    """
    Busca casos relacionados dentro de las soluciones rápidas
    registradas en AstreaS.

    Usa esta herramienta cuando el usuario pregunte por un caso
    frecuente, una solución rápida o quiera saber si existe una
    FAQ relacionada con determinado problema.

    Esta herramienta consulta data/faqs.json y no sustituye
    la recuperación semántica del sistema RAG.

    Args:
        consulta:
            Tema, problema o palabras que desea buscar el usuario.

    Returns:
        Hasta cinco soluciones rápidas relacionadas.
    """

    consulta_normalizada = _normalizar(
        consulta
    )

    if not consulta_normalizada:
        return (
            "No se indicó ningún término "
            "para realizar la búsqueda."
        )

    faqs = _cargar_faqs()

    if not faqs:
        return (
            "Actualmente no hay soluciones rápidas "
            "disponibles para consultar."
        )

    palabras_consulta = set(
        consulta_normalizada.split()
    )

    coincidencias: list[
        tuple[int, dict[str, Any]]
    ] = []

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

        # Coincidencia directa de toda la consulta.
        if consulta_normalizada in contenido:
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
            f"relacionadas con '{consulta_normalizada}'."
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


# Diccionario para encontrar rápidamente
# una herramienta a partir del nombre
# solicitado por el modelo.
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
    Permite que el modelo determine si necesita ejecutar
    alguna de las herramientas registradas en AstreaS.

    Flujo:

    1. Las tools se enlazan al LLM mediante bind_tools().
    2. Se envía la pregunta del usuario al modelo.
    3. El modelo decide si necesita ejecutar una tool.
    4. Python ejecuta la herramienta solicitada.
    5. El resultado se devuelve mediante ToolMessage.
    6. El modelo utiliza ese resultado para generar
       la respuesta final.

    Si ninguna herramienta es necesaria, devuelve None.
    De esta forma, la consulta puede continuar normalmente
    por el pipeline RAG.
    """

    pregunta = str(
        pregunta
    ).strip()

    if not pregunta:
        return None


    # -----------------------------------------------------
    # 1. Enlazar herramientas al modelo
    # -----------------------------------------------------

    llm_con_tools = llm.bind_tools(
        TOOLS
    )


    # -----------------------------------------------------
    # 2. Crear mensaje inicial
    # -----------------------------------------------------

    mensajes = [
        HumanMessage(
            content=pregunta
        )
    ]


    # -----------------------------------------------------
    # 3. El modelo decide si necesita una herramienta
    # -----------------------------------------------------

    respuesta_modelo = (
        llm_con_tools.invoke(
            mensajes
        )
    )


    # Acceso defensivo a tool_calls.
    tool_calls = getattr(
        respuesta_modelo,
        "tool_calls",
        [],
    ) or []


    # -----------------------------------------------------
    # 4. Sin tool -> continuar con RAG
    # -----------------------------------------------------

    if not tool_calls:
        return None


    # Se agrega al historial la respuesta del modelo
    # donde solicitó las tools.
    mensajes.append(
        respuesta_modelo
    )


    herramientas_usadas = []


    # -----------------------------------------------------
    # 5. Ejecutar cada herramienta solicitada
    # -----------------------------------------------------

    for tool_call in tool_calls:

        nombre = tool_call.get(
            "name"
        )

        argumentos = tool_call.get(
            "args",
            {},
        )

        tool_call_id = tool_call.get(
            "id"
        )


        # Si el modelo genera una llamada incompleta,
        # no se intenta ejecutar.
        if not nombre or not tool_call_id:
            continue


        herramienta = TOOLS_BY_NAME.get(
            nombre
        )


        # -------------------------------------------------
        # Tool inexistente
        # -------------------------------------------------

        if herramienta is None:

            resultado = (
                f"La herramienta '{nombre}' "
                "no está disponible."
            )


        # -------------------------------------------------
        # Tool existente
        # -------------------------------------------------

        else:

            try:

                resultado = herramienta.invoke(
                    argumentos
                )

                # Solo se registra como utilizada
                # cuando realmente se ejecutó.
                herramientas_usadas.append(
                    nombre
                )

            except Exception:

                # No se envían detalles internos
                # de excepciones al modelo.
                resultado = (
                    f"No fue posible ejecutar "
                    f"la herramienta '{nombre}'."
                )


        # -------------------------------------------------
        # 6. Enviar resultado de la tool al modelo
        # -------------------------------------------------

        mensajes.append(
            ToolMessage(
                content=str(resultado),
                tool_call_id=tool_call_id,
            )
        )


    # -----------------------------------------------------
    # 7. El LLM redacta la respuesta final
    # -----------------------------------------------------

    respuesta_final = (
        llm_con_tools.invoke(
            mensajes
        )
    )


    contenido_final = str(
        respuesta_final.content
    ).strip()


    if not contenido_final:
        contenido_final = (
            "La herramienta fue ejecutada, "
            "pero no fue posible generar una respuesta final."
        )


    return {
        "respuesta": contenido_final,
        "tools_usadas": herramientas_usadas,
    }
