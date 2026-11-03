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

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from src import config
from src.vectorstore import index_exists


# =========================================================
# RUTAS DEL PROYECTO
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# config.PDF_DIR está definido en src/config.py.
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

        return [
            item
            for item in datos
            if isinstance(item, dict)
        ]

    except (OSError, json.JSONDecodeError):
        return []


def _normalizar(texto: str) -> str:
    """
    Normaliza texto para búsquedas simples.
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

    Utiliza esta herramienta únicamente cuando el usuario pregunte
    por el estado de la base de conocimiento, la indexación,
    los documentos indexados o la disponibilidad del índice vectorial.

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

    Utiliza esta herramienta cuando el usuario pregunte
    qué documentos, manuales o archivos PDF tiene disponibles
    AstreaS para consulta.

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

    Utiliza esta herramienta cuando el usuario pregunte
    expresamente por una solución rápida, una pregunta frecuente,
    una FAQ o un caso frecuente registrado.

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

        # Coincidencia directa de la consulta completa.
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


# Permite localizar rápidamente una tool
# a partir del nombre solicitado por el modelo.
TOOLS_BY_NAME = {
    herramienta.name: herramienta
    for herramienta in TOOLS
}


# =========================================================
# CONFIGURACIÓN DEL EJECUTOR
# =========================================================

# Número máximo de rondas de Tool Calling.
# Evita ciclos infinitos si el modelo continúa
# solicitando herramientas.
MAX_TOOL_ITERATIONS = 5


TOOL_SYSTEM_PROMPT = """
Eres el orquestador de herramientas de AstreaS,
un asistente especializado en soporte técnico.

Tu única responsabilidad en esta etapa es decidir si la consulta
del usuario necesita utilizar alguna de las herramientas disponibles.

Herramientas disponibles:

1. consultar_estado_base
   Úsala únicamente cuando el usuario pregunte por:
   - estado de la base de conocimiento,
   - indexación,
   - existencia del índice vectorial,
   - disponibilidad de documentos indexados.

2. listar_documentos
   Úsala únicamente cuando el usuario pregunte:
   - qué documentos existen,
   - qué manuales están disponibles,
   - qué archivos PDF tiene disponibles AstreaS.

3. buscar_solucion_rapida
   Úsala únicamente cuando el usuario pregunte expresamente por:
   - soluciones rápidas,
   - preguntas frecuentes,
   - FAQs,
   - casos frecuentes registrados.

Reglas obligatorias:

- NO utilices estas herramientas para responder procedimientos
  técnicos contenidos en los manuales.

- NO utilices estas herramientas para preguntas cuyo contenido
  deba recuperarse mediante el sistema RAG.

- Si ninguna herramienta es apropiada, no intentes responder
  la pregunta en esta etapa.

- Si ninguna herramienta es apropiada, simplemente no solicites
  ninguna tool. La pregunta continuará posteriormente hacia el RAG.

- No utilices conocimiento general para reemplazar al RAG.

- Nunca inventes herramientas.

- Nunca inventes resultados de herramientas.

- Puedes solicitar varias herramientas en la misma ronda cuando
  la consulta realmente necesite información de varias de ellas.

- Después de recibir los resultados de una herramienta puedes
  solicitar otra herramienta únicamente si es necesaria para
  completar la misma consulta.

- No repitas innecesariamente una herramienta que ya fue ejecutada
  con los mismos argumentos.

- Cuando ya tengas toda la información necesaria proveniente
  de las tools, genera una respuesta clara y directa para el usuario.
""".strip()


# =========================================================
# EJECUTOR DE TOOLS
# =========================================================

def ejecutar_tools(
    llm,
    pregunta: str,
) -> dict | None:
    """
    Ejecuta el flujo de Tool Calling de AstreaS.

    Flujo:

    1. Enlaza las tools al modelo mediante bind_tools().
    2. Envía la pregunta al modelo.
    3. El modelo decide si necesita una o varias tools.
    4. Si no solicita ninguna, devuelve None y la consulta
       continúa por el pipeline RAG.
    5. Si solicita tools, Python ejecuta las funciones.
    6. Los resultados se devuelven mediante ToolMessage.
    7. El modelo puede solicitar nuevas tools si todavía
       son necesarias.
    8. Si buscar_solucion_rapida no encuentra coincidencias,
       devuelve None para continuar automáticamente por RAG.
    9. Cuando deja de solicitar tools, genera la respuesta final.
    10. El ciclo está limitado para evitar ejecuciones infinitas.

    Returns:
        None:
            Cuando ninguna herramienta aplica o cuando una búsqueda
            de solución rápida no encuentra coincidencias y la consulta
            debe continuar por el RAG.

        dict:
            Cuando una o varias herramientas fueron utilizadas
            y resolvieron la consulta.
    """

    pregunta = str(
        pregunta
    ).strip()

    if not pregunta:
        return None


    # =====================================================
    # 1. ENLAZAR TOOLS AL MODELO
    # =====================================================

    llm_con_tools = llm.bind_tools(
        TOOLS
    )


    # =====================================================
    # 2. HISTORIAL INTERNO
    # =====================================================

    mensajes = [
        SystemMessage(
            content=TOOL_SYSTEM_PROMPT
        ),
        HumanMessage(
            content=pregunta
        ),
    ]


    herramientas_usadas: list[str] = []

    # Para evitar repetir exactamente la misma llamada.
    llamadas_ejecutadas: set[str] = set()

    hubo_tool = False

    rondas_ejecutadas = 0


    # =====================================================
    # 3. CICLO CONTROLADO DE TOOL CALLING
    # =====================================================

    for ronda in range(
        1,
        MAX_TOOL_ITERATIONS + 1,
    ):

        rondas_ejecutadas = ronda


        # -------------------------------------------------
        # EL MODELO ANALIZA LA CONVERSACIÓN
        # -------------------------------------------------

        try:
            respuesta_modelo = llm_con_tools.invoke(
                mensajes
            )

        except Exception:
            # Una falla del modelo en esta etapa
            # no debe impedir el fallback hacia RAG
            # cuando todavía no se ha usado ninguna tool.
            if not hubo_tool:
                return None

            return {
                "respuesta": (
                    "No fue posible completar la consulta "
                    "utilizando las herramientas disponibles."
                ),
                "tools_usadas": herramientas_usadas,
                "iteraciones": rondas_ejecutadas,
            }


        # -------------------------------------------------
        # OBTENER TOOL CALLS DE FORMA DEFENSIVA
        # -------------------------------------------------

        tool_calls = getattr(
            respuesta_modelo,
            "tool_calls",
            [],
        ) or []


        # =================================================
        # EL MODELO NO SOLICITA MÁS TOOLS
        # =================================================

        if not tool_calls:

            # Nunca se usó ninguna tool:
            # la consulta debe continuar por el RAG.
            if not hubo_tool:
                return None


            # Ya hubo tools:
            # esta es la respuesta final.
            contenido = str(
                respuesta_modelo.content or ""
            ).strip()


            if not contenido:
                contenido = (
                    "La consulta fue procesada con las "
                    "herramientas disponibles, pero no fue "
                    "posible generar una respuesta final."
                )


            return {
                "respuesta": contenido,
                "tools_usadas": herramientas_usadas,
                "iteraciones": rondas_ejecutadas,
            }


        # =================================================
        # EXISTEN TOOL CALLS
        # =================================================

        hubo_tool = True


        # La respuesta del asistente que contiene los
        # tool_calls debe agregarse antes de los ToolMessage.
        mensajes.append(
            respuesta_modelo
        )


        # =================================================
        # 4. EJECUTAR TOOLS SOLICITADAS
        # =================================================

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


            # ---------------------------------------------
            # VALIDAR LLAMADA
            # ---------------------------------------------

            if not nombre or not tool_call_id:
                continue


            if not isinstance(
                argumentos,
                dict,
            ):
                argumentos = {}


            herramienta = TOOLS_BY_NAME.get(
                nombre
            )


            # ---------------------------------------------
            # IDENTIFICADOR DE LLAMADA
            # ---------------------------------------------

            try:
                argumentos_serializados = json.dumps(
                    argumentos,
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                )

            except Exception:
                argumentos_serializados = str(
                    argumentos
                )


            firma_llamada = (
                f"{nombre}:"
                f"{argumentos_serializados}"
            )


            # ---------------------------------------------
            # EVITAR REPETICIÓN EXACTA
            # ---------------------------------------------

            if firma_llamada in llamadas_ejecutadas:

                resultado = (
                    f"La herramienta '{nombre}' ya fue ejecutada "
                    "previamente con los mismos argumentos. "
                    "Utiliza el resultado anterior."
                )

                mensajes.append(
                    ToolMessage(
                        content=resultado,
                        tool_call_id=tool_call_id,
                    )
                )

                continue


            llamadas_ejecutadas.add(
                firma_llamada
            )


            # ---------------------------------------------
            # TOOL NO REGISTRADA
            # ---------------------------------------------

            if herramienta is None:

                resultado = (
                    f"La herramienta '{nombre}' "
                    "no está registrada en AstreaS."
                )


            # ---------------------------------------------
            # TOOL REGISTRADA
            # ---------------------------------------------

            else:

                try:
                    resultado = herramienta.invoke(
                        argumentos
                    )


                    # =====================================
                    # FALLBACK AUTOMÁTICO FAQ -> RAG
                    # =====================================
                    #
                    # Si la tool buscar_solucion_rapida
                    # no encuentra coincidencias, no debe
                    # cerrar la consulta.
                    #
                    # Devuelve None para que rag_pipeline()
                    # continúe automáticamente por ChromaDB.
                    # =====================================

                    if (
                        nombre == "buscar_solucion_rapida"
                        and str(resultado).startswith(
                            "No se encontraron soluciones rápidas"
                        )
                    ):
                        return None


                    herramientas_usadas.append(
                        nombre
                    )


                except Exception:
                    # No se exponen excepciones internas
                    # al modelo ni al usuario.
                    resultado = (
                        f"No fue posible ejecutar "
                        f"la herramienta '{nombre}'."
                    )


            # =================================================
            # 5. DEVOLVER RESULTADO MEDIANTE TOOLMESSAGE
            # =================================================

            mensajes.append(
                ToolMessage(
                    content=str(resultado),
                    tool_call_id=tool_call_id,
                )
            )


    # =====================================================
    # 6. LÍMITE MÁXIMO ALCANZADO
    # =====================================================

    if hubo_tool:

        return {
            "respuesta": (
                "No fue posible completar la consulta con las "
                "herramientas disponibles dentro del límite "
                "de procesamiento permitido."
            ),
            "tools_usadas": herramientas_usadas,
            "iteraciones": MAX_TOOL_ITERATIONS,
        }


    return None
