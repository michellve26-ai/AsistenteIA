"""Herramientas del agente Astrea (Avance 3).

Las restricciones de volumen, dominio y salida a Internet estan implementadas aqui,
no dependen exclusivamente de las instrucciones del modelo.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import PurePath
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from langchain_core.tools import tool

# Controles no modificables mediante argumentos del modelo.
MAX_FRAGMENTS = 7
MAX_FRAGMENT_CHARS = 850
MAX_QUERY_CHARS = 220
MAX_WEB_RESULTS = 3
MAX_EXCERPT_CHARS = 260
ALLOWED_WEB_DOMAINS = (
    "dian.gov.co",
    "gov.co",
    "mincit.gov.co",
    "funcionpublica.gov.co",
    "sic.gov.co",
    "ugpp.gov.co",
    "mintic.gov.co",
)


@dataclass
class AgentToolState:
    docs: list = field(default_factory=list)
    web_sources: list[dict] = field(default_factory=list)
    base_called: bool = False
    query_fragments: list[str] = field(default_factory=list)


def _page(metadata: dict) -> str:
    number = metadata.get("page")
    if isinstance(number, int) and number >= 0:
        # PyPDFLoader usa paginacion con base cero.
        return str(number + 1)
    return str(metadata.get("page_label") or metadata.get("section") or "sin pagina")[:60]


def _safe_name(value: object) -> str:
    return PurePath(str(value or "desconocido").replace("\\", "/")).name[:100]


def _allowed_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        host = (parsed.hostname or "").lower().rstrip(".")
        return parsed.scheme == "https" and any(
            host == domain or host.endswith("." + domain)
            for domain in ALLOWED_WEB_DOMAINS
        )
    except ValueError:
        return False


def _safe_public_query(query: str, snippets: list[str]) -> bool:
    """Evita exfiltrar cadenas reconocibles del corpus o datos privados a la web.

    La comprobacion de coincidencias es deliberadamente restrictiva. Una consulta
    completamente publica debe ser reformulada sin copiar ningun fragmento local.
    """
    if not query or len(query) > MAX_QUERY_CHARS:
        return False
    # No son permitidas busquedas arbitrarias: debe tratarse de una consulta
    # institucional/publica del dominio. Esto complementa el prompt.
    public_topics = ("dian", "norma", "ley", "resolucion", "resolución", "decreto",
                     "facturacion electronica", "facturación electrónica", "gov.co",
                     "ministerio", "actual", "vigente", "tributaria", "tributario",
                     "impuesto", "regulacion", "regulación", "tramite", "trámite",
                     "publico", "público", "requisito")
    if not any(term in query.casefold() for term in public_topics):
        return False
    forbidden = (
        r"\b(?:token|clave|password|contrase(?:n|ñ)a|api[_ -]?key|secreto)\b",
        r"\b(?:documento interno|manual privado|manual completo|base de datos interna|corpus)\b",
        r"\b\d{7,}\b",  # NIT, cedulas o identificadores largos
        r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b",
    )
    if any(re.search(pattern, query, flags=re.I) for pattern in forbidden):
        return False
    normalized_query = " ".join(query.casefold().split())
    for snippet in snippets:
        words = " ".join(snippet.casefold().split()).split()
        if len(words) < 7:
            continue
        # No transmitir frases consecutivas extraidas de la base.
        for i in range(len(words) - 6):
            if " ".join(words[i:i + 7]) in normalized_query:
                return False
    return True


def _money(value: float) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("El valor debe ser numerico") from exc
    if not result.is_finite() or result < 0 or result > Decimal("1000000000000"):
        raise ValueError("Valor fuera del rango permitido")
    return result


def create_agent_tools(vector_store, *, k: int = 5):
    """Crea tools por consulta con estado aislado para impedir mezclar sesiones.

    Devuelve (herramientas, estado). La tool de base se puede probar directamente
    sin LLM con buscar_en_base.invoke({'consulta': 'crear factura'}).
    """
    state = AgentToolState()
    max_k = min(max(int(k), 1), MAX_FRAGMENTS)

    @tool("buscar_en_base")
    def buscar_en_base(consulta: str) -> str:
        """Busca procedimientos de Astrea/OficinaPro en manuales locales.

        Usar SIEMPRE primero para preguntas de soporte, manuales y procedimientos.
        Puede invocarse varias veces con distintas consultas para dos temas.
        Cada resultado indica nombre del PDF y pagina. No devuelve documentos completos.
        """
        query = str(consulta or "").strip()[:MAX_QUERY_CHARS]
        if not query:
            return "BASE: la consulta esta vacia."
        if vector_store is None:
            return "BASE: el indice no esta disponible."
        # El marcador se activa solo al hacer una busqueda real.
        state.base_called = True
        docs = vector_store.similarity_search(query, k=max_k)
        if not docs:
            return "BASE: no se encontraron fragmentos relacionados."
        parts = []
        for document in docs[:max_k]:
            metadata = document.metadata or {}
            name = _safe_name(metadata.get("source"))
            page = _page(metadata)
            content = " ".join(str(document.page_content or "").split())[:MAX_FRAGMENT_CHARS]
            if not content:
                continue
            # Deduplicacion para metricas Ragas y fuentes de la interfaz.
            identity = (name, page, content[:100])
            if not any((
                _safe_name((d.metadata or {}).get("source")),
                _page(d.metadata or {}),
                " ".join(d.page_content.split())[:100],
            ) == identity for d in state.docs):
                # Las metricas Ragas usan exactamente el fragmento entregado al LLM,
                # no el contenido original sin recortar del documento.
                from langchain_core.documents import Document
                state.docs.append(Document(page_content=content, metadata=dict(metadata)))
            state.query_fragments.append(content)
            parts.append(f"[Fuente: {name}, pag. {page}] {content}")
        if not parts:
            return "BASE: no se encontraron textos utilizables."
        return "Origen: BASE local.\n" + "\n---\n".join(parts)

    @tool("calcular_saldo_factura")
    def calcular_saldo_factura(total: float, abono: float) -> str:
        """Calcula el saldo pendiente exacto de una factura, sin modificar datos.

        Usar cuando el usuario proporcione monto total y abono; extraer ambos
        argumentos de la pregunta. Es un calculo ilustrativo, sin acceso contable.
        """
        try:
            a, b = _money(total), _money(abono)
        except ValueError as exc:
            return f"ERROR DE CALCULO: {exc}"
        pendiente = a - b
        return (
            "Resultado del calculo (no altera la plataforma): "
            f"total={a:.2f} COP; abono={b:.2f} COP; "
            f"saldo pendiente={max(pendiente, Decimal('0')):.2f} COP; "
            f"saldo a favor={max(-pendiente, Decimal('0')):.2f} COP."
        )

    @tool("buscar_en_web")
    def buscar_en_web(consulta: str) -> str:
        """Busca fuentes PUBLICAS ACTUALES en sitios oficiales de Colombia.

        Usar solo DESPUES de consultar la base local y comprobar que esta no
        responde la pregunta publica/actual. Prohibido enviar fragmentos,
        nombres privados, credenciales, identificadores o el corpus. El modelo
        NO elige los dominios autorizados. Como maximo devuelve 3 resultados.
        """
        if not state.base_called:
            return "WEB BLOQUEADA: primero consulta buscar_en_base."
        query = " ".join(str(consulta or "").split())
        if not _safe_public_query(query, state.query_fragments):
            return "WEB BLOQUEADA: consulta no publica o contiene datos del corpus."
        api_key = os.environ.get("TAVILY_API_KEY", "").strip()
        if not api_key:
            return "WEB NO DISPONIBLE: TAVILY_API_KEY no configurada."
        payload = json.dumps({
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": MAX_WEB_RESULTS,
            "include_domains": list(ALLOWED_WEB_DOMAINS),
            "include_answer": False,
            "include_raw_content": False,
        }).encode("utf-8")
        request = Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=12) as response:
                data = json.loads(response.read(120000).decode("utf-8"))
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            return f"WEB NO DISPONIBLE: {type(exc).__name__}"
        entries = []
        results = data.get("results", []) if isinstance(data, dict) else []
        if not isinstance(results, list):
            results = []
        for item in results[:MAX_WEB_RESULTS]:
            if not isinstance(item, dict):
                continue
            link = str(item.get("url", "")).strip()
            if not _allowed_url(link):
                continue
            title = re.sub(r"\s+", " ", str(item.get("title", ""))).strip()[:130]
            excerpt = re.sub(r"<[^>]+>", "", str(item.get("content", "")))
            excerpt = " ".join(excerpt.split())[:MAX_EXCERPT_CHARS]
            source = {"titulo": title or "Sin titulo", "url": link, "extracto": excerpt, "origen": "web"}
            state.web_sources.append(source)
            entries.append(f"Origen WEB: {title} [web: {link}] {excerpt}")
        return "\n\n".join(entries) if entries else "WEB: sin resultados en los sitios permitidos."

    # Se conservan las tools previas, sin cambiar rutas ni funciones.
    from .tools import consultar_estado_base, listar_documentos, buscar_solucion_rapida

    return [
        buscar_en_base,
        calcular_saldo_factura,
        buscar_en_web,
        consultar_estado_base,
        listar_documentos,
        buscar_solucion_rapida,
    ], state
