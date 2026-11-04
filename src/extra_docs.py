"""
Manuales extra: PDF que se suben manualmente desde el chat.

Es una función ADICIONAL y está aislada de la indexación automática:

  - Los PDF se guardan en su propia carpeta (UPLOAD_DIR, por defecto
    data/uploads) y se indexan en su propia base Chroma (EXTRA_PERSIST_DIR,
    por defecto chroma_db_extra), NO en data/pdfs ni en la base principal.
  - La indexación automática (data/pdfs + src/index_state.py) no los ve,
    no los borra y no se entera de que existen, aunque reconstruya su base.
  - Al consultar, CombinedVectorStore junta los resultados de la base
    principal y de los manuales extra con una búsqueda HÍBRIDA: semántica
    (embeddings) + por palabras clave (BM25), sin fragmentos repetidos.
    Además responde preguntas sobre el archivo que el usuario acaba de
    subir ("¿qué te mandé?"). Si algo falla, cae al retriever clásico de
    la base principal. Con HYBRID_SEARCH=0 se desactiva la búsqueda híbrida.

Seguridad opcional: si defines la variable de entorno UPLOAD_TOKEN, subir o
quitar manuales exige esa clave (cabecera X-Upload-Token). Sin la variable,
cualquiera que abra la app puede subir manuales.
"""

import hmac
import math
import os
import re
import threading
import time
import unicodedata
import uuid
from collections import Counter, defaultdict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from . import config
from .ingest import split_documents


def _int_env(nombre: str, por_defecto: int) -> int:
    try:
        return int(os.getenv(nombre, str(por_defecto)))
    except ValueError:
        return por_defecto


EXTRA_PERSIST_DIR = os.getenv("EXTRA_PERSIST_DIR", "chroma_db_extra")
EXTRA_COLLECTION = "manuales_extra"

UPLOAD_DIR = os.getenv(
    "UPLOAD_DIR",
    os.path.join(os.path.dirname(config.PDF_DIR) or ".", "uploads"),
)

MAX_UPLOAD_MB = _int_env("MAX_UPLOAD_MB", 15)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_EXTRA_DOCS = _int_env("MAX_EXTRA_DOCS", 20)

UPLOAD_TOKEN = os.getenv("UPLOAD_TOKEN", "")


class ExtraDocsError(Exception):
    """Error de validación al subir o quitar un manual extra."""


def token_valido(token: str | None) -> bool:
    """True si no hay clave configurada o si la clave enviada es correcta."""

    if not UPLOAD_TOKEN:
        return True

    return hmac.compare_digest(
        (token or "").encode("utf-8"),
        UPLOAD_TOKEN.encode("utf-8"),
    )


def sanitize_name(nombre: str) -> str:
    """Nombre de archivo seguro (sin rutas ni caracteres raros) y con .pdf."""

    nombre = os.path.basename((nombre or "").replace("\\", "/"))
    nombre = unicodedata.normalize("NFC", nombre)
    nombre = re.sub(r"[^\w .()\-]", "_", nombre)
    nombre = re.sub(r"\s+", " ", nombre).strip(" .")

    raiz = nombre[:-4] if nombre.lower().endswith(".pdf") else nombre
    raiz = raiz.strip(" .")[:100] or "documento"

    return f"{raiz}.pdf"


class ExtraDocs:
    """Base vectorial aparte para los manuales subidos manualmente."""

    def __init__(self, embeddings_provider):
        self._embeddings_provider = embeddings_provider
        self._lock = threading.RLock()
        self._chroma_obj = None
        # Sube cada vez que cambian los manuales extra (invalida el índice léxico).
        self._revision = 0

    # -- acceso a Chroma ------------------------------------------

    def _existe(self) -> bool:
        return self._chroma_obj is not None or os.path.isdir(
            EXTRA_PERSIST_DIR
        )

    def _chroma(self) -> Chroma:
        with self._lock:
            if self._chroma_obj is None:
                self._chroma_obj = Chroma(
                    persist_directory=EXTRA_PERSIST_DIR,
                    collection_name=EXTRA_COLLECTION,
                    embedding_function=self._embeddings_provider(),
                    collection_metadata={"hnsw:space": "cosine"},
                )
            return self._chroma_obj

    # -- consulta --------------------------------------------------

    def count(self) -> int:
        """Cantidad de fragmentos extra indexados (0 si no hay ninguno)."""

        if not self._existe():
            return 0

        try:
            return int(self._chroma()._collection.count())
        except Exception:
            return 0

    def buscar(self, consulta: str, k: int) -> list:
        """Lista de (Document, distancia); vacía si no hay manuales extra."""

        if self.count() == 0:
            return []

        return self._chroma().similarity_search_with_score(consulta, k=k)

    def listar(self) -> list[dict]:
        """Manuales extra subidos: archivo, fragmentos, páginas y fecha."""

        if not self._existe():
            return []

        with self._lock:
            metadatos = self._chroma()._collection.get(
                include=["metadatas"]
            ).get("metadatas") or []

        acumulado: dict[str, dict] = {}

        for meta in metadatos:
            nombre = (meta or {}).get("extra_doc")

            if not nombre:
                continue

            doc = acumulado.setdefault(
                nombre,
                {
                    "archivo": nombre,
                    "fragmentos": 0,
                    "paginas": meta.get("total_pages"),
                    "subido": None,
                },
            )
            doc["fragmentos"] += 1

        for doc in acumulado.values():
            ruta = os.path.join(UPLOAD_DIR, doc["archivo"])

            if os.path.exists(ruta):
                doc["subido"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%S",
                    time.localtime(os.path.getmtime(ruta)),
                )

        return sorted(
            acumulado.values(),
            key=lambda d: d["archivo"].lower(),
        )

    def documentos(self) -> list:
        """Todos los fragmentos extra como Documents (para la búsqueda por palabras)."""

        if self.count() == 0:
            return []

        with self._lock:
            datos = self._chroma()._collection.get(
                include=["documents", "metadatas"]
            )

        return _a_documentos(datos)

    def fragmentos_de(self, nombre: str, limite: int = 3) -> list:
        """Primeros fragmentos (por página) de un manual extra."""

        if self.count() == 0:
            return []

        with self._lock:
            datos = self._chroma()._collection.get(
                where={"extra_doc": nombre},
                include=["documents", "metadatas"],
            )

        docs = _a_documentos(datos)
        docs.sort(key=lambda d: _numero(d.metadata.get("page")))

        return docs[:limite]

    # -- altas y bajas ---------------------------------------------

    def agregar_pdf(self, nombre: str, datos: bytes) -> dict:
        """
        Guarda el PDF, lo divide en fragmentos y los agrega a la base extra.

        Si ya existía un manual extra con el mismo nombre, lo reemplaza
        (los fragmentos nuevos se agregan primero y luego se retiran los
        viejos, así una falla a mitad de camino no deja el manual a medias).
        """

        nombre = sanitize_name(nombre)

        if b"%PDF-" not in datos[:1024]:
            raise ExtraDocsError("El archivo no parece ser un PDF válido.")

        if os.path.exists(os.path.join(config.PDF_DIR, nombre)):
            raise ExtraDocsError(
                "Ya existe un manual base con ese nombre. "
                "Renombra el archivo y súbelo de nuevo."
            )

        with self._lock:
            existentes = {d["archivo"] for d in self.listar()}
            reemplazado = nombre in existentes

            if not reemplazado and len(existentes) >= MAX_EXTRA_DOCS:
                raise ExtraDocsError(
                    f"Ya hay {MAX_EXTRA_DOCS} manuales extra. "
                    "Quita alguno antes de subir otro."
                )

            os.makedirs(UPLOAD_DIR, exist_ok=True)
            destino = os.path.join(UPLOAD_DIR, nombre)
            temporal = destino + ".tmp"

            with open(temporal, "wb") as archivo:
                archivo.write(datos)

            try:
                try:
                    paginas = PyPDFLoader(temporal).load()
                except Exception as error:
                    raise ExtraDocsError(
                        "No se pudo leer el PDF "
                        "(¿está dañado o protegido con contraseña?)."
                    ) from error

                paginas = [p for p in paginas if p.page_content.strip()]

                if not paginas:
                    raise ExtraDocsError(
                        "El PDF no tiene texto seleccionable "
                        "(¿es un escaneo?). No se puede indexar."
                    )

                total_paginas = max(
                    (p.metadata.get("total_pages") or 0) for p in paginas
                ) or len(paginas)

                fragmentos = split_documents(paginas)
                version = uuid.uuid4().hex

                for fragmento in fragmentos:
                    fragmento.metadata = {
                        **fragmento.metadata,
                        "source": nombre,
                        "extra_doc": nombre,
                        "origen": "manual",
                        "version": version,
                    }

                chroma = self._chroma()

                try:
                    for i in range(0, len(fragmentos), 200):
                        chroma.add_documents(fragmentos[i:i + 200])
                except Exception:
                    chroma._collection.delete(where={"version": version})
                    raise

                if reemplazado:
                    chroma._collection.delete(
                        where={
                            "$and": [
                                {"extra_doc": nombre},
                                {"version": {"$ne": version}},
                            ]
                        }
                    )

                os.replace(temporal, destino)
                self._revision += 1

            finally:
                if os.path.exists(temporal):
                    os.remove(temporal)

        print(
            f"[EXTRA] '{nombre}' indexado: {len(fragmentos)} fragmentos.",
            flush=True,
        )

        return {
            "archivo": nombre,
            "paginas": total_paginas,
            "fragmentos": len(fragmentos),
            "reemplazado": reemplazado,
        }

    def eliminar(self, nombre: str) -> int | None:
        """
        Quita un manual extra (sus fragmentos y su archivo).

        Devuelve cuántos fragmentos se quitaron, o None si no existía.
        """

        nombre = sanitize_name(nombre)

        with self._lock:
            quitados = 0

            if self._existe():
                chroma = self._chroma()
                ids = chroma._collection.get(
                    where={"extra_doc": nombre}
                ).get("ids") or []
                quitados = len(ids)

                if ids:
                    chroma._collection.delete(ids=ids)

            self._revision += 1
            ruta = os.path.join(UPLOAD_DIR, nombre)
            existia_archivo = os.path.exists(ruta)

            if existia_archivo:
                os.remove(ruta)

            if quitados == 0 and not existia_archivo:
                return None

            return quitados


# =============================================================
# BÚSQUEDA HÍBRIDA (semántica + palabras clave) PARA EL CHAT
# =============================================================

USAR_HIBRIDO = os.getenv("HYBRID_SEARCH", "1") != "0"

_STOPWORDS = frozenset(
    """a al algo ante aqui as asi aun con contra cual cuales cuando cuanto
    de del desde donde el ella ellas ello ellos en entre era eran es esa
    esas ese eso esos esta estas este esto estos fue ha han has hay la las
    le les lo los me mi mis mucho muy ni no nos o otra otro para pero por
    porque que quien se si sin sobre son su sus te ti tu tus un una uno
    unos y ya yo puedo puede pueden podria hago hace hacer quiero necesito
    debo debe favor ayudame ayudarme ayuda como""".split()
)


def _plano(texto: str) -> str:
    """Minúsculas y sin tildes (para comparar palabras)."""

    texto = unicodedata.normalize("NFKD", (texto or "").lower())

    return "".join(c for c in texto if not unicodedata.combining(c))


def _raiz(palabra: str) -> str:
    """Raíz muy simple: quita plurales y deja 6 letras (reversar ~ reverso)."""

    if len(palabra) > 4 and palabra.endswith("es"):
        palabra = palabra[:-2]
    elif len(palabra) > 3 and palabra.endswith("s"):
        palabra = palabra[:-1]

    return palabra[:6]


def _tokens(texto: str) -> list[str]:
    """Palabras clave (raíces) + pares de palabras seguidas."""

    palabras = [
        _raiz(p)
        for p in re.findall(r"[a-z0-9]+", _plano(texto))
        if len(p) > 1 and p not in _STOPWORDS
    ]

    pares = [f"{a}_{b}" for a, b in zip(palabras, palabras[1:])]

    return palabras + pares


def _numero(valor) -> float:
    try:
        return float(valor)
    except (TypeError, ValueError):
        return 0.0


def _a_documentos(datos: dict) -> list:
    textos = datos.get("documents") or []
    metas = datos.get("metadatas") or []

    return [
        Document(page_content=texto or "", metadata=meta or {})
        for texto, meta in zip(textos, metas)
    ]


class _IndiceLexico:
    """Índice BM25 en memoria (sin dependencias extra)."""

    K1 = 1.5
    B = 0.75

    def __init__(self, docs: list):
        self.docs = docs
        self.n = len(docs)
        self.postings: dict[str, list] = defaultdict(list)
        self.longitudes: list[int] = []

        for i, doc in enumerate(docs):
            tokens = _tokens(doc.page_content)
            self.longitudes.append(len(tokens))

            for termino, veces in Counter(tokens).items():
                self.postings[termino].append((i, veces))

        self.promedio = (
            sum(self.longitudes) / self.n if self.n else 0.0
        ) or 1.0

    def buscar(self, consulta: str, k: int) -> list:
        puntajes: dict[int, float] = defaultdict(float)

        for termino in set(_tokens(consulta)):
            lista = self.postings.get(termino)

            if not lista:
                continue

            idf = math.log(
                1 + (self.n - len(lista) + 0.5) / (len(lista) + 0.5)
            )

            for i, veces in lista:
                norma = veces + self.K1 * (
                    1 - self.B + self.B * self.longitudes[i] / self.promedio
                )
                puntajes[i] += idf * veces * (self.K1 + 1) / norma

        orden = sorted(puntajes.items(), key=lambda p: p[1], reverse=True)

        return [(self.docs[i], puntaje) for i, puntaje in orden[:k]]


_cache_lexico: dict = {"clave": None, "indice": None}
_lock_lexico = threading.Lock()


def _indice_lexico(principal, extra: ExtraDocs) -> _IndiceLexico:
    """Índice BM25 de la base principal + manuales extra (se reconstruye solo si cambian)."""

    clave = (
        id(principal),
        int(principal._collection.count()),
        extra._revision,
        extra.count(),
    )

    with _lock_lexico:
        if _cache_lexico["clave"] != clave:
            datos = principal._collection.get(
                include=["documents", "metadatas"]
            )
            docs = _a_documentos(datos) + extra.documentos()

            _cache_lexico["indice"] = _IndiceLexico(docs)
            _cache_lexico["clave"] = clave

            print(
                f"[BUSQUEDA] Índice por palabras listo: {len(docs)} fragmentos.",
                flush=True,
            )

        return _cache_lexico["indice"]


def _clave_doc(doc) -> tuple:
    meta = doc.metadata or {}

    return (
        os.path.basename(str(meta.get("source", ""))),
        meta.get("page"),
        hash(doc.page_content),
    )


def _fusionar(listas: list, constante: int = 60) -> list:
    """Reciprocal Rank Fusion: junta varios rankings en uno."""

    puntajes: dict = defaultdict(float)
    docs: dict = {}

    for lista in listas:
        for posicion, doc in enumerate(lista):
            clave = _clave_doc(doc)
            docs.setdefault(clave, doc)
            puntajes[clave] += 1.0 / (constante + posicion + 1)

    orden = sorted(puntajes, key=lambda c: puntajes[c], reverse=True)

    return [docs[c] for c in orden]


def _diversificar(docs: list, k: int, umbral: float = 0.8) -> list:
    """Toma los k mejores sin fragmentos casi idénticos (el texto se repite mucho)."""

    elegidos: list = []
    conjuntos: list[set] = []

    for doc in docs:
        palabras = set(_tokens(doc.page_content))

        repetido = any(
            palabras
            and otro
            and len(palabras & otro) / len(palabras | otro) >= umbral
            for otro in conjuntos
        )

        if repetido:
            continue

        elegidos.append(doc)
        conjuntos.append(palabras)

        if len(elegidos) >= k:
            break

    return elegidos


# Preguntas del estilo "¿qué te mandé?", "el archivo que subí", "qué manuales cargué".
_PATRON_CARGAS = re.compile(
    r"\b("
    r"(te|les?) (mande|envie|subi|cargue|adjunte|pase)\b"
    r"|te acab\w* de (mandar|enviar|subir|cargar|adjuntar|pasar)"
    r"|acab\w* de (mandar|enviar|subir|cargar|adjuntar|pasar)(te|le)\b"
    r"|acab\w* de (subir|cargar|adjuntar)\b"
    r"|usuario (mando|envio|subio|cargo|adjunto)\b"
    r"|usuario acaba de (mandar|enviar|subir|cargar|adjuntar|pasar)"
    r"|(archivo|documento|pdf|manual|adjunto)s? (que )?(te )?"
    r"(mande|envie|subi|cargue|adjunte|pase|cargaste|recibiste)\b"
    r"|que (archivos?|documentos?|pdfs?|manuales?|adjuntos?)"
    r"( extra| adicionales?| nuevos?)? "
    r"(tienes|has recibido|recibiste|cargue|subi|cargaron|subieron)\b"
    r")"
)


def _respuesta_sobre_cargas(consulta: str, extra: ExtraDocs, k: int):
    """Si preguntan por el archivo subido, devuelve su resumen + primeros fragmentos."""

    if not _PATRON_CARGAS.search(_plano(consulta)):
        return None

    lista = extra.listar()

    if not lista:
        return None

    ultimo = max(lista, key=lambda d: d.get("subido") or "")

    detalle = "; ".join(
        f"«{d['archivo']}» ({d.get('paginas') or '?'} págs., "
        f"{d['fragmentos']} fragmentos)"
        for d in lista
    )

    cabecera = Document(
        page_content=(
            "Documentos que el usuario cargó manualmente en el chat "
            f"como referencia extra: {detalle}. "
            f"El más reciente es «{ultimo['archivo']}». "
            "Abajo están sus primeros fragmentos."
        ),
        metadata={
            "source": "Documentos cargados por el usuario",
            "page": "-",
        },
    )

    return [cabecera] + extra.fragmentos_de(
        ultimo["archivo"], max(k - 1, 1)
    )


class _HibridoRetriever:
    """Retriever mínimo (solo .invoke): semántico + palabras clave + sin repetidos."""

    def __init__(self, principal, extra: ExtraDocs, k: int):
        self._principal = principal
        self._extra = extra
        self._k = k

    def _buscar(self, consulta: str) -> list:
        hay_extra = self._extra.count() > 0

        if hay_extra:
            sobre_cargas = _respuesta_sobre_cargas(
                consulta, self._extra, self._k
            )

            if sobre_cargas:
                return sobre_cargas

        pool = max(self._k * 4, 20)

        semantico = list(
            self._principal.similarity_search_with_score(consulta, k=pool)
        )

        if hay_extra:
            semantico += self._extra.buscar(consulta, pool)

        # Distancia coseno: menor = más parecido.
        semantico.sort(key=lambda par: par[1])

        try:
            palabras = _indice_lexico(self._principal, self._extra).buscar(
                consulta, pool
            )
        except Exception as error:
            print(
                f"[BUSQUEDA] Sin búsqueda por palabras: {error}",
                flush=True,
            )
            palabras = []

        fusion = _fusionar(
            [
                [doc for doc, _ in semantico],
                [doc for doc, _ in palabras],
            ]
        )

        return _diversificar(fusion, self._k)

    def invoke(self, consulta: str, *args, **kwargs) -> list:
        try:
            return self._buscar(consulta)

        except Exception as error:
            # A prueba de fallos: si la búsqueda híbrida falla, la pregunta
            # se responde con el retriever clásico de la base principal.
            print(
                f"[BUSQUEDA] Se usó la búsqueda clásica por un error: {error}",
                flush=True,
            )

            return self._principal.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self._k},
            ).invoke(consulta)


class CombinedVectorStore:
    """
    Se comporta como el vectorstore principal, pero su retriever hace una
    búsqueda híbrida e incluye los manuales extra. rag_pipeline() lo usa
    sin cambios.
    """

    def __init__(self, principal, extra: ExtraDocs):
        self._principal = principal
        self._extra = extra

    def as_retriever(self, search_type="similarity", search_kwargs=None, **kwargs):
        if not USAR_HIBRIDO:
            return self._principal.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs,
                **kwargs,
            )

        k = int((search_kwargs or {}).get("k", config.TOP_K))
        return _HibridoRetriever(self._principal, self._extra, k)

    def __getattr__(self, nombre):
        # Cualquier otro uso se delega a la base principal.
        return getattr(self._principal, nombre)
