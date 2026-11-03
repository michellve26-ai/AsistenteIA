"""
Control de cambios de los documentos indexados en AstreaS.

Este módulo calcula una huella de los PDF disponibles
para determinar si la base vectorial necesita actualizarse.
"""

import hashlib
import json
from pathlib import Path

from src import config


BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / config.PDF_DIR

STATE_FILE = BASE_DIR / "data" / "index_state.json"


def _hash_archivo(ruta: Path) -> str:
    """
    Calcula el SHA-256 de un archivo.
    """

    sha256 = hashlib.sha256()

    with ruta.open("rb") as archivo:
        while True:
            bloque = archivo.read(1024 * 1024)

            if not bloque:
                break

            sha256.update(bloque)

    return sha256.hexdigest()


def obtener_estado_actual() -> dict[str, str]:
    """
    Obtiene el estado actual de todos los PDF.

    Returns:
        Diccionario:
        {
            "archivo.pdf": "hash_sha256"
        }
    """

    if not PDF_DIR.exists():
        return {}

    if not PDF_DIR.is_dir():
        return {}

    estado = {}

    for archivo in sorted(PDF_DIR.glob("*.pdf")):

        if not archivo.is_file():
            continue

        try:
            estado[archivo.name] = _hash_archivo(
                archivo
            )

        except OSError:
            continue

    return estado


def cargar_estado_guardado() -> dict[str, str]:
    """
    Lee el estado de los PDF correspondiente
    a la última indexación válida.
    """

    if not STATE_FILE.exists():
        return {}

    try:
        datos = json.loads(
            STATE_FILE.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(datos, dict):
            return {}

        return {
            str(nombre): str(hash_archivo)
            for nombre, hash_archivo in datos.items()
        }

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}


def guardar_estado_actual() -> None:
    """
    Guarda el estado actual de los PDF.
    """

    estado = obtener_estado_actual()

    STATE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    STATE_FILE.write_text(
        json.dumps(
            estado,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def documentos_cambiaron() -> bool:
    """
    Determina si los documentos actuales son
    diferentes a los utilizados en la última indexación.

    Detecta:

    - PDF nuevos.
    - PDF eliminados.
    - PDF modificados.
    """

    estado_actual = obtener_estado_actual()
    estado_guardado = cargar_estado_guardado()

    return estado_actual != estado_guardado
