"""Funciones de cuota de Ragas/Groq, puras y comprobables sin usar API.

Groq anuncia los reintentos en segundos, milisegundos o formatos combinados.
Un 429 nunca constituye una puntuacion Ragas de cero.
"""
from __future__ import annotations

import re


def quota_delay_seconds(error: BaseException) -> float | None:
    """Convierte los plazos de 'Please try again in ...' a segundos."""
    message = str(error)
    match = re.search(
        r"(?:please\s+)?try again in\s+"
        r"(?:(?P<h>\d+(?:\.\d+)?)h\s*)?"
        r"(?:(?P<m>\d+(?:\.\d+)?)m\s*)?"
        r"(?:(?P<s>\d+(?:\.\d+)?)s|(?P<ms>\d+(?:\.\d+)?)ms)?",
        message,
        flags=re.IGNORECASE,
    )
    if match and any(match.group(x) is not None for x in ("h", "m", "s", "ms")):
        return (
            float(match.group("h") or 0) * 3600
            + float(match.group("m") or 0) * 60
            + float(match.group("s") or 0)
            + float(match.group("ms") or 0) / 1000
        )
    headers = getattr(getattr(error, "response", None), "headers", None)
    if headers:
        try:
            value = float(headers.get("retry-after", ""))
            if 0 <= value < float("inf"):
                return value
        except (TypeError, ValueError):
            pass
    return None


def retry_wait_seconds(error: BaseException, attempt: int, max_wait: float) -> float | None:
    """Espera conservadora con margen y backoff acotado; None si excede limite."""
    advertised = quota_delay_seconds(error)
    if advertised is None:
        advertised = 60.0
    wait = max(12.0, advertised + 8.0, min(90.0, 12.0 * (attempt + 1)))
    return wait if wait <= max_wait else None
