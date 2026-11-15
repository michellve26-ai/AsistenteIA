"""Funciones de cuota de Ragas/Groq, puras y comprobables sin usar API.

Groq anuncia los reintentos en segundos, milisegundos o formatos combinados.
Un 429 nunca constituye una puntuacion Ragas de cero.
"""
from __future__ import annotations

import re


def quota_delay_seconds(error: BaseException) -> float | None:
    """Convierte plazos Groq en ms, s, m o h sin confundir 'ms' con minutos."""
    message = str(error)
    match = re.search(
        r"(?:please\s+)?try again in\s+"
        r"(?P<duration>(?:\d+(?:\.\d+)?(?:ms|h|m|s)\s*)+)",
        message,
        flags=re.IGNORECASE,
    )
    if match:
        units = re.findall(
            r"(\d+(?:\.\d+)?)(ms|h|m|s)",
            match.group("duration"),
            flags=re.IGNORECASE,
        )
        factors = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
        if units:
            return sum(float(value) * factors[unit.lower()] for value, unit in units)
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
