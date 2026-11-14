"""Registro JSONL de ejecucion de herramientas, sin almacenar respuestas ni corpus."""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(os.environ.get("AGENT_LOG_PATH", "logs/agent_tools.jsonl"))


def _redact(value):
    if isinstance(value, dict):
        return {str(k): "[PROTEGIDO]" if re.search(r"token|secret|passw|clave|api.?key", str(k), re.I)
                else _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value[:10]]
    if isinstance(value, str):
        # Log estructurado sin almacenar el texto de consultas ni del corpus.
        return f"[TEXTO {len(value)} caracteres]"
    return value


def record_agent_run(*, trace: list, llm_calls: int, tokens: int, seconds: float, status: str):
    """Guarda tool, claves de argumentos y costo, sin texto del corpus."""
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tools": _redact(trace),
        "llm_calls": llm_calls,
        "tokens": tokens,
        "time_seconds": round(seconds, 3),
        "status": status,
    }
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_WRONLY | os.O_APPEND
        fd = os.open(str(LOG_PATH), flags, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:
        # Una falla del registro no debe bloquear la consulta del usuario.
        pass
