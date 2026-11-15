"""Prueba real .invoke() contra un indice ya creado; no cambia documentos ni vectores."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.agent_tools import create_agent_tools
from src.vectorstore import get_embeddings, load_vectorstore

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluacion" / "resultados_avance3" / "prueba_buscar_en_base.json"


def main() -> None:
    store = load_vectorstore(get_embeddings())  # Solo lectura del indice de prueba.
    tools, state = create_agent_tools(store, k=5)
    search = next(item for item in tools if item.name == "buscar_en_base")
    query = "¿Cómo se registra un producto de inventario en Astrea?"
    output = search.invoke({"consulta": query})
    evidence = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "herramienta": search.name,
        "argumentos": {"consulta": query},
        "salida": output,
        "fragmentos": len(state.docs),
        "fuentes": [
            {"archivo": Path(str(d.metadata.get("source", "desconocido"))).name,
             "pagina_cero_basada": d.metadata.get("page")}
            for d in state.docs
        ],
        "sin_llm": True,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[TOOL] Resultado guardado: {OUT}; fragmentos={len(state.docs)}")
    if not state.docs:
        raise SystemExit("BUSQUEDA SIN FRAGMENTOS: revisar corpus y consulta de prueba.")


if __name__ == "__main__":
    main()
