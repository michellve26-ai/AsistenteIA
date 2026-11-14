"""Compara costo real del RAG fijo y el agente sobre las mismas 15 preguntas.

Requiere indice y credencial Groq. No inventa resultados ni modifica el baseline.
"""
import csv
import json
import time
from pathlib import Path

from src.rag_chain import get_llm, get_prompt_template, rag_pipeline, rag_pipeline_fijo
from src.vectorstore import get_embeddings, load_vectorstore

ROOT = Path(__file__).resolve().parents[1]


class MeasuredLLM:
    def __init__(self, delegate, counters=None):
        self.delegate = delegate
        self.counters = counters if counters is not None else {"llamadas": 0, "tokens": 0}

    def bind_tools(self, tools):
        return MeasuredLLM(self.delegate.bind_tools(tools), self.counters)

    def invoke(self, *args, **kwargs):
        answer = self.delegate.invoke(*args, **kwargs)
        self.counters["llamadas"] += 1
        usage = getattr(answer, "usage_metadata", None) or {}
        if not usage:
            usage = (getattr(answer, "response_metadata", {}) or {}).get("token_usage", {})
        self.counters["tokens"] += int(usage.get("total_tokens") or 0)
        return answer


def main():
    dataset = json.loads((ROOT / "evaluacion" / "preguntas_ragas.json").read_text(encoding="utf-8"))
    if len(dataset) < 15:
        raise RuntimeError("Se requieren al menos 15 preguntas.")
    store = load_vectorstore(get_embeddings())
    prompt = get_prompt_template()
    rows = []
    for question in dataset:
        for mode, pipeline in (("Avance 2 fijo", rag_pipeline_fijo), ("Avance 3 agente", rag_pipeline)):
            llm = MeasuredLLM(get_llm())
            begin = time.perf_counter()
            result = pipeline(store, llm, prompt, question["user_input"], k=5, historial=[])
            rows.append({
                "id": question["id"], "modo": mode,
                "llamadas_llm": llm.counters["llamadas"],
                "tokens": llm.counters["tokens"],
                "tiempo_s": round(time.perf_counter() - begin, 3),
                "respuesta": result["respuesta"],
            })
    destination = ROOT / "evaluacion" / "resultados_agente" / "costos_comparados.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for mode in ("Avance 2 fijo", "Avance 3 agente"):
        values = [r for r in rows if r["modo"] == mode]
        print(mode, {field: round(sum(r[field] for r in values) / len(values), 3)
                     for field in ("llamadas_llm", "tokens", "tiempo_s")})
    print("Detalle:", destination)


if __name__ == "__main__":
    main()
