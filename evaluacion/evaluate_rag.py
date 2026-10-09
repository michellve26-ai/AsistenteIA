"""Evaluacion Ragas 0.4: comparar recuperacion top_k=5 y top_k=7."""
import asyncio
import csv
import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import AsyncGroq
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)

from src import config
from src.rag_chain import get_llm, get_prompt_template, rag_pipeline
from src.vectorstore import get_embeddings, load_vectorstore

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evaluacion" / "preguntas_ragas.json"
OUTPUT_DIR = ROOT / "evaluacion" / "resultados"
NO_ENCONTRADO = (
    "No encontré esto en los manuales. "
    "Te recomiendo escalar el caso al equipo correspondiente."
)
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


class FastEmbedRagasAdapter(BaseRagasEmbedding):
    """Reutiliza embeddings locales del RAG para relevancia semantica en Ragas."""

    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return self.embedding_model.embed_query(text)

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return await asyncio.to_thread(self.embed_text, text)


def save_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


async def score_one(metrics, row):
    if row["expected_behavior"] != "answer":
        return {"id": row["id"], **{name: None for name in METRIC_NAMES}}
    inputs = {
        "user_input": row["question"],
        "response": row["answer"],
        "reference": row["ground_truth"],
        "retrieved_contexts": row["contexts"],
    }
    scores = {"id": row["id"]}
    for name, metric in metrics.items():
        try:
            result = await metric.ascore(**inputs)
            value = float(result.value)
            if not math.isfinite(value):
                raise ValueError("Resultado no finito")
            scores[name] = value
        except Exception as exc:
            scores[name] = None
            print(f"ERROR metrica {name} / {row['id']}: {type(exc).__name__}: {exc}", flush=True)
    return scores


async def main():
    load_dotenv(ROOT / ".env")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Falta GROQ_API_KEY")
    questions = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if len(questions) < 15:
        raise ValueError("Se requieren al menos 15 preguntas")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings()
    vectorstore = load_vectorstore(embeddings)
    generator = get_llm()
    template = get_prompt_template()

    judge_client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    judge = llm_factory(config.GROQ_MODEL, provider="groq", client=judge_client)
    metrics = {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=FastEmbedRagasAdapter(embeddings)),
        "context_precision": ContextPrecisionWithReference(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }
    incomplete = False
    summaries = []

    for top_k in (5, 7):
        rows = []
        for i, item in enumerate(questions, start=1):
            print(f"[k={top_k}] Pregunta {i}/{len(questions)}: {item['id']}", flush=True)
            result = await asyncio.to_thread(
                rag_pipeline, vectorstore, generator, template,
                item["user_input"], top_k, [],
            )
            rows.append({
                "id": item["id"],
                "question": item["user_input"],
                "answer": result["respuesta"],
                "contexts": [d.page_content for d in result["fragmentos"]],
                "ground_truth": item["reference"],
                "expected_behavior": item["expected_behavior"],
                "source_document": item.get("source_document"),
                "source_page": item.get("source_page"),
            })
        save_json(OUTPUT_DIR / f"respuestas_k{top_k}.json", rows)
        score_rows = []
        for i, row in enumerate(rows, start=1):
            print(f"[k={top_k}] Metricas {i}/{len(rows)}: {row['id']}", flush=True)
            score_rows.append(await score_one(metrics, row))

        with (OUTPUT_DIR / f"metricas_k{top_k}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=("id", *METRIC_NAMES))
            writer.writeheader()
            writer.writerows(score_rows)
        eligible = sum(r["expected_behavior"] == "answer" for r in rows)
        summary = {"top_k": top_k, "total_preguntas": len(rows), "evaluables": eligible, "metricas": {}}
        for metric in METRIC_NAMES:
            values = [r[metric] for r in score_rows if r[metric] is not None]
            summary["metricas"][metric] = {
                "promedio": round(sum(values) / len(values), 4) if values else None,
                "evaluadas": len(values),
                "esperadas": eligible,
            }
            if len(values) != eligible:
                incomplete = True
        abstentions = [r for r in rows if r["expected_behavior"] == "abstain"]
        summary["abstenciones_correctas"] = sum(r["answer"].strip() == NO_ENCONTRADO for r in abstentions)
        summary["total_fuera_de_alcance"] = len(abstentions)
        summaries.append(summary)
        save_json(OUTPUT_DIR / f"resumen_k{top_k}.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    save_json(OUTPUT_DIR / "comparacion.json", summaries)
    if incomplete:
        raise RuntimeError("Evaluacion parcial: revisar errores de metricas en logs y CSV")
    print("Evaluacion completa. Descargar artefactos de GitHub Actions.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
