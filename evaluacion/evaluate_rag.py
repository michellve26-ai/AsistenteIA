"""Evaluacion Ragas 0.4: top_k=5 frente a top_k=7 para OficinaPro.

Ejecutar desde la raiz: python -m evaluacion.evaluate_rag
Los resultados son experimentales y solo son validos cuando las metricas
se calculan para todas las preguntas evaluables.
"""

import asyncio
import csv
import json
import math
import os
import random
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
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
METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)
TOP_K_VALUES = (5, 7)
MAX_RETRIES = 8
# Pausa entre llamadas; modificable en GitHub Actions sin tocar Render.
REQUEST_PAUSE = float(os.getenv("RAGAS_REQUEST_PAUSE", "8"))


class FastEmbedRagasAdapter(BaseRagasEmbedding):
    """Usa el mismo modelo local para la metrica AnswerRelevancy."""

    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return self.embedding_model.embed_query(text)

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return await asyncio.to_thread(self.embed_text, text)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def save_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", *METRIC_NAMES))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def is_rate_limit_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    message = str(exc).lower()
    return status == 429 or "ratelimit" in type(exc).__name__.lower() or (
        "429" in message and ("rate" in message or "token" in message)
    )


def retry_delay(exc: Exception, attempt: int) -> float:
    """Respeta Retry-After cuando sea posible; limita el backoff."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    try:
        retry_after = float(headers.get("retry-after", 0))
    except (TypeError, ValueError):
        retry_after = 0.0
    match = re.search(r"(?:try again in|retry after)\s+([0-9.]+)\s*s", str(exc), re.I)
    if match:
        retry_after = max(retry_after, float(match.group(1)))
    return min(120.0, max(retry_after + 2, 15.0 * (2 ** min(attempt, 3))) + random.uniform(0, 2))


async def with_rate_limit_retry(operation, label: str):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await operation()
        except Exception as exc:
            if not is_rate_limit_error(exc) or attempt == MAX_RETRIES:
                raise
            delay = retry_delay(exc, attempt)
            print(f"[429] {label}: esperando {delay:.1f}s; intento {attempt + 1}/{MAX_RETRIES}", flush=True)
            await asyncio.sleep(delay)


def create_evaluator():
    client = AsyncOpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
        timeout=90.0,
        max_retries=1,
    )
    judge = llm_factory(config.GROQ_MODEL, provider="openai", client=client)
    print(f"[RAGAS] Evaluador Groq: {config.GROQ_MODEL}", flush=True)
    return judge, client


def create_metrics(judge, embeddings):
    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(
            llm=judge, embeddings=FastEmbedRagasAdapter(embeddings)
        ),
        "context_precision": ContextPrecisionWithReference(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }


def metric_inputs(name, row):
    common = {"user_input": row["question"]}
    if name == "faithfulness":
        return {**common, "response": row["answer"], "retrieved_contexts": row["contexts"]}
    if name == "answer_relevancy":
        return {**common, "response": row["answer"]}
    if name in ("context_precision", "context_recall"):
        return {**common, "reference": row["ground_truth"], "retrieved_contexts": row["contexts"]}
    raise ValueError(f"Metrica desconocida: {name}")


async def generate_row(item, top_k, store, generator, template):
    async def call():
        return await asyncio.to_thread(
            rag_pipeline, store, generator, template,
            item["user_input"], top_k, [],
        )

    result = await with_rate_limit_retry(call, f"generacion {item['id']} k={top_k}")
    return {
        "id": item["id"],
        "question": item["user_input"],
        "answer": str(result["respuesta"]),
        "contexts": [doc.page_content for doc in result["fragmentos"]],
        "ground_truth": item["reference"],
        "expected_behavior": item["expected_behavior"],
        "source_document": item.get("source_document"),
        "source_page": item.get("source_page"),
    }


async def score_row(row, metrics, top_k):
    scores = {"id": row["id"], **{name: None for name in METRIC_NAMES}}
    if row["expected_behavior"] != "answer":
        return scores
    for name, metric in metrics.items():
        async def call():
            return await metric.ascore(**metric_inputs(name, row))

        try:
            result = await with_rate_limit_retry(call, f"{name} {row['id']} k={top_k}")
            value = float(result.value)
            if not math.isfinite(value):
                raise ValueError("Valor no finito")
            scores[name] = value
            print(f"[k={top_k}] {row['id']} {name}: {value:.4f}", flush=True)
        except Exception as exc:
            print(f"[ERROR] k={top_k}, {row['id']}, {name}: {type(exc).__name__}: {exc}", flush=True)
        await asyncio.sleep(REQUEST_PAUSE)
    return scores


def build_summary(rows, score_rows, top_k):
    eligible = sum(row["expected_behavior"] == "answer" for row in rows)
    abstentions = [row for row in rows if row["expected_behavior"] == "abstain"]
    summary = {
        "top_k": top_k,
        "total_preguntas": len(rows),
        "evaluables": eligible,
        "metricas": {},
        "abstenciones_correctas": sum(row["answer"].strip() == NO_ENCONTRADO for row in abstentions),
        "total_fuera_de_alcance": len(abstentions),
    }
    incomplete = False
    for metric in METRIC_NAMES:
        values = [row[metric] for row in score_rows if row[metric] is not None]
        summary["metricas"][metric] = {
            "promedio": round(sum(values) / len(values), 4) if values else None,
            "evaluadas": len(values),
            "esperadas": eligible,
        }
        if len(values) != eligible:
            incomplete = True
    return summary, incomplete


async def evaluate_configuration(k, questions, store, generator, template, metrics):
    responses_path = OUTPUT_DIR / f"respuestas_k{k}.json"
    scores_path = OUTPUT_DIR / f"metricas_k{k}.csv"
    summary_path = OUTPUT_DIR / f"resumen_k{k}.json"

    # Reutiliza respuestas completas de una ejecucion previa si se guardaron
    # en el mismo runner. En Actions, cada ejecucion comienza con disco nuevo.
    rows = []
    if responses_path.exists():
        saved = json.loads(responses_path.read_text(encoding="utf-8"))
        if isinstance(saved, list):
            rows = saved
    ids = [row["id"] for row in rows]
    expected_ids = [item["id"] for item in questions]
    if ids != expected_ids[:len(ids)]:
        raise ValueError("Las respuestas guardadas no corresponden al dataset actual")

    for i, item in enumerate(questions[len(rows):], start=len(rows) + 1):
        print(f"[k={k}] Generando {i}/{len(questions)}: {item['id']}", flush=True)
        rows.append(await generate_row(item, k, store, generator, template))
        save_json(responses_path, rows)
        await asyncio.sleep(REQUEST_PAUSE)

    score_rows = []
    for i, row in enumerate(rows, 1):
        print(f"[k={k}] Metricas {i}/{len(rows)}: {row['id']}", flush=True)
        score_rows.append(await score_row(row, metrics, k))
        save_csv(scores_path, score_rows)

    summary, incomplete = build_summary(rows, score_rows, k)
    save_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary, incomplete


async def main():
    load_dotenv(ROOT / ".env")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Falta GROQ_API_KEY")
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Falta {DATASET_PATH}")
    questions = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or len(questions) < 15:
        raise ValueError("Se requieren al menos 15 preguntas")
    if len({item["id"] for item in questions}) != len(questions):
        raise ValueError("Hay IDs de preguntas duplicados")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings()
    store = load_vectorstore(embeddings)
    generator = get_llm()
    template = get_prompt_template()
    judge, judge_client = create_evaluator()
    metrics = create_metrics(judge, embeddings)
    summaries = []
    incomplete = False

    try:
        for k in TOP_K_VALUES:
            summary, partial = await evaluate_configuration(
                k, questions, store, generator, template, metrics
            )
            summaries.append(summary)
            incomplete = incomplete or partial
            save_json(OUTPUT_DIR / "comparacion.json", summaries)
    finally:
        await judge_client.close()

    if incomplete:
        raise RuntimeError("Evaluacion parcial: revisar metricas no calculadas")
    print("[OK] Evaluacion Ragas completa", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
