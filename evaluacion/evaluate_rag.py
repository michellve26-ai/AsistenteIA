
"""
Evaluación Ragas 0.4 del asistente OficinaPro.

Compara la recuperación con top_k=5 y top_k=7.
Calcula:
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

Guarda respuestas, métricas y comparación.
"""

import asyncio
import csv
import json
import math
import os
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
from src.rag_chain import (
    get_llm,
    get_prompt_template,
    rag_pipeline,
)
from src.vectorstore import (
    get_embeddings,
    load_vectorstore,
)


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


class FastEmbedRagasAdapter(BaseRagasEmbedding):
    """
    Adaptador para reutilizar los embeddings locales
    del asistente durante la evaluación Ragas.
    """

    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return self.embedding_model.embed_query(text)

    async def aembed_text(
        self,
        text: str,
        **kwargs,
    ) -> list[float]:
        return await asyncio.to_thread(
            self.embed_text,
            text,
        )


def save_json(path: Path, payload):
    """Guarda resultados en JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def create_evaluator():
    """
    Configura Ragas utilizando Groq mediante
    la API compatible con OpenAI.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "No se encontró GROQ_API_KEY."
        )

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=60.0,
        max_retries=2,
    )

    judge = llm_factory(
        config.GROQ_MODEL,
        provider="openai",
        client=client,
    )

    print(
        "[RAGAS] Evaluador configurado mediante Groq.",
        flush=True,
    )

    return judge, client


def create_metrics(judge, embeddings):
    """
    Inicializa las cuatro métricas requeridas.
    """

    ragas_embeddings = FastEmbedRagasAdapter(
        embeddings
    )

    return {
        "faithfulness": Faithfulness(
            llm=judge,
        ),

        "answer_relevancy": AnswerRelevancy(
            llm=judge,
            embeddings=ragas_embeddings,
        ),

        "context_precision": ContextPrecisionWithReference(
            llm=judge,
        ),

        "context_recall": ContextRecall(
            llm=judge,
        ),
    }


async def score_one(metrics, row):
    """
    Evalúa una respuesta con las cuatro métricas.
    Los casos fuera del corpus se analizan aparte.
    """

    scores = {
        "id": row["id"],
    }

    if row["expected_behavior"] != "answer":
        for name in METRIC_NAMES:
            scores[name] = None

        return scores

    inputs = {
        "user_input": row["question"],
        "response": row["answer"],
        "reference": row["ground_truth"],
        "retrieved_contexts": row["contexts"],
    }

    for name, metric in metrics.items():
        try:
            result = await metric.ascore(**inputs)

            value = float(result.value)

            if not math.isfinite(value):
                raise ValueError(
                    "La métrica devolvió un valor no finito."
                )

            scores[name] = value

            print(
                f"[RAGAS] {row['id']} - "
                f"{name}: {value:.4f}",
                flush=True,
            )

        except Exception as exc:
            scores[name] = None

            print(
                f"[ERROR] Métrica {name} "
                f"en pregunta {row['id']}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    return scores


async def evaluate_configuration(
    top_k,
    questions,
    vectorstore,
    generator,
    template,
    metrics,
):
    """
    Ejecuta las preguntas con una configuración
    determinada de recuperación.
    """

    print(
        f"\n[RAGAS] Iniciando evaluación top_k={top_k}",
        flush=True,
    )

    rows = []

    for i, item in enumerate(
        questions,
        start=1,
    ):
        print(
            f"[k={top_k}] Pregunta "
            f"{i}/{len(questions)}: {item['id']}",
            flush=True,
        )

        result = await asyncio.to_thread(
            rag_pipeline,
            vectorstore,
            generator,
            template,
            item["user_input"],
            top_k,
            [],
        )

        rows.append({
            "id": item["id"],
            "question": item["user_input"],
            "answer": result["respuesta"],
            "contexts": [
                document.page_content
                for document in result["fragmentos"]
            ],
            "ground_truth": item["reference"],
            "expected_behavior": item["expected_behavior"],
            "source_document": item.get(
                "source_document"
            ),
            "source_page": item.get(
                "source_page"
            ),
        })

    # Guardar respuestas antes de calcular métricas.
    save_json(
        OUTPUT_DIR / f"respuestas_k{top_k}.json",
        rows,
    )

    score_rows = []

    for i, row in enumerate(
        rows,
        start=1,
    ):
        print(
            f"[k={top_k}] Evaluando métricas "
            f"{i}/{len(rows)}: {row['id']}",
            flush=True,
        )

        scores = await score_one(
            metrics,
            row,
        )

        score_rows.append(scores)

        # Pausa pequeña para reducir ráfagas
        # de solicitudes al evaluador.
        await asyncio.sleep(1)

    # Guardar resultados individuales.
    csv_path = (
        OUTPUT_DIR / f"metricas_k{top_k}.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "id",
                *METRIC_NAMES,
            ),
        )

        writer.writeheader()
        writer.writerows(score_rows)

    eligible = sum(
        row["expected_behavior"] == "answer"
        for row in rows
    )

    summary = {
        "top_k": top_k,
        "total_preguntas": len(rows),
        "evaluables": eligible,
        "metricas": {},
    }

    incomplete = False

    for metric in METRIC_NAMES:
        values = [
            row[metric]
            for row in score_rows
            if row[metric] is not None
        ]

        summary["metricas"][metric] = {
            "promedio": (
                round(
                    sum(values) / len(values),
                    4,
                )
                if values
                else None
            ),
            "evaluadas": len(values),
            "esperadas": eligible,
        }

        if len(values) != eligible:
            incomplete = True

    # Evaluación independiente de abstenciones.
    abstentions = [
        row
        for row in rows
        if row["expected_behavior"] == "abstain"
    ]

    summary["abstenciones_correctas"] = sum(
        row["answer"].strip() == NO_ENCONTRADO
        for row in abstentions
    )

    summary["total_fuera_de_alcance"] = len(
        abstentions
    )

    save_json(
        OUTPUT_DIR / f"resumen_k{top_k}.json",
        summary,
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    return summary, incomplete


async def main():
    """Ejecuta la evaluación completa del RAG."""

    load_dotenv(ROOT / ".env")

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "Falta configurar GROQ_API_KEY."
        )

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"No existe el conjunto de evaluación: "
            f"{DATASET_PATH}"
        )

    questions = json.loads(
        DATASET_PATH.read_text(
            encoding="utf-8"
        )
    )

    if len(questions) < 15:
        raise ValueError(
            "El conjunto debe contener mínimo 15 preguntas."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[RAGAS] Cargando embeddings y ChromaDB...",
        flush=True,
    )

    embeddings = get_embeddings()

    vectorstore = load_vectorstore(
        embeddings
    )

    generator = get_llm()
    template = get_prompt_template()

    judge, judge_client = create_evaluator()

    metrics = create_metrics(
        judge,
        embeddings,
    )

    summaries = []
    incomplete = False

    try:
        for top_k in (5, 7):
            summary, partial = await evaluate_configuration(
                top_k,
                questions,
                vectorstore,
                generator,
                template,
                metrics,
            )

            summaries.append(summary)

            if partial:
                incomplete = True

    finally:
        await judge_client.close()

    save_json(
        OUTPUT_DIR / "comparacion.json",
        summaries,
    )

    if incomplete:
        raise RuntimeError(
            "Evaluación parcial: algunas métricas "
            "fallaron. Revisa los CSV y los logs."
        )

    print(
        "[OK] Evaluación Ragas finalizada correctamente.",
        flush=True,
    )

    print(
        "[OK] Resultados disponibles en "
        "evaluacion/resultados/",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
