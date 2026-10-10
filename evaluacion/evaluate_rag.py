"""Evaluacion Ragas 0.4 reanudable (k=5 y k=7).

Los resultados se restauran desde un artefacto previo de GitHub Actions.
No reemplazar ni cambiar corpus, modelo de embeddings o ground truths entre ejecuciones.
"""
import asyncio
import csv
import json
import math
import os
import re
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
import instructor
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorLLM, InstructorModelArgs
from ragas.metrics.collections import (
    AnswerRelevancy, ContextPrecisionWithReference, ContextRecall, Faithfulness,
)

from src import config
from src.rag_chain import get_llm, get_prompt_template, rag_pipeline
from src.vectorstore import get_embeddings, load_vectorstore

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evaluacion" / "preguntas_ragas.json"
OUTPUT_DIR = ROOT / "evaluacion" / "resultados"
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
NO_ENCONTRADO = (
    "No encontré esto en los manuales. "
    "Te recomiendo escalar el caso al equipo correspondiente."
)
# Numero maximo de operaciones (generaciones O metricas) por ejecucion.
MAX_OPERATIONS = int(os.getenv("RAGAS_MAX_OPERATIONS", "100"))
K_VALUES = tuple(int(x.strip()) for x in os.getenv("RAGAS_K_VALUES", "5").split(",") if x.strip())
MAX_RATE_RETRIES = int(os.getenv("RAGAS_MAX_RATE_RETRIES", "80"))
MAX_WAIT_SECONDS = float(os.getenv("RAGAS_MAX_WAIT_SECONDS", "3600"))
JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", config.GROQ_MODEL)
JUDGE_MODE = os.getenv("RAGAS_JUDGE_MODE", "MD_JSON").strip().upper()
JUDGE_MAX_TOKENS = int(os.getenv("RAGAS_JUDGE_MAX_TOKENS", "4096"))
INTEGRATION_VERSION = "instructor_groq_extra_body_v2"
REASONING_EFFORT = os.getenv("RAGAS_REASONING_EFFORT", "low").lower()
PAUSE_SECONDS = float(os.getenv("RAGAS_PAUSE_SECONDS", "15"))
RETRY_JSON_ERRORS = os.getenv("RAGAS_RETRY_JSON_ERRORS", "false").lower() == "true"
ERRORS_PATH = OUTPUT_DIR / "errores_json_ultima_ejecucion.json"
# Filtro opcional para reevaluaciones dirigidas (no genera ni puntua otros IDs).
ONLY_IDS = {x.strip() for x in os.getenv("RAGAS_ONLY_IDS", "").split(",") if x.strip()}
ONLY_METRICS = {x.strip() for x in os.getenv("RAGAS_ONLY_METRICS", "").split(",") if x.strip()}



class GroqInstructorLLM(InstructorLLM):
    """Send Groq GPT-OSS reasoning options via OpenAI extra_body.

    Use provider-specific mapping rather than passing mode or other
    arguments through llm_factory (which breaks in 0.4.3).
    """

    def _map_openai_params(self):
        args = super()._map_openai_params()
        if self.model.startswith("openai/gpt-oss-"):
            # Groq GPT-OSS does not support reasoning_format. The OpenAI
            # Python client accepts Groq-specific settings via extra_body.
            body = dict(args.get("extra_body") or {})
            body["reasoning_effort"] = REASONING_EFFORT
            body["include_reasoning"] = False
            args["extra_body"] = body
        return args


class FastEmbedRagasAdapter(BaseRagasEmbedding):
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return self.embedding_model.embed_query(text)

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return await asyncio.to_thread(self.embed_text, text)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def read_json(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def read_scores(path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        result = {}
        for row in csv.DictReader(f):
            result[row["id"]] = {
                key: parse_score(row.get(key)) for key in METRIC_NAMES
            }
        return result


def parse_score(value):
    if value in (None, "", "None", "nan"):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def write_scores(path, ids, scores):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("id", *METRIC_NAMES))
        writer.writeheader()
        for qid in ids:
            if qid in scores:
                writer.writerow({"id": qid, **scores[qid]})
    temp.replace(path)


def metric_inputs(name, row):
    inputs = {"user_input": row["question"]}
    if name == "faithfulness":
        return {**inputs, "response": row["answer"], "retrieved_contexts": row["contexts"]}
    if name == "answer_relevancy":
        return {**inputs, "response": row["answer"]}
    return {**inputs, "reference": row["ground_truth"], "retrieved_contexts": row["contexts"]}


def quota_delay_seconds(error):
    text = str(error)
    # Esperas anunciadas por Groq en minutos, segundos u horas.
    m = re.search(r"try again in\s*(?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", text, re.I)
    if m:
        return 3600 * int(m.group(1) or 0) + 60 * int(m.group(2) or 0) + float(m.group(3))
    return None


def is_json_error(error):
    msg = str(error).lower()
    return ("json_validate_failed" in msg or "failed to validate json" in msg or
            "instructorretryexception" in type(error).__name__.lower() or
            "incompleteoutputexception" in type(error).__name__.lower() or
            "output is incomplete" in msg)


def is_quota_error(error):
    return getattr(error, "status_code", None) == 429 or "429" in str(error) and "rate" in str(error).lower()


def summarize(k, rows, scores, expected_ids):
    answer_rows = [r for r in rows if r["expected_behavior"] == "answer"]
    out = {
        "top_k": k, "preguntas_respondidas": len(rows),
        "total_preguntas": len(expected_ids),
        "evaluables": len(answer_rows),
        "metricas": {},
        "abstenciones_correctas": sum(
            r["answer"].strip() == NO_ENCONTRADO
            for r in rows if r["expected_behavior"] == "abstain"
        ),
        "total_fuera_de_alcance": sum(r["expected_behavior"] == "abstain" for r in rows),
    }
    for name in METRIC_NAMES:
        values = [scores.get(r["id"], {}).get(name) for r in answer_rows]
        valid = [v for v in values if v is not None]
        out["metricas"][name] = {
            "promedio": round(sum(valid) / len(valid), 4) if valid else None,
            "evaluadas": len(valid), "esperadas": sum(r["expected_behavior"] == "answer" for r in expected_ids),
        }
    return out


async def main():
    load_dotenv(ROOT / ".env")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Falta GROQ_API_KEY")
    questions = read_json(DATASET_PATH, [])
    if len(questions) < 15 or len(set(q["id"] for q in questions)) != len(questions):
        raise RuntimeError("Se necesitan >=15 preguntas con ids distintos")
    if not K_VALUES or any(k not in (5, 7) for k in K_VALUES):
        raise ValueError("RAGAS_K_VALUES debe ser 5, 7 o 5,7")
    if MAX_OPERATIONS <= 0:
        raise ValueError("RAGAS_MAX_OPERATIONS debe ser positivo")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if ONLY_IDS and not ONLY_IDS.issubset({q["id"] for q in questions}):
        raise ValueError(f"RAGAS_ONLY_IDS desconocidos: {ONLY_IDS - {q['id'] for q in questions}}")
    if ONLY_METRICS and not ONLY_METRICS.issubset(set(METRIC_NAMES)):
        raise ValueError(f"RAGAS_ONLY_METRICS invalidas: {ONLY_METRICS - set(METRIC_NAMES)}")

    embeddings = get_embeddings()
    store = load_vectorstore(embeddings)
    generator = get_llm()
    template = get_prompt_template()
    client = AsyncOpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1", timeout=90, max_retries=0,
    )
    try:
        instructor_mode = getattr(instructor.Mode, JUDGE_MODE)
    except AttributeError as exc:
        raise ValueError(f"RAGAS_JUDGE_MODE no soportado: {JUDGE_MODE}") from exc
    # Ragas 0.4.3 no acepta mode= en llm_factory: lo pasa a la API como argumento.
    # Envolver primero el cliente OpenAI con Instructor y luego entregarlo a Ragas.
    structured_client = instructor.from_openai(client, mode=instructor_mode)
    judge = GroqInstructorLLM(
        client=structured_client, model=JUDGE_MODEL, provider="openai",
        model_args=InstructorModelArgs(max_tokens=JUDGE_MAX_TOKENS),
    )
    print(f"[RAGAS] Modelo evaluador: {JUDGE_MODEL}; modo: {JUDGE_MODE}; max_tokens: {JUDGE_MAX_TOKENS}; razonamiento: {REASONING_EFFORT}", flush=True)
    metrics = {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(
            llm=judge, embeddings=FastEmbedRagasAdapter(embeddings)
        ),
        "context_precision": ContextPrecisionWithReference(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }
    operations = 0
    blocked = False
    run_failures = read_json(ERRORS_PATH, [])
    if not isinstance(run_failures, list):
        raise ValueError("El archivo de errores JSON debe contener una lista")
    skip_json = {
        (r.get("top_k"), r.get("id"), r.get("metrica"), r.get("modo", "JSON"))
        for r in run_failures
        if r.get("error") == "json_validate_failed"
        and r.get("evaluador") == JUDGE_MODEL
        and r.get("modo", "JSON") == JUDGE_MODE
        and r.get("integracion") == INTEGRATION_VERSION
    }
    print(
        f"[REANUDAR] Errores JSON previos del evaluador actual: {len(skip_json)}; "
        f"reintentar={RETRY_JSON_ERRORS}", flush=True,
    )
    try:
        for k in K_VALUES:
            if blocked or operations >= MAX_OPERATIONS:
                break
            responses_path = OUTPUT_DIR / f"respuestas_k{k}.json"
            scores_path = OUTPUT_DIR / f"metricas_k{k}.csv"
            rows = read_json(responses_path, [])
            expected_ids = [q["id"] for q in questions]
            if [r["id"] for r in rows] != expected_ids[:len(rows)]:
                raise RuntimeError(f"respuestas_k{k} no coincide con el dataset actual")
            scores = read_scores(scores_path)
            saved_lookup = {r["id"]: r for r in rows}
            if ONLY_IDS and len(rows) != len(questions):
                raise RuntimeError("Modo dirigido requiere respuestas completas; no se regeneraran preguntas")
            try:
                # Generar solo respuestas faltantes; respetar el mismo orden.
                for item in questions[len(rows):]:
                    if operations >= MAX_OPERATIONS:
                        break
                    print(f"[k={k}] Generando {item['id']}", flush=True)
                    result = await asyncio.to_thread(
                        rag_pipeline, store, generator, template,
                        item["user_input"], k, [],
                    )
                    row = {
                        "id": item["id"], "question": item["user_input"],
                        "answer": str(result["respuesta"]),
                        "contexts": [d.page_content for d in result["fragmentos"]],
                        "ground_truth": item["reference"],
                        "expected_behavior": item["expected_behavior"],
                        "source_document": item.get("source_document"),
                        "source_page": item.get("source_page"),
                    }
                    rows.append(row)
                    saved_lookup[row["id"]] = row
                    save_json(responses_path, rows)
                    operations += 1
                    await asyncio.sleep(PAUSE_SECONDS)

                # No puntuar preguntas con respuesta pendiente hasta completar el conjunto.
                if len(rows) == len(questions):
                    # Priorizar operaciones nunca bloqueadas; reintentar formatos fallidos al final.
                    ordered_rows = sorted(
                        rows,
                        key=lambda r: sum(
                            (k, r["id"], metric, JUDGE_MODE) in skip_json
                            for metric in METRIC_NAMES
                        ),
                    )
                    for row in ordered_rows:
                        if operations >= MAX_OPERATIONS:
                            break
                        if row["expected_behavior"] != "answer":
                            continue
                        if ONLY_IDS and row["id"] not in ONLY_IDS:
                            continue
                        record = scores.setdefault(
                            row["id"], {name: None for name in METRIC_NAMES}
                        )
                        for name in METRIC_NAMES:
                            if operations >= MAX_OPERATIONS:
                                break
                            if ONLY_METRICS and name not in ONLY_METRICS:
                                continue
                            if record.get(name) is not None:
                                continue
                            key = (k, row["id"], name, JUDGE_MODE)
                            if key in skip_json and not RETRY_JSON_ERRORS:
                                print(
                                    f"[OMITIR JSON] k={k} {row['id']} {name}: "
                                    "ya fallo con este evaluador; sigue pendiente.", flush=True,
                                )
                                continue
                            print(f"[k={k}] {row['id']} {name}", flush=True)
                            operations += 1  # Contar también las operaciones que fallan.
                            try:
                                # Esperar automaticamente si la cuota por minuto/dia se agota.
                                for rate_attempt in range(MAX_RATE_RETRIES + 1):
                                    try:
                                        result = await metrics[name].ascore(**metric_inputs(name, row))
                                        break
                                    except Exception as rate_exc:
                                        if not is_quota_error(rate_exc):
                                            raise
                                        if rate_attempt == MAX_RATE_RETRIES:
                                            raise
                                        advertised = quota_delay_seconds(rate_exc)
                                        if advertised is None:
                                            advertised = 120
                                        wait = max(15.0, advertised + 10.0)
                                        if wait > MAX_WAIT_SECONDS:
                                            print(f"[CUOTA] Espera requerida {wait:.1f}s supera el maximo {MAX_WAIT_SECONDS:.1f}s", flush=True)
                                            raise
                                        print(f"[CUOTA] 429; espera automatica {wait:.1f}s; reintento {rate_attempt+1}/{MAX_RATE_RETRIES}", flush=True)
                                        await asyncio.sleep(wait)
                                val = float(result.value)
                                if not math.isfinite(val):
                                    raise RuntimeError(f"Metrica no finita: {name} {row['id']}")
                            except Exception as exc:
                                if is_quota_error(exc):
                                    raise  # Detener por cuota y conservar el avance.
                                if is_json_error(exc):
                                    # Fallo de salida estructurada: NO asignar 0 ni detener todo.
                                    failure = {
                                        "top_k": k, "id": row["id"], "metrica": name,
                                        "error": "json_validate_failed",
                                        "evaluador": JUDGE_MODEL, "modo": JUDGE_MODE,
                                        "integracion": INTEGRATION_VERSION,
                                    }
                                    if key not in skip_json:
                                        run_failures.append(failure)
                                        skip_json.add(key)
                                    print(f"[FORMATO] {row['id']} {name}: {type(exc).__name__}; sigue pendiente y se conservan resultados.", flush=True)
                                    save_json(ERRORS_PATH, run_failures)
                                    await asyncio.sleep(PAUSE_SECONDS)
                                    continue
                                raise
                            record[name] = val
                            if key in skip_json:
                                run_failures = [
                                    e for e in run_failures
                                    if (e.get("top_k"), e.get("id"), e.get("metrica"), e.get("modo", "JSON")) != key
                                    or e.get("evaluador") != JUDGE_MODEL
                                    or e.get("integracion") != INTEGRATION_VERSION
                                ]
                                skip_json.discard(key)
                                save_json(ERRORS_PATH, run_failures)
                            write_scores(scores_path, expected_ids, scores)
                            print(f"[k={k}] {row['id']} {name}={val:.4f}", flush=True)
                            await asyncio.sleep(PAUSE_SECONDS)
            except Exception as exc:
                if is_quota_error(exc):
                    wait = quota_delay_seconds(exc)
                    print(f"[CUOTA] Groq alcanzo el limite; esperar {wait or 'periodo de renovacion'} segundos y reanudar en otra ejecucion.", flush=True)
                    blocked = True
                else:
                    # No ocultar errores de programacion o de datos.
                    raise
            finally:
                write_scores(scores_path, expected_ids, scores)
                save_json(OUTPUT_DIR / f"resumen_k{k}.json", summarize(k, rows, scores, questions))
    finally:
        await client.close()

    comparisons = []
    for k in (5, 7):
        file = OUTPUT_DIR / f"resumen_k{k}.json"
        if file.exists():
            comparisons.append(read_json(file, {}))
    save_json(OUTPUT_DIR / "comparacion.json", comparisons)
    complete = all(
        (item := next((r for r in comparisons if r.get("top_k") == k), None)) is not None
        and item["preguntas_respondidas"] == len(questions)
        and all(v["evaluadas"] == v["esperadas"] for v in item["metricas"].values())
        for k in K_VALUES
    )
    print(f"[ESTADO] filtro ids={sorted(ONLY_IDS) if ONLY_IDS else 'todos'}; metricas={sorted(ONLY_METRICS) if ONLY_METRICS else 'todas'}", flush=True)
    print(f"[ESTADO] valores de k procesados={K_VALUES}", flush=True)
    print(
        f"[ESTADO] operaciones nuevas={operations}; "
        f"bloqueos_json={len(skip_json)}; evaluacion_completa={complete}", flush=True,
    )
    if not complete:
        print("[PENDIENTE] Descargar el artefacto y reanudar con su run ID en GitHub Actions.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
