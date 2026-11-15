"""Comprobacion reproducible del conjunto y corpus; no crea ni modifica indices."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluacion" / "preguntas_ragas.json"
PDF_DIR = ROOT / "data" / "pdfs"
OUTPUT = ROOT / "evaluacion" / "resultados_avance3" / "preflight.json"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    questions = json.loads(DATASET.read_text(encoding="utf-8"))
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    ids = [entry.get("id") for entry in questions]
    counts = {
        "respondibles": sum(x.get("expected_behavior") == "answer" for x in questions),
        "abstenciones": sum(x.get("expected_behavior") == "abstain" for x in questions),
    }
    valid = (
        len(questions) == 15
        and len(ids) == len(set(ids))
        and counts == {"respondibles": 12, "abstenciones": 3}
        and bool(pdfs)
    )
    reference_names = sorted({
        str(entry["source_document"])
        for entry in questions if entry.get("source_document")
    })
    present = {p.name.casefold() for p in pdfs}
    missing = [name for name in reference_names if name.casefold() not in present]
    manifest = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hash_file(DATASET),
        "preguntas": len(questions),
        "distribucion": counts,
        "top_k": 5,
        "pdfs_actuales": [{"archivo": p.name, "sha256": hash_file(p)} for p in pdfs],
        "referencias_documentales_no_presentes": missing,
        "comparable_con_metricas_historicas": not bool(missing),
        "comparacion_pareada_actual_permitida": valid,
        "advertencia": (
            "Las metricas historicas no son comparables directamente: los nombres de "
            "documentos de referencia no coinciden con el corpus actual. "
            "Evaluar RAG fijo y agente desde cero sobre los mismos PDF."
            if missing else "Verificar hashes historicos antes de afirmar comparabilidad."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "preguntas": manifest["preguntas"],
        "distribucion": counts,
        "pdfs_detectados": len(pdfs),
        "referencias_faltantes": missing,
        "comparacion_pareada_actual_permitida": valid,
        "archivo_resultado": str(OUTPUT),
    }, ensure_ascii=False, indent=2))
    if not valid:
        raise SystemExit("PREFLIGHT FALLIDO: no se ejecutaran evaluaciones.")


if __name__ == "__main__":
    main()
