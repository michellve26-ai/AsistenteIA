"""Comprobacion reproducible del conjunto y corpus; no crea ni modifica indices."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluacion" / "preguntas_ragas.json"
PDF_DIR = ROOT / os.getenv("AVANCE3_PDF_DIR", "data/pdfs")
CORPUS_REF = os.getenv("AVANCE3_CORPUS_REF", "working-tree")
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
    # Nombre real historico BaseDeConociemiento.pdf; dataset BaseDeConocimiento.pdf.
    aliases = {"basedeconocimiento.pdf": "basedeconociemiento.pdf"}
    matched = {name: name if name.casefold() in present else aliases.get(name.casefold())
               for name in reference_names}
    missing = [name for name, candidate in matched.items() if not candidate or candidate.casefold() not in present]
    manifest = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_ref": CORPUS_REF,
        "directorio_corpus": str(PDF_DIR.relative_to(ROOT)),
        "aliases_referencias": aliases,
        "dataset_sha256": hash_file(DATASET),
        "preguntas": len(questions),
        "distribucion": counts,
        "top_k": 5,
        "pdfs_actuales": [{"archivo": p.name, "sha256": hash_file(p)} for p in pdfs],
        "referencias_documentales_no_presentes": missing,
        "comparable_con_metricas_historicas": (not bool(missing) and CORPUS_REF != "working-tree"),
        "comparacion_pareada_actual_permitida": (valid and not missing),
        "advertencia": (
            "La referencia documental no coincide con el corpus versionado."
            if missing else ("Corpus historico recuperado de Git. Comparar fijo y agente "
                             "con las mismas 15 preguntas. Los promedios antiguos "
                             "no sustituyen la reevaluacion.")
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        previous = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if (previous.get("dataset_sha256") != manifest["dataset_sha256"]
                or previous.get("corpus_ref") != manifest["corpus_ref"]
                or previous.get("pdfs_actuales") != manifest["pdfs_actuales"]):
            raise SystemExit(
                "Los hashes del corpus o las preguntas cambiaron. "
                "No reutilizar resultados de una ejecucion anterior."
            )
    OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "preguntas": manifest["preguntas"],
        "distribucion": counts,
        "pdfs_detectados": len(pdfs),
        "referencias_faltantes": missing,
        "comparacion_pareada_actual_permitida": valid,
        "archivo_resultado": str(OUTPUT),
    }, ensure_ascii=False, indent=2))
    if not valid or missing:
        raise SystemExit("PREFLIGHT FALLIDO: no se ejecutaran evaluaciones.")


if __name__ == "__main__":
    main()
