"""Resume las evidencias actuales sin presentar metricas ausentes como reales."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "evaluacion" / "resultados_avance3"


def read_json(name, default):
    path = FOLDER / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def metrics(name):
    rows = read_json(f"{name}/resumen_k5.json", {})
    return rows.get("metricas", {})


def main() -> None:
    FOLDER.mkdir(parents=True, exist_ok=True)
    preflight = read_json("preflight.json", {})
    cases = read_json("casos_comportamiento.json", [])
    fixed = metrics("fijo")
    agent = metrics("agente")
    cost_path = FOLDER / "costos_comparados.csv"
    costs = []
    if cost_path.exists():
        with cost_path.open(newline="", encoding="utf-8") as handle:
            costs = list(csv.DictReader(handle))
    lines = [
        "# Astrea - Estado verificado del Avance 3",
        "",
        "Generado: " + datetime.now(timezone.utc).isoformat(),
        "",
        "## Corpus y comparabilidad",
        f"- Preguntas: {preflight.get('preguntas', 'sin ejecutar')}",
        f"- Referencias que no coinciden con PDFs versionados: {preflight.get('referencias_documentales_no_presentes', 'pendiente')}",
        "- Las metricas del Avance 2 historico NO son directamente comparables si el corpus ha cambiado.",
        "",
        "## Casos de comportamiento",
        f"- Casos ejecutados: {len(cases)} de 7",
        f"- Casos con comprobaciones automaticas preliminares positivas: {sum(x.get('cumple_preliminar', False) for x in cases)}",
        "- El cumplimiento final necesita revision humana de las respuestas y fuentes.",
        "",
        "## Metricas Ragas sobre corpus pareado actual",
        "| Metrica | RAG fijo (re-evaluado) | Agente | Diferencia |",
        "|---|---:|---:|---:|",
    ]
    for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        x, y = fixed.get(metric, {}), agent.get(metric, {})
        a, b = x.get("promedio"), y.get("promedio")
        delta = round(b - a, 4) if a is not None and b is not None else None
        show = lambda value: str(value) if value is not None else "Pendiente"
        lines.append(f"| {metric} | {show(a)} ({x.get('evaluadas', 0)}/12) | {show(b)} ({y.get('evaluadas', 0)}/12) | {show(delta)} |")
    lines.extend(["", "## Costos promedios sobre mismas preguntas"])
    for mode in ("Avance 2 fijo", "Avance 3 agente"):
        entries = [r for r in costs if r["modo"] == mode]
        if entries:
            means = {
                name: round(sum(float(r[name]) for r in entries) / len(entries), 3)
                for name in ("llamadas_llm", "tokens", "tiempo_s")
            }
            lines.append(f"- {mode}: {len(entries)} preguntas; {means}")
        else:
            lines.append(f"- {mode}: pendiente de ejecucion")
    lines += ["", "## Evidencias que faltan", "- Capturas reales de saludo, PDF, calculo y web desde Render.", "- Revision de las siete respuestas reales y excepciones.", "- Exportar PDF completando la plantilla oficial sin inventar datos."]
    out = FOLDER / "resumen_avance3.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
