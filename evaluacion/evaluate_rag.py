"""Evaluacion Ragas del RAG de OficinaPro. Ejecutar desde raiz del repositorio."""
import json
from pathlib import Path

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from src.rag_chain import get_llm, get_prompt_template, rag_pipeline
from src.vectorstore import get_embeddings, load_vectorstore

load_dotenv()
DATASET_PATH = Path('evaluacion/preguntas_ragas.json')
OUTPUT_DIR = Path('evaluacion/resultados')
METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def main():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f'No existe {DATASET_PATH}. Copia ahi preguntas_ragas.json')
    questions = json.loads(DATASET_PATH.read_text(encoding='utf-8'))
    assert len(questions) >= 15, 'Se requieren al menos 15 preguntas'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings()
    vectorstore = load_vectorstore(embeddings)
    llm = get_llm()
    prompt = get_prompt_template()

    for top_k in (5, 7):
        samples = []
        for number, item in enumerate(questions, 1):
            print(f'[top_k={top_k}] {number}/{len(questions)}: {item["id"]}', flush=True)
            result = rag_pipeline(
                vectorstore, llm, prompt, item['user_input'], k=top_k,
                historial=[],
            )
            samples.append({
                'question': item['user_input'],
                'answer': result['respuesta'],
                'contexts': [doc.page_content for doc in result['fragmentos']],
                'ground_truth': item['reference'],
                'id': item['id'],
                'expected_behavior': item['expected_behavior'],
                'source_document': item['source_document'],
                'source_page': item['source_page'],
            })
        # Guardar respuestas ANTES de ejecutar evaluadores, para conservar evidencia.
        (OUTPUT_DIR / f'respuestas_k{top_k}.json').write_text(
            json.dumps(samples, ensure_ascii=False, indent=2), encoding='utf-8'
        )

        # Las metricas de precision/recall de contexto no siempre son interpretables
        # para preguntas sin respaldo; las analizamos aparte como abstenciones.
        metric_data = [
            {k: row[k] for k in ('question', 'answer', 'contexts', 'ground_truth')}
            for row in samples if row['expected_behavior'] == 'answer'
        ]
        scores = evaluate(
            Dataset.from_list(metric_data),
            metrics=METRICS,
            llm=LangchainLLMWrapper(llm),
            embeddings=LangchainEmbeddingsWrapper(embeddings),
            raise_exceptions=False,
        )
        score_df = scores.to_pandas()
        score_df.to_csv(OUTPUT_DIR / f'metricas_k{top_k}.csv', index=False)
        abstain = [row for row in samples if row['expected_behavior'] == 'abstain']
        expected_text = 'No encontré esto en los manuales. Te recomiendo escalar el caso al equipo correspondiente.'
        abstain_ok = sum(row['answer'].strip() == expected_text for row in abstain)
        summary = {
            'top_k': top_k,
            'metricas': {
                m: float(score_df[m].mean()) for m in
                ('faithfulness', 'answer_relevancy', 'context_precision', 'context_recall')
            },
            'abstenciones_correctas': abstain_ok,
            'total_fuera_de_alcance': len(abstain),
        }
        (OUTPUT_DIR / f'resumen_k{top_k}.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    print('Evaluacion terminada. Revisa evaluacion/resultados/.')


if __name__ == '__main__':
    main()
