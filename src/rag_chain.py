"""
Construcción del prompt aumentado y pipeline RAG (retrieval + generación con Groq).
"""
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from . import config

PROMPT_TEMPLATE_RAW = """Eres el {assistant_name} de {empresa}, una empresa del sector financiero.
Tu trabajo es ayudar a los agentes de soporte técnico a resolver solicitudes rápido, usando ÚNICAMENTE la información de los manuales internos que se te entregan como contexto.

Reglas:
- Responde de forma clara, breve y accionable (pasos numerados si aplica).
- Al final de cada dato que uses, cita la fuente entre paréntesis, ej: (Fuente: manual_tarjetas.pdf, Pág. 12).
- Si la información no está en el contexto, responde exactamente: "No encontré esto en los manuales. Te recomiendo escalar el caso al equipo correspondiente."
- No inventes procedimientos ni políticas que no estén en el contexto.

Contexto recuperado de los manuales:
{context}

Pregunta del agente de soporte: {question}

Respuesta:"""


def get_llm() -> ChatGroq:
    return ChatGroq(
        model=config.GROQ_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=config.GROQ_API_KEY,
    )


def get_prompt_template() -> ChatPromptTemplate:
    filled = PROMPT_TEMPLATE_RAW.format(
        assistant_name=config.ASSISTANT_NAME,
        empresa=config.EMPRESA,
        context="{context}",
        question="{question}",
    )
    return ChatPromptTemplate.from_template(filled)


def _build_context(docs: list) -> str:
    return "\n\n---\n\n".join(
        f"[Fuente: {os.path.basename(d.metadata.get('source', '?'))} — Pág. {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


def rag_pipeline(vector_store, llm, prompt_template, pregunta: str, k: int = config.TOP_K) -> dict:
    """Pipeline RAG de extremo a extremo: recuperación -> prompt aumentado -> respuesta."""
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.invoke(pregunta)
    contexto = _build_context(docs)
    prompt = prompt_template.invoke({"context": contexto, "question": pregunta})
    respuesta = llm.invoke(prompt).content

    return {
        "pregunta": pregunta,
        "fragmentos": docs,
        "contexto": contexto,
        "respuesta": respuesta,
    }
