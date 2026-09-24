"""
Interfaz de chat — Asistente de Soporte Técnico (RAG)
Ejecutar con: streamlit run app.py
"""
import streamlit as st

from src import config
from src.ingest import load_pdfs, split_documents
from src.vectorstore import get_embeddings, build_vectorstore, load_vectorstore, index_exists
from src.rag_chain import get_llm, get_prompt_template, rag_pipeline

st.set_page_config(page_title=config.ASSISTANT_NAME, page_icon="💬", layout="centered")

st.markdown(
    """
    <style>
    .source-tag {
        display: inline-block; background: #eef2ff; color: #3730a3;
        padding: 2px 8px; border-radius: 999px; font-size: 0.75rem; margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(f"💬 {config.ASSISTANT_NAME}")
st.caption(f"Consulta los manuales internos de soporte de {config.EMPRESA} en lenguaje natural.")


# ---------- Carga de recursos (cacheados entre mensajes) ----------
@st.cache_resource(show_spinner="Cargando modelo de embeddings...")
def _get_embeddings():
    return get_embeddings()


@st.cache_resource(show_spinner="Cargando modelo de lenguaje (Groq)...")
def _get_llm():
    return get_llm()


@st.cache_resource(show_spinner="Cargando base vectorial...")
def _get_vectorstore(_embeddings):
    return load_vectorstore(_embeddings)


embeddings = _get_embeddings()
llm = _get_llm()
prompt_template = get_prompt_template()

# ---------- Sidebar: gestión del índice ----------
with st.sidebar:
    st.header("⚙️ Base de conocimiento")
    st.write(f"Carpeta de manuales: `{config.PDF_DIR}`")

    if index_exists():
        st.success("Índice vectorial encontrado.")
    else:
        st.warning("Aún no hay índice vectorial. Indexa los manuales para empezar.")

    if st.button("🔄 (Re)indexar manuales PDF"):
        with st.spinner("Cargando y procesando PDFs..."):
            documents = load_pdfs()
            chunks = split_documents(documents)
            build_vectorstore(chunks, embeddings, rebuild=True)
        st.cache_resource.clear()
        st.success("Índice actualizado. Recarga la página para usarlo.")

    st.divider()
    top_k = st.slider("Fragmentos a recuperar (k)", 2, 10, config.TOP_K)
    if st.button("🧹 Borrar historial del chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Estado del chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Entrada del usuario ----------
pregunta = st.chat_input("Escribe el problema o la duda del cliente/agente...")

if pregunta:
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        if not index_exists():
            respuesta_txt = (
                "Aún no hay manuales indexados. Ve al panel lateral y pulsa "
                "**(Re)indexar manuales PDF**."
            )
            st.markdown(respuesta_txt)
        else:
            with st.spinner("Buscando en los manuales..."):
                vector_store = _get_vectorstore(embeddings)
                resultado = rag_pipeline(vector_store, llm, prompt_template, pregunta, k=top_k)
                respuesta_txt = resultado["respuesta"]

            st.markdown(respuesta_txt)

            with st.expander(f"📎 Fuentes consultadas ({len(resultado['fragmentos'])})"):
                for i, doc in enumerate(resultado["fragmentos"], 1):
                    fuente = doc.metadata.get("source", "?").split("/")[-1]
                    pagina = doc.metadata.get("page", "?")
                    st.markdown(f"**[{i}] {fuente} — Pág. {pagina}**")
                    st.caption(doc.page_content[:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": respuesta_txt})
