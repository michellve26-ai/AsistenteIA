"""
Interfaz — Asistente de Soporte Técnico (RAG)
Ejecutar con: streamlit run app.py
"""

import html
import re

import streamlit as st

from src import config
from src.faqs import FAQS
from src.ingest import load_pdfs, split_documents
from src.vectorstore import (
    get_embeddings,
    build_vectorstore,
    load_vectorstore,
    index_exists,
)
from src.rag_chain import (
    get_llm,
    get_prompt_template,
    rag_pipeline,
)

st.set_page_config(
    page_title=f"{config.ASSISTANT_NAME} | OficinaPro",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.html("""
<style>
:root {
    --navy:#163B65;
    --navy-soft:#1F5F99;
    --blue:#2F80ED;
    --blue-light:#EEF5FC;
    --background:#F4F7FA;
    --surface:#FFFFFF;
    --surface-soft:#F8FAFC;
    --text:#172033;
    --muted:#65758B;
    --border:#D9E2EC;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--background);
}

[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    max-width: 1500px;
    padding-top: 1rem;
    padding-bottom: 3rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
}

.op-header {
    width: 100%;
    background: var(--navy);
    border-radius: 14px;
    padding: 17px 22px;
    margin-bottom: 22px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 20px;
    box-shadow: 0 5px 18px rgba(23,32,51,.08);
}

.op-brand {
    display:flex;
    align-items:center;
    gap:12px;
}

.op-logo {
    width:40px;
    height:40px;
    border-radius:10px;
    display:flex;
    align-items:center;
    justify-content:center;
    background:rgba(255,255,255,.12);
    color:#fff;
    font-size:20px;
    font-weight:800;
}

.op-brand-name {
    color:#fff;
    font-size:20px;
    font-weight:800;
    line-height:1.1;
}

.op-brand-name span {
    color:#79B8F3;
}

.op-brand-subtitle {
    color:rgba(255,255,255,.72);
    font-size:12px;
    margin-top:3px;
}

.op-status {
    color:rgba(255,255,255,.92);
    font-size:12px;
    display:flex;
    align-items:center;
    gap:7px;
}

.op-status-dot {
    width:8px;
    height:8px;
    border-radius:50%;
    background:#5DD49A;
}

.op-menu-title {
    font-size:11px;
    font-weight:800;
    color:var(--muted);
    letter-spacing:.09em;
    text-transform:uppercase;
    margin-bottom:9px;
}

.op-nav-info {
    margin-top:16px;
    padding:12px 13px;
    background:var(--surface-soft);
    border:1px solid var(--border);
    border-radius:10px;
    font-size:12px;
    color:var(--muted);
    line-height:1.5;
}

.op-nav-info strong {
    color:var(--text);
}

.op-page-heading {
    margin-bottom:18px;
}

.op-page-title {
    font-size:29px;
    font-weight:800;
    color:var(--text);
    line-height:1.15;
}

.op-page-description {
    margin-top:6px;
    font-size:14px;
    color:var(--muted);
    max-width:760px;
    line-height:1.55;
}

.stButton > button {
    width:100%;
    min-height:43px;
    background:var(--surface);
    color:var(--text);
    border:1px solid var(--border);
    border-radius:10px;
    font-weight:650;
    text-align:left;
    transition:transform .18s ease,border-color .18s ease,background .18s ease,box-shadow .18s ease;
}

.stButton > button:hover {
    transform:translateY(-1px);
    border-color:#9FC5ED;
    background:#F7FAFD;
    box-shadow:0 5px 14px rgba(23,59,101,.07);
}

.stButton > button[kind="primary"] {
    background:var(--navy);
    color:white;
    border-color:var(--navy);
}

.stButton > button[kind="primary"]:hover {
    background:var(--navy-soft);
    border-color:var(--navy-soft);
}

.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span {
    color:white !important;
}

div[data-testid="stTextInput"] input,
[data-testid="stChatInput"] textarea {
    background:white !important;
    border:1px solid #CBD6E2 !important;
    border-radius:11px !important;
    color:var(--text) !important;
}

div[data-testid="stTextInput"] input:focus,
[data-testid="stChatInput"] textarea:focus {
    border-color:var(--blue) !important;
    box-shadow:0 0 0 3px rgba(47,128,237,.08) !important;
}

[data-testid="stRadio"] > div {
    gap:7px;
}

[data-testid="stRadio"] label {
    background:white;
    border:1px solid var(--border);
    border-radius:999px;
    padding:5px 10px;
    transition:all .18s ease;
}

[data-testid="stRadio"] label:hover {
    border-color:#9FC5ED;
    background:var(--blue-light);
}

.faq-card {
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:12px;
    padding:14px 15px;
    margin-bottom:8px;
    transition:transform .18s ease,box-shadow .18s ease,border-color .18s ease;
}

.faq-card:hover {
    transform:translateY(-2px);
    border-color:#A7C7E8;
    box-shadow:0 7px 18px rgba(23,59,101,.07);
}

.faq-category {
    display:inline-block;
    background:var(--blue-light);
    color:#245E96;
    border-radius:999px;
    padding:3px 8px;
    font-size:10px;
    font-weight:750;
    margin-bottom:8px;
}

.faq-question {
    font-size:14px;
    font-weight:750;
    color:var(--text);
    line-height:1.4;
}

.faq-preview {
    margin-top:5px;
    color:var(--muted);
    font-size:12px;
    line-height:1.45;
}

.detail-title {
    margin-top:8px;
    font-size:21px;
    font-weight:800;
    color:var(--text);
    line-height:1.35;
}

.detail-source {
    margin-top:14px;
    padding-top:11px;
    border-top:1px solid #E9EEF4;
    color:var(--muted);
    font-size:11px;
}

.chat-intro {
    background:white;
    border:1px solid var(--border);
    border-radius:12px;
    padding:14px 16px;
    margin-bottom:18px;
    color:var(--muted);
    font-size:13px;
    line-height:1.55;
}

.chat-intro strong {
    display:block;
    color:var(--text);
    margin-bottom:3px;
    font-size:14px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background:white;
    border-color:var(--border) !important;
    border-radius:13px !important;
    box-shadow:0 3px 12px rgba(23,59,101,.035);
}

[data-testid="stChatMessage"] {
    background:white;
    border:1px solid #E1E8F0;
    border-radius:12px;
    padding:.65rem .8rem;
    margin-bottom:.7rem;
    box-shadow:0 2px 8px rgba(23,59,101,.025);
}

[data-testid="stSidebar"] {
    background:#FBFCFE;
    border-right:1px solid var(--border);
}
/* ---------------------------------------------------------
   LEGIBILIDAD GENERAL
--------------------------------------------------------- */

.stApp,
.stApp p,
.stApp span,
.stApp div,
.stApp label,
.stApp li {
    color:#172033;
}

[data-testid="stMarkdownContainer"] {
    color:#172033;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color:#172033 !important;
    opacity:1 !important;
}

[data-testid="stCaptionContainer"] {
    color:#65758B !important;
    opacity:1 !important;
}

[data-testid="stCaptionContainer"] p {
    color:#65758B !important;
    opacity:1 !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span {
    color:#172033 !important;
    opacity:1 !important;
}

[data-testid="stTextInput"] input {
    color:#172033 !important;
    opacity:1 !important;
}

[data-testid="stTextInput"] input::placeholder,
[data-testid="stChatInput"] textarea::placeholder {
    color:#7A889A !important;
    opacity:1 !important;
}

[data-testid="stRadio"] label p {
    color:#172033 !important;
    opacity:1 !important;
}

.stButton > button p,
.stButton > button span {
    color:#172033 !important;
    opacity:1 !important;
}

.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span {
    color:#FFFFFF !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color:#172033;
    opacity:1 !important;
}

/* ---------------------------------------------------------
   CABECERA - FORZAR TEXTO CLARO
--------------------------------------------------------- */

.op-header,
.op-header div,
.op-header span {
    color:#FFFFFF !important;
}

.op-brand-name {
    color:#FFFFFF !important;
}

.op-brand-name span {
    color:#79B8F3 !important;
}

.op-brand-subtitle {
    color:#D6E3F2 !important;
    opacity:1 !important;
}

.op-status {
    color:#EAF2FA !important;
    opacity:1 !important;
}

.op-status-dot {
    background:#5DD49A !important;
}
@media (max-width:900px) {
    .block-container {
        padding-left:.75rem;
        padding-right:.75rem;
    }

    .op-status {
        display:none;
    }

    .op-page-title {
        font-size:24px;
    }
}
</style>
""")


def faq_preview(answer: str) -> str:
    clean = re.sub(r"[#>*`]+", "", answer)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:115].rstrip() + "..." if len(clean) > 115 else clean


def category_icon(category: str) -> str:
    category = category.lower()
    if "factur" in category:
        return "🧾"
    if "invent" in category:
        return "📦"
    if "cartera" in category:
        return "💳"
    if "cliente" in category or "proveedor" in category:
        return "👥"
    if "banco" in category or "caja" in category:
        return "🏦"
    if "nota" in category:
        return "📄"
    return "📘"


def safe_index_exists() -> bool:
    try:
        return index_exists()
    except Exception:
        return False


if "messages" not in st.session_state:
    st.session_state.messages = []

if "main_view" not in st.session_state:
    st.session_state.main_view = "Soluciones rápidas"

if "selected_faq" not in st.session_state:
    st.session_state.selected_faq = 0

if "faq_category" not in st.session_state:
    st.session_state.faq_category = "Todas"


@st.cache_resource(show_spinner="Cargando modelo de embeddings...")
def _get_embeddings():
    return get_embeddings()


@st.cache_resource(show_spinner="Cargando modelo de lenguaje...")
def _get_llm():
    return get_llm()


@st.cache_resource(show_spinner="Cargando base vectorial...")
def _get_vectorstore(_embeddings):
    return load_vectorstore(_embeddings)


indice_actual = safe_index_exists()
estado_texto = (
    "Base de conocimiento activa"
    if indice_actual
    else "Base pendiente de indexación"
)

st.html(
    f'<div class="op-header">'
    f'<div class="op-brand">'
    f'<div class="op-logo">O</div>'
    f'<div>'
    f'<div class="op-brand-name">OFICINA<span>PRO.CO</span></div>'
    f'<div class="op-brand-subtitle">Asistente de soporte</div>'
    f'</div></div>'
    f'<div class="op-status">'
    f'<div class="op-status-dot"></div>'
    f'{html.escape(estado_texto)}'
    f'</div></div>'
)

with st.sidebar:
    st.header("Base de conocimiento")

    if safe_index_exists():
        st.success("Índice disponible")
    else:
        st.warning("Índice pendiente")

    st.caption(f"Directorio de manuales: `{config.PDF_DIR}`")

    top_k = st.slider(
        "Fragmentos a recuperar",
        min_value=2,
        max_value=10,
        value=config.TOP_K,
    )

    st.divider()

    if st.button("Reindexar manuales PDF", use_container_width=True):
        try:
            with st.spinner("Procesando documentos..."):
                embeddings = _get_embeddings()
                documents = load_pdfs()
                chunks = split_documents(documents)
                build_vectorstore(chunks, embeddings, rebuild=True)

            st.cache_resource.clear()
            st.success("Índice actualizado correctamente.")
        except Exception as exc:
            st.error("No fue posible indexar los manuales.")
            with st.expander("Detalle técnico"):
                st.code(str(exc))

    if st.button("Borrar historial del chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


nav_col, content_col = st.columns([1.15, 4.85], gap="large")

with nav_col:
    with st.container(border=True):
        st.html('<div class="op-menu-title">Soporte</div>')

        if st.button(
            "💬  Chat\n\nConsulta al asistente",
            type="primary" if st.session_state.main_view == "Chat" else "secondary",
            use_container_width=True,
            key="nav_chat",
        ):
            st.session_state.main_view = "Chat"
            st.rerun()

        if st.button(
            "📖  Soluciones rápidas\n\nCasos frecuentes",
            type="primary"
            if st.session_state.main_view == "Soluciones rápidas"
            else "secondary",
            use_container_width=True,
            key="nav_solutions",
        ):
            st.session_state.main_view = "Soluciones rápidas"
            st.rerun()

        estado_indice = (
            "Índice listo para consultar"
            if safe_index_exists()
            else "Pendiente de indexación"
        )

        st.html(
            f'<div class="op-nav-info">'
            f'<strong>Estado del conocimiento</strong><br>'
            f'{html.escape(estado_indice)}<br><br>'
            f'La administración técnica está disponible desde el menú lateral.'
            f'</div>'
        )

with content_col:
    if st.session_state.main_view == "Soluciones rápidas":
        st.html(
            '<div class="op-page-heading">'
            '<div class="op-page-title">Soluciones rápidas</div>'
            '<div class="op-page-description">'
            'Encuentra procedimientos frecuentes antes de realizar una consulta al asistente.'
            '</div></div>'
        )

        filtro = st.text_input(
            "Buscar",
            placeholder="Buscar una solución: inventario, factura, cartera, cliente...",
            label_visibility="collapsed",
            key="faq_search",
        ).strip().lower()

        categorias = ["Todas"] + sorted({faq["category"] for faq in FAQS})

        categoria_actual = st.radio(
            "Categorías",
            categorias,
            index=(
                categorias.index(st.session_state.faq_category)
                if st.session_state.faq_category in categorias
                else 0
            ),
            horizontal=True,
            label_visibility="collapsed",
            key="faq_categories",
        )

        st.session_state.faq_category = categoria_actual
        faqs_visibles = []

        for original_index, faq in enumerate(FAQS):
            coincide_categoria = (
                categoria_actual == "Todas"
                or faq["category"] == categoria_actual
            )

            contenido_busqueda = " ".join(
                [
                    faq.get("question", ""),
                    faq.get("category", ""),
                    faq.get("keywords", ""),
                    faq.get("answer", ""),
                ]
            ).lower()

            if coincide_categoria and (
                not filtro or filtro in contenido_busqueda
            ):
                faqs_visibles.append((original_index, faq))

        if not faqs_visibles:
            st.info("No encontré una solución rápida con ese término.")
        else:
            indices_visibles = [index for index, _ in faqs_visibles]

            if st.session_state.selected_faq not in indices_visibles:
                st.session_state.selected_faq = indices_visibles[0]

            list_col, detail_col = st.columns([2.1, 2.7], gap="medium")

            with list_col:
                with st.container(border=True):
                    st.html('<div class="op-menu-title">Casos frecuentes</div>')

                    for original_index, faq in faqs_visibles:
                        icon = category_icon(faq["category"])

                        st.html(
                            f'<div class="faq-card">'
                            f'<div class="faq-category">{html.escape(faq["category"])}</div>'
                            f'<div class="faq-question">'
                            f'{icon}&nbsp;{html.escape(faq["question"])}'
                            f'</div>'
                            f'<div class="faq-preview">'
                            f'{html.escape(faq_preview(faq["answer"]))}'
                            f'</div></div>'
                        )

                        selected = (
                            original_index == st.session_state.selected_faq
                        )

                        if st.button(
                            "Solución seleccionada" if selected else "Ver solución →",
                            key=f"faq_select_{original_index}",
                            type="primary" if selected else "secondary",
                            use_container_width=True,
                        ):
                            st.session_state.selected_faq = original_index
                            st.rerun()

            with detail_col:
                faq = FAQS[st.session_state.selected_faq]

                with st.container(border=True):
                    st.html(
                        f'<div class="faq-category">{html.escape(faq["category"])}</div>'
                        f'<div class="detail-title">{html.escape(faq["question"])}</div>'
                    )

                    st.caption(
                        "Procedimiento recomendado según la base de conocimiento disponible."
                    )
                    st.markdown(faq["answer"])

                    st.html(
                        f'<div class="detail-source">'
                        f'Fuente: {html.escape(faq.get("source", "Base de conocimiento"))}'
                        f'</div>'
                    )

                    st.write("")

                    if st.button(
                        "💬 Consultar este caso en el chat",
                        type="primary",
                        use_container_width=True,
                        key="faq_to_chat",
                    ):
                        st.session_state.pending_question = faq["question"]
                        st.session_state.main_view = "Chat"
                        st.rerun()

    else:
        st.html(
            '<div class="op-page-heading">'
            '<div class="op-page-title">Chat de soporte</div>'
            '<div class="op-page-description">'
            'Consulta los manuales y la base de conocimiento de OficinaPro mediante el asistente.'
            '</div></div>'
        )

        st.html(
            '<div class="chat-intro">'
            '<strong>¿Qué necesitas resolver?</strong>'
            'Describe el inconveniente tal como fue reportado '
            'por el cliente o el agente de soporte.'
            '</div>'
        )

        if not safe_index_exists():
            st.warning(
                "La base de conocimiento todavía no está indexada. "
                "Abre el menú lateral y usa 'Reindexar manuales PDF'."
            )

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        pregunta_pendiente = st.session_state.pop("pending_question", None)
        pregunta = pregunta_pendiente or st.chat_input(
            "Describe el problema o la duda..."
        )

        if pregunta:
            st.session_state.messages.append(
                {"role": "user", "content": pregunta}
            )

            with st.chat_message("user"):
                st.markdown(pregunta)

            with st.chat_message("assistant"):
                if not safe_index_exists():
                    respuesta_txt = (
                        "Aún no hay manuales indexados. "
                        "Abre la configuración lateral y reindexa los manuales PDF."
                    )
                    st.markdown(respuesta_txt)
                else:
                    try:
                        with st.spinner(
                            "Consultando la base de conocimiento..."
                        ):
                            embeddings = _get_embeddings()
                            llm = _get_llm()
                            prompt_template = get_prompt_template()
                            vector_store = _get_vectorstore(embeddings)

                            resultado = rag_pipeline(
                                vector_store,
                                llm,
                                prompt_template,
                                pregunta,
                                k=top_k,
                                historial=st.session_state.messages[:-1],
                            )

                            respuesta_txt = resultado["respuesta"]

                        st.markdown(respuesta_txt)

                        fragmentos = resultado.get("fragmentos", [])

                        if fragmentos:
                            with st.expander(
                                f"Fuentes consultadas ({len(fragmentos)})"
                            ):
                                for i, doc in enumerate(fragmentos, 1):
                                    fuente = (
                                        doc.metadata
                                        .get("source", "?")
                                        .replace("\\", "/")
                                        .split("/")[-1]
                                    )
                                    pagina = doc.metadata.get("page", "?")

                                    st.markdown(
                                        f"**[{i}] {fuente} — Pág. {pagina}**"
                                    )

                                    contenido = doc.page_content.strip()
                                    st.caption(
                                        contenido[:300]
                                        + ("..." if len(contenido) > 300 else "")
                                    )

                    except Exception as exc:
                        respuesta_txt = (
                            "No fue posible completar la consulta en este momento."
                        )
                        st.error(respuesta_txt)

                        with st.expander("Detalle técnico"):
                            st.code(str(exc))

            st.session_state.messages.append(
                {"role": "assistant", "content": respuesta_txt}
            )
