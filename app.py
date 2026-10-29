"""Interfaz de Astrea. Ejecutar con: streamlit run app.py."""

import html
import inspect
import re
from contextlib import nullcontext

import streamlit as st

from src import config
from src.faqs import FAQS
from src.ingest import load_pdfs, split_documents
from src.rag_chain import get_llm, get_prompt_template, rag_pipeline
from src.vectorstore import (
    build_vectorstore,
    get_embeddings,
    index_exists,
    load_vectorstore,
)


UI_NAME = "Astrea"

st.set_page_config(
    page_title=UI_NAME,
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="auto",
)

st.html("""
<style>
/* Astrea: presentación de Streamlit. Las interacciones son componentes nativos. */
.stApp {
  --astrea-ink: #1c3553;
  --astrea-muted: #5e748e;
  --astrea-line: #dbe6f2;
  --astrea-panel: #ffffff;
  --astrea-soft: #f0f6fc;
  --astrea-blue: #2365ad;
  --astrea-navy: #123253;
  color: var(--astrea-ink);
  background: radial-gradient(ellipse at 92% 8%, #d9ecff 0, transparent 52%), #f1f5fb;
}

[data-testid="stHeader"] { background: #f1f5fbf0; }
/* El botón móvil vive dentro del toolbar: solo ocultamos sus acciones técnicas. */
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"] { display: none; }
.stApp [data-testid="stMainBlockContainer"], .stApp .main .block-container {
  max-width: 1120px;
  padding: calc(60px + 1.5rem) 2rem 6rem;
}
.stMain [data-testid="stMarkdownContainer"],
.stMain [data-testid="stWidgetLabel"] { color: var(--astrea-ink); }
.stMain [data-testid="stCaptionContainer"] { color: var(--astrea-muted); }

.astrea-header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  padding: 2px 0 21px;
  margin-bottom: 24px;
  border-bottom: 1px solid #ccdef0;
}
.astrea-header h1 { margin: 0; padding: 0; font-size: 19px; line-height: 1.4; font-weight: 650; color: var(--astrea-ink); }
.astrea-header p { margin: 4px 0 0; color: var(--astrea-muted); font-size: 12px; }
.astrea-status {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 7px 11px;
  font-size: 12px;
  border-radius: 99px;
  background: var(--astrea-panel);
  border: 1px solid var(--astrea-line);
  box-shadow: 0 3px 10px #14385d08;
}
.astrea-status > span { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.astrea-status.is-ready { color: #237b59; }
.astrea-status.is-pending { color: #916b18; background: #fff8e7; border-color: #eddfb9; }

/* El sidebar nativo conserva su control de apertura en móviles. */
[data-testid="stSidebar"] {
  background: linear-gradient(160deg, #173e6b, #102d50 58%, #0c2441);
  color: #e8f1ff;
  border-right: 1px solid #153858;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: .4rem; }
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: .7rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stExpander"] summary { color: #c3d7ee; }
[data-testid="stSidebar"] [data-testid="stExpander"] { border-color: #ffffff26; border-radius: 13px; }
[data-testid="stSidebar"] [data-testid="stAlert"] { color: #1c3553; }
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button { color: #d6e7fa; }
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"] { color: #d6e7fa; }
.astrea-brand { display: flex; align-items: center; gap: 11px; margin: 5px 0 27px; }
.astrea-brand-mark {
  display: grid;
  place-items: center;
  width: 39px;
  height: 41px;
  flex: 0 0 auto;
  color: #fff;
  font-size: 24px;
  font-weight: 700;
  border-radius: 12px;
  border: 1px solid #ffffff40;
  background: linear-gradient(145deg, #4b96d3, #2163a3);
  box-shadow: 0 5px 16px #00182c38, inset 0 1px 0 #ffffff30;
}
.astrea-brand-name { color: #f6faff; font-size: 29px; font-weight: 700; line-height: 1.1; letter-spacing: -1px; }
.astrea-brand-description { color: #b6cfe8; margin-top: 4px; font-size: 12px; }
.astrea-menu-label { color: #92b1d0; font-size: 11px; font-weight: 600; letter-spacing: 1px; margin: 22px 0 8px; }
.astrea-sidebar-note { padding: 16px; margin: 25px 0 19px; background: #ffffff09; border: 1px solid #ffffff15; border-radius: 13px; }
.astrea-sidebar-note strong { color: #e4effd; font-size: 13px; font-weight: 500; }
.astrea-sidebar-note p { color: #a8bfd8; font-size: 12px; line-height: 1.6; margin: 5px 0 0; }
.astrea-sidebar-footer { color: #ddeafa; font-size: 14px; font-weight: 600; margin-top: 25px; padding-top: 18px; border-top: 1px solid #ffffff1c; }
.astrea-sidebar-footer span { display: block; font-size: 11px; font-weight: 400; color: #96b3d0; margin-top: 3px; }

.stButton > button {
  min-height: 43px;
  border: 1px solid var(--astrea-line);
  background: var(--astrea-panel);
  color: var(--astrea-ink);
  border-radius: 11px;
  padding: .6rem .9rem;
  transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
}
.stButton > button:hover { border-color: #8eb6de; transform: translateY(-2px); box-shadow: 0 7px 15px #153d6c14; color: #1c548e; }
.stButton > button:focus-visible { outline: 3px solid #75a9d6; outline-offset: 3px; }
.stButton > button[kind="primary"] { background: #205f9e; border-color: #205f9e; color: #ffffff; }
.stButton > button[kind="primary"] p { color: inherit; }
.stButton > button:disabled { transform: none; box-shadow: none; }
[data-testid="stSidebar"] .stButton > button { background: #ffffff08; border-color: #ffffff19; color: #d5e4f6; text-align: left; }
[data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] .stButton > button p { color: inherit; }
[data-testid="stSidebar"] .stButton > button:hover { background: #ffffff12; border-color: #ffffff30; color: #ffffff; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] { background: #edf5ff; border-color: #edf5ff; color: #1e4d7d; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] .stButton > button[kind="primary"] p { color: #1e4d7d; }
[data-testid="stSidebar"] .st-key-astrea_new_chat_area .stButton > button[kind="primary"] { background: linear-gradient(115deg,#2e79c8,#2165af); color: #ffffff; border-color: #4587c7; box-shadow: 0 5px 13px #001c3a40; }
[data-testid="stSidebar"] .st-key-astrea_new_chat_area .stButton > button[kind="primary"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] .st-key-astrea_new_chat_area .stButton > button[kind="primary"] p { color: #ffffff; }

.astrea-welcome { text-align: center; max-width: 670px; margin: 2rem auto 1.7rem; }
.astrea-welcome-mark {
  display: grid; place-items: center;
  margin: 0 auto 20px;
  width: 50px; height: 52px;
  border-radius: 16px;
  color: #fff; font-size: 29px; font-weight: 600;
  background: linear-gradient(145deg,#559fdd,#1c5fa1 65%,#154a83);
  border: 1px solid #75a9d6;
  box-shadow: 0 12px 25px #326b9e25, inset 0 1px 0 #ffffff66;
  transform: rotate(-6deg);
  animation: astrea-float 5s ease-in-out infinite;
}
.astrea-welcome > p { font-size: 14px; font-weight: 500; color: var(--astrea-blue); margin: 0 0 9px; }
.astrea-welcome h2 { color: var(--astrea-ink); font-size: clamp(27px,3.7vw,39px); line-height: 1.2; font-weight: 650; letter-spacing: -1.3px; margin: 0 0 13px; padding: 0; }
.astrea-welcome h2 span { color: var(--astrea-blue); }
.astrea-welcome > div:last-child { color: var(--astrea-muted); font-size: 14px; line-height: 1.6; }
.astrea-welcome-hint { text-align: center; font-size: 12px; color: var(--astrea-muted); margin: 13px 0 20px; }
[class*="st-key-astrea_topic_card_"] {
  background: var(--astrea-panel);
  border-color: var(--astrea-line);
  border-radius: 18px;
  box-shadow: 0 8px 22px #153c6610;
  transition: transform 220ms ease, box-shadow 220ms ease;
  animation: astrea-card-in 360ms ease-out backwards;
}
.st-key-astrea_topic_card_1 { animation-delay: 50ms; }
.st-key-astrea_topic_card_2 { animation-delay: 100ms; }
.st-key-astrea_topic_card_3 { animation-delay: 150ms; }
[class*="st-key-astrea_topic_card_"]:hover { transform: translateY(-5px); box-shadow: 0 15px 29px #163e6b1a; }
.astrea-topic-card { display: flex; align-items: center; gap: 12px; padding: 3px 0 5px; }
.astrea-topic-icon {
  display: grid; place-items: center;
  flex: 0 0 auto; width: 40px; height: 42px;
  border-radius: 12px; border: 1px solid #dbeafa;
  background: #edf5ff; color: #2668b0;
  font-size: 18px; font-weight: 600;
}
.astrea-topic-icon svg { display: block; width: 22px; height: 22px; }
.astrea-topic-card strong { font-size: 15px; color: var(--astrea-ink); font-weight: 600; }
.astrea-topic-card p { font-size: 12px; color: var(--astrea-muted); margin: 3px 0 0; }
[class*="st-key-astrea_topic_card_"] .stButton > button { color: #28649f; background: #f6faff; border-color: #e0ebf7; }
.astrea-topic-heading { color: var(--astrea-ink); font-size: 23px; font-weight: 600; margin-bottom: 12px; letter-spacing: -.4px; }

/* El diálogo y sus controles conservan el foco y el cierre nativos. */
[data-testid="stDialog"] [role="dialog"] { border-radius: 22px; border: 1px solid #dce7f3; box-shadow: 0 25px 70px #08244835; background: var(--astrea-panel); color: var(--astrea-ink); animation: astrea-card-in 180ms ease-out; }
[data-testid="stDialog"] [data-testid="stMarkdownContainer"],
[data-testid="stDialog"] [data-testid="stWidgetLabel"] { color: var(--astrea-ink); }

[data-testid="stChatMessage"] { border: 1px solid var(--astrea-line); background: var(--astrea-panel); border-radius: 7px 18px 18px 18px; padding: 1.15rem; margin-bottom: .8rem; box-shadow: 0 7px 20px #143b6310; overflow-wrap: anywhere; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: #e5effb; border-color: #d6e4f4; border-radius: 17px 17px 5px 17px; }
.stMain [data-testid="stChatMessageAvatarUser"] { background-color: #d2e3f8; color: #255d95; }
.stMain [data-testid="stChatMessageAvatarAssistant"] { background-color: #205f9e; color: #ffffff; }
.stMain [data-testid="stChatMessageAvatarUser"] svg,
.stMain [data-testid="stChatMessageAvatarAssistant"] svg { color: inherit; fill: currentColor; }
.astrea-message-name { font-weight: 600; color: var(--astrea-blue); font-size: 13px; margin-bottom: 7px; }
[data-testid="stChatMessage"] [data-testid="stExpander"] { background: #f7faff; border-color: #e2eaf3; border-radius: 10px; }
[data-testid="stChatInput"] { background: var(--astrea-panel); border: 1px solid #c9dcef; border-radius: 18px; box-shadow: 0 10px 25px #173f6e12; }
[data-testid="stChatInput"]:focus-within { border-color: #73a4d3; box-shadow: 0 12px 28px #173f6e20; }
[data-testid="stChatInput"] textarea { background: var(--astrea-panel); color: var(--astrea-ink); font-size: 16px; }
[data-testid="stChatInput"] textarea::placeholder { color: #657e98; opacity: 1; }
[data-testid="stBottomBlockContainer"] { background: transparent; padding-top: 1rem; }
.st-key-astrea_latest_response { animation: astrea-message-in 260ms ease-out; }
@keyframes astrea-message-in { from { opacity: .4; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }
@keyframes astrea-card-in { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
@keyframes astrea-float { 0%, 100% { transform: translateY(0) rotate(-6deg); } 50% { transform: translateY(-6px) rotate(-3deg); } }

.astrea-page-intro { margin: 2px 0 20px; }
.astrea-page-intro h2 { font-size: 28px; font-weight: 600; letter-spacing: -.7px; margin: 0 0 6px; padding: 0; color: var(--astrea-ink); }
.astrea-page-intro p { color: var(--astrea-muted); font-size: 14px; margin: 0; }
.astrea-category { display: inline-block; font-size: 11px; font-weight: 600; padding: 4px 8px; border-radius: 7px; background: #eaf3fe; color: #2b659f; margin-bottom: 5px; }
.astrea-detail-title { font-size: 22px; font-weight: 600; line-height: 1.45; margin: 5px 0 16px; padding: 0; color: var(--astrea-ink); }
[class*="st-key-astrea_faq_"] { border-radius: 16px; background: var(--astrea-panel); border-color: var(--astrea-line); box-shadow: 0 6px 16px #12355509; }
[data-testid="stTextInput"] input { color: var(--astrea-ink); background: var(--astrea-panel); }
[data-testid="stSelectbox"] [data-baseweb="select"] > div { color: var(--astrea-ink); background: var(--astrea-panel); border-color: var(--astrea-line); border-radius: 10px; }

@media (max-width: 760px) {
  .stApp [data-testid="stMainBlockContainer"], .stApp .main .block-container { padding: calc(60px + 1.5rem) 1rem 6rem; }
  .astrea-header { margin-bottom: 17px; padding-bottom: 17px; gap: 10px; }
  .astrea-header h1 { font-size: 17px; }
  .astrea-welcome { margin-top: 1.2rem; }
  .astrea-welcome h2 { letter-spacing: -.8px; }
  .astrea-topic-card strong { font-size: 14px; }
  [data-testid="stChatMessage"] { padding: .8rem; }
  .astrea-detail-title { font-size: 20px; }
}
@media (prefers-reduced-motion: reduce) {
  .stButton > button,
  [class*="st-key-astrea_topic_card_"],
  [data-testid="stDialog"] [role="dialog"],
  .astrea-welcome-mark,
  .st-key-astrea_latest_response { transition: none; animation: none; }
  .stButton > button:hover,
  [class*="st-key-astrea_topic_card_"]:hover { transform: none; }
}
</style>
""")


_PREVIOUS_NAME = re.compile(r"\boficina[\s_-]*pro(?:\.co)?\b", re.IGNORECASE)
_MARKDOWN_PART = re.compile(
    r"(?P<code>```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]+`)"
    r"|(?P<link>\[(?P<label>[^\]\n]*)\]\((?P<target>[^\)\n]+)\))"
    r"|(?P<url>https?://[^\s<>]+)",
    re.IGNORECASE,
)


def display_text(value: str) -> str:
    """Adapta texto de presentación conservando código y destinos de enlaces.

    Se usa al dibujar respuestas y guías, nunca antes de enviar una pregunta
    al pipeline, guardar el historial o identificar una fuente documental.
    """
    value = str(value)
    result = []
    start = 0

    for part in _MARKDOWN_PART.finditer(value):
        result.append(_PREVIOUS_NAME.sub(UI_NAME, value[start:part.start()]))

        if part.group("code") is not None:
            result.append(part.group(0))
        elif part.group("link") is not None:
            label = _PREVIOUS_NAME.sub(UI_NAME, part.group("label"))
            result.append(f"[{label}]({part.group('target')})")
        else:
            url = part.group("url")
            target = url.rstrip(".,;:!?")
            while target.endswith(")") and target.count(")") > target.count("("):
                target = target[:-1]
            punctuation = url[len(target):]
            # Se cambia la etiqueta, nunca la dirección existente.
            result.append(
                f"[Abrir enlace]({target}){punctuation}"
                if _PREVIOUS_NAME.search(target) else url
            )

        start = part.end()

    result.append(_PREVIOUS_NAME.sub(UI_NAME, value[start:]))
    return "".join(result)

# Las versiones anteriores que no admiten key conservan los contenedores nativos.
CONTAINER_HAS_KEY = "key" in inspect.signature(st.container).parameters
DIALOG_AVAILABLE = callable(getattr(st, "dialog", None))


def ui_container(key: str, *, border: bool = True):
    kwargs = {"border": border}
    if CONTAINER_HAS_KEY:
        kwargs["key"] = key
    return st.container(**kwargs)


def escaped(value: str) -> str:
    return html.escape(display_text(str(value)))


def faq_preview(answer: str) -> str:
    clean = re.sub(r"[#>*`]+", "", display_text(answer))
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:130].rstrip() + "…" if len(clean) > 130 else clean


def category_icon(category: str) -> str:
    """Iconos locales: no necesitan librerías ni peticiones externas."""
    icons = {
        "Facturación": '<path d="M6 3h12v18l-3-2-3 2-3-2-3 2V3Z"/><path d="M9 7h6M9 11h6M9 15h3"/>',
        "Inventario": '<path d="m12 3 9 5v8l-9 5-9-5V8l9-5Z"/><path d="m3 8 9 5 9-5M12 13v8M7.5 5.5l9 5"/>',
        "Cartera": '<rect x="3" y="5" width="18" height="15" rx="3"/><path d="M3 9h18M15 14h6"/><circle cx="16" cy="14" r=".5"/>',
        "Cajas y bancos": '<path d="m3 9 9-6 9 6H3ZM5 10v7M10 10v7M14 10v7M19 10v7M3 21h18M4 18h16"/>',
    }
    paths = icons.get(category, '<path d="M4 4h7a3 3 0 0 1 3 3v14a4 4 0 0 0-4-2H4V4ZM14 7a3 3 0 0 1 3-3h3v15h-2a4 4 0 0 0-4 2"/>')
    return (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" '
        f'aria-hidden="true">{paths}</svg>'
    )


def safe_index_exists() -> bool:
    try:
        return index_exists()
    except Exception:
        return False


for name, default in (
    ("messages", []),
    ("main_view", "Chat"),
    ("selected_faq", 0),
    ("faq_category", "Todas"),
    ("astrea_message_details", {}),
):
    if name not in st.session_state:
        st.session_state[name] = default


@st.cache_resource(show_spinner="Preparando la búsqueda documental…")
def _get_embeddings():
    return get_embeddings()


@st.cache_resource(show_spinner="Preparando el asistente…")
def _get_llm():
    return get_llm()


@st.cache_resource(show_spinner="Abriendo la base de conocimiento…")
def _get_vectorstore(_embeddings):
    return load_vectorstore(_embeddings)


def new_chat() -> None:
    st.session_state.messages = []
    st.session_state.astrea_message_details = {}
    st.session_state.main_view = "Chat"
    st.session_state.pop("pending_question", None)
    st.session_state.pop("astrea_open_topic", None)


def select_view(view: str) -> None:
    st.session_state.main_view = view
    st.session_state.pop("astrea_open_topic", None)


def queue_faq(index: int) -> None:
    st.session_state.selected_faq = index
    st.session_state.pending_question = FAQS[index]["question"]
    st.session_state.main_view = "Chat"
    st.session_state.pop("astrea_open_topic", None)


def queue_chat_question() -> None:
    # El callback permite cambiar del formulario de bienvenida al chat fijo
    # antes del rerun, sin perder ni transformar la pregunta escrita.
    st.session_state.pending_question = st.session_state.astrea_chat_input


def show_category(category: str) -> None:
    st.session_state.faq_category = category
    st.session_state.faq_categories = category
    st.session_state.main_view = "Soluciones rápidas"
    st.session_state.pop("astrea_open_topic", None)


def render_sidebar(index_ready: bool) -> int:
    with st.sidebar:
        st.html(
            '<div class="astrea-brand">'
            '<div class="astrea-brand-mark" aria-hidden="true">A</div>'
            '<div><div class="astrea-brand-name">'
            f'{escaped(UI_NAME)}</div>'
            '<div class="astrea-brand-description">Tu asistente de soporte</div>'
            '</div></div>'
        )
        with ui_container("astrea_new_chat_area", border=False):
            st.button(
                "+ Nueva consulta",
                key="astrea_new_chat",
                type="primary",
                use_container_width=True,
                on_click=new_chat,
            )
        st.html('<p class="astrea-menu-label">TU ESPACIO DE AYUDA</p>')
        st.button(
            "Chat de soporte",
            key="nav_chat",
            type="primary" if st.session_state.main_view == "Chat" else "secondary",
            use_container_width=True,
            on_click=select_view,
            args=("Chat",),
        )
        st.button(
            "Soluciones rápidas",
            key="nav_solutions",
            type="primary" if st.session_state.main_view == "Soluciones rápidas" else "secondary",
            use_container_width=True,
            on_click=select_view,
            args=("Soluciones rápidas",),
        )
        st.html(
            '<div class="astrea-sidebar-note">'
            '<strong>Las fuentes, a la mano.</strong>'
            '<p>Consulta los documentos que acompañan cada respuesta.</p></div>'
        )
        with st.expander("Configuración de la base", expanded=False):
            st.caption("Índice disponible" if index_ready else "Índice pendiente")
            st.caption(f"Directorio de manuales: `{config.PDF_DIR}`")
            top_k = st.slider(
                "Fragmentos a recuperar",
                min_value=2,
                max_value=10,
                value=config.TOP_K,
                key="astrea_top_k",
            )
            if st.button(
                "Reindexar manuales PDF",
                key="astrea_reindex",
                use_container_width=True,
            ):
                try:
                    with st.spinner("Procesando documentos…"):
                        embeddings = _get_embeddings()
                        documents = load_pdfs()
                        chunks = split_documents(documents)
                        build_vectorstore(chunks, embeddings, rebuild=True)
                    st.cache_resource.clear()
                    st.session_state.astrea_notice = "Índice actualizado correctamente."
                    st.rerun()
                except Exception as exc:
                    st.error("No fue posible indexar los manuales.")
                    with st.expander("Detalle técnico"):
                        st.code(str(exc), language=None)
            st.button(
                "Borrar historial del chat",
                key="astrea_clear_history",
                use_container_width=True,
                on_click=new_chat,
            )
        st.html(
            '<div class="astrea-sidebar-footer">'
            f'{escaped(UI_NAME)}<span>Centro de soporte</span></div>'
        )
    return top_k


def render_header(index_ready: bool) -> None:
    is_chat = st.session_state.main_view == "Chat"
    title = "Centro de soporte" if is_chat else "Soluciones rápidas"
    state_class = "is-ready" if index_ready else "is-pending"
    state_label = "Índice disponible" if index_ready else "Índice pendiente"
    st.html(
        '<header class="astrea-header">'
        f'<div><h1>{title}</h1><p>Tu espacio de ayuda con {escaped(UI_NAME)}</p></div>'
        f'<span class="astrea-status {state_class}"><span aria-hidden="true"></span>{state_label}</span>'
        '</header>'
    )


def render_topic_contents(category: str) -> None:
    indices = [i for i, faq in enumerate(FAQS) if faq["category"] == category]
    st.html(f'<div class="astrea-topic-heading">{escaped(category)}</div>')
    if not indices:
        st.info("Aún no hay soluciones rápidas para este tema.")
        return
    selected = st.selectbox(
        "Elige tu consulta",
        indices,
        format_func=lambda index: display_text(FAQS[index]["question"]),
        key=f"astrea_topic_selection_{category}",
    )
    faq = FAQS[selected]
    st.markdown(display_text(faq["answer"]))
    st.caption(f"Fuente: {faq.get('source', 'Base de conocimiento')}")
    if st.button(
        "Consultar este caso en el chat",
        type="primary",
        use_container_width=True,
        key=f"astrea_topic_ask_{category}",
    ):
        queue_faq(selected)
        st.rerun()
    if st.button(
        "Ver todas las soluciones de este tema",
        use_container_width=True,
        key=f"astrea_topic_more_{category}",
    ):
        show_category(category)
        st.rerun()


if DIALOG_AVAILABLE:
    @st.dialog("Soluciones de Astrea")
    def topic_dialog(category: str) -> None:
        render_topic_contents(category)


def render_welcome() -> None:
    st.html(
        '<section class="astrea-welcome">'
        '<div class="astrea-welcome-mark" aria-hidden="true">A</div>'
        f'<p>Hola, soy {escaped(UI_NAME)}.</p>'
        '<h2>¿Qué <span>resolvemos</span> hoy?</h2>'
        '<div>Encuentra ayuda con pasos claros y fuentes a la mano.</div>'
        '</section>'
    )
    available = {faq["category"] for faq in FAQS}
    preferred = ("Facturación", "Inventario", "Cartera", "Cajas y bancos")
    categories = [category for category in preferred if category in available]
    categories.extend(sorted(available.difference(categories)))
    categories = categories[:4]
    descriptions = {
        "Facturación": "Facturas y vencimientos",
        "Inventario": "Ayuda con tus productos",
        "Cartera": "Consultas de este módulo",
        "Cajas y bancos": "Orientación paso a paso",
    }
    selected_topic = None
    for row_start in range(0, len(categories), 2):
        columns = st.columns(2, gap="medium")
        for column, category in zip(columns, categories[row_start:row_start + 2]):
            position = categories.index(category)
            with column:
                with ui_container(f"astrea_topic_card_{position}"):
                    st.html(
                        '<div class="astrea-topic-card">'
                        f'<span class="astrea-topic-icon" aria-hidden="true">{category_icon(category)}</span>'
                        '<div>'
                        f'<strong>{escaped(category)}</strong>'
                        f'<p>{escaped(descriptions.get(category, "Guías de soporte"))}</p>'
                        '</div></div>'
                    )
                    if st.button(
                        f"Explorar {display_text(category)} →",
                        key=f"astrea_open_{position}",
                        use_container_width=True,
                    ):
                        selected_topic = category
    if selected_topic:
        if DIALOG_AVAILABLE:
            topic_dialog(selected_topic)
        else:
            st.session_state.astrea_open_topic = selected_topic
    if not DIALOG_AVAILABLE and st.session_state.get("astrea_open_topic"):
        with st.expander("Ficha de la solución", expanded=True):
            render_topic_contents(st.session_state.astrea_open_topic)
            if st.button("Cerrar ficha", key="astrea_close_topic"):
                st.session_state.pop("astrea_open_topic", None)
                st.rerun()
    st.html('<p class="astrea-welcome-hint">Elige un tema o escribe tu pregunta abajo.</p>')


def serialize_sources(documents: list) -> list[dict]:
    sources = []
    for document in documents:
        metadata = document.metadata or {}
        filename = str(metadata.get("source", "?")).replace("\\", "/").split("/")[-1]
        content = document.page_content.strip()
        sources.append({
            "archivo": filename,
            "pagina": metadata.get("page", "?"),
            "fragmento": content[:300] + ("…" if len(content) > 300 else ""),
        })
    return sources


def render_message(message: dict, index: int, *, new: bool = False) -> None:
    details = st.session_state.astrea_message_details.get(index, {})
    role = message.get("role", "assistant")
    content = str(message.get("content", ""))
    wrapper = ui_container("astrea_latest_response", border=False) if new else nullcontext()
    with wrapper:
        with st.chat_message(role):
            if role == "assistant":
                st.html(f'<div class="astrea-message-name">{escaped(UI_NAME)}</div>')
                if details.get("error"):
                    st.error(display_text(content))
                else:
                    st.markdown(display_text(content))
                sources = details.get("sources", [])
                if sources:
                    with st.expander(f"Fuentes consultadas ({len(sources)})"):
                        for index, source in enumerate(sources, 1):
                            st.text(f"[{index}] {source['archivo']} — Pág. {source['pagina']}")
                            st.caption(display_text(source.get("fragmento", "")))
                with st.expander("Copiar respuesta"):
                    st.caption("Usa el botón de copiar del bloque.")
                    st.code(display_text(content), language=None)
                if details.get("error_detail"):
                    with st.expander("Detalle técnico"):
                        st.code(details["error_detail"], language=None)
            else:
                st.markdown(content)


def answer_question(question: str, top_k: int, index_ready: bool) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    render_message(st.session_state.messages[-1], len(st.session_state.messages) - 1)
    answer = {"role": "assistant", "content": ""}
    details = {"sources": []}
    if not index_ready:
        answer["content"] = (
            "La base de conocimiento todavía no está preparada. "
            "Puedes consultar las soluciones rápidas mientras se indexan los documentos. "
            "La opción 'Reindexar manuales PDF' está en Configuración de la base."
        )
    else:
        try:
            with st.spinner("Consultando la documentación…"):
                embeddings = _get_embeddings()
                llm = _get_llm()
                prompt_template = get_prompt_template()
                vector_store = _get_vectorstore(embeddings)
                result = rag_pipeline(
                    vector_store,
                    llm,
                    prompt_template,
                    question,
                    k=top_k,
                    historial=st.session_state.messages[:-1],
                )
            answer["content"] = result["respuesta"]
            details["sources"] = serialize_sources(result.get("fragmentos", []))
        except Exception as exc:
            answer["content"] = "No fue posible completar la consulta en este momento."
            details["error"] = True
            details["error_detail"] = str(exc)
    # Las fuentes y los detalles visuales no se envían al modelo.
    index = len(st.session_state.messages)
    st.session_state.messages.append(answer)
    st.session_state.astrea_message_details[index] = details
    render_message(answer, index, new=True)


def render_chat(index_ready: bool, top_k: int) -> None:
    pending_question = st.session_state.pop("pending_question", None)
    if not index_ready:
        st.warning(
            "La información de soporte está pendiente de indexación. "
            "Las soluciones rápidas siguen disponibles."
        )
    show_welcome = not st.session_state.messages and not pending_question
    if show_welcome:
        render_welcome()
    # Un formulario inline evita el auto-scroll de Streamlit en la bienvenida.
    # En una conversación, el mismo componente nativo queda fijo al pie.
    input_context = ui_container("astrea_welcome_input", border=False) if show_welcome else nullcontext()
    with input_context:
        typed_question = st.chat_input(
            "Escribe tu pregunta a Astrea…",
            key="astrea_chat_input",
            on_submit=queue_chat_question,
        )
    question = pending_question or typed_question
    for index, message in enumerate(st.session_state.messages):
        render_message(message, index)
    if question and question.strip():
        answer_question(question, top_k, index_ready)


def render_solutions() -> None:
    st.html(
        '<div class="astrea-page-intro"><h2>Encuentra una solución</h2>'
        '<p>Consulta una guía o continúa tu pregunta en el chat.</p></div>'
    )
    search_col, category_col = st.columns([2, 1], gap="medium")
    with search_col:
        search = st.text_input(
            "Buscar una solución",
            placeholder="Factura, inventario, cartera…",
            key="faq_search",
        ).strip().casefold()
    categories = ["Todas"] + sorted({faq["category"] for faq in FAQS})
    with category_col:
        category = st.selectbox(
            "Categoría",
            categories,
            index=categories.index(st.session_state.faq_category) if st.session_state.faq_category in categories else 0,
            key="faq_categories",
        )
    st.session_state.faq_category = category
    matches = []
    for index, faq in enumerate(FAQS):
        searchable = display_text(" ".join(str(faq.get(field, "")) for field in ("question", "category", "keywords", "answer"))).casefold()
        if (category == "Todas" or faq["category"] == category) and (not search or search in searchable):
            matches.append((index, faq))
    if not matches:
        st.info("No encontré una solución rápida con ese término. Prueba otra palabra o categoría.")
        return
    visible_indices = [index for index, _ in matches]
    if st.session_state.selected_faq not in visible_indices:
        st.session_state.selected_faq = visible_indices[0]
    list_col, detail_col = st.columns([1, 1.5], gap="large")
    with list_col:
        for index, faq in matches:
            with ui_container(f"astrea_faq_{index}"):
                st.html(f'<span class="astrea-category">{escaped(faq["category"])}</span>')
                if st.button(
                    display_text(faq["question"]),
                    key=f"faq_select_{index}",
                    type="primary" if index == st.session_state.selected_faq else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.selected_faq = index
                    st.rerun()
                st.caption(faq_preview(faq["answer"]))
    with detail_col:
        faq = FAQS[st.session_state.selected_faq]
        with ui_container("astrea_faq_detail"):
            st.html(
                f'<span class="astrea-category">{escaped(faq["category"])}</span>'
                f'<h2 class="astrea-detail-title">{escaped(faq["question"])}</h2>'
            )
            st.markdown(display_text(faq["answer"]))
            st.caption(f"Fuente: {faq.get('source', 'Base de conocimiento')}")
            st.button(
                "Consultar este caso en el chat",
                key="faq_to_chat",
                type="primary",
                use_container_width=True,
                on_click=queue_faq,
                args=(st.session_state.selected_faq,),
            )


index_ready = safe_index_exists()
top_k = render_sidebar(index_ready)
render_header(index_ready)
notice = st.session_state.pop("astrea_notice", None)
if notice:
    st.success(notice)
if st.session_state.main_view == "Soluciones rápidas":
    render_solutions()
else:
    render_chat(index_ready, top_k)
