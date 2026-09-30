import html
import re

import streamlit as st

print("1 - Streamlit importado correctamente")

from src import config
print("2 - config cargado correctamente")

from src.faqs import FAQS
print("3 - FAQS cargadas correctamente")

from src.ingest import load_pdfs, split_documents
print("4 - ingest cargado correctamente")

from src.vectorstore import (
    get_embeddings,
    build_vectorstore,
    load_vectorstore,
    index_exists,
)
print("5 - vectorstore cargado correctamente")

from src.rag_chain import (
    get_llm,
    get_prompt_template,
    rag_pipeline,
)
print("6 - rag_chain cargado correctamente")


# -----------------------------------------------------------------------------
# Configuración de página
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title=f"{config.ASSISTANT_NAME} | OficinaPro",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

print("7 - set_page_config ejecutado correctamente")


# -----------------------------------------------------------------------------
# Estilos visuales
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>
        :root {
            --op-navy: #12355b;
            --op-navy-2: #173f6d;
            --op-blue: #1478f2;
            --op-blue-soft: #eaf4ff;
            --op-text: #152744;
            --op-muted: #667892;
            --op-border: #dfe7f1;
            --op-bg: #f7f9fc;
            --op-card: #ffffff;
            --op-green: #119b5b;
            --op-green-soft: #edf9f2;
            --op-yellow: #b87800;
            --op-yellow-soft: #fff8e8;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: var(--op-bg);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 1.1rem;
            padding-bottom: 2rem;
            padding-left: 1.4rem;
            padding-right: 1.4rem;
        }

        .op-topbar {
            background: linear-gradient(110deg, #12355b 0%, #163f6d 100%);
            color: white;
            border-radius: 18px;
            min-height: 78px;
            padding: 16px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
            margin-bottom: 18px;
            box-shadow: 0 8px 28px rgba(18, 53, 91, 0.12);
        }

        .op-brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .op-brand-icon {
            width: 42px;
            height: 42px;
            border-radius: 12px;
            background: rgba(255,255,255,.12);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
        }

        .op-brand-title {
            font-size: 25px;
            font-weight: 800;
            line-height: 1.05;
            letter-spacing: -.4px;
        }

        .op-brand-title span {
            color: #55a8ff;
        }

        .op-brand-subtitle {
            margin-top: 3px;
            font-size: 12px;
            color: rgba(255,255,255,.76);
        }

        .op-top-help {
            font-size: 13px;
            color: rgba(255,255,255,.92);
            white-space: nowrap;
        }

        .op-nav-card,
        .op-panel,
        .op-faq-detail {
            background: var(--op-card);
            border: 1px solid var(--op-border);
            border-radius: 16px;
            box-shadow: 0 8px 26px rgba(18, 53, 91, 0.045);
        }

        .op-nav-card {
            padding: 14px;
            min-height: 560px;
        }

        .op-nav-title {
            color: var(--op-muted);
            font-size: 11px;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin: 2px 4px 10px;
        }

        .op-nav-current {
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 10px;
            background: var(--op-blue-soft);
            border: 1px solid #cde4ff;
            color: #0b64c9;
            font-weight: 700;
            font-size: 13px;
        }

        .op-nav-note {
            margin-top: 18px;
            padding: 12px;
            background: #f8fafc;
            border: 1px solid #e8eef5;
            border-radius: 12px;
            color: var(--op-muted);
            font-size: 12px;
            line-height: 1.45;
        }

        .op-page-title-wrap {
            padding: 4px 2px 12px;
        }

        .op-page-title {
            color: var(--op-text);
            font-size: 30px;
            line-height: 1.15;
            font-weight: 800;
            letter-spacing: -.45px;
            margin: 0;
        }

        .op-page-subtitle {
            color: var(--op-muted);
            font-size: 14px;
            margin-top: 6px;
        }

        .op-panel {
            padding: 16px;
        }

        .op-section-label {
            font-size: 11px;
            font-weight: 800;
            color: var(--op-muted);
            text-transform: uppercase;
            letter-spacing: .07em;
            margin-bottom: 8px;
        }

        .op-status-ok,
        .op-status-warn {
            border-radius: 12px;
            padding: 11px 12px;
            font-size: 12px;
            line-height: 1.4;
            margin: 10px 0;
        }

        .op-status-ok {
            color: #087044;
            background: var(--op-green-soft);
            border: 1px solid #bfe9d1;
        }

        .op-status-warn {
            color: #8a5a00;
            background: var(--op-yellow-soft);
            border: 1px solid #f4dfa7;
        }

        .op-faq-card {
            border: 1px solid var(--op-border);
            border-radius: 13px;
            background: #fff;
            padding: 12px 14px;
            margin-bottom: 8px;
            transition: .18s ease;
        }

        .op-faq-card:hover {
            border-color: #b6d7ff;
            box-shadow: 0 5px 16px rgba(20, 120, 242, .06);
        }

        .op-faq-category {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 4px 9px;
            background: var(--op-blue-soft);
            color: #0b64c9;
            font-size: 11px;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .op-faq-question {
            font-size: 14px;
            font-weight: 750;
            color: var(--op-text);
            line-height: 1.35;
        }

        .op-faq-detail {
            padding: 19px 20px;
        }

        .op-detail-heading {
            font-size: 23px;
            font-weight: 800;
            color: var(--op-text);
            letter-spacing: -.3px;
            line-height: 1.25;
            margin: 4px 0 8px;
        }

        .op-detail-source {
            color: var(--op-muted);
            font-size: 11px;
            margin-top: 10px;
            padding-top: 9px;
            border-top: 1px solid #edf1f6;
        }

        .op-steps-box {
            border: 1px solid #cde4ff;
            background: #f3f8ff;
            border-radius: 13px;
            padding: 14px 16px;
            margin-top: 12px;
        }

        .op-info-box {
            border: 1px solid #bee6ce;
            background: #f0faf4;
            border-radius: 13px;
            padding: 13px 15px;
            margin-top: 12px;
            color: #195f3d;
            font-size: 13px;
            line-height: 1.55;
        }

        .op-chat-intro {
            border: 1px solid #dce8f5;
            background: linear-gradient(135deg, #ffffff 0%, #f5f9ff 100%);
            border-radius: 16px;
            padding: 17px 18px;
            margin-bottom: 14px;
        }

        .op-chat-intro strong {
            color: var(--op-text);
            font-size: 15px;
        }

        .op-chat-intro div {
            color: var(--op-muted);
            margin-top: 4px;
            font-size: 13px;
        }

        div[data-testid="stTextInput"] input {
            min-height: 45px;
            border-radius: 12px;
            border-color: #dbe5f0;
        }

        div[data-testid="stTextInput"] input:focus {
            border-color: #1478f2;
            box-shadow: 0 0 0 1px #1478f2;
        }

        .stButton > button {
            border-radius: 11px;
            min-height: 42px;
            border: 1px solid #dbe5f0;
            font-weight: 650;
        }

        .stButton > button[kind="primary"] {
            background: #1478f2;
            border-color: #1478f2;
        }

        .stButton > button:hover {
            border-color: #8ec3ff;
        }

        [data-testid="stChatMessage"] {
            border: 1px solid #e5ebf3;
            border-radius: 14px;
            padding: .55rem .75rem;
            margin-bottom: .65rem;
            background: #fff;
        }

        [data-testid="stChatInput"] textarea {
            border-radius: 14px;
        }

        [data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid #e7edf4;
        }

        @media (max-width: 900px) {
            .op-top-help {
                display: none;
            }

            .op-brand-title {
                font-size: 21px;
            }

            .op-page-title {
                font-size: 25px;
            }

            .op-nav-card {
                min-height: auto;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

print("8 - estilos principales cargados")


# -----------------------------------------------------------------------------
# Estado de interfaz
# -----------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "main_view" not in st.session_state:
    st.session_state.main_view = "Soluciones rápidas"

if "selected_faq" not in st.session_state:
    st.session_state.selected_faq = 0

if "faq_category" not in st.session_state:
    st.session_state.faq_category = "Todas"

print("9 - session_state inicializado")


# -----------------------------------------------------------------------------
# Recursos RAG cacheados
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando modelo de embeddings...")
def _get_embeddings():
    print("10 - entrando a get_embeddings")
    return get_embeddings()


@st.cache_resource(show_spinner="Cargando modelo de lenguaje...")
def _get_llm():
    print("11 - entrando a get_llm")
    return get_llm()


@st.cache_resource(show_spinner="Cargando base vectorial...")
def _get_vectorstore(_embeddings):
    print("12 - entrando a load_vectorstore")
    return load_vectorstore(_embeddings)


def _resolver_pregunta(pregunta: str, top_k: int) -> dict | None:
    print("13 - resolver pregunta")

    if not index_exists():
        print("13.1 - no existe índice")
        return None

    print("13.2 - índice encontrado")

    embeddings = _get_embeddings()
    llm = _get_llm()
    prompt_template = get_prompt_template()
    vector_store = _get_vectorstore(embeddings)

    return rag_pipeline(
        vector_store,
        llm,
        prompt_template,
        pregunta,
        k=top_k,
    )


def _mostrar_fuentes(fragmentos: list) -> None:
    with st.expander(f"📎 Fuentes consultadas ({len(fragmentos)})"):
        for i, doc in enumerate(fragmentos, 1):
            fuente = (
                doc.metadata.get("source", "?")
                .split("/")[-1]
                .split("\\")[-1]
            )
            pagina = doc.metadata.get("page", "?")

            st.markdown(f"**[{i}] {fuente} — Pág. {pagina}**")

            contenido = doc.page_content.strip()

            st.caption(
                contenido[:300]
                + ("..." if len(contenido) > 300 else "")
            )


def _faq_preview(answer: str) -> str:
    clean = re.sub(r"[#>*`]+", "", answer)
    clean = re.sub(r"\s+", " ", clean).strip()

    if len(clean) > 100:
        return clean[:100].rstrip() + "..."

    return clean


def _category_icon(category: str) -> str:
    cat = category.lower()

    if "factur" in cat:
        return "🧾"

    if "invent" in cat:
        return "📦"

    if "cartera" in cat:
        return "💵"

    if "cliente" in cat or "proveedor" in cat:
        return "👥"

    if "banco" in cat or "caja" in cat:
        return "🏦"

    if "nota" in cat:
        return "📄"

    return "📘"


print("14 - funciones definidas correctamente")


# -----------------------------------------------------------------------------
# Cabecera
# -----------------------------------------------------------------------------

st.markdown(
    """
    <div class="op-topbar">
        <div class="op-brand">
            <div class="op-brand-icon">✦</div>

            <div>
                <div class="op-brand-title">
                    OFICINA<span>PRO.CO</span>
                </div>

                <div class="op-brand-subtitle">
                    Asistente virtual de soporte
                </div>
            </div>
        </div>

        <div class="op-top-help">
            ⓘ &nbsp; Tu aliado en la solución de dudas de OficinaPro
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

print("15 - cabecera mostrada")


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Base de conocimiento")

    st.caption(f"Manuales: `{config.PDF_DIR}`")

    print("16 - comprobando index_exists sidebar")

    try:
        indice_disponible = index_exists()
        print(f"17 - index_exists sidebar: {indice_disponible}")
    except Exception as exc:
        print(f"ERROR index_exists sidebar: {exc}")
        indice_disponible = False

    if indice_disponible:
        st.success("Índice vectorial disponible.")
    else:
        st.warning("Aún no hay índice vectorial.")

    top_k = st.slider(
        "Fragmentos a recuperar (k)",
        2,
        10,
        config.TOP_K,
    )

    if st.button(
        "🔄 (Re)indexar manuales PDF",
        use_container_width=True,
    ):
        with st.spinner("Cargando y procesando PDFs..."):
            embeddings = _get_embeddings()
            documents = load_pdfs()
            chunks = split_documents(documents)

            build_vectorstore(
                chunks,
                embeddings,
                rebuild=True,
            )

        st.cache_resource.clear()
        st.success("Índice actualizado correctamente.")

    if st.button(
        "🧹 Borrar historial del chat",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


print("18 - sidebar cargado")


# -----------------------------------------------------------------------------
# Layout principal
# -----------------------------------------------------------------------------

nav_col, content_col = st.columns(
    [1.15, 4.85],
    gap="large",
)


with nav_col:
    st.markdown(
        '<div class="op-nav-card">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="op-nav-title">Soporte</div>',
        unsafe_allow_html=True,
    )

    chat_active = st.session_state.main_view == "Chat"

    if st.button(
        "💬  Chat\n\nConsulta tus dudas",
        use_container_width=True,
        type="primary" if chat_active else "secondary",
        key="nav_chat",
    ):
        st.session_state.main_view = "Chat"
        st.rerun()

    faq_active = (
        st.session_state.main_view == "Soluciones rápidas"
    )

    if st.button(
        "📖  Soluciones rápidas\n\nPreguntas frecuentes",
        use_container_width=True,
        type="primary" if faq_active else "secondary",
        key="nav_faq",
    ):
        st.session_state.main_view = "Soluciones rápidas"
        st.rerun()

    try:
        estado_indice = index_exists()
    except Exception as exc:
        print(f"ERROR index_exists navegación: {exc}")
        estado_indice = False

    st.markdown(
        f"""
        <div class="op-nav-note">
            <b>Estado del conocimiento</b><br>

            {
                '● Índice listo para consultar'
                if estado_indice
                else '● Pendiente de indexación'
            }

            <br><br>

            La configuración técnica sigue disponible
            en el menú lateral de Streamlit.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


print("19 - navegación cargada")


# -----------------------------------------------------------------------------
# Vista principal
# -----------------------------------------------------------------------------

with content_col:

    if st.session_state.main_view == "Soluciones rápidas":

        st.markdown(
            """
            <div class="op-page-title-wrap">
                <div class="op-page-title">
                    📖 Soluciones rápidas
                </div>

                <div class="op-page-subtitle">
                    Encuentra respuestas a los casos más comunes
                    de OficinaPro y, si necesitas más detalle,
                    continúa directamente en el chat.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        filtro = st.text_input(
            "Buscar solución",
            placeholder=(
                "🔎 Buscar por palabra clave: inventario, "
                "factura, cartera, banco, cliente..."
            ),
            label_visibility="collapsed",
            key="faq_filter_visual",
        ).strip().lower()

        categorias = [
            "Todas"
        ] + sorted(
            {
                faq["category"]
                for faq in FAQS
            }
        )

        categoria_actual = st.radio(
            "Categoría",
            categorias,
            index=(
                categorias.index(
                    st.session_state.faq_category
                )
                if st.session_state.faq_category in categorias
                else 0
            ),
            horizontal=True,
            label_visibility="collapsed",
            key="faq_category_radio",
        )

        st.session_state.faq_category = categoria_actual

        faqs_visibles = []

        for original_index, faq in enumerate(FAQS):

            coincide_categoria = (
                categoria_actual == "Todas"
                or faq["category"] == categoria_actual
            )

            coincide_texto = (
                not filtro
                or filtro in faq["question"].lower()
                or filtro in faq["category"].lower()
                or filtro in faq.get("keywords", "").lower()
                or filtro in faq["answer"].lower()
            )

            if coincide_categoria and coincide_texto:
                faqs_visibles.append(
                    (
                        original_index,
                        faq,
                    )
                )

        if not faqs_visibles:

            st.info(
                "No encontré una solución rápida con ese término. "
                "Puedes consultarlo en el Chat."
            )

        else:

            visible_indexes = [
                idx
                for idx, _ in faqs_visibles
            ]

            if (
                st.session_state.selected_faq
                not in visible_indexes
            ):
                st.session_state.selected_faq = (
                    visible_indexes[0]
                )

            list_col, detail_col = st.columns(
                [2.15, 2.65],
                gap="medium",
            )

            with list_col:

                st.markdown(
                    '<div class="op-panel">',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div class="op-section-label">'
                    'Casos frecuentes'
                    '</div>',
                    unsafe_allow_html=True,
                )

                for original_index, faq in faqs_visibles:

                    icon = _category_icon(
                        faq["category"]
                    )

                    selected = (
                        original_index
                        == st.session_state.selected_faq
                    )

                    st.markdown(
                        f"""
                        <div class="op-faq-card">

                            <div class="op-faq-category">
                                {html.escape(faq['category'])}
                            </div>

                            <div class="op-faq-question">
                                {icon} &nbsp;
                                {html.escape(faq['question'])}
                            </div>

                            <div style="
                                color:#77879d;
                                font-size:12px;
                                margin-top:5px;
                                line-height:1.35;
                            ">
                                {
                                    html.escape(
                                        _faq_preview(
                                            faq["answer"]
                                        )
                                    )
                                }
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if st.button(
                        (
                            "✓ Solución seleccionada"
                            if selected
                            else "Ver solución"
                        ),
                        key=f"select_faq_{original_index}",
                        use_container_width=True,
                        type=(
                            "primary"
                            if selected
                            else "secondary"
                        ),
                    ):
                        st.session_state.selected_faq = (
                            original_index
                        )
                        st.rerun()

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )

            with detail_col:

                faq = FAQS[
                    st.session_state.selected_faq
                ]

                st.markdown(
                    '<div class="op-faq-detail">',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="op-faq-category">
                        {html.escape(faq["category"])}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="op-detail-heading">
                        {html.escape(faq["question"])}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.caption(
                    "Procedimiento recomendado según "
                    "la base de conocimiento disponible."
                )

                st.markdown(
                    '<div class="op-steps-box">',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    faq["answer"]
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="op-detail-source">
                        Fuente:
                        {html.escape(faq["source"])}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )

                st.write("")

                if st.button(
                    "💬 Consultar este caso en el chat",
                    type="primary",
                    use_container_width=True,
                    key="faq_to_chat_visual",
                ):
                    st.session_state.pending_question = (
                        faq["question"]
                    )

                    st.session_state.main_view = "Chat"

                    st.rerun()


    else:

        st.markdown(
            """
            <div class="op-page-title-wrap">

                <div class="op-page-title">
                    💬 Chat de soporte
                </div>

                <div class="op-page-subtitle">
                    Consulta los manuales y la base de conocimiento
                    de OficinaPro mediante el asistente RAG.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="op-chat-intro">

                <strong>
                    ¿Qué necesitas resolver?
                </strong>

                <div>
                    Describe el problema tal como lo reportaría
                    un cliente o un agente de soporte.
                    El asistente buscará información relacionada
                    en los documentos indexados.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            indice_chat = index_exists()
        except Exception as exc:
            print(f"ERROR index_exists chat: {exc}")
            indice_chat = False

        if not indice_chat:

            st.markdown(
                """
                <div class="op-status-warn">
                    ⚠️ La base vectorial aún no está disponible.
                    Abre la configuración lateral y usa
                    <b>(Re)indexar manuales PDF</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                """
                <div class="op-status-ok">
                    ● Base de conocimiento lista para consultas.
                </div>
                """,
                unsafe_allow_html=True,
            )

        for msg in st.session_state.messages:

            with st.chat_message(
                msg["role"]
            ):
                st.markdown(
                    msg["content"]
                )

        pregunta_pendiente = (
            st.session_state.pop(
                "pending_question",
                None,
            )
        )

        pregunta = (
            pregunta_pendiente
            or st.chat_input(
                "Escribe el problema o la duda "
                "del cliente/agente..."
            )
        )

        if pregunta:

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": pregunta,
                }
            )

            with st.chat_message("user"):
                st.markdown(
                    pregunta
                )

            with st.chat_message("assistant"):

                if not indice_chat:

                    respuesta_txt = (
                        "Aún no hay manuales indexados. "
                        "Abre la configuración lateral y pulsa "
                        "**(Re)indexar manuales PDF**."
                    )

                    st.markdown(
                        respuesta_txt
                    )

                else:

                    try:

                        with st.spinner(
                            "Buscando en los manuales..."
                        ):

                            resultado = (
                                _resolver_pregunta(
                                    pregunta,
                                    top_k,
                                )
                            )

                        if resultado is None:

                            respuesta_txt = (
                                "No se encontró un índice "
                                "vectorial disponible."
                            )

                            st.warning(
                                respuesta_txt
                            )

                        else:

                            respuesta_txt = (
                                resultado["respuesta"]
                            )

                            st.markdown(
                                respuesta_txt
                            )

                            _mostrar_fuentes(
                                resultado["fragmentos"]
                            )

                    except Exception as exc:

                        print(
                            f"ERROR ejecutando RAG: "
                            f"{type(exc).__name__}: {exc}"
                        )

                        respuesta_txt = (
                            "No fue posible completar la consulta "
                            "en este momento. La interfaz sigue "
                            "disponible; revisa la conexión del "
                            "modelo y vuelve a intentar."
                        )

                        st.error(
                            respuesta_txt
                        )

                        with st.expander(
                            "Detalle técnico"
                        ):
                            st.code(
                                str(exc)
                            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": respuesta_txt,
                }
            )


print("20 - app.py terminó de renderizar")