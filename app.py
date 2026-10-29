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
