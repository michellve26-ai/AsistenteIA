// =========================================================
// CONFIGURACIÓN DE MARCA
// =========================================================

const BRAND_FALLBACK = {
  assistant_name: "AstreaS",
  empresa: "Soporte técnico",
};


// =========================================================
// ENDPOINTS DEL BACKEND
// =========================================================

const API = {
  status: "/api/status",
  faqs: "/api/faqs",
  chat: "/api/chat",
  index: "/api/index",
};


// =========================================================
// CONFIGURACIÓN DEL HISTORIAL
// =========================================================

const CHAT_STORAGE_KEY = "soporte_chat_historial";

// Máximo de mensajes anteriores que se enviarán al backend.
// 12 mensajes equivalen aproximadamente a 6 turnos.
const MAX_HISTORY_MESSAGES = 12;


// =========================================================
// NAVEGACIÓN ENTRE VISTAS
// =========================================================

const navItems = document.querySelectorAll(".nav-item");
const views = document.querySelectorAll(".view");

const topbarTitle = document.getElementById("topbar-title");
const topbarSub = document.getElementById("topbar-sub");


const VIEW_META = {
  chat: {
    title: "Chat",
    sub: "Pregunta lo que necesites sobre los manuales internos.",
  },

  faqs: {
    title: "Soluciones rápidas",
    sub: "Encuentra aquí las respuestas a los casos más comunes.",
  },
};


function switchView(name) {

  navItems.forEach((btn) => {
    btn.classList.toggle(
      "is-active",
      btn.dataset.view === name
    );
  });


  views.forEach((view) => {
    view.classList.toggle(
      "is-active",
      view.id === `view-${name}`
    );
  });


  const meta = VIEW_META[name];

  if (!meta) {
    return;
  }


  topbarTitle.textContent = meta.title;
  topbarSub.textContent = meta.sub;
}


navItems.forEach((btn) => {

  btn.addEventListener(
    "click",
    () => switchView(
      btn.dataset.view
    )
  );

});


// =========================================================
// ESTADO / MARCA / ÍNDICE
// =========================================================

async function loadStatus() {

  const statusEl = document.getElementById(
    "index-status"
  );

  const statusText = document.getElementById(
    "index-status-text"
  );


  try {

    const res = await fetch(
      API.status
    );


    if (!res.ok) {
      throw new Error(
        "No fue posible consultar el estado."
      );
    }


    const data = await res.json();


    const brandName = document.getElementById(
      "brand-name"
    );

    const brandSub = document.getElementById(
      "brand-sub"
    );


    if (brandName) {

      brandName.textContent =
        data.assistant_name ||
        BRAND_FALLBACK.assistant_name;

    }


    if (brandSub) {

      brandSub.textContent =
        data.empresa ||
        BRAND_FALLBACK.empresa;

    }


    if (data.indexed) {

      statusEl.classList.add(
        "is-ready"
      );

      statusEl.classList.remove(
        "is-error"
      );

      statusText.textContent =
        "Manuales indexados";

    } else {

      statusEl.classList.remove(
        "is-ready"
      );

      statusEl.classList.remove(
        "is-error"
      );

      renderIndexPrompt(
        statusText
      );

    }


  } catch (err) {

    statusEl.classList.remove(
      "is-ready"
    );

    statusEl.classList.add(
      "is-error"
    );

    statusText.textContent =
      "No se pudo conectar con el backend";

  }

}


// =========================================================
// BOTÓN DE INDEXACIÓN
// =========================================================

function renderIndexPrompt(statusText) {

  statusText.innerHTML = "";


  const span = document.createElement(
    "span"
  );

  span.textContent =
    "Manuales sin indexar — ";


  const btn = document.createElement(
    "button"
  );

  btn.textContent =
    "indexar ahora";


  btn.style.cssText = `
    background:none;
    border:none;
    color:#E0A73B;
    text-decoration:underline;
    cursor:pointer;
    font:inherit;
    padding:0;
  `;


  btn.addEventListener(
    "click",
    async () => {

      btn.textContent =
        "indexando…";

      btn.disabled = true;


      try {

        const res = await fetch(
          API.index,
          {
            method: "POST",
          }
        );


        const data = await res.json();


        if (!res.ok) {

          throw new Error(
            data.detail ||
            "Error al indexar"
          );

        }


        await loadStatus();


      } catch (err) {

        statusText.textContent =
          `Error: ${err.message}`;

      }

    }
  );


  statusText.appendChild(
    span
  );

  statusText.appendChild(
    btn
  );

}


// =========================================================
// CHAT
// =========================================================

const chatWindow = document.getElementById(
  "chat-window"
);

const chatEmpty = document.getElementById(
  "chat-empty"
);

const chatForm = document.getElementById(
  "chat-form"
);

const chatInput = document.getElementById(
  "chat-input"
);

const chatSend = document.getElementById(
  "chat-send"
);


// =========================================================
// HISTORIAL LOCAL
// =========================================================

function loadHistory() {

  try {

    const saved = localStorage.getItem(
      CHAT_STORAGE_KEY
    );


    if (!saved) {
      return [];
    }


    const parsed = JSON.parse(
      saved
    );


    if (!Array.isArray(parsed)) {
      return [];
    }


    return parsed.filter(
      (msg) =>
        msg &&
        (
          msg.role === "user" ||
          msg.role === "assistant"
        ) &&
        typeof msg.content === "string"
    );


  } catch {

    return [];

  }

}


function saveHistory(history) {

  try {

    localStorage.setItem(
      CHAT_STORAGE_KEY,
      JSON.stringify(history)
    );


  } catch {

    /*
      localStorage no disponible.

      El historial no persistirá,
      pero el chat puede continuar funcionando.
    */

  }

}


let history = loadHistory();


// =========================================================
// PREPARAR HISTORIAL PARA EL BACKEND
// =========================================================

function prepareHistoryForBackend(
  historyItems
) {

  return historyItems
    .filter(
      (msg) =>
        msg &&
        (
          msg.role === "user" ||
          msg.role === "assistant"
        ) &&
        typeof msg.content === "string" &&
        msg.content.trim()
    )
    .slice(
      -MAX_HISTORY_MESSAGES
    )
    .map(
      (msg) => ({
        role: msg.role,
        content: msg.content,
      })
    );

}


// =========================================================
// RENDERIZAR HISTORIAL
// =========================================================

function renderHistory() {

  chatWindow.innerHTML = "";


  if (history.length === 0) {

    chatWindow.appendChild(
      chatEmpty
    );

    return;

  }


  history.forEach(
    (msg) => {

      appendMessage(
        msg.role,
        msg.content,
        msg.sources,
        false
      );

    }
  );


  chatWindow.scrollTop =
    chatWindow.scrollHeight;

}


// =========================================================
// AGREGAR MENSAJE AL CHAT
// =========================================================

function appendMessage(
  role,
  content,
  sources,
  scroll = true
) {

  if (
    chatWindow.contains(
      chatEmpty
    )
  ) {

    chatEmpty.remove();

  }


  const bubble = document.createElement(
    "div"
  );


  bubble.className =
    `msg ${role}`;


  bubble.textContent =
    content;


  // =====================================================
  // FUENTES
  // =====================================================

  if (
    Array.isArray(sources) &&
    sources.length > 0
  ) {

    const details = document.createElement(
      "details"
    );

    details.className =
      "msg-sources";


    const summary = document.createElement(
      "summary"
    );

    summary.textContent =
      `📎 Fuentes consultadas (${sources.length})`;


    details.appendChild(
      summary
    );


    sources.forEach(
      (source) => {

        const item = document.createElement(
          "div"
        );

        item.className =
          "source-item";


        const archivo = document.createElement(
          "b"
        );

        archivo.textContent =
          source.archivo ||
          "Documento";


        item.appendChild(
          archivo
        );


        item.appendChild(
          document.createTextNode(
            ` — pág. ${
              source.pagina ?? "?"
            }`
          )
        );


        details.appendChild(
          item
        );

      }
    );


    bubble.appendChild(
      details
    );

  }


  chatWindow.appendChild(
    bubble
  );


  if (scroll) {

    chatWindow.scrollTop =
      chatWindow.scrollHeight;

  }


  return bubble;

}


// =========================================================
// ENVIAR MENSAJE
// =========================================================

async function sendMessage(question) {

  const pregunta = String(
    question || ""
  ).trim();


  if (!pregunta) {
    return;
  }


  /*
    IMPORTANTE:

    Preparamos el historial ANTES de agregar
    la pregunta actual.

    Así el backend recibe:

    historial anterior
    +
    pregunta actual por separado

    y no duplica la pregunta.
  */

  const previousHistory =
    prepareHistoryForBackend(
      history
    );


  // =====================================================
  // 1. GUARDAR MENSAJE DEL USUARIO
  // =====================================================

  history.push(
    {
      role: "user",
      content: pregunta,
    }
  );


  saveHistory(
    history
  );


  appendMessage(
    "user",
    pregunta
  );


  // =====================================================
  // 2. MENSAJE TEMPORAL
  // =====================================================

  const pending = appendMessage(
    "assistant pending",
    "AstreaS está respondiendo…"
  );


  chatSend.disabled = true;


  try {

    // ===================================================
    // 3. ENVIAR PREGUNTA + HISTORIAL
    // ===================================================

    const res = await fetch(
      API.chat,
      {
        method: "POST",

        headers: {
          "Content-Type":
            "application/json",
        },

        body: JSON.stringify(
          {
            question: pregunta,
            historial: previousHistory,
          }
        ),
      }
    );


    // ===================================================
    // 4. PROCESAR RESPUESTA
    // ===================================================

    let data;


    try {

      data = await res.json();

    } catch {

      throw new Error(
        "El servidor devolvió una respuesta no válida."
      );

    }


    if (!res.ok) {

      throw new Error(
        data.detail ||
        "Error al consultar el asistente"
      );

    }


    pending.remove();


    const respuesta =
      data.respuesta ||
      "No fue posible generar una respuesta.";


    const fuentes =
      Array.isArray(
        data.fuentes
      )
        ? data.fuentes
        : [];


    // ===================================================
    // 5. MOSTRAR RESPUESTA
    // ===================================================

    appendMessage(
      "assistant",
      respuesta,
      fuentes
    );


    // ===================================================
    // 6. GUARDAR RESPUESTA EN HISTORIAL
    // ===================================================

    history.push(
      {
        role: "assistant",
        content: respuesta,
        sources: fuentes,
      }
    );


    saveHistory(
      history
    );


  } catch (err) {

    pending.remove();


    const mensajeError =
      err instanceof Error
        ? err.message
        : "Ocurrió un error inesperado.";


    appendMessage(
      "assistant",
      `⚠️ ${mensajeError}`
    );


    /*
      No guardamos el mensaje de error como respuesta
      del asistente dentro del historial conversacional.

      Así evitamos que un error temporal se utilice
      posteriormente para contextualizar preguntas.
    */


  } finally {

    chatSend.disabled = false;

    chatInput.focus();

  }

}


// =========================================================
// FORMULARIO DEL CHAT
// =========================================================

chatForm.addEventListener(
  "submit",
  (event) => {

    event.preventDefault();


    const question =
      chatInput.value.trim();


    if (!question) {
      return;
    }


    chatInput.value = "";

    chatInput.style.height =
      "auto";


    sendMessage(
      question
    );

  }
);


// =========================================================
// TEXTAREA AUTOAJUSTABLE
// =========================================================

chatInput.addEventListener(
  "input",
  () => {

    chatInput.style.height =
      "auto";


    chatInput.style.height =
      Math.min(
        chatInput.scrollHeight,
        140
      ) + "px";

  }
);


// =========================================================
// ENTER PARA ENVIAR
// SHIFT + ENTER PARA SALTO DE LÍNEA
// =========================================================

chatInput.addEventListener(
  "keydown",
  (event) => {

    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {

      event.preventDefault();

      chatForm.requestSubmit();

    }

  }
);


// =========================================================
// NUEVO CHAT / BORRAR HISTORIAL
// =========================================================

const newChatButton =
  document.getElementById(
    "btn-new-chat"
  );


if (newChatButton) {

  newChatButton.addEventListener(
    "click",
    () => {

      history = [];

      saveHistory(
        history
      );

      renderHistory();

      chatInput.focus();

    }
  );

}


// =========================================================
// CONSULTAR FAQ EN EL CHAT
// =========================================================

function askInChat(question) {

  switchView(
    "chat"
  );


  sendMessage(
    question
  );

}


// =========================================================
// SOLUCIONES RÁPIDAS
// =========================================================

let allFaqs = [];

let activeCategory =
  "Todas";

let activeFaqId =
  null;


// =========================================================
// CARGAR FAQS
// =========================================================

async function loadFaqs() {

  try {

    const res = await fetch(
      API.faqs
    );


    if (!res.ok) {

      throw new Error(
        "No fue posible cargar las soluciones rápidas."
      );

    }


    const data = await res.json();


    allFaqs =
      Array.isArray(data)
        ? data
        : [];


  } catch {

    allFaqs = [];

  }


  renderCategories();

  renderFaqList();

}


// =========================================================
// CATEGORÍAS FAQ
// =========================================================

function renderCategories() {

  const wrap =
    document.getElementById(
      "faq-categories"
    );


  if (!wrap) {
    return;
  }


  const categories = [
    "Todas",
    ...new Set(
      allFaqs
        .map(
          (faq) =>
            faq.category
        )
        .filter(Boolean)
    ),
  ];


  wrap.innerHTML = "";


  categories.forEach(
    (category) => {

      const chip =
        document.createElement(
          "button"
        );


      chip.className =
        "chip" +
        (
          category === activeCategory
            ? " is-active"
            : ""
        );


      chip.textContent =
        category;


      chip.addEventListener(
        "click",
        () => {

          activeCategory =
            category;

          activeFaqId =
            null;

          renderCategories();

          renderFaqList();

        }
      );


      wrap.appendChild(
        chip
      );

    }
  );

}


// =========================================================
// FILTRAR FAQS
// =========================================================

function getFilteredFaqs() {

  const search =
    document.getElementById(
      "faq-search"
    );


  const query =
    search
      ? search.value
          .trim()
          .toLowerCase()
      : "";


  return allFaqs.filter(
    (faq) => {

      const category =
        String(
          faq.category || ""
        );


      const title =
        String(
          faq.title || ""
        );


      const summary =
        String(
          faq.summary || ""
        );


      const matchesCategory =
        activeCategory === "Todas" ||
        category === activeCategory;


      const matchesQuery =
        !query ||
        title
          .toLowerCase()
          .includes(query) ||
        summary
          .toLowerCase()
          .includes(query) ||
        category
          .toLowerCase()
          .includes(query);


      return (
        matchesCategory &&
        matchesQuery
      );

    }
  );

}


// =========================================================
// LISTADO DE FAQS
// =========================================================

function renderFaqList() {

  const list =
    document.getElementById(
      "faqs-list"
    );


  if (!list) {
    return;
  }


  const filtered =
    getFilteredFaqs();


  list.innerHTML = "";


  if (
    filtered.length === 0
  ) {

    const empty =
      document.createElement(
        "p"
      );

    empty.className =
      "muted";

    empty.style.padding =
      "12px";

    empty.textContent =
      "No hay casos que coincidan con la búsqueda.";


    list.appendChild(
      empty
    );

    return;

  }


  /*
    Si no existe una FAQ activa dentro
    del conjunto filtrado, seleccionamos
    automáticamente la primera.
  */

  const selectedExists =
    filtered.some(
      (faq) =>
        faq.id === activeFaqId
    );


  if (!selectedExists) {

    activeFaqId =
      filtered[0].id;

  }


  filtered.forEach(
    (faq) => {

      const card =
        document.createElement(
          "div"
        );


      card.className =
        "faq-card" +
        (
          faq.id === activeFaqId
            ? " is-active"
            : ""
        );


      const category =
        document.createElement(
          "span"
        );

      category.className =
        "cat";

      category.textContent =
        faq.category ||
        "General";


      const title =
        document.createElement(
          "h3"
        );

      title.textContent =
        faq.title ||
        "Sin título";


      const summary =
        document.createElement(
          "p"
        );

      summary.textContent =
        faq.summary ||
        "";


      card.appendChild(
        category
      );

      card.appendChild(
        title
      );

      card.appendChild(
        summary
      );


      card.addEventListener(
        "click",
        () => {

          activeFaqId =
            faq.id;

          renderFaqList();

          renderFaqDetail(
            faq
          );

        }
      );


      list.appendChild(
        card
      );

    }
  );


  const selectedFaq =
    filtered.find(
      (faq) =>
        faq.id === activeFaqId
    );


  if (selectedFaq) {

    renderFaqDetail(
      selectedFaq
    );

  }

}


// =========================================================
// DETALLE DE FAQ
// =========================================================

function renderFaqDetail(faq) {

  const detail =
    document.getElementById(
      "faqs-detail"
    );


  if (!detail) {
    return;
  }


  detail.innerHTML = "";


  // Categoría
  const category =
    document.createElement(
      "span"
    );

  category.className =
    "cat-badge";

  category.textContent =
    faq.category ||
    "General";


  // Título
  const title =
    document.createElement(
      "h2"
    );

  title.textContent =
    faq.title ||
    "Sin título";


  // Resumen
  const summary =
    document.createElement(
      "p"
    );

  summary.className =
    "summary";

  summary.textContent =
    faq.summary ||
    "";


  detail.appendChild(
    category
  );

  detail.appendChild(
    title
  );

  detail.appendChild(
    summary
  );


  // =====================================================
  // PASOS
  // =====================================================

  if (
    Array.isArray(
      faq.steps
    ) &&
    faq.steps.length > 0
  ) {

    const block =
      document.createElement(
        "div"
      );

    block.className =
      "detail-block";


    const heading =
      document.createElement(
        "h4"
      );

    heading.textContent =
      "Pasos a seguir";


    const list =
      document.createElement(
        "ol"
      );

    list.className =
      "steps";


    faq.steps.forEach(
      (step) => {

        const item =
          document.createElement(
            "li"
          );

        item.textContent =
          step;

        list.appendChild(
          item
        );

      }
    );


    block.appendChild(
      heading
    );

    block.appendChild(
      list
    );

    detail.appendChild(
      block
    );

  }


  // =====================================================
  // NOTAS
  // =====================================================

  if (
    Array.isArray(
      faq.notes
    ) &&
    faq.notes.length > 0
  ) {

    const block =
      document.createElement(
        "div"
      );

    block.className =
      "detail-block notes-box";


    const heading =
      document.createElement(
        "h4"
      );

    heading.textContent =
      "Ten en cuenta";


    const list =
      document.createElement(
        "ul"
      );


    faq.notes.forEach(
      (note) => {

        const item =
          document.createElement(
            "li"
          );

        item.textContent =
          note;

        list.appendChild(
          item
        );

      }
    );


    block.appendChild(
      heading
    );

    block.appendChild(
      list
    );

    detail.appendChild(
      block
    );

  }


  // =====================================================
  // CONSEJO
  // =====================================================

  if (faq.tip) {

    const block =
      document.createElement(
        "div"
      );

    block.className =
      "detail-block tip-box";


    const heading =
      document.createElement(
        "h4"
      );

    heading.textContent =
      "Consejo";


    const paragraph =
      document.createElement(
        "p"
      );

    paragraph.style.margin =
      "0";

    paragraph.textContent =
      faq.tip;


    block.appendChild(
      heading
    );

    block.appendChild(
      paragraph
    );

    detail.appendChild(
      block
    );

  }


  // =====================================================
  // CONSULTAR EN CHAT
  // =====================================================

  const consultButton =
    document.createElement(
      "button"
    );


  consultButton.className =
    "btn-consult";


  const mainText =
    document.createElement(
      "span"
    );

  mainText.textContent =
    "💬 Consultar este caso en el chat";


  const subText =
    document.createElement(
      "span"
    );

  subText.className =
    "sub";

  subText.textContent =
    "Obtener una respuesta más completa con el asistente";


  consultButton.appendChild(
    mainText
  );

  consultButton.appendChild(
    subText
  );


  consultButton.addEventListener(
    "click",
    () => {

      askInChat(
        faq.chat_query ||
        faq.title
      );

    }
  );


  detail.appendChild(
    consultButton
  );

}


// =========================================================
// BUSCADOR FAQ
// =========================================================

const faqSearch =
  document.getElementById(
    "faq-search"
  );


if (faqSearch) {

  faqSearch.addEventListener(
    "input",
    () => {

      activeFaqId =
        null;

      renderFaqList();

    }
  );

}


// =========================================================
// ARRANQUE
// =========================================================

loadStatus();

loadFaqs();

renderHistory();
