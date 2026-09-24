// ---------- Configuración de marca (ajusta si cambias EMPRESA/ASSISTANT_NAME en src/config.py) ----------
const BRAND_FALLBACK = { assistant_name: "Asistente", empresa: "Soporte técnico" };

const API = {
  status: "/api/status",
  faqs: "/api/faqs",
  chat: "/api/chat",
  index: "/api/index",
};

const CHAT_STORAGE_KEY = "soporte_chat_historial";

// ---------- Navegación entre vistas ----------
const navItems = document.querySelectorAll(".nav-item");
const views = document.querySelectorAll(".view");
const topbarTitle = document.getElementById("topbar-title");
const topbarSub = document.getElementById("topbar-sub");

const VIEW_META = {
  chat: { title: "Chat", sub: "Pregunta lo que necesites sobre los manuales internos." },
  faqs: { title: "Soluciones rápidas", sub: "Encuentra aquí las respuestas a los casos más comunes." },
};

function switchView(name) {
  navItems.forEach((btn) => btn.classList.toggle("is-active", btn.dataset.view === name));
  views.forEach((v) => v.classList.toggle("is-active", v.id === `view-${name}`));
  topbarTitle.textContent = VIEW_META[name].title;
  topbarSub.textContent = VIEW_META[name].sub;
}

navItems.forEach((btn) => btn.addEventListener("click", () => switchView(btn.dataset.view)));

// ---------- Estado / marca e índice ----------
async function loadStatus() {
  const statusEl = document.getElementById("index-status");
  const statusText = document.getElementById("index-status-text");
  try {
    const res = await fetch(API.status);
    const data = await res.json();

    document.getElementById("brand-name").textContent = data.assistant_name || BRAND_FALLBACK.assistant_name;
    document.getElementById("brand-sub").textContent = data.empresa || BRAND_FALLBACK.empresa;

    if (data.indexed) {
      statusEl.classList.add("is-ready");
      statusEl.classList.remove("is-error");
      statusText.textContent = "Manuales indexados";
    } else {
      statusEl.classList.remove("is-ready");
      renderIndexPrompt(statusText);
    }
  } catch (err) {
    statusEl.classList.add("is-error");
    statusText.textContent = "No se pudo conectar con el backend";
  }
}

function renderIndexPrompt(statusText) {
  statusText.innerHTML = "";
  const span = document.createElement("span");
  span.textContent = "Manuales sin indexar — ";
  const btn = document.createElement("button");
  btn.textContent = "indexar ahora";
  btn.style.cssText = "background:none;border:none;color:#E0A73B;text-decoration:underline;cursor:pointer;font:inherit;padding:0;";
  btn.addEventListener("click", async () => {
    btn.textContent = "indexando…";
    btn.disabled = true;
    try {
      const res = await fetch(API.index, { method: "POST" });
      if (!res.ok) throw new Error((await res.json()).detail || "Error al indexar");
      await loadStatus();
    } catch (err) {
      statusText.textContent = `Error: ${err.message}`;
    }
  });
  statusText.appendChild(span);
  statusText.appendChild(btn);
}

// ---------- Chat ----------
const chatWindow = document.getElementById("chat-window");
const chatEmpty = document.getElementById("chat-empty");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(CHAT_STORAGE_KEY)) || [];
  } catch {
    return [];
  }
}

function saveHistory(history) {
  try {
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(history));
  } catch {
    /* localStorage no disponible: el historial no persiste, pero el chat sigue funcionando */
  }
}

let history = loadHistory();

function renderHistory() {
  chatWindow.innerHTML = "";
  if (history.length === 0) {
    chatWindow.appendChild(chatEmpty);
    return;
  }
  history.forEach((msg) => appendMessage(msg.role, msg.content, msg.sources, false));
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function appendMessage(role, content, sources, scroll = true) {
  if (chatWindow.contains(chatEmpty)) chatEmpty.remove();

  const bubble = document.createElement("div");
  bubble.className = `msg ${role}`;
  bubble.textContent = content;

  if (sources && sources.length) {
    const details = document.createElement("details");
    details.className = "msg-sources";
    const summary = document.createElement("summary");
    summary.textContent = `📎 Fuentes consultadas (${sources.length})`;
    details.appendChild(summary);
    sources.forEach((s) => {
      const item = document.createElement("div");
      item.className = "source-item";
      item.innerHTML = `<b>${s.archivo}</b> — pág. ${s.pagina}`;
      details.appendChild(item);
    });
    bubble.appendChild(details);
  }

  chatWindow.appendChild(bubble);
  if (scroll) chatWindow.scrollTop = chatWindow.scrollHeight;
  return bubble;
}

async function sendMessage(question) {
  history.push({ role: "user", content: question });
  saveHistory(history);
  appendMessage("user", question);

  const pending = appendMessage("assistant pending", "Buscando en los manuales…");
  chatSend.disabled = true;

  try {
    const res = await fetch(API.chat, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Error al consultar el asistente");

    pending.remove();
    appendMessage("assistant", data.respuesta, data.fuentes);
    history.push({ role: "assistant", content: data.respuesta, sources: data.fuentes });
    saveHistory(history);
  } catch (err) {
    pending.remove();
    appendMessage("assistant", `⚠️ ${err.message}`);
  } finally {
    chatSend.disabled = false;
  }
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = chatInput.value.trim();
  if (!question) return;
  chatInput.value = "";
  chatInput.style.height = "auto";
  sendMessage(question);
});

chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + "px";
});

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

document.getElementById("btn-new-chat").addEventListener("click", () => {
  history = [];
  saveHistory(history);
  renderHistory();
});

// Permite que un caso de "Soluciones rápidas" abra el chat con una pregunta precargada.
function askInChat(question) {
  switchView("chat");
  sendMessage(question);
}

// ---------- Soluciones rápidas (FAQs) ----------
let allFaqs = [];
let activeCategory = "Todas";
let activeFaqId = null;

async function loadFaqs() {
  try {
    const res = await fetch(API.faqs);
    allFaqs = await res.json();
  } catch {
    allFaqs = [];
  }
  renderCategories();
  renderFaqList();
}

function renderCategories() {
  const wrap = document.getElementById("faq-categories");
  const categories = ["Todas", ...new Set(allFaqs.map((f) => f.category))];
  wrap.innerHTML = "";
  categories.forEach((cat) => {
    const chip = document.createElement("button");
    chip.className = "chip" + (cat === activeCategory ? " is-active" : "");
    chip.textContent = cat;
    chip.addEventListener("click", () => {
      activeCategory = cat;
      renderCategories();
      renderFaqList();
    });
    wrap.appendChild(chip);
  });
}

function getFilteredFaqs() {
  const query = document.getElementById("faq-search").value.trim().toLowerCase();
  return allFaqs.filter((f) => {
    const matchesCategory = activeCategory === "Todas" || f.category === activeCategory;
    const matchesQuery =
      !query ||
      f.title.toLowerCase().includes(query) ||
      f.summary.toLowerCase().includes(query) ||
      f.category.toLowerCase().includes(query);
    return matchesCategory && matchesQuery;
  });
}

function renderFaqList() {
  const list = document.getElementById("faqs-list");
  const filtered = getFilteredFaqs();
  list.innerHTML = "";

  if (filtered.length === 0) {
    list.innerHTML = `<p class="muted" style="padding:12px;">No hay casos que coincidan con la búsqueda.</p>`;
    return;
  }

  filtered.forEach((faq) => {
    const card = document.createElement("div");
    card.className = "faq-card" + (faq.id === activeFaqId ? " is-active" : "");
    card.innerHTML = `
      <span class="cat">${faq.category}</span>
      <h3>${faq.title}</h3>
      <p>${faq.summary}</p>
    `;
    card.addEventListener("click", () => {
      activeFaqId = faq.id;
      renderFaqList();
      renderFaqDetail(faq);
    });
    list.appendChild(card);
  });

  // Si nada está seleccionado aún, muestra el primer resultado.
  if (!activeFaqId && filtered.length) {
    activeFaqId = filtered[0].id;
    renderFaqList();
    renderFaqDetail(filtered[0]);
  }
}

function renderFaqDetail(faq) {
  const detail = document.getElementById("faqs-detail");
  detail.innerHTML = `
    <span class="cat-badge">${faq.category}</span>
    <h2>${faq.title}</h2>
    <p class="summary">${faq.summary}</p>

    <div class="detail-block">
      <h4>Pasos a seguir</h4>
      <ol class="steps">
        ${faq.steps.map((s) => `<li>${s}</li>`).join("")}
      </ol>
    </div>

    ${faq.notes && faq.notes.length ? `
    <div class="detail-block notes-box">
      <h4>Ten en cuenta</h4>
      <ul>${faq.notes.map((n) => `<li>${n}</li>`).join("")}</ul>
    </div>` : ""}

    ${faq.tip ? `
    <div class="detail-block tip-box">
      <h4>Consejo</h4>
      <p style="margin:0;">${faq.tip}</p>
    </div>` : ""}

    <button class="btn-consult" id="btn-consult">
      <span>💬 Consultar este caso en el chat</span>
      <span class="sub">Obtener una respuesta más completa con el asistente</span>
    </button>
  `;
  document.getElementById("btn-consult").addEventListener("click", () => {
    askInChat(faq.chat_query || faq.title);
  });
}

document.getElementById("faq-search").addEventListener("input", renderFaqList);

// ---------- Arranque ----------
loadStatus();
loadFaqs();
renderHistory();
