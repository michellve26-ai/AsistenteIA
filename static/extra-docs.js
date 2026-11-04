// =========================================================
// MANUALES EXTRA — agregar un PDF desde el chat
// =========================================================
//
// Archivo independiente: se carga DESPUÉS de app.js y no lo modifica.
// Crea por sí solo el botón 📎 dentro del formulario del chat y permite
// arrastrar un PDF sobre el chat. El PDF se envía a /api/extra-docs y
// queda como referencia extra (no toca la indexación automática).
//
// Usa de app.js solo dos funciones que ya existen: appendMessage y switchView.

(function () {
  "use strict";

  const ENDPOINT = "/api/extra-docs";
  const TOKEN_KEY = "soporte_upload_token";

  const form = document.getElementById("chat-form");
  const viewChat = document.getElementById("view-chat");

  if (
    !form ||
    typeof appendMessage !== "function" ||
    typeof switchView !== "function"
  ) {
    // La página no tiene lo necesario: la función simplemente no se activa.
    return;
  }

  let maxMb = 15;
  let maxMbLeido = false;
  let subiendo = false;


  // ---------- Interfaz: botón 📎 y selector de archivo ----------

  const input = document.createElement("input");
  input.type = "file";
  input.accept = "application/pdf,.pdf";
  input.multiple = true;
  input.hidden = true;

  const boton = document.createElement("button");
  boton.type = "button";
  boton.id = "btn-attach";
  boton.className = "btn-attach";
  boton.title = "Agregar un manual PDF como referencia extra";
  boton.setAttribute("aria-label", "Agregar un manual PDF como referencia extra");
  boton.textContent = "📎";

  form.insertBefore(boton, form.firstChild);
  form.appendChild(input);

  boton.addEventListener("click", () => input.click());
  input.addEventListener("change", () => subirPdfs(input.files));


  // ---------- Clave opcional (UPLOAD_TOKEN del servidor) ----------

  function leerClave() {
    try {
      return sessionStorage.getItem(TOKEN_KEY) || "";
    } catch {
      return "";
    }
  }

  function guardarClave(valor) {
    try {
      if (valor) {
        sessionStorage.setItem(TOKEN_KEY, valor);
      } else {
        sessionStorage.removeItem(TOKEN_KEY);
      }
    } catch {
      /* sin sessionStorage: se pedirá la clave cada vez */
    }
  }

  async function leerDetalle(res, respaldo) {
    try {
      const data = await res.json();
      return typeof data.detail === "string" ? data.detail : respaldo;
    } catch {
      return respaldo;
    }
  }

  // Si el servidor responde 401, pide la clave y reintenta una vez.
  async function enviar(url, opciones) {
    const intentar = () => {
      const headers = { ...(opciones.headers || {}) };
      const clave = leerClave();

      if (clave) {
        headers["X-Upload-Token"] = clave;
      }

      return fetch(url, { ...opciones, headers });
    };

    let res = await intentar();

    if (res.status === 401) {
      const escrita = window.prompt(
        "Esta función está protegida. Escribe la clave para agregar manuales:"
      );

      if (!escrita) {
        throw new Error("Se necesita la clave para agregar manuales.");
      }

      guardarClave(escrita.trim());
      res = await intentar();

      if (res.status === 401) {
        guardarClave("");
        throw new Error("Clave incorrecta.");
      }
    }

    return res;
  }


  // ---------- Subida ----------

  async function leerLimite() {
    if (maxMbLeido) {
      return;
    }

    maxMbLeido = true;

    try {
      const res = await fetch(ENDPOINT);

      if (res.ok) {
        const data = await res.json();

        if (data.max_mb) {
          maxMb = data.max_mb;
        }
      }
    } catch {
      /* se usa el límite por defecto */
    }
  }

  async function subirPdf(archivo) {
    if (!/\.pdf$/i.test(archivo.name)) {
      appendMessage("assistant", `⚠️ «${archivo.name}» no es un PDF. Solo se pueden agregar archivos .pdf.`);
      return;
    }

    if (archivo.size > maxMb * 1024 * 1024) {
      appendMessage("assistant", `⚠️ «${archivo.name}» supera el máximo de ${maxMb} MB.`);
      return;
    }

    const pendiente = appendMessage(
      "assistant pending",
      `Indexando «${archivo.name}»… puede tardar un momento.`
    );

    try {
      const res = await enviar(
        `${ENDPOINT}?nombre=${encodeURIComponent(archivo.name)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/pdf" },
          body: archivo,
        }
      );

      if (!res.ok) {
        throw new Error(await leerDetalle(res, "No fue posible agregar el manual."));
      }

      const data = await res.json();

      pendiente.remove();

      appendMessage(
        "assistant",
        `📎 Listo: ${data.reemplazado ? "actualicé" : "agregué"} «${data.archivo}» como referencia extra ` +
        `(${data.paginas} págs., ${data.fragmentos} fragmentos). Ya puedes preguntarme sobre su contenido.`
      );

    } catch (error) {
      pendiente.remove();

      appendMessage(
        "assistant",
        `⚠️ ${error instanceof Error ? error.message : "No fue posible agregar el manual."}`
      );
    }
  }

  async function subirPdfs(archivos) {
    const lista = Array.from(archivos || []);

    if (subiendo || lista.length === 0) {
      return;
    }

    subiendo = true;
    boton.disabled = true;

    switchView("chat");

    try {
      await leerLimite();

      for (const archivo of lista) {
        await subirPdf(archivo);
      }
    } finally {
      subiendo = false;
      boton.disabled = false;
      input.value = "";
    }
  }


  // ---------- Arrastrar un PDF sobre el chat ----------

  if (viewChat) {
    const traeArchivos = (evento) =>
      evento.dataTransfer &&
      Array.from(evento.dataTransfer.types || []).includes("Files");

    ["dragenter", "dragover"].forEach((tipo) => {
      viewChat.addEventListener(tipo, (evento) => {
        if (!traeArchivos(evento)) {
          return;
        }

        evento.preventDefault();
        viewChat.classList.add("is-dragover");
      });
    });

    viewChat.addEventListener("dragleave", (evento) => {
      if (!viewChat.contains(evento.relatedTarget)) {
        viewChat.classList.remove("is-dragover");
      }
    });

    viewChat.addEventListener("drop", (evento) => {
      if (!traeArchivos(evento)) {
        return;
      }

      evento.preventDefault();
      viewChat.classList.remove("is-dragover");
      subirPdfs(evento.dataTransfer.files);
    });
  }
})();
