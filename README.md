# Asistente de Soporte Técnico (RAG con Groq)

Chatbot que responde preguntas del equipo de soporte técnico usando **únicamente**
la información de los manuales internos (PDF), para evitar que tengan que
buscar manualmente dentro de ellos.

## Estructura del proyecto

```
asistente-soporte-rag/
├── data/
│   └── pdfs/              ← coloca aquí los manuales/reglamentos en PDF
├── src/
│   ├── __init__.py
│   ├── config.py           ← configuración (modelo, chunking, rutas, nombre del asistente)
│   ├── ingest.py            ← carga de PDFs + chunking
│   ├── vectorstore.py       ← embeddings locales + ChromaDB
│   └── rag_chain.py         ← prompt aumentado + pipeline RAG con Groq
├── chroma_db/                ← se genera automáticamente al indexar (no tocar a mano)
├── static/
│   ├── index.html             ← interfaz nueva (chat + soluciones rápidas)
│   ├── style.css
│   └── app.js
├── api.py                     ← backend FastAPI (sirve la interfaz + endpoints /api/*)
├── app.py                     ← interfaz alterna en Streamlit (opcional, ya no es la principal)
├── requirements.txt
├── .env.example
└── README.md
```

## 1. Instalación (VS Code)

1. Abre la carpeta `asistente-soporte-rag` en VS Code.
2. Crea un entorno virtual:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Configura tu API key de Groq

1. Copia `.env.example` a `.env`:
   ```bash
   cp .env.example .env
   ```
2. Consigue tu key gratis en https://console.groq.com (API Keys → Create API Key).
3. Pégala en `.env` en `GROQ_API_KEY`.

## 3. Agrega los manuales

Copia los PDF de soporte técnico dentro de `data/pdfs/`.

## 4. Lanza la interfaz

```bash
uvicorn api:app --reload
```

Abre en el navegador: **http://127.0.0.1:8000**

La interfaz tiene dos pestañas en el menú lateral:

- **Chat** — conversación libre con el asistente. El historial se guarda en el
  navegador (localStorage), así que persiste entre recargas de la página.
- **Soluciones rápidas** — un panel de casos frecuentes con buscador y filtro
  por categoría, editable en `data/faqs.json`. Cada caso trae pasos, notas y un
  botón **"Consultar este caso en el chat"** que abre el chat con la pregunta
  ya escrita para pedir más detalle al RAG.

La primera vez (o cuando cambies los PDF de `data/pdfs`), pulsa
**"indexar ahora"** en el aviso que aparece abajo del menú lateral.

### Personalizar "Soluciones rápidas"

Edita `data/faqs.json`: cada objeto es un caso, con `category`, `title`,
`summary`, `steps` (lista de pasos), `notes` (advertencias, opcional), `tip`
(opcional) y `chat_query` (la pregunta que se envía al chat si pulsan el
botón). Los que vienen por defecto son solo ejemplos — reemplázalos por los
casos reales de tu empresa.

### Interfaz alterna en Streamlit (opcional)

Si prefieres la versión más simple, `app.py` sigue funcionando igual:
```bash
streamlit run app.py
```
Ambas interfaces comparten el mismo backend RAG en `src/`.

## Notas

- Los embeddings corren localmente (Sentence Transformers, CPU) — no gastan cuota de API.
- Solo la generación de la respuesta final usa la API de Groq (gratuita, con límites generosos).
- Personaliza el nombre del asistente y de la empresa en `src/config.py`
  (`ASSISTANT_NAME`, `EMPRESA`) y el tono de las respuestas en `src/rag_chain.py`
  (`PROMPT_TEMPLATE_RAW`).
- Si quieres correr todo por consola sin interfaz (como en el notebook original),
  puedes importar directamente las funciones de `src/` desde un script propio.
