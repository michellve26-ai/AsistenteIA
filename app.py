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
