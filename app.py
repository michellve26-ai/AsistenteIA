"""Interfaz de Astrea. Ejecutar con: streamlit run app.py."""

import base64
import html
import inspect
import re
from contextlib import nullcontext

import streamlit as st
from streamlit.errors import StreamlitAPIException

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

PAGE_OPTIONS = {"page_title": UI_NAME, "page_icon": "✨", "layout": "wide"}
try:
    st.set_page_config(**PAGE_OPTIONS, initial_sidebar_state="locked")
except StreamlitAPIException as exc:
    if "locked" not in str(exc):
        raise
    st.set_page_config(**PAGE_OPTIONS, initial_sidebar_state="expanded")

st.html("""
<style>
/* Astrea: tema oscuro y componentes nativos de Streamlit. */
:root, .stApp {
  --astrea-bg: #0b1726;
  --astrea-panel: #192b3d;
  --astrea-panel-hover: #20364d;
  --astrea-ink: #edf4fc;
  --astrea-muted: #aac1d8;
  --astrea-line: #30465d;
  --astrea-blue: #91c8ff;
  --astrea-accent: #2b79c5;
  color-scheme: dark;
}
html, body, .stApp { background: #080f1b; color: var(--astrea-ink); }
.stApp { font-family: Arial, Helvetica, sans-serif; }
.stApp :is(h1, h2, h3, p, label, input, textarea, button), [data-testid="stDialog"] :is(h1, h2, h3, p, label, input, textarea, button) { font-family: Arial, Helvetica, sans-serif !important; }
.stApp .stButton p { letter-spacing: normal; }
[data-testid="stAppViewContainer"] {
  background: radial-gradient(ellipse at 100% 0, #194e78 0, #112d48 31%, transparent 65%), var(--astrea-bg);
}
.stMain, [data-testid="stMain"] { background: transparent; color: var(--astrea-ink); }
[data-testid="stHeader"] { background: transparent !important; color: var(--astrea-muted); }
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"], #MainMenu, footer { display: none; }
[data-testid="stHeader"] button, [data-testid="stSidebarCollapseButton"] button { color: #bad4ee !important; }
[data-testid="stHeader"] [data-testid="stIconMaterial"], [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"] { color: #bad4ee !important; }
[data-testid="stSidebarCollapsedControl"] button { background: #143352; border-radius: 9px; color: #e7f2ff !important; }
.stApp [data-testid="stMainBlockContainer"], .stApp .main .block-container {
  max-width: 870px;
  padding: 24px 28px 26px;
}
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stWidgetLabel"],
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] { color: var(--astrea-ink); }
.stApp [data-testid="stCaptionContainer"] { color: var(--astrea-muted); }
.stApp [data-testid="stCaptionContainer"] p, [data-testid="stDialog"] [data-testid="stCaptionContainer"] p { color: var(--astrea-muted) !important; }
[data-testid="stCaptionContainer"] { opacity: 1 !important; color: #abc3db !important; }
[data-testid="stMarkdownContainer"] blockquote { opacity: 1 !important; color: #c1d3e6 !important; border-left-color: #547da3 !important; }
.stApp [data-testid="stMarkdownContainer"] a { color: #9accff; }

/* Cabecera del espacio de soporte. */
.astrea-header { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; padding: 0 0 16px; margin: 0 0 12px; border-bottom: 1px solid #ffffff14; }
.astrea-header h1 { margin: 0; padding: 0; font-size: 15px; line-height: 1.4; font-weight: 700; color: var(--astrea-ink); }
.astrea-header p { color: var(--astrea-muted); margin: 5px 0 0; font-size: 12px; line-height: 1.5; }
.astrea-status { display: inline-flex; align-items: center; gap: 7px; white-space: nowrap; padding: 8px 12px; font-size: 12px; color: #95c9ff; border: 1px solid #45688b55; border-radius: 99px; background: #142b41c9; box-shadow: 0 5px 12px #06111f22; }
.astrea-status img { display: block; width: 16px; height: 16px; }
.astrea-status.is-pending { color: #e6c288; border-color: #a57b3b60; background: #3b3026; }

/* El panel conserva su apertura y cierre nativos. */
[data-testid="stSidebar"] { background: linear-gradient(155deg, #183f6a 0, #123255 50%, #0e2948 100%); border-right: 1px solid #244461; color: var(--astrea-muted); overflow: hidden; }
[data-testid="stSidebarContent"] { position: relative; z-index: 1; }
[data-testid="stSidebarUserContent"] { padding: .6rem .85rem 1.3rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .7rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #b5cce3; }
.astrea-brand { display: flex; align-items: center; gap: 9px; margin: 0 0 25px; }
.astrea-brand-mark { display: grid; place-items: center; flex: 0 0 auto; width: 35px; height: 37px; border-radius: 11px; border: 1px solid #80b9ed70; color: #fff; font-size: 25px; font-weight: 600; background: linear-gradient(140deg, #5aabeb, #2369a9 75%); box-shadow: inset 0 1px 0 #d8efff30, 0 5px 12px #06152a25; }
.astrea-brand-name { color: #f4f8fe; font-size: 26px; line-height: 1.1; font-weight: 700; letter-spacing: -.9px; }
.astrea-brand-description { color: #a9cae8; font-size: 12px; line-height: 1.5; margin-top: 5px; }
.astrea-menu-label { margin: 25px 12px 7px; color: #87aecf; font-size: 10px; font-weight: 700; letter-spacing: 1.5px; }
.astrea-sidebar-footer { display: flex; align-items: center; gap: 10px; margin-top: 35px; padding: 18px 10px 0; border-top: 1px solid #ffffff16; color: #e5f1ff; font-size: 12px; font-weight: 700; }
.astrea-sidebar-footer span { display: block; color: #98b8d6; font-size: 12px; font-weight: 400; margin-top: 4px; }
.astrea-footer-mark { display: grid; place-items: center; flex: 0 0 auto; width: 29px; height: 29px; border: 1px solid #365876; border-radius: 8px; color: #a8d5ff; font-size: 15px; }

/* Botones y navegación, con foco visible para teclado. */
.stButton > button, [data-testid="stFormSubmitButton"] > button { min-height: 42px; border: 1px solid var(--astrea-line); border-radius: 11px; background: var(--astrea-panel); color: var(--astrea-ink); padding: .6rem .9rem; transition: background 180ms ease, border-color 180ms ease, transform 180ms ease, box-shadow 180ms ease; }
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { color: #fff; border-color: #5480aa; background: var(--astrea-panel-hover); }
.stButton > button:focus-visible, [data-testid="stFormSubmitButton"] > button:focus-visible { outline: 3px solid #90c4fa; outline-offset: 3px; }
.stButton > button p, .stButton > button [data-testid="stMarkdownContainer"] { color: inherit; }
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"] { background: #276fb2; border-color: #3785cb; color: #fff; }
.stButton > button:disabled { opacity: .55; transform: none; box-shadow: none; }
[data-testid="stSidebar"] .stButton > button { justify-content: flex-start; border-color: transparent; background: transparent; color: #b8d1e8; text-align: left; font-size: 14px; }
[data-testid="stSidebar"] .stButton > button:hover { background: #ffffff0b; border-color: #ffffff12; color: #f2f8ff; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] { background: #eff6ff; border-color: #d2e4f6; color: #244f76; }
[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"] { color: inherit; }
.st-key-astrea_new_chat_area { margin-bottom: 16px; }
[data-testid="stSidebar"] .st-key-astrea_new_chat_area .stButton > button { min-height: 48px; justify-content: center; background: linear-gradient(105deg, #3186d3, #2465ad); border: 1px solid #488fd0; color: #fff; box-shadow: 0 6px 14px #04182b26; font-weight: 600; }
[data-testid="stSidebar"] .st-key-astrea_new_chat_area .stButton > button:hover { background: linear-gradient(105deg, #3b91dd, #2a70bc); }
.st-key-astrea_nav_chat .stButton > button, .st-key-astrea_nav_solutions .stButton > button { min-height: 48px; }
.st-key-astrea_nav_chat .stButton > button::before, .st-key-astrea_nav_solutions .stButton > button::before,
[class*="st-key-astrea_sidebar_topic_"] .stButton > button::before { content: ""; flex: 0 0 auto; width: 18px; height: 20px; margin-right: 8px; background-image: var(--astrea-card-icon); background-position: center; background-repeat: no-repeat; background-size: 18px; }
[class*="st-key-astrea_sidebar_topic_"] .stButton > button { min-height: 38px; padding: .4rem .7rem; }
[class*="st-key-astrea_sidebar_topic_"] .stButton > button p { font-size: 13px; }
:is([class*="st-key-astrea_topic_card_"], .st-key-astrea_nav_chat, .st-key-astrea_nav_solutions, [class*="st-key-astrea_sidebar_topic_"]) .stButton > button > div { flex: 1; min-width: 0; justify-content: flex-start !important; text-align: left; }
:is([class*="st-key-astrea_topic_card_"], .st-key-astrea_nav_chat, .st-key-astrea_nav_solutions, [class*="st-key-astrea_sidebar_topic_"]) .stButton > button > div > span { justify-content: flex-start !important; text-align: left; }

/* Bienvenida y tarjetas compactas: todo el botón abre la ficha. */
.astrea-welcome { max-width: 600px; text-align: center; margin: 10px auto 24px; }
.astrea-welcome-mark { position: relative; display: grid; place-items: center; margin: 0 auto 25px; width: 54px; height: 56px; border-radius: 18px; border: 1px solid #71ade0; background: linear-gradient(135deg, #4b9ddd, #1c61a0 75%); color: #eef8ff; box-shadow: inset 0 1px 0 #c2e7ff55, 0 12px 25px #020f2150; transform: rotate(-7deg); animation: astrea-float 5.5s ease-in-out infinite; }
.astrea-welcome-mark::before { content: ""; position: absolute; inset: -7px; border: 1px solid #6b91b650; border-radius: 23px; }
.astrea-welcome-mark img { width: 25px; height: 25px; transform: rotate(7deg); }
.astrea-welcome > p { margin: 0 0 12px; color: #8dc6ff; font-size: 14px; font-weight: 600; }
.astrea-welcome h2 { margin: 0 0 13px; padding: 0; color: #edf4fe; font-size: clamp(27px, 3.1vw, 34px); line-height: 1.2; font-weight: 700; letter-spacing: -1.2px; }
.astrea-welcome h2 span { color: #8fc8ff; }
.astrea-welcome > div:last-child { max-width: 560px; margin: 0 auto; color: var(--astrea-muted); font-size: 14px; line-height: 1.6; }
[class*="st-key-astrea_topic_card_"] { background: transparent; border: 0; padding: 0; animation: astrea-card-in 320ms ease-out backwards; }
.st-key-astrea_topic_card_1 { animation-delay: 45ms; }
.st-key-astrea_topic_card_2 { animation-delay: 90ms; }
.st-key-astrea_topic_card_3 { animation-delay: 135ms; }
[class*="st-key-astrea_topic_card_"] .stButton > button { display: flex; justify-content: flex-start; align-items: center; gap: 11px; width: 100%; min-height: 88px; padding: 15px 13px; border: 1px solid #354c647d; border-radius: 19px; background: #192b3de8; box-shadow: 0 9px 18px #030d1d28; text-align: left; color: #adc6dd; }
[class*="st-key-astrea_topic_card_"] .stButton > button::before { content: ""; display: block; flex: 0 0 auto; width: 36px; height: 40px; border: 1px solid #40638670; border-radius: 11px; background: var(--astrea-card-icon) center / 19px 19px no-repeat, #203d59; }
[class*="st-key-astrea_topic_card_"] .stButton > button [data-testid="stMarkdownContainer"] { text-align: left; min-width: 0; }
[class*="st-key-astrea_topic_card_"] .stButton > button p { color: #adc6dd; font-size: 12px; line-height: 1.6; margin: 0; }
[class*="st-key-astrea_topic_card_"] .stButton > button strong { display: block; margin: 0 0 3px; color: #f0f5fc; font-size: 14px; line-height: 1.35; font-weight: 600; }
[class*="st-key-astrea_topic_card_"] .stButton > button br { display: none; }
[class*="st-key-astrea_topic_card_"] .stButton > button:hover { transform: translateY(-4px); border-color: #608db7; background: #21374de8; box-shadow: 0 14px 23px #020c1d45; }
.astrea-welcome-hint { margin: 20px 0 28px; color: #abc4da; text-align: center; font-size: 12px; line-height: 1.6; }

/* Compositor amplio, siempre oscuro aunque el navegador use otro tema. */
[data-testid="stChatInput"] { position: relative; min-height: 128px; background: #192b3d !important; border: 1px solid #405975 !important; border-radius: 21px !important; color: var(--astrea-ink); padding: 12px 16px 40px; box-shadow: 0 11px 25px #030c1b30; }
[data-testid="stChatInput"]:focus-within { border-color: #75a7d5 !important; box-shadow: 0 0 0 2px #367db826, 0 12px 26px #030c1b45; }
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="base-input"],
[data-testid="stChatInput"] [data-baseweb="input"] { background: transparent !important; border: 0 !important; border-radius: 0 !important; box-shadow: none !important; }
[data-testid="stChatInput"] div { position: static !important; }
[data-testid="stChatInput"] > div { padding: 0 !important; }
[data-testid="stChatInput"] textarea { background: transparent !important; color: #e8f2ff !important; caret-color: #a5d3ff; border: 0 !important; box-shadow: none !important; outline: none; min-height: 48px !important; font-family: inherit; font-size: 14px; line-height: 1.65; padding: 4px 2px !important; }
[data-testid="stChatInput"] textarea::placeholder { color: #adc6de !important; opacity: 1; }
[data-testid="stChatInput"]::after { content: "↵  Enter para enviar"; position: absolute; left: 18px; bottom: 18px; pointer-events: none; color: #abc5de; font-size: 11px; line-height: 1.4; }
[data-testid="stChatInputSubmitButton"] { position: absolute; right: 12px; bottom: 11px; width: 86px; min-height: 35px; padding: 7px 10px; gap: 5px; border: 1px solid #2f6b9e !important; border-radius: 11px; background: #205b92 !important; color: #d9edff !important; }
[data-testid="stChatInputSubmitButton"]::before { content: "Enviar"; font-size: 12px; }
[data-testid="stChatInputSubmitButton"]:disabled { opacity: .48; }
[data-testid="stChatInputSubmitButton"] svg { width: 15px; height: 15px; color: inherit; }
[data-testid="stBottomBlockContainer"], [data-testid="stBottom"] { background: #0b1726 !important; }
[data-testid="stBottomBlockContainer"] { max-width: 870px; padding: 1rem 28px .8rem !important; }
.astrea-footer, .astrea-composer-footer { text-align: center; color: #96b3ce; font-size: 11px; line-height: 1.6; margin: 12px 0 0; }

/* Conversación, documentación y formularios. */
[data-testid="stChatMessage"] { background: #172a3c; border: 1px solid #334b62; border-radius: 7px 18px 18px 18px; padding: 1.1rem; margin-bottom: .8rem; color: var(--astrea-ink); overflow-wrap: anywhere; box-shadow: 0 6px 16px #030c1b22; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: #17395a; border-color: #315878; border-radius: 18px 18px 5px 18px; }
[data-testid="stChatMessageAvatarUser"] { background: #305779 !important; color: #e3f2ff !important; }
[data-testid="stChatMessageAvatarAssistant"] { background: #286ea9 !important; color: #fff !important; }
.astrea-message-name { color: #93c9ff; font-size: 12px; font-weight: 600; margin: 0 0 8px; }
[data-testid="stExpander"] { background: #13263b; border: 1px solid #34516c !important; border-radius: 12px !important; color: #c5d9ed; }
[data-testid="stExpander"] summary { color: #c5d9ed !important; font-size: 13px; }
[data-testid="stSidebar"] [data-testid="stExpander"] { background: #0f2c49a6; border-color: #42618166 !important; margin-top: 15px; }
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea { background: #172a3d !important; color: #eaf4ff !important; caret-color: #a5d3ff; }
[data-testid="stTextInput"] [data-baseweb="input"], [data-testid="stTextInput"] [data-baseweb="base-input"],
[data-testid="stTextArea"] [data-baseweb="textarea"], [data-testid="stSelectbox"] [data-baseweb="select"] > div { background: #172a3d !important; color: #eaf4ff !important; border-color: #405a74 !important; border-radius: 11px; }
[data-testid="stTextInput"] input::placeholder, [data-testid="stTextArea"] textarea::placeholder { color: #94afca; opacity: 1; }
[data-testid="stSelectbox"] svg { fill: #aac9e5; }
[data-testid="stSelectbox"] [role="group"] { background: #172a3d !important; color: #eaf4ff !important; border-color: #405a74 !important; border-radius: 11px; }
[data-testid="stSelectbox"] input[role="combobox"] { background: transparent !important; color: #eaf4ff !important; }
[data-testid="stSelectbox"] button[aria-label="Open"] { background: transparent !important; color: #aac9e5 !important; }
[data-testid="stSelectboxVirtualDropdown"], [data-testid="stSelectboxVirtualDropdown"] [role="listbox"] { background: #172a3d !important; color: #eaf4ff !important; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"] { color: #eaf4ff !important; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"], [data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused] { background: #244969 !important; }
[data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] [role="option"] { background: #162d43; color: #e5f1ff; }
[data-baseweb="popover"] [role="option"]:hover, [data-baseweb="popover"] [aria-selected="true"] { background: #244969; }
[data-testid="stCode"] pre, [data-testid="stCode"] code { background: #0d1c2c !important; color: #d6e7f8 !important; }
[data-testid="stCode"] { border: 1px solid #2d455d; border-radius: 10px; }
[data-testid="stDialog"] [role="dialog"] { background: #102237 !important; color: #eef6ff !important; border: 1px solid #426181; border-radius: 23px; box-shadow: 0 25px 80px #0009; animation: astrea-card-in 180ms ease-out; }
[data-testid="stDialog"] [data-testid="stWidgetLabel"], [data-testid="stDialog"] h2 { color: #edf5ff !important; }
[data-testid="stDialog"] [data-testid="stCaptionContainer"] { color: #abc3db; }
[data-testid="stDialog"] button[aria-label="Close"] { color: #c4d9ef; }
.astrea-topic-heading { margin-bottom: 13px; color: #edf5ff; font-size: 22px; font-weight: 600; }
.astrea-page-intro { margin: 0 0 19px; }
.astrea-page-intro h2 { margin: 0 0 8px; padding: 0; font-size: 27px; color: #edf4fc; letter-spacing: -.8px; }
.astrea-page-intro p { margin: 0; color: #aac1d8; font-size: 14px; }
.astrea-category { display: inline-block; margin-bottom: 7px; padding: 5px 9px; border: 1px solid #355d80; border-radius: 8px; background: #1a3b5a; color: #a5d0fa; font-size: 11px; font-weight: 600; }
.astrea-detail-title { margin: 3px 0 17px; padding: 0; color: #edf5ff; font-size: 22px; line-height: 1.4; }
[class*="st-key-astrea_faq_"] { background: #14283b; border-color: #334d66; border-radius: 17px; }
.st-key-astrea_latest_response { animation: astrea-message-in 240ms ease-out; }
@keyframes astrea-message-in { from { opacity: .4; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
@keyframes astrea-card-in { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }
@keyframes astrea-float { 0%, 100% { transform: translateY(0) rotate(-7deg); } 50% { transform: translateY(-5px) rotate(-4deg); } }

@media (min-width: 761px) {
  .stApp .stMain { height: calc(100dvh - 34px) !important; max-height: calc(100dvh - 34px); min-height: 0; }
  [data-testid="stAppViewContainer"] { position: absolute; inset: 16px; width: auto; height: auto; max-width: 1120px; margin: 0 auto; border: 1px solid #293e53; border-radius: 28px; overflow: hidden; box-shadow: 0 18px 45px #0003; }
  [data-testid="stSidebar"] { width: 244px !important; min-width: 244px !important; max-width: 244px !important; height: calc(100% - 20px); margin: 10px 0 10px 10px; border-radius: 20px; border-right: 0; }
  [data-testid="stSidebarHeader"] { display: none; }
  [data-testid="stSidebarContent"] { height: 100%; padding: 20px 14px 16px !important; overflow-y: auto; box-sizing: border-box; }
  [data-testid="stSidebarUserContent"] { margin: 0 !important; padding: 0 !important; }
  [data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] { min-height: calc(100dvh - 90px); }
  [data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has(> .st-key-astrea_sidebar_footer) { margin-top: auto; }
  .st-key-astrea_sidebar_footer { margin-top: auto; }
  .astrea-sidebar-footer { margin-top: 0; }
  [data-testid="stSidebar"]::after { content: ""; position: absolute; right: -160px; bottom: 55px; width: 330px; height: 330px; border: 28px solid #25517913; border-radius: 50%; pointer-events: none; }
}
@media (max-width: 760px) {
  [data-testid="stBottomBlockContainer"] { padding-left: 1rem !important; padding-right: 1rem !important; }
  [data-testid="stAppViewContainer"] { inset: 0; margin: 0; width: 100%; max-width: none; border: 0; border-radius: 0; }
  .stApp [data-testid="stMainBlockContainer"], .stApp .main .block-container { padding: 4rem 1rem 2rem; }
  [data-testid="stHeader"] { background: #0e2035f5 !important; }
  .astrea-header { margin-bottom: 22px; padding-bottom: 17px; gap: 10px; }
  .astrea-header h1 { font-size: 15px; }
  .astrea-header p { font-size: 11px; }
  .astrea-status { font-size: 10px; padding: 7px 9px; }
  .astrea-welcome { margin: .9rem auto 1.5rem; }
  .astrea-welcome h2 { font-size: 29px; letter-spacing: -.9px; }
  .astrea-welcome > div:last-child { font-size: 13px; }
  .astrea-welcome-mark { width: 49px; height: 51px; }
  .astrea-welcome-hint { margin: 14px 0 20px; }
  [class*="st-key-astrea_topic_card_"] .stButton > button { min-height: 84px; }
  [data-testid="stChatInput"] textarea { font-size: 16px; }
  [data-testid="stChatMessage"] { padding: .85rem; }
  .astrea-detail-title { font-size: 20px; }
}
@media (prefers-reduced-motion: reduce) {
  .stButton > button, [class*="st-key-astrea_topic_card_"], .astrea-welcome-mark,
  .st-key-astrea_latest_response, [data-testid="stDialog"] [role="dialog"] { transition: none; animation: none; }
  [class*="st-key-astrea_topic_card_"] .stButton > button:hover { transform: none; }
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


def icon_data(category: str, color: str = "#a4c9ea") -> str:
    """Iconos locales, compatibles con la sanitización de HTML de Streamlit."""
    icons = {
        "Facturación": '<path d="M6 3h12v18l-3-2-3 2-3-2-3 2V3Z"/><path d="M9 7h6M9 11h6M9 15h3"/>',
        "Inventario": '<path d="m12 3 9 5v8l-9 5-9-5V8l9-5Z"/><path d="m3 8 9 5 9-5M12 13v8M7.5 5.5l9 5"/>',
        "Cartera": '<rect x="3" y="5" width="18" height="15" rx="3"/><path d="M3 9h18M15 14h6"/><circle cx="16" cy="14" r=".5"/>',
        "Cajas y bancos": '<path d="m3 9 9-6 9 6H3ZM5 10v7M10 10v7M14 10v7M19 10v7M3 21h18M4 18h16"/>',
        "sparkles": '<path d="m10 3 2.1 5.9L18 11l-5.9 2.1L10 19l-2.1-5.9L2 11l5.9-2.1L10 3ZM19 2l.9 2.1L22 5l-2.1.9L19 8l-.9-2.1L16 5l2.1-.9L19 2ZM19 15l1.2 2.8L23 19l-2.8 1.2L19 23l-1.2-2.8L15 19l2.8-1.2L19 15Z"/>',
        "chat": '<path d="M4 3h12a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2H8l-5 4V5a2 2 0 0 1 1-2Z"/><path d="M9 17h7l5 4V10"/>',
        "book": '<path d="M12 5v16M3 4h5a4 4 0 0 1 4 2 4 4 0 0 1 4-2h5v15h-5a4 4 0 0 0-4 2 4 4 0 0 0-4-2H3V4Z"/>',
    }
    paths = icons.get(category, '<path d="M4 4h7a3 3 0 0 1 3 3v14a4 4 0 0 0-4-2H4V4ZM14 7a3 3 0 0 1 3-3h3v15h-2a4 4 0 0 0-4 2"/>')
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" '
        'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" '
        f'>{paths}</svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def icon_image(name: str, size: int = 22, color: str = "#a4c9ea") -> str:
    return (
        f'<img src="{icon_data(name, color)}" width="{size}" height="{size}" '
        'alt="" aria-hidden="true">'
    )


def help_categories() -> list[str]:
    available = {faq["category"] for faq in FAQS}
    preferred = ("Facturación", "Inventario", "Cartera", "Cajas y bancos")
    categories = [category for category in preferred if category in available]
    categories.extend(sorted(available.difference(categories)))
    return categories[:4]


def render_icon_styles() -> None:
    rules = []
    for position, category in enumerate(help_categories()):
        rules.append(
            f'.st-key-astrea_topic_card_{position}, .st-key-astrea_sidebar_topic_{position}'
            f'{{ --astrea-card-icon: url("{icon_data(category)}"); }}'
        )
    for key, name in (("astrea_nav_chat", "chat"), ("astrea_nav_solutions", "book")):
        rules.append(f'.st-key-{key} {{ --astrea-card-icon: url("{icon_data(name)}"); }}')
        rules.append(
            f'.st-key-{key} button[kind="primary"]'
            f'{{ --astrea-card-icon: url("{icon_data(name, "#315e87")}"); }}'
        )
    st.html("<style>" + "".join(rules) + "</style>")


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
        with ui_container("astrea_nav_chat", border=False):
            st.button(
                "Chat de soporte",
                key="nav_chat",
                type="primary" if st.session_state.main_view == "Chat" else "secondary",
                use_container_width=True,
                on_click=select_view,
                args=("Chat",),
            )
        with ui_container("astrea_nav_solutions", border=False):
            st.button(
                "Soluciones rápidas",
                key="nav_solutions",
                type="primary" if st.session_state.main_view == "Soluciones rápidas" else "secondary",
                use_container_width=True,
                on_click=select_view,
                args=("Soluciones rápidas",),
            )
        st.html('<p class="astrea-menu-label">TEMAS DE AYUDA</p>')
        for position, category in enumerate(help_categories()):
            with ui_container(f"astrea_sidebar_topic_{position}", border=False):
                st.button(
                    display_text(category),
                    key=f"astrea_sidebar_topic_button_{position}",
                    use_container_width=True,
                    on_click=show_category,
                    args=(category,),
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
        with ui_container("astrea_sidebar_footer", border=False):
            st.html(
                '<div class="astrea-sidebar-footer">'
                '<div class="astrea-footer-mark" aria-hidden="true">A</div>'
                f'<div>{escaped(UI_NAME).upper()}<span>Centro de soporte</span></div></div>'
            )
    return top_k


def render_header(index_ready: bool) -> None:
    is_chat = st.session_state.main_view == "Chat"
    title = "Centro de soporte" if is_chat else "Soluciones rápidas"
    st.html(
        '<header class="astrea-header">'
        f'<div><h1>{title}</h1><p>Tu espacio de ayuda con {escaped(UI_NAME)}</p></div>'
        f'<span class="astrea-status">{icon_image("sparkles", 16)}Asistente virtual</span>'
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
        f'<div class="astrea-welcome-mark" aria-hidden="true">{icon_image("sparkles", 25, "#e5f4ff")}</div>'
        f'<p>Hola, soy {escaped(UI_NAME)}.</p>'
        '<h2>¿Qué <span>resolvemos</span> hoy?</h2>'
        '<div>Un espacio para tus preguntas. Encuentra ayuda con pasos claros y fuentes a la mano.</div>'
        '</section>'
    )
    categories = help_categories()
    descriptions = {
        "Facturación": "Facturas y vencimientos",
        "Inventario": "Ayuda con tus productos",
        "Cartera": "Consultas sobre este módulo",
        "Cajas y bancos": "Orientación paso a paso",
    }
    selected_topic = None
    for row_start in range(0, len(categories), 2):
        columns = st.columns(2, gap="medium")
        for column, category in zip(columns, categories[row_start:row_start + 2]):
            position = categories.index(category)
            with column:
                with ui_container(f"astrea_topic_card_{position}", border=False):
                    if st.button(
                        f"**{display_text(category)}**  \n{descriptions.get(category, 'Guías de soporte')}",
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
        input_options = {"height": 130} if "height" in inspect.signature(st.chat_input).parameters else {}
        typed_question = st.chat_input(
            "Escribe tu pregunta a Astrea…",
            key="astrea_chat_input",
            on_submit=queue_chat_question,
            **input_options,
        )
    if show_welcome:
        st.html('<p class="astrea-footer">Soporte de Astrea · Consulta las fuentes de cada respuesta.</p>')
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


render_icon_styles()
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

# FIN DEL ARCHIVO APP.PY
