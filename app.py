from __future__ import annotations

import io
import math
import html as html_lib
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

# Default every Plotly figure to the same high-contrast dark instrument theme.
pio.templates.default = "plotly_dark"
import streamlit as st

# ============================================================================
# QUANTITATIVE NUCLEAR MEDICINE LABORATORY 3.3.8 - custom navigation edition
# Educational simulator only. Not an instrument controller or validated tool.
# ============================================================================

st.set_page_config(
    page_title="Quantitative Nuclear Medicine Laboratory 3.3.8 | Instrument Guide & Practical Workshop",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RADIONUCLIDES: Dict[str, Dict[str, float | str]] = {
    "18F": {"half_life_min": 109.77, "energy_keV": 511.0},
    "68Ga": {"half_life_min": 67.71, "energy_keV": 511.0},
    "89Zr": {"half_life_min": 78.41 * 60, "energy_keV": 909.0},
    "99mTc": {"half_life_min": 6.0067 * 60, "energy_keV": 140.5},
    "64Cu": {"half_life_min": 12.701 * 60, "energy_keV": 511.0},
    "177Lu": {"half_life_min": 6.647 * 24 * 60, "energy_keV": 208.4},
    "131I": {"half_life_min": 8.025 * 24 * 60, "energy_keV": 364.5},
}
DEFAULT_SENS = {"18F": .145, "68Ga": .132, "89Zr": .085, "99mTc": .110, "64Cu": .125, "177Lu": .060, "131I": .075}
DETECTOR_REL = {"A": .965, "B": 1.018, "C": .992, "D": 1.041, "E": .984, "F": 1.012}
PALETTE = ["#58B9FF", "#55E0B0", "#FFC45B", "#C6A5FF", "#FF829A", "#46D9D0", "#E1EAF0"]

# ----------------------------------- styling ---------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500;600;700&family=DM+Sans:wght@400;500;600;700;800&display=swap');
:root {
  color-scheme: dark;
  --bg: #08131e;
  --surface: #112231;
  --surface-2: #192f40;
  --surface-3: #233d50;
  --border: #3d596c;
  --text: #f7fafc;
  --text-2: #e1eaf0;
  --muted: #c4d2dc;
  --accent: #55e0b0;
  --accent-2: #ffc45b;
  --tab-idle: #263f52;
  --tab-hover: #36596f;
  --tab-active: #55e0b0;
}
html, body, [class*="css"] {font-family:'DM Sans',Arial,sans-serif;color:var(--text);color-scheme:dark;}
.stApp, [data-testid="stAppViewContainer"] {background:var(--bg)!important;color:var(--text)!important;}
.block-container {max-width:1580px;padding-top:1.1rem;padding-bottom:2.4rem;}
/* Global text: use bright, non-muted text on the dark canvas. */
.stApp p,.stApp li,.stApp label,.stApp small,.stApp span,.stApp strong,
.stApp [data-testid="stMarkdownContainer"],.stApp [data-testid="stMarkdownContainer"] *,
.stApp [data-testid="stWidgetLabel"],.stApp [data-testid="stWidgetLabel"] *,
.stApp [data-testid="stCaptionContainer"],.stApp [data-testid="stCaptionContainer"] *,
.stApp [data-testid="stHelp"],.stApp [data-testid="stTooltipContent"] * {color:var(--text-2)!important;}
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp h6 {color:#ffffff!important;font-weight:800!important;}
.stApp [data-testid="stCaptionContainer"] p,.stApp .note {color:var(--muted)!important;}
/* Sidebar: deep navy with bright text and clearly surfaced controls. */
[data-testid="stSidebar"] {background:#0d1c29!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] * {color:var(--text-2)!important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {color:#ffffff!important;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {color:#d7e3eb!important;}
/* Inputs: dark inset controls, bright values, no low-opacity placeholders. */
.stApp input,.stApp textarea,.stApp [data-baseweb="input"] input,
.stApp [data-baseweb="select"] input {background:#0b1823!important;color:#ffffff!important;
  -webkit-text-fill-color:#ffffff!important;caret-color:#55e0b0!important;border-color:#557184!important;}
.stApp input::placeholder,.stApp textarea::placeholder {color:#c4d2dc!important;opacity:1!important;}
.stApp [data-baseweb="select"]>div,.stApp [data-baseweb="select"] [role="combobox"] {
  background:#0b1823!important;color:#ffffff!important;border-color:#557184!important;}
.stApp [data-baseweb="select"] *,[data-baseweb="popover"] * {color:#ffffff!important;}
[data-baseweb="popover"] [role="option"] {background:#142a39!important;color:#ffffff!important;}
[data-baseweb="popover"] [role="option"]:hover {background:#28536a!important;color:#ffffff!important;}
.stApp [data-baseweb="tag"] {background:#164c43!important;color:#ffffff!important;}
.stApp [data-baseweb="tag"] * {color:#ffffff!important;}
.stApp [data-testid="stSelectbox"] label,.stApp [data-testid="stMultiSelect"] label,
.stApp [data-testid="stTextInput"] label,.stApp [data-testid="stNumberInput"] label,
.stApp [data-testid="stSlider"] label,.stApp [data-testid="stCheckbox"] label,
.stApp [data-testid="stRadio"] label,.stApp [data-testid="stToggle"] label,
.stApp [data-testid="stFileUploader"] label {color:#ffffff!important;font-weight:800!important;}
.stApp [data-testid="stSlider"] output,.stApp [data-testid="stSlider"] [data-testid="stMarkdownContainer"] * {color:#ffffff!important;}
/* Tabs: individual raised tab-buttons, high-contrast idle/active/hover/focus states. */
.stApp [data-testid="stTabs"] [data-baseweb="tab-list"],
.stApp [role="tablist"] {background:#0b1823!important;border:1px solid #3b586b!important;
  border-radius:14px!important;padding:7px!important;gap:8px!important;}
.stApp [data-testid="stTabs"] button[role="tab"],
.stApp button[data-baseweb="tab"],.stApp [data-baseweb="tab"] {
  background:var(--tab-idle)!important;color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;
  opacity:1!important;border:1px solid #638096!important;border-radius:10px!important;
  min-height:48px!important;padding:10px 18px!important;font-size:14px!important;letter-spacing:.15px!important;font-weight:800!important;text-transform:none!important;
  box-shadow:0 2px 0 #07111a!important;transition:background .12s ease,border-color .12s ease!important;}
.stApp [data-testid="stTabs"] button[role="tab"] *,
.stApp button[data-baseweb="tab"] *,.stApp [data-baseweb="tab"] * {
  color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;opacity:1!important;
  font-weight:800!important;text-shadow:none!important;}
.stApp [data-testid="stTabs"] button[role="tab"][aria-selected="true"],
.stApp button[data-baseweb="tab"][aria-selected="true"],
.stApp [data-baseweb="tab"][aria-selected="true"] {
  background:var(--tab-active)!important;color:#06251e!important;-webkit-text-fill-color:#06251e!important;
  border:2px solid #9affd8!important;box-shadow:0 0 0 2px #55e0b033,0 3px 0 #07111a!important;}
.stApp [data-testid="stTabs"] button[role="tab"][aria-selected="true"] *,
.stApp button[data-baseweb="tab"][aria-selected="true"] *,
.stApp [data-baseweb="tab"][aria-selected="true"] * {
  color:#06251e!important;-webkit-text-fill-color:#06251e!important;}
.stApp [data-testid="stTabs"] button[role="tab"]:hover:not([aria-selected="true"]),
.stApp button[data-baseweb="tab"]:hover:not([aria-selected="true"]),
.stApp [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
  background:var(--tab-hover)!important;color:#ffffff!important;border-color:#8eb2c7!important;}
.stApp [data-testid="stTabs"] button[role="tab"]:hover:not([aria-selected="true"]) *,
.stApp button[data-baseweb="tab"]:hover:not([aria-selected="true"]) *,
.stApp [data-baseweb="tab"]:hover:not([aria-selected="true"]) * {color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;}
.stApp button[role="tab"]:focus-visible,.stApp [data-baseweb="tab"]:focus-visible {
  outline:3px solid #ffc45b!important;outline-offset:3px!important;}
.stApp button[role="tab"]:disabled,.stApp button[role="tab"][aria-disabled="true"] {
  background:#26333d!important;color:#cbd5dc!important;-webkit-text-fill-color:#cbd5dc!important;opacity:1!important;}
/* Workflow overview cards make the two stages visually distinct before the tabs. */
.stage-overview {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:8px 0 16px 0;}
.stage-overview-card {border-radius:14px;padding:18px 20px;min-height:104px;border:1px solid #52748a;background:#132b3c;box-shadow:0 5px 14px rgba(0,0,0,.18);}
.stage-overview-card.stage-one {border-top:4px solid #55e0b0;background:linear-gradient(135deg,#123b39,#142b3b);}
.stage-overview-card.stage-two {border-top:4px solid #58b9ff;background:linear-gradient(135deg,#183552,#142b3b);}
.stage-overview-card .stage-tag {font-family:'DM Mono',monospace;font-size:10px;letter-spacing:1.3px;font-weight:800;color:#8cefd0;margin-bottom:7px;}
.stage-overview-card.stage-two .stage-tag {color:#9bd5ff;}
.stage-overview-card .stage-name {font-size:18px;font-weight:800;color:#ffffff;margin-bottom:5px;}
.stage-overview-card .stage-copy {font-size:13px;line-height:1.55;color:#e5eef4;}
@media(max-width:760px){.stage-overview{grid-template-columns:1fr;}}
/* Tab controls are distinct raised tiles rather than faint text labels. */
.stApp [data-testid="stTabs"] [data-baseweb="tab-list"] {flex-wrap:wrap!important;align-items:stretch!important;}
.stApp [data-testid="stTabs"] button[role="tab"] {min-height:54px!important;white-space:normal!important;line-height:1.25!important;}
/* Final tab treatment: remove the default red underline and use a clear luminous selected tile. */
.stApp [data-testid="stTabs"] [data-baseweb="tab-highlight"],
.stApp [data-baseweb="tab-highlight"],
.stApp [data-testid="stTabs"] [data-baseweb="tab-border"] {display:none!important;height:0!important;opacity:0!important;background:transparent!important;}
.stApp [data-testid="stTabs"] button[role="tab"][aria-selected="true"],
.stApp button[data-baseweb="tab"][aria-selected="true"],
.stApp [data-baseweb="tab"][aria-selected="true"] {
  background:linear-gradient(135deg,#55e0b0 0%,#38c9b2 55%,#2da8b5 100%)!important;
  color:#06251e!important;-webkit-text-fill-color:#06251e!important;
  border:1px solid #a5ffe0!important;border-top:3px solid #d2fff0!important;
  border-radius:11px!important;
  box-shadow:0 0 0 1px rgba(85,224,176,.50),0 0 13px rgba(85,224,176,.38),0 0 25px rgba(56,201,178,.18),inset 0 1px 0 rgba(255,255,255,.25)!important;
  transform:translateY(-1px)!important;
}
.stApp [data-testid="stTabs"] button[role="tab"][aria-selected="true"] *,
.stApp button[data-baseweb="tab"][aria-selected="true"] *,
.stApp [data-baseweb="tab"][aria-selected="true"] * {color:#06251e!important;-webkit-text-fill-color:#06251e!important;text-shadow:none!important;}
.stApp [data-testid="stTabs"] button[role="tab"]:not([aria-selected="true"]),
.stApp button[data-baseweb="tab"]:not([aria-selected="true"]) {border-bottom:1px solid #638096!important;}
.stApp [data-testid="stTabs"] button[role="tab"]:not([aria-selected="true"]):hover,
.stApp button[data-baseweb="tab"]:not([aria-selected="true"]):hover {
  box-shadow:0 0 0 1px rgba(142,178,199,.45),0 0 12px rgba(88,185,255,.18)!important;
  transform:translateY(-1px)!important;
}
/* Panels, metrics, expanders, alerts and upload areas. */
.stApp [data-testid="stMetric"] {background:#142a39!important;border:1px solid #456174!important;
  padding:12px 14px!important;border-radius:12px!important;}
.stApp [data-testid="stMetricLabel"],.stApp [data-testid="stMetricLabel"] * {color:#d5e2ea!important;}
.stApp [data-testid="stMetricValue"],.stApp [data-testid="stMetricValue"] * {color:#ffffff!important;font-family:'DM Mono',monospace!important;}
.stApp [data-testid="stExpander"] {background:#122635!important;border:1px solid #456174!important;border-radius:10px!important;}
.stApp [data-testid="stExpander"] summary,.stApp [data-testid="stExpander"] summary * {color:#ffffff!important;}
.stApp [data-testid="stAlert"],.stApp [data-testid="stAlert"] * {color:#ffffff!important;}
.stApp [data-testid="stFileUploaderDropzone"] {background:#132a39!important;border:1px dashed #7895a8!important;}
.stApp [data-testid="stFileUploaderDropzone"] * {color:#ffffff!important;}
.stApp [data-testid="stDataFrame"],.stApp [data-testid="stTable"] {background:#142a39!important;color:#ffffff!important;border:1px solid #456174!important;}
/* Buttons: saturated teal, never faint. */
.stApp .stButton>button,.stApp .stDownloadButton>button {
  background:#087f68!important;color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;
  border:1px solid #55e0b0!important;border-radius:10px!important;font-weight:800!important;}
.stApp .stButton>button *,.stApp .stDownloadButton>button * {color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;}
.stApp .stButton>button:hover,.stApp .stDownloadButton>button:hover {background:#0aa982!important;border-color:#9affd8!important;}
.stApp .stButton>button:disabled {background:#354653!important;color:#d5dfe5!important;-webkit-text-fill-color:#d5dfe5!important;}
/* Hero/readout components retain intentional high-contrast instrument styling. */
.hero {background:linear-gradient(110deg,#102b3d,#1a4355);color:#fff;padding:23px 27px;border-radius:17px;margin-bottom:17px;position:relative;overflow:hidden;border:1px solid #426579;}
.hero,.hero * {color:#f7fafc!important;}
.hero .eyebrow {color:#8cf3c9!important;font-family:'DM Mono',monospace;letter-spacing:2px;font-size:11px;font-weight:700;}
.hero h1 {color:#fff!important;font-size:32px;}
.hero p {color:#e7f0f5!important;}
.kicker {color:#73eac0!important;font-family:'DM Mono',monospace;font-size:11px;letter-spacing:1.4px;font-weight:800;}
.section-title {font-size:20px;font-weight:800;color:#ffffff!important;margin:0 0 5px 0;}
.note {color:#d2dee6!important;font-size:12px;}
.readout,.readout * {background:#061019;color:#93ffc8!important;padding:10px 14px;border-radius:8px;font-family:'DM Mono',monospace;letter-spacing:.3px;}
.formula {background:#142f3d;border:1px solid #456b7c;border-left:4px solid #55e0b0;border-radius:7px;padding:12px 15px;color:#ffffff!important;}
hr {border-color:#3b586b!important;} footer {visibility:hidden;}

/* Editorial booklet treatment for the theory pages: quieter hierarchy and generous reading width. */
.book-cover {background:#102838;border:1px solid #426276;border-radius:12px;padding:30px 34px 27px 34px;margin:6px 0 30px 0;max-width:1160px;box-shadow:0 8px 24px #0003;}
.book-series {font-family:'DM Mono',monospace;font-size:10px;letter-spacing:2px;color:#65e2ba!important;font-weight:700;margin-bottom:15px;}
.book-chapter {font-family:'DM Mono',monospace;font-size:11px;letter-spacing:1.2px;text-transform:uppercase;color:#b6c9d5!important;margin-bottom:9px;}
.book-title {font-family:'DM Sans',Arial,sans-serif;font-size:30px;line-height:1.18;font-weight:800;color:#ffffff!important;margin-bottom:13px;}
.book-lede {font-size:16px;line-height:1.75;color:#e0eaf0!important;max-width:980px;}
.book-section-label {font-family:'DM Mono',monospace;font-size:11px;letter-spacing:1.35px;font-weight:700;color:#61dfb4!important;border-bottom:1px solid #365467;padding:17px 0 8px 0;margin:16px 0 10px 0;}
.book-prose {font-size:15px;line-height:1.9;color:#e1eaf0!important;max-width:1120px;margin:0 0 13px 0;}
.book-example {background:#152f40;border-left:4px solid #ffc45b;border-radius:5px;padding:18px 22px;color:#f5f8fa!important;font-size:14px;line-height:1.85;margin:8px 0 15px 0;max-width:1120px;}
.book-question {background:#123c39;border-left:4px solid #55e0b0;border-radius:5px;padding:17px 21px;color:#ffffff!important;font-size:15px;line-height:1.7;font-weight:600;margin:8px 0 18px 0;max-width:1120px;}
.book-question-label {font-family:'DM Mono',monospace;font-size:10px;letter-spacing:1.3px;font-weight:800;color:#70f0c4!important;margin-bottom:7px;text-transform:uppercase;}
.book-question div:not(.book-question-label) {color:#ffffff!important;font-weight:600;}

/* Full teaching content is presented in readable coloured blocks. */
.book-block {background:#142d3e;border:1px solid #426276;border-left:5px solid #55e0b0;border-radius:9px;padding:17px 21px;margin:8px 0 15px 0;max-width:1160px;color:#f3f7fa!important;}
.book-block * {color:#f3f7fa!important;}
.book-block .book-block-heading {font-family:'DM Mono',monospace;font-size:10px;letter-spacing:1.2px;text-transform:uppercase;color:#75edc6!important;font-weight:800;margin:0 0 9px 0;}
.book-block .book-block-body {font-size:14px;line-height:1.85;color:#edf3f7!important;margin:0;}
.book-block ul,.book-block ol {margin:5px 0 0 18px;padding:0;}
.book-block li {font-size:14px;line-height:1.8;color:#edf3f7!important;margin:3px 0;}
.book-block-gold {background:#342a19;border-left-color:#ffc45b;border-color:#6e5936;}
.book-block-gold .book-block-heading {color:#ffd27a!important;}
.book-block-blue {background:#172c48;border-left-color:#58b9ff;border-color:#3f5f83;}
.book-block-blue .book-block-heading {color:#8dcfff!important;}
.book-block-purple {background:#2b2440;border-left-color:#c6a5ff;border-color:#594a7c;}
.book-block-purple .book-block-heading {color:#d7c0ff!important;}
.book-block-green {background:#123c39;border-left-color:#55e0b0;border-color:#34786d;}
.book-block-green .book-block-heading {color:#80f1cf!important;}
.book-symbol-table {width:100%;border-collapse:collapse;font-size:13px;line-height:1.55;margin-top:4px;}
.book-symbol-table th {text-align:left;padding:9px 11px;background:#203e50;color:#9eeed7!important;border-bottom:1px solid #547185;}
.book-symbol-table td {padding:9px 11px;border-bottom:1px solid #385568;color:#f0f5f8!important;vertical-align:top;}
.book-symbol-table tr:nth-child(even) td {background:#183548;}
/* Streamlit's KaTeX element sits in its own element container; give every equation its own visible card. */
.stApp div[data-testid="stElementContainer"]:has(.katex-display) {background:#172f42;border:1px solid #45677e;border-left:4px solid #58b9ff;border-radius:8px;padding:8px 15px;margin:7px 0 12px 0;}
.stApp div[data-testid="stElementContainer"] .katex {color:#ffffff!important;}
.book-practical-divider {display:flex;align-items:center;gap:16px;margin:26px 0 10px 0;padding:15px 18px;border-top:1px solid #4a6678;border-bottom:1px solid #4a6678;background:#0d1d29;}
.book-practical-divider span {font-family:'DM Mono',monospace;letter-spacing:1.5px;font-size:10px;color:#ffc45b!important;font-weight:700;}
.book-practical-divider strong {font-size:15px;color:#ffffff!important;}


/* Teaching theory panels: clear separation before every experiment. */
.theory-panel {background:linear-gradient(135deg,#122b3d 0%,#0d1d2a 100%);border:1px solid #4c7189;
 border-left:5px solid #55e0b0;border-radius:14px;padding:18px 20px;margin:10px 0 18px 0;}
.theory-panel .theory-kicker {color:#55e0b0;font-family:'DM Mono',monospace;font-size:11px;letter-spacing:1.1px;font-weight:700;text-transform:uppercase;margin-bottom:7px;}
.theory-panel .theory-title {color:#ffffff;font-size:20px;font-weight:800;margin-bottom:7px;}
.theory-panel .theory-copy {color:#e4edf3;font-size:14px;line-height:1.55;margin-bottom:7px;}
.theory-panel .theory-practice {color:#ffd27a;font-size:13px;font-weight:700;margin-top:8px;}

.guide-banner {background:linear-gradient(120deg,#123b39,#142c43);border:1px solid #3e746a;border-left:6px solid #55e0b0;border-radius:15px;padding:22px 26px;margin:6px 0 18px 0;}
.guide-banner .guide-kicker {font:700 10px 'DM Mono',monospace;letter-spacing:1.5px;color:#79e8c5;margin-bottom:8px;}
.guide-banner .guide-heading {font-size:26px;line-height:1.15;font-weight:800;color:#ffffff;margin-bottom:8px;}
.guide-banner .guide-copy {font-size:14px;line-height:1.6;color:#e2edf3;max-width:1000px;}
.guide-card {background:#132b3b;border:1px solid #456477;border-radius:13px;padding:15px 17px;margin:8px 0;}
.guide-card h4 {color:#ffffff!important;margin:0 0 8px 0;font-size:15px;}
.guide-card p,.guide-card li {color:#e3edf3!important;line-height:1.58;font-size:13px;}
.guide-chip {display:inline-block;padding:5px 9px;border-radius:7px;background:#254b61;color:#ffffff;font-weight:800;font-size:11px;margin:2px 5px 2px 0;}
.guide-step {background:#142b3b;border:1px solid #416075;border-radius:12px;padding:12px 14px;min-height:96px;}
.guide-step.active {background:#16483f;border:2px solid #55e0b0;box-shadow:0 0 0 2px #55e0b022;}
.guide-step .num {font:700 10px 'DM Mono',monospace;color:#8debd0;letter-spacing:1px;}
.guide-step .name {font-weight:800;color:#ffffff;font-size:14px;margin-top:5px;}
.guide-step .desc {font-size:11px;line-height:1.4;color:#d6e4eb;margin-top:4px;}
.guide-alert {background:#49361b;border:1px solid #b98738;border-left:5px solid #ffc45b;border-radius:10px;padding:13px 16px;color:#fff2d9;margin:10px 0;line-height:1.55;}
.guide-note {background:#153b39;border:1px solid #3b8477;border-left:5px solid #55e0b0;border-radius:10px;padding:13px 16px;color:#e8fff7;margin:10px 0;line-height:1.55;}


/* -------------------------------------------------------------------------
   Robust tab styling override for current Streamlit/BaseWeb tab markup.
   Target both documented BaseWeb attributes and ARIA roles without relying
   on a single Streamlit test-id wrapper.
   ------------------------------------------------------------------------- */
:root {
  --qnl-tab-idle-bg: #1b3548;
  --qnl-tab-idle-border: #6b8ba0;
  --qnl-tab-hover-bg: #28516a;
  --qnl-tab-active-bg: #54e4b5;
  --qnl-tab-active-edge: #c2ffea;
  --qnl-tab-active-ink: #06251e;
}
/* The tab rail itself should read as a distinct instrument-console tray. */
.stApp [data-baseweb="tab-list"],
.stApp [role="tablist"],
[data-testid="stTabs"] [data-baseweb="tab-list"],
[data-testid="stTabs"] [role="tablist"] {
  display: flex !important;
  align-items: stretch !important;
  flex-wrap: wrap !important;
  gap: 9px !important;
  padding: 9px !important;
  margin: 5px 0 14px 0 !important;
  background: #07121c !important;
  border: 1px solid #466276 !important;
  border-radius: 15px !important;
  box-shadow: inset 0 2px 8px rgba(0,0,0,.45), 0 4px 14px rgba(0,0,0,.22) !important;
}
/* Idle tabs: each is a fully filled, raised tile (not just a text label). */
.stApp [data-baseweb="tab-list"] > button[data-baseweb="tab"],
.stApp [data-baseweb="tab-list"] > button[role="tab"],
.stApp [role="tablist"] > button[role="tab"],
[data-testid="stTabs"] button[data-baseweb="tab"],
[data-testid="stTabs"] button[role="tab"] {
  appearance: none !important;
  -webkit-appearance: none !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  min-height: 52px !important;
  padding: 11px 19px !important;
  margin: 0 !important;
  white-space: normal !important;
  text-align: center !important;
  background: linear-gradient(180deg, #29475c 0%, #1a3347 100%) !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  border: 1px solid #7896aa !important;
  border-bottom: 3px solid #7896aa !important;
  border-radius: 11px !important;
  opacity: 1 !important;
  font-family: 'DM Sans', Arial, sans-serif !important;
  font-size: 13px !important;
  font-weight: 800 !important;
  letter-spacing: .35px !important;
  line-height: 1.25 !important;
  box-shadow: 0 3px 0 #06111a, 0 5px 12px rgba(0,0,0,.25) !important;
  transform: translateY(0) !important;
  transition: background .16s ease, box-shadow .16s ease, transform .16s ease, border-color .16s ease !important;
}
/* Force nested text elements to inherit the correct high-contrast colour. */
.stApp [data-baseweb="tab-list"] > button[data-baseweb="tab"] *,
.stApp [role="tablist"] > button[role="tab"] *,
[data-testid="stTabs"] button[role="tab"] * {
  color: inherit !important;
  -webkit-text-fill-color: inherit !important;
  opacity: 1 !important;
  font-weight: 800 !important;
}
/* Active tab: saturated filled tile with a luminous edge and halo. */
.stApp [data-baseweb="tab-list"] > button[data-baseweb="tab"][aria-selected="true"],
.stApp [role="tablist"] > button[role="tab"][aria-selected="true"],
[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
.stApp button[data-baseweb="tab"][aria-selected="true"] {
  background: linear-gradient(135deg, #82f5d0 0%, #54e4b5 48%, #25bfa6 100%) !important;
  color: var(--qnl-tab-active-ink) !important;
  -webkit-text-fill-color: var(--qnl-tab-active-ink) !important;
  border: 2px solid var(--qnl-tab-active-edge) !important;
  border-bottom: 4px solid #caffed !important;
  border-radius: 12px !important;
  box-shadow: 0 0 0 2px rgba(84,228,181,.28),
              0 0 14px rgba(84,228,181,.55),
              0 0 28px rgba(84,228,181,.27),
              0 4px 0 #08745f,
              inset 0 1px 0 rgba(255,255,255,.65) !important;
  transform: translateY(-2px) !important;
  position: relative !important;
  z-index: 2 !important;
}
.stApp [data-baseweb="tab-list"] > button[data-baseweb="tab"][aria-selected="true"] *,
.stApp [role="tablist"] > button[role="tab"][aria-selected="true"] *,
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] * {
  color: #06251e !important;
  -webkit-text-fill-color: #06251e !important;
  text-shadow: none !important;
}
/* Hover and keyboard focus are visibly distinct from both states. */
.stApp [data-baseweb="tab-list"] > button[data-baseweb="tab"]:hover:not([aria-selected="true"]),
.stApp [role="tablist"] > button[role="tab"]:hover:not([aria-selected="true"]),
[data-testid="stTabs"] button[role="tab"]:hover:not([aria-selected="true"]) {
  background: linear-gradient(180deg, #376783 0%, #28516a 100%) !important;
  border-color: #b5dfff !important;
  box-shadow: 0 0 0 1px rgba(181,223,255,.25), 0 0 14px rgba(88,185,255,.30), 0 4px 0 #07131d !important;
  transform: translateY(-1px) !important;
}
.stApp [role="tablist"] > button[role="tab"]:focus-visible,
.stApp [data-baseweb="tab-list"] > button[data-baseweb="tab"]:focus-visible {
  outline: 3px solid #ffc45b !important;
  outline-offset: 3px !important;
}
/* Hide BaseWeb's moving underline/highlight: selected tile provides the cue. */
.stApp [data-baseweb="tab-highlight"],
.stApp [data-baseweb="tab-border"],
.stApp [data-testid="stTabs"] [data-baseweb="tab-highlight"],
.stApp [data-testid="stTabs"] [data-baseweb="tab-border"] {
  display: none !important;
  visibility: hidden !important;
  opacity: 0 !important;
  height: 0 !important;
  border: 0 !important;
  background: transparent !important;
}


/* Version 3.3.8: card-style navigation implemented with Streamlit buttons.
   This deliberately replaces native st.tabs, avoiding BaseWeb underline and
   DOM-selector differences between Streamlit releases. */
.stApp [data-testid="stButton"] > button[data-testid="stBaseButton-primary"],
.stApp [data-testid="stButton"] > button[kind="primary"] {
  min-height:58px!important; width:100%!important; white-space:normal!important;
  line-height:1.25!important; padding:12px 14px!important;
  color:#06251e!important; -webkit-text-fill-color:#06251e!important;
  background:linear-gradient(135deg,#9affdc 0%,#55e0b0 48%,#23b99e 100%)!important;
  border:2px solid #d1fff0!important; border-radius:13px!important;
  font-weight:900!important; letter-spacing:.2px!important;
  box-shadow:0 0 0 2px rgba(85,224,176,.22),0 0 18px rgba(85,224,176,.34),
             0 4px 0 #08745f,inset 0 1px 0 rgba(255,255,255,.55)!important;
  transform:translateY(-1px)!important; transition:all .14s ease!important;
}
.stApp [data-testid="stButton"] > button[data-testid="stBaseButton-secondary"],
.stApp [data-testid="stButton"] > button[kind="secondary"] {
  min-height:58px!important; width:100%!important; white-space:normal!important;
  line-height:1.25!important; padding:12px 14px!important;
  color:#ffffff!important; -webkit-text-fill-color:#ffffff!important;
  background:linear-gradient(180deg,#31536b 0%,#203b50 100%)!important;
  border:1px solid #7896aa!important; border-bottom:3px solid #7896aa!important;
  border-radius:13px!important; font-weight:800!important; letter-spacing:.15px!important;
  box-shadow:0 3px 0 #06111a,0 5px 12px rgba(0,0,0,.22)!important;
  transition:all .14s ease!important;
}
.stApp [data-testid="stButton"] > button[data-testid="stBaseButton-primary"] *,
.stApp [data-testid="stButton"] > button[kind="primary"] * {
  color:#06251e!important;-webkit-text-fill-color:#06251e!important;
}
.stApp [data-testid="stButton"] > button[data-testid="stBaseButton-secondary"] *,
.stApp [data-testid="stButton"] > button[kind="secondary"] * {
  color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;
}
.stApp [data-testid="stButton"] > button[data-testid="stBaseButton-secondary"]:hover,
.stApp [data-testid="stButton"] > button[kind="secondary"]:hover {
  background:linear-gradient(180deg,#416f8c 0%,#2c5872 100%)!important;
  border-color:#c5e8ff!important; box-shadow:0 0 0 1px rgba(181,223,255,.25),
  0 0 16px rgba(88,185,255,.28),0 4px 0 #07131d!important; transform:translateY(-1px)!important;
}
.stApp [data-testid="stButton"] > button:focus-visible {outline:3px solid #ffc45b!important;outline-offset:3px!important;}
.qnl-nav-label {font-family:'DM Mono',monospace;font-size:11px;font-weight:800;letter-spacing:1.25px;color:#8cefd0!important;margin:10px 0 5px 2px;text-transform:uppercase;}

</style>
""", unsafe_allow_html=True)

# --------------------------------- physics -----------------------------------
def decay_factor(dt_min: float, half_life_min: float) -> float:
    return math.exp(-math.log(2.0) * dt_min / half_life_min)


def np_deadtime(true_rate: float, tau_s: float) -> float:
    return true_rate / (1.0 + true_rate * tau_s)


def np_deadtime_correct(rate_obs: float, tau_s: float) -> float:
    denom = 1.0 - rate_obs * tau_s
    return rate_obs / denom if denom > 1e-8 else float("nan")


def acquire(activity_bq: float, sensitivity: float, count_s: float, bg_cps: float,
            tau_us: float, gain: float = 1.0, geom: float = 1.0,
            matrix: float = 1.0, seed: int | None = None) -> dict[str, float]:
    """Generate independent Poisson sample and background acquisitions."""
    rng = np.random.default_rng(seed)
    tau = tau_us * 1e-6
    sample_true = max(activity_bq, 0) * sensitivity * gain * geom * matrix
    gross_true = sample_true + bg_cps
    gross_obs_rate = np_deadtime(gross_true, tau)
    bg_obs_rate = np_deadtime(bg_cps, tau)
    gross_n = int(rng.poisson(max(gross_obs_rate * count_s, 0)))
    bg_n = int(rng.poisson(max(bg_obs_rate * count_s, 0)))
    gross_obs = gross_n / count_s
    bg_obs = bg_n / count_s
    gross_corr = np_deadtime_correct(gross_obs, tau)
    bg_corr = np_deadtime_correct(bg_obs, tau)
    net_corr = max(gross_corr - bg_corr, 0.0) if np.isfinite(gross_corr) and np.isfinite(bg_corr) else float("nan")
    sigma_net = math.sqrt(max(gross_n, 1) + max(bg_n, 1)) / count_s
    return {
        "gross_counts": gross_n, "background_counts": bg_n,
        "gross_cps_observed": gross_obs, "background_cps_observed": bg_obs,
        "gross_cps_corrected": gross_corr, "background_cps_corrected": bg_corr,
        "net_cps_corrected": net_corr, "sigma_net_cps": sigma_net,
        "true_net_cps": sample_true,
        "deadtime_fraction": gross_true * tau,
    }


# Energies are established nuclear/atomic emissions where indicated. The weights
# below are *teaching-shape weights*, not a substitute for energy-dependent NaI(Tl)
# efficiency, evaluated decay data, or a measured instrument response function.
SPECTRUM_LINES = {
    "18F": [(511.0, 1.93, "annihilation"), (1022.0, 0.025, "sum_peak")],
    "68Ga": [(511.0, 1.78, "annihilation"), (1077.0, 0.03, "gamma"), (1022.0, 0.018, "sum_peak")],
    "89Zr": [
        (14.96, 0.48, "characteristic_xray"), (16.74, 0.10, "characteristic_xray"),
        (511.0, 0.451, "annihilation"), (909.14, 0.99871, "gamma"),
        (1022.0, 0.004, "sum_peak"),
        (1620.76, 0.00072, "gamma"), (1657.26, 0.00107, "gamma"),
        (1712.8, 0.00763, "gamma"), (1744.47, 0.00129, "gamma"),
    ],
    "99mTc": [(140.5, 0.89, "gamma")],
    "64Cu": [(511.0, 0.36, "annihilation"), (1345.8, 0.004, "gamma")],
    "177Lu": [(112.9, 0.06, "gamma"), (208.4, 0.11, "gamma")],
    "131I": [(364.5, 0.817, "gamma"), (637.0, 0.072, "gamma")],
}


def simulate_spectrum(radionuclide: str, activity_bq: float, live_time_s: float,
                      sensitivity_cps_bq: float, background_cps: float,
                      resolution_511_pct: float, gain_kev_per_channel: float,
                      offset_kev: float, deadtime_us: float = 2.5, channels: int = 2048,
                      seed: int | None = None, half_life_min: float | None = None,
                      geometry_response_factor: float = 1.0,
                      scatter_transfer: float = 0.0,
                      coincidence_sum_factor: float = 1.0,
                      lead_fluorescence_cps: float = 0.0) -> pd.DataFrame:
    """Generate a teaching spectrum using Poisson events and integrated radioactive decay.

    The activity parameter is the activity at acquisition start. Sample events are
    integrated over the count interval; ambient/background events are time-constant.
    The sensitivity-defined total expected sample counts are partitioned across
    photopeak, Compton-like continuum and backscatter components, so the full
    spectrum integrates to the sensitivity-defined expected count total before
    dead-time response. Optional geometry, scatter-transfer, coincidence-summing
    and lead-fluorescence terms are simplified teaching controls. The spectrum is
    not a Monte Carlo photon-transport or validated instrument-response model.
    """
    rng = np.random.default_rng(seed)
    ch = np.arange(channels, dtype=float)
    energy = offset_kev + gain_kev_per_channel * ch
    live_time_s = max(float(live_time_s), 1e-9)

    if half_life_min and half_life_min > 0:
        lam = math.log(2.0) / (float(half_life_min) * 60.0)
        effective_sample_time_s = -math.expm1(-lam * live_time_s) / lam
        average_activity_fraction = effective_sample_time_s / live_time_s
    else:
        lam = 0.0
        effective_sample_time_s = live_time_s
        average_activity_fraction = 1.0

    sample_expected = np.zeros(channels, dtype=float)
    background_expected = np.zeros(channels, dtype=float)

    continuum_shape = (0.35 + 0.9 * np.exp(-np.maximum(energy, 0) / 260.0)) * np.exp(-np.maximum(energy, 0) / 2600.0)
    continuum_shape[energy < 0] = 0.0
    continuum_shape /= max(float(continuum_shape.sum()), 1.0)
    background_total = max(float(background_cps), 0.0) * live_time_s
    background_expected += continuum_shape * background_total * 0.72
    background_tail_shape = np.maximum(1.0 + 0.25 * np.sin(ch / 90.0), 0.0)
    background_tail_shape /= max(float(background_tail_shape.sum()), 1.0)
    background_expected += background_tail_shape * background_total * 0.28

    # Optional environmental lead K-fluorescence feature. This is a separate
    # background component, not an intrinsic 18F or 89Zr emission.
    lead_rate = max(float(lead_fluorescence_cps), 0.0)
    if lead_rate > 0:
        for xray_energy, relative in [(74.97, 2.0), (84.94, 1.0)]:
            centre_x = (xray_energy - offset_kev) / max(gain_kev_per_channel, 1e-12)
            fwhm_x = resolution_511_pct / 100.0 * 511.0 * math.sqrt(xray_energy / 511.0)
            sigma_x = max((fwhm_x / 2.355) / max(gain_kev_per_channel, 1e-12), 0.7)
            xray_shape = np.exp(-0.5 * ((ch-centre_x)/sigma_x)**2)
            xray_shape /= max(float(xray_shape.sum()), 1e-12)
            background_expected += xray_shape * lead_rate * live_time_s * relative / 3.0

    lines = SPECTRUM_LINES.get(radionuclide, [(RADIONUCLIDES[radionuclide]["energy_keV"], 1.0, "gamma")])
    adjusted_lines = []
    for e, weight, kind in lines:
        w = float(weight)
        if kind == "sum_peak":
            w *= max(float(coincidence_sum_factor), 0.0)
        adjusted_lines.append((e, w, kind))
    visible_lines = [line for line in adjusted_lines if energy.min() <= float(line[0]) <= energy.max()]
    total_line_strength = max(sum(max(float(line[1]), 0.0) for line in visible_lines), 1e-12)
    geometry_response_factor = float(np.clip(geometry_response_factor, 0.0, 2.0))
    total_detected = max(float(activity_bq), 0.0) * max(float(sensitivity_cps_bq), 0.0) * geometry_response_factor * effective_sample_time_s
    # Allocate the sensitivity-defined total sample counts across the displayed
    # lines and spectral components. This keeps the full-spectrum expected count
    # total consistent with sensitivity × activity × integrated acquisition time.
    for e, yield_rel, line_kind in visible_lines:
        line_n = total_detected * max(float(yield_rel), 0.0) / total_line_strength
        centre = (float(e) - offset_kev) / gain_kev_per_channel
        fwhm_kev = resolution_511_pct / 100.0 * 511.0 * math.sqrt(max(float(e), 1.0) / 511.0)
        sigma_ch = max((fwhm_kev / 2.355) / gain_kev_per_channel, 0.7)
        peak_shape = np.exp(-0.5 * ((ch - centre) / sigma_ch) ** 2)
        peak_shape /= max(float(peak_shape.sum()), 1e-12)

        if line_kind == "characteristic_xray":
            # The 89Zr electron-capture daughter Y produces K X-rays at ~14.9
            # and ~16.7 keV. Their visibility is threshold/attenuation sensitive.
            peak_fraction, continuum_fraction, back_fraction = 0.92, 0.08, 0.0
        elif line_kind == "sum_peak":
            # Teaching approximation for a coincidence-summing feature. Its
            # apparent intensity depends strongly on geometry and resolving time.
            peak_fraction, continuum_fraction, back_fraction = 0.88, 0.12, 0.0
        else:
            transfer = float(np.clip(scatter_transfer, 0.0, 0.18))
            peak_fraction = 0.745 - transfer
            continuum_fraction = 0.220 + transfer
            back_fraction = 0.035

        sample_expected += line_n * peak_fraction * peak_shape
        if continuum_fraction > 0:
            compton_edge = float(e) * (1.0 - 1.0 / (1.0 + 2.0 * float(e) / 511.0))
            edge_width = max(2.5 * gain_kev_per_channel, 0.015 * max(compton_edge, 1.0))
            edge_gate = 1.0 / (1.0 + np.exp(np.clip((energy - compton_edge) / edge_width, -60, 60)))
            low_energy_shape = np.exp(-np.maximum(energy, 0.0) / max(compton_edge, 1.0) * 0.55)
            compton_shape = np.maximum(edge_gate * low_energy_shape, 0.0)
            compton_shape /= max(float(compton_shape.sum()), 1e-12)
            sample_expected += line_n * continuum_fraction * compton_shape
        if back_fraction > 0:
            backscatter = float(e) / (1.0 + 2.0 * float(e) / 511.0)
            back_sigma = max((0.035 * backscatter) / gain_kev_per_channel, 1.2)
            back_shape = np.exp(-0.5 * ((energy - backscatter) / (back_sigma * gain_kev_per_channel)) ** 2)
            back_shape /= max(float(back_shape.sum()), 1e-12)
            sample_expected += line_n * back_fraction * back_shape

    true_rate_average = float((sample_expected.sum() + background_expected.sum()) / live_time_s)
    tau_s = max(float(deadtime_us), 0.0) * 1e-6
    response_scale = 1.0 / (1.0 + true_rate_average * tau_s)
    sample_observed_expected = np.maximum(sample_expected * response_scale, 0.0)
    background_observed_expected = np.maximum(background_expected * response_scale, 0.0)
    expected_observed = sample_observed_expected + background_observed_expected
    observed = rng.poisson(expected_observed).astype(int)

    result = pd.DataFrame({
        "Channel": ch.astype(int),
        "Energy (keV)": energy,
        "Counts": observed,
        "Expected counts (teaching)": expected_observed,
        "Expected sample counts (teaching)": sample_observed_expected,
        "Expected background counts (teaching)": background_observed_expected,
    })
    result.attrs["deadtime_loss_fraction"] = 1.0 - response_scale
    result.attrs["average_activity_fraction"] = average_activity_fraction
    result.attrs["effective_sample_time_s"] = effective_sample_time_s
    result.attrs["half_life_min"] = float(half_life_min) if half_life_min else None
    result.attrs["geometry_response_factor"] = geometry_response_factor
    result.attrs["scatter_transfer"] = float(np.clip(scatter_transfer, 0.0, 0.18))
    result.attrs["coincidence_sum_factor"] = float(coincidence_sum_factor)
    result.attrs["lead_fluorescence_cps"] = lead_rate
    return result


def make_live_acquisition_figure(
    spectrum: pd.DataFrame,
    live_time_s: float,
    n_frames: int = 36,
    y_scale: str = "Linear",
    energy_range: tuple[float, float] | None = None,
    frame_duration_ms: int = 220,
    dominant_line: float | None = None,
    isotope_lines: list[tuple[float, float, str]] | None = None,
    half_life_min: float | None = None,
) -> go.Figure:
    """Animate a Poisson acquisition with time-resolved count-rate diagnostics.

    The channel totals are generated by the spectrum model. Their allocation to
    time bins is conditional on the expected decaying sample and constant
    background contributions. The final animation frame equals the saved
    channel-by-channel spectrum exactly.

    The cumulative-count panel follows the elapsed acquisition window and
    expands its axes as time advances. The interval-CPS panel shows the measured
    rate in each bin with Poisson uncertainty plus the model-expected interval
    rate. For long-lived radionuclides over short acquisitions, the expected
    interval rate is correctly almost constant; visibly large decay requires
    an acquisition duration that is a meaningful fraction of the half-life.
    """
    energies = spectrum["Energy (keV)"].to_numpy(dtype=float)
    final_counts = np.maximum(spectrum["Counts"].to_numpy(dtype=int), 0)
    live_time_s = max(float(live_time_s), 1e-9)
    n_frames = int(max(8, min(n_frames, 96)))
    dt = live_time_s / n_frames
    edges = np.linspace(0.0, live_time_s, n_frames + 1)
    elapsed = edges[1:]
    midpoints = (edges[:-1] + edges[1:]) / 2.0

    sample_expected = spectrum.get(
        "Expected sample counts (teaching)", pd.Series(np.zeros(len(spectrum)))
    ).to_numpy(dtype=float)
    background_expected = spectrum.get(
        "Expected background counts (teaching)",
        pd.Series(spectrum["Expected counts (teaching)"].to_numpy(dtype=float))
    ).to_numpy(dtype=float)
    if sample_expected.sum() <= 0 and background_expected.sum() <= 0:
        sample_expected = spectrum["Expected counts (teaching)"].to_numpy(dtype=float)
        background_expected = np.zeros_like(sample_expected)

    if half_life_min and half_life_min > 0:
        lam = math.log(2.0) / (float(half_life_min) * 60.0)
        interval_integrals = (np.exp(-lam * edges[:-1]) - np.exp(-lam * edges[1:])) / lam
        sample_prob = interval_integrals / max(float(interval_integrals.sum()), 1e-15)
    else:
        lam = 0.0
        sample_prob = np.full(n_frames, 1.0 / n_frames)
    background_prob = np.full(n_frames, 1.0 / n_frames)

    # Apportion each channel's final Poisson count across time bins. This is a
    # conditional multinomial allocation, preserving the exported final spectrum.
    rng = np.random.default_rng(76123 + int(final_counts.sum()) % 1000003)
    increments = np.zeros((n_frames, len(final_counts)), dtype=np.int64)
    for channel, n in enumerate(final_counts):
        if n <= 0:
            continue
        weights = sample_expected[channel] * sample_prob + background_expected[channel] * background_prob
        probs = weights / weights.sum() if weights.sum() > 0 else np.full(n_frames, 1.0 / n_frames)
        increments[:, channel] = rng.multinomial(int(n), probs)

    cumulative = np.cumsum(increments, axis=0)
    totals = cumulative.sum(axis=1)
    interval_counts = increments.sum(axis=1)
    interval_cps = interval_counts / dt
    # Poisson standard uncertainty for each interval rate: sqrt(N_i) / dt.
    interval_sd_cps = np.sqrt(interval_counts.astype(float)) / dt

    expected_sample_total = float(sample_expected.sum())
    expected_background_total = float(background_expected.sum())
    expected_interval_counts = (
        expected_sample_total * sample_prob + expected_background_total * background_prob
    )
    expected_interval_cps = expected_interval_counts / dt
    expected_interval_sd = np.sqrt(np.maximum(expected_interval_counts, 0.0)) / dt

    expected_cumulative = np.cumsum(expected_interval_counts)
    cumulative_x = np.r_[0.0, elapsed]
    cumulative_y = np.r_[0, totals]
    expected_y = np.r_[0.0, expected_cumulative]

    log_display = str(y_scale).lower().startswith("log")
    initial_y = cumulative[0].astype(float) + (1.0 if log_display else 0.0)
    custom_counts = cumulative[0].astype(int)

    # Axes 2 (cumulative) advance with acquisition, rather than reserving the
    # entire eventual run range from the first frame. This makes early changes
    # visible while preserving the true elapsed-time axis.
    initial_xmax = max(float(elapsed[0]) * 1.12, dt)
    initial_ymax = max(float(totals[0]), float(expected_cumulative[0]), 1.0) * 1.18
    # Zoom the interval-rate panel around the physically relevant rate band.
    # A zero-based axis can visually flatten ordinary Poisson variation when
    # the mean rate is high. The zoom is explicit in the axis title and retains
    # all observed points and their uncertainty bars.
    rate_lower_candidates = [
        expected_interval_cps - 2.0 * expected_interval_sd,
        interval_cps - 2.0 * interval_sd_cps,
    ]
    rate_upper_candidates = [
        expected_interval_cps + 2.0 * expected_interval_sd,
        interval_cps + 2.0 * interval_sd_cps,
    ]
    rate_low = max(0.0, min(float(np.min(v)) for v in rate_lower_candidates if len(v)))
    rate_high = max(float(np.max(v)) for v in rate_upper_candidates if len(v))
    rate_span = max(rate_high - rate_low, max(rate_high, 1.0) * 0.02, 0.5)
    rate_padding = 0.12 * rate_span
    interval_ymin = max(0.0, rate_low - rate_padding)
    interval_ymax = max(interval_ymin + 0.5, rate_high + rate_padding)

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.58, 0.24, 0.18],
        vertical_spacing=0.105,
        subplot_titles=(
            "Pulse-height spectrum · cumulative events",
            "Cumulative counts versus elapsed acquisition time",
            "Interval count rate · observed and expected",
        ),
    )
    fig.add_trace(go.Scatter(
        x=energies, y=initial_y, customdata=custom_counts, mode="lines",
        line=dict(color="#55E0B0", width=1.8),
        fill="tozeroy" if not log_display else None,
        fillcolor="rgba(22,132,91,0.14)", name="Accumulated spectrum",
        hovertemplate="Energy: %{x:.2f} keV<br>Counts/channel: %{customdata:,}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0.0, elapsed[0]], y=[0, totals[0]], mode="lines+markers",
        line=dict(color="#FFC45B", width=2.8), marker=dict(size=5),
        name="Observed cumulative counts",
        hovertemplate="Elapsed: %{x:.2f} s<br>Observed counts: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[0.0, elapsed[0]], y=[0.0, expected_cumulative[0]], mode="lines",
        line=dict(color="#58B9FF", width=2, dash="dash"), name="Model expectation",
        hovertemplate="Elapsed: %{x:.2f} s<br>Expected counts: %{y:,.1f}<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[midpoints[0]], y=[interval_cps[0]], mode="markers",
        marker=dict(color="#C6A5FF", size=7),
        error_y=dict(type="data", array=[interval_sd_cps[0]], visible=True, color="#C6A5FF"),
        name="Observed interval CPS",
        hovertemplate="Interval midpoint: %{x:.2f} s<br>Observed rate: %{y:,.2f} cps<extra></extra>",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=midpoints[:1], y=expected_interval_cps[:1], mode="lines",
        line=dict(color="#55E0B0", width=2.2, dash="dash"),
        name="Expected interval CPS",
        hovertemplate="Interval midpoint: %{x:.2f} s<br>Expected rate: %{y:,.2f} cps<extra></extra>",
    ), row=3, col=1)

    frames = []
    for i in range(n_frames):
        frame_y = cumulative[i].astype(float) + (1.0 if log_display else 0.0)
        xmax = max(float(elapsed[i]) * 1.08, dt)
        ymax = max(float(totals[i]), float(expected_cumulative[i]), 1.0) * 1.18
        frames.append(go.Frame(
            name=str(i + 1),
            data=[
                go.Scatter(x=energies, y=frame_y, customdata=cumulative[i].astype(int)),
                go.Scatter(x=cumulative_x[:i + 2], y=cumulative_y[:i + 2]),
                go.Scatter(x=cumulative_x[:i + 2], y=expected_y[:i + 2]),
                go.Scatter(x=midpoints[:i + 1], y=interval_cps[:i + 1],
                           error_y=dict(type="data", array=interval_sd_cps[:i + 1], visible=True, color="#C6A5FF")),
                go.Scatter(x=midpoints[:i + 1], y=expected_interval_cps[:i + 1]),
            ],
            traces=[0, 1, 2, 3, 4],
            layout=go.Layout(
                xaxis2=dict(range=[0.0, xmax], autorange=False),
                yaxis2=dict(range=[0.0, ymax], autorange=False),
            ),
        ))
    fig.frames = frames
    slider_steps = [dict(
        method="animate", label=f"{elapsed[i]:.1f} s", args=[[str(i + 1)],
        {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]
    ) for i in range(n_frames)]

    fig.update_layout(
        template="plotly_dark", height=980,
        margin=dict(l=82, r=38, t=130, b=118),
        paper_bgcolor="#0B1823", plot_bgcolor="#102231",
        font=dict(family="DM Sans, Arial", color="#F7FAFC", size=13),
        title=dict(text="LIVE PULSE-HEIGHT ACQUISITION · spectrum, cumulative counts and interval CPS", x=0.015,
                   xanchor="left", font=dict(size=18, color="#FFFFFF")),
        showlegend=True, legend=dict(orientation="h", y=1.04, x=0.43, font=dict(color="#FFFFFF", size=11)),
        updatemenus=[dict(
            type="buttons", direction="left", x=0.015, y=1.105,
            buttons=[
                dict(label="▶ PLAY", method="animate", args=[None,
                     {"fromcurrent": True, "frame": {"duration": int(frame_duration_ms), "redraw": True},
                      "transition": {"duration": 30}, "mode": "immediate"}]),
                dict(label="Ⅱ PAUSE", method="animate", args=[[None],
                     {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]),
            ], bgcolor="#263F52", bordercolor="#55E0B0", font=dict(color="#FFFFFF", size=12),
        )],
        sliders=[dict(
            active=0, x=0.07, y=-0.095, len=0.89,
            currentvalue={"prefix": "Elapsed acquisition: ", "suffix": f" / {live_time_s:.1f} s"},
            steps=slider_steps, font=dict(color="#FFFFFF", size=11),
        )],
    )
    fig.update_xaxes(title_text="Energy (keV)", showgrid=True, gridcolor="#294353", zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="Counts per channel" + (" (+1 display offset)" if log_display else ""),
                     type="log" if log_display else "linear", rangemode="tozero" if not log_display else None,
                     showgrid=True, gridcolor="#294353", row=1, col=1)
    fig.update_xaxes(title_text="Elapsed acquisition time (s)", range=[0.0, initial_xmax], autorange=False,
                     showgrid=True, gridcolor="#294353", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative counts", range=[0.0, initial_ymax], autorange=False,
                     showgrid=True, gridcolor="#294353", row=2, col=1)
    fig.update_xaxes(title_text="Elapsed time (s)", range=[0.0, live_time_s], autorange=False,
                     showgrid=True, gridcolor="#294353", row=3, col=1)
    fig.update_yaxes(title_text="Interval count rate (cps; zoomed scale)", range=[interval_ymin, interval_ymax], autorange=False,
                     showgrid=True, gridcolor="#294353", row=3, col=1)

    if energy_range is not None:
        lo, hi = sorted((float(energy_range[0]), float(energy_range[1])))
        fig.update_xaxes(range=[lo, hi], row=1, col=1)
    if isotope_lines:
        for energy, intensity, line_type in isotope_lines:
            if energies.min() <= energy <= energies.max():
                is_primary = dominant_line is not None and abs(energy - dominant_line) < 1e-6
                fig.add_vline(x=float(energy), row=1, col=1, line_width=1.5 if is_primary else 1.0,
                              line_dash="solid" if is_primary else "dot",
                              line_color="#FFC45B" if is_primary else "#8FAABD", opacity=0.85)
                if is_primary:
                    fig.add_annotation(x=float(energy), y=1.0, yref="y domain", row=1, col=1,
                                       text=f"Primary line · {energy:.1f} keV", showarrow=False,
                                       yshift=14, font=dict(color="#FFC45B", size=10))
    return fig


def peak_integral(spectrum: pd.DataFrame, centre_kev: float, half_width_kev: float) -> dict[str, float]:
    """ROI integration with a sideband-derived local linear continuum estimate."""
    e=spectrum["Energy (keV)"].to_numpy(); y=spectrum["Counts"].to_numpy(dtype=float)
    roi=(e>=centre_kev-half_width_kev)&(e<=centre_kev+half_width_kev)
    side=((e>=centre_kev-3*half_width_kev)&(e<centre_kev-2*half_width_kev)) | ((e>centre_kev+2*half_width_kev)&(e<=centre_kev+3*half_width_kev))
    if roi.sum()==0 or side.sum()<2:
        return {"gross_peak_counts":0.0,"estimated_continuum_counts":0.0,"net_peak_counts":0.0,"net_peak_rate_cps":0.0,"counting_sd":0.0}
    xside=e[side]; yside=y[side]
    slope, intercept=np.polyfit(xside,yside,1)
    continuum=np.maximum(slope*e[roi]+intercept,0)
    gross=float(y[roi].sum()); cont=float(continuum.sum()); net=max(gross-cont,0.0)
    return {"gross_peak_counts":gross,"estimated_continuum_counts":cont,
            "net_peak_counts":net,"net_peak_rate_cps":net/max(float(spectrum.attrs.get("live_time_s",1)),1e-9),
            "counting_sd":math.sqrt(max(gross+cont,0.0))}



# ----------------------------- datasets --------------------------------------
def new_record(df: pd.DataFrame, experiment: str, radionuclide: str, notes: str) -> pd.DataFrame:
    out=df.copy()
    out["Experiment"] = experiment
    out["Radionuclide"] = radionuclide
    out["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out["Scenario notes"] = notes
    return out


def demo_datasets() -> dict[str, pd.DataFrame]:
    rng=np.random.default_rng(240623)
    # Calibration standards
    acts=np.array([500,1000,2500,5000,10000,20000,40000],dtype=float)
    rows=[]
    for i,a in enumerate(acts):
        r=acquire(a,.145,60,16,2.5,seed=int(rng.integers(1,1e9)))
        rows.append({"Standard":f"STD-{i+1:02d}","Known activity (Bq)":a,"Gross counts":r["gross_counts"],
                     "Background counts":r["background_counts"],"Net rate (cps)":r["net_cps_corrected"],
                     "Net rate SD (cps)":r["sigma_net_cps"],"Estimated activity (Bq)":r["net_cps_corrected"]/.145})
    cal=new_record(pd.DataFrame(rows),"Calibration example","18F","Synthetic known-activity standards; 60 s counts.")
    # Detector normalization
    gains=DETECTOR_REL
    rows=[]
    for det,g in gains.items():
        for rep in range(1,5):
            r=acquire(12000,.145,45,14,2.5,gain=g,seed=int(rng.integers(1,1e9)))
            sens=r["net_cps_corrected"]/12000
            rows.append({"Detector":det,"Replicate":rep,"Reference activity (Bq)":12000,"Net rate (cps)":r["net_cps_corrected"],
                         "Observed sensitivity (cps/Bq)":sens,"Raw response relative to fleet mean":g})
    norm=pd.DataFrame(rows)
    mean_s=norm.groupby("Detector")["Observed sensitivity (cps/Bq)"].mean()
    factors=float(mean_s.mean())/mean_s
    norm["Normalisation factor"] = norm["Detector"].map(factors)
    norm["Normalised sensitivity (cps/Bq)"] = norm["Observed sensitivity (cps/Bq)"]*norm["Normalisation factor"]
    norm=new_record(norm,"Detector normalisation example","18F","Four repeated reference-source measurements per detector.")
    # Tissue biodistribution synthetic study
    organs=[("Blood",.035,.42),("Liver",.185,1.25),("Kidney",.115,.31),("Tumour",.082,.24),("Bone",.048,.36),("Muscle",.026,.58),("Lung",.071,.19),("Spleen",.092,.11)]
    inj=2.5e6; elapsed=95; f=decay_factor(elapsed,RADIONUCLIDES["18F"]["half_life_min"])
    rows=[]
    for organ,frac,mass in organs:
        a0=inj*frac; at=a0*f
        r=acquire(at,.145,60,16,2.5,seed=int(rng.integers(1,1e9)))
        est=r["net_cps_corrected"]/.145
        rows.append({"Organ":organ,"Mass (g)":mass,"Injected activity (Bq)":inj,"Elapsed time (min)":elapsed,
                     "Activity at count (Bq)":at,"Gross counts":r["gross_counts"],"Background counts":r["background_counts"],
                     "Net rate (cps)":r["net_cps_corrected"],"Estimated activity (Bq)":est,
                     "%ID (estimated, count time)":100*est/(inj*f),"%ID/g":100*est/(inj*f*mass)})
    tissue=new_record(pd.DataFrame(rows),"Hypothetical biodistribution","18F","Synthetic organ uptake pattern; reference activity at injection.")
    # Matched synthetic PET ROI summaries for the cross-modality workshop.
    pet_rows=[]
    for _,tr in tissue.iterrows():
        true_conc=float(tr["Activity at count (Bq)"])*1.05/float(tr["Mass (g)"])
        measured=float(true_conc*rng.normal(1.0,0.025))
        pet_rows.append({"ROI":tr["Organ"],"Modality":"PET","Concentration measured (Bq/mL)":measured,
                         "Time from reference (min)":elapsed,"Common reference time (min)":elapsed,
                         "Decay-corrected concentration (Bq/mL)":measured/decay_factor(elapsed,RADIONUCLIDES["18F"]["half_life_min"]),
                         "Recovery coefficient":0.95,"ROI volume (mL)":float(tr["Mass (g)"])/1.05,
                         "Reference concentration (Bq/mL)":measured})
    pet_demo=new_record(pd.DataFrame(pet_rows),"Imported PET ROI summary","18F","Synthetic PET ROI summary matched to the demonstration ex-vivo tissue dataset; illustrative values.")
    # Geometry/matrix response
    geo=[("13 mm tube",1.00),("3 mL vial",.98),("5 mL tube",.95),("20 mL vial",.90),("High-fill vial",.86)]
    rows=[]
    for name,g in geo:
        for rep in range(1,4):
            r=acquire(10000,.145,60,16,2.5,geom=g,seed=int(rng.integers(1,1e9)))
            rows.append({"Geometry":name,"Relative geometry factor (truth)":g,"Replicate":rep,
                         "Net rate (cps)":r["net_cps_corrected"],"Uncorrected activity estimate (Bq)":r["net_cps_corrected"]/.145,
                         "Geometry-corrected activity estimate (Bq)":r["net_cps_corrected"]/(.145*g)})
    geometry=new_record(pd.DataFrame(rows),"Geometry challenge","18F","Same nominal activity in different sample geometries; illustrative response factors.")

    # Troubleshooting case: high-activity standards exhibit rate-dependent dead-time loss.
    sat_rows=[]
    sat_activities=np.array([5_000,10_000,25_000,50_000,100_000,200_000,400_000],dtype=float)
    sat_tau=4.5e-6; sat_bg=18.0; sat_time=60.0; sat_sens=.145
    sat_bg_obs=np_deadtime(sat_bg,sat_tau)
    for i,a in enumerate(sat_activities,1):
        true_total=a*sat_sens+sat_bg
        gross_obs=np_deadtime(true_total,sat_tau)
        gross_n=int(rng.poisson(gross_obs*sat_time)); bg_n=int(rng.poisson(sat_bg_obs*sat_time))
        net_obs=max(gross_n/sat_time-bg_n/sat_time,0.0)
        sat_rows.append({"Standard":f"QC-HR-{i:02d}","Known activity (Bq)":a,"Gross counts":gross_n,
                         "Background counts":bg_n,"Count time (s)":sat_time,"Observed net rate (cps)":net_obs,
                         "Naive activity estimate (Bq)":net_obs/sat_sens,
                         "Relative deviation from expected (%)":100*(net_obs/(a*sat_sens)-1),
                         "Challenge clue":"High-rate response may be non-linear"})
    sat_cal=new_record(pd.DataFrame(sat_rows),"Troubleshooting: high-rate calibration non-linearity","18F",
                       "Synthetic standards intentionally generated with a non-paralysable response; investigate rate-dependent count loss.")

    # Troubleshooting case: a persistently low detector response should be distinguished from random variation.
    trouble_gain={"A":1.00,"B":1.03,"C":.98,"D":.76,"E":1.02,"F":.99}
    det_rows=[]
    for det,gain in trouble_gain.items():
        for rep in range(1,6):
            rr=acquire(15000,.145,60,14,2.5,gain=gain,seed=int(rng.integers(1,1e9)))
            det_rows.append({"Detector":det,"Replicate":rep,"Reference activity (Bq)":15000,
                             "Gross counts":rr["gross_counts"],"Background counts":rr["background_counts"],
                             "Count time (s)":60,"Net rate (cps)":rr["net_cps_corrected"],
                             "Observed sensitivity (cps/Bq)":rr["net_cps_corrected"]/15000,
                             "QC status":"Investigate persistent low channel" if det=="D" else "Within illustrative comparison range"})
    trouble_norm=pd.DataFrame(det_rows)
    det_mean=trouble_norm.groupby("Detector")["Observed sensitivity (cps/Bq)"].mean()
    det_factor=det_mean.mean()/det_mean
    trouble_norm["Normalisation factor"]=trouble_norm["Detector"].map(det_factor)
    trouble_norm["Normalised sensitivity (cps/Bq)"]=trouble_norm["Observed sensitivity (cps/Bq)"]*trouble_norm["Normalisation factor"]
    trouble_norm=new_record(trouble_norm,"Troubleshooting: detector D low response","18F",
                             "Synthetic reference-source QC with detector D deliberately assigned a stable low response; assess whether to correct or investigate.")

    # Troubleshooting case: rack-position-dependent under-response.
    rack_rows=[]
    for pos in range(1,13):
        position_factor=.74 if pos==8 else 1.0
        rr=acquire(12000,.145,45,12,2.5,gain=position_factor,seed=int(rng.integers(1,1e9)))
        rack_rows.append({"Rack position":pos,"Sample ID":f"QC-VIAL-{pos:02d}","Nominal activity (Bq)":12000,
                          "Net rate (cps)":rr["net_cps_corrected"],"Apparent activity (Bq)":rr["net_cps_corrected"]/.145,
                          "Position response factor (teaching only)":position_factor})
    rack_case=new_record(pd.DataFrame(rack_rows),"Troubleshooting: rack position response","18F",
                         "Synthetic rack QC where position 8 has a persistent position-dependent under-response; test whether the deviation follows sample or location.")

    # Troubleshooting case: small-ROI recovery loss in imaging relative to ex-vivo measurement.
    pet_case_rows=[]; gamma_case_rows=[]
    paired=[("Liver",62000,.00120,.97,1.20), ("Kidney",28000,.00030,.91,.30),
            ("Tumour",18500,.00022,.62,.22), ("Muscle",5200,.00150,.98,1.50)]
    for roi,activity,mass,rc,vol in paired:
        gamma_case_rows.append({"Tissue":roi,"Sample ID":roi,"Estimated activity at count (Bq)":activity,
                                "Mass (g)":mass,"Elapsed time (min)":40.0,"Volume (mL)":vol})
        gamma_conc=activity/vol
        pet_measured=gamma_conc*rc*float(rng.normal(1.0,.025))
        pet_case_rows.append({"ROI":roi,"Modality":"PET","Concentration measured (Bq/mL)":pet_measured,
                              "Time from reference (min)":48.0,"Common reference time (min)":0.0,
                              "Recovery coefficient":rc,"RC correction applied":False,
                              "Reference concentration (Bq/mL)":pet_measured/decay_factor(48.0,RADIONUCLIDES["18F"]["half_life_min"])})
    pet_recovery=new_record(pd.DataFrame(pet_case_rows),"Troubleshooting: PET small-ROI recovery","18F",
                            "Synthetic paired ROI dataset with deliberately reduced recovery for small structures and an 8-minute timing offset.")
    gamma_recovery=new_record(pd.DataFrame(gamma_case_rows),"Troubleshooting: matched ex-vivo tissue","18F",
                              "Matched synthetic gamma-counter tissue activities and masses for the PET recovery troubleshooting case.")

    return {"DEMO · Calibration standards":cal,"DEMO · Detector normalisation":norm,
            "DEMO · Hypothetical tissue study":tissue,"DEMO · PET ROI summary":pet_demo,
            "DEMO · Geometry challenge":geometry,
            "CASE · High-rate calibration non-linearity":sat_cal,
            "CASE · Detector D low response":trouble_norm,
            "CASE · Rack position response":rack_case,
            "CASE · PET small-ROI recovery":pet_recovery,
            "CASE · Matched ex-vivo tissue":gamma_recovery}


THEORY_LESSONS = {
    "01": {
        "objectives": ["Distinguish detected count rate from physical activity (Bq).", "Apply background, dead-time and response corrections in the right order.", "Report activity at a stated reference time and normalise to tissue mass when needed."],
        "mechanism": "A gamma counter registers electrical pulses produced after energy deposition in its scintillator. The observed event rate depends on activity, photon emission probability, intrinsic detector efficiency, energy-window selection and sample presentation. Therefore, a count rate is not an activity until it is linked to an empirical sensitivity measured with a suitable standard.",
        "terms": [("A(t₀)", "Activity at the chosen reference time, in Bq (s⁻¹)."), ("Rgross, Rbg, Rnet", "Gross, background and net count rates, in counts s⁻¹ (cps)."), ("Scal", "Empirical sensitivity in cps/Bq for the radionuclide, window, detector and geometry used."), ("Gdet, Ggeom, Fmatrix", "Dimensionless detector, geometry and matrix response factors; define their convention before applying them."), ("m", "Tissue mass in grams for %ID/g." )],
        "example": "Suppose the dead-time-corrected gross rate is 1,420 cps and the separately measured background is 120 cps. Then Rnet = 1,300 cps. With Scal = 0.130 cps/Bq and an overall response factor of 0.95, Aest = 1,300/(0.130 × 0.95) = 10,526 Bq at the count time. For a 0.250 g tissue, convert to %ID/g only after decay-correcting to the injection reference time and dividing by injected activity.",
        "caution": "Use a background measurement with its own counting duration and propagate its uncertainty. Do not subtract counts acquired over unequal times without first converting to rates. The example assumes the response factors are known and that count-rate linearity is valid.",
        "observe": "Increase background or shorten the count time. Watch both the activity estimate and its uncertainty; a numerically similar net rate can have very different precision."
    },
    "02": {
        "objectives": ["Explain why standards must match the sample measurement conditions.", "Estimate sensitivity and intercept using regression.", "Inspect residuals and identify non-linearity or saturation."],
        "mechanism": "Absolute calibration establishes the conversion between net count rate and known activity. A set of standards spanning the intended activity range is measured with the same radionuclide, energy window, detector, vial type, fill height and matrix as the unknowns. The slope is the empirical sensitivity. An intercept can represent residual background or an offset, but should be investigated rather than automatically accepted.",
        "terms": [
            ("n", "Number of distinct standard activity levels used in the regression (dimensionless). Replicate measurements are averaged within each level before the fit in this simulator."),
            ("A_bar", "Mean known standard activity, (1/n) times the sum of A_i, in Bq."),
            ("Rnet_bar", "Mean background-subtracted and dead-time-corrected net count rate across the n standard levels, in cps."),
            ("A_i", "Known activity of calibration standard i at the stated reference time, in Bq."),
            ("Rnet_i", "Mean measured net count rate for standard level i after the simulator's non-paralysable dead-time correction, in cps."),
            ("S_cal", "Fitted calibration slope (empirical sensitivity), in cps/Bq."),
            ("b", "Fitted regression intercept, in cps. A non-zero intercept should be investigated."),
            ("epsilon_i", "Residual for standard i: observed mean rate minus fitted mean rate, in cps."),
            ("R_squared", "Coefficient of determination. R-squared = 1 - SSE/SST for this intercept-included ordinary least-squares fit. It is not R-bar and is not, by itself, proof of linearity or calibration validity."),
        ],
        "example": "A fitted line R_net = 0.142 A + 8 gives a sensitivity of 0.142 cps/Bq and an 8 cps intercept. An unknown measured at 1,430 cps gives A_est = (1,430 - 8)/0.142 = 10,014 Bq. This estimate is defensible only within the calibrated range and under matched conditions.",
        "caution": "Do not force the regression through zero unless the physics and blank measurements justify it. Inspect residuals, replicate variability, uncertainty weighting and high-activity dead-time losses. R^2 alone is not a calibration acceptance criterion. If the calibration rates were already corrected for detector response, geometry or matrix, those same corrections are already represented in the fitted sensitivity: do not apply them a second time to the unknown. If the unknown requires an additional correction not represented in the standards, document and apply it once.",
        "observe": "Generate a standard series and inspect the fitted slope, intercept and residuals. Consider what happens when the highest standards begin to depart from linearity."
    },
    "03": {
        "objectives": ["Distinguish relative detector normalisation from absolute calibration.", "Calculate channel-specific sensitivity and correction factors.", "Understand why normalisation requires a common reference source and matched geometry."],
        "mechanism": "Detector normalisation compensates for relative sensitivity differences between detector positions or channels. The same reference source is measured in every channel. Each channel's sensitivity is compared with the fleet mean, and a multiplicative factor maps its response to that mean. This equalises relative response; it does not independently establish activity in Bq.",
        "terms": [("Sj", "Measured sensitivity of detector j, usually net cps divided by reference activity."), ("S̄", "Mean sensitivity across the selected detector set."), ("Fj", "Dimensionless correction factor S̄/Sj."), ("Rnorm,j", "Measured net rate multiplied by Fj."), ("Aref", "Known activity of the common normalisation source." )],
        "example": "If the six-detector mean sensitivity is 0.140 cps/Bq and detector D measures 0.147 cps/Bq, FD = 0.140/0.147 = 0.952. A raw rate of 1,470 cps from D becomes 1,400 cps after relative normalisation. A separate absolute calibration is still needed to convert that rate to Bq.",
        "caution": "Use the same radionuclide, reference time, vial, fill height, positioning, energy window and background protocol. The factor can drift with detector service, gain or energy-window changes; repeat normalisation after relevant changes.",
        "observe": "Compare raw detector responses and corrected responses. The corrected channels should converge around a common mean, while random counting scatter remains."
    },
    "04": {
        "objectives": ["Explain how vial shape, fill height and matrix composition affect efficiency.", "Apply geometry and matrix response factors only within their validated scope.", "Propagate calibration and correction-factor uncertainty."],
        "mechanism": "Even with identical activity, sample geometry changes the fraction of emitted photons that reach the detector and fall inside the selected energy window. Fill height, vial diameter, wall material, source distribution and matrix attenuation/scatter all matter. A calibration made with one geometry is not automatically transferable to another.",
        "terms": [("Ggeom", "Relative response for the sample geometry compared with the calibration geometry."), ("Fmatrix", "Relative response associated with composition or matrix, using the convention defined in the model."), ("Scal", "Sensitivity under the calibration reference conditions."), ("u(x)", "Standard uncertainty of quantity x." )],
        "example": "A sample produces 1,235 cps. With Scal = 0.145 cps/Bq, Ggeom = 0.90 and Fmatrix = 0.97, the corrected activity is 1,235/(0.145 × 0.90 × 0.97) = 9,802 Bq. Ignoring geometry and matrix would underestimate the activity relative to this model.",
        "caution": "Correction factors are empirical and container-specific. Avoid applying a geometry correction twice if the calibration sensitivity already incorporates that geometry. For substantial attenuation or complex geometries, establish matched standards or use a validated physical model.",
        "observe": "Hold nominal activity constant and change vial geometry. Compare the naive activity estimate with the corrected estimate and discuss the extra uncertainty introduced by the correction."
    },
    "05": {
        "objectives": ["Decay-correct tissue activity to a common reference time.", "Calculate %ID and %ID/g without confusing the two quantities.", "Interpret tumour-to-background ratios and study-arm assumptions."],
        "mechanism": "Ex-vivo biodistribution measures activity in harvested tissues. Samples are weighed and counted, then the measured activity is decay-corrected to the injection (or another explicitly defined) reference time. %ID describes the fraction of injected activity in an organ; %ID/g additionally divides by tissue mass and therefore describes uptake concentration by mass.",
        "terms": [("Atissue(t)", "Measured tissue activity at counting time t."), ("Ainj(t₀)", "Injected activity at reference time t₀, decay-corrected consistently."), ("m", "Tissue mass in grams."), ("%ID", "100 × tissue activity at t₀ / injected activity at t₀."), ("%ID/g", "%ID divided by tissue mass in grams."), ("TBR", "Tumour %ID/g divided by the selected reference-tissue %ID/g." )],
        "example": "If 12,000 Bq is measured in a tissue 60 min after injection, with a 60 min half-life, its activity at injection is 24,000 Bq. For an injected activity of 2.0 MBq, %ID = 100 × 24,000/2,000,000 = 1.20%. If tissue mass is 0.30 g, %ID/g = 4.00 %ID/g.",
        "caution": "The injected activity must be defined consistently (e.g. net administered activity after residual syringe activity). Record exact count time, mass, background, aliquot/dilution and radionuclide. %ID/g is not the same as activity concentration in Bq/mL.",
        "observe": "Compare tumour %ID with tumour %ID/g. A small tumour can have a high %ID/g without containing the largest total activity."
    },
    "06": {
        "objectives": ["Match in-vivo image ROIs to ex-vivo samples by tissue and time.", "Convert gamma-counter activity to concentration using sample volume.", "Interpret bias and limits of agreement rather than relying only on correlation."],
        "mechanism": "PET/SPECT estimates activity concentration in reconstructed voxels or ROIs, while a gamma counter measures total activity in a harvested sample or aliquot. For a fair comparison, use the same tissue definition, correct both measurements to a common time, convert total sample activity to concentration, and account for sample mass, density, volume, recovery coefficient and uncertainty.",
        "terms": [("Agamma", "Ex-vivo sample activity at the gamma count time."), ("Vsample", "Counted sample volume; may be estimated as mass/density when appropriate."), ("Cgamma", "Ex-vivo activity concentration in Bq/mL."), ("Cimage", "Image-derived ROI activity concentration in Bq/mL."), ("RC", "Recovery coefficient; apply only if image data have not already been recovery-corrected."), ("Bias / LoA", "Mean paired difference and approximate 95% limits of agreement." )],
        "example": "A 0.25 g tissue at an assumed density of 1.05 g/mL has volume 0.238 mL. If its decay-corrected ex-vivo activity is 4,760 Bq, Cgamma = 20,000 Bq/mL. Compare this with the matched image ROI after aligning both to the same reference time and documenting any recovery correction.",
        "caution": "A strong correlation does not establish agreement. ROI mismatch, partial-volume effects, spill-in/out, segmentation, tissue heterogeneity, sample loss and activity timing can create systematic differences. Do not apply RC twice.",
        "observe": "Inspect the identity line and Bland–Altman differences. Ask whether the difference changes with concentration or tissue size."
    },
    "07": {
        "objectives": ["Explain why sample identity and position metadata are part of quantitative quality control.", "Calculate net count rates for unequal sample and background durations.", "Recognise position effects, carryover and sample swaps."],
        "mechanism": "An automated rack repeats a defined acquisition protocol across multiple labelled positions. Each result must retain sample ID, rack position, detector assignment, count duration, background, timing and calibration metadata. The data pipeline is only as reliable as the identity and traceability attached to each measurement.",
        "terms": [("Ni", "Observed counts for sample i."), ("ti", "Live counting duration for sample i."), ("Rsample", "Sample counts divided by sample counting time."), ("Rbackground", "Background counts divided by background counting time."), ("Rnet", "Difference of sample and background rates."), ("RSD", "Relative standard deviation, often 100 × SD/mean." )],
        "example": "A sample yields 18,240 counts in 60 s (304 cps). A background run yields 900 counts in 300 s (3 cps). The net rate is 301 cps. The background count duration differs, so subtracting raw counts (18,240 − 900) would be incorrect.",
        "caution": "A teaching simulator does not model all real transport faults, carryover, contamination or barcode failures. In a real lab, use independent sample identity checks, blanks and QC positions.",
        "observe": "Look for an outlier by rack position or detector. Determine whether the variation follows the sample, the detector or the position."
    },
    "08": {
        "objectives": ["Explain the pulse-height-to-energy relationship in a scintillation detector.", "Calibrate channel number to energy and interpret photopeak resolution.", "Estimate peak area after accounting for continuum beneath the ROI."],
        "mechanism": "Energy deposited in the scintillator produces light. The photosensor converts this light into an electrical pulse whose height is approximately related to deposited energy. A multichannel analyser bins pulse heights into channels. The spectrum therefore contains photopeaks, Compton continua and other response features; the exact shape depends on radionuclide emissions and detector response.",
        "terms": [("ch", "Multichannel analyser channel index."), ("E0, g", "Energy-calibration offset (keV) and gain (keV/channel)."), ("FWHM", "Full width at half maximum of a fitted photopeak."), ("RE", "Percentage energy resolution: 100 × FWHM / peak energy."), ("NROI", "Counts within the selected peak window."), ("Ncontinuum", "Estimated underlying continuum contribution." )],
        "example": "If two reference peaks at 140.5 and 511.0 keV are observed at channels 94 and 341, the gain is (511.0−140.5)/(341−94) = 1.50 keV/channel. If a 511 keV peak has a 46 keV FWHM, the energy resolution is 100×46/511 = 9.0%.",
        "caution": "The teaching spectrum is a simplified response model. Real spectra may require nonlinear energy calibration, peak fitting, pile-up assessment, radionuclide-specific branching data and validated continuum estimation. ROI boundaries affect net peak area.",
        "observe": "Change gain, offset, resolution and ROI width. Watch how peak location, width and the estimated net peak area change."
    },
    "09": {
        "objectives": ["Identify the minimum metadata required for quantitative image and counter data.", "Convert Bq, kBq and MBq consistently.", "Decay-normalise data without silently inferring missing units or time origins."],
        "mechanism": "Data from different instruments often use different units, clocks, ROI labels, volume definitions and calibration conventions. Before comparison, standardise units, radionuclide, reference time, activity concentration, ROI/tissue identity and uncertainty fields. Missing metadata should be flagged rather than guessed.",
        "terms": [("C", "Activity concentration, preferably stored in a single canonical unit such as Bq/mL."), ("t−t₀", "Elapsed time from measurement to the chosen reference time."), ("T1/2", "Radionuclide physical half-life."), ("ρ", "Tissue density used to convert mass to volume when justified."), ("RC", "Image recovery coefficient, with correction status explicitly recorded." )],
        "example": "A value of 12.5 kBq/mL equals 12,500 Bq/mL. For a radionuclide with a 60 min half-life measured 30 min after reference time, the reference-time concentration is 12,500 × 2^(30/60) = 17,678 Bq/mL.",
        "caution": "Confirm whether timestamps are injection time, acquisition start, midpoint or reconstruction reference. Confirm whether units are Bq/mL, kBq/mL, SUV or arbitrary counts before importing.",
        "observe": "Import the template, map each field explicitly and inspect the standardised output before sending it to the agreement analysis."
    }
}

LIVE_LESSONS = {
    "Unknown sample assay": ("Counts become activity only through a calibrated response", "An unknown assay combines an independent background measurement with the sensitivity established for the radionuclide and acquisition configuration.", [r"R_{\mathrm{net}}=\frac{N_{\mathrm{gross}}}{t_{\mathrm{gross}}}-\frac{N_{\mathrm{bg}}}{t_{\mathrm{bg}}}", r"A_{\mathrm{est}}=\frac{R_{\mathrm{net,corr}}}{S_{\mathrm{cal}}G_{\mathrm{det}}G_{\mathrm{geom}}F_{\mathrm{matrix}}}", r"u(R_{\mathrm{net}})=\sqrt{\frac{N_{\mathrm{gross}}}{t_{\mathrm{gross}}^2}+\frac{N_{\mathrm{bg}}}{t_{\mathrm{bg}}^2}}"], "For 1,420 cps gross and 120 cps background, the net rate is 1,300 cps. At 0.130 cps/Bq and combined response 0.95, the estimate is approximately 10.5 kBq.", "Use the same isotope, energy window and validated geometry for calibration and unknowns. Background measurements have their own duration and uncertainty."),
    "Calibration standards sequence": ("Empirical sensitivity is the slope of response against activity", "A calibration is valid only for the isotope, energy window, detector, geometry and matrix represented by the standards. Fit net count rate against known activity and examine residuals, not only R-squared.", [r"R_{\mathrm{net},i}=S_{\mathrm{cal}}A_i+b+\varepsilon_i", r"S_{\mathrm{cal}}=\frac{\mathrm{d}R_{\mathrm{net}}}{\mathrm{d}A}", r"A_{\mathrm{unknown}}=\frac{R_{\mathrm{unknown}}-b}{S_{\mathrm{cal}}}"], "If mean rates are 150, 750 and 1,510 cps for 1, 5 and 10 kBq, the approximate sensitivity is 150 cps/kBq. Inspect residuals before transferring the fit.", "Check linearity, background, uncertainty and high-activity dead-time effects. Do not extrapolate beyond the validated calibration range."),
    "Detector normalisation": ("Normalisation equalises relative channel response", "Measure one common reference source in every detector position. Each channel's sensitivity is compared with the fleet mean and a multiplicative correction maps the response to that mean. This is a relative correction, not an absolute activity calibration.", [r"S_j=\frac{R_{\mathrm{net},j}}{A_{\mathrm{ref}}}", r"F_j=\frac{\overline{S}}{S_j}", r"R_{\mathrm{norm},j}=F_jR_{\mathrm{net},j}"], "If a channel is 5% more sensitive than the fleet mean, its factor is approximately 1/1.05 = 0.952. Random counting scatter remains after normalisation.", "Use the same radionuclide, reference time, vial, fill height, positioning, energy window and background protocol."),
    "Geometry and matrix challenge": ("Sample presentation is part of the measurement model", "Vial diameter, fill height, wall composition, source distribution and matrix can alter counting efficiency and scatter. A geometry factor is specific to the reference configuration and method used to establish it.", [r"R_{\mathrm{net}}=S_{\mathrm{cal}}A G_{\mathrm{geom}}F_{\mathrm{matrix}}", r"A_{\mathrm{corr}}=\frac{R_{\mathrm{net}}}{S_{\mathrm{cal}}G_{\mathrm{geom}}F_{\mathrm{matrix}}}"], "If geometry response is 0.90 and matrix response is 0.97, the combined response is 0.873. Without correction, the response is 12.7% below the reference configuration.", "Do not apply correction factors twice or transfer them outside their validated geometry and matrix scope."),
    "Hypothetical tissue study": ("Biodistribution requires time, activity and mass normalisation", "Each harvested tissue is counted and weighed. Activity is corrected to injection reference time, then expressed as a fraction of injected activity and divided by tissue mass for %ID/g.", [r"A_{\mathrm{tissue}}(t_0)=A_{\mathrm{tissue}}(t)e^{\lambda(t-t_0)}", r"\%\,\mathrm{ID}=100\frac{A_{\mathrm{tissue}}(t_0)}{A_{\mathrm{injected}}(t_0)}", r"\%\,\mathrm{ID}/\mathrm{g}=\frac{\%\,\mathrm{ID}}{m_{\mathrm{tissue}}}", r"\mathrm{TBR}=\frac{(\%\,\mathrm{ID}/\mathrm{g})_{\mathrm{tumour}}}{(\%\,\mathrm{ID}/\mathrm{g})_{\mathrm{reference\ tissue}}}"], "A tissue containing 24 kBq at injection reference, with 2 MBq injected and a mass of 0.30 g, has 1.2%ID and 4.0 %ID/g.", "The three simulated study arms are redistribution scenarios, not pharmacokinetic predictions. State the reference time and use consistent injected activity."),
    "PET/SPECT cross-calibration": ("Compare like with like: same tissue, volume and reference time", "Image-derived concentration and ex-vivo total activity are different quantities. Convert sample activity into concentration and align timing before calculating paired differences. Apply recovery correction only when it has not already been applied during image reconstruction or analysis.", [r"V_{\mathrm{sample}}\approx\frac{m_{\mathrm{sample}}}{\rho}", r"C_{\gamma}(t_0)=\frac{A_{\gamma}(t)e^{\lambda(t-t_0)}}{V_{\mathrm{sample}}}", r"C_{\mathrm{image,corr}}=\frac{C_{\mathrm{image}}}{RC}"], "A 0.25 g sample at 1.05 g/mL has a volume of 0.238 mL. At 4,760 Bq, its concentration is approximately 20 kBq/mL.", "Check recovery-correction status, ROI/tissue matching, sample density and the reference clock."),
    "Automated sample rack": ("Traceability is part of quantitative quality assurance", "Every result must remain linked to sample identity, rack position, detector, timing, background and calibration metadata. Subtract sample and background rates after converting each count total using its own live time.", [r"R_i=\frac{N_i}{t_i}", r"R_{\mathrm{net},i}=R_{\mathrm{sample},i}-R_{\mathrm{bg}}", r"u(N_i)=\sqrt{N_i}"], "18,240 counts in 60 s is 304 cps. A 900-count blank in 300 s is 3 cps, giving 301 cps net.", "Subtract rates, not raw counts, when sample and background durations differ. Retain the rack map and all sample identifiers."),
    "Gamma spectroscopy and ROI analysis": ("The multichannel analyser turns pulse heights into an energy spectrum", "Energy calibration maps channel number to energy. Peak width reflects detector resolution; a net ROI area requires an estimate of the continuum beneath the photopeak.", [r"E_i=E_0+g\,i", r"\mathrm{FWHM}=2\sqrt{2\ln 2}\,\sigma_E", r"R_E=100\frac{\mathrm{FWHM}}{E_{\mathrm{peak}}}", r"N_{\mathrm{net,ROI}}=N_{\mathrm{gross,ROI}}-\widehat{N}_{\mathrm{continuum,ROI}}"], "A 46 keV FWHM at a 511 keV peak corresponds to 9.0% energy resolution.", "ROI boundaries, continuum estimation and neighbouring emissions influence the net peak area. A peak's centroid must be checked after gain or offset changes."),
    "Background subtraction": ("Background is a second measurement with its own uncertainty", "Background may arise from environmental radiation, detector materials, contamination and electronic contributions. Independent blank counting reduces bias but contributes statistical uncertainty.", [r"R_{\mathrm{net}}=\frac{N_{\mathrm{gross}}}{t_{\mathrm{gross}}}-\frac{N_{\mathrm{bg}}}{t_{\mathrm{bg}}}", r"u(R_{\mathrm{net}})=\sqrt{\frac{N_{\mathrm{gross}}}{t_{\mathrm{gross}}^2}+\frac{N_{\mathrm{bg}}}{t_{\mathrm{bg}}^2}}"], "For 1,200 counts in 10 s and 300 counts in 30 s, the net rate is 110 cps and its Poisson standard uncertainty is approximately 1.20 cps.", "Do not treat background as exact or subtract counts from measurements with unequal live times."),
    "Dead-time behaviour": ("The detector has a finite event-processing capacity", "At high event rates, pulses may arrive while the system is unable to process a previous event. The simulator uses a non-paralysable model: an event arriving during dead-time is lost but does not extend the dead period.", [r"R_{\mathrm{obs}}=\frac{R_{\mathrm{true}}}{1+R_{\mathrm{true}}\tau}", r"R_{\mathrm{true}}=\frac{R_{\mathrm{obs}}}{1-R_{\mathrm{obs}}\tau}", r"L=1-\frac{R_{\mathrm{obs}}}{R_{\mathrm{true}}}"], "For a true rate of 20,000 cps and a dead-time of 2.5 μs, the observed rate is approximately 19,048 cps, a loss of 4.8%.", "The inversion becomes unstable as R_obs τ approaches 1. Real instruments may use different timing and correction implementations; remain within the validated operating range."),
    "Decay during acquisition": ("A finite acquisition integrates a changing activity", "The event rate decreases during measurement as the radionuclide decays. Cumulative counts continue to rise, but the expected count rate in successive time intervals decreases. The size of this decline depends on the acquisition duration relative to the physical half-life.", [r"A(t)=A_0e^{-\lambda t}", r"\lambda=\frac{\ln 2}{T_{1/2}}", r"E[N_s(T)]=S_{\mathrm{cal}}A_0\frac{1-e^{-\lambda T}}{\lambda}"], "If the count duration equals one half-life, activity falls to 50% of its initial value by the end. For 60 s of 89Zr counting, the physical decline is very small because its half-life is long.", "Background is modelled as constant. Dead-time and detector response are simplified; distinguish real physical decay from Poisson fluctuations."),
    "Counting precision": ("Poisson counting uncertainty decreases with the square root of counts", "For independent Poisson events, the standard deviation of a count N is approximately the square root of N. Longer counting improves relative precision, but with diminishing returns.", [r"u(N)=\sqrt{N}", r"\frac{u(N)}{N}=\frac{1}{\sqrt{N}}", r"\frac{u(R_{\mathrm{net}})}{R_{\mathrm{net}}}\approx\frac{\sqrt{R_s+2R_b}}{R_s\sqrt{t}}"], "At 100 counts, relative standard uncertainty is 10%; at 400 counts it is 5%; at 10,000 counts it is 1%.", "For net measurements, include background uncertainty and calibration/correction-factor uncertainty where relevant."),
}


THEORY_EXPANDED = {
    "01": [
        "A gamma counter does not measure activity directly. It registers electrical pulses generated when radiation deposits energy in the detector. The number of pulses recorded per second is a count rate, and the relationship between that rate and activity depends on the radionuclide, emission spectrum, detector, energy window and sample presentation. A result in counts per second (cps) is therefore not interchangeable with a result in becquerels (Bq).",
        "The bridge between count rate and activity is an empirical sensitivity. It is established by measuring a standard with a known activity under defined conditions. For an unknown, the measured gross response is corrected for background and, where required, dead-time and relative response effects before division by the calibration sensitivity. Every factor must have a stated definition: a correction factor may multiply the measured response or modify the sensitivity, but the same convention must be used consistently throughout the calculation.",
        "A defensible assay also reports when the activity applies. The measured activity refers to the counting time; comparison with an injection-time value requires physical-decay correction. For tissue samples, mass normalisation is a further calculation performed after activity has been estimated. Keeping these steps separate prevents common errors such as treating cps as Bq, subtracting raw counts acquired for different durations, or confusing %ID with %ID/g."
    ],
    "02": [
        "Absolute calibration estimates the response of a measurement system to known amounts of a particular radionuclide. A standard series is more informative than a single point because it tests both sensitivity and the range over which the response is acceptably linear. The standards should span the activity range expected in the unknown samples and should be prepared in containers and matrices representative of the intended work.",
        "For each standard, convert gross and blank acquisitions to rates using their own live times, then apply the prescribed corrections. A linear model relates corrected net rate to reference activity. The slope is sensitivity in cps/Bq; the intercept captures a residual offset and should be interpreted rather than automatically forced to zero. Residuals show whether deviations are random or systematic. Curvature at high activity may indicate dead-time losses, while a changing response at low activity may reflect background or detection-limit effects.",
        "Calibration is conditional, not universal. Changing the isotope, energy window, detector position, vial, fill height, matrix or electronics settings can change the response. Record the calibration date, source certificate and reference date, radionuclide, geometry, window, count duration, background protocol, fit model, slope uncertainty and residual diagnostics. Use the fitted sensitivity only within the validated range and re-check it after relevant instrument or method changes."
    ],
    "03": [
        "Detector normalisation is a relative correction used when multiple detectors or counting positions do not have identical sensitivities. Small differences can arise from crystal dimensions, optical coupling, photomultiplier gain, electronics, geometry and energy-window response. If the same reference source is measured in each channel using the same vial and positioning protocol, the measured responses can be compared on a common basis.",
        "First calculate each detector sensitivity as background-corrected count rate divided by the reference activity at the measurement time. A fleet reference is then calculated, commonly the arithmetic mean or another explicitly specified robust statistic. The channel factor is the fleet sensitivity divided by the individual channel sensitivity. A detector that is more sensitive than the fleet mean receives a factor below one; a less sensitive detector receives a factor above one. Applying the factor brings relative responses towards a common value.",
        "Normalisation does not establish activity in Bq: it removes relative differences between channels. Absolute calibration is still required to convert the normalised count rate into activity. Normalisation also cannot repair unstable channels, contamination, non-linearity or mismatched geometries. Replicate measurements are valuable because correction factors calculated from noisy measurements inherit uncertainty. A good practical check is to compare the spread of channel responses before and after correction and confirm that no single channel remains an outlier."
    ],
    "04": [
        "Counting efficiency is affected by how the sample is presented to the detector. A small-volume tube, a wide vial and a high-fill container distribute activity differently relative to the detector well. Photon attenuation and scattering depend on the material, density, fill height and composition of the sample. Consequently, two samples containing the same activity can produce different count rates even when the radionuclide and instrument settings are unchanged.",
        "A geometry or matrix factor expresses the response relative to a defined reference arrangement. The factor must be measured or validated for the intended containers and matrix; it should not be treated as a universal physical constant. The direction of the factor must be clear. In this simulator, a factor below one represents a reduced response compared with the reference condition, so the estimated activity is obtained by dividing the measured rate by sensitivity multiplied by the response factor.",
        "The most reliable approach is to match standards to samples. When that is not practical, prepare a controlled geometry study with known activity, identical radionuclide and reproducible fill heights. Measure replicates, plot response against geometry, and quantify uncertainty in the correction. Avoid double correction: if the calibration sensitivity already includes the sample geometry, applying the same geometry factor again will bias the result."
    ],
    "05": [
        "Ex-vivo biodistribution links tissue uptake to the amount of administered radioactivity. After tissue collection, each sample is weighed and measured in a gamma counter. The counter result is converted into activity using the appropriate calibration and response corrections. Because samples are usually counted after injection, their activity must be referred to a common time before comparison with the injected activity or with other animals and tissues.",
        "Physical decay follows an exponential law. To correct a sample measured at time t back to an earlier reference time t0, multiply by exp[lambda(t-t0)], where lambda is the decay constant. The injected activity must use the same reference time and definition, typically net administered activity after accounting for residual activity in the syringe or delivery device. The fraction of injected dose in a tissue is then calculated as %ID. Dividing %ID by tissue mass gives %ID/g, a mass-normalised uptake index.",
        "%ID and %ID/g answer different questions. %ID describes the total fraction of the injected activity present in the whole tissue; %ID/g describes uptake per gram and can be high in a small tissue even when its total activity is modest. Report tissue mass, count time, injection reference time, radionuclide half-life, background, calibration and any aliquot or dilution factors. The simulator's control, tumour-enriched and tumour-reduced arms are hypothetical redistribution scenarios, not biological predictions."
    ],
    "06": [
        "PET and SPECT quantify activity within image voxels or segmented regions, generally reported as activity concentration such as Bq/mL. A gamma counter measures the total activity contained in a harvested tissue sample or aliquot. These are different measurands. A direct comparison between a PET concentration and a gamma-counter total activity is dimensionally invalid until the ex-vivo activity has been converted to concentration using a defensible sample volume.",
        "For a weighed tissue, volume may be estimated from mass divided by density, provided the density assumption is appropriate and clearly stated. The ex-vivo activity is decay-corrected to the image reference time, then divided by sample volume. Image data must also be checked for decay-correction convention, calibration, reconstruction scaling, ROI definition and partial-volume recovery. A recovery coefficient should only be applied when the image values have not already been corrected; applying it twice creates a systematic error.",
        "Agreement is more informative than correlation alone. Two methods can correlate strongly while one consistently overestimates the other. Examine paired differences, relative differences and a Bland-Altman plot, and consider whether bias changes with concentration, organ size or tissue type. Matching sample identity, acquisition timing, ROI definition and uncertainty is essential. Differences may arise from partial-volume effects, segmentation, spill-in/out, tissue heterogeneity, sample loss or calibration mismatch."
    ],
    "07": [
        "An automated rack increases throughput, but automation does not remove the need for measurement traceability. Each result must remain linked to a unique sample identifier, rack position, detector assignment, acquisition start and stop time, live time, background protocol, radionuclide, calibration and any correction factors. Without these fields, a numerically plausible result can still be assigned to the wrong sample or interpreted using the wrong calibration.",
        "For each position, calculate a sample rate from counts divided by sample live time. Calculate the background rate independently using the background acquisition time, then subtract rates rather than raw counts when acquisition times differ. Replicate or quality-control positions can reveal drift, position effects and unusual variability. The relative standard deviation is useful for summarising repeated measurements, but it should be interpreted alongside the expected Poisson uncertainty and any detector or geometry effects.",
        "A real laboratory workflow should include a defined rack map, barcode or other identity verification, blank positions where appropriate, checks for contamination or carryover, and review of missing or duplicated positions. This simulator focuses on the quantitative data path; it does not reproduce every mechanical fault, barcode failure or contamination pathway of a commercial sample changer."
    ],
    "08": [
        "A scintillation detector converts deposited radiation energy into light. A photosensor converts the light pulse into an electrical pulse, and the pulse height is approximately related to the energy deposited. A multichannel analyser sorts pulses into channel bins, producing a pulse-height spectrum. The horizontal axis is channel number until an energy calibration maps channels to keV; the vertical axis is the number of events recorded in each channel.",
        "A photopeak is formed when the detector absorbs the relevant photon energy within its response. Compton scattering and incomplete energy deposition contribute a continuum and other spectral structures. The photopeak position is used for energy identification, while its width reflects energy resolution and several detector-response contributions. FWHM is measured at half the peak height; percentage resolution is FWHM divided by peak energy, multiplied by 100. Better resolution means a narrower peak relative to its energy.",
        "Quantifying a peak requires defining a region of interest and estimating the continuum beneath it. A simple sideband or local-background estimate can be useful for teaching, but real data may require validated peak fitting, multiplet deconvolution and a documented continuum model. ROI width is a trade-off: too narrow a window loses true peak events, while too wide a window admits more continuum and possibly neighbouring emissions. The spectra in this application are illustrative isotope-dependent teaching models, not measured response functions for a particular commercial counter."
    ],
    "09": [
        "Quantitative analysis often combines files from scanners, gamma counters, dose calibrators and laboratory worksheets. These systems may use different units, clock conventions, ROI labels and definitions of activity concentration. Before comparing results, create a canonical data structure and document each transformation. Missing information should be flagged explicitly rather than inferred silently.",
        "Unit conversion is straightforward only after the original unit is known: 1 MBq is 10^6 Bq and 1 kBq is 10^3 Bq. Time metadata require equal care. A timestamp may refer to injection, acquisition start, acquisition midpoint, scan completion, reconstruction reference or tissue counting. Decay correction is only valid when the time origin and radionuclide half-life are known. The same activity should not be corrected twice if the source data are already decay-corrected.",
        "A robust import workflow validates required columns, parses numeric values, standardises units, checks duplicate sample labels and matches ROI names to tissue names. Record the source file and mapping decisions. For cross-modality comparisons, also document whether the image concentration includes attenuation, scatter, calibration and recovery corrections. A clean dataset is part of the measurement method, not merely a formatting convenience."
    ],
}

LIVE_EXPANDED = {
    "Unknown sample assay": "Treat the unknown as a complete measurement chain. First collect gross counts from the sample and a separate blank measurement. Convert both to rates using their own live times, then subtract background. Correct the response for dead-time using the model selected for the exercise, and convert the net rate to activity using the calibration sensitivity and only those detector, geometry and matrix factors that are not already embedded in that sensitivity. The result is an estimate at the measurement time; decay-correct it only when a different reference time is required.",
    "Calibration standards sequence": "A standard series tests whether the instrument response is proportional to activity under a fixed configuration. The independent variable is the known standard activity at a common reference time; the dependent variable is the corrected net count rate. Fit slope and intercept, inspect residuals and consider uncertainty in both axes where relevant. A visually straight line or a high R-squared value does not by itself establish a valid calibration: the standards must bracket the samples and the residual pattern must be acceptable.",
    "Detector normalisation": "The normalisation experiment deliberately measures one common source in several detector channels. Any persistent difference in mean cps/Bq represents relative channel response, while scatter between repeat measurements reflects counting statistics and reproducibility. Calculate a factor for each channel relative to the selected fleet reference, apply it to the raw response and compare the channel spread before and after correction. Normalisation equalises the relative response scale; it does not prove that the mean sensitivity is accurate in Bq or remove random measurement uncertainty.",
    "Geometry and matrix challenge": "Keep activity fixed while changing only the sample presentation. Differences in count rate then demonstrate how the measurement geometry and sample matrix affect response. Compare the uncorrected activity estimate with the estimate after applying the declared relative response factor. In real work, factors should be established with standards or validated calculations for the actual container, fill volume and matrix, with uncertainty retained when the factor is transferred to unknowns.",
    "Hypothetical tissue study": "This scenario follows an animal-study biodistribution workflow. Uptake fractions define the activity assigned to each tissue at a stated reference time. The tissue activity at collection is reduced by physical decay, a synthetic count measurement is generated, and the estimated activity is returned to the reference time before calculating %ID and %ID/g. Compare total organ uptake with mass-normalised uptake: a small tissue can have high %ID/g without containing the greatest total fraction of the injected dose.",
    "PET/SPECT cross-calibration": "This exercise compares two measurement methods that do not initially report the same quantity. Imaging provides an ROI activity concentration, while ex-vivo counting provides total activity in a sample. Convert the sample result to concentration using measured mass and a stated density or volume, decay-align the methods to a common reference time, and confirm recovery-correction status. Interpret paired differences and agreement limits rather than relying on correlation alone.",
    "Automated sample rack": "The rack exercise treats each position as a separate traceable measurement. Sample identity, rack position, detector assignment, acquisition time and calibration metadata are retained with the result. Examine whether apparent outliers follow a sample, a detector or a position. If sample and background live times differ, subtract rates rather than counts. In a real workflow, identity checks and blank/QC positions are as important as the acquisition itself.",
    "Gamma spectroscopy and ROI analysis": "The spectrum exercise focuses on the transformation from pulse height to energy and from a displayed photopeak to a quantitative net ROI area. Calibrate channel number to energy with reference peaks, inspect the peak position and width, and define the ROI and continuum estimation method. Change the resolution and window width to understand the balance between including photopeak events and excluding continuum or neighbouring emissions.",
    "Background subtraction": "The background run is an independent measurement, not a fixed number known without uncertainty. Gross and blank counts are generated over their own live times and converted to rates before subtraction. The variance of the net rate is the sum of the rate variances for independent Poisson measurements. Increasing background or shortening either acquisition reduces the precision of the net result, even if the expected net rate is unchanged.",
    "Dead-time behaviour": "Dead-time is the interval during which the system is unavailable to register another event after processing a pulse. As the true input rate rises, the observed rate increasingly under-reports the input. This exercise uses a non-paralysable model, in which events arriving during dead-time are lost but do not extend the dead period. The model is useful for illustrating rate loss, but actual instruments may use different electronics and correction implementations.",
    "Decay during acquisition": "A count is accumulated over a finite interval, during which the activity is changing. The expected sample counts are therefore the time integral of the decaying event rate, not simply the initial count rate multiplied by the full duration. Background is treated as constant in the teaching model. Compare the cumulative-count curve with the interval-rate curve: the cumulative total continues to rise, while the instantaneous sample rate falls gradually.",
    "Counting precision": "For independent Poisson events, the standard deviation of a gross count is approximately the square root of the count. Relative uncertainty therefore decreases as the square root of measurement time when the rate is stable. This is a diminishing-return relationship: halving relative uncertainty requires approximately four times as many counts and therefore four times the counting duration. For net rates, include both sample and background counting uncertainty, then add calibration and correction-factor uncertainty where appropriate.",
}



def render_math(expression: str) -> None:
    """Render LaTeX with consistent upright text subscripts across the guide.

    Accepts already-correct LaTeX unchanged and normalises common legacy forms
    such as R_{net}, S_{cal}, %ID/g, and A_{tissue} before handing them to
    Streamlit's MathJax renderer.
    """
    import re as _re
    eq = str(expression)
    symbol_map = [
        (r"\%ID/g", r"\%\,\mathrm{ID}/\mathrm{g}"),
        (r"\%ID", r"\%\,\mathrm{ID}"),
        (r"R_{net,corr}", r"R_{\mathrm{net,corr}}"),
        (r"R_{net,ROI}", r"R_{\mathrm{net,ROI}}"),
        (r"R_{net,i}", r"R_{\mathrm{net},i}"),
        (r"R_{norm,j}", r"R_{\mathrm{norm},j}"),
        (r"R_{gross,ROI}", r"R_{\mathrm{gross,ROI}}"),
        (r"N_{gross,ROI}", r"N_{\mathrm{gross,ROI}}"),
        (r"N_{continuum,ROI}", r"N_{\mathrm{continuum,ROI}}"),
        (r"A_{image,corr}", r"A_{\mathrm{image,corr}}"),
        (r"C_{image,corr}", r"C_{\mathrm{image,corr}}"),
        (r"A_{tissue}", r"A_{\mathrm{tissue}}"),
        (r"A_{injected}", r"A_{\mathrm{injected}}"),
        (r"A_{unknown}", r"A_{\mathrm{unknown}}"),
        (r"A_{est}", r"A_{\mathrm{est}}"),
        (r"A_{corr}", r"A_{\mathrm{corr}}"),
        (r"A_{gamma}", r"A_{\gamma}"),
        (r"A_{sample}", r"A_{\mathrm{sample}}"),
        (r"A_{ref}", r"A_{\mathrm{ref}}"),
        (r"R_{gross}", r"R_{\mathrm{gross}}"),
        (r"R_{bg}", r"R_{\mathrm{bg}}"),
        (r"R_{net}", r"R_{\mathrm{net}}"),
        (r"R_{obs}", r"R_{\mathrm{obs}}"),
        (r"R_{true}", r"R_{\mathrm{true}}"),
        (r"R_{unknown}", r"R_{\mathrm{unknown}}"),
        (r"R_{sample}", r"R_{\mathrm{sample}}"),
        (r"R_{ROI}", r"R_{\mathrm{ROI}}"),
        (r"S_{cal}", r"S_{\mathrm{cal}}"),
        (r"S_{norm,j}", r"S_{\mathrm{norm},j}"),
        (r"G_{det}", r"G_{\mathrm{det}}"),
        (r"G_{geom}", r"G_{\mathrm{geom}}"),
        (r"F_{matrix}", r"F_{\mathrm{matrix}}"),
        (r"N_{gross}", r"N_{\mathrm{gross}}"),
        (r"N_{bg}", r"N_{\mathrm{bg}}"),
        (r"N_{net}", r"N_{\mathrm{net}}"),
        (r"N_{sample}", r"N_{\mathrm{sample}}"),
        (r"N_{ROI}", r"N_{\mathrm{ROI}}"),
        (r"t_{gross}", r"t_{\mathrm{gross}}"),
        (r"t_{bg}", r"t_{\mathrm{bg}}"),
        (r"t_{sample}", r"t_{\mathrm{sample}}"),
        (r"t_{live}", r"t_{\mathrm{live}}"),
        (r"V_{sample}", r"V_{\mathrm{sample}}"),
        (r"C_{gamma}", r"C_{\gamma}"),
        (r"C_{image}", r"C_{\mathrm{image}}"),
        (r"E_{photopeak}", r"E_{\mathrm{photopeak}}"),
        (r"%ID/g", r"\%\,\mathrm{ID}/\mathrm{g}"),
        (r"%ID", r"\%\,\mathrm{ID}"),
    ]
    for old, new in symbol_map:
        eq = eq.replace(old, new)
    # Prevent a second pass from changing a command already formatted above.
    st.latex(eq)


def _book_block(label: str, body: str, tone: str = "teal") -> None:
    """Render teaching prose inside a high-contrast, coloured reading block."""
    safe_label = html_lib.escape(str(label))
    safe_body = html_lib.escape(str(body)).replace("\n\n", "</p><p class='book-block-body'>").replace("\n", "<br>")
    st.markdown(
        f"<div class='book-block book-block-{tone}'>"
        f"<div class='book-block-heading'>{safe_label}</div>"
        f"<p class='book-block-body'>{safe_body}</p></div>",
        unsafe_allow_html=True,
    )


def _book_list_block(label: str, items: list[str], ordered: bool = False, tone: str = "blue") -> None:
    """Render a readable list within a coloured teaching block."""
    tag = "ol" if ordered else "ul"
    rows = "".join(f"<li>{html_lib.escape(str(item))}</li>" for item in items)
    st.markdown(
        f"<div class='book-block book-block-{tone}'>"
        f"<div class='book-block-heading'>{html_lib.escape(label)}</div>"
        f"<{tag}>{rows}</{tag}></div>",
        unsafe_allow_html=True,
    )


def _book_terms_block(terms: list[tuple[str, str]]) -> None:
    """Render symbols and units as a compact, coloured reference block."""
    symbol_html = {
        "Rgross, Rbg, Rnet": "<i>R</i><sub>gross</sub>, <i>R</i><sub>bg</sub>, <i>R</i><sub>net</sub>",
        "Scal": "<i>S</i><sub>cal</sub>", "Gdet, Ggeom, Fmatrix": "<i>G</i><sub>det</sub>, <i>G</i><sub>geom</sub>, <i>F</i><sub>matrix</sub>",
        "Ai": "<i>A</i><sub>i</sub>", "Ri": "<i>R</i><sub>i</sub>", "εi": "ε<sub>i</sub>",
        "n": "<i>n</i>", "A_bar": "<span style='text-decoration:overline'><i>A</i></span>",
        "Rnet_bar": "<span style='text-decoration:overline'><i>R</i></span><sub>net</sub>",
        "A_i": "<i>A</i><sub>i</sub>", "Rnet_i": "<i>R</i><sub>net,i</sub>",
        "S_cal": "<i>S</i><sub>cal</sub>", "epsilon_i": "ε<sub>i</sub>",
        "R_squared": "<i>R</i><sup>2</sup>",
        "Sj": "<i>S</i><sub>j</sub>", "S̄": "<span style='text-decoration:overline'><i>S</i></span>",
        "Fj": "<i>F</i><sub>j</sub>", "Rnorm,j": "<i>R</i><sub>norm,j</sub>", "Aref": "<i>A</i><sub>ref</sub>",
        "Ggeom": "<i>G</i><sub>geom</sub>", "Fmatrix": "<i>F</i><sub>matrix</sub>",
        "Atissue(t)": "<i>A</i><sub>tissue</sub>(t)", "Ainj(t₀)": "<i>A</i><sub>inj</sub>(t₀)",
        "T1/2": "<i>T</i><sub>1/2</sub>", "Cgamma": "<i>C</i><sub>γ</sub>", "Cimage": "<i>C</i><sub>image</sub>",
        "Rsample": "<i>R</i><sub>sample</sub>", "Rbackground": "<i>R</i><sub>background</sub>",
        "Rnet": "<i>R</i><sub>net</sub>", "RSD": "RSD", "RC": "RC",
        "Agamma": "<i>A</i><sub>γ</sub>", "Vsample": "<i>V</i><sub>sample</sub>",
        "Atissue(t)": "<i>A</i><sub>tissue</sub>(t)", "Ainj(t₀)": "<i>A</i><sub>inj</sub>(t₀)",
        "Ni": "<i>N</i><sub>i</sub>", "ti": "<i>t</i><sub>i</sub>",
        "T1/2": "<i>T</i><sub>1/2</sub>", "εi": "ε<sub>i</sub>",
    }
    rows = "".join(
        "<tr><td><b>" + symbol_html.get(str(symbol), html_lib.escape(str(symbol))) + "</b></td><td>" +
        html_lib.escape(str(description)) + "</td></tr>"
        for symbol, description in terms
    )
    st.markdown(
        "<div class='book-block book-block-purple'>"
        "<div class='book-block-heading'>SYMBOLS AND UNITS</div>"
        "<table class='book-symbol-table'><thead><tr><th>Symbol / quantity</th>"
        "<th>Meaning and units</th></tr></thead><tbody>" + rows +
        "</tbody></table></div>",
        unsafe_allow_html=True,
    )


def make_teaching_figure(chapter: str) -> go.Figure:
    """Create a deterministic, illustrative teaching figure for each chapter."""
    fig = go.Figure()
    common = dict(template="plotly_dark", paper_bgcolor="#102231", plot_bgcolor="#0b1823",
                  font=dict(color="#f2f7fa", size=12), margin=dict(l=48, r=25, t=66, b=48),
                  height=360, hovermode="closest")
    if chapter == "01":
        # Measurement chain, drawn as connected stages.
        stages = [("Radioactive\nsample", 0.08), ("Photon\ninteractions", 0.30),
                  ("Scintillation\nlight", 0.52), ("Pulse-height\nanalysis", 0.74),
                  ("Calibrated\nactivity", 0.92)]
        for i, (label, x) in enumerate(stages):
            fig.add_shape(type="rect", x0=x-.075, x1=x+.075, y0=.39, y1=.67,
                          line=dict(color="#55e0b0", width=2), fillcolor="#173c4a")
            fig.add_annotation(x=x, y=.53, text=label.replace("\n", "<br>"), showarrow=False,
                               font=dict(size=11, color="#ffffff"))
            if i < len(stages)-1:
                nx=stages[i+1][1]
                fig.add_annotation(x=(x+nx)/2, y=.53, ax=x+.08, ay=.53,
                                   axref="x", ayref="y", text="", showarrow=True,
                                   arrowhead=3, arrowsize=1.2, arrowwidth=2, arrowcolor="#ffc45b")
        fig.update_xaxes(range=(0,1), visible=False); fig.update_yaxes(range=(0,1), visible=False)
        fig.update_layout(title="From radioactive decay to a reported activity", **common)
    elif chapter == "02":
        activity=np.array([500,1000,2500,5000,10000,20000,35000],float)
        sensitivity=.145; rate=sensitivity*activity+11
        offsets=np.array([-4,3,-2,5,-6,7,-3])
        fig.add_trace(go.Scatter(x=activity,y=rate+offsets,mode="markers",name="Reference standards",
                                 marker=dict(size=10,color="#55e0b0",line=dict(color="#ffffff",width=1))))
        xx=np.linspace(0,37000,100)
        fig.add_trace(go.Scatter(x=xx,y=sensitivity*xx+11,mode="lines",name="Linear calibration fit",
                                 line=dict(color="#ffc45b",width=3)))
        fig.update_layout(title="Calibration: net count rate versus known activity",xaxis_title="Known activity (Bq)",
                          yaxis_title="Net count rate (cps)",legend=dict(orientation="h",y=-.24),**common)
    elif chapter == "03":
        det=list("ABCDEF"); observed=np.array([.965,1.018,.992,1.041,.984,1.012]); mean=observed.mean()
        fig.add_trace(go.Bar(x=det,y=observed,name="Before normalisation",marker_color="#58b9ff"))
        fig.add_trace(go.Bar(x=det,y=observed*(mean/observed),name="After normalisation",marker_color="#55e0b0"))
        fig.add_hline(y=mean,line_dash="dash",line_color="#ffc45b",annotation_text="Fleet mean",annotation_font_color="#ffc45b")
        fig.update_layout(title="Relative detector response before and after normalisation",barmode="group",
                          xaxis_title="Detector channel",yaxis_title="Relative sensitivity",legend=dict(orientation="h",y=-.24),**common)
    elif chapter == "04":
        names=["Reference tube","Small vial","3 mL vial","5 mL tube","High-fill vial"]
        response=np.array([1.00,.98,.95,.90,.86])
        fig.add_trace(go.Bar(x=names,y=response,marker_color=["#55e0b0","#58b9ff","#58b9ff","#ffc45b","#ff829a"],
                             text=[f"{v:.2f}" for v in response],textposition="outside"))
        fig.update_layout(title="Illustrative response change with sample presentation",xaxis_title="Sample geometry",
                          yaxis_title="Relative response (reference = 1.00)",yaxis=dict(range=(0,1.18)),**common)
    elif chapter == "05":
        tissues=["Blood","Liver","Kidney","Tumour","Bone","Muscle"]
        pid=np.array([3.5,18.5,11.5,8.2,4.8,2.6]); mass=np.array([.42,1.25,.31,.24,.36,.58])
        pidg=pid/mass
        fig.add_trace(go.Bar(x=tissues,y=pid,name="%ID (whole tissue)",marker_color="#58b9ff"))
        fig.add_trace(go.Bar(x=tissues,y=pidg,name="%ID/g",marker_color="#55e0b0"))
        fig.update_layout(title="Total uptake and mass-normalised uptake answer different questions",barmode="group",
                          xaxis_title="Tissue",yaxis_title="Value (different units; compare patterns, not bar heights)",
                          legend=dict(orientation="h",y=-.24),**common)
    elif chapter == "06":
        pet=np.array([1800,4200,7600,11200,15800,20400,25800],float)
        gamma=pet*np.array([1.06,.96,1.03,.91,1.08,.98,1.02])
        lim=max(pet.max(),gamma.max())*1.08
        fig.add_trace(go.Scatter(x=pet,y=gamma,mode="markers+text",text=["ROI 1","ROI 2","ROI 3","ROI 4","ROI 5","ROI 6","ROI 7"],
                                 textposition="top center",name="Matched tissue/ROI pairs",marker=dict(size=10,color="#55e0b0")))
        fig.add_trace(go.Scatter(x=[0,lim],y=[0,lim],mode="lines",name="Identity line",line=dict(color="#ffc45b",dash="dash",width=2)))
        fig.update_layout(title="Paired PET and ex-vivo gamma-counter concentrations",xaxis_title="PET/SPECT concentration (Bq/mL)",
                          yaxis_title="Gamma-counter concentration (Bq/mL)",legend=dict(orientation="h",y=-.24),**common)
    elif chapter == "07":
        rows=[]
        for pos in range(1,21): rows.append((pos,(pos-1)%5,4-(pos-1)//5))
        for pos,x,y in rows:
            fig.add_trace(go.Scatter(x=[x],y=[y],mode="markers+text",text=[f"{pos:02d}"],textposition="middle center",
                                     marker=dict(size=43,color="#173c4a",line=dict(color="#55e0b0",width=2)),
                                     textfont=dict(color="#ffffff",size=10),showlegend=False,hovertemplate=f"Rack position {pos}<extra></extra>"))
        fig.update_xaxes(range=(-.6,4.6),visible=False); fig.update_yaxes(range=(-.6,4.6),visible=False,scaleanchor="x")
        fig.update_layout(title="Illustrative 20-position sample rack and identity map",**common)
    elif chapter == "08":
        x=np.linspace(0,700,2048); rng=np.random.default_rng(808)
        y=5+22*np.exp(-x/220)+850*np.exp(-.5*((x-140)/10)**2)+520*np.exp(-.5*((x-511)/18)**2)
        y=np.maximum(rng.poisson(y),0)
        fig.add_trace(go.Scatter(x=x,y=y,mode="lines",name="Illustrative spectrum",line=dict(color="#55e0b0",width=1.6)))
        fig.add_vrect(x0=485,x1=537,fillcolor="#ffc45b",opacity=.16,line_width=0,
                      annotation_text="Photopeak ROI",annotation_position="top left",annotation_font_color="#ffc45b")
        fig.update_layout(title="Pulse-height spectrum: continuum and photopeak regions",xaxis_title="Energy (keV)",
                          yaxis_title="Counts per channel",**common)
    else:
        labels=["Import raw\nmeasurements","Validate units\nand labels","Align reference\ntime","Apply defined\ncorrections","Compare and\nreport"]
        xs=np.linspace(.1,.9,len(labels))
        for i,(lab,xv) in enumerate(zip(labels,xs)):
            fig.add_shape(type="rect",x0=xv-.085,x1=xv+.085,y0=.39,y1=.65,
                          line=dict(color="#58b9ff",width=2),fillcolor="#172c48")
            fig.add_annotation(x=xv,y=.52,text=lab.replace("\n","<br>"),showarrow=False,font=dict(color="#ffffff",size=10))
            if i<len(xs)-1:
                fig.add_annotation(x=(xv+xs[i+1])/2,y=.52,ax=xv+.09,ay=.52,axref="x",ayref="y",
                                   text="",showarrow=True,arrowhead=3,arrowcolor="#ffc45b",arrowwidth=2)
        fig.update_xaxes(range=(0,1),visible=False); fig.update_yaxes(range=(0,1),visible=False)
        fig.update_layout(title="A traceable data-standardisation workflow",**common)
    return fig


def render_chapter_figure(chapter: str, live: bool = False) -> None:
    """Show a chapter-specific visual with a stable, unique Streamlit widget key."""
    if live:
        live_figure_chapters = {
            "Unknown sample assay": "01",
            "Calibration standards sequence": "02",
            "Detector normalisation": "03",
            "Geometry and matrix challenge": "04",
            "Hypothetical tissue study": "05",
            "PET/SPECT cross-calibration": "06",
            "Automated sample rack": "07",
            "Gamma spectroscopy and ROI analysis": "08",
            "Background subtraction": "01",
            "Dead-time behaviour": "01",
            "Decay during acquisition": "01",
            "Counting precision": "01",
        }
        chapter = live_figure_chapters.get(chapter, "01")
    chapter = chapter if chapter in {f"{i:02d}" for i in range(1,10)} else "01"
    st.markdown("<div class='book-section-label'>VISUAL GUIDE · CONCEPT TO MEASUREMENT</div>", unsafe_allow_html=True)
    fig = make_teaching_figure(chapter)
    st.plotly_chart(fig, use_container_width=True, key=f"{'live' if live else 'chapter'}_teaching_figure_{chapter}",
                    config={"displaylogo": False, "scrollZoom": False})
    takeaways = {
        "01":"Follow the chain from radioactive transformation to electrical pulse, then from count rate to calibrated activity.",
        "02":"The calibration slope is the system sensitivity. Residuals and the intercept help reveal departures from the assumed linear response.",
        "03":"Normalisation reduces relative differences between channels; it does not replace absolute calibration in cps/Bq.",
        "04":"The same activity can produce different count rates when sample geometry or matrix changes.",
        "05":"%ID describes the total fraction in the tissue, whereas %ID/g describes uptake per unit mass.",
        "06":"Compare matched concentrations and inspect agreement around the identity line; correlation alone is not sufficient.",
        "07":"Every result must remain linked to its sample identity and physical rack position.",
        "08":"A photopeak is quantified within an ROI after accounting for the underlying continuum.",
        "09":"Unit validation and time alignment must precede quantitative comparison."
    }
    _book_block("HOW TO READ THIS FIGURE", takeaways[chapter], tone="green")



TROUBLESHOOTING_CASES = {
    "single": {
        "title": "Unexpectedly low activity estimate",
        "setting": "A tissue aliquot is measured at a lower activity than expected from the preparation record.",
        "observations": ["The calibration standard was counted in a narrow 13 mm tube.", "The tissue sample was transferred into a wider vial with a different fill height.", "The background was measured separately and has a different live time."],
        "task": "Check the count-rate calculation first, then decide whether the calibration sensitivity is transferable to the sample geometry. Quantify the effect of applying a matched geometry factor.",
        "diagnosis": "A geometry mismatch can bias the inferred activity. Unequal sample/background times must be handled as rates, not by subtracting raw counts."
    },
    "calibration": {
        "title": "High-activity standards fall below the calibration line",
        "setting": "Low and mid-range standards look linear, but the highest standards produce less signal than expected.",
        "observations": ["The deviation increases with count rate.", "The calibration intercept is small at low activity.", "The highest standard approaches the counter's high-rate operating region."],
        "task": "Start with Demonstration strength = Clear non-linearity (recommended), Maximum activity = 2,000,000 Bq, and the current dead-time setting (for example 2.5 µs). Generate the standards and compare the orange raw observed-rate curve with the green dead-time-corrected measurements and blue ideal response. Then switch to Near-linear and Strong count-rate loss to see how the curvature changes. Plot residuals against activity and decide whether the highest standards remain inside the validated linear range.",
        "diagnosis": "Rate-dependent count loss is consistent with dead-time effects. A single straight-line sensitivity should not be extrapolated beyond the validated linear range."
    },
    "normalisation": {
        "title": "One detector channel fails the reference-source check",
        "setting": "The same sealed reference source gives reproducible but systematically different count rates across detector channels.",
        "observations": ["Five channels cluster close to the fleet mean.", "One channel is persistently low across replicate measurements.", "The source, count duration and energy window were held constant."],
        "task": "Calculate channel sensitivity and normalisation factors. Determine whether the outlying channel can be corrected by a stable multiplicative factor or whether it should be investigated before routine use.",
        "diagnosis": "A stable relative response difference can be normalised for comparison, but a large or changing deviation may indicate detector, electronics, positioning or energy-window problems. Normalisation does not replace QC."
    },
    "geometry": {
        "title": "A sample reads low after transfer to a different container",
        "setting": "A QC solution passes in the calibration tube but appears low when measured in a larger vial.",
        "observations": ["Nominal activity is unchanged.", "The vial has a greater diameter and a different fill height.", "A calibration sensitivity from the original tube is being reused."],
        "task": "Compare the same activity across each container. Estimate relative response factors and test whether geometry correction restores agreement with the reference measurement.",
        "diagnosis": "The sensitivity is conditional on geometry. A correction factor should be established with matched standards and an uncertainty assigned to its transfer."
    },
    "tissue": {
        "title": "The organ with the highest %ID is not the highest in %ID/g",
        "setting": "A biodistribution report gives different rankings when tissues are compared by total uptake and by mass-normalised uptake.",
        "observations": ["A large organ contains substantial total activity.", "A small tissue contains less total activity but has a smaller mass.", "Collection times and injected-activity reference times must be reconciled."],
        "task": "Decay-correct all tissues to the same reference time. Calculate %ID and %ID/g separately, then compare tumour-to-reference tissue ratios using %ID/g.",
        "diagnosis": "%ID represents the fraction of injected activity in the whole tissue; %ID/g is uptake per unit mass. The two metrics answer different biological questions."
    },
    "pet": {
        "title": "PET concentration disagrees with ex-vivo counting",
        "setting": "The PET-derived concentration for a small tumour ROI is lower than the concentration inferred from a harvested tissue sample.",
        "observations": ["The tumour ROI is close to the scanner's effective spatial resolution.", "The PET and gamma-counter measurements were not made at exactly the same time.", "A recovery coefficient is available but its application status is unclear."],
        "task": "Confirm units and timestamps, decay-align both measurements, convert ex-vivo activity to concentration, and inspect whether recovery correction has already been applied.",
        "diagnosis": "Partial-volume loss, timing differences, ROI definition, tissue volume/density assumptions and calibration bias can all contribute. Do not apply a recovery correction twice."
    },
    "rack": {
        "title": "A repeatable low result follows one rack position",
        "setting": "Different samples appear low when they are counted at the same position in an automated rack.",
        "observations": ["The deviation follows the rack position rather than the sample identity.", "Re-counting the sample in another position restores the expected response.", "The remaining detector channels pass reference checks."],
        "task": "Plot response by rack position and by sample ID. Separate a position-dependent effect from a detector-channel or sample-specific effect.",
        "diagnosis": "Possible causes include sample seating, positioning, rack indexing or position-specific detector response. Traceability requires retaining both sample ID and physical position."
    },
    "spectroscopy": {
        "title": "The photopeak is displaced from the selected ROI",
        "setting": "A known radionuclide's photopeak appears shifted, and the standard ROI returns an unexpectedly low net peak area.",
        "observations": ["The peak shape remains broadly similar.", "The centroid differs from the reference energy.", "The original channel-to-energy calibration has not been checked recently."],
        "task": "Recalibrate energy using reference peaks, inspect centroid and FWHM, then repeat ROI integration with a defined continuum estimate.",
        "diagnosis": "Gain or offset drift can move events outside a fixed ROI. A low ROI result does not automatically mean low activity."
    },
    "import": {
        "title": "Imported PET and gamma results differ by orders of magnitude",
        "setting": "An imported dataset produces implausibly large or small cross-modality differences.",
        "observations": ["One file reports concentration in kBq/mL while another reports Bq/mL.", "Time columns use different reference conventions.", "ROI/tissue labels are not fully harmonised."],
        "task": "Inspect source units and metadata before conversion. Standardise units, sample identities and time references, then compare matched quantities.",
        "diagnosis": "Unit scaling, inconsistent reference times, mismatched labels or duplicated decay correction can create large apparent disagreement."
    },
}

def render_troubleshooting_case(case_key: str) -> None:
    """Show a realistic laboratory case for attendees to diagnose before running the practical."""
    case = TROUBLESHOOTING_CASES.get(case_key)
    if not case:
        return
    _book_block("LABORATORY CASE FILE · " + case["title"], case["setting"], tone="gold")
    _book_list_block("OBSERVATIONS", case["observations"], tone="blue")
    _book_block("YOUR TASK", case["task"], tone="purple")
    with st.expander("Instructor note · likely interpretation", expanded=False):
        st.write(case["diagnosis"])


def theory_panel(kicker: str, title: str, overview: str, equations: list[str], practical: str, assumptions: str = "") -> None:
    """Render a booklet-style lesson with every teaching section in a coloured block."""
    lesson = None
    lesson_key = None
    is_live = "LIVE PRACTICAL THEORY" in kicker
    if is_live:
        scenario_label = kicker.split("/", 1)[-1].strip().casefold()
        match_key = next((key for key in LIVE_LESSONS if key.casefold() == scenario_label), None)
        if match_key:
            lesson = LIVE_LESSONS[match_key]
            lesson_key = match_key
    else:
        match = __import__("re").search(r"THEORY\s+(\d{2})", kicker)
        if match:
            lesson_key = match.group(1)
            lesson = THEORY_LESSONS.get(lesson_key)

    chapter_label = kicker.replace("THEORY", "CHAPTER").replace("LIVE PRACTICAL", "PRACTICAL CHAPTER")
    st.markdown(
        f"<div class='book-cover'><div class='book-series'>QUANTITATIVE NUCLEAR MEDICINE LABORATORY  /  FIELD GUIDE</div>"
        f"<div class='book-chapter'>{html_lib.escape(chapter_label)}</div>"
        f"<div class='book-title'>{html_lib.escape(title)}</div>"
        f"<div class='book-lede'>{html_lib.escape(overview)}</div></div>",
        unsafe_allow_html=True,
    )

    # The visual guide sits immediately after the chapter opening, before the full reading text.
    if lesson_key:
        render_chapter_figure(lesson_key, live=is_live)

    if lesson and not is_live:
        _book_list_block("01 · WHAT YOU WILL LEARN", lesson.get("objectives", []), tone="blue")
        detail_paragraphs = THEORY_EXPANDED.get(lesson_key, [lesson.get("mechanism", "")])
        for index, para in enumerate(detail_paragraphs, start=1):
            _book_block(f"02 · THE SCIENCE BEHIND THE MEASUREMENT · {index}", para, tone="teal")

        workflow = {
            "01": ["Acquire the sample and a separate background or blank measurement.", "Convert each count total to a rate using its own live time.", "Apply the defined response corrections and calibration sensitivity.", "Report activity at the stated count time, then decay-correct if another reference time is needed."],
            "02": ["Prepare a series of traceable standards across the expected sample range.", "Measure standards and blanks with matched geometry and acquisition settings.", "Fit net count rate against known activity; retain slope, intercept and residuals.", "Verify linearity and uncertainty before applying the fitted sensitivity to unknowns."],
            "03": ["Measure the same reference source in each detector channel using matched positioning.", "Calculate each channel sensitivity and the chosen fleet reference.", "Derive and apply the channel correction factor.", "Compare response spread before and after normalisation; retain replicate uncertainty."],
            "04": ["Choose a reference container and define the nominal activity.", "Change one geometry or matrix feature at a time.", "Measure replicates and calculate relative response factors.", "Apply the factors once and assess the remaining uncertainty."],
            "05": ["Record injected activity, residual activity, injection time and tissue collection time.", "Weigh each tissue and measure its gross and background counts.", "Convert counts to activity and decay-correct to the reference time.", "Calculate %ID and %ID/g, then compare tissues and study arms."],
            "06": ["Match image ROIs and harvested tissues by identity and anatomy.", "Confirm units, timing, decay convention and recovery-correction status.", "Convert ex-vivo total activity into concentration using volume.", "Assess paired bias and agreement, not correlation alone."],
            "07": ["Define a rack map and unique sample identifiers before acquisition.", "Retain position, detector, live time and calibration metadata for every sample.", "Convert sample and background measurements to rates.", "Review missing positions, outliers, replicates and QC samples."],
            "08": ["Calibrate channel number to energy using reference peaks.", "Inspect peak centroid, FWHM and energy resolution.", "Define an ROI around the photopeak and estimate the local continuum.", "Compare net peak area as gain, resolution and ROI boundaries change."],
            "09": ["Inspect the input file and identify original units and time references.", "Map sample/ROI identity and required metadata explicitly.", "Convert to canonical units and decay-align only once.", "Review the standardised table and preserve the import mapping."],
        }.get(lesson_key, [])
        _book_list_block("03 · MEASUREMENT WORKFLOW", workflow, ordered=True, tone="blue")
        _book_block("04 · EQUATIONS USED IN THIS CHAPTER", "These equations define the measurement model used by the practical. Keep units consistent and confirm whether a correction factor is already included in the calibration sensitivity.", tone="purple")
        for equation in equations:
            render_math(equation)
        if lesson_key == "02":
            _book_block(
                "READING THE REGRESSION NOTATION",
                "The overbar denotes an arithmetic mean across the n distinct standard activity levels. A-bar is the mean of the known activities A_i; R-net-bar is the mean of the corresponding mean net count rates. In this simulator, replicate rates are averaged within each activity level and an unweighted ordinary least-squares fit is applied to those level means. R-squared is the coefficient of determination: R² = 1 - SSE/SST, where SSE is the residual sum of squares and SST is the total sum of squares about R-net-bar. It is not R-bar. For this intercept-included in-sample fit, R² ranges from 0 to 1; a high value does not rule out systematic curvature, an influential point or an unsuitable calibration range.",
                tone="blue",
            )
            _book_block(
                "WHEN IS A CORRECTION ALREADY INCLUDED?",
                "Correction convention in this simulator: the calibration response R_net,i is formed by subtracting the independently measured background rate after applying the same non-paralysable dead-time correction to gross and background rates. Unknown measurements use that same correction pathway; do not apply a second dead-time correction after using S_cal. The calibration standards are simulated at the reference detector, geometry and matrix settings (relative factors equal to 1.0), so those factors are not absorbed into S_cal. If an unknown is measured in a different detector, vial geometry or matrix, apply the relevant validated relative-response factor once, or preferably establish a matched calibration. In a real laboratory, document exactly which corrections were applied before fitting the slope.",
                tone="gold",
            )
        terms = lesson.get("terms", [])
        if terms:
            _book_terms_block(terms)
        _book_block("06 · WORKED EXAMPLE", lesson.get("example", ""), tone="gold")
        _book_block("07 · ASSUMPTIONS, LIMITATIONS AND GOOD PRACTICE", (lesson.get("caution", "") + " " + assumptions).strip(), tone="blue")
        _book_block("08 · DISCUSSION POINT", lesson.get("observe", practical), tone="green")

    elif lesson and is_live:
        _label, mechanism, detailed_equations, worked, caution = lesson
        _book_block("01 · HOW THE MEASUREMENT WORKS", LIVE_EXPANDED.get(lesson_key, mechanism), tone="teal")
        _book_block("02 · EQUATIONS AND QUANTITATIVE MODEL", "These expressions describe the quantities calculated in the selected scenario. The exact corrections applied by the simulator are shown in the measurement output and should be interpreted using the stated model assumptions.", tone="purple")
        for eq in detailed_equations:
            render_math(eq)
        _book_block("03 · WORKED EXAMPLE", worked, tone="gold")
        _book_block("04 · ASSUMPTIONS AND LIMITATIONS", (caution + " " + assumptions).strip(), tone="blue")
        _book_block("05 · DISCUSSION POINT", practical, tone="green")

    else:
        _book_block("EQUATIONS AND PRACTICAL NOTES", "The following relationships are used in this practical. Check the units and confirm that each correction is applied only once.", tone="purple")
        for equation in equations:
            render_math(equation)
        if assumptions:
            _book_block("ASSUMPTIONS AND LIMITATIONS", assumptions, tone="blue")
        _book_block("DISCUSSION POINT", practical, tone="green")

    st.markdown("<div class='book-practical-divider'><span>END OF CHAPTER</span><strong>Continue to the practical workspace below</strong></div>", unsafe_allow_html=True)
    st.divider()




# ======================= INSTRUMENT GUIDE SIMULATORS ========================
GUIDE_ACQUISITION_STEPS = [
    ("Photon interaction", "A gamma photon deposits energy in the NaI(Tl) crystal. Depending on photon energy and interaction pathway, energy is transferred through photoelectric absorption and Compton scattering."),
    ("Scintillation light", "The thallium-activated sodium iodide crystal converts deposited energy into a brief flash of visible photons. The light yield is approximately proportional to deposited energy over the useful operating range."),
    ("Photocathode and PMT", "Scintillation photons reach the photomultiplier photocathode and release photoelectrons. A dynode chain multiplies the electron population; the anode collects the charge pulse."),
    ("Preamplifier", "The charge-sensitive preamplifier converts the small anode charge into a voltage pulse. The pulse amplitude carries information about the energy deposited in the scintillator."),
    ("Shaping and threshold", "The shaping amplifier filters the pulse to improve signal-to-noise and establish a reproducible pulse shape. A lower-level discriminator can reject pulses below a selected threshold."),
    ("ADC conversion", "The analogue-to-digital converter samples or digitises the shaped pulse height. The resulting digital value is assigned to a channel in the analyser."),
    ("MCA spectrum", "The multichannel analyser increments the channel corresponding to each accepted pulse. Repeating this process builds a pulse-height spectrum: counts as a function of channel or calibrated energy."),
]

GUIDE_STEP_SHORT = [
    ("NaI(Tl) well", "Energy deposition"),
    ("Scintillation", "Optical photons"),
    ("PMT", "Photoelectrons × dynode gain"),
    ("Preamp", "Charge-to-voltage pulse"),
    ("Shaper / discriminator", "Pulse conditioning and threshold"),
    ("ADC", "Pulse-height digitisation"),
    ("MCA", "Channel histogram"),
]



def make_detector_cutaway_figure(active_step: int) -> go.Figure:
    """Conceptual cross-section of a well counter detector head."""
    fig=go.Figure()
    # Lead shield body and detector well.
    fig.add_shape(type="rect",x0=.24,x1=.76,y0=.24,y1=.90,fillcolor="#435565",line=dict(color="#a9bac5",width=2))
    fig.add_annotation(x=.5,y=.865,text="LEAD SHIELDING",showarrow=False,font=dict(color="#ffffff",size=11))
    # NaI(Tl) crystal arranged around the sample well.
    crystal_fill="#17604f" if active_step in (0,1) else "#174b63"
    crystal_edge="#7cf0c7" if active_step in (0,1) else "#58b9ff"
    for x0,x1,y0,y1 in [(.32,.43,.35,.78),(.57,.68,.35,.78),(.32,.68,.25,.35)]:
        fig.add_shape(type="rect",x0=x0,x1=x1,y0=y0,y1=y1,fillcolor=crystal_fill,line=dict(color=crystal_edge,width=2))
    fig.add_annotation(x=.37,y=.55,text="NaI(Tl)\ncrystal",showarrow=False,font=dict(color="#ffffff",size=10))
    fig.add_annotation(x=.63,y=.55,text="NaI(Tl)\ncrystal",showarrow=False,font=dict(color="#ffffff",size=10))
    # Sample tube and sample liquid.
    fig.add_shape(type="rect",x0=.455,x1=.545,y0=.37,y1=.79,fillcolor="#e7eef2",line=dict(color="#ffffff",width=2))
    fig.add_shape(type="rect",x0=.465,x1=.535,y0=.43,y1=.57,fillcolor="#8d6dc4",line=dict(color="#d8caff",width=1))
    fig.add_shape(type="rect",x0=.445,x1=.555,y0=.79,y1=.83,fillcolor="#d7e0e6",line=dict(color="#ffffff",width=1))
    fig.add_annotation(x=.50,y=.63,text="Sample\ntube",showarrow=False,font=dict(color="#10212b",size=10))
    fig.add_annotation(x=.50,y=.49,text="Sample",showarrow=False,font=dict(color="#ffffff",size=9))
    # Gamma rays from sample to the crystal.
    for x0,y0,x1,y1 in [(.50,.62,.40,.66),(.50,.60,.60,.67),(.50,.52,.39,.48),(.50,.54,.61,.45)]:
        fig.add_annotation(x=x1,y=y1,ax=x0,ay=y0,axref="x",ayref="y",text="",showarrow=True,arrowhead=2,
                           arrowcolor="#ffc45b",arrowwidth=2,arrowsize=1.1)
    fig.add_annotation(x=.50,y=.96,text="γ photons deposit energy in the surrounding scintillator",showarrow=False,
                       font=dict(color="#ffc45b",size=10))
    # PMT below crystal and electrical output.
    pmt_fill="#175645" if active_step==2 else "#183247"
    fig.add_shape(type="rect",x0=.405,x1=.595,y0=.08,y1=.22,fillcolor=pmt_fill,line=dict(color="#78e9c7" if active_step==2 else "#64869b",width=2))
    fig.add_annotation(x=.50,y=.15,text="PHOTOMULTIPLIER\n(PMT)",showarrow=False,font=dict(color="#ffffff",size=10))
    fig.add_annotation(x=.70,y=.14,ax=.595,ay=.14,axref="x",ayref="y",text="",showarrow=True,arrowhead=3,arrowcolor="#55e0b0",arrowwidth=2)
    fig.add_annotation(x=.84,y=.14,text="Pulse to\nelectronics",showarrow=False,font=dict(color="#55e0b0",size=10))
    fig.add_annotation(x=.50,y=.025,text="Conceptual well-detector cross-section · exact geometry varies by instrument",showarrow=False,
                       font=dict(color="#c9d8e1",size=9))
    fig.update_xaxes(range=(0,1),visible=False); fig.update_yaxes(range=(0,1),visible=False,scaleanchor="x")
    fig.update_layout(template="plotly_dark",height=430,margin=dict(l=10,r=10,t=45,b=10),
                      title=dict(text="Inside the detector head: sample well, scintillator and shielding",font=dict(size=16,color="#ffffff")),
                      paper_bgcolor="#0d1e2b",plot_bgcolor="#0d1e2b",font=dict(color="#f4f8fb"),showlegend=False)
    return fig


def make_guide_chain_figure(active_step: int) -> go.Figure:
    """Conceptual signal path. Not a proprietary WIZARD electronics schematic."""
    fig = go.Figure()
    labels = ["NaI(Tl) well", "Scintillation\nphotons", "Photocathode\n+ dynodes", "Preamplifier", "Shaping /\ndiscriminator", "ADC", "MCA\nspectrum"]
    subtitles = ["γ energy", "light burst", "electron gain", "charge pulse", "filter + threshold", "digital channel", "counts / channel"]
    xs = np.linspace(.075, .925, len(labels))
    for i, (x, label, sub) in enumerate(zip(xs, labels, subtitles)):
        fill = "#17604f" if i == active_step else "#173348"
        line = "#7cf0c7" if i == active_step else "#5a8299"
        fig.add_shape(type="rect", x0=x-.058, x1=x+.058, y0=.43, y1=.68,
                      line=dict(color=line, width=2.5 if i == active_step else 1.5),
                      fillcolor=fill, layer="below")
        fig.add_annotation(x=x, y=.565, text=label.replace("\n", "<br>"), showarrow=False,
                           font=dict(color="#ffffff", size=10), align="center")
        fig.add_annotation(x=x, y=.32, text=sub, showarrow=False,
                           font=dict(color="#bfd2dd", size=9), align="center")
        fig.add_annotation(x=x, y=.78, text=f"{i+1:02d}", showarrow=False,
                           font=dict(color="#76e7c2" if i == active_step else "#9fb7c7", size=10))
        if i < len(xs)-1:
            fig.add_annotation(x=(x+xs[i+1])/2, y=.555, ax=x+.063, ay=.555,
                               axref="x", ayref="y", text="", showarrow=True,
                               arrowhead=3, arrowsize=1.1, arrowwidth=2,
                               arrowcolor="#ffc45b")
    fig.add_annotation(x=.5, y=.12, text="PHYSICAL CONVERSION  →  ELECTRICAL PULSE  →  DIGITAL HISTOGRAM",
                       showarrow=False, font=dict(color="#e4edf2", size=11))
    fig.update_xaxes(range=(0,1), visible=False); fig.update_yaxes(range=(0,1), visible=False)
    fig.update_layout(template="plotly_dark", height=245, margin=dict(l=10,r=10,t=40,b=18),
                      title=dict(text="Complete acquisition signal path · conceptual overview",font=dict(size=16,color="#ffffff")),
                      paper_bgcolor="#0d1e2b",plot_bgcolor="#0d1e2b",font=dict(color="#f4f8fb"),
                      showlegend=False)
    return fig


def make_circuit_schematic_figure(active_step: int) -> go.Figure:
    """Simplified educational block/circuit diagram; not the manufacturer's circuit."""
    fig = go.Figure()
    # Signal path blocks.
    blocks = [
        (0.05, .60, .16, .82, "NaI(Tl)\n+ PMT"),
        (.27, .60, .39, .82, "Charge\npreamplifier"),
        (.46, .60, .59, .82, "Shaping\namp"),
        (.66, .60, .78, .82, "Discriminator"),
        (.85, .60, .97, .82, "ADC / MCA"),
    ]
    active_for_block = [2,3,4,4,5]
    for bi, ((x0,y0,x1,y1,label), idx) in enumerate(zip(blocks, active_for_block)):
        active = idx == active_step or (active_step == 6 and idx == 5)
        fill = "#175645" if active else "#183247"
        edge = "#78e9c7" if active else "#64869b"
        if bi in (1,3):
            # Operational-amplifier / comparator triangle symbols.
            path=f"M {x0:.3f},{y0:.3f} L {x0:.3f},{y1:.3f} L {x1:.3f},{(y0+y1)/2:.3f} Z"
            fig.add_shape(type="path",path=path,fillcolor=fill,line=dict(color=edge,width=2 if active else 1.2))
            symbol="A1" if bi==1 else "CMP"
            fig.add_annotation(x=x0+.035,y=.71,text=symbol,showarrow=False,font=dict(color="#ffffff",size=9),align="center")
            fig.add_annotation(x=x0+.012,y=.755,text="−",showarrow=False,font=dict(color="#ffffff",size=9))
            fig.add_annotation(x=x0+.012,y=.665,text="+",showarrow=False,font=dict(color="#ffffff",size=9))
            fig.add_annotation(x=(x0+x1)/2,y=.55,text="Charge preamp" if bi==1 else "Threshold comparator",showarrow=False,
                               font=dict(color="#d6e4eb",size=8),align="center")
        else:
            fig.add_shape(type="rect",x0=x0,x1=x1,y0=y0,y1=y1,fillcolor=fill,
                          line=dict(color=edge,width=2 if active else 1.2))
            fig.add_annotation(x=(x0+x1)/2,y=.71,text=label.replace("\n","<br>"),showarrow=False,
                               font=dict(color="#ffffff",size=10),align="center")
    # Signal arrows between the modules.
    for x0,x1 in [(.21,.27),(.39,.46),(.59,.66),(.78,.85)]:
        fig.add_annotation(x=x1,y=.71,ax=x0,ay=.71,axref="x",ayref="y",text="",showarrow=True,
                           arrowhead=3,arrowcolor="#ffc45b",arrowwidth=2)
    # High-voltage supply symbol feeding the dynode chain (conceptual only).
    fig.add_shape(type="rect",x0=.055,x1=.135,y0=.88,y1=.97,fillcolor="#49361b",line=dict(color="#ffc45b",width=1.5))
    fig.add_annotation(x=.095,y=.925,text="+HV",showarrow=False,font=dict(color="#fff2d9",size=10))
    fig.add_shape(type="line",x0=.095,x1=.095,y0=.88,y1=.80,line=dict(color="#ffc45b",width=1.5,dash="dot"))
    fig.add_annotation(x=.16,y=.91,text="PMT bias supply",showarrow=False,font=dict(color="#ffc45b",size=9),xanchor="left")
    # PMT internal symbolic detail: photocathode and three dynodes.
    fig.add_shape(type="rect",x0=.075,x1=.135,y0=.635,y1=.785,fillcolor="#0d1f2b",line=dict(color="#91b3c4",width=1))
    fig.add_shape(type="line",x0=.088,x1=.088,y0=.65,y1=.77,line=dict(color="#55e0b0",width=3))
    for yy in [.66,.69,.72,.75]:
        fig.add_shape(type="line",x0=.105,x1=.122,y0=yy,y1=yy+.012,line=dict(color="#ffc45b",width=2))
    fig.add_annotation(x=.105,y=.615,text="photocathode → dynodes → anode",showarrow=False,font=dict(color="#cad9e1",size=8))
    # A small equivalent shaping network under the signal path.
    fig.add_shape(type="line",x0=.30,x1=.70,y0=.37,y1=.37,line=dict(color="#b7d0df",width=2))
    # Resistor zig-zag.
    zig_x=np.linspace(.37,.50,9); zig_y=[.37,.405,.335,.405,.335,.405,.335,.37,.37]
    fig.add_trace(go.Scatter(x=zig_x,y=zig_y,mode="lines",line=dict(color="#58b9ff",width=2),showlegend=False,hoverinfo="skip"))
    fig.add_annotation(x=.435,y=.435,text="R",showarrow=False,font=dict(color="#58b9ff",size=10))
    # Capacitor to ground.
    fig.add_shape(type="line",x0=.58,x1=.58,y0=.37,y1=.27,line=dict(color="#b7d0df",width=2))
    fig.add_shape(type="line",x0=.555,x1=.605,y0=.31,y1=.31,line=dict(color="#ffc45b",width=3))
    fig.add_shape(type="line",x0=.555,x1=.605,y0=.295,y1=.295,line=dict(color="#ffc45b",width=3))
    fig.add_shape(type="line",x0=.58,x1=.58,y0=.295,y1=.25,line=dict(color="#b7d0df",width=2))
    fig.add_shape(type="line",x0=.55,x1=.61,y0=.25,y1=.25,line=dict(color="#b7d0df",width=1.5))
    fig.add_shape(type="line",x0=.56,x1=.60,y0=.235,y1=.235,line=dict(color="#b7d0df",width=1.5))
    fig.add_annotation(x=.58,y=.205,text="C · shaping time constant",showarrow=False,font=dict(color="#cad9e1",size=9))
    # Comparator threshold reference.
    fig.add_shape(type="line",x0=.76,x1=.93,y0=.37,y1=.37,line=dict(color="#b7d0df",width=2))
    fig.add_shape(type="line",x0=.82,x1=.82,y0=.37,y1=.48,line=dict(color="#ffc45b",width=2))
    fig.add_annotation(x=.82,y=.50,text="LLD threshold",showarrow=False,font=dict(color="#ffc45b",size=9))
    fig.add_annotation(x=.50,y=.08,text="Simplified equivalent signal-conditioning circuit · component values and topology are illustrative, not a WIZARD service schematic",
                       showarrow=False,font=dict(color="#c6d6df",size=9))
    fig.update_xaxes(range=(0,1),visible=False); fig.update_yaxes(range=(0,1),visible=False)
    fig.update_layout(template="plotly_dark",height=340,margin=dict(l=10,r=10,t=40,b=15),
                      title=dict(text="From PMT anode pulse to a digitised spectrum",font=dict(size=16,color="#ffffff")),
                      paper_bgcolor="#0d1e2b",plot_bgcolor="#0d1e2b",font=dict(color="#f4f8fb"),showlegend=False)
    return fig



def progressive_spectrum_counts(spectrum: pd.DataFrame, progress_pct: float,
                                live_time_s: float, half_life_min: float | None,
                                seed: int = 76123, frames: int = 100) -> np.ndarray:
    """Return a monotonic cumulative spectrum at a selected acquisition fraction.

    Final channel counts are partitioned into time bins. Sample events follow the
    exponential-decay time distribution when a half-life is supplied; background
    events are uniform in time. Every progress value therefore uses the same event
    allocation, and the 100% frame exactly matches the saved final spectrum.
    """
    final_counts=np.maximum(spectrum["Counts"].to_numpy(dtype=np.int64),0)
    pct=float(np.clip(progress_pct,0.0,100.0))
    if pct <= 0:
        return np.zeros_like(final_counts)
    if pct >= 100:
        return final_counts.copy()
    n_frames=max(10,int(frames))
    edges=np.linspace(0.0,max(float(live_time_s),1e-9),n_frames+1)
    sample_expected=spectrum.get("Expected sample counts (teaching)",pd.Series(np.zeros(len(spectrum)))).to_numpy(dtype=float)
    background_expected=spectrum.get("Expected background counts (teaching)",pd.Series(np.zeros(len(spectrum)))).to_numpy(dtype=float)
    if half_life_min and half_life_min>0:
        lam=math.log(2.0)/(float(half_life_min)*60.0)
        intervals=(np.exp(-lam*edges[:-1])-np.exp(-lam*edges[1:]))/lam
        sample_prob=intervals/max(float(intervals.sum()),1e-15)
    else:
        sample_prob=np.full(n_frames,1.0/n_frames)
    background_prob=np.full(n_frames,1.0/n_frames)
    rng=np.random.default_rng(int(seed))
    cumulative=np.zeros((n_frames,len(final_counts)),dtype=np.int64)
    for ch,n in enumerate(final_counts):
        if n<=0:
            continue
        se=max(float(sample_expected[ch]),0.0)
        be=max(float(background_expected[ch]),0.0)
        total=se+be
        if total>0:
            weights=se*sample_prob+be*background_prob
            probs=weights/max(float(weights.sum()),1e-15)
        else:
            probs=background_prob
        increments=rng.multinomial(int(n),probs)
        cumulative[:,ch]=np.cumsum(increments)
    frame_idx=min(n_frames-1,int(math.ceil(pct/100.0*n_frames))-1)
    return cumulative[frame_idx]


def make_pulse_shape_figure(gain_factor: float, resolution_pct: float, threshold: float) -> go.Figure:
    t=np.linspace(0,8,500)
    pre=(1-np.exp(-t/.12))*np.exp(-t/1.15)
    pre=pre/max(pre.max(),1e-12)*gain_factor
    shaped=(t/.8)**2*np.exp(2-t/.8)
    shaped[t<0]=0
    shaped=shaped/max(shaped.max(),1e-12)*gain_factor
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=t,y=pre,mode="lines",name="PMT / preamp pulse",line=dict(color="#58b9ff",width=2.4)))
    fig.add_trace(go.Scatter(x=t,y=shaped,mode="lines",name="Shaped pulse",line=dict(color="#55e0b0",width=2.7)))
    fig.add_hline(y=threshold/100,line_dash="dash",line_color="#ffc45b",annotation_text="LLD threshold",annotation_font_color="#ffc45b")
    fig.update_layout(template="plotly_dark",height=260,margin=dict(l=10,r=12,t=35,b=20),
                      title=dict(text=f"Illustrative pulse processing · resolution setting {resolution_pct:.1f}% FWHM",font=dict(size=14,color="#ffffff")),
                      xaxis_title="Time after event (arbitrary units)",yaxis_title="Normalised pulse amplitude",
                      paper_bgcolor="#0d1e2b",plot_bgcolor="#0d1e2b",font=dict(color="#f4f8fb"),legend=dict(orientation="h",y=-.28))
    return fig


def estimate_sample_geometry_response(volume_ml: float, inner_diameter_mm: float,
                                       vertical_offset_mm: float, matrix_mu_cm: float,
                                       wall_material: str, wall_thickness_mm: float) -> dict[str, float]:
    """Transparent, deliberately simplified relative-response teaching model.

    It captures fill-height/active-zone overlap, radial self attenuation and
    positioning. It is not a validated Monte Carlo model or WIZARD calibration.
    """
    vol=max(float(volume_ml),0.01); diam=max(float(inner_diameter_mm),1.0)
    area_mm2=math.pi*(diam/2.0)**2
    height_mm=1000.0*vol/area_mm2  # 1 mL = 1000 mm^3
    active_h=50.0
    ref_height=1000.0*3.0/(math.pi*(8.0/2.0)**2)
    def axial_factor(h):
        # Smooth active-zone overlap approximation, normalised later to reference.
        return (0.78+0.22*min(h/active_h,1.0)) * min(1.0,active_h/max(h,1e-9))
    axial=axial_factor(height_mm)/axial_factor(ref_height)
    mu=max(float(matrix_mu_cm),0.0)
    path_cm=(diam/10.0)*0.5
    ref_mu=0.096
    ref_path_cm=(8.0/10.0)*0.5
    matrix_atten=math.exp(-mu*path_cm)/math.exp(-ref_mu*ref_path_cm)
    wall_mu={"Polypropylene":0.11,"Borosilicate glass":0.22,"Thin low-attenuation tube":0.07}.get(wall_material,0.11)
    wall_atten=math.exp(-wall_mu*max(float(wall_thickness_mm),0.0)/10.0)
    ref_wall=math.exp(-0.11*0.4/10.0)
    wall_factor=wall_atten/ref_wall
    position_factor=math.exp(-0.5*(float(vertical_offset_mm)/18.0)**2)
    response=float(np.clip(axial*matrix_atten*wall_factor*position_factor,0.35,1.5))
    # More material/path length redirects a modest fraction of full-energy events
    # into the continuum. This is a relative teaching adjustment, not a transport code.
    scatter=float(np.clip(0.045*max(mu*path_cm-ref_mu*ref_path_cm,0.0)/0.10
                          + 0.025*max(height_mm/ref_height-1.0,0.0),0.0,0.15))
    return {"height_mm":height_mm,"axial_factor":axial,"attenuation_factor":matrix_atten,
            "wall_factor":wall_factor,"position_factor":position_factor,
            "response_factor":response,"scatter_transfer":scatter}


def make_sample_vial_figure(volume_ml: float, diameter_mm: float, height_mm: float,
                            position_offset_mm: float) -> go.Figure:
    fig=go.Figure()
    vial_half=max(diameter_mm/2.0,2.0)
    h=max(float(height_mm),2.0)
    display_height=max(100.0,h*1.12)
    vial_base=float(position_offset_mm)
    fig.add_shape(type="rect",x0=-vial_half,x1=vial_half,y0=vial_base,y1=vial_base+display_height,
                  line=dict(color="#d9e5ed",width=2),fillcolor="rgba(180,205,220,0.04)")
    fig.add_shape(type="rect",x0=-vial_half+0.5,x1=vial_half-0.5,y0=vial_base,y1=vial_base+h,
                  line=dict(color="#55e0b0",width=1),fillcolor="rgba(85,224,176,0.32)")
    fig.add_shape(type="rect",x0=-12,x1=12,y0=0,y1=50,
                  line=dict(color="#58b9ff",width=1,dash="dash"),fillcolor="rgba(88,185,255,0.05)")
    fig.add_annotation(x=vial_half+3,y=vial_base+h/2,text=f"{volume_ml:.2f} mL",showarrow=False,
                       font=dict(color="#55e0b0",size=11),xanchor="left")
    fig.add_annotation(x=0,y=vial_base+display_height+4,text=f"Sample centre offset: {position_offset_mm:+.1f} mm",showarrow=False,
                       font=dict(color="#ffc45b",size=10))
    fig.add_annotation(x=0,y=25,text="Illustrative\nactive zone",showarrow=False,
                       font=dict(color="#58b9ff",size=9))
    fig.update_xaxes(range=(-25,35),visible=False); fig.update_yaxes(range=(min(-20.0,vial_base-4.0),max(80.0,vial_base+display_height+12)),visible=False)
    fig.update_layout(template="plotly_dark",height=330,margin=dict(l=4,r=4,t=30,b=4),
                      title="Vial fill height and detector active zone",paper_bgcolor="#0d1e2b",
                      plot_bgcolor="#0d1e2b",font=dict(color="#f4f8fb"),showlegend=False)
    return fig


def render_choice_tiles(options: list[str], state_key: str, columns: int = 4, section_label: str | None = None) -> str:
    """Render high-contrast, card-like navigation using Streamlit buttons, not st.tabs.

    This intentionally avoids Streamlit/BaseWeb's version-dependent tab DOM and
    underline indicator. The selected tile is a primary button; other tiles are
    secondary buttons. Selection persists in session state.
    """
    if not options:
        raise ValueError("Navigation requires at least one option.")
    if state_key not in st.session_state or st.session_state[state_key] not in options:
        st.session_state[state_key] = options[0]
    if section_label:
        st.markdown(f"<div class='qnl-nav-label'>{html_lib.escape(section_label)}</div>", unsafe_allow_html=True)
    with st.container(border=True):
        for start in range(0, len(options), max(1, columns)):
            row_options = options[start:start + max(1, columns)]
            cols = st.columns(len(row_options), gap="small")
            for col, option in zip(cols, row_options):
                option_index = options.index(option)
                active = st.session_state[state_key] == option
                with col:
                    if st.button(
                        option,
                        key=f"{state_key}_tile_{option_index}",
                        type="primary" if active else "secondary",
                        use_container_width=True,
                    ):
                        if not active:
                            st.session_state[state_key] = option
                            st.rerun()
    return str(st.session_state[state_key])


def render_acquisition_guide(radionuclide: str, sensitivity: float, bg_default: float, deadtime_default: float) -> None:
    st.markdown("""
    <div class="guide-banner"><div class="guide-kicker">INSTRUMENT FIELD GUIDE · INTERACTIVE WALK-THROUGH</div>
    <div class="guide-heading">How a well-type gamma counter works</div>
    <div class="guide-copy">Follow an event from gamma-ray interaction in the scintillation crystal through the photomultiplier and pulse-processing electronics to the multichannel analyser. Move through the acquisition timeline and watch the energy spectrum build as accepted pulses accumulate.</div></div>
    """,unsafe_allow_html=True)
    guide_labels=["01 · ACQUISITION MECHANISM","02 · FAULT-FINDING BENCH","03 · RACK & NORMALISATION PREPARATION","04 · SAMPLE GEOMETRY & PREPARATION"]
    guide_nav=render_choice_tiles(guide_labels,"qnl_guide_navigation",columns=4,section_label="GAMMA COUNTER GUIDE")

    # ------------------------ Acquisition mechanism -------------------------
    if guide_nav == guide_labels[0]:
        st.markdown("### Follow one gamma event through the counter")
        st.write("Select a component to highlight it in the detector schematic. The spectrum beside it also builds progressively, linking the selected stage to the pulse-height distribution. In the real instrument, the MCA accumulates accepted pulses; the staged view is a teaching visualisation of that event-processing chain.")

        # The conceptual preview cards are the stage selector. Each card is a real
        # Streamlit button, so no separate stage slider is needed.
        selected_idx = int(st.session_state.get("guide_acq_stage_index", 0))
        selected_idx = max(0, min(selected_idx, len(GUIDE_ACQUISITION_STEPS) - 1))
        stage_layout = [range(0, 4), range(4, len(GUIDE_ACQUISITION_STEPS))]
        for row_indices in stage_layout:
            cols = st.columns(4)
            for col, idx in zip(cols, row_indices):
                with col:
                    with st.container(border=True):
                        is_selected = idx == selected_idx
                        label = f"{idx + 1:02d} · {GUIDE_STEP_SHORT[idx][0]}"
                        if st.button(
                            label,
                            key=f"guide_stage_tile_333_{idx}",
                            type="primary" if is_selected else "secondary",
                            use_container_width=True,
                        ):
                            st.session_state["guide_acq_stage_index"] = idx
                            st.rerun()
                        st.caption(GUIDE_STEP_SHORT[idx][1])

        active = int(st.session_state.get("guide_acq_stage_index", 0))
        active = max(0, min(active, len(GUIDE_ACQUISITION_STEPS) - 1))
        stage_name, stage_summary = GUIDE_ACQUISITION_STEPS[active]
        _book_block(
            f"STAGE {active + 1:02d} · {stage_name.upper()}",
            stage_summary,
            tone="teal",
        )

        component_details = [
            "A gamma photon interacts in the NaI(Tl) crystal and transfers energy to electrons in the material. Photoelectric absorption can deposit most or all of the photon energy, whereas Compton scattering may deposit only part of it. Photons that scatter out of the detector contribute to the continuum rather than the full-energy peak.",
            "The deposited energy excites the scintillator, which emits a brief flash of visible light. Within the useful operating range, the total light output is approximately proportional to the energy deposited. Variations in light production and collection contribute to the energy resolution of the detector.",
            "Light reaching the photocathode releases photoelectrons. These electrons are multiplied through the dynode stages and collected at the anode as a measurable charge pulse. The photomultiplier gain controls the size of the electrical pulse; changing gain can move peaks in channel space if the energy calibration is not updated.",
            "The preamplifier converts the small charge from the PMT into a voltage pulse suitable for further processing. The pulse amplitude retains information about the energy deposited in the scintillator. Poor connections, excess noise or saturation can affect pulse stability and measurement quality.",
            "Pulse shaping filters the signal and establishes a consistent pulse form. A lower-level discriminator rejects pulses below a selected threshold. If the threshold is too high, genuine low-amplitude events may be removed; if it is too low, electronic noise may be accepted as counts.",
            "The analogue-to-digital converter maps the shaped pulse height to a digital value. This value is assigned to a channel. The relationship between channel and energy depends on the energy calibration; changes in gain or offset can therefore shift the displayed peak energy.",
            "The multichannel analyser increments the channel associated with each accepted pulse. Repeated events build a pulse-height histogram. Once an energy calibration is applied, the horizontal axis can be displayed in keV. Peak position, peak width, ROI boundaries and continuum subtraction then become important for quantitative analysis.",
        ]

        left, right = st.columns([1.0, 1.0], gap="large")
        with left:
            st.plotly_chart(
                make_detector_cutaway_figure(active),
                use_container_width=True,
                key=f"guide_detector_cutaway_333_{active}",
            )
        with right:
            _book_block("WHAT HAPPENS AT THIS STAGE", component_details[active], tone="blue")
            _book_block(
                "WHY IT MATTERS FOR MEASUREMENT",
                [
                    "The interaction pathway influences whether the event contributes to the full-energy peak or the continuum. This is why energy-window selection and detector geometry matter.",
                    "Scintillation yield and optical collection influence the number of photoelectrons and therefore the statistical spread in pulse height.",
                    "Photomultiplier gain affects pulse amplitude. A gain change can shift the apparent peak and requires the energy scale to be checked.",
                    "The preamplifier must preserve the charge information while adding as little noise as practical. Instability here can affect both peak position and width.",
                    "Threshold selection affects which pulses are accepted. A threshold problem can reduce counts in selected energy regions and alter the spectrum.",
                    "The ADC calibration determines how pulse amplitude maps to channel number. An incorrect slope or offset gives an incorrect energy axis.",
                    "The MCA forms the final histogram. The spectrum is the accumulated record of accepted pulses and is the basis for peak and ROI analysis.",
                ][active],
                tone="gold",
            )

        # The selected component also controls how far the illustrative spectrum
        # has accumulated. The real MCA accumulates the pulses; this staged view
        # links the event-processing chain to the appearance of the histogram.
        guide_spectrum = simulate_spectrum(
            radionuclide, 25000.0, 60.0, sensitivity, float(bg_default), 9.5,
            1.0, 0.0, float(deadtime_default), seed=44021,
            half_life_min=float(RADIONUCLIDES[radionuclide]["half_life_min"]),
        )
        stage_progress = [10, 22, 35, 49, 64, 82, 100][active]
        stage_counts = progressive_spectrum_counts(
            guide_spectrum, stage_progress, 60.0,
            float(RADIONUCLIDES[radionuclide]["half_life_min"]),
            seed=44021, frames=100,
        )
        energy = guide_spectrum["Energy (keV)"].to_numpy(dtype=float)
        expected_final = guide_spectrum["Expected counts (teaching)"].to_numpy(dtype=float)
        spec_col, info_col = st.columns([1.7, 0.8], gap="large")
        with spec_col:
            spectrum_fig = go.Figure()
            spectrum_fig.add_trace(go.Scatter(
                x=energy, y=expected_final, mode="lines", name="Expected final spectrum",
                line=dict(color="#8BA0AF", width=1.2, dash="dot"), opacity=0.55,
            ))
            spectrum_fig.add_trace(go.Scatter(
                x=energy, y=stage_counts, mode="lines", name="Accumulated events at selected stage",
                line=dict(color="#55E0B0", width=1.9), fill="tozeroy",
                fillcolor="rgba(85,224,176,0.10)",
            ))
            line_specs = SPECTRUM_LINES.get(radionuclide, [])
            xray_energies=[float(e) for e,_w,k in line_specs if k=="characteristic_xray"]
            for line_energy, _weight, line_kind in line_specs:
                if energy.min() <= float(line_energy) <= energy.max():
                    if line_kind == "characteristic_xray":
                        spectrum_fig.add_vline(x=float(line_energy), line_dash="dot", line_color="#C6A5FF", opacity=.75)
                        continue
                    label = {"annihilation":"annihilation", "gamma":"gamma", "sum_peak":"coincidence sum"}.get(line_kind, line_kind)
                    spectrum_fig.add_vline(x=float(line_energy), line_dash="dot", line_color="#FFC45B", opacity=.65)
                    spectrum_fig.add_annotation(x=float(line_energy), y=1.02, yref="paper", text=f"{float(line_energy):g} keV · {label}",
                        showarrow=False, textangle=-90, font=dict(color="#FFC45B", size=9), xanchor="left")
            if xray_energies:
                spectrum_fig.add_annotation(x=float(np.mean(xray_energies)),y=.90,yref="paper",
                    text="Y K X-ray cluster · 14.9–16.7 keV",showarrow=True,arrowhead=2,arrowcolor="#C6A5FF",
                    font=dict(color="#DCCBFF",size=9),ax=45,ay=-32)
            # Mark Compton edge/backscatter for the two most important gamma lines.
            major = [x for x in line_specs if x[2] in ("annihilation", "gamma")]
            for line_energy, _weight, _kind in sorted(major, key=lambda z:z[1], reverse=True)[:2]:
                e0=float(line_energy)
                ce=e0*(1.0-1.0/(1.0+2.0*e0/511.0))
                bs=e0/(1.0+2.0*e0/511.0)
                spectrum_fig.add_vline(x=ce, line_dash="dash", line_color="#58B9FF", opacity=.45)
                spectrum_fig.add_vline(x=bs, line_dash="dash", line_color="#FF9B61", opacity=.45)
                spectrum_fig.add_annotation(x=ce,y=.72,yref="paper",text=f"Compton edge · {ce:.0f} keV",
                    showarrow=False,font=dict(color="#8BCBFF",size=8),textangle=-90)
                spectrum_fig.add_annotation(x=bs,y=.72,yref="paper",text=f"Backscatter · {bs:.0f} keV",
                    showarrow=False,font=dict(color="#FFB18A",size=8),textangle=-90)
            spectrum_fig.update_layout(template="plotly_dark", height=420, margin=dict(l=12,r=12,t=75,b=35),
                title=f"{radionuclide} spectrum formation · {stage_progress}% illustrative accumulation",
                xaxis_title="Energy (keV)", yaxis_title="Counts per channel",
                paper_bgcolor="#0d1e2b", plot_bgcolor="#0d1e2b", font=dict(color="#f4f8fb"),
                legend=dict(orientation="h", y=-.22))
            st.plotly_chart(spectrum_fig, use_container_width=True, key=f"guide_acquisition_progress_spectrum_333_{active}")
        with info_col:
            st.metric("Illustrative accumulation", f"{stage_progress}%")
            st.metric("Events displayed", f"{int(stage_counts.sum()):,}")
            st.metric("Final expected events", f"{int(round(expected_final.sum())):,}")
            st.markdown("**How to read the spectrum**")
            st.markdown("""- **Photopeak:** full-energy deposition.
- **Compton continuum / edge:** partial energy transfer.
- **Backscatter feature:** photons scattered in surrounding material before detection.
- **Coincidence sum:** two photons recorded together as one higher-energy event.
- **Characteristic X-rays:** atomic relaxation following electron capture or fluorescence in surrounding material.""")
        if radionuclide == "18F":
            _book_block("INTERPRETING THE SMALL LOW-ENERGY FEATURE", "The small feature near 170 keV in this model is the 511-keV backscatter contribution (E' = E/[1 + 2E/511]). It is not a lead X-ray. Lead K fluorescence, if generated by interactions in the shielding, occurs around 75–85 keV and is treated separately as an optional environmental/background component in the geometry experiment. The 1022-keV coincidence-summing peak is produced when the two 511-keV annihilation photons are recorded together in the well detector; it may lie outside a configured 15–1000 keV analysis window.", tone="gold")
        elif radionuclide == "89Zr":
            _book_block("INTERPRETING THE ⁸⁹Zr SPECTRUM", "The main gamma photopeak is at about 909 keV. The smaller 511-keV feature is associated with the positron branch. The low-energy 14–17 keV features are characteristic yttrium K X-rays following electron capture; the 511- and 909-keV photons also produce Compton continua and backscatter features near 170 and 204 keV. A small 1022-keV coincidence-summing feature can arise from the two annihilation photons being detected together. The 909-keV transition is delayed through the ⁸⁹mY state (about 16 s), so it should not be treated as prompt 909+511 coincidence summing. The 1022-keV feature may fall outside a configured 15–1000 keV counting window, so its visibility depends on the actual instrument/protocol energy range.", tone="gold")
        else:
            _book_block("SPECTRUM MODEL NOTE", "The displayed lines use evaluated principal emissions where available, but the line weights, continuum partition and detector response remain a teaching approximation. Actual peak areas depend on energy-dependent efficiency, threshold, geometry, matrix, energy window and electronics. Do not use these spectra as instrument calibration data.", tone="blue")
        with st.expander("Why these spectral features appear · references and model assumptions", expanded=False):
            st.markdown("""
**¹⁸F** — The dominant full-energy feature is the 511 keV annihilation peak. In a well geometry, both annihilation photons can be detected within the resolving time and create a coincidence-sum feature at 1022 keV. A backscatter feature from 511 keV photons is expected near 170 keV. The small feature near 170 keV in the base simulator is therefore modelled as backscatter, not lead fluorescence. Lead K fluorescence is a separate environmental contribution near 75 and 85 keV.

**⁸⁹Zr** — The principal nuclear gamma ray is 909.14 keV; the positron branch also produces 511 keV annihilation photons. Electron capture leaves atomic vacancies in the daughter yttrium, producing K X-rays near 14.9 and 16.7 keV. The 511 and 909 keV photons have backscatter features near 170 and 204 keV. A 1022 keV feature can arise when both annihilation photons are detected together. The 909 keV transition follows the metastable ⁸⁹ᵐY state (about 16 s), so do not interpret it as prompt 909 + 511 keV coincidence summing.

**Lead shielding** — When photons interact in lead, characteristic K X-rays can be produced near 75–85 keV. The geometry experiment includes an optional control to add these as a separate background component; they are not intrinsic emissions of ¹⁸F or ⁸⁹Zr.

**Model limits** — The line energies are isotope-informed, but relative line weights, continuum partition, resolution and coincidence-sum amplitude are pedagogical approximations. The model is useful for recognising spectral structures and parameter effects, not for predicting an individual counter's measured spectrum. The WIZARD² 2470 published energy range is specified as 15–1000 keV; a 1022 keV sum feature may therefore lie beyond a particular configured counting window.

References: [Lodge et al., EJNMMI Physics (2015), ¹⁸F spectra and 1022 keV coincidence-summing peak](https://pmc.ncbi.nlm.nih.gov/articles/PMC4452125/); [Silosky, JACMP (2026), NaI(Tl) spectra and count-rate response for ⁸⁹Zr](https://doi.org/10.1002/acm2.70788); [INL ⁸⁹Zr gamma-ray catalogue](https://gammaray.inl.gov/SiteAssets/catalogs/ge/pdf/zr89.pdf); [NIST X-ray transition energies for yttrium](https://physics.nist.gov/cgi-bin/XrayTrans/search.pl?element=Y&lower=&units=eV&upper=); [WIZARD² 2470 specifications](https://resources.revvity.com/pdfs/84900%20SPC%20RUO%20Wizard_V5.pdf).
            """)

    # --------------------------- Fault-finding -------------------------------
    elif guide_nav == guide_labels[1]:
        st.markdown("### Investigate why a spectrum or assay result is wrong")
        _book_block("FAULT-FINDING APPROACH","Change one setting at a time. Compare the reference spectrum with the disturbed spectrum, then decide whether the fault changes peak position, peak width, ROI recovery, background or total count rate. In a real instrument, changes to high voltage, gain, thresholds or service settings must follow the authorised instrument procedure.",tone="gold")
        fleft,fright=st.columns([.95,1.05])
        with fleft:
            hv_factor=st.slider("PMT high-voltage / pulse-gain factor (%)",70,130,100,1,key="fault_hv_factor")/100.0
            adc_factor=st.slider("ADC energy-scale gain (%)",80,120,100,1,key="fault_adc_gain")/100.0
            energy_offset=st.slider("Energy offset (keV)",-60.0,60.0,0.0,1.0,key="fault_energy_offset")
            fault_resolution=st.slider("Intrinsic detector resolution (% FWHM at 511 keV)",4.0,24.0,9.5,.5,key="fault_resolution")
            electronic_noise=st.slider("Electronic noise contribution (% FWHM)",0.0,12.0,0.0,.5,key="fault_electronic_noise")
            threshold_keV=st.slider("Lower-level discriminator threshold (keV)",0.0,250.0,15.0,5.0,key="fault_threshold")
            fault_bg=st.slider("Background rate (cps)",0,2500,16,5,key="fault_background")
            fault_tau=st.slider("Non-paralysable dead-time (µs)",0.0,25.0,2.5,.5,key="fault_deadtime")
            fault_activity=st.number_input("Source activity (Bq)",100.0,1e8,150000.0,1000.0,key="fault_activity")
            fault_count_time=st.slider("Acquisition time (s)",5,300,60,5,key="fault_count_time")
        base_keV_per_channel=1.0
        ideal=simulate_spectrum(radionuclide,fault_activity,fault_count_time,sensitivity,16.0,9.5,base_keV_per_channel,0.0,2.5,seed=99271,
            half_life_min=float(RADIONUCLIDES[radionuclide]["half_life_min"]))
        # PMT gain changes pulse height/channel. ADC gain/offset changes the channel-to-energy mapping.
        effective_resolution=math.sqrt(float(fault_resolution)**2+float(electronic_noise)**2)
        disturbed=simulate_spectrum(radionuclide,fault_activity,fault_count_time,sensitivity,float(fault_bg),effective_resolution,
            base_keV_per_channel/max(hv_factor,.05),0.0,float(fault_tau),seed=99271,
            half_life_min=float(RADIONUCLIDES[radionuclide]["half_life_min"]))
        ideal_energy=ideal["Channel"].to_numpy(dtype=float)*base_keV_per_channel
        disturbed_energy=energy_offset+disturbed["Channel"].to_numpy(dtype=float)*base_keV_per_channel*adc_factor
        ideal_y=ideal["Counts"].to_numpy(dtype=float)
        disturbed_y=disturbed["Counts"].to_numpy(dtype=float).copy()
        disturbed_y[disturbed_energy < threshold_keV]=0.0
        with fright:
            fault_fig=go.Figure()
            fault_fig.add_trace(go.Scatter(x=ideal_energy,y=ideal_y,mode="lines",name="Reference / calibrated spectrum",line=dict(color="#58b9ff",width=1.5)))
            fault_fig.add_trace(go.Scatter(x=disturbed_energy,y=disturbed_y,mode="lines",name="Disturbed spectrum",line=dict(color="#ff9b61",width=1.8)))
            fault_fig.update_layout(template="plotly_dark",height=460,margin=dict(l=10,r=10,t=48,b=25),title="Reference versus disturbed spectrum",
                xaxis_title="Displayed energy using current calibration (keV)",yaxis_title="Counts / channel",paper_bgcolor="#0d1e2b",plot_bgcolor="#0d1e2b",font=dict(color="#f4f8fb"),legend=dict(orientation="h",y=-.25))
            st.plotly_chart(fault_fig,use_container_width=True,key="fault_spectrum_overlay_33")
        lines=SPECTRUM_LINES.get(radionuclide,[(float(RADIONUCLIDES[radionuclide]["energy_keV"]),1.0,"gamma")])
        main_line=float(max(lines,key=lambda x:x[1])[0])
        expected_centroid=main_line*hv_factor*adc_factor+energy_offset
        default_fwhm=9.5/100*511*math.sqrt(max(main_line,1)/511)
        disturbed_fwhm=effective_resolution/100*511*math.sqrt(max(main_line,1)/511)*hv_factor*adc_factor
        roi_half=max(2.5*default_fwhm,12.0)
        ref_roi=float(ideal_y[(ideal_energy>=main_line-roi_half)&(ideal_energy<=main_line+roi_half)].sum())
        fault_roi=float(disturbed_y[(disturbed_energy>=main_line-roi_half)&(disturbed_energy<=main_line+roi_half)].sum())
        roi_recovery=100.0*fault_roi/ref_roi if ref_roi>0 else 0.0
        true_rate=fault_activity*sensitivity+fault_bg
        dead_loss=100.0*(true_rate*fault_tau*1e-6)/(1.0+true_rate*fault_tau*1e-6)
        mm=st.columns(4)
        mm[0].metric("Expected displayed centroid",f"{expected_centroid:.1f} keV",delta=f"{expected_centroid-main_line:+.1f} keV vs reference")
        mm[1].metric("Displayed FWHM",f"{disturbed_fwhm:.1f} keV")
        mm[2].metric("Fixed-ROI recovery",f"{roi_recovery:.1f}%")
        mm[3].metric("Modelled dead-time loss",f"{dead_loss:.2f}%")
        fault_notes=[]
        if abs(hv_factor-1)>0.015: fault_notes.append("PMT pulse gain is displaced from its calibrated setting: the peak moves in channel space if the MCA energy mapping is not recalibrated.")
        if abs(adc_factor-1)>0.015 or abs(energy_offset)>1: fault_notes.append("The energy calibration slope/offset has shifted: measured peak energies will no longer match reference energies.")
        if effective_resolution>11: fault_notes.append("Broader intrinsic resolution and/or electronic noise spread events across more channels and can reduce peak height or fixed-window recovery even when the total peak area is similar.")
        if threshold_keV>50: fault_notes.append("The discriminator threshold is high enough to remove lower-energy events and alter the continuum/low-energy spectrum.")
        if fault_bg>100: fault_notes.append("Elevated background increases the continuum and uncertainty of net peak estimation.")
        if dead_loss>5: fault_notes.append("The selected rate/dead-time combination produces more than 5% modelled count loss; verify the usable rate range and instrument-specific correction.")
        if not fault_notes: fault_notes.append("The selected settings are close to the reference configuration. Increase one fault factor at a time to observe its characteristic signature.")
        _book_list_block("INTERPRETATION OF THE CURRENT SETTINGS",fault_notes,tone="gold")
        st.caption("Teaching model: PMT gain, ADC calibration and discriminator threshold are separated conceptually. A high-voltage factor is a relative pulse-gain proxy, not a recommended voltage setting. The plot is not a WIZARD service calibration tool.")

    # -------------------------- Rack preparation -----------------------------
    elif guide_nav == guide_labels[2]:
        st.markdown("### Prepare a routine assay rack or a normalisation rack")
        st.write("The label arrangement depends on the instrument generation. Select the model profile before following the workflow; the legacy 1470 ID-clip system must not be assumed to apply to the WIZARD² 2470.")
        model=st.radio("Instrument model profile",["WIZARD² 2470 · rack barcode / normalisation cassette","Wallac 1470 WIZARD · legacy removable ID clip"],horizontal=False,key="rack_model_profile")
        purpose=st.selectbox("Rack purpose",["Routine assay","Isotope normalisation","Background normalisation","GLP / performance test"],key="rack_purpose")
        if "1470" in model:
            rack_id_number=st.number_input("Optional rack number barcode (1–99)",1,99,2,1,key="guide_rack_id_1470")
            protocol_number=st.number_input("Protocol number / isotope code field (1–99)",1,99,18,1,key="guide_protocol_1470")
            isotope_code=st.text_input("For NORM/TEST: enter the isotope code configured in this instrument",str(int(protocol_number)),key="guide_isotope_code_1470")
            rack_id=f"{int(rack_id_number):02d}"
            protocol_id=f"{int(protocol_number):02d}"
        else:
            rack_id=st.text_input("Rack barcode / identifier","RACK-01",key="guide_rack_id_2470")
            protocol_number=st.number_input("Protocol barcode value (1–999)",1,999,18,1,key="guide_protocol_2470")
            barcode_format=st.selectbox("Rack barcode symbology",["Code 128","Interleaved 2 of 5","Code 39","Codabar"],key="guide_barcode_format")
            sample_barcode_option=st.checkbox("Show optional sample-vial barcode option (if fitted)",False,key="guide_sample_barcode_option")
            protocol_id=str(int(protocol_number))
            isotope_code="Defined in normalisation cassette / isotope setup"
        rack_positions=10
        rack_fig=go.Figure()
        for i in range(1,rack_positions+1):
            x=(i-1)%5; y=1-(i-1)//5
            is_source=(purpose in ["Isotope normalisation","GLP / performance test"] and i==10 and "1470" in model)
            is_blank=(purpose=="Background normalisation" or (purpose in ["Isotope normalisation","GLP / performance test"] and i<10 and "1470" in model))
            fill="#8b5b22" if is_source else ("#263b4a" if is_blank else "#17604f")
            line="#ffc45b" if is_source else ("#627b8c" if is_blank else "#55e0b0")
            rack_fig.add_trace(go.Scatter(x=[x],y=[y],mode="markers+text",text=["SOURCE" if is_source else ("EMPTY" if is_blank else f"{i:02d}")],
                textposition="middle center",marker=dict(size=58,color=fill,line=dict(color=line,width=2)),
                textfont=dict(color="#ffffff",size=9),hovertemplate=f"Position {i}<extra></extra>",showlegend=False))
        rack_fig.update_xaxes(range=(-.7,4.7),visible=False); rack_fig.update_yaxes(range=(-.6,1.6),visible=False,scaleanchor="x")
        rack_fig.update_layout(template="plotly_dark",height=245,margin=dict(l=5,r=5,t=35,b=5),title="Illustrative 10-position rack map",
            paper_bgcolor="#0d1e2b",plot_bgcolor="#0d1e2b",font=dict(color="#f4f8fb"),showlegend=False)
        left,right=st.columns([1,1.15])
        with left:
            st.plotly_chart(rack_fig,use_container_width=True,key="rack_prep_map_33")
        with right:
            if "1470" in model:
                special_label = {"Routine assay": f"RACK {rack_id}", "Isotope normalisation": "NORM", "Background normalisation": "BKG", "GLP / performance test": "TEST"}[purpose]
                protocol_label = isotope_code if purpose in ["Isotope normalisation", "GLP / performance test"] else protocol_id
                st.markdown(
                    f"<div class='guide-card'><h4>Legacy 1470 removable ID clip</h4><div style='display:grid;grid-template-columns:1fr 1fr;gap:10px'>"
                    f"<div style='background:#f7f7f2;color:#10212b;border:2px solid #d7d8cf;border-radius:8px;padding:10px;text-align:center'><div style='font-size:10px;font-weight:800;letter-spacing:1px'>RACK / SPECIAL</div><div style='font-size:22px;font-weight:900;color:#10212b'>{html_lib.escape(special_label)}</div><div style='height:28px;margin-top:6px;background:repeating-linear-gradient(90deg,#111 0 2px,#fff 2px 4px,#111 4px 5px,#fff 5px 8px)'></div></div>"
                    f"<div style='background:#f7f7f2;color:#10212b;border:2px solid #d7d8cf;border-radius:8px;padding:10px;text-align:center'><div style='font-size:10px;font-weight:800;letter-spacing:1px'>PROTOCOL</div><div style='font-size:22px;font-weight:900;color:#10212b'>{html_lib.escape(str(protocol_label))}</div><div style='height:28px;margin-top:6px;background:repeating-linear-gradient(90deg,#111 0 2px,#fff 2px 4px,#111 4px 5px,#fff 5px 8px)'></div></div>"
                    f"</div><p style='margin-top:8px'>Illustrative two-field ID clip; labels shown are training examples, not print-ready manufacturer barcode artwork.</p></div>",unsafe_allow_html=True)
            else:
                if purpose == "Routine assay":
                    barcode_text = f"PROTOCOL {protocol_id} · RACK {html_lib.escape(str(rack_id))} · {barcode_format}"
                else:
                    barcode_text = "RACK ID / protocol identification · normalisation uses the isotope-specific cassette"
                st.markdown(
                    f"<div class='guide-card'><h4>WIZARD² 2470 rack identity</h4><div style='background:#f7f7f2;color:#10212b;border:2px solid #d7d8cf;border-radius:8px;padding:12px;text-align:center'><div style='font-size:12px;font-weight:900;color:#10212b'>{html_lib.escape(barcode_text)}</div><div style='height:34px;margin:8px auto 0 auto;max-width:300px;background:repeating-linear-gradient(90deg,#111 0 2px,#fff 2px 4px,#111 4px 5px,#fff 5px 8px)'></div></div><p style='margin-top:8px'>Schematic label only; use labels and barcode format approved for the installed instrument and workflow.</p></div>",unsafe_allow_html=True)
            if "1470" in model:
                if purpose=="Routine assay":
                    _book_block("LEGACY 1470 · ROUTINE RACK","Attach a numerical protocol barcode in the PROTOCOL field of the removable ID clip. An optional numerical rack number can be placed in RACK/SPECIAL. The manual describes protocol numbers 1–99 and optional rack numbers 1–99. The protocol defines the counting conditions; the rack code identifies the rack.",tone="teal")
                    _book_block("CHECK BEFORE LOADING","Fit the clip into its designated recess. Load the rack in the correct orientation: the manual specifies the ID labels face away from the operator and the white dot is on the right. Keep the same orientation for unlabelled racks.",tone="blue")
                elif purpose=="Isotope normalisation":
                    _book_block("LEGACY 1470 · NORM LABEL ARRANGEMENT","Place the NORM instruction barcode in RACK/SPECIAL. Place the numerical isotope code in PROTOCOL. The source is placed in position 10 only; positions 1–9 are empty and their holders are removed. Use the isotope code configured for that instrument/software library.",tone="gold")
                    _book_alert="Do not treat this as a generic 2470 sticker layout. The position-10 source and separate NORM / isotope-code clip arrangement are specific to the documented legacy 1470 workflow. Confirm the current local SOP and isotope source requirements before using a real instrument."
                    st.markdown(f"<div class='guide-alert'>{html_lib.escape(_book_alert)}</div>",unsafe_allow_html=True)
                elif purpose=="Background normalisation":
                    _book_block("LEGACY 1470 · BKG LABEL ARRANGEMENT","Place the BKG instruction barcode in RACK/SPECIAL. The background-normalisation rack is empty: no sample tubes should be present. Run it according to the instrument's background-normalisation procedure.",tone="gold")
                else:
                    _book_block("LEGACY 1470 · TEST / GLP LABEL ARRANGEMENT","Place TEST in RACK/SPECIAL and the appropriate isotope code in PROTOCOL. The isotope source is in the final position (position 10); positions 1–9 are empty and have no holders. This is a GLP/performance-test normalisation workflow, distinct from routine sample counting.",tone="gold")
                _book_block("AFTER A SPECIAL RACK","Remove NORM, BKG and TEST racks after their operation so they are not inadvertently processed again on a later conveyor run.",tone="blue")
            else:
                if purpose=="Routine assay":
                    _book_block("WIZARD² 2470 · ROUTINE RACK",f"The published 2470 specification describes 10-position plastic sample racks for 13 mm tubes. Rack barcodes identify protocol and rack number. Protocol barcode values are described as 1–999; supported barcode formats include Code 128, Interleaved 2 of 5, Code 39 and Codabar. Selected training barcode format: {barcode_format}.",tone="teal")
                    if sample_barcode_option:
                        _book_block("OPTIONAL SAMPLE-VIAL BARCODE","The manufacturer lists a sample-vial barcode option. Its use depends on the option being installed and the local validated workflow; it is not a substitute for the rack/protocol identification.",tone="blue")
                elif purpose=="Isotope normalisation":
                    _book_block("WIZARD² 2470 · ISOTOPE NORMALISATION","The 2470 specification describes automatic normalisation using a normalisation cassette for each defined nuclide. Do not copy the legacy 1470 NORM clip / position-10 procedure into a 2470 workflow unless the applicable 2470 manual and local SOP explicitly require it.",tone="gold")
                elif purpose=="Background normalisation":
                    _book_block("WIZARD² 2470 · BACKGROUND","Use the model-specific background / normalisation workflow and the appropriate cassette or supplied accessories as defined in the current operating procedure. The public specification confirms automatic normalisation by nuclide-specific cassette but does not document the detailed operator sequence.",tone="blue")
                else:
                    _book_block("WIZARD² 2470 · PERFORMANCE / QC","Use the performance/QC functions and cassette specified for the installed software and model configuration. Do not infer the 1470 TEST barcode arrangement for a 2470 system.",tone="blue")
                _book_block("RACK AND SAMPLE IDENTITY","Keep rack ID, protocol ID, sample ID, vial position, radionuclide, count protocol and reference-time metadata linked. The rack barcode does not replace sample-level traceability in your laboratory worksheet or LIMS.",tone="teal")
        with st.expander("Model-specific documentation and source notes",expanded=False):
            st.markdown("""
**WIZARD² 2470:** Revvity specification sheet (10-position racks, rack/protocol barcode identification, supported symbologies, protocol code range and isotope-specific normalisation cassette): [2470 WIZARD² specifications](https://resources.revvity.com/pdfs/84900%20SPC%20RUO%20Wizard_V5.pdf).

**Wallac 1470 WIZARD:** Instrument manual sections 2.5–2.10 and 6.1 describe ID clips, PROTOCOL / RACK-SPECIAL fields, NORM/BKG/TEST instruction labels and the position-10 isotope source arrangement: [1470 WIZARD instrument manual](https://extranetcemtl.cemtl.rtss.qc.ca/fileadmin/intranet/enseignement-recherche/Recherche/Manuels_operations/Equipements_Marcel-Lamoureux/Perkin-1470wizard-EN.pdf).

The app reproduces these workflows as a teaching aid. Always follow the exact model, software version, isotope library and local operating procedure installed in your facility.
            """)


    # ----------------------- Sample geometry / preparation -------------------
    elif guide_nav == guide_labels[3]:
        st.markdown("### Explore how sample preparation changes the measurement")
        st.write("Keep the radionuclide and detector configuration fixed, then change the way the sample is presented. The spectrum and expected count rate update automatically whenever a control changes; no separate acquisition button is needed for this comparative demonstration.")
        gleft, gright = st.columns([0.92, 1.08], gap="large")
        with gleft:
            volume_ml=st.slider("Sample volume (mL)",0.2,6.0,3.0,0.1,key="geom_volume_333")
            diameter_mm=st.slider("Tube internal diameter (mm)",5.0,14.0,8.0,0.5,key="geom_diameter_333")
            pos_offset=st.slider("Vertical positioning offset from reference (mm)",-15.0,15.0,0.0,0.5,key="geom_position_333")
            matrix_name=st.selectbox("Sample matrix",["Water / buffer","Saline","Serum-like","Tissue homogenate","Dense / high-solute suspension"],key="geom_matrix_333")
            matrix_mu_nominal={"Water / buffer":0.096,"Saline":0.098,"Serum-like":0.100,"Tissue homogenate":0.108,"Dense / high-solute suspension":0.140}[matrix_name]
            matrix_mu=st.slider("Effective matrix attenuation coefficient μ (cm⁻¹)",0.04,0.20,float(matrix_mu_nominal),0.002,key="geom_mu_333")
            st.caption("Teaching parameter only: an effective attenuation coefficient for this simplified model, not an energy-specific material lookup. Actual attenuation depends on photon energy, composition and density; 511-keV water is used only as the nominal reference point.")
            wall_material=st.selectbox("Tube material",["Polypropylene","Borosilicate glass","Thin low-attenuation tube"],key="geom_wall_333")
            wall_thickness=st.slider("Wall thickness (mm)",0.1,1.5,0.4,0.05,key="geom_wall_thickness_333")
            activity_mode=st.radio("Activity definition",["Fixed total activity","Fixed activity concentration"],key="geom_activity_mode_333")
            if activity_mode=="Fixed total activity":
                total_activity=st.number_input("Total sample activity (Bq)",100.0,1e7,25000.0,500.0,key="geom_total_activity_333")
                active_bq=total_activity
            else:
                concentration=st.number_input("Activity concentration (Bq/mL)",10.0,1e7,8333.0,100.0,key="geom_activity_conc_333")
                active_bq=concentration*volume_ml
            if activity_mode=="Fixed total activity":
                st.caption("Fixed total activity: increasing volume changes fill height and concentration, but does not change the total Bq placed in the vial.")
            else:
                st.caption("Fixed activity concentration: increasing volume also increases total activity, so the raw count rate reflects both sample presentation and more activity.")
            geom_bg=st.number_input("Background rate (cps)",0.0,500.0,16.0,1.0,key="geom_bg_333")
            geom_time=st.slider("Live time (s)",5,300,60,5,key="geom_time_333")
            lead_xray=st.slider("Optional lead-shield fluorescence (cps)",0.0,100.0,0.0,0.5,key="geom_lead_xray_333")
        geo=estimate_sample_geometry_response(volume_ml,diameter_mm,pos_offset,matrix_mu,wall_material,wall_thickness)
        with gright:
            st.plotly_chart(make_sample_vial_figure(volume_ml,diameter_mm,geo["height_mm"],pos_offset),
                            use_container_width=True,key="sample_vial_geometry_333")
            gm1,gm2,gm3=st.columns(3)
            gm1.metric("Calculated fill height",f"{geo['height_mm']:.1f} mm")
            gm2.metric("Relative response",f"{geo['response_factor']:.3f}")
            gm3.metric("Scatter transfer",f"{100*geo['scatter_transfer']:.1f}%")
        geom_spec=simulate_spectrum(
            radionuclide,active_bq,geom_time,sensitivity,geom_bg,9.5,1.0,0.0,deadtime_default,
            seed=73331,half_life_min=float(RADIONUCLIDES[radionuclide]["half_life_min"]),
            geometry_response_factor=geo["response_factor"],scatter_transfer=geo["scatter_transfer"],
            coincidence_sum_factor=geo["response_factor"],lead_fluorescence_cps=lead_xray)
        ref_geo=estimate_sample_geometry_response(3.0,8.0,0.0,0.096,"Polypropylene",0.4)
        ref_activity = total_activity if activity_mode=="Fixed total activity" else concentration*3.0
        ref_spec=simulate_spectrum(
            radionuclide,ref_activity,geom_time,sensitivity,geom_bg,9.5,1.0,0.0,deadtime_default,
            seed=73331,half_life_min=float(RADIONUCLIDES[radionuclide]["half_life_min"]),
            geometry_response_factor=ref_geo["response_factor"],scatter_transfer=ref_geo["scatter_transfer"],
            coincidence_sum_factor=ref_geo["response_factor"],lead_fluorescence_cps=lead_xray)
        current_rate=float(geom_spec["Expected sample counts (teaching)"].sum()/geom_time)
        ref_rate=float(ref_spec["Expected sample counts (teaching)"].sum()/geom_time)
        delta=100.0*(current_rate/ref_rate-1.0) if ref_rate>0 else 0.0
        gm=st.columns(4)
        gm[0].metric("Expected sample rate",f"{current_rate:.1f} cps",delta=f"{delta:+.1f}% vs reference")
        gm[1].metric("Reference rate",f"{ref_rate:.1f} cps")
        expected_gross_rate=float(geom_spec["Expected counts (teaching)"].sum()/geom_time)
        gm[2].metric("Expected gross rate",f"{expected_gross_rate:.1f} cps")
        gm[3].metric("Expected sample counts",f"{int(round(current_rate*geom_time)):,}")
        sg=go.Figure()
        sg.add_trace(go.Scatter(x=ref_spec["Energy (keV)"],y=ref_spec["Expected counts (teaching)"],
            mode="lines",name="Reference geometry",line=dict(color="#58B9FF",width=1.5,dash="dash")))
        sg.add_trace(go.Scatter(x=geom_spec["Energy (keV)"],y=geom_spec["Expected counts (teaching)"],
            mode="lines",name="Current preparation",line=dict(color="#55E0B0",width=2),fill="tozeroy",fillcolor="rgba(85,224,176,0.08)"))
        if lead_xray>0:
            for e in (74.97,84.94):
                sg.add_vline(x=e,line_dash="dot",line_color="#FFC45B")
            st.info("The optional 75 and 85 keV features represent lead K-fluorescence in the surrounding shielding. They are environmental background features, not emissions from the selected radionuclide.")
        sg.update_layout(template="plotly_dark",height=430,margin=dict(l=10,r=10,t=45,b=30),
            title=f"Live geometry response · {radionuclide}",xaxis_title="Energy (keV)",yaxis_title="Expected counts per channel",
            paper_bgcolor="#0d1e2b",plot_bgcolor="#0d1e2b",font=dict(color="#f4f8fb"),legend=dict(orientation="h",y=-.22))
        st.plotly_chart(sg,use_container_width=True,key="geometry_spectrum_live_333")
        # Show the continuous expected count-rate trend across the volume range.
        volume_grid=np.linspace(0.2,6.0,45)
        rate_grid=[]
        lam_geom=math.log(2.0)/(float(RADIONUCLIDES[radionuclide]["half_life_min"])*60.0)
        decay_average=-math.expm1(-lam_geom*geom_time)/(lam_geom*geom_time) if lam_geom>0 else 1.0
        for v_scan in volume_grid:
            activity_scan=(total_activity if activity_mode=="Fixed total activity" else concentration*v_scan)
            g_scan=estimate_sample_geometry_response(v_scan,diameter_mm,pos_offset,matrix_mu,wall_material,wall_thickness)
            true_scan=max(activity_scan,0.0)*sensitivity*g_scan["response_factor"]*decay_average
            total_scan=true_scan+geom_bg+lead_xray
            rate_grid.append(true_scan/(1.0+total_scan*deadtime_default*1e-6))
        trend=go.Figure()
        trend.add_trace(go.Scatter(x=volume_grid,y=rate_grid,mode="lines",name="Expected sample rate",line=dict(color="#58B9FF",width=2.4)))
        trend.add_trace(go.Scatter(x=[volume_ml],y=[current_rate],mode="markers",name="Selected volume",marker=dict(color="#FFC45B",size=12,line=dict(color="#ffffff",width=1.5))))
        trend.update_layout(template="plotly_dark",height=300,margin=dict(l=10,r=10,t=38,b=28),title="Expected sample count rate across sample volume",
            xaxis_title="Sample volume (mL)",yaxis_title="Expected observed sample rate (cps)",paper_bgcolor="#0d1e2b",plot_bgcolor="#0d1e2b",font=dict(color="#f4f8fb"),legend=dict(orientation="h",y=-.25))
        st.plotly_chart(trend,use_container_width=True,key="geometry_volume_response_curve_333")
        _book_block("HOW SAMPLE PRESENTATION ENTERS THE MEASUREMENT",
            "The same activity can produce different count rates when its distribution within the vial changes. Volume and tube diameter determine fill height; the average photon path through the sample influences self-attenuation; wall material contributes additional attenuation; and positioning changes the fraction of activity located in the detector's most sensitive region. The plotted spectrum separates the reference preparation from the current preparation so the response change can be inspected directly.", tone="teal")
        render_math(r"h=\frac{V}{\pi(d/2)^2},\qquad R_{\mathrm{sample}}\approx R_{\mathrm{ref}}\,F_{\mathrm{active}}F_{\mathrm{atten}}F_{\mathrm{wall}}F_{\mathrm{position}}")
        st.caption("This deliberately transparent model uses an approximate active-zone overlap and exponential attenuation. It is not a Monte Carlo transport calculation or a validated WIZARD calibration. Use matched standards or measured geometry factors for quantitative experimental work.")
        with st.expander("Published context: why fill volume matters",expanded=False):
            st.markdown("Lodge et al. reported a measurable sample-volume dependence for a NaI(Tl) gamma counter: for 18F, a 4 mL sample had an efficiency approximately 0.94 relative to a 0.1 mL sample in their system. This is evidence that volume can matter, not a correction factor to transfer to another counter, vial or energy window. [Read the study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4452125/).")
        if st.button("Save current geometry experiment to Stage 2",type="primary",key="save_geom_run_333"):
            geom_df=geom_spec.copy()
            geom_df["Radionuclide"]=radionuclide
            geom_df["Volume (mL)"]=volume_ml
            geom_df["Tube diameter (mm)"]=diameter_mm
            geom_df["Fill height (mm)"]=geo["height_mm"]
            geom_df["Geometry response factor"]=geo["response_factor"]
            geom_df["Matrix"]=matrix_name
            geom_df["Tube material"]=wall_material
            geom_df["Position offset (mm)"]=pos_offset
            ds_name=f"GEOMETRY · {radionuclide} · {volume_ml:.1f}mL · {datetime.now().strftime('%H%M%S')}"
            st.session_state.datasets[ds_name]=new_record(geom_df,"Sample geometry and preparation",radionuclide,
                f"Teaching simulation; volume={volume_ml:.2f}mL; diameter={diameter_mm:.1f}mm; fill={geo['height_mm']:.1f}mm; relative response={geo['response_factor']:.3f}; matrix={matrix_name}.")
            st.success(f"Saved '{ds_name}' to Stage 2.")



# ----------------------------- state -----------------------------------------
if "datasets" not in st.session_state:
    st.session_state.datasets=demo_datasets()
if "selected_sample" not in st.session_state: st.session_state.selected_sample=""
if "acq_history" not in st.session_state: st.session_state.acq_history=[]
if "seed" not in st.session_state: st.session_state.seed=923741
if "calibrated_sensitivities" not in st.session_state: st.session_state.calibrated_sensitivities={}
if "spectrum_results" not in st.session_state: st.session_state.spectrum_results={}

# -------------------------------- header --------------------------------------
st.markdown("""<div class='hero'><div class='eyebrow'>QUANTITATIVE RADIOACTIVITY MEASUREMENT  /  INTERACTIVE WORKSHOP</div>
<h1>QUANTITATIVE NUCLEAR MEDICINE <span style='color:#55e0b0;font-weight:800'>LABORATORY</span></h1>
<p>An interactive learning environment for radiometric measurement, gamma spectroscopy, detector calibration, biodistribution and quantitative PET/SPECT comparison. Work through laboratory cases, acquire virtual measurements and analyse the resulting datasets.</p></div>""",unsafe_allow_html=True)

# -------------------------------- sidebar ------------------------------------
with st.sidebar:
    st.markdown("## GLOBAL SETTINGS")
    st.caption("Shared defaults used by the experiment pages.")
    with st.expander("Measurement configuration", expanded=True):
        radionuclide=st.selectbox("Radionuclide",list(RADIONUCLIDES),index=0,key="sidebar_radionuclide")
        st.caption(f"Reference photon energy: {RADIONUCLIDES[radionuclide]['energy_keV']:.1f} keV")
        detector=st.selectbox("Selected detector",list(DETECTOR_REL),index=2,key="sidebar_detector")
        count_time_global=st.slider("Default count duration (s)",5,300,60,5,key="sidebar_count_duration")
        bg_global=st.number_input("Background rate (cps)",0.0,5000.0,16.0,1.0,key="sidebar_background_rate")
        deadtime_global=st.number_input("Dead-time constant (µs)",0.0,100.0,2.5,.1,key="sidebar_deadtime")
        calibration_choices=["Illustrative default"]
        if radionuclide in st.session_state.calibrated_sensitivities:
            calibration_choices.append("Last fitted calibration")
        calibration_source=st.selectbox("Sensitivity source",calibration_choices,index=0,key="sidebar_sensitivity_source")
        default_sens=(st.session_state.calibrated_sensitivities[radionuclide]
                      if calibration_source=="Last fitted calibration" else DEFAULT_SENS[radionuclide])
        sensitivity_global=st.number_input("Sensitivity (cps/Bq)",.000001,10.0,float(default_sens),format="%.6f",key=f"sensitivity_{radionuclide}_{calibration_source}")
    with st.expander("Data session", expanded=False):
        st.metric("Saved experiment datasets",len(st.session_state.datasets))
        st.metric("Total saved rows",sum(len(x) for x in st.session_state.datasets.values()))
        if st.button("Restore demonstration datasets",width="stretch"):
            st.session_state.datasets=demo_datasets(); st.session_state.acq_history=[]; st.rerun()
        if st.button("Clear all generated datasets",width="stretch"):
            st.session_state.datasets={}; st.session_state.acq_history=[]; st.rerun()
        st.caption("Demonstration parameters are illustrative. The virtual counter is not connected to hardware.")

# ------------------------- spectrum-first console ----------------------------
last_acq = st.session_state.acq_history[-1] if st.session_state.acq_history else None
st.markdown("<div class='kicker'>CURRENT MEASUREMENT CONFIGURATION</div>", unsafe_allow_html=True)
status_cols = st.columns(4)
status_cols[0].metric("Selected radionuclide", radionuclide)
status_cols[1].metric("Reference photon energy", f"{RADIONUCLIDES[radionuclide]['energy_keV']:.1f} keV")
status_cols[2].metric("Detector channel", detector)
status_cols[3].metric("Default acquisition", f"{count_time_global} s")
if last_acq:
    st.markdown(
        f"<div class='readout'>LAST SAMPLE MEASUREMENT  |  {last_acq['Sample ID']}  |  "
        f"{last_acq['Net rate (cps)']:,.1f} CPS  |  SELECT THE LIVE ACQUISITION TAB TO BUILD A NEW SPECTRUM</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div class='readout'>SYSTEM READY  |  SELECT A RADIONUCLIDE AND OPEN LIVE ACQUISITION TO GENERATE A 2048-CHANNEL SPECTRUM</div>",
        unsafe_allow_html=True,
    )
st.caption("The emphasis is the measurement itself: energy calibration, peak formation, continuum, counting statistics and acquisition-time dependence.")

st.markdown("---")
st.markdown("<div class='kicker'>WORKSHOP WORKSPACE</div>",unsafe_allow_html=True)
st.markdown("<div class='qnl-nav-label'>BUILD 3.3.8 · CUSTOM BUTTON NAVIGATION</div>",unsafe_allow_html=True)
st.markdown("<div class='section-title'>From measurement to analysis</div>",unsafe_allow_html=True)
st.markdown("**Stage 1 — Experiment & acquisition:** work through a laboratory case and generate measurements using the virtual instrument. **Stage 2 — Analysis & visualisation:** select those same datasets for regression, comparison, plotting and export.")

st.markdown("""
<div class="stage-overview">
  <div class="stage-overview-card stage-one">
    <div class="stage-tag">STAGE 1 · EXPERIMENT & ACQUISITION</div>
    <div class="stage-name">Laboratory workbench</div>
    <div class="stage-copy">Read the chapter, review a realistic QC or troubleshooting case, configure the virtual instrument and generate measurement data.</div>
  </div>
  <div class="stage-overview-card stage-two">
    <div class="stage-tag">STAGE 2 · ANALYSIS & VISUALISATION</div>
    <div class="stage-name">Quantitative analysis workspace</div>
    <div class="stage-copy">Use the same saved measurements for calibration fits, normalisation, comparisons, uncertainty review, plots and data export.</div>
  </div>
</div>
""", unsafe_allow_html=True)
main_labels=["01 · GAMMA COUNTER GUIDE", "02 · PRACTICAL LABORATORY"]
main_nav=render_choice_tiles(main_labels,"qnl_main_navigation",columns=2,section_label="MAIN WORKSPACE")

if main_nav == main_labels[0]:
    render_acquisition_guide(radionuclide, sensitivity_global, bg_global, deadtime_global)

else:
    stage_labels=["STAGE 1  ·  EXPERIMENT & ACQUISITION","STAGE 2  ·  ANALYSIS & VISUALISATION"]
    stage_nav=render_choice_tiles(stage_labels,"qnl_stage_navigation",columns=2,section_label="WORKFLOW STAGE")

    # ============================== STAGE 1 ======================================
    if stage_nav == stage_labels[0]:
        experiment_labels=["01 · Sample assay","02 · Calibration","03 · Normalisation","04 · Geometry & matrix","05 · Tissue study","06 · PET–gamma","07 · Sample rack","08 · Spectroscopy","09 · Data import","10 · Live workbench"]
        experiment_nav=render_choice_tiles(experiment_labels,"qnl_experiment_navigation",columns=5,section_label="SELECT A PRACTICAL")

        # Single sample
        if experiment_nav == experiment_labels[0]:
            render_troubleshooting_case("single")
            theory_panel('THEORY 01 / QUANTITATIVE ASSAY',
                'From detected events to an activity estimate',
                'A well counter measures photon interactions, not becquerels directly. Activity is inferred from background-corrected count rate and calibrated sensitivity, with decay, detector response, geometry and matrix corrections.',
                [r'A(t)=A(t_0)e^{-\lambda(t-t_0)},\quad \lambda=\frac{\ln 2}{T_{1/2}}', r'R_{\mathrm{net}}=R_{\mathrm{gross}}-R_{\mathrm{bg}},\quad A_{\mathrm{est}}=\frac{R_{\mathrm{net,corr}}}{S_{\mathrm{cal}}G_{\mathrm{det}}G_{\mathrm{geom}}F_{\mathrm{matrix}}}', r'\%\,\mathrm{ID}/\mathrm{g}=100\frac{A_{\mathrm{tissue}}(t_0)}{A_{\mathrm{injected}}(t_0)m_{\mathrm{tissue}}}'],
                'How does the estimate change when geometry, background or counting time changes?',
                'Calibration sensitivity and correction factors are assumed known; counts are Poisson distributed and dead time is approximated as non-paralysable.')
            st.markdown("<div class='kicker'>EXPERIMENT 01 / UNKNOWN SAMPLE</div><div class='section-title'>Acquire an unknown sample</div>",unsafe_allow_html=True)
            a,b,c=st.columns(3)
            with a:
                sample_id=st.text_input("Sample identifier","UNKNOWN-001")
                true_activity=st.number_input("Teaching ground-truth activity at reference (Bq)",0.0,1e9,18000.0,500.0)
                mass=st.number_input("Tissue mass (g)",.001,100.0,.250,.01,format="%.3f")
            with b:
                elapsed=st.number_input("Elapsed time from reference (min)",0.0,100000.0,25.0,1.0)
                count_t=st.number_input("Sample count duration (s)",1.0,3600.0,float(count_time_global),5.0)
                bg_t=st.number_input("Background count duration (s)",1.0,3600.0,float(count_time_global),5.0)
            with c:
                geom=st.slider("Relative geometry response",.5,1.1,1.0,.01)
                matrix=st.slider("Relative matrix response",.5,1.1,.97,.01)
                dyn_norm=st.checkbox("Apply detector normalisation",True)
            half=float(RADIONUCLIDES[radionuclide]["half_life_min"]); at=true_activity*decay_factor(elapsed,half)
            render_math(r"A(t)=A(t_0)\,e^{-\lambda(t-t_0)},\qquad \lambda=\frac{\ln 2}{T_{1/2}}")
            st.caption(f"Simulated activity at count start: {at:,.1f} Bq. Ground truth is shown only for teaching and is not used by the estimator.")
            if st.button("▶ ACQUIRE UNKNOWN SAMPLE",type="primary",width="stretch",key="run_single"):
                st.session_state.seed+=1
                g=DETECTOR_REL[detector]
                raw=acquire(at,sensitivity_global,count_t,bg_global,deadtime_global,gain=g,geom=geom,matrix=matrix,seed=st.session_state.seed)
                # Independently simulated background with its own duration.
                rng=np.random.default_rng(st.session_state.seed+88); tau=deadtime_global*1e-6
                bn=int(rng.poisson(np_deadtime(bg_global,tau)*bg_t)); br=bn/bg_t
                brc=np_deadtime_correct(br,tau); grc=raw["gross_cps_corrected"]
                net=max(grc-brc,0) if np.isfinite(grc) and np.isfinite(brc) else np.nan
                norm_factor=float(np.mean(list(DETECTOR_REL.values()))/g) if dyn_norm else 1.0
                net_n=net*norm_factor
                mean_gain=float(np.mean(list(DETECTOR_REL.values()))) if dyn_norm else 1.0
                denominator=sensitivity_global*mean_gain*geom*matrix
                est=net_n/denominator if denominator>0 else np.nan
                # Propagate independent gross/background Poisson counting errors through
                # the non-paralysable correction using each acquisition's own duration.
                dg=1.0/max((1.0-raw["gross_cps_observed"]*tau)**2,1e-12)
                db=1.0/max((1.0-br*tau)**2,1e-12)
                sigma_g=(math.sqrt(max(raw["gross_counts"],1))/count_t)*dg
                sigma_b=(math.sqrt(max(bn,1))/bg_t)*db
                sigma=math.sqrt(sigma_g**2+sigma_b**2)*norm_factor/denominator if denominator>0 else np.nan
                row={"Sample ID":sample_id,"Radionuclide":radionuclide,"Detector":detector,"Ground truth activity at count (Bq) [teaching only]":at,
                     "Ground truth activity at reference (Bq) [teaching only]":true_activity,"Mass (g)":mass,"Elapsed time (min)":elapsed,
                     "Count time (s)":count_t,"Background time (s)":bg_t,"Gross counts":raw["gross_counts"],"Background counts":bn,
                     "Gross rate corrected (cps)":grc,"Background rate corrected (cps)":brc,"Net rate (cps)":net,
                     "Detector normalisation factor":norm_factor,"Normalised net rate (cps)":net_n,"Sensitivity (cps/Bq)":sensitivity_global,
                     "Geometry factor":geom,"Matrix factor":matrix,"Estimated activity at count (Bq)":est,"Activity SD (Bq)":sigma,
                     "%ID/g":100*est/(true_activity*decay_factor(elapsed,half)*mass) if true_activity>0 else np.nan}
                df=new_record(pd.DataFrame([row]),"Single sample acquisition",radionuclide,"Unknown sample measurement.")
                st.session_state.datasets[f"RUN · Single sample · {sample_id} · {datetime.now().strftime('%H%M%S')}"]=df
                st.session_state.acq_history.append({"Sample ID":sample_id,"Net rate (cps)":net,"Estimated activity (Bq)":est})
                st.success("Sample acquired and saved to Stage 2 analysis.")
                q1,q2,q3,q4=st.columns(4); q1.metric("Gross counts",f"{raw['gross_counts']:,}"); q2.metric("Background counts",f"{bn:,}"); q3.metric("Net corrected rate",f"{net:,.2f} cps"); q4.metric("Estimated activity",f"{est:,.1f} Bq")
                st.dataframe(df.drop(columns=["Timestamp","Scenario notes"]),width="stretch",hide_index=True)
                st.info("The ground-truth activity column is retained for teaching validation. Remove it before treating the exported table as blinded unknown-sample results.")

        # Calibration series
        elif experiment_nav == experiment_labels[1]:
            render_troubleshooting_case("calibration")
            theory_panel('THEORY 02 / ABSOLUTE CALIBRATION',
                'Determine the count-rate-to-activity sensitivity',
                'Known-activity standards are measured under matched radionuclide, energy-window, vial-geometry, fill-height and matrix conditions. The fitted slope is the empirical sensitivity.',
                [r'R_{\mathrm{net},i}=S_{\mathrm{cal}}A_i+b+\varepsilon_i', r'\bar A=\frac{1}{n}\sum_{i=1}^{n}A_i,\qquad \overline{R_{\mathrm{net}}}=\frac{1}{n}\sum_{i=1}^{n}R_{\mathrm{net},i}', r'\hat S_{\mathrm{cal}}=\frac{\sum_{i=1}^{n}(A_i-\bar A)(R_{\mathrm{net},i}-\overline{R_{\mathrm{net}}})}{\sum_{i=1}^{n}(A_i-\bar A)^2}', r'\hat b=\overline{R_{\mathrm{net}}}-\hat S_{\mathrm{cal}}\bar A,\qquad \hat\varepsilon_i=R_{\mathrm{net},i}-(\hat S_{\mathrm{cal}}A_i+\hat b)', r'\hat R_{\mathrm{net},i}=\hat S_{\mathrm{cal}}A_i+\hat b', r'\mathrm{SSE}=\sum_{i=1}^{n}(R_{\mathrm{net},i}-\hat R_{\mathrm{net},i})^2,\quad \mathrm{SST}=\sum_{i=1}^{n}(R_{\mathrm{net},i}-\overline{R_{\mathrm{net}}})^2', r'R^2=1-\frac{\mathrm{SSE}}{\mathrm{SST}}', r'A_{\mathrm{unknown}}=\frac{R_{\mathrm{unknown}}-\hat b}{\hat S_{\mathrm{cal}}}'],
                'Does the fitted slope remain stable across the activity range?',
                'Linear response is assumed; dead time, geometry and matrix effects may cause non-linearity.')
            st.markdown("<div class='kicker'>EXPERIMENT 02 / ABSOLUTE SENSITIVITY</div><div class='section-title'>Generate a multi-point calibration curve</div>",unsafe_allow_html=True)
            st.write("Measure traceable standards across an activity range. Fit net corrected count rate against known activity, then use the fitted slope as the empirical sensitivity for this radionuclide, energy window and geometry.")
            st.markdown("#### Configure the calibration challenge")
            st.caption("The key to revealing dead-time loss is to extend the standards into a sufficiently high count-rate range. The plot will show both the raw observed response and the dead-time-corrected response.")
            preset_col, range_col = st.columns([1.0, 1.4])
            with preset_col:
                nonlin_preset = st.selectbox(
                    "Demonstration strength",
                    ["Near-linear", "Moderate loss", "Clear non-linearity (recommended)", "Strong count-rate loss"],
                    index=2, key="cal_nonlin_preset")
            preset_max = {
                "Near-linear": 100_000.0,
                "Moderate loss": 500_000.0,
                "Clear non-linearity (recommended)": 2_000_000.0,
                "Strong count-rate loss": 5_000_000.0,
            }[nonlin_preset]
            x1,x2,x3=st.columns(3)
            with x1:
                nstd=st.slider("Number of standards",4,12,8,key="cal_nstd")
                minact=st.number_input("Minimum activity (Bq)",10.0,1e7,500.0,100.0,key="cal_min_activity")
            with x2:
                maxact=st.number_input("Maximum activity (Bq)",100.0,1e9,preset_max,1000.0,key=f"cal_max_activity_{nonlin_preset}")
                reps=st.slider("Replicates per standard",1,6,3,key="cal_replicates")
            with x3:
                cal_t=st.number_input("Count time per replicate (s)",5.0,600.0,60.0,5.0,key="cal_count_time")
                cal_bg=st.number_input("Background rate (cps)",0.0,5000.0,float(bg_global),1.0,key="calibration_background_rate")
            cal_tau=st.number_input("Dead-time (µs) for calibration",0.0,100.0,float(deadtime_global),.1,key="cal_deadtime_us")
            expected_top_rate=maxact*sensitivity_global
            expected_loss=100.0*(1.0-(1.0/(1.0+expected_top_rate*cal_tau*1e-6))) if cal_tau>0 else 0.0
            st.info(f"At the highest standard, the ideal sample rate is approximately {expected_top_rate:,.0f} cps. With τ = {cal_tau:g} µs, the non-paralysable model predicts about {expected_loss:.1f}% count-rate loss before correction. For a clear demonstration, aim for at least 10-20% predicted loss.")
            if maxact<=minact: st.error("Maximum activity must exceed minimum activity.")
            if st.button("GENERATE CALIBRATION MEASUREMENTS",type="primary",width="stretch",key="run_cal") and maxact>minact:
                rng=np.random.default_rng(st.session_state.seed+100); st.session_state.seed+=101
                standards=np.geomspace(minact,maxact,nstd); rows=[]
                for si,activity in enumerate(standards,1):
                    for rep in range(1,reps+1):
                        r=acquire(activity,sensitivity_global,cal_t,cal_bg,cal_tau,seed=int(rng.integers(1,2**31-1)))
                        observed_net=max(r["gross_cps_observed"]-r["background_cps_observed"],0.0)
                        rows.append({"Standard":f"STD-{si:02d}","Replicate":rep,"Known activity (Bq)":activity,"Gross counts":r["gross_counts"],
                                     "Background counts":r["background_counts"],"Observed net rate (cps)":observed_net,
                                     "Net rate (cps)":r["net_cps_corrected"],"Net rate SD (cps)":r["sigma_net_cps"],
                                     "Count time (s)":cal_t,"Dead-time loss estimate (%)":100.0*(1.0-observed_net/max(activity*sensitivity_global,1e-12)),
                                     "Ground truth sensitivity (cps/Bq) [teaching only]":sensitivity_global})
                df=new_record(pd.DataFrame(rows),"Calibration series",radionuclide,"Poisson-simulated standards retaining both raw observed and dead-time-corrected responses.")
                name=f"RUN · Calibration · {radionuclide} · {datetime.now().strftime('%H%M%S')}"
                st.session_state.datasets[name]=df
                agg=df.groupby("Known activity (Bq)",as_index=False).agg({"Observed net rate (cps)":"mean","Net rate (cps)":"mean","Net rate SD (cps)":"mean","Dead-time loss estimate (%)":"mean"})
                # The corrected response is used for the transferable sensitivity. The uncorrected response
                # is retained and plotted to make rate-dependent loss visible to learners.
                slope,inter=np.polyfit(agg["Known activity (Bq)"],agg["Net rate (cps)"],1)
                obs_slope,obs_inter=np.polyfit(agg["Known activity (Bq)"],agg["Observed net rate (cps)"],1)
                fit=slope*agg["Known activity (Bq)"]+inter
                ssr=float(np.sum((agg["Net rate (cps)"]-fit)**2)); sst=float(np.sum((agg["Net rate (cps)" ]-agg["Net rate (cps)"].mean())**2)); r2=1-ssr/sst if sst else np.nan
                obs_fit=obs_slope*agg["Known activity (Bq)"]+obs_inter
                obs_ssr=float(np.sum((agg["Observed net rate (cps)"]-obs_fit)**2)); obs_sst=float(np.sum((agg["Observed net rate (cps)"]-agg["Observed net rate (cps)"].mean())**2)); obs_r2=1-obs_ssr/obs_sst if obs_sst else np.nan
                st.session_state.calibrated_sensitivities[radionuclide]=float(slope)
                st.success(f"Calibration dataset saved. Corrected sensitivity = {slope:.6f} cps/Bq (intercept {inter:.2f} cps; R² {r2:.5f}). This corrected slope is available in the instrument sensitivity selector.")
                mc1,mc2,mc3=st.columns(3)
                mc1.metric("Corrected fit R²",f"{r2:.5f}")
                mc2.metric("Observed fit R²",f"{obs_r2:.5f}")
                mc3.metric("Mean loss at top standard",f"{agg['Dead-time loss estimate (%)'].iloc[-1]:.1f}%")
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=agg["Known activity (Bq)"],y=agg["Observed net rate (cps)"],mode="markers+lines",name="Observed net rate (no correction)",marker=dict(color="#ffb45e",size=9)))
                fig.add_trace(go.Scatter(x=agg["Known activity (Bq)"],y=agg["Net rate (cps)"],mode="markers",name="Dead-time corrected net rate",marker=dict(color="#55e0b0",size=9),error_y=dict(type="data",array=agg["Net rate SD (cps)"],visible=True)))
                xx=np.linspace(0,maxact,150)
                fig.add_trace(go.Scatter(x=xx,y=sensitivity_global*xx,mode="lines",name="Ideal linear response",line=dict(color="#7db8ff",dash="dash")))
                fig.add_trace(go.Scatter(x=xx,y=slope*xx+inter,mode="lines",name="Corrected linear fit",line=dict(color="#55e0b0",width=2)))
                fig.update_layout(height=430,xaxis_title="Known activity (Bq)",yaxis_title="Net count rate (cps)",template="plotly_dark",legend=dict(orientation="h",y=-0.25),margin=dict(b=100))
                st.plotly_chart(fig,width="stretch",key="calibration_observed_vs_corrected_plot")
                st.markdown("#### How to interpret the curves")
                st.write("The orange observed response bends below the ideal line as the rate rises. The green corrected measurements should return towards linearity because the simulator applies the specified non-paralysable correction. In a real counter, correction is only appropriate within the instrument's validated rate range and with its verified dead-time model.")
                st.dataframe(df,width="stretch",hide_index=True)

        # Normalisation
        elif experiment_nav == experiment_labels[2]:
            render_troubleshooting_case("normalisation")
            theory_panel('THEORY 03 / RELATIVE DETECTOR NORMALISATION',
                'Separate relative response from absolute calibration',
                'A shared reference source is counted in each detector. Relative factors map each measured sensitivity to the fleet mean; this does not establish activity in Bq.',
                [r'S_j=\frac{R_{\mathrm{net},j}}{A_{\mathrm{ref}}},\quad \overline{S}=\frac{1}{n}\sum_{j=1}^{n}S_j', r'F_j=\frac{\overline{S}}{S_j},\quad R_{\mathrm{norm},j}=F_jR_{\mathrm{net},j}', r'S_{\mathrm{norm},j}=F_jS_j\approx\overline{S}'],
                'After correction, do detector sensitivities converge around the fleet mean?',
                'The same standard, geometry and timing must be used for every channel; replicates reduce random uncertainty.')
            st.markdown("<div class='kicker'>EXPERIMENT 03 / DETECTOR-TO-DETECTOR NORMALISATION</div><div class='section-title'>Measure detector response using a common reference</div>",unsafe_allow_html=True)
            st.write("The same reference source is measured in each detector. Estimate each detector's sensitivity and calculate a correction factor that maps the detector response to the fleet mean.")
            n1,n2,n3=st.columns(3)
            with n1: ref_activity=st.number_input("Reference activity (Bq)",100.0,1e8,12000.0,500.0)
            with n2: norm_reps=st.slider("Replicates per detector",1,10,5)
            with n3: norm_t=st.number_input("Count time (s)",5.0,600.0,45.0,5.0)
            norm_bg=st.number_input("Background cps",0.0,5000.0,float(bg_global),1.0)
            if st.button("RUN DETECTOR NORMALISATION",type="primary",width="stretch",key="run_norm"):
                rng=np.random.default_rng(st.session_state.seed+200); st.session_state.seed+=201; rows=[]
                for det,g in DETECTOR_REL.items():
                    for rep in range(1,norm_reps+1):
                        r=acquire(ref_activity,sensitivity_global,norm_t,norm_bg,deadtime_global,gain=g,seed=int(rng.integers(1,2**31-1)))
                        rows.append({"Detector":det,"Replicate":rep,"Reference activity (Bq)":ref_activity,"Gross counts":r["gross_counts"],
                                     "Background counts":r["background_counts"],"Net rate (cps)":r["net_cps_corrected"],
                                     "Observed sensitivity (cps/Bq)":r["net_cps_corrected"]/ref_activity})
                df=pd.DataFrame(rows); sens=df.groupby("Detector")["Observed sensitivity (cps/Bq)"].mean(); fleet=sens.mean(); factors=fleet/sens
                df["Fleet mean sensitivity (cps/Bq)"]=fleet; df["Normalisation factor"]=df["Detector"].map(factors)
                df["Normalised sensitivity (cps/Bq)"]=df["Observed sensitivity (cps/Bq)"]*df["Normalisation factor"]
                df=new_record(df,"Detector normalisation",radionuclide,"Reference-source measurements repeated across detector channels.")
                name=f"RUN · Detector normalisation · {radionuclide} · {datetime.now().strftime('%H%M%S')}"; st.session_state.datasets[name]=df
                st.success("Detector normalisation experiment saved to Stage 2.")
                summary=df.groupby("Detector",as_index=False).agg({"Observed sensitivity (cps/Bq)":"mean","Normalisation factor":"first","Normalised sensitivity (cps/Bq)":"mean"})
                f=go.Figure(); f.add_trace(go.Bar(x=summary.Detector,y=summary["Observed sensitivity (cps/Bq)"],name="Raw sensitivity")); f.add_trace(go.Bar(x=summary.Detector,y=summary["Normalised sensitivity (cps/Bq)"],name="After normalisation"))
                f.update_layout(barmode="group",height=350,yaxis_title="Sensitivity (cps/Bq)",template="plotly_dark")
                st.plotly_chart(f,width="stretch",key="stage1_normalisation_results_plot"); st.dataframe(summary,width="stretch",hide_index=True)
                render_math(r"F_j=\frac{\overline{S}}{S_j},\qquad S_{j,\mathrm{norm}}=F_jS_j")

        # Geometry and matrix
        elif experiment_nav == experiment_labels[3]:
            render_troubleshooting_case("geometry")
            theory_panel('THEORY 04 / GEOMETRY & MATRIX RESPONSE',
                'Why identical activity can produce different count rates',
                'Observed efficiency depends on sample dimensions, fill height, vial material, source distribution and matrix composition. A geometry factor is valid only for the conditions in which it was established.',
                [r'R_{\mathrm{net}}=S_{\mathrm{cal}}A G_{\mathrm{geom}}F_{\mathrm{matrix}}', r'A_{\mathrm{corr}}=\frac{R_{\mathrm{net}}}{S_{\mathrm{cal}}G_{\mathrm{geom}}F_{\mathrm{matrix}}}', r'\left(\frac{u(A_{\mathrm{corr}})}{A_{\mathrm{corr}}}\right)^2\simeq\left(\frac{u(R_{\mathrm{net}})}{R_{\mathrm{net}}}\right)^2+\left(\frac{u(S_{\mathrm{cal}})}{S_{\mathrm{cal}}}\right)^2+\left(\frac{u(G_{\mathrm{geom}})}{G_{\mathrm{geom}}}\right)^2+\left(\frac{u(F_{\mathrm{matrix}})}{F_{\mathrm{matrix}}}\right)^2'],
                'Can one cps/Bq factor be transferred between a small tube and a high-fill vial?',
                'Geometry and matrix factors are illustrative; real factors require matched standards and intended containers.')
            st.markdown("<div class='kicker'>EXPERIMENT 04 / GEOMETRY & MATRIX EFFECT</div><div class='section-title'>Compare vial, tube and sample-matrix responses</div>",unsafe_allow_html=True)
            st.write("Hold activity constant and vary the sample geometry or matrix attenuation. The simulator applies a declared relative response factor so participants can see the bias introduced by using a calibration factor outside its validated geometry.")
            geom_defs={"13 mm counting tube":1.0,"3 mL vial":.98,"5 mL tube":.95,"20 mL vial":.90,"High-fill / unusual geometry":.86}
            geom_act=st.number_input("Nominal activity (Bq)",100.0,1e8,10000.0,500.0,key="geom_act")
            geom_reps=st.slider("Replicates per geometry",1,8,4,key="geom_reps")
            selected_geoms=st.multiselect("Geometries to compare",list(geom_defs),default=list(geom_defs))
            matrix_effect=st.slider("Common sample matrix response",.5,1.05,.97,.01)
            if st.button("RUN GEOMETRY EXPERIMENT",type="primary",width="stretch",key="run_geom") and selected_geoms:
                rng=np.random.default_rng(st.session_state.seed+300); st.session_state.seed+=301; rows=[]
                for name in selected_geoms:
                    for rep in range(1,geom_reps+1):
                        g=geom_defs[name]; r=acquire(geom_act,sensitivity_global,count_time_global,bg_global,deadtime_global,geom=g,matrix=matrix_effect,seed=int(rng.integers(1,2**31-1)))
                        naive=r["net_cps_corrected"]/sensitivity_global
                        corrected=r["net_cps_corrected"]/(sensitivity_global*g*matrix_effect)
                        rows.append({"Geometry":name,"Replicate":rep,"Nominal activity (Bq)":geom_act,"Relative geometry factor":g,
                                     "Matrix factor":matrix_effect,"Net rate (cps)":r["net_cps_corrected"],"Naive activity estimate (Bq)":naive,
                                     "Geometry/matrix corrected activity (Bq)":corrected})
                df=new_record(pd.DataFrame(rows),"Geometry and matrix experiment",radionuclide,"Constant activity measured in different sample geometries.")
                st.session_state.datasets[f"RUN · Geometry challenge · {datetime.now().strftime('%H%M%S')}"]=df
                st.success("Geometry dataset saved to Stage 2.")
                fig=px.box(df,x="Geometry",y="Naive activity estimate (Bq)",color="Geometry",points="all",title="Naive activity estimate: geometry-dependent bias")
                fig.update_layout(height=370,showlegend=False,template="plotly_dark"); st.plotly_chart(fig,width="stretch",key="stage1_geometry_bias_plot")
                st.dataframe(df,width="stretch",hide_index=True)

        # Hypothetical tissue study
        elif experiment_nav == experiment_labels[4]:
            render_troubleshooting_case("tissue")
            theory_panel('THEORY 05 / IN-VITRO BIODISTRIBUTION',
                'Convert ex-vivo activity into %ID and %ID/g',
                'Each harvested tissue is weighed and counted. Activity is corrected to a shared reference time, then normalised to injected activity and tissue mass.',
                [r'A_{\mathrm{tissue}}(t_0)=A_{\mathrm{tissue}}(t)e^{\lambda(t-t_0)}', r'\%\,\mathrm{ID}=100\frac{A_{\mathrm{tissue}}(t_0)}{A_{\mathrm{injected}}(t_0)},\quad \%\,\mathrm{ID}/\mathrm{g}=\frac{\%\,\mathrm{ID}}{m_{\mathrm{tissue}}}', r'\mathrm{TBR}=\frac{(\%\,\mathrm{ID}/\mathrm{g})_{\mathrm{tumour}}}{(\%\,\mathrm{ID}/\mathrm{g})_{\mathrm{reference\ tissue}}}'],
                'Does tumour enrichment improve tumour-to-background ratio rather than merely changing total uptake?',
                'Study-arm examples are synthetic redistribution models, not pharmacokinetic or therapeutic predictions.')
            st.markdown("<div class='kicker'>EXPERIMENT 05 / IN-VIVO TO EX-VIVO</div><div class='section-title'>Generate a hypothetical biodistribution study</div>",unsafe_allow_html=True)
            st.write("Create a synthetic animal biodistribution dataset with user-defined injected activity, uptake fractions, tissue masses and collection timing. Simulated counts are then acquired for each tissue sample.")
            t1,t2,t3=st.columns(3)
            with t1: injected=st.number_input("Injected activity at reference (Bq)",1000.0,1e10,2.5e6,1e5,format="%.0f")
            with t2: uptake_time=st.number_input("Tissue collection time (min)",0.0,100000.0,90.0,5.0)
            with t3: tissue_count_t=st.number_input("Count duration per tissue (s)",5.0,600.0,60.0,5.0)
            organs_default=pd.DataFrame({"Tissue":["Blood","Liver","Kidney","Tumour","Bone","Muscle","Lung","Spleen"],
                                         "Uptake (%ID)": [3.5,18.5,11.5,8.2,4.8,2.6,7.1,9.2],
                                         "Mass (g)": [.42,1.25,.31,.24,.36,.58,.19,.11]})
            edited=st.data_editor(organs_default,width="stretch",num_rows="dynamic",key="tissue_editor",
                                  column_config={"Uptake (%ID)":st.column_config.NumberColumn(min_value=0,max_value=100,step=.1),"Mass (g)":st.column_config.NumberColumn(min_value=.001,max_value=100,step=.01)})
            st.caption("The entered values define the CONTROL arm. Their sum may be below 100% because activity can remain in unlisted tissues, the carcass or excreta. The scenarios below preserve the total uptake across the listed tissues by redistributing the tumour-related change proportionally across non-tumour tissues.")
            enrich_fold=st.slider("Tumour-enriched arm: tumour uptake multiplier",1.05,3.00,1.50,0.05,key="arm_enrich_fold")
            reduce_fold=st.slider("Tumour-reduced arm: tumour uptake multiplier",0.10,0.95,0.65,0.05,key="arm_reduce_fold")
            study_arms=st.multiselect("Hypothetical study arms",["Control","Tumour-enriched","Tumour-reduced"],default=["Control","Tumour-enriched"])
            st.markdown("**What do the three arms mean?** Control uses the baseline uptake values you entered. Tumour-enriched increases tumour uptake relative to baseline, then proportionally reduces the listed non-tumour tissue shares so the total for the listed tissues is conserved. Tumour-reduced decreases tumour uptake, then redistributes the difference proportionally across the other listed tissues. These are deliberately simplified hypothetical redistribution scenarios, not biological predictions.")
            preview_input = edited[edited["Tissue"].notna()].copy()
            preview_base = {str(x["Tissue"]): max(float(x["Uptake (%ID)"]), 0.0) for _, x in preview_input.iterrows()}
            preview_mass = {str(x["Tissue"]): max(float(x["Mass (g)"]), .001) for _, x in preview_input.iterrows()}
            preview_tumour = next((k for k in preview_base if k.strip().lower() in {"tumour", "tumor"}), None)
            preview_liver = next((k for k in preview_base if k.strip().lower() == "liver"), None)
            preview_total = sum(preview_base.values())
            preview_rows = []
            for preview_arm, preview_fold in [("Control",1.0),("Tumour-enriched",float(enrich_fold)),("Tumour-reduced",float(reduce_fold))]:
                values = dict(preview_base)
                if preview_tumour and preview_total > 0 and preview_arm != "Control":
                    tumour_new = min(preview_base[preview_tumour] * preview_fold, preview_total)
                    other_total = preview_total - preview_base[preview_tumour]
                    scale_other = max(preview_total - tumour_new,0.0) / other_total if other_total > 0 else 0.0
                    values[preview_tumour] = tumour_new
                    for key in values:
                        if key != preview_tumour:
                            values[key] = preview_base[key] * scale_other
                tbr_preview = ((values[preview_tumour] / preview_mass[preview_tumour]) /
                               (values[preview_liver] / preview_mass[preview_liver])) if preview_tumour and preview_liver and values[preview_liver] > 0 else np.nan
                preview_rows.append({"Study arm":preview_arm,"Tumour uptake (%ID)":values.get(preview_tumour,np.nan),
                                     "Liver uptake (%ID)":values.get(preview_liver,np.nan),
                                     "Tumour/liver %ID/g ratio":tbr_preview,"Listed-tissue uptake total (%ID)":sum(values.values())})
            st.markdown("#### Study-arm assumptions preview")
            st.dataframe(pd.DataFrame(preview_rows),width="stretch",hide_index=True)
            if st.button("GENERATE TISSUE COUNTING STUDY",type="primary",width="stretch",key="run_tissue") and study_arms:
                rng=np.random.default_rng(st.session_state.seed+400); st.session_state.seed+=401; half=float(RADIONUCLIDES[radionuclide]["half_life_min"]); fdec=decay_factor(uptake_time,half); rows=[]
                for arm in study_arms:
                    for _,rr in edited.iterrows():
                        if pd.isna(rr.get("Tissue")) or str(rr.get("Tissue")).strip()=="":
                            continue
                        name=str(rr["Tissue"]); mass_g=max(float(rr["Mass (g)"]),.001); base_pct=max(float(rr["Uptake (%ID)"]),0)
                        # The current row uses a mass-balanced redistribution of the listed-tissue uptake pool.
                        # Recompute the map for this arm; the listed total is conserved while tumour share changes.
                        base_rows = edited[edited["Tissue"].notna()].copy()
                        base_map = {str(x["Tissue"]): max(float(x["Uptake (%ID)"]), 0.0) for _, x in base_rows.iterrows()}
                        tumour_key = next((k for k in base_map if k.strip().lower() in {"tumour", "tumor"}), None)
                        target_map = dict(base_map)
                        listed_total = float(sum(base_map.values()))
                        if tumour_key and listed_total > 0 and arm != "Control":
                            tumour_base = base_map[tumour_key]
                            fold = float(enrich_fold if arm == "Tumour-enriched" else reduce_fold)
                            tumour_new = min(tumour_base * fold, listed_total)
                            other_total = listed_total - tumour_base
                            other_scale = max(listed_total - tumour_new, 0.0) / other_total if other_total > 0 else 0.0
                            target_map[tumour_key] = tumour_new
                            for tissue_key in target_map:
                                if tissue_key != tumour_key:
                                    target_map[tissue_key] = base_map[tissue_key] * other_scale
                        pct=float(target_map.get(name, base_pct)); a_ref=injected*pct/100; a_count=a_ref*fdec
                        r=acquire(a_count,sensitivity_global,tissue_count_t,bg_global,deadtime_global,seed=int(rng.integers(1,2**31-1)))
                        est_count=r["net_cps_corrected"]/sensitivity_global
                        est_ref=est_count/fdec if fdec>0 else np.nan
                        rows.append({"Study arm":arm,"Tissue":name,"Mass (g)":mass_g,"Injected activity (Bq)":injected,"Collection time (min)":uptake_time,
                                     "True uptake (%ID) [teaching only]":pct,"True activity at count (Bq) [teaching only]":a_count,
                                     "Gross counts":r["gross_counts"],"Background counts":r["background_counts"],"Net rate (cps)":r["net_cps_corrected"],
                                     "Estimated activity at count (Bq)":est_count,"Estimated activity at injection reference (Bq)":est_ref,
                                     "Estimated %ID":100*est_ref/injected,"Estimated %ID/g":100*est_ref/(injected*mass_g)})
                if not rows:
                    st.error("Add at least one tissue row with a tissue name before generating the study.")
                    st.stop()
                df=new_record(pd.DataFrame(rows),"Hypothetical biodistribution",radionuclide,"Synthetic organ/tissue uptake and Poisson counting. Control uses entered baseline; tumour-enriched/reduced arms change tumour uptake and proportionally redistribute listed non-tumour uptake to conserve the listed-tissue total.")
                st.session_state.datasets[f"RUN · Tissue study · {radionuclide} · {datetime.now().strftime('%H%M%S')}"]=df
                st.success("Hypothetical tissue dataset saved to Stage 2 analysis.")
                u,v=st.columns(2)
                with u:
                    ff=px.bar(df,x="Tissue",y="Estimated %ID/g",color="Study arm",barmode="group",title="Estimated %ID/g by tissue and study arm"); ff.update_layout(height=350,template="plotly_dark"); st.plotly_chart(ff,width="stretch",key="stage1_tissue_percent_idg_plot")
                with v:
                    ff2=px.bar(df,x="Tissue",y="Estimated %ID",color="Study arm",barmode="group",title="Estimated %ID by tissue and study arm"); ff2.update_layout(height=350,template="plotly_dark"); st.plotly_chart(ff2,width="stretch",key="stage1_tissue_percent_id_plot")
                st.dataframe(df.drop(columns=["Timestamp","Scenario notes"]),width="stretch",hide_index=True)

        # PET-gamma comparison
        elif experiment_nav == experiment_labels[5]:
            render_troubleshooting_case("pet")
            theory_panel('THEORY 06 / CROSS-MODALITY VALIDATION',
                'Compare PET/SPECT concentration with ex-vivo counting',
                'PET/SPECT reports reconstructed activity concentration; a well counter measures activity in a harvested sample. Match tissue labels, reference time and sampled volume before comparing.',
                [r'C_{\gamma}(t_0)=\frac{A_{\gamma}(t)e^{\lambda(t-t_0)}}{V_{\mathrm{sample}}},\quad V_{\mathrm{sample}}=\frac{m}{\rho}', r'C_{\mathrm{image,corr}}(t_0)=\frac{C_{\mathrm{image}}(t)e^{\lambda(t-t_0)}}{RC}', r'\mathrm{Bias}=\overline{C_{\mathrm{image}}-C_{\gamma}},\quad \mathrm{LoA}=\mathrm{Bias}\pm1.96s_{\mathrm{diff}}'],
                'Are both modalities measuring the same volume, time point and activity definition?',
                'Recovery-coefficient correction is optional and must not be applied twice; document tissue density and ROI matching.')
            st.markdown("<div class='kicker'>EXPERIMENT 06 / CROSS-MODALITY COMPARISON</div><div class='section-title'>Compare PET concentration with ex-vivo counting</div>",unsafe_allow_html=True)
            p1,p2,p3=st.columns(3)
            with p1: pet_conc=st.number_input("PET ROI concentration (Bq/mL)",1.0,1e9,12500.0,100.0)
            with p2: gamma_activity=st.number_input("Gamma-counter aliquot activity (Bq)",1.0,1e9,61000.0,100.0)
            with p3: aliquot_vol=st.number_input("Aliquot volume (mL)",.001,1000.0,5.0,.1)
            time_offset=st.number_input("Time difference: gamma count after PET (min)",0.0,100000.0,15.0,1.0)
            if st.button("SAVE PET–GAMMA COMPARISON",type="primary",width="stretch",key="run_cross"):
                half=float(RADIONUCLIDES[radionuclide]["half_life_min"]); gamma_conc=gamma_activity/aliquot_vol
                gamma_at_pet=gamma_conc/decay_factor(time_offset,half)
                factor=gamma_at_pet/pet_conc
                df=pd.DataFrame([{"PET ROI concentration (Bq/mL)":pet_conc,"Gamma aliquot activity (Bq)":gamma_activity,"Aliquot volume (mL)":aliquot_vol,
                                  "Gamma concentration at count (Bq/mL)":gamma_conc,"Time offset (min)":time_offset,
                                  "Gamma concentration at PET time (Bq/mL)":gamma_at_pet,"Gamma/PET ratio":factor,"Difference (%)":100*(factor-1)}])
                df=new_record(df,"PET-gamma comparison",radionuclide,"Hypothetical cross-modality phantom comparison.")
                st.session_state.datasets[f"RUN · PET-gamma comparison · {datetime.now().strftime('%H%M%S')}"]=df
                st.success(f"Comparison saved. Time-matched gamma/PET ratio = {factor:.4f}.")
                z=go.Figure(go.Bar(x=["PET ROI","Time-matched gamma aliquot"],y=[pet_conc,gamma_at_pet],text=[f"{pet_conc:,.0f}",f"{gamma_at_pet:,.0f}"],textposition="outside",marker_color=[PALETTE[0],PALETTE[1]]))
                z.update_layout(height=330,yaxis_title="Activity concentration (Bq/mL)",template="plotly_dark"); st.plotly_chart(z,width="stretch",key="stage1_pet_gamma_comparison_plot")

        # Automated sample rack / batch acquisition
        elif experiment_nav == experiment_labels[6]:
            render_troubleshooting_case("rack")
            theory_panel('THEORY 07 / AUTOMATED SAMPLE RACK',
                'Batch acquisition and sample identity control',
                'A rack acquisition repeats the same measurement sequence across labelled positions. Sample identity, position, detector assignment, background and count duration should be retained for traceability.',
                [r'N_i\sim\mathrm{Poisson}(R_it_i),\quad u(N_i)=\sqrt{N_i}', r'R_{\mathrm{net},i}=\frac{N_{\mathrm{sample},i}}{t_{\mathrm{sample}}}-\frac{N_{\mathrm{bg}}}{t_{\mathrm{bg}}}', r'\mathrm{RSD}(N_i)=\frac{1}{\sqrt{N_i}}'],
                'Can you identify a position effect or sample-label error from the rack output?',
                'The virtual rack assumes sequential counting; carryover, contamination and transport faults are not modelled.')
            st.markdown("<div class='kicker'>EXPERIMENT 07 / AUTOMATED SAMPLE HANDLING</div><div class='section-title'>Rack loading and batch acquisition</div>",unsafe_allow_html=True)
            st.write("Prepare a rack, assign each position a sample, and acquire a batch using a shared measurement protocol. The rack sequence and all generated counts are retained for downstream analysis.")
            rack_n=st.select_slider("Rack capacity",options=[6,10,12,20],value=10)
            defaults=pd.DataFrame({"Position":np.arange(1,rack_n+1),"Sample ID":[f"TISSUE-{i:02d}" for i in range(1,rack_n+1)],
                                   "Activity at reference (Bq)":np.linspace(2500,22000,rack_n).round(0),
                                   "Elapsed time (min)":np.full(rack_n,20.0),"Mass (g)":np.full(rack_n,.25),
                                   "Geometry factor":np.ones(rack_n),"Matrix factor":np.full(rack_n,.97)})
            rack_df=st.data_editor(defaults,num_rows="fixed",width="stretch",key=f"rack_editor_{rack_n}",
                column_config={"Position":st.column_config.NumberColumn(disabled=True),
                               "Activity at reference (Bq)":st.column_config.NumberColumn(min_value=0,format="%.0f"),
                               "Elapsed time (min)":st.column_config.NumberColumn(min_value=0,format="%.1f"),
                               "Mass (g)":st.column_config.NumberColumn(min_value=.001,format="%.3f"),
                               "Geometry factor":st.column_config.NumberColumn(min_value=.1,max_value=1.2,format="%.3f"),
                               "Matrix factor":st.column_config.NumberColumn(min_value=.1,max_value=1.2,format="%.3f")})
            rb1,rb2,rb3=st.columns(3)
            with rb1: rack_count_s=st.number_input("Count time per vial (s)",5.0,600.0,30.0,5.0,key="rack_time")
            with rb2: rack_bg=st.number_input("Background rate (cps)",0.0,5000.0,float(bg_global),1.0,key="rack_bg")
            with rb3: auto_detector=st.checkbox("Rotate through detector channels A–F",False,key="rack_auto_detector")
            if st.button("▶ LOAD RACK AND RUN BATCH",type="primary",width="stretch",key="run_rack"):
                rng=np.random.default_rng(st.session_state.seed+719); st.session_state.seed+=720
                batch_rows=[]; half=float(RADIONUCLIDES[radionuclide]["half_life_min"])
                for ix,rrow in rack_df.iterrows():
                    did=list(DETECTOR_REL)[int(ix)%len(DETECTOR_REL)] if auto_detector else detector
                    age=float(rrow["Elapsed time (min)"]); at=float(rrow["Activity at reference (Bq)"])*decay_factor(age,half)
                    geomf=float(rrow["Geometry factor"]); matf=float(rrow["Matrix factor"]); massf=float(rrow["Mass (g)"])
                    meas=acquire(at,sensitivity_global,rack_count_s,rack_bg,deadtime_global,
                                 gain=DETECTOR_REL[did],geom=geomf,matrix=matf,seed=int(rng.integers(1,2**31-1)))
                    normf=float(np.mean(list(DETECTOR_REL.values()))/DETECTOR_REL[did])
                    net=meas["net_cps_corrected"]; est=net*normf/(sensitivity_global*np.mean(list(DETECTOR_REL.values()))*geomf*matf)
                    batch_rows.append({"Rack position":int(rrow["Position"]),"Sample ID":str(rrow["Sample ID"]),"Detector":did,
                        "Radionuclide":radionuclide,"Activity at reference (Bq) [teaching only]":float(rrow["Activity at reference (Bq)"]),
                        "Activity at count start (Bq) [teaching only]":at,"Elapsed time (min)":age,"Mass (g)":massf,
                        "Gross counts":meas["gross_counts"],"Background counts":meas["background_counts"],"Count time (s)":rack_count_s,
                        "Net rate (cps)":net,"Detector normalisation factor":normf,"Geometry factor":geomf,"Matrix factor":matf,
                        "Estimated activity at count (Bq)":est,"Estimated %ID/g":100*est/(float(rrow["Activity at reference (Bq)"])*massf) if float(rrow["Activity at reference (Bq)"])>0 else np.nan})
                batch=new_record(pd.DataFrame(batch_rows),"Automated rack batch",radionuclide,"Synthetic sequential rack run; per-vial Poisson acquisition.")
                st.session_state.datasets[f"RUN · Rack batch · {radionuclide} · {datetime.now().strftime('%H%M%S')}"]=batch
                st.success(f"Batch complete: {len(batch)} positions acquired and saved to Stage 2.")
                k1,k2,k3=st.columns(3); k1.metric("Positions acquired",len(batch)); k2.metric("Total gross counts",f"{int(batch['Gross counts'].sum()):,}"); k3.metric("Mean estimated activity",f"{batch['Estimated activity at count (Bq)'].mean():,.1f} Bq")
                bfig=px.bar(batch,x="Sample ID",y="Estimated activity at count (Bq)",color="Detector",title="Batch result by rack position")
                bfig.update_layout(template="plotly_dark",font=dict(color="#f4f8fb"),height=360); st.plotly_chart(bfig,width="stretch",key="stage1_rack_batch_plot")
                st.dataframe(batch.drop(columns=["Timestamp","Scenario notes"]),width="stretch",hide_index=True)

        # Advanced 2048-channel spectrum acquisition
        elif experiment_nav == experiment_labels[7]:
            render_troubleshooting_case("spectroscopy")
            theory_panel('THEORY 08 / PULSE-HEIGHT SPECTROSCOPY',
                'From deposited energy to a multichannel spectrum',
                'Scintillation light is converted into electrical pulses related to deposited energy. A multichannel analyser sorts pulse heights into channels; energy calibration maps channels to keV and ROI integration estimates peak area above continuum.',
                [r'E_i=E_0+g\,i', r'\mathrm{FWHM}=2\sqrt{2\ln 2}\,\sigma_E,\quad R_E=100\frac{\mathrm{FWHM}}{E_{\mathrm{photopeak}}}', r'N_{\mathrm{net,ROI}}=N_{\mathrm{ROI}}-\widehat{N}_{\mathrm{continuum,ROI}},\quad R_{\mathrm{ROI}}=\frac{N_{\mathrm{net,ROI}}}{t_{\mathrm{live}}}'],
                'How do energy calibration, resolution and ROI width affect peak integration?',
                'The spectrum is a simplified teaching model, not a full Monte Carlo or validated instrument response.')
            st.markdown("<div class='kicker'>EXPERIMENT 08 / MULTICHANNEL ANALYSER</div><div class='section-title'>2048-channel gamma spectrum acquisition</div>",unsafe_allow_html=True)
            st.write("Generate an illustrative pulse-height spectrum with radionuclide-specific photopeaks, a Compton-like continuum, background and Poisson channel statistics. Explore energy calibration, resolution and peak-window integration.")
            sp1,sp2,sp3=st.columns(3)
            with sp1: sp_act=st.number_input("Source activity (Bq)",1.0,1e8,18000.0,500.0,key="spec_activity"); sp_time=st.number_input("Live time (s)",1.0,3600.0,120.0,5.0,key="spec_livetime")
            with sp2: sp_bg=st.number_input("Continuum background (cps)",0.0,10000.0,25.0,1.0,key="spec_background"); sp_res=st.slider("Resolution at 511 keV (% FWHM)",4.0,20.0,9.5,.5,key="spec_resolution")
            with sp3: sp_gain=st.number_input("Energy gain (keV/channel)",.1,5.0,1.5,.01,key="spec_gain"); sp_offset=st.number_input("Energy offset (keV)",-100.0,100.0,0.0,.5,key="spec_offset")
            use_energy_fit=st.checkbox("Derive energy calibration from two reference peaks",False,key="spec_energy_fit")
            if use_energy_fit:
                ec1,ec2=st.columns(2)
                with ec1:
                    ref_e1=st.number_input("Reference peak 1 energy (keV)",1.0,3000.0,140.5,.5,key="spec_ref_e1")
                    ref_ch1=st.number_input("Observed channel 1",0.0,2047.0,93.7,.1,key="spec_ref_ch1")
                with ec2:
                    ref_e2=st.number_input("Reference peak 2 energy (keV)",1.0,3000.0,511.0,.5,key="spec_ref_e2")
                    ref_ch2=st.number_input("Observed channel 2",0.0,2047.0,340.7,.1,key="spec_ref_ch2")
                if ref_ch2<=ref_ch1 or ref_e2<=ref_e1:
                    st.error("For a positive linear calibration, reference energy 2 and channel 2 must both exceed reference 1.")
                else:
                    sp_gain=(ref_e2-ref_e1)/(ref_ch2-ref_ch1)
                    sp_offset=ref_e1-sp_gain*ref_ch1
                    st.info(f"Two-point calibration: E = {sp_offset:.3f} + {sp_gain:.5f} × channel (keV).")
            spec_lines=SPECTRUM_LINES.get(radionuclide,SPECTRUM_LINES["18F"])
            visible_lines=[x for x in spec_lines if x[0] <= sp_offset+2047*sp_gain]
            target_options=[f"{e:.1f} keV · {kind}" for e,y,kind in visible_lines]
            target_idx=max(range(len(visible_lines)),key=lambda i:visible_lines[i][1]) if visible_lines else 0
            target_label=st.selectbox("Photopeak / line for ROI integration",target_options,index=target_idx if target_options else 0,key="spec_target") if target_options else None
            roi_half=st.slider("ROI half-width (keV)",2.0,80.0,20.0,1.0,key="spec_roi_width")
            if st.button("▶ ACQUIRE 2048-CHANNEL SPECTRUM",type="primary",width="stretch",key="run_spectrum") and (not use_energy_fit or (ref_ch2>ref_ch1 and ref_e2>ref_e1)):
                st.session_state.seed+=1
                spec=simulate_spectrum(radionuclide,sp_act,sp_time,sensitivity_global,sp_bg,sp_res,sp_gain,sp_offset,deadtime_global,seed=st.session_state.seed)
                spec.attrs["live_time_s"]=sp_time
                deadtime_loss=float(spec.attrs.get("deadtime_loss_fraction",0.0))
                target_e=float(target_label.split(" keV")[0]) if target_label else float(RADIONUCLIDES[radionuclide]["energy_keV"])
                integ=peak_integral(spec,target_e,roi_half)
                spec["ROI selected"]=(spec["Energy (keV)"]>=target_e-roi_half)&(spec["Energy (keV)"]<=target_e+roi_half)
                # Metadata repeated for portable CSV/Excel analysis.
                spec["Radionuclide"]=radionuclide; spec["Live time (s)"]=sp_time; spec["ROI centre (keV)"]=target_e
                name=f"RUN · 2048ch spectrum · {radionuclide} · {datetime.now().strftime('%H%M%S')}"
                spec_record=new_record(spec,"2048-channel gamma spectrum",radionuclide,
                    f"Synthetic 2048-channel spectrum; gain={sp_gain:.4f} keV/ch; offset={sp_offset:.2f} keV; FWHM@511={sp_res:.1f}%; ROI={target_e:.1f}±{roi_half:.1f} keV; dead-time loss={100*deadtime_loss:.2f}%.")
                st.session_state.datasets[name]=spec_record
                st.session_state.spectrum_results={"dataset":name,"centre":target_e,"half_width":roi_half,**integ,"live_time":sp_time}
                st.success("Spectrum acquired and saved as a 2048-row dataset for Stage 2 analysis.")
                ss1,ss2,ss3,ss4,ss5=st.columns(5); ss1.metric("Gross ROI counts",f"{integ['gross_peak_counts']:,.0f}"); ss2.metric("Estimated continuum",f"{integ['estimated_continuum_counts']:,.0f}"); ss3.metric("Net peak counts",f"{integ['net_peak_counts']:,.0f}"); ss4.metric("Net peak rate",f"{integ['net_peak_rate_cps']:,.2f} cps"); ss5.metric("Dead-time loss",f"{100*deadtime_loss:.2f}%")
                sf=go.Figure(); sf.add_trace(go.Scatter(x=spec["Energy (keV)"],y=spec["Counts"],mode="lines",name="Observed counts",line=dict(color="#55E0B0",width=1.8)))
                sf.add_vrect(x0=target_e-roi_half,x1=target_e+roi_half,fillcolor="#F4B942",opacity=.2,line_width=0)
                sf.update_layout(template="plotly_dark",font=dict(color="#f4f8fb"),height=430,title=f"{radionuclide} simulated pulse-height spectrum",xaxis_title="Energy (keV)",yaxis_title="Counts / channel")
                st.plotly_chart(sf,width="stretch",key="stage1_spectroscopy_roi_plot")
                render_math(r"N_{\mathrm{net,ROI}}=N_{\mathrm{gross,ROI}}-\widehat{N}_{\mathrm{continuum,ROI}},\qquad R_{\mathrm{peak}}=\frac{N_{\mathrm{net,ROI}}}{t_{\mathrm{live}}}")
                st.caption("This spectrum is a pedagogical model, not a detector-response simulation validated for a specific counter. Peak yields, continuum and energy resolution are illustrative.")

        # PET/SPECT ROI summary import
        elif experiment_nav == experiment_labels[8]:
            render_troubleshooting_case("import")
            theory_panel('THEORY 09 / DATA IMPORT & STANDARDISATION',
                'Make external datasets comparable before analysis',
                'Imported data may have different units, time origins, ROI labels and uncertainty fields. Standardisation converts units and decay-normalises to a common reference time.',
                ['A(t_0)=A(t)2^{(t-t_0)/T_{1/2}}', r'C_{\mathrm{Bq/mL}}=10^3C_{\mathrm{kBq/mL}}=10^6C_{\mathrm{MBq/mL}}', r'C_{\gamma}=\frac{A_{\mathrm{sample}}}{m/\rho}'],
                'Are units, time stamps, sample labels and mass/volume definitions explicit?',
                'Missing metadata are not inferred; verify all field mappings and units before import.')
            st.markdown("<div class='kicker'>EXPERIMENT 09 / IN-VIVO IMAGE DATA</div><div class='section-title'>Import PET or SPECT ROI measurements</div>",unsafe_allow_html=True)
            st.write("Import a CSV exported from your image-analysis workflow. Use ROI summary measurements (e.g. mean activity concentration, volume and acquisition time); this module does not directly parse DICOM or reconstruct image volumes.")
            template_roi=pd.DataFrame({"ROI":["Tumour","Liver","Kidney","Muscle"],"Mean concentration (Bq/mL)":[18500,9200,14100,2100],"Concentration SD (Bq/mL)":[1200,650,900,250],"ROI volume (mL)":[0.42,1.8,0.31,0.65],"Time from reference (min)":[45,45,45,45],"Recovery coefficient":[0.82,0.95,0.91,0.98]})
            st.download_button("Download example PET/SPECT ROI CSV template",template_roi.to_csv(index=False).encode(),"pet_spect_roi_template.csv","text/csv",key="roi_template")
            uploaded=st.file_uploader("Upload PET/SPECT ROI summary CSV",type=["csv"],key="modality_csv_upload")
            if uploaded is not None:
                try:
                    raw_roi=pd.read_csv(uploaded)
                    if raw_roi.empty or len(raw_roi.columns)==0: st.error("The uploaded CSV contains no data.")
                    else:
                        st.dataframe(raw_roi.head(8),width="stretch",hide_index=True)
                        cols=list(raw_roi.columns)
                        im1,im2,im3=st.columns(3)
                        with im1: modality=st.selectbox("Modality",["PET","SPECT"],key="import_modality"); roi_col=st.selectbox("ROI / tissue label",["(row number)"]+cols,index=roi_default,key="import_roi_col")
                        conc_default=next((i for i,c in enumerate(cols) if "concentration" in c.lower()),0)
                        roi_default=next((i+1 for i,c in enumerate(cols) if c.lower() in ["roi","organ","tissue","region"]),0)
                        time_default=next((i+1 for i,c in enumerate(cols) if "time" in c.lower()),0)
                        vol_default=next((i+1 for i,c in enumerate(cols) if "volume" in c.lower()),0)
                        sd_default=next((i+1 for i,c in enumerate(cols) if "sd" in c.lower() or "uncert" in c.lower()),0)
                        rc_default=next((i+1 for i,c in enumerate(cols) if "recovery" in c.lower() or c.lower()=="rc"),0)
                        with im2: conc_col=st.selectbox("Activity concentration column",cols,index=conc_default,key="import_conc_col"); units=st.selectbox("Concentration units",["Bq/mL","kBq/mL","MBq/mL"],key="import_conc_units")
                        with im3: time_col=st.selectbox("Time from reference (min)",["(all zero)"]+cols,index=time_default,key="import_time_col"); vol_col=st.selectbox("ROI volume (mL), optional",["(not supplied)"]+cols,index=vol_default,key="import_vol_col")
                        im4,im5,im6=st.columns(3)
                        with im4: sd_col=st.selectbox("Concentration SD column, optional",["(not supplied)"]+cols,index=sd_default,key="import_sd_col")
                        with im5: rc_col=st.selectbox("Recovery coefficient column, optional",["(not supplied)"]+cols,index=rc_default,key="import_rc_col")
                        with im6: apply_rc=st.checkbox("Apply recovery coefficient correction",False,key="import_apply_rc")
                        common_ref=st.number_input("Common reference time offset (min)",0.0,100000.0,0.0,1.0,key="import_ref_time")
                        if st.button("IMPORT AND NORMALISE ROI DATA",type="primary",width="stretch",key="do_import_roi"):
                            half=float(RADIONUCLIDES[radionuclide]["half_life_min"])
                            scale={"Bq/mL":1.0,"kBq/mL":1e3,"MBq/mL":1e6}[units]
                            imported=[]
                            for idx,row0 in raw_roi.iterrows():
                                conc=float(row0[conc_col])*scale
                                t=float(row0[time_col]) if time_col!="(all zero)" and pd.notna(row0[time_col]) else 0.0
                                # Correct measured concentration to the requested common reference time.
                                dt=t-common_ref
                                factor=1.0/decay_factor(dt,half) if dt>=0 else decay_factor(-dt,half)
                                rc=float(row0[rc_col]) if rc_col!="(not supplied)" and pd.notna(row0[rc_col]) else 1.0
                                if rc<=0 or rc>1.5: raise ValueError(f"Invalid recovery coefficient in row {idx+1}: expected 0 < RC <= 1.5.")
                                corrected=conc*factor/(rc if apply_rc else 1.0)
                                label=str(row0[roi_col]) if roi_col!="(row number)" else f"ROI-{idx+1:03d}"
                                item={"Modality":modality,"ROI":label,"Concentration measured (Bq/mL)":conc,"Time from reference (min)":t,
                                      "Common reference time (min)":common_ref,"Decay-corrected concentration (Bq/mL)":conc*factor,
                                      "Recovery coefficient":rc,"RC correction applied":bool(apply_rc),"Reference concentration (Bq/mL)":corrected}
                                if vol_col!="(not supplied)" and pd.notna(row0[vol_col]): item["ROI volume (mL)"]=float(row0[vol_col])
                                if sd_col!="(not supplied)" and pd.notna(row0[sd_col]): item["Concentration SD (Bq/mL)"]=float(row0[sd_col])*scale*factor/(rc if apply_rc else 1.0)
                                imported.append(item)
                            imported_df=new_record(pd.DataFrame(imported),f"Imported {modality} ROI summary",radionuclide,
                                "User-supplied CSV; units converted to Bq/mL and decay-normalised to the selected common reference time. RC correction only applied when selected.")
                            keyname=f"IMPORT · {modality} ROIs · {datetime.now().strftime('%H%M%S')}"
                            st.session_state.datasets[keyname]=imported_df
                            st.success(f"Imported {len(imported_df)} ROI measurements. The normalised dataset is available in Stage 2.")
                            st.dataframe(imported_df,width="stretch",hide_index=True)
                            st.download_button("Download normalised ROI dataset",imported_df.to_csv(index=False).encode(),"normalised_pet_spect_roi.csv","text/csv",key="download_imported_roi")
                except Exception as exc:
                    st.error(f"Could not process this CSV: {exc}")

            st.markdown("---")
            st.markdown("#### Import ex-vivo gamma-counter tissue measurements")
            st.write("You can also import real gamma-counter tissue results. The app standardises these to the same fields used by the PET/SPECT agreement workflow. Activity values or net count rates can be supplied; net count rates are converted using the radionuclide-specific sensitivity entered below.")
            gamma_template=pd.DataFrame({"Tissue":["Tumour","Liver","Kidney","Muscle"],"Activity at count (Bq)":[42000,58000,22000,3500],"Mass (g)":[0.24,1.18,0.29,0.61],"Elapsed time (min)":[45,45,45,45]})
            st.download_button("Download example gamma tissue CSV template",gamma_template.to_csv(index=False).encode(),"gamma_tissue_template.csv","text/csv",key="gamma_tissue_template")
            gamma_upload=st.file_uploader("Upload ex-vivo gamma tissue CSV",type=["csv"],key="gamma_tissue_csv_upload")
            if gamma_upload is not None:
                try:
                    graw=pd.read_csv(gamma_upload)
                    st.dataframe(graw.head(8),width="stretch",hide_index=True)
                    gcols=list(graw.columns)
                    numeric_cols=[c for c in gcols if pd.api.types.is_numeric_dtype(graw[c])]
                    if not numeric_cols:
                        st.error("The gamma-counter CSV needs at least one numeric measurement column.")
                    else:
                        gid_default=next((i for i,c in enumerate(gcols) if c.lower() in ["tissue","organ","sample id","sample","roi"]),0)
                        gact_default=next((i for i,c in enumerate(numeric_cols) if "activity" in c.lower()),0)
                        gmass_default=next((i for i,c in enumerate(gcols) if "mass" in c.lower()),0)
                        gtime_default=next((i+1 for i,c in enumerate(gcols) if "elapsed" in c.lower() or "time from" in c.lower()),0)
                        gi1,gi2,gi3=st.columns(3)
                        with gi1: glabel=st.selectbox("Tissue / sample label",["(row number)"]+gcols,index=gid_default+1 if gid_default>=0 else 0,key="gimport_label")
                        with gi2: gbasis=st.selectbox("Measurement type",["Activity at count time","Net count rate"],key="gimport_basis")
                        with gi3: gvalue=st.selectbox("Measurement column",numeric_cols,index=gact_default,key="gimport_value")
                        gi4,gi5,gi6=st.columns(3)
                        with gi4: gmass=st.selectbox("Tissue mass column (g)",gcols,index=gmass_default,key="gimport_mass")
                        with gi5: gtime=st.selectbox("Elapsed time column (min)",["(reference time)"]+gcols,index=gtime_default,key="gimport_time")
                        with gi6: gmass_unit=st.selectbox("Mass unit",["g","mg"],key="gimport_mass_unit")
                        gi7,gi8=st.columns(2)
                        with gi7: gunit=st.selectbox("Activity unit",["Bq","kBq","MBq"],key="gimport_activity_unit")
                        with gi8: gsens=st.number_input("Sensitivity for net-cps conversion (cps/Bq)",.000001,10.0,float(sensitivity_global),format="%.6f",key="gimport_sensitivity")
                        if st.button("IMPORT AND STANDARDISE GAMMA TISSUE DATA",type="primary",width="stretch",key="do_import_gamma"):
                            scale_activity={"Bq":1.0,"kBq":1e3,"MBq":1e6}[gunit]
                            scale_mass=1.0 if gmass_unit=="g" else 0.001
                            grow=[]
                            for idx,rr in graw.iterrows():
                                val=float(rr[gvalue])
                                act=val*scale_activity if gbasis=="Activity at count time" else (val/max(gsens,1e-12))
                                mass_g=float(rr[gmass])*scale_mass
                                if mass_g<=0 or act<0:
                                    raise ValueError(f"Invalid activity or mass in row {idx+1}; activity must be non-negative and mass must be positive.")
                                age=float(rr[gtime]) if gtime!="(reference time)" and pd.notna(rr[gtime]) else 0.0
                                label=str(rr[glabel]) if glabel!="(row number)" else f"Sample-{idx+1:03d}"
                                grow.append({"Sample ID":label,"Tissue":label,"Radionuclide":radionuclide,
                                             "Estimated activity at count (Bq)":act,"Mass (g)":mass_g,"Elapsed time (min)":age,
                                             "Activity input basis":gbasis,"Sensitivity used (cps/Bq)":gsens if gbasis=="Net count rate" else np.nan})
                            gimported=new_record(pd.DataFrame(grow),"Imported gamma-counter tissue summary",radionuclide,
                                "User-supplied ex-vivo gamma-counter results standardised to Bq, g and elapsed minutes. Net cps converted using entered sensitivity when selected.")
                            gname=f"IMPORT · Gamma tissue · {radionuclide} · {datetime.now().strftime('%H%M%S')}"
                            st.session_state.datasets[gname]=gimported
                            st.success(f"Imported {len(gimported)} gamma-counter tissue measurements. This dataset is available for PET/SPECT agreement analysis.")
                            st.dataframe(gimported,width="stretch",hide_index=True)
                            st.download_button("Download standardised gamma tissue dataset",gimported.to_csv(index=False).encode(),"standardised_gamma_tissue.csv","text/csv",key="download_gamma_import")
                except Exception as exc:
                    st.error(f"Could not process the gamma-counter CSV: {exc}")

        # Live acquisition playback: spectrum and count-rate buildup
        elif experiment_nav == experiment_labels[9]:
            st.markdown("<div class='kicker'>THEORY-LED WORKBENCH / SHARED ACQUISITION ENGINE</div><div class='section-title'>Select a practical task, study its theory, then acquire</div>", unsafe_allow_html=True)
            st.markdown("<div class='kicker'>EXPERIMENT 10 / LIVE PULSE-HEIGHT ACQUISITION</div><div class='section-title'>Watch the gamma counter build a spectrum</div>", unsafe_allow_html=True)
            st.write("This is the shared acquisition workbench for the Stage 1 exercises. Select the laboratory task, read its theory and equations below the selector, configure the sample and detector, then acquire the spectrum. The selected task also generates its own calibration, normalisation, geometry, tissue, rack or PET/SPECT comparison dataset for Stage 2.")
            live_scenario = st.selectbox(
                "Choose the Stage 1 laboratory task for this acquisition",
                ["Unknown sample assay", "Calibration standards sequence", "Detector normalisation",
                 "Geometry and matrix challenge", "Hypothetical tissue study", "PET/SPECT cross-calibration",
                 "Automated sample rack", "Gamma spectroscopy and ROI analysis",
                 "Background subtraction", "Dead-time behaviour", "Decay during acquisition", "Counting precision"],
                key="live_scenario",
                help="The same live spectrum engine is reused for each task. The selected task creates its own dedicated measurement dataset for Stage 2."
            )
            scenario_help = {
                "Unknown sample assay": "Measure an unknown sample, subtract an independently counted background and estimate activity using the selected calibration sensitivity and response corrections.",
                "Calibration standards sequence": "Acquire a series of known-activity standards, fit net count rate versus activity, and save the fitted sensitivity and residual data.",
                "Detector normalisation": "Measure one reference source across all six detector channels, derive relative correction factors and compare raw with normalised response.",
                "Geometry and matrix challenge": "Repeat the same nominal activity across tubes/vials and matrix-response conditions to see how cps/Bq changes with sample presentation.",
                "Hypothetical tissue study": "Generate a matched multi-tissue dataset for Control, Tumour-enriched or Tumour-reduced uptake, with count statistics, activity, %ID and %ID/g.",
                "PET/SPECT cross-calibration": "Create matched image-derived ROI concentration and ex-vivo gamma tissue activity values, decay-normalised to a common reference time.",
                "Automated sample rack": "Simulate sequential acquisition of a labelled multi-position rack and save each sample result with its position and detector assignment.",
                "Gamma spectroscopy and ROI analysis": "Focus on the isotope-specific 2048-channel spectrum, energy calibration, FWHM, photopeak ROI and continuum-subtracted peak area.",
                "Background subtraction": "Visualise gross, background and net sample counts accumulating over time, including counting uncertainty.",
                "Dead-time behaviour": "Compare true and observed count rates as activity increases and show the operating point for this sample.",
                "Decay during acquisition": "Show the expected fall in sample activity and count rate during the acquisition interval.",
                "Counting precision": "Compare relative Poisson uncertainty for short and long counting times, with and without a background measurement.",
            }
            st.info(scenario_help[live_scenario])
            live_theory = {
                "Unknown sample assay": ("Quantify an unknown sample", [r"N_{net}=N_{gross}-N_{bg}", r"A_{est}=\frac{R_{net,corr}}{S_{cal}G_{det}G_{geom}F_{matrix}}", r"u(N_{net})=\sqrt{N_{gross}+N_{bg}}"], "How close is the measured activity estimate to the teaching ground truth?"),
                "Calibration standards sequence": ("Establish absolute detector sensitivity", [r"R_{net,i}=S_{cal}A_i+b", r"S_{cal}=\frac{\mathrm{d}R_{net}}{\mathrm{d}A}", r"A_{unknown}=\frac{R_{unknown}-b}{S_{cal}}"], "Does the standard series support a linear cps/Bq response?"),
                "Detector normalisation": ("Correct relative detector-to-detector response", [r"S_j=\frac{R_{net,j}}{A_{ref}}", r"F_j=\frac{\bar S}{S_j}", r"R_{norm,j}=F_jR_{net,j}"], "Do six detectors converge to a common relative sensitivity after correction?"),
                "Geometry and matrix challenge": ("Test sample-presentation dependence", [r"R_{net}=S_{cal}A G_{geom}F_{matrix}", r"A_{corr}=\frac{R_{net}}{S_{cal}G_{geom}F_{matrix}}"], "How much apparent activity bias is introduced by vial geometry or matrix?"),
                "Hypothetical tissue study": ("Translate tissue counts into biodistribution", [r"A(t_0)=A(t)e^{\lambda(t-t_0)}", r"\%ID=100\frac{A_{tissue}(t_0)}{A_{injected}(t_0)}", r"\%ID/g=\frac{\%ID}{m_{tissue}}"], "Does a tumour uptake shift change tumour-to-liver %ID/g?"),
                "PET/SPECT cross-calibration": ("Compare reconstructed concentration with ex-vivo activity", [r"C_{gamma}=\frac{A_{gamma}}{V_{sample}}", r"C_{image,corr}=\frac{C_{image}}{RC}", r"\Delta C=C_{image}-C_{gamma}"], "Is the modality difference consistent with the selected image bias?"),
                "Automated sample rack": ("Acquire a sequence of identified samples", [r"N_i\sim\mathrm{Poisson}(R_i t_i)", r"R_{net,i}=\frac{N_{sample,i}}{t_{sample}}-\frac{N_{bg}}{t_{bg}}"], "Can sample position and detector assignment explain measured variability?"),
                "Gamma spectroscopy and ROI analysis": ("Interpret the pulse-height distribution", [r"E_i=E_0+g i", r"FWHM=2\sqrt{2\ln2}\,\sigma_E", r"N_{net,ROI}=N_{ROI}-\widehat N_{continuum}"], "How do resolution and ROI width affect net peak counts?"),
                "Background subtraction": ("Separate sample signal from ambient counts", [r"R_{net}=R_{gross}-R_{bg}", r"\sigma_{R_{net}}=\sqrt{\frac{N_g}{t_g^2}+\frac{N_b}{t_b^2}}"], "How does increasing background change the uncertainty of the net result?"),
                "Dead-time behaviour": ("Understand count losses at high event rates", [r"R_{obs}=\frac{R_{true}}{1+R_{true}\tau}", r"R_{true}=\frac{R_{obs}}{1-R_{obs}\tau}"], "At what operating rate does dead-time loss become consequential?"),
                "Decay during acquisition": ("Integrate activity over a finite counting interval", [r"A(t)=A_0e^{-\lambda t}", r"N_s(T)=\epsilon S A_0\frac{1-e^{-\lambda T}}{\lambda}"], "How much does the mean event rate change during the selected acquisition?"),
                "Counting precision": ("Relate counting time to statistical uncertainty", [r"\frac{\sigma_N}{N}\approx\frac{1}{\sqrt{N}}", r"\frac{\sigma_{R_{net}}}{R_{net}}=\frac{\sqrt{R_s+2R_b}}{R_s\sqrt{t}}"], "How much longer must you count to halve relative uncertainty?"),
            }
            live_title, live_equations, live_question = live_theory[live_scenario]
            # Use the fully typeset, notation-audited equations from LIVE_LESSONS.
            if live_scenario in LIVE_LESSONS:
                live_equations = LIVE_LESSONS[live_scenario][2]
            theory_panel("LIVE PRACTICAL THEORY / " + live_scenario.upper(), live_title,
                         scenario_help[live_scenario], live_equations, live_question,
                         "The simulator uses synthetic Poisson counts and idealised calibration/response factors. Interpret this as a teaching model, not a validated instrument protocol.")
            live_case_key = {
                "Unknown sample assay":"single", "Calibration standards sequence":"calibration",
                "Detector normalisation":"normalisation", "Geometry and matrix challenge":"geometry",
                "Hypothetical tissue study":"tissue", "PET/SPECT cross-calibration":"pet",
                "Automated sample rack":"rack", "Gamma spectroscopy and ROI analysis":"spectroscopy",
                "Background subtraction":"single", "Dead-time behaviour":"calibration",
                "Decay during acquisition":"tissue", "Counting precision":"single"
            }.get(live_scenario, "single")
            render_troubleshooting_case(live_case_key)

            live1, live2, live3 = st.columns(3)
            with live1:
                live_sample_id = st.text_input("Sample ID", "LIVE-001", key="live_sample_id")
                live_activity = st.number_input("Sample activity at count start (Bq)", 1.0, 2.0e6, 18000.0, 500.0, key="live_activity")
            with live2:
                live_time = st.slider("Acquisition live time (s)", 5, 3600, 60, 5, key="live_time")
                st.caption("Longer durations are available for decay demonstrations. A 60 s acquisition will show almost no physical decay for long-lived radionuclides such as 89Zr; a longer 18F acquisition makes the interval-rate trend easier to see.")
                live_bg = st.number_input("Continuum/background (cps)", 0.0, 5000.0, float(bg_global), 1.0, key="live_bg")
            with live3:
                live_res = st.slider("Detector resolution at 511 keV (% FWHM)", 4.0, 20.0, 9.5, 0.5, key="live_res")
                live_frames = st.slider("Playback resolution (frames)", 12, 72, 40, 4, key="live_frames")

            live_gain_col, live_offset_col, live_geom_col = st.columns(3)
            with live_gain_col:
                live_gain = st.number_input("Energy gain (keV/channel)", 0.1, 5.0, 1.5, 0.01, key="live_gain")
            with live_offset_col:
                live_offset = st.number_input("Energy offset (keV)", -100.0, 100.0, 0.0, 0.5, key="live_offset")
            with live_geom_col:
                live_geom = st.slider("Relative geometry response", 0.5, 1.1, 1.0, 0.01, key="live_geom")

            view_a, view_b, view_c = st.columns(3)
            with view_a:
                live_y_scale = st.selectbox("Spectrum intensity scale", ["Linear", "Logarithmic"], key="live_y_scale")
            with view_b:
                live_energy_view = st.selectbox(
                    "Energy display window",
                    ["Full 2048-channel spectrum", "Primary photopeak focus", "All listed isotope lines"],
                    key="live_energy_view",
                )
            with view_c:
                live_playback_ms = st.slider("Playback speed (ms/frame)", 60, 600, 220, 20, key="live_playback_ms")

            # Extra controls are exposed only for the selected task, while the same live acquisition engine remains common.
            if live_scenario == "Calibration standards sequence":
                cal_a, cal_b = st.columns(2)
                with cal_a: live_cal_points = st.slider("Number of standards", 4, 10, 6, key="live_cal_points")
                with cal_b: live_cal_max = st.number_input("Maximum standard activity (Bq)", 1000.0, 1e7, 50000.0, 1000.0, key="live_cal_max")
            elif live_scenario == "Geometry and matrix challenge":
                st.caption("The common sample activity is counted under multiple geometry/matrix responses.")
                live_geometry_set = st.multiselect("Geometry conditions", ["Small tube", "Standard vial", "Large vial", "High fill height"], default=["Small tube", "Standard vial", "Large vial", "High fill height"], key="live_geometry_set")
            elif live_scenario == "Hypothetical tissue study":
                t_a, t_b, t_c = st.columns(3)
                with t_a: live_injected = st.number_input("Injected activity (Bq)", 1000.0, 1e9, 2.5e6, 1e4, key="live_tissue_injected")
                with t_b: live_collection = st.number_input("Collection time after injection (min)", 0.0, 100000.0, 90.0, 5.0, key="live_tissue_collection")
                with t_c: live_arm = st.selectbox("Study arm", ["Control", "Tumour-enriched", "Tumour-reduced"], key="live_tissue_arm")
                live_fold_a, live_fold_b = st.columns(2)
                with live_fold_a: live_enrich_fold = st.slider("Tumour-enriched multiplier", 1.05, 3.00, 1.50, 0.05, key="live_arm_enrich_fold")
                with live_fold_b: live_reduce_fold = st.slider("Tumour-reduced multiplier", 0.10, 0.95, 0.65, 0.05, key="live_arm_reduce_fold")
            elif live_scenario == "PET/SPECT cross-calibration":
                p_a, p_b = st.columns(2)
                with p_a: live_phantom_volume = st.number_input("Phantom / tissue volume (mL)", 0.1, 1000.0, 5.0, 0.1, key="live_phantom_volume")
                with p_b: live_pet_bias = st.slider("Simulated image bias (%)", -20.0, 20.0, 3.0, 0.5, key="live_pet_bias")
            elif live_scenario == "Automated sample rack":
                r_a, r_b = st.columns(2)
                with r_a: live_rack_n = st.selectbox("Rack positions", [6, 10, 12, 20], index=1, key="live_rack_n")
                with r_b: live_rack_min = st.number_input("Minimum sample activity (Bq)", 100.0, 1e6, 2000.0, 100.0, key="live_rack_min")

            live_roi = float(RADIONUCLIDES[radionuclide]["energy_keV"])
            st.markdown(
                f"<div class='readout'>READY  |  SAMPLE {live_sample_id or 'UNSPECIFIED'}  |  {radionuclide}  |  DETECTOR {detector}  |  TARGET LINE {live_roi:.1f} keV</div>",
                unsafe_allow_html=True,
            )
            st.caption("The simulator generates Poisson counts using the activity integrated over the acquisition interval. The cumulative-count panel expands its elapsed-time and count axes as acquisition proceeds, keeping the early build-up visible. The lower panel calculates CPS separately in each time bin (bin counts divided by bin duration), shows Poisson error bars, and overlays the expected interval-rate trend. For long-lived isotopes over short acquisitions, the expected CPS is correctly almost flat. The final animation frame exactly matches the saved 2048-channel dataset. This is a simulated acquisition, not a hardware-connected detector.")

            if st.button("▶ START MEASUREMENT & BUILD LIVE SPECTRUM", type="primary", width="stretch", key="start_live_acq"):
                if not live_sample_id.strip():
                    st.error("Please enter a sample identifier.")
                else:
                    st.session_state.seed += 1
                    live_detector_gain = float(DETECTOR_REL[detector])
                    live_sensitivity = float(sensitivity_global) * live_detector_gain * float(live_geom)
                    live_spec = simulate_spectrum(
                        radionuclide, live_activity, float(live_time), live_sensitivity,
                        float(live_bg), float(live_res), float(live_gain), float(live_offset),
                        float(deadtime_global), seed=int(st.session_state.seed),
                        half_life_min=float(RADIONUCLIDES[radionuclide]["half_life_min"]),
                    )
                    live_spec.attrs["live_time_s"] = float(live_time)
                    live_spec.attrs["deadtime_loss_fraction"] = float(live_spec.attrs.get("deadtime_loss_fraction", 0.0))
                    # Save a portable channel-by-channel dataset with measurement metadata.
                    live_spec["Radionuclide"] = radionuclide
                    live_spec["Live time (s)"] = float(live_time)
                    live_spec["Detector"] = detector
                    live_spec["Sample ID"] = live_sample_id.strip()
                    live_spec["Sample activity at count start (Bq) [teaching only]"] = float(live_activity)
                    live_spec["Geometry response"] = float(live_geom)
                    live_spec["Background setting (cps)"] = float(live_bg)
                    live_spec["Resolution at 511 keV (% FWHM)"] = float(live_res)
                    live_spec["Radionuclide half-life (min)"] = float(RADIONUCLIDES[radionuclide]["half_life_min"])
                    live_spec["Average activity fraction during acquisition"] = float(live_spec.attrs.get("average_activity_fraction", 1.0))

                    # Peak integration is performed around the radionuclide's dominant listed line.
                    available_lines = SPECTRUM_LINES.get(radionuclide, [(live_roi, 1.0, "gamma")])
                    dominant_line = max(available_lines, key=lambda item: item[1])[0]
                    roi_half_width = max(10.0, 0.045 * dominant_line)
                    live_integral = peak_integral(live_spec, float(dominant_line), float(roi_half_width))
                    energy_min = float(live_spec["Energy (keV)"].min())
                    energy_max = float(live_spec["Energy (keV)"].max())
                    fwhm_kev = float(live_res) / 100.0 * 511.0 * math.sqrt(max(float(dominant_line), 1.0) / 511.0)
                    energy_window = None
                    if live_energy_view == "Primary photopeak focus":
                        energy_window = (max(energy_min, dominant_line - 4.0 * fwhm_kev),
                                         min(energy_max, dominant_line + 4.0 * fwhm_kev))
                    elif live_energy_view == "All listed isotope lines":
                        visible_lines = [line[0] for line in available_lines if energy_min <= line[0] <= energy_max]
                        if visible_lines:
                            pad = max(35.0, 0.08 * (max(visible_lines) - min(visible_lines) + 1.0))
                            energy_window = (max(energy_min, min(visible_lines) - pad),
                                             min(energy_max, max(visible_lines) + pad))
                    live_fig = make_live_acquisition_figure(
                        live_spec, float(live_time), int(live_frames),
                        y_scale=live_y_scale, energy_range=energy_window,
                        frame_duration_ms=int(live_playback_ms), dominant_line=float(dominant_line),
                        isotope_lines=available_lines,
                        half_life_min=float(RADIONUCLIDES[radionuclide]["half_life_min"]),
                    )
                    live_spec["ROI selected"] = (
                        (live_spec["Energy (keV)"] >= dominant_line - roi_half_width) &
                        (live_spec["Energy (keV)"] <= dominant_line + roi_half_width)
                    )
                    live_dataset = new_record(
                        live_spec, "Live 2048-channel acquisition", radionuclide,
                        f"Time-resolved Poisson acquisition playback; detector={detector}; sample={live_sample_id}; "
                        f"live time={live_time:.1f} s; energy calibration={live_offset:.2f}+{live_gain:.5f}×channel; "
                        f"geometry response={live_geom:.3f}; FWHM@511={live_res:.1f}%."
                    )
                    live_dataset_name = f"RUN · LIVE spectrum · {live_sample_id} · {datetime.now().strftime('%H%M%S')}"
                    st.session_state.datasets[live_dataset_name] = live_dataset

                    scenario_norm_summary = None
                    if live_scenario == "Detector normalisation":
                        norm_rng = np.random.default_rng(int(st.session_state.seed) + 90210)
                        norm_rows = []
                        for det_name, det_gain in DETECTOR_REL.items():
                            det_result = acquire(
                                float(live_activity), float(sensitivity_global), float(live_time),
                                float(live_bg), float(deadtime_global), gain=float(det_gain),
                                seed=int(norm_rng.integers(1, 2**31 - 1)),
                            )
                            det_sens = det_result["net_cps_corrected"] / max(float(live_activity), 1e-12)
                            norm_rows.append({
                                "Detector": det_name,
                                "Reference activity (Bq)": float(live_activity),
                                "Count time (s)": float(live_time),
                                "Gross counts": int(det_result["gross_counts"]),
                                "Background counts": int(det_result["background_counts"]),
                                "Net corrected rate (cps)": float(det_result["net_cps_corrected"]),
                                "Rate uncertainty (cps)": float(det_result["sigma_net_cps"]),
                                "Observed sensitivity (cps/Bq)": float(det_sens),
                            })
                        scenario_norm_summary = pd.DataFrame(norm_rows)
                        fleet_sensitivity = float(scenario_norm_summary["Observed sensitivity (cps/Bq)"].mean())
                        scenario_norm_summary["Normalisation factor"] = fleet_sensitivity / scenario_norm_summary["Observed sensitivity (cps/Bq)"].clip(lower=1e-12)
                        scenario_norm_summary["Normalised sensitivity (cps/Bq)"] = (
                            scenario_norm_summary["Observed sensitivity (cps/Bq)"] * scenario_norm_summary["Normalisation factor"]
                        )
                        scenario_norm_summary["Normalised equivalent rate (cps)"] = (
                            scenario_norm_summary["Net corrected rate (cps)"] * scenario_norm_summary["Normalisation factor"]
                        )
                        norm_saved = new_record(
                            scenario_norm_summary, "Live detector normalisation", radionuclide,
                            "One reference-source acquisition per detector; correction factors map each measured detector sensitivity to the six-detector fleet mean. Relative normalisation only; absolute calibration remains separate."
                        )
                        norm_saved_name = f"RUN · LIVE normalisation · {live_sample_id.strip()} · {datetime.now().strftime('%H%M%S')}"
                        st.session_state.datasets[norm_saved_name] = norm_saved

                    # Task-specific measured datasets created from the same universal acquisition setup.
                    task_rng = np.random.default_rng(int(st.session_state.seed) + 55123)
                    if live_scenario == "Unknown sample assay":
                        assay = acquire(float(live_activity), float(sensitivity_global), float(live_time),
                                        float(live_bg), float(deadtime_global), gain=live_detector_gain,
                                        geom=float(live_geom), seed=int(task_rng.integers(1, 2**31-1)))
                        assay_row = {"Sample ID": live_sample_id.strip(), "Radionuclide": radionuclide,
                                     "Reference / count-start activity (Bq) [teaching only]": float(live_activity),
                                     "Gross counts": assay["gross_counts"], "Background counts": assay["background_counts"],
                                     "Net rate (cps)": assay["net_cps_corrected"], "Net rate SD (cps)": assay["sigma_net_cps"],
                                     "Sensitivity (cps/Bq)": float(sensitivity_global),
                                     "Estimated activity (Bq)": assay["net_cps_corrected"] / max(float(sensitivity_global)*live_detector_gain*float(live_geom),1e-12),
                                     "Count time (s)": float(live_time), "Detector": detector, "Geometry response": float(live_geom)}
                        ass_df = new_record(pd.DataFrame([assay_row]), "Live unknown-sample assay", radionuclide,
                                            "Generated by the universal live acquisition workbench; activity estimator uses the selected sensitivity and detector/geometry response.")
                        st.session_state.datasets[f"RUN · Live unknown assay · {live_sample_id} · {datetime.now().strftime('%H%M%S')}"] = ass_df

                    elif live_scenario == "Calibration standards sequence":
                        acts = np.geomspace(max(100.0, float(live_activity)/20.0), max(float(live_cal_max), float(live_activity)), int(live_cal_points))
                        cal_rows = []
                        for k, act in enumerate(acts, 1):
                            rr = acquire(float(act), float(sensitivity_global), float(live_time), float(live_bg),
                                         float(deadtime_global), gain=live_detector_gain, geom=float(live_geom),
                                         seed=int(task_rng.integers(1, 2**31-1)))
                            cal_rows.append({"Standard": f"LIVE-STD-{k:02d}", "Known activity (Bq)": float(act),
                                             "Gross counts": rr["gross_counts"], "Background counts": rr["background_counts"],
                                             "Net rate (cps)": rr["net_cps_corrected"], "Net rate SD (cps)": rr["sigma_net_cps"],
                                             "Count time (s)": float(live_time), "Detector": detector,
                                             "Geometry response": float(live_geom)})
                        cal_live_df = new_record(pd.DataFrame(cal_rows), "Live calibration standards sequence", radionuclide,
                                                 "Multiple known activity standards acquired using the same universal live-workbench detector and geometry settings.")
                        cal_live_name = f"RUN · LIVE calibration · {radionuclide} · {datetime.now().strftime('%H%M%S')}"
                        st.session_state.datasets[cal_live_name] = cal_live_df
                        cfit = np.polyfit(cal_live_df["Known activity (Bq)"], cal_live_df["Net rate (cps)"], 1)
                        st.session_state.calibrated_sensitivities[radionuclide] = float(cfit[0])

                    elif live_scenario == "Geometry and matrix challenge":
                        geometry_factors = {"Small tube": 1.00, "Standard vial": 0.96, "Large vial": 0.90, "High fill height": 0.84}
                        geom_rows = []
                        for condition in live_geometry_set:
                            gf = geometry_factors[condition]
                            rr = acquire(float(live_activity), float(sensitivity_global), float(live_time), float(live_bg),
                                         float(deadtime_global), gain=live_detector_gain, geom=gf,
                                         matrix=1.0 if condition == "Small tube" else 0.97,
                                         seed=int(task_rng.integers(1, 2**31-1)))
                            mf = 1.0 if condition == "Small tube" else 0.97
                            geom_rows.append({"Geometry": condition, "Nominal activity (Bq)": float(live_activity),
                                              "Geometry factor (teaching truth)": gf, "Matrix factor (teaching truth)": mf,
                                              "Gross counts": rr["gross_counts"], "Background counts": rr["background_counts"],
                                              "Net rate (cps)": rr["net_cps_corrected"],
                                              "Uncorrected activity estimate (Bq)": rr["net_cps_corrected"] / max(float(sensitivity_global),1e-12),
                                              "Geometry/matrix-corrected estimate (Bq)": rr["net_cps_corrected"] / max(float(sensitivity_global)*gf*mf,1e-12)})
                        geo_live_df = new_record(pd.DataFrame(geom_rows), "Live geometry and matrix challenge", radionuclide,
                                                 "Same nominal source measured in several illustrative geometries; geometry factors are synthetic teaching inputs.")
                        st.session_state.datasets[f"RUN · LIVE geometry · {datetime.now().strftime('%H%M%S')}"] = geo_live_df

                    elif live_scenario == "Hypothetical tissue study":
                        organs = [("Blood",3.5,.42),("Liver",18.5,1.25),("Kidney",11.5,.31),("Tumour",8.2,.24),
                                  ("Bone",4.8,.36),("Muscle",2.6,.58),("Lung",7.1,.19),("Spleen",9.2,.11)]
                        base = {n: pct for n,pct,_ in organs}; total = sum(base.values()); tumour0 = base["Tumour"]
                        fold = 1.0 if live_arm == "Control" else (float(live_enrich_fold) if live_arm == "Tumour-enriched" else float(live_reduce_fold))
                        tumour_new = min(tumour0 * fold, total); other0 = total - tumour0
                        other_scale = (total - tumour_new)/other0 if other0 > 0 else 0.0
                        fdec_t = decay_factor(float(live_collection), float(RADIONUCLIDES[radionuclide]["half_life_min"]))
                        tissue_rows = []
                        for organ, base_pct, mass_g in organs:
                            pct = tumour_new if organ == "Tumour" else base_pct*other_scale
                            activity_ref = float(live_injected)*pct/100.0
                            activity_count = activity_ref*fdec_t
                            tr = acquire(activity_count, float(sensitivity_global), float(live_time), float(live_bg),
                                         float(deadtime_global), gain=live_detector_gain,
                                         seed=int(task_rng.integers(1,2**31-1)))
                            est_count = tr["net_cps_corrected"]/max(float(sensitivity_global)*live_detector_gain,1e-12)
                            est_ref = est_count/max(fdec_t,1e-12)
                            tissue_rows.append({"Study arm":live_arm,"Tissue":organ,"Mass (g)":mass_g,
                                                "Injected activity (Bq)":float(live_injected),"Collection time (min)":float(live_collection),
                                                "True uptake (%ID) [teaching only]":pct,"Gross counts":tr["gross_counts"],
                                                "Background counts":tr["background_counts"],"Net rate (cps)":tr["net_cps_corrected"],
                                                "Estimated activity at count (Bq)":est_count,
                                                "Estimated activity at injection reference (Bq)":est_ref,
                                                "Estimated %ID":100*est_ref/float(live_injected),
                                                "Estimated %ID/g":100*est_ref/(float(live_injected)*mass_g)})
                        tissue_live_df = new_record(pd.DataFrame(tissue_rows), "Live hypothetical tissue study", radionuclide,
                                                    "Synthetic tissue-counting arm. Tumour share is modified and non-tumour tissues are proportionally rescaled to conserve the listed-tissue uptake total.")
                        st.session_state.datasets[f"RUN · LIVE tissue · {live_arm} · {datetime.now().strftime('%H%M%S')}"] = tissue_live_df

                    elif live_scenario == "PET/SPECT cross-calibration":
                        true_conc = float(live_activity)/float(live_phantom_volume)
                        paired = acquire(float(live_activity), float(sensitivity_global), float(live_time),
                                         float(live_bg), float(deadtime_global), gain=live_detector_gain,
                                         geom=float(live_geom), seed=int(task_rng.integers(1,2**31-1)))
                        gamma_activity_est = paired["net_cps_corrected"] / max(float(sensitivity_global)*live_detector_gain*float(live_geom),1e-12)
                        gamma_conc = gamma_activity_est/float(live_phantom_volume)
                        pet_conc = true_conc*(1.0+float(live_pet_bias)/100.0)*float(task_rng.normal(1.0,0.01))
                        ref_time_min = 0.0
                        image_df = new_record(pd.DataFrame([{"ROI":"Uniform phantom","Modality":"PET/SPECT",
                                                             "Concentration measured (Bq/mL)":pet_conc,
                                                             "Common reference time (min)":ref_time_min,
                                                             "Reference concentration (Bq/mL)":pet_conc,
                                                             "ROI volume (mL)":float(live_phantom_volume),
                                                             "Simulated image bias (%)":float(live_pet_bias)}]),
                                              "Live PET/SPECT phantom ROI", radionuclide,
                                              "Hypothetical image concentration with user-selected bias at the common reference time.")
                        gamma_df = new_record(pd.DataFrame([{"Tissue":"Uniform phantom","Sample ID":"Uniform phantom",
                                                              "Estimated activity at count (Bq)":gamma_activity_est,
                                                              "Mass (g)":float(live_phantom_volume)*1.05,
                                                              "Elapsed time (min)":ref_time_min,
                                                              "Gamma concentration (Bq/mL)":gamma_conc}]),
                                              "Live gamma phantom assay", radionuclide,
                                              "Ex-vivo phantom activity estimated from independent gross/background counting and converted to concentration using density 1.0 g/mL.")
                        stamp = datetime.now().strftime('%H%M%S')
                        st.session_state.datasets[f"RUN · LIVE PET ROI · {stamp}"] = image_df
                        st.session_state.datasets[f"RUN · LIVE gamma phantom · {stamp}"] = gamma_df

                    elif live_scenario == "Automated sample rack":
                        rack_rows = []
                        rack_activities = np.geomspace(float(live_rack_min), max(float(live_activity),float(live_rack_min)*2), int(live_rack_n))
                        det_names = list(DETECTOR_REL)
                        for pos, act in enumerate(rack_activities,1):
                            dname = det_names[(pos-1) % len(det_names)]
                            rr = acquire(float(act), float(sensitivity_global), float(live_time), float(live_bg),
                                         float(deadtime_global), gain=float(DETECTOR_REL[dname]), geom=float(live_geom),
                                         seed=int(task_rng.integers(1,2**31-1)))
                            rack_rows.append({"Rack position":pos,"Sample ID":f"RACK-{pos:02d}","Detector":dname,
                                              "Nominal activity (Bq) [teaching only]":float(act),"Gross counts":rr["gross_counts"],
                                              "Background counts":rr["background_counts"],"Net rate (cps)":rr["net_cps_corrected"],
                                              "Estimated activity (Bq)":rr["net_cps_corrected"]/max(float(sensitivity_global)*float(DETECTOR_REL[dname])*float(live_geom),1e-12),
                                              "Count time (s)":float(live_time)})
                        rack_df = new_record(pd.DataFrame(rack_rows), "Live automated sample rack", radionuclide,
                                             "Sequential rack positions measured using a single common acquisition configuration; detector assignment cycles through six teaching channels.")
                        st.session_state.datasets[f"RUN · LIVE rack · {datetime.now().strftime('%H%M%S')}"] = rack_df

                    total_live_counts = int(live_spec["Counts"].sum())
                    gross_rate = total_live_counts / float(live_time)
                    dead_loss = float(live_spec.attrs.get("deadtime_loss_fraction", 0.0))
                    lm1, lm2, lm3, lm4 = st.columns(4)
                    lm1.metric("Total recorded counts", f"{total_live_counts:,}")
                    lm2.metric("Mean count rate", f"{gross_rate:,.1f} cps")
                    lm3.metric("Live time", f"{live_time:.0f} s")
                    lm4.metric("Modelled dead-time loss", f"{100*dead_loss:.2f}%")

                    st.markdown("#### Live 2048-channel spectrum")
                    st.markdown(
                        f"<div class='readout'>ACQUISITION COMPLETE  |  SAMPLE {live_sample_id.strip()}  |  "
                        f"{radionuclide}  |  {detector}  |  {live_time:.0f} s LIVE TIME  |  "
                        f"{gross_rate:,.1f} CPS  |  {live_y_scale.upper()} DISPLAY</div>",
                        unsafe_allow_html=True,
                    )
                    st.plotly_chart(
                        live_fig, width="stretch",
                        config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False},
                        key="live_spectrum_animation",
                    )
                    st.caption("Use PLAY/PAUSE and the elapsed-time slider. The cumulative-count panel expands its time and count axes as acquisition progresses, so the displayed window follows the measurement rather than reserving the full final count range from the start. The interval-CPS panel is vertically zoomed around the expected and measured rate band (the axis is labelled accordingly) to make Poisson fluctuations visible. Each purple point is that time bin's counts divided by its own duration, with a Poisson uncertainty bar; the green dashed curve is the expected rate. For long-lived radionuclides such as 89Zr, physical decay over a short count is negligible, so the expected CPS is nearly flat. A longer 18F acquisition makes the decay trend easier to see.")
                    with st.expander("How the count-versus-time plots are calculated", expanded=False):
                        render_math(r"R_i=\frac{N_i}{\Delta t}")
                        render_math(r"u(R_i)\approx\frac{\sqrt{N_i}}{\Delta t}")
                        render_math(r"E[N_i]\approx f_{\mathrm{DT}}\left[S_{\mathrm{cal}}A_0\int_{t_i}^{t_i+\Delta t}e^{-\lambda t}\,\mathrm{d}t+R_{\mathrm{bg}}\Delta t\right]")
                        st.write("Here, N_i is the number of detected events in time bin i, Δt is the bin duration, and f_DT is the acquisition-average dead-time response factor used by this teaching model. The observed interval rate is recalculated from the counts allocated to each bin; it is not the overall mean rate repeated at every time point. Purple points show the measured interval CPS with approximate Poisson uncertainty, while the green dashed line shows the model expectation.")
                        st.caption("The interval-rate panel uses a labelled zoomed vertical scale so small but meaningful fluctuations remain visible. Read the axis limits before comparing apparent variation between runs.")
                    st.success("Measurement complete. The spectrum has been saved to Stage 2. The final frame exactly reproduces the stored 2048-channel spectrum.")

                    st.markdown("#### Scenario visualisation")
                    st.info("The acquisition spectrum and any task-specific calibration, geometry, tissue, rack or cross-modality results have been saved as separate Stage 2 datasets. Open Stage 2 → Analysis & Visualisation to inspect and export them.")
                    t_s = np.linspace(0.0, float(live_time), 100)
                    tau_s_live = max(float(deadtime_global), 0.0) * 1e-6
                    selected_gain = float(DETECTOR_REL[detector]) * float(live_geom)
                    selected_signal_rate = float(live_activity) * float(sensitivity_global) * selected_gain
                    selected_background_rate = float(live_bg)

                    if live_scenario == "Detector normalisation" and scenario_norm_summary is not None:
                        norm_plot = go.Figure()
                        norm_plot.add_trace(go.Bar(
                            x=scenario_norm_summary["Detector"],
                            y=scenario_norm_summary["Observed sensitivity (cps/Bq)"],
                            name="Before normalisation", marker_color="#FFC45B",
                        ))
                        norm_plot.add_trace(go.Bar(
                            x=scenario_norm_summary["Detector"],
                            y=scenario_norm_summary["Normalised sensitivity (cps/Bq)"],
                            name="After normalisation", marker_color="#55E0B0",
                        ))
                        norm_plot.update_layout(
                            title="Reference-source sensitivity by detector",
                            barmode="group", height=390, template="plotly_dark",
                            paper_bgcolor="#0b1823", plot_bgcolor="#102231",
                            font=dict(color="#f4f8fb"),
                            xaxis_title="Detector channel", yaxis_title="Measured sensitivity (cps/Bq)",
                        )
                        st.plotly_chart(norm_plot, width="stretch", key="live_norm_sensitivity_plot")

                        selected_norm = scenario_norm_summary[scenario_norm_summary["Detector"] == detector].iloc[0]
                        raw_rate = float(selected_norm["Net corrected rate (cps)"])
                        norm_rate = float(selected_norm["Normalised equivalent rate (cps)"])
                        time_fig = go.Figure()
                        time_fig.add_trace(go.Scatter(x=t_s, y=raw_rate*t_s, mode="lines", name=f"{detector}: raw accumulated net counts", line=dict(color="#FFC45B", width=3)))
                        time_fig.add_trace(go.Scatter(x=t_s, y=norm_rate*t_s, mode="lines", name=f"{detector}: normalised equivalent counts", line=dict(color="#55E0B0", width=3, dash="dash")))
                        time_fig.update_layout(title=f"Effect of normalisation on {detector} accumulated net counts", height=340,
                                               template="plotly_dark", paper_bgcolor="#0b1823", plot_bgcolor="#102231",
                                               font=dict(color="#f4f8fb"), xaxis_title="Elapsed time (s)", yaxis_title="Accumulated net counts (equivalent)")
                        st.plotly_chart(time_fig, width="stretch", key="live_norm_time_plot")
                        st.dataframe(scenario_norm_summary, width="stretch", hide_index=True)
                        render_math(r"F_j=\frac{\overline{S}}{S_j},\qquad S_{j,\mathrm{norm}}=F_jS_j")
                        st.caption("The bar chart uses measured sensitivities from independent Poisson acquisitions. The time trace visualises the selected detector's measured net rate before and after applying its relative correction factor. Normalisation equalises relative detector response; it does not establish the absolute cps/Bq calibration.")
                        st.success(f"Normalisation dataset saved to Stage 2: {norm_saved_name}")

                    elif live_scenario == "Background subtraction":
                        gross = (selected_signal_rate + selected_background_rate) * t_s
                        background = selected_background_rate * t_s
                        net = selected_signal_rate * t_s
                        gross_sd = np.sqrt(np.maximum(gross, 0.0))
                        background_sd = np.sqrt(np.maximum(background, 0.0))
                        bg_fig = go.Figure()
                        bg_fig.add_trace(go.Scatter(x=t_s, y=gross, mode="lines", name="Gross counts (sample + background)", line=dict(color="#FFC45B", width=3)))
                        bg_fig.add_trace(go.Scatter(x=t_s, y=background, mode="lines", name="Background counts", line=dict(color="#AFC4D2", width=2, dash="dot")))
                        bg_fig.add_trace(go.Scatter(x=t_s, y=net, mode="lines", name="Net sample counts (expected)", line=dict(color="#55E0B0", width=3)))
                        bg_fig.update_layout(title="Gross, background and net counts accumulate with time", height=360,
                                              template="plotly_dark", paper_bgcolor="#0b1823", plot_bgcolor="#102231",
                                              font=dict(color="#f4f8fb"), xaxis_title="Elapsed time (s)", yaxis_title="Cumulative counts")
                        st.plotly_chart(bg_fig, width="stretch", key="live_background_scenario_plot")
                        render_math(r"N_{\mathrm{net}}=N_{\mathrm{gross}}-N_{\mathrm{background}},\quad \sigma_{N_{\mathrm{net}}}=\sqrt{N_{\mathrm{gross}}+N_{\mathrm{background}}}")
                        st.caption("Expected curves are shown to teach the relationship. Actual measured values fluctuate according to Poisson counting statistics; background uncertainty must be included in the net-count uncertainty.")

                    elif live_scenario == "Dead-time behaviour":
                        operating_true = selected_signal_rate + selected_background_rate
                        rate_max = max(operating_true * 3.0, 100.0)
                        true_rates = np.linspace(0.0, rate_max, 250)
                        observed_rates = true_rates / (1.0 + true_rates * tau_s_live)
                        operating_observed = operating_true / (1.0 + operating_true * tau_s_live)
                        dt_fig = go.Figure()
                        dt_fig.add_trace(go.Scatter(x=true_rates, y=true_rates, mode="lines", name="Ideal response", line=dict(color="#AFC4D2", dash="dot")))
                        dt_fig.add_trace(go.Scatter(x=true_rates, y=observed_rates, mode="lines", name="Non-paralysable observed response", line=dict(color="#55E0B0", width=3)))
                        dt_fig.add_trace(go.Scatter(x=[operating_true], y=[operating_observed], mode="markers", name="Current sample operating point", marker=dict(size=12, color="#FFC45B", line=dict(color="#fff", width=1.5))))
                        dt_fig.update_layout(title="Dead-time response: true rate versus observed rate", height=370,
                                             template="plotly_dark", paper_bgcolor="#0b1823", plot_bgcolor="#102231",
                                             font=dict(color="#f4f8fb"), xaxis_title="True event rate (cps)", yaxis_title="Observed event rate (cps)")
                        st.plotly_chart(dt_fig, width="stretch", key="live_deadtime_scenario_plot")
                        loss_pct = 100.0 * (1.0 - operating_observed / max(operating_true, 1e-12))
                        st.metric("Modelled count-rate loss at this operating point", f"{loss_pct:.2f}%")
                        render_math(r"R_{\mathrm{obs}}=\frac{R_{\mathrm{true}}}{1+R_{\mathrm{true}}\tau}")
                        st.caption("This visual uses a non-paralysable dead-time model. The actual instrument model and validated dead-time parameter must be established experimentally.")

                    elif live_scenario == "Decay during acquisition":
                        half_life_s = float(RADIONUCLIDES[radionuclide]["half_life_min"]) * 60.0
                        lam_s = math.log(2.0) / half_life_s
                        activity_curve = float(live_activity) * np.exp(-lam_s*t_s)
                        rate_curve = activity_curve * float(sensitivity_global) * selected_gain
                        decay_fig = go.Figure()
                        decay_fig.add_trace(go.Scatter(x=t_s, y=activity_curve, mode="lines", name="Activity in sample", line=dict(color="#55E0B0", width=3)))
                        decay_fig.update_layout(title=f"{radionuclide}: activity during acquisition", height=340,
                                                template="plotly_dark", paper_bgcolor="#0b1823", plot_bgcolor="#102231",
                                                font=dict(color="#f4f8fb"), xaxis_title="Elapsed time (s)", yaxis_title="Activity (Bq)")
                        st.plotly_chart(decay_fig, width="stretch", key="live_decay_activity_plot")
                        rate_fig = go.Figure()
                        rate_fig.add_trace(go.Scatter(x=t_s, y=rate_curve, mode="lines", name="Expected sample count rate", line=dict(color="#C6A5FF", width=3)))
                        rate_fig.add_hline(y=float(live_bg), line_dash="dot", line_color="#AFC4D2", annotation_text="Background rate")
                        rate_fig.update_layout(title="Expected sample count rate decreases as activity decays", height=320,
                                               template="plotly_dark", paper_bgcolor="#0b1823", plot_bgcolor="#102231",
                                               font=dict(color="#f4f8fb"), xaxis_title="Elapsed time (s)", yaxis_title="Expected sample cps")
                        st.plotly_chart(rate_fig, width="stretch", key="live_decay_rate_plot")
                        st.caption(f"Acquisition-average activity is {100*float(live_spec.attrs.get('average_activity_fraction', 1.0)):.3f}% of the activity at count start. Sample events are decay-integrated; background is treated as time-constant.")

                    elif live_scenario == "Counting precision":
                        t_precision = np.linspace(1.0, max(float(live_time), 1.0), 150)
                        sig_rate = max(selected_signal_rate, 1e-12)
                        rel_signal = 100.0 / np.sqrt(np.maximum(sig_rate*t_precision, 1.0))
                        rel_bg = 100.0 * np.sqrt(np.maximum((sig_rate + 2.0*selected_background_rate)*t_precision, 1.0)) / (sig_rate*t_precision)
                        precision_fig = go.Figure()
                        precision_fig.add_trace(go.Scatter(x=t_precision, y=rel_signal, mode="lines", name="Signal-only Poisson precision", line=dict(color="#55E0B0", width=3)))
                        precision_fig.add_trace(go.Scatter(x=t_precision, y=rel_bg, mode="lines", name="Net rate with equal-duration background count", line=dict(color="#FFC45B", width=3)))
                        precision_fig.update_layout(title="Relative counting uncertainty improves with acquisition time", height=360,
                                                    template="plotly_dark", paper_bgcolor="#0b1823", plot_bgcolor="#102231",
                                                    font=dict(color="#f4f8fb"), xaxis_title="Count duration (s)", yaxis_title="Relative standard uncertainty (%)")
                        st.plotly_chart(precision_fig, width="stretch", key="live_precision_scenario_plot")
                        render_math(r"\frac{\sigma_N}{N}\approx\frac{1}{\sqrt{N}},\qquad \frac{\sigma_{R_{\mathrm{net}}}}{R_{\mathrm{net}}}=\frac{\sqrt{R_s+2R_b}}{R_s\sqrt{t}}")
                        st.caption("The second curve assumes independent sample-plus-background and background measurements with equal count durations. It illustrates why a background measurement contributes its own Poisson uncertainty.")

                    st.markdown("#### Peak analysis at completion")
                    pm1, pm2, pm3, pm4 = st.columns(4)
                    pm1.metric("Dominant line", f"{dominant_line:.1f} keV")
                    pm2.metric("Gross ROI counts", f"{live_integral['gross_peak_counts']:,.0f}")
                    pm3.metric("Estimated continuum", f"{live_integral['estimated_continuum_counts']:,.0f}")
                    pm4.metric("Net ROI counts", f"{live_integral['net_peak_counts']:,.0f}")
                    render_math(r"N_{\mathrm{net,ROI}}=N_{\mathrm{gross,ROI}}-\widehat{N}_{\mathrm{continuum,ROI}},\qquad R_{\mathrm{ROI}}=\frac{N_{\mathrm{net,ROI}}}{t_{\mathrm{live}}}")
                    st.download_button(
                        "Download acquired 2048-channel spectrum (CSV)",
                        live_dataset.to_csv(index=False).encode("utf-8"),
                        f"{live_sample_id.strip()}_{radionuclide}_live_spectrum.csv",
                        "text/csv", key="download_live_spectrum",
                    )

    # ============================== STAGE 2 ======================================
    elif stage_nav == stage_labels[1]:
        st.markdown("<div class='kicker'>ANALYSIS WORKSPACE / GENERATED MEASUREMENTS</div><div class='section-title'>Analyse the data produced in Stage 1</div>",unsafe_allow_html=True)
        st.write("Every experiment generated above is retained in this session. Select a dataset to inspect the raw measurements, examine its response, fit a model and export the exact underlying table.")
        if not st.session_state.datasets:
            st.warning("No datasets are currently available. Generate an experiment in Stage 1 or restore the demonstration pack using the sidebar.")
        else:
            ds_name=st.selectbox("Select experiment dataset",list(st.session_state.datasets.keys()))
            df=st.session_state.datasets[ds_name].copy()
            # Dataset summary metrics
            nrows=len(df); exp_type=str(df["Experiment"].iloc[0]) if "Experiment" in df else "Unknown"; iso=str(df["Radionuclide"].iloc[0]) if "Radionuclide" in df else "—"
            q1,q2,q3,q4=st.columns(4); q1.metric("Measurements / rows",f"{nrows}"); q2.metric("Experiment type",exp_type); q3.metric("Radionuclide",iso); q4.metric("Saved datasets",len(st.session_state.datasets))
            analysis_labels=["OVERVIEW & PLOTS","CALIBRATION / NORMALISATION","TABLE & EXPORT","COMPARE SCENARIOS","PET/SPECT ↔ GAMMA AGREEMENT"]
            analysis_nav=render_choice_tiles(analysis_labels,"qnl_analysis_navigation",columns=5,section_label="ANALYSIS TOOLS")
            if analysis_nav == analysis_labels[0]:
                num_cols=df.select_dtypes(include=np.number).columns.tolist()
                candidates=[c for c in num_cols if "teaching only" not in c.lower()]
                is_spectrum = "Channel" in df.columns and "Energy (keV)" in df.columns and "Counts" in df.columns and len(df)>=128
                if len(candidates)>=2 and not is_spectrum:
                    xcol=st.selectbox("X-axis",candidates,index=0,key="analysis_x")
                    ycol=st.selectbox("Y-axis",candidates,index=min(1,len(candidates)-1),key="analysis_y")
                    color_candidates=[c for c in df.columns if df[c].dtype==object and c not in ["Timestamp","Scenario notes"]]
                    ccol=st.selectbox("Group / colour",["(none)"]+color_candidates,key="analysis_color")
                    if ccol=="(none)": fig=px.scatter(df,x=xcol,y=ycol,title=f"{ycol} versus {xcol}",hover_data=[c for c in ["Detector","Tissue","Standard","Replicate"] if c in df.columns])
                    else: fig=px.scatter(df,x=xcol,y=ycol,color=ccol,title=f"{ycol} versus {xcol}",hover_data=[c for c in ["Detector","Tissue","Standard","Replicate"] if c in df.columns])
                    fig.update_layout(height=430,template="plotly_dark",legend_title_text=ccol if ccol!="(none)" else "")
                    st.plotly_chart(fig,width="stretch",key="stage2_dataset_scatter_plot")
                elif not is_spectrum:
                    st.info("This dataset does not contain two numerical columns suitable for a generic scatter plot.")
                if "Channel" in df.columns and "Energy (keV)" in df.columns and "Counts" in df.columns and len(df)>=128:
                    spec_fig=go.Figure()
                    spec_fig.add_trace(go.Scatter(x=df["Energy (keV)"],y=df["Counts"],mode="lines",name="Pulse-height spectrum",line=dict(color="#55E0B0",width=1.8)))
                    spec_fig.update_layout(title="Imported / simulated multichannel spectrum",xaxis_title="Energy (keV)",yaxis_title="Counts per channel",template="plotly_dark",font=dict(color="#f4f8fb"),height=420)
                    st.plotly_chart(spec_fig,width="stretch",key="stage2_imported_spectrum_plot")
                    if "ROI selected" in df.columns:
                        roi_df=df[df["ROI selected"].astype(bool)]
                        st.metric("Counts in selected ROI",f"{roi_df['Counts'].sum():,.0f}")
                if "Tissue" in df.columns and "%ID/g" in df.columns:
                    fig=px.bar(df,x="Tissue",y="%ID/g",color="Tissue",title="Biodistribution: %ID/g"); fig.update_layout(showlegend=False,template="plotly_dark"); st.plotly_chart(fig,width="stretch",key="stage2_biodistribution_percent_idg_plot")
                elif "Tissue" in df.columns and "Estimated %ID/g" in df.columns:
                    colour="Study arm" if "Study arm" in df.columns else "Tissue"
                    fig=px.bar(df,x="Tissue",y="Estimated %ID/g",color=colour,barmode="group",title="Hypothetical biodistribution: estimated %ID/g"); fig.update_layout(template="plotly_dark",height=390); st.plotly_chart(fig,width="stretch",key="stage2_hypothetical_tissue_percent_idg_plot")
                if "Reference concentration (Bq/mL)" in df.columns and "Modality" in df.columns:
                    st.markdown("#### Cross-modality ROI concentration summary")
                    roi_sum=df.groupby("Modality")["Reference concentration (Bq/mL)"].agg(["count","mean","std","median"]).reset_index()
                    st.dataframe(roi_sum,width="stretch",hide_index=True)
                    roi_fig=px.bar(df,x="ROI",y="Reference concentration (Bq/mL)",color="Modality",barmode="group",title="Decay-normalised ROI activity concentration")
                    roi_fig.update_layout(template="plotly_dark",font=dict(color="#f4f8fb"),height=380)
                    st.plotly_chart(roi_fig,width="stretch",key="stage2_cross_modality_roi_plot")
                if "Detector" in df.columns and "Normalisation factor" in df.columns:
                    sm=df.groupby("Detector",as_index=False).agg({"Observed sensitivity (cps/Bq)":"mean","Normalisation factor":"first","Normalised sensitivity (cps/Bq)":"mean"})
                    fig=go.Figure(); fig.add_bar(x=sm.Detector,y=sm["Observed sensitivity (cps/Bq)"],name="Before"); fig.add_bar(x=sm.Detector,y=sm["Normalised sensitivity (cps/Bq)"],name="After"); fig.update_layout(barmode="group",title="Detector response before and after normalisation",template="plotly_dark"); st.plotly_chart(fig,width="stretch",key="stage2_detector_normalisation_plot")
            elif analysis_nav == analysis_labels[1]:
                if "Known activity (Bq)" in df.columns and "Net rate (cps)" in df.columns:
                    grouped=df.groupby("Known activity (Bq)",as_index=False).agg({"Net rate (cps)":"mean"})
                    if len(grouped)>=2:
                        slope,inter=np.polyfit(grouped["Known activity (Bq)"],grouped["Net rate (cps)"],1)
                        pred=slope*grouped["Known activity (Bq)"]+inter
                        ssr=np.sum((grouped["Net rate (cps)"]-pred)**2); sst=np.sum((grouped["Net rate (cps)"]-grouped["Net rate (cps)"].mean())**2); r2=1-ssr/sst if sst else np.nan
                        u1,u2,u3=st.columns(3); u1.metric("Sensitivity slope",f"{slope:.6f} cps/Bq"); u2.metric("Intercept",f"{inter:.3f} cps"); u3.metric("R²",f"{r2:.5f}")
                        f=go.Figure(); f.add_trace(go.Scatter(x=grouped["Known activity (Bq)"],y=grouped["Net rate (cps)"],mode="markers",name="Measured means")); xx=np.linspace(0,grouped["Known activity (Bq)"].max(),100); f.add_trace(go.Scatter(x=xx,y=slope*xx+inter,mode="lines",name="OLS linear fit")); f.update_layout(title="Calibration fit",xaxis_title="Known activity (Bq)",yaxis_title="Net rate (cps)",template="plotly_dark",height=380); st.plotly_chart(f,width="stretch",key="stage2_calibration_regression_plot")
                        render_math(r"R_{\mathrm{net}}=S_{\mathrm{cal}}A+b,\qquad \widehat{A}=\frac{R_{\mathrm{net}}-b}{S_{\mathrm{cal}}}")
                        st.caption("OLS is used here for transparency. For formal calibration, consider weighting by measurement variance, checking residuals and using a validated model over the intended activity range.")
                elif "Detector" in df.columns and "Observed sensitivity (cps/Bq)" in df.columns:
                    summ=df.groupby("Detector",as_index=False).agg({"Observed sensitivity (cps/Bq)":"mean"})
                    fleet=summ["Observed sensitivity (cps/Bq)"].mean(); summ["Derived normalisation factor"]=fleet/summ["Observed sensitivity (cps/Bq)"]
                    summ["Expected normalised sensitivity"]=summ["Observed sensitivity (cps/Bq)"]*summ["Derived normalisation factor"]
                    st.dataframe(summ,width="stretch",hide_index=True)
                    f=px.bar(summ,x="Detector",y=["Observed sensitivity (cps/Bq)","Expected normalised sensitivity"],barmode="group",title="Detector response normalisation"); f.update_layout(template="plotly_dark"); st.plotly_chart(f,width="stretch",key="stage2_detector_normalisation_summary_plot")
                    render_math(r"F_j=\frac{\overline{S}}{S_j},\qquad S_{j,\mathrm{norm}}=F_j S_j")
                else:
                    st.info("This dataset is not a calibration or detector-normalisation experiment. Use Overview & Plots for generic visualisation.")
            elif analysis_nav == analysis_labels[2]:
                show_teaching=st.checkbox("Show teaching ground-truth columns",value=True)
                display=df.copy()
                if not show_teaching:
                    display=display[[c for c in display.columns if "teaching only" not in c.lower() and not c.lower().startswith("true ")]]
                st.dataframe(display,width="stretch",hide_index=True)
                csv=display.to_csv(index=False).encode("utf-8")
                st.download_button("Download selected dataset (CSV)",csv,file_name=f"{ds_name.replace(' ','_').replace('·','-')}.csv",mime="text/csv")
                # Excel workbook with selected data, plus metadata sheet
                xbuf=io.BytesIO()
                try:
                    with pd.ExcelWriter(xbuf,engine="openpyxl") as writer:
                        display.to_excel(writer,index=False,sheet_name="Measurements")
                        pd.DataFrame({"Property":["Dataset","Rows","Exported at","Use limitation"],"Value":[ds_name,len(display),datetime.now().isoformat(timespec="seconds"),"Synthetic teaching data; not experimental observations"]}).to_excel(writer,index=False,sheet_name="Metadata")
                    st.download_button("Download selected dataset (Excel)",xbuf.getvalue(),file_name="gamma_lab_dataset.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except Exception as exc:
                    st.caption(f"Excel export is unavailable in this environment ({exc}); use CSV export.")
                st.markdown("#### Export the complete workshop session")
                all_rows=pd.concat(list(st.session_state.datasets.values()),ignore_index=True,sort=False)
                st.download_button("Download all datasets (combined CSV)",all_rows.to_csv(index=False).encode("utf-8"),"gamma_lab_all_experiments.csv","text/csv")
                allbuf=io.BytesIO()
                try:
                    with pd.ExcelWriter(allbuf,engine="openpyxl") as writer:
                        for idx,(nm,table) in enumerate(st.session_state.datasets.items(),1):
                            sheet=f"Exp{idx:02d}_{nm.split('·')[0].strip()}"[:31]
                            # Ensure unique sheet names after truncation.
                            if sheet in writer.book.sheetnames: sheet=f"Exp{idx:02d}"
                            table.to_excel(writer,index=False,sheet_name=sheet)
                    st.download_button("Download all datasets (multi-sheet Excel)",allbuf.getvalue(),"gamma_lab_all_experiments.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except Exception as exc:
                    st.caption(f"Complete Excel export unavailable ({exc}); combined CSV remains available.")
            elif analysis_nav == analysis_labels[3]:
                if len(st.session_state.datasets)>=2:
                    names=list(st.session_state.datasets)
                    chosen=st.multiselect("Choose datasets to compare",names,default=names[:min(3,len(names))])
                    if chosen:
                        combined=pd.concat([st.session_state.datasets[n] for n in chosen],ignore_index=True,sort=False)
                        st.dataframe(combined,width="stretch",hide_index=True)
                        numeric=combined.select_dtypes(include=np.number).columns.tolist()
                        if numeric:
                            metric=st.selectbox("Metric to compare",numeric,index=0,key="scenario_metric")
                            if "Experiment" in combined.columns:
                                summary=combined.groupby("Experiment",dropna=False)[metric].agg(["count","mean","std","median"]).reset_index()
                                st.dataframe(summary,width="stretch",hide_index=True)
                                fig=px.box(combined,x="Experiment",y=metric,color="Experiment",points="all",title=f"Scenario comparison: {metric}")
                                fig.update_layout(showlegend=False,template="plotly_dark",height=420); st.plotly_chart(fig,width="stretch",key="stage2_combined_scenario_boxplot")
                        st.download_button("Export combined comparison (CSV)",combined.to_csv(index=False).encode(),"gamma_lab_combined_scenarios.csv","text/csv")
                else:
                    st.info("Generate or restore more than one experiment dataset to compare scenarios.")

            elif analysis_nav == analysis_labels[4]:
                st.markdown("#### Cross-modality quantitative agreement")
                st.write("Match PET/SPECT ROI summary measurements to ex-vivo gamma-counted tissue samples. Both modalities are decay-normalised to a shared reference time. Gamma activity is divided by tissue volume (mass / density) to obtain Bq/mL.")
                image_sets=[n for n,d in st.session_state.datasets.items() if "Reference concentration (Bq/mL)" in d.columns and "ROI" in d.columns]
                gamma_sets=[n for n,d in st.session_state.datasets.items() if any(c in d.columns for c in ["Estimated activity at count (Bq)","Estimated activity (Bq)"]) and "Mass (g)" in d.columns]
                if not image_sets or not gamma_sets:
                    st.info("To use this analysis, first import PET/SPECT ROI measurements and generate a tissue-counting or automated-rack dataset. The example datasets are available in the session for practice.")
                else:
                    cm1,cm2=st.columns(2)
                    with cm1: image_name=st.selectbox("PET/SPECT ROI dataset",image_sets,key="cm_image_dataset")
                    with cm2: gamma_name=st.selectbox("Gamma-counter tissue dataset",gamma_sets,key="cm_gamma_dataset")
                    idf=st.session_state.datasets[image_name].copy(); gdf=st.session_state.datasets[gamma_name].copy()
                    image_iso=str(idf["Radionuclide"].iloc[0]) if "Radionuclide" in idf.columns else radionuclide
                    gamma_iso=str(gdf["Radionuclide"].iloc[0]) if "Radionuclide" in gdf.columns else radionuclide
                    cross_iso=image_iso if image_iso==gamma_iso else gamma_iso
                    if image_iso!=gamma_iso:
                        st.warning(f"Radionuclide mismatch: image dataset is {image_iso}, gamma dataset is {gamma_iso}. Select the radionuclide whose decay correction should be used only after checking the source records.")
                        cross_iso=st.selectbox("Decay-correction radionuclide",list(RADIONUCLIDES),index=list(RADIONUCLIDES).index(gamma_iso) if gamma_iso in RADIONUCLIDES else 0,key="cm_iso")
                    image_key=st.selectbox("Image ROI matching field",[c for c in idf.columns if idf[c].dtype==object and c not in ["Timestamp","Scenario notes","Experiment","Radionuclide","Modality"]],index=0,key="cm_image_key")
                    gamma_keys=[c for c in gdf.columns if gdf[c].dtype==object and c not in ["Timestamp","Scenario notes","Experiment","Radionuclide"]]
                    preferred=next((c for c in gamma_keys if c.lower() in ["tissue","organ","sample id"]),gamma_keys[0] if gamma_keys else None)
                    if preferred is None:
                        st.warning("The selected gamma-counter dataset does not contain a text field to match sample / tissue names.")
                    else:
                        gamma_key=st.selectbox("Gamma-counter tissue matching field",gamma_keys,index=gamma_keys.index(preferred),key="cm_gamma_key")
                        activity_opts=[c for c in ["Estimated activity at count (Bq)","Estimated activity (Bq)"] if c in gdf.columns]
                        ga1,ga2,ga3=st.columns(3)
                        with ga1: gamma_activity_col=st.selectbox("Gamma activity field",activity_opts,key="cm_activity_col")
                        with ga2: gamma_mass_col=st.selectbox("Tissue mass field",[c for c in gdf.columns if "mass" in c.lower() and "g" in c.lower()],key="cm_mass_col")
                        with ga3:
                            tcols=[c for c in gdf.columns if "elapsed time" in c.lower()]
                            gamma_time_col=st.selectbox("Gamma count time field",["(reference time)"]+tcols,key="cm_time_col")
                        d1,d2,d3=st.columns(3)
                        with d1: density=st.number_input("Tissue density (g/mL)",.5,2.0,1.05,.01,key="cm_density")
                        image_ref_default=float(pd.to_numeric(idf["Common reference time (min)"],errors="coerce").dropna().median()) if "Common reference time (min)" in idf.columns and pd.to_numeric(idf["Common reference time (min)"],errors="coerce").notna().any() else 0.0
                        with d2: common_ref=st.number_input("Common reference time (min)",0.0,100000.0,image_ref_default,1.0,key="cm_ref")
                        with d3: image_value_col=st.selectbox("Image concentration field",["Reference concentration (Bq/mL)"],key="cm_image_value")
                        if st.button("MATCH AND CALCULATE AGREEMENT",type="primary",width="stretch",key="cm_run"):
                            half=float(RADIONUCLIDES[cross_iso]["half_life_min"])
                            im=idf[[image_key,image_value_col]].copy(); im.columns=["Match key","Image concentration (Bq/mL)"]
                            gm=gdf[[gamma_key,gamma_activity_col,gamma_mass_col]+([] if gamma_time_col=="(reference time)" else [gamma_time_col])].copy()
                            gm.columns=["Match key","Gamma activity (Bq)","Mass (g)"]+([] if gamma_time_col=="(reference time)" else ["Gamma elapsed time (min)"])
                            im["Match key"]=im["Match key"].astype(str).str.strip().str.casefold()
                            gm["Match key"]=gm["Match key"].astype(str).str.strip().str.casefold()
                            im["Image concentration (Bq/mL)"]=pd.to_numeric(im["Image concentration (Bq/mL)"],errors="coerce")
                            if "Common reference time (min)" in idf.columns and pd.to_numeric(idf["Common reference time (min)"],errors="coerce").notna().any():
                                img_ref=pd.to_numeric(idf["Common reference time (min)"],errors="coerce").fillna(image_ref_default).to_numpy()
                                img_dt=common_ref-img_ref
                                img_factor=np.where(img_dt>=0,np.array([decay_factor(float(x),half) for x in np.maximum(img_dt,0)]),np.array([1.0/decay_factor(float(-x),half) for x in np.minimum(img_dt,0)]))
                                im["Image concentration (Bq/mL)"]=im["Image concentration (Bq/mL)"]*img_factor
                            gm["Gamma activity (Bq)"]=pd.to_numeric(gm["Gamma activity (Bq)"],errors="coerce")
                            gm["Mass (g)"]=pd.to_numeric(gm["Mass (g)"],errors="coerce")
                            if "Gamma elapsed time (min)" in gm:
                                gm["Gamma elapsed time (min)"]=pd.to_numeric(gm["Gamma elapsed time (min)"],errors="coerce").fillna(0)
                                dt=gm["Gamma elapsed time (min)"]-common_ref
                                factors=np.where(dt>=0,1/np.array([decay_factor(float(x),half) for x in np.maximum(dt,0)]),np.array([decay_factor(float(-x),half) for x in np.minimum(dt,0)]))
                            else:
                                factors=np.ones(len(gm))
                            gm["Gamma activity at common reference (Bq)"]=gm["Gamma activity (Bq)"]*factors
                            gm["Gamma concentration (Bq/mL)"]=gm["Gamma activity at common reference (Bq)"]*density/gm["Mass (g)"].replace(0,np.nan)
                            im=im.groupby("Match key",as_index=False)["Image concentration (Bq/mL)"].mean()
                            gm=gm.groupby("Match key",as_index=False).agg({"Gamma concentration (Bq/mL)":"mean","Gamma activity at common reference (Bq)":"mean","Mass (g)":"mean"})
                            merged=im.merge(gm,on="Match key",how="inner")
                            merged=merged.replace([np.inf,-np.inf],np.nan).dropna(subset=["Image concentration (Bq/mL)","Gamma concentration (Bq/mL)"])
                            if merged.empty:
                                st.error("No matching ROI/tissue labels were found. Check the selected matching fields and spelling in the source datasets.")
                            else:
                                merged["Difference (Gamma - image) (Bq/mL)"]=merged["Gamma concentration (Bq/mL)"]-merged["Image concentration (Bq/mL)"]
                                merged["Mean of methods (Bq/mL)"]=(merged["Gamma concentration (Bq/mL)"]+merged["Image concentration (Bq/mL)"])/2
                                merged["Gamma / image ratio"]=merged["Gamma concentration (Bq/mL)"]/merged["Image concentration (Bq/mL)"].replace(0,np.nan)
                                merged["Relative difference (%)"]=100*merged["Difference (Gamma - image) (Bq/mL)"]/merged["Image concentration (Bq/mL)"].replace(0,np.nan)
                                st.session_state.datasets[f"ANALYSIS · PET-SPECT gamma agreement · {datetime.now().strftime('%H%M%S')}"]=new_record(merged,"PET-SPECT gamma agreement",cross_iso,
                                    "Matched ROI/tissue labels; gamma activity converted to Bq/mL using mass and selected density; decay-normalised to common reference time.")
                                aa,bb,cc=st.columns(3)
                                aa.metric("Matched ROIs",len(merged)); bb.metric("Mean gamma/image ratio",f"{merged['Gamma / image ratio'].mean():.3f}"); cc.metric("Mean relative difference",f"{merged['Relative difference (%)'].mean():+.1f}%")
                                st.dataframe(merged,width="stretch",hide_index=True)
                                f1=go.Figure(); f1.add_trace(go.Scatter(x=merged["Image concentration (Bq/mL)"],y=merged["Gamma concentration (Bq/mL)"],mode="markers+text",text=merged["Match key"],textposition="top center",name="Matched ROIs"))
                                lo=float(min(merged["Image concentration (Bq/mL)"].min(),merged["Gamma concentration (Bq/mL)"].min())); hi=float(max(merged["Image concentration (Bq/mL)"].max(),merged["Gamma concentration (Bq/mL)"].max()))
                                f1.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode="lines",line=dict(dash="dash",color="#FFC45B"),name="Identity line"))
                                f1.update_layout(title="PET/SPECT versus gamma-counter concentration",xaxis_title="Image concentration (Bq/mL)",yaxis_title="Gamma-counter concentration (Bq/mL)",template="plotly_dark",font=dict(color="#f4f8fb"),height=430)
                                st.plotly_chart(f1,width="stretch",key="stage2_pet_gamma_agreement_scatter")
                                dif=merged["Difference (Gamma - image) (Bq/mL)"]; avg=merged["Mean of methods (Bq/mL)"]
                                bias=float(dif.mean()); loa=1.96*float(dif.std(ddof=1)) if len(dif)>1 else np.nan
                                f2=go.Figure(); f2.add_trace(go.Scatter(x=avg,y=dif,mode="markers+text",text=merged["Match key"],textposition="top center",name="ROI difference"))
                                f2.add_hline(y=bias,line_dash="dash",annotation_text=f"Bias = {bias:.2g}")
                                if np.isfinite(loa): f2.add_hline(y=bias+loa,line_dash="dot",annotation_text="+1.96 SD"); f2.add_hline(y=bias-loa,line_dash="dot",annotation_text="−1.96 SD")
                                f2.update_layout(title="Bland–Altman agreement analysis",xaxis_title="Mean concentration (Bq/mL)",yaxis_title="Gamma − image (Bq/mL)",template="plotly_dark",font=dict(color="#f4f8fb"),height=430)
                                st.plotly_chart(f2,width="stretch",key="stage2_pet_gamma_bland_altman")
                                st.download_button("Download matched agreement table (CSV)",merged.to_csv(index=False).encode(),"pet_spect_gamma_agreement.csv","text/csv",key="cm_export")
                                st.caption("This is a comparison of ROI summary measurements, not voxel-wise image validation. Density, recovery coefficients, calibration and reference-time assumptions must be justified for real data.")

            st.markdown("---")
            with st.expander("Data dictionary and interpretation notes"):
                st.markdown("""
    - **Gross counts:** total events recorded in the sample measurement window.
    - **Background counts:** independently simulated background acquisition. In a physical workflow, sample and background durations may differ; rate scaling is then required.
    - **Net rate:** background-subtracted and dead-time-corrected rate.
    - **Sensitivity (cps/Bq):** empirical count-rate response per unit activity for a specified radionuclide, energy window, instrument and geometry.
    - **Normalisation factor:** multiplicative correction relative to the selected detector fleet/reference mean.
    - **%ID/g:** decay-corrected tissue activity divided by the injected activity at the same reference time and tissue mass.
    - **Teaching ground truth:** available in synthetic experiments to allow attendees to assess bias and recovery. It is not measured data.

    All datasets are held in Streamlit session memory. Download them before closing or refreshing the session if you need to preserve the workshop output.
    """)

    # -------------------------------- footer --------------------------------------
    st.markdown("---")
    st.markdown("<div style='display:flex;justify-content:space-between;gap:18px;flex-wrap:wrap;color:#d6e2ea;font-size:12px'><div><b style='color:#55e0b0'>QUANTITATIVE NUCLEAR MEDICINE LABORATORY</b> · instrument guide and interactive virtual practicals</div><div>Synthetic Poisson counts · 2048-channel spectra · calibration · biodistribution · PET/SPECT</div></div>",unsafe_allow_html=True)
    with st.expander("Scientific scope and limitations"):
        st.markdown("""
    **Educational simulator only.** This spectrum-first application is not an instrument controller or a manufacturer interface. All generated measurements and hypothetical tissue uptake values are synthetic. The default cps/Bq sensitivities, relative detector gains, geometry and matrix response factors are illustrative, not instrument calibration data.

    The count-generation model uses Poisson sampling and a non-paralysable dead-time approximation. The 2048-channel spectrum is an isotope-informed teaching model with Gaussian photopeaks, approximate Compton/backscatter components, optional characteristic X-ray and coincidence-summing features; it is not a Monte Carlo detector-response model. Real gamma-counter response may require a different dead-time model, isotope-specific energy windows, spillover correction, geometry- and fill-height-specific calibration, matrix validation and traceable standards. Detector normalisation is demonstrated as a relative response correction and should not be confused with absolute activity calibration.

    For any real biodistribution study, decay-correction reference times, sample/background duration scaling, radionuclide-specific sensitivity, uncertainty propagation, sample mass and injected activity measurement must be specified and validated.
    """)
