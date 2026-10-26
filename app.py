"""
The AI Unit Economics Deal Desk  (v2.7 - levers & goal-seek, traffic mix, CAC/LTV, scenario compare & export, feedback form)
------------------------------------------------------------------------------------------
FP&A planning model for direct AI inference, vector storage, infrastructure step-costs,
enterprise contract cohorts (term / renewal / expansion), promotional-credit drawdown,
invoice-lag cash timing, scenario sensitivities and a like-for-like pricing comparison.

All prices, rates and presets are user-editable ESTIMATES. Verify against vendor billing
portals, billing-system exports and accounting policy before using for decisions.

TOKEN PRICES: model prices per 1M tokens are illustrative placeholders (not live vendor quotes).
They are held FLAT unless the user turns on the "AI cost path" (efficiency / vendor price change).
Free-tier inference can be booked to COGS or to sales & marketing; extra COGS lines can be added
to reach a fully loaded margin. See the disclaimers at the bottom of the page.
"""
import calendar
import datetime
import html
import inspect
import io
import json
import math
import re
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

APP_NAME = "The AI Unit Economics Deal Desk"
APP_VERSION = "v2.9"

st.set_page_config(
    page_title="AI Deal Desk | Will my AI product make money?",
    page_icon="⚡",
    layout="wide",
)

# ==========================================
# 0. STYLING & SMALL UI HELPERS
# ==========================================
def _detect_dark_theme():
    """True / False when Streamlit reports the active theme; None when unknown (older Streamlit, first load)."""
    try:
        t = st.context.theme.type
        if t in ("dark", "light"):
            return t == "dark"
    except Exception:
        pass
    try:
        base = st.get_option("theme.base")
        if base in ("dark", "light"):
            return base == "dark"
    except Exception:
        pass
    return None


IS_DARK = _detect_dark_theme()
_LIGHT = dict(app_bg="#ffffff", text="#252936", side_bg="#fbfcff", side_border="#dfe1ea", label="#4a4e5c",
              card_bg="#ffffff", card_border="#e2e7f0", metric_label="#5b5f6e", hdr_from="#ffffff", hdr_to="#f6f8fd",
              hdr_text="#2c2f39", hdr_sub="#5b5f6e", note_bg="#fcfcfd", note_border="#dfe1ea", note_accent="#1834c6",
              note_h4="#0b2cbc", note_text="#3d4150", flag_bg="#fffaf0", flag_border="#f6d6a8", flag_accent="#dd6b20",
              flag_b="#c05621", flag_text="#4a5568",
              accent="#0b2cbc", on_accent="#ffffff", navy="#131733", ring="#e4e7f6", chip_bg="#e3e7f8", chip_text="#0b2cbc",
              good="#1f7a4d", good_bg="#e1f3e9", warn="#9a5b00", warn_bg="#fdf0d5", bad="#b42318", bad_bg="#fde4e1",
              accent_dark="#0a1f8a", seg_bg="#e6e8f0", seg_on="#ffffff", violet="#8a5cf6", sky="#33c2ff")
_DARK = dict(app_bg="#0f1226", text="#e8eaf3", side_bg="#151937", side_border="#262b52", label="#c5c9e0",
             card_bg="#171b3a", card_border="#2a3060", metric_label="#a3a9cc", hdr_from="#1a1f45", hdr_to="#151937",
             hdr_text="#f5f6fb", hdr_sub="#a3a9cc", note_bg="#171b3a", note_border="#2a3060", note_accent="#6f8bff",
             note_h4="#9db0ff", note_text="#c5c9e0", flag_bg="#2a2118", flag_border="#5c4326", flag_accent="#ed8936",
             flag_b="#fbd38d", flag_text="#e8eaf3",
             accent="#5b7bff", on_accent="#0f1226", navy="#0a0c1c", ring="#222861", chip_bg="#232a5e", chip_text="#aebcff",
             good="#5fd08f", good_bg="#14301f", warn="#f3b25c", warn_bg="#33270f", bad="#ff8a80", bad_bg="#3a1816",
             accent_dark="#2a3fb0", seg_bg="#1f2552", seg_on="#2a3060", violet="#a78bfa", sky="#5cd0ff")


def _css_vars(pal):
    return "".join(f"--{k.replace('_', '-')}:{v};" for k, v in pal.items())


if IS_DARK is True:
    _ROOT_CSS = f":root{{{_css_vars(_DARK)}}}"
elif IS_DARK is False:
    _ROOT_CSS = f":root{{{_css_vars(_LIGHT)}}}"
else:  # theme not reported: light by default, follow the operating-system setting when it is dark
    _ROOT_CSS = f":root{{{_css_vars(_LIGHT)}}} @media (prefers-color-scheme: dark){{:root{{{_css_vars(_DARK)}}}}}"

# Plotly: dark template + transparent backgrounds in dark mode so charts sit on the page colour.
CHART_KW = (dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            if IS_DARK else dict(template="plotly_white"))


def tone(light, dark):
    """Pick a chart colour that stays readable on the active theme."""
    return dark if IS_DARK else light


_STATIC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
.stApp {background:var(--app-bg);color:var(--text);font-family:'DM Sans', sans-serif;}
section[data-testid="stSidebar"] {background:var(--side-bg) !important; border-right: 1px solid var(--side-border);}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    font-size: 0.95rem !important; margin-top: 6px !important; margin-bottom: 2px !important;
    padding-bottom: 0px !important; font-weight: 700 !important;
}
section[data-testid="stSidebar"] label {font-size: 0.75rem !important; color: var(--label) !important;}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap: .45rem !important;}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {margin-bottom: .12rem !important;}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] button {min-height: 1.9rem !important;}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {min-height: 1.9rem !important;}
section[data-testid="stSidebar"] [data-testid="stSlider"] {padding-top: 0 !important; padding-bottom: 0 !important;}
section[data-testid="stSidebar"] hr {margin: .45rem 0 !important;}
div[data-testid="stMetric"] {
    background:var(--card-bg); border:1px solid var(--card-border); padding:6px 10px !important;
    border-radius:6px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}
div[data-testid="stMetric"] label {font-size:0.75rem !important; color:var(--metric-label) !important;}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {font-size:1.1rem !important; font-weight:700;}
.compact-header {
    background: linear-gradient(135deg,var(--hdr-from),var(--hdr-to)); padding: 8px 14px; border-radius: 6px;
    border: 1px solid var(--card-border); margin-bottom: 8px;
}
.compact-header h1 {color:var(--hdr-text); margin:0 0 2px 0; font-size:1.25rem; font-weight:700;}
.compact-header p {color:var(--hdr-sub); margin:0; font-size:0.8rem;}
.cfo-commentary {
    background: var(--note-bg); border: 1px solid var(--note-border); border-left: 4px solid var(--note-accent);
    padding: 10px 14px; border-radius: 6px; margin-bottom: 12px;
}
.cfo-commentary h4 {margin:0 0 4px 0; font-size:0.9rem; color:var(--note-h4); font-weight:700;}
.cfo-commentary ul {margin:0; padding-left:16px; font-size:0.82rem; color:var(--note-text);}
.cfo-commentary li {margin-bottom:2px;}
.assumption-flag {
    background: var(--flag-bg); border: 1px solid var(--flag-border); border-left: 4px solid var(--flag-accent);
    padding: 8px 12px; border-radius: 6px; margin-bottom: 12px; font-size: 0.8rem; color: var(--flag-text);
}
.assumption-flag b {color: var(--flag-b);}
.assumption-flag ul {margin: 4px 0 0 0; padding-left: 16px;}
.assumption-flag li {margin-bottom: 2px;}
.hero {padding: 10px 4px 0 4px;}
div.block-container {padding-top: 2.4rem !important; padding-bottom: 2rem !important;}
header[data-testid="stHeader"] {background: transparent;}
.eyebrow {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; letter-spacing: .14em; text-transform: uppercase;
          font-size: .72rem; color: var(--accent); font-weight: 600;}
.hero h1 {font-size: clamp(1.7rem, 2.9vw, 2.4rem) !important; line-height: 1.15 !important; font-weight: 600 !important; color: var(--text); margin: 10px 0 12px 0 !important; padding: 0 !important; letter-spacing: -0.02em;}
.hero h1 .accent, .chapter h2 .accent {color: var(--accent);}
.hero p.sub {font-size: 1rem; line-height: 1.55; color: var(--label); max-width: 62rem; margin: 0 0 12px 0;}
.skipbar {font-size: .8rem; color: var(--label); margin: 4px 0 6px 0;}
.skipbar a {color: var(--accent); text-decoration: none; margin-right: 14px; font-weight: 600;}
.chapter {margin: 26px 0 6px 0; padding-top: 18px; border-top: 1px solid var(--card-border);}
.chapter h2 {font-size: 1.45rem !important; font-weight: 600 !important; padding: 0 !important; margin: 6px 0 8px 0 !important; color: var(--text); letter-spacing: -0.01em;}
.chapter p.lede {font-size: .95rem; line-height: 1.5; color: var(--label); margin: 0 0 14px 0; max-width: 62rem;}
.kpi-card {height: 100%; min-height: 238px; box-sizing: border-box; display: flex; flex-direction: column; border: 1.5px solid transparent; border-radius: 14px; padding: 16px 18px; margin-bottom: 10px;
           background: linear-gradient(var(--card-bg), var(--card-bg)) padding-box, linear-gradient(135deg, var(--accent), var(--violet) 60%, var(--sky)) border-box;}
.kpi-card .row {display: flex; align-items: flex-end; justify-content: space-between; gap: 14px; margin: 2px 0 6px 0;}
.kpi-card .row svg {flex: none;}
.kpi-card .trend {font-size: .76rem; color: var(--metric-label); margin-bottom: 6px;}
.kpi-card .delta {font-size: .78rem; font-weight: 600;} .kpi-card .delta.up {color: var(--good);} .kpi-card .delta.down {color: var(--bad);}
.rank {list-style: none; margin: 6px 0 0 0; padding: 0;}
.rank li {display: flex; align-items: center; gap: 10px; padding: 6px 0; border-top: 1px solid var(--card-border); font-size: .85rem;}
.rank .n {width: 20px; height: 20px; border-radius: 50%; background: var(--text); color: var(--app-bg); font-size: .7rem; display: flex; align-items: center; justify-content: center; flex: none;}
.rank .bar {height: 6px; border-radius: 9px; background: var(--seg-bg); flex: 1; max-width: 110px;} .rank .bar i {display: block; height: 100%; border-radius: 9px; background: var(--accent);}
.rank .amt {margin-left: auto; font-weight: 600;}
.kpi-card .k {font-size: .74rem; font-weight: 600; color: var(--metric-label); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
              letter-spacing: .08em; text-transform: uppercase;}
.kpi-card .v {font-size: 2.1rem; font-weight: 650; color: var(--text); line-height: 1.15; margin: 0;}
.kpi-card .d {font-size: .8rem; color: var(--label); min-height: 3.2em; margin-top: auto; padding-top: 8px;}
.founder-art {max-width: 320px; margin: 4px auto 0; padding: 8px;}
.founder-art svg {display:block; width:100%; height:auto; overflow:visible;}
.founder-question {text-align:center; color:var(--label); font-size:.9rem; line-height:1.45; margin-top:2px;}
.chapter h2 em {font-family: Georgia, 'Times New Roman', serif; font-weight:500; font-style:italic; letter-spacing:-.025em;}
.hero h1, .chapter h2, .hero p.sub, .chapter p.lede {font-family:'DM Sans', sans-serif;}
@media (max-width: 900px) {.founder-art {max-width:230px;} .kpi-card {min-height:220px;}}
.pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .72rem; font-weight: 600; margin-bottom: 4px;
       background: var(--chip-bg); color: var(--chip-text);}
.pill.good {background: var(--good-bg); color: var(--good);} .pill.warn {background: var(--warn-bg); color: var(--warn);}
.pill.bad {background: var(--bad-bg); color: var(--bad);}
.takeaway {background: var(--note-bg); border: 1px solid var(--note-border); border-left: 4px solid var(--accent);
           padding: 10px 14px; border-radius: 10px; margin: 12px 0; font-size: .92rem; color: var(--note-text);}
.bridge {border-left: 3px solid var(--accent); padding: 4px 12px; color: var(--label); font-style: italic; margin: 14px 0 6px 0;}
.stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {
    border-radius: 12px; font-weight: 600; background: var(--card-bg); color: var(--text); border: 1px solid var(--card-border);
    box-shadow: 0 3px 0 var(--card-border); transition: transform .06s, box-shadow .06s;}
.stButton > button:active, .stDownloadButton > button:active, div[data-testid="stFormSubmitButton"] > button:active {transform: translateY(3px); box-shadow: 0 0 0 var(--card-border);}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
    background: var(--accent); color: var(--on-accent); border: 0; box-shadow: 0 4px 0 var(--accent-dark);}
.stButton > button[kind="primary"]:active {box-shadow: 0 1px 0 var(--accent-dark);}
.st-key-chips .stButton > button {border-radius: 999px; background: var(--chip-bg); color: var(--chip-text); border: 0; box-shadow: none; padding: 4px 14px; font-size: .82rem;}
.st-key-chips .stButton > button:active {transform: none;}
.st-key-profile [data-testid="stHorizontalBlock"] {gap: 0.5rem !important; align-items: stretch !important;}
.st-key-profile [data-testid="column"] {min-width: 0 !important; flex: 1 1 0% !important;}
.st-key-profile .stButton {width: 100% !important; margin-bottom: 0 !important;}
.st-key-profile [data-testid="stButton"] {width: 100% !important; display: block !important;}
.st-key-profile .stButton > button {display: flex !important; align-items: center !important; justify-content: center !important; width: 100% !important; max-width: 100% !important; min-width: 0 !important; padding: 15px 8px !important; font-size: 1.02rem !important; min-height: 54px !important; box-sizing: border-box !important; margin: 0 !important; white-space: normal !important;}
.st-key-profile .stage-story {display: block; width: 100%; text-align: center; font-size: 0.84rem; line-height: 1.4; min-height: 5.2rem; padding: 8px 4px 0; color: var(--label); overflow-wrap: anywhere; box-sizing: border-box;}
"""
st.markdown(f"<style>{_ROOT_CSS}\n{_STATIC_CSS}</style>", unsafe_allow_html=True)


_VIEWS = ["Guided story", "Analyst view"]
_seg = getattr(st, "segmented_control", None)
if _seg is not None:
    _v = _seg("View", _VIEWS, default=st.session_state.get("_view", _VIEWS[0]), key="view_seg", label_visibility="collapsed")
else:
    _v = st.radio("View", _VIEWS, index=_VIEWS.index(st.session_state.get("_view", _VIEWS[0])), horizontal=True, key="view_radio", label_visibility="collapsed")
VIEW = _v if _v in _VIEWS else st.session_state.get("_view", _VIEWS[0])     # a click on the active segment deselects it: keep the last view
st.session_state["_view"] = VIEW
GUIDED = VIEW == "Guided story"


def _keyed_container(key):
    try:
        return st.container(key=key)
    except TypeError:   # older Streamlit without container keys: still works, just unstyled
        return st.container()


def _finite(vals):
    return [float(x) for x in vals if x is not None and not pd.isna(x)]


def _spark_svg(vals, color="var(--accent)", w=120, h=44):
    """Small trend line with a soft area and an end dot. Fixed size (never stretched), padded so the line is not clipped."""
    v = _finite(vals)
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    span = (hi - lo) or 1.0
    pad = 6
    xy = [(pad + i / (len(v) - 1) * (w - 2 * pad), pad + (1 - (x - lo) / span) * (h - 2 * pad)) for i, x in enumerate(v)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
    area = f"{xy[0][0]:.1f},{h - pad} " + line + f" {xy[-1][0]:.1f},{h - pad}"
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}"><polygon points="{area}" fill="{color}" opacity="0.10"/>'
            f'<polyline fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="{line}"/>'
            f'<circle cx="{xy[-1][0]:.1f}" cy="{xy[-1][1]:.1f}" r="3" fill="{color}"/></svg>')


def kpi_card(label, value, note="", pill=None, pill_kind="", series=None, trend_note=None):
    _p = f'<span class="pill {pill_kind}">{html.escape(pill)}</span><br>' if pill else ""
    v = _finite(series) if series is not None else []
    _sp = _spark_svg(series) if len(v) >= 2 else ""
    if trend_note is not None:
        _tr = f'<div class="trend">{html.escape(trend_note)}</div>'
    elif len(v) >= 2:
        _tr = f'<div class="trend">Month 1: {v[0]:.1f}% &rarr; latest month: {v[-1]:.1f}%</div>'
    else:
        _tr = ""
    return (f'<div class="kpi-card">{_p}<div class="k">{html.escape(label)}</div><div class="row"><div class="v">{html.escape(value)}</div>{_sp}</div>{_tr}'
            f'<div class="d">{html.escape(note)}</div></div>')


def ring_card(label, pct, note=""):
    p_ = 0.0 if pct is None or pd.isna(pct) else max(0.0, min(100.0, float(pct)))
    txt = "n/a" if pct is None or pd.isna(pct) else f"{p_:.0f}%"
    return (f'<div class="kpi-card"><div class="k">{html.escape(label)}</div>'
            f'<svg width="100%" height="120" viewBox="0 0 120 120"><circle cx="60" cy="60" r="42" fill="none" stroke="var(--seg-bg)" stroke-width="14"/>'
            f'<circle cx="60" cy="60" r="42" fill="none" stroke="var(--accent)" stroke-width="14" stroke-linecap="round" stroke-dasharray="{p_ / 100 * 264:.1f} 264" transform="rotate(-90 60 60)"/>'
            f'<text x="60" y="67" text-anchor="middle" font-size="22" font-weight="700" fill="var(--text)">{txt}</text></svg>'
            f'<div class="d">{html.escape(note)}</div></div>')


def drivers_card(label, items):
    items = [(n, float(v)) for n, v in items if v and not pd.isna(v) and v > 0]
    items.sort(key=lambda t: -t[1])
    top = items[0][1] if items else 1.0
    rows = "".join(f'<li><span class="n">{i + 1}</span>{html.escape(n)}<div class="bar"><i style="width:{v / top * 100:.0f}%"></i></div><span class="amt">${v:,.0f}</span></li>'
                   for i, (n, v) in enumerate(items[:5]))
    return f'<div class="kpi-card"><div class="k">{html.escape(label)}</div><ul class="rank">{rows or "<li>No cost yet</li>"}</ul></div>'


def chapter_head(anchor, num, label, question_html, lede):
    return (f'<div class="chapter" id="{anchor}"><div class="eyebrow">Chapter {num} &middot; {html.escape(label)}</div>'
            f'<h2>{question_html}</h2><p class="lede">{html.escape(lede)}</p></div>')


if GUIDED:
    # Presentation-only redesign: illustration and story copy. Calculation engine is unchanged.
    _hero_text, _hero_art = st.columns([1.55, 0.75], gap="large", vertical_alignment="center")
    with _hero_text:
        st.markdown("""
<div class="hero">
  <div class="eyebrow">AI Deal Desk &middot; A founder's guide to AI unit economics</div>
  <h1>Will my AI product <span class="accent">make money?</span></h1>
  <p class="sub">Alex has an AI product, a growing user base, and a bill that keeps changing. The question is not just whether customers pay — it is whether the product can earn a healthy margin as usage grows.</p>
  <div class="skipbar">Follow the story: <a href="#ch0">Start with the margins</a> <a href="#ch1">Then price one prompt</a> <a href="#deeper">Explore the full model</a></div>
</div>
""", unsafe_allow_html=True)
    with _hero_art:
        st.markdown("""
<div class="founder-art" role="img" aria-label="Illustration of a founder reviewing growth and profit charts">
<svg viewBox="0 0 320 230" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
<rect x="34" y="20" width="252" height="178" rx="18" fill="var(--card-bg)" stroke="var(--card-border)" stroke-width="2"/>
<rect x="56" y="43" width="90" height="10" rx="5" fill="var(--seg-bg)"/><rect x="56" y="61" width="54" height="7" rx="3.5" fill="var(--seg-bg)"/>
<path d="M58 157 L95 132 L126 141 L161 99 L193 113 L236 76" fill="none" stroke="var(--accent)" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/><circle cx="236" cy="76" r="7" fill="var(--accent)"/>
<path d="M58 170 H258" stroke="var(--card-border)" stroke-width="2"/><rect x="185" y="140" width="20" height="30" rx="4" fill="var(--chip-bg)"/><rect x="213" y="123" width="20" height="47" rx="4" fill="var(--chip-bg)"/><rect x="241" y="104" width="17" height="66" rx="4" fill="var(--accent)" opacity=".8"/>
<circle cx="58" cy="204" r="13" fill="var(--chip-bg)"/><path d="M52 204 l4 4 8 -9" fill="none" stroke="var(--accent)" stroke-width="3" stroke-linecap="round"/><text x="80" y="209" font-size="12" fill="var(--label)" font-family="sans-serif">Growth → profit?</text>
</svg></div><div class="founder-question"><em>The founder's question: can growth turn into profit?</em></div>
""", unsafe_allow_html=True)
    st.markdown(f'<div class="chapter" id="ch0" style="border-top:0;margin-top:6px"><div class="eyebrow">Step 1 &middot; Start here</div>'
                f'<h2>More users. More revenue. <span class="accent"><em>More profit?</em></span></h2>'
                f'<p class="lede">Meet Alex, an AI product founder. Before changing prices or chasing growth, Alex needs to understand three different margin views. Choose a starting profile; the cards update using the model below.</p></div>',
                unsafe_allow_html=True)
    slot_pick = st.container()
    slot_cards = st.container()
    st.markdown(chapter_head("ch1", 1, "The first bill", 'What does <span class="accent">one prompt</span> cost?',
                             "Alex starts with a simple question: what happens when someone presses ‘Ask’? Input tokens, output tokens, model choice and usage discounts all affect the bill. Change the usage control below and see how the estimate moves."),
                unsafe_allow_html=True)
    slot_ch1_ctl = st.container()
    slot_ch1_out = st.container()
    slot_chat = st.container()
    st.markdown('<div class="chapter" id="deeper"><div class="eyebrow">The working model</div><h2>Now test the <span class="accent">business decisions</span></h2>'
                '<p class="lede">The story starts with margins and the cost of one prompt. The tabs below let you investigate who is using the AI budget, forecast cost and cash, test pricing and usage levers, and evaluate a customer deal. These views use the same calculations.</p></div>',
                unsafe_allow_html=True)
else:
    st.markdown("""
<div class="compact-header">
    <h1>⚡ The AI Unit Economics Deal Desk</h1>
    <p>Founder presets, team contract cohorts, typical/heavy-user costs, like-for-like pricing simulation, credit runway and cash timing.</p>
</div>
""", unsafe_allow_html=True)


OTHER_BASIS = ["$ / month", "% of revenue"]
TREAT_LABEL = {"COGS": "COGS", "S&M": "sales & marketing", "SPLIT": "trials in sales & marketing, freemium in COGS"}


def ai_unit_cost_index(p, k):
    """Multiplier on AI inference (API) cost in forecast month k (1-based): efficiency programme x one-time vendor price change.
    Efficiency: from month `eff_start`, cost per prompt falls `eff_mom` per month, never below (1 - `eff_cap`) of the starting level.
    Vendor price change: a one-time step of `price_step_pct` (signed) from month `price_step_month`. 1.0 = no change."""
    eff = 1.0
    if p["eff_mom"] > 0:
        n_eff = max(0, k - int(p["eff_start"]) + 1)
        if n_eff > 0:
            eff = max(1.0 - p["eff_cap"], (1.0 - p["eff_mom"]) ** n_eff)
    step = (1.0 + p["price_step_pct"]) if (p["price_step_pct"] != 0 and k >= int(p["price_step_month"])) else 1.0
    return eff * step


def ai_path_is_flat(p):
    return p["eff_mom"] <= 0 and p["price_step_pct"] == 0


def ai_path_text(p, horizon=None):
    """Plain-English description of the AI cost path, built from live inputs."""
    if ai_path_is_flat(p):
        return "none (token prices held flat for the whole horizon)"
    parts = []
    if p["eff_mom"] > 0:
        parts.append(f"{p['eff_mom'] * 100:.1f}% MoM reduction in inference cost per prompt from month {int(p['eff_start'])} "
                     f"(capped at {p['eff_cap'] * 100:.0f}% cumulative)")
    if p["price_step_pct"] != 0:
        parts.append(f"{p['price_step_pct'] * 100:+.0f}% one-time vendor price change from month {int(p['price_step_month'])}")
    txt = "; ".join(parts)
    if horizon:
        txt += f"; net AI unit-cost index x{ai_unit_cost_index(p, int(horizon)):.2f} by month {int(horizon)}"
    return txt


def token_price_note(p, horizon=None):
    """Token-price disclaimer built from the LIVE inputs, so it can never quote stale prices or a stale cost path."""
    base = (f"<b>Token-price assumption:</b> frontier ${p['f_in']:.2f} in / ${p['f_out']:.2f} out and standard ${p['s_in']:.4f} in / ${p['s_out']:.4f} out "
            "per 1M tokens are illustrative, user-editable placeholders, not live vendor quotes. Verify against current vendor rate cards. ")
    if ai_path_is_flat(p):
        path = ("They are held <b>flat for the entire horizon</b> (no price decline or increase is modeled) while tokens per prompt grow; "
                "right-sizing, model-routing gains and vendor price moves are not reflected unless you set the AI cost path in the sidebar. ")
    else:
        path = (f"An <b>AI cost path</b> is applied on top of those prices: {ai_path_text(p, horizon)}. "
                "This is a planning assumption, not a vendor commitment: efficiency gains must be delivered and vendor prices can move either way. ")
    tail = ("COGS covers AI inference, vector DB, hosting, payment fees and any other COGS lines you add; costs you do not enter are not in the margin. "
            "Read margins as a planning range, not a forecast.")
    return base + path + tail


# ---------- feedback form helpers ----------
FEEDBACK_CATEGORIES = ["🐛 Bug report or performance issue", "💡 Feature idea or suggestion", "❓ Question about how it works"]
FEEDBACK_WELCOME = """
**Welcome to the Deal Desk Feedback Space! 👋**

Thank you for trying The AI Unit Economics Deal Desk! This is an independent, non-commercial portfolio project built for fun and learning.
There are no paid plans, and your feedback directly helps me learn and shape future experiments.

Use this form to share:

* 🐛 Bug reports or performance issues
* 💡 Feature ideas and suggestions
* ❓ General questions about how it works

*A note on the numbers: all figures are illustrative planning estimates, not financial advice. Please don't include confidential or personal financial details in your message.*

As this is a solo personal project, I review submissions as often as I can. Thank you for your support and patience!
"""
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def feedback_config():
    """Where feedback goes, read from Streamlit secrets so no address or key lives in the code:
    [feedback] webhook_url = "https://..."  (required to enable the form)   form_url = "..."  email = "..."  (optional fallbacks)."""
    out = {"webhook_url": "", "form_url": "", "email": ""}
    try:
        fb = st.secrets.get("feedback", {})
        for k in out:
            out[k] = str(fb.get(k, "") or "").strip()
    except Exception:
        pass
    return out


def clean_text(txt, limit):
    txt = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(txt or "")).strip()
    return txt[:limit]


def safe_cell(v):
    """Stop spreadsheet apps from running text as a formula: prefix a quote when a string starts with = + - @ or a control character."""
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return v


def safe_frame(d):
    d = d.copy()
    d.columns = [safe_cell(str(c)) for c in d.columns]
    for c in d.columns:
        if not pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].map(safe_cell)
    return d


def validate_feedback(category, message, email):
    """Return an error string, or None when the submission is acceptable."""
    if category not in FEEDBACK_CATEGORIES:
        return "Please choose what you would like to share."
    if len(clean_text(message, 5000)) < 10:
        return "Please write a little more (at least 10 characters) so I can act on it."
    if email and not _EMAIL_RE.match(email.strip()):
        return "That email address doesn't look right. Leave it blank if you'd rather not share one."
    return None


def post_feedback(url, payload, timeout=8):
    """POST the payload as JSON to the configured webhook (Google Apps Script, Formspree, Zapier, ...). Returns (ok, message).
    Only https URLs are used (plain http is allowed for localhost testing only)."""
    if not (url.startswith("https://") or url.startswith("http://127.0.0.1") or url.startswith("http://localhost")):
        return False, "The feedback destination must be an https URL."
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if 200 <= resp.status < 300:
                return True, "sent"
            return False, f"The feedback service answered with status {resp.status}."
    except urllib.error.HTTPError as e:
        return False, f"The feedback service rejected the message (HTTP {e.code})."
    except urllib.error.URLError as e:
        return False, f"Could not reach the feedback service ({type(e.reason).__name__})."
    except Exception as e:  # timeouts, bad URLs, anything unexpected: never crash the app over a feedback form
        return False, f"Could not send ({type(e).__name__})."


def _supports_stretch(fn):
    """True only if this Streamlit build's `fn` really accepts width='stretch' (older builds take an int or only **kwargs)."""
    try:
        prm = inspect.signature(fn).parameters.get("width")
    except (TypeError, ValueError):
        return False
    return prm is not None and ("stretch" in str(prm.annotation) or prm.default == "stretch")


def _stretch(fn, obj, **kw):
    """width='stretch' on current Streamlit; use_container_width on older versions."""
    if _supports_stretch(fn):
        try:
            return fn(obj, width="stretch", **kw)
        except Exception:
            pass
    return fn(obj, use_container_width=True, **kw)


def show_df(obj, **kw):
    return _stretch(st.dataframe, obj, **kw)


def show_chart(fig):
    return _stretch(st.plotly_chart, fig)


def style_frame(df):
    """Format a DataFrame from column-name conventions (no hand-typed keys that can silently miss)."""
    fmt = {}
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        if "1,000 Prompts" in c:
            fmt[c] = "${:,.4f}"
        elif "/ Paid" in c or "/ Account-Month" in c:
            fmt[c] = "${:,.2f}"
        elif c.startswith("Δ") and "(pp)" in c:
            fmt[c] = "{:+.1f}pp"
        elif c.startswith("Δ") and "($)" in c:
            fmt[c] = "${:+,.0f}"
        elif "(%)" in c:
            fmt[c] = "{:.1f}%"
        elif "($)" in c:
            fmt[c] = "${:,.0f}"
        elif "Index" in c:
            fmt[c] = "{:.3f}"
        elif "Average" in c or c.startswith("Avg"):
            fmt[c] = "{:,.1f}"
        else:
            fmt[c] = "{:,.0f}"
    return df.style.format(fmt, na_rep="—")


# ==========================================
# 1. SIDEBAR ASSUMPTIONS
# ==========================================
FOUNDER_PRESETS = {
    # p1_vol / p2_vol: seasonal peak prompts / user / day. Calibrated at ~1.7x and ~2.1x each preset's own
    # baseline (previously a fixed 35 / 45 for every preset, which made the baseline slider irrelevant).
    "Pre-launch": {
        "dau": 100, "pro": 0, "ent": 0, "growth": 5.0, "deals": 0.0,
        "prompts": 3, "infra": 500.0, "ai_credits": 0.0, "cloud_credits": 0.0,
        "front_mix": 0.70, "cache_rate": 0.10, "free_mult": 0.50,
        "p1_vol": 5, "p2_vol": 6,
    },
    "Early traction": {
        "dau": 5000, "pro": 150, "ent": 2, "growth": 4.0, "deals": 0.5,
        "prompts": 12, "infra": 2500.0, "ai_credits": 10000.0, "cloud_credits": 15000.0,
        "front_mix": 0.50, "cache_rate": 0.20, "free_mult": 0.40,
        "p1_vol": 20, "p2_vol": 25,
    },
    "Scaling": {
        "dau": 12000, "pro": 600, "ent": 10, "growth": 5.0, "deals": 2.0,
        "prompts": 10, "infra": 6000.0, "ai_credits": 25000.0, "cloud_credits": 40000.0,
        "front_mix": 0.30, "cache_rate": 0.45, "free_mult": 0.20,
        "p1_vol": 17, "p2_vol": 21,
    },
}

_STAGE_DESC = {
    "Pre-launch": "Alex is testing the idea. There may be little or no revenue yet, but every prompt still has a cost.",
    "Early traction": "Alex has first paying customers. Now the question is whether subscription revenue covers the cost of serving them.",
    "Scaling": "Alex is growing usage and revenue. The challenge is making sure AI and infrastructure costs do not grow faster than sales.",
}
_stage_default = st.session_state.get("_stage", "Early traction")
def _set_stage(name):
    st.session_state["_stage"] = name


if GUIDED:
    with slot_pick:
        _cur = st.session_state.get("_stage", "Early traction")
        with _keyed_container("profile"):
            _pcols = st.columns(3, gap="small")
            for _pc, _nm in zip(_pcols, ["Pre-launch", "Early traction", "Scaling"]):
                with _pc:
                    st.button(_nm, key=f"stage_{_nm}", type="primary" if _nm == _cur else "secondary", on_click=_set_stage, args=(_nm,), use_container_width=True)
                    st.markdown(f'<div class="stage-story">{html.escape(_STAGE_DESC[_nm])}</div>', unsafe_allow_html=True)
    founder_stage = st.session_state.get("_stage", "Early traction")
    st.sidebar.header("Forecast setup")
else:
    st.sidebar.header("0. Founder Onboarding")
    founder_stage = st.sidebar.selectbox(
        "Choose a starting profile", ["Early traction", "Pre-launch", "Scaling"],
        index=["Early traction", "Pre-launch", "Scaling"].index(_stage_default), key="stage_side",
        help="Presets populate starting values only. All assumptions remain editable.",
    )
st.session_state["_stage"] = founder_stage
preset = FOUNDER_PRESETS[founder_stage]
current_now = datetime.datetime.now()
_setup_a, _setup_b = st.sidebar.columns(2, gap="small")
with _setup_a:
    start_year = int(st.number_input("Start year", min_value=2024, max_value=2035, value=current_now.year))
with _setup_b:
    start_month = int(st.selectbox("Start month", list(range(1, 13)), index=current_now.month - 1, format_func=lambda m: calendar.month_abbr[m]))
forecast_horizon = int(st.sidebar.select_slider("Forecast horizon", options=[3, 6, 9, 12, 18, 24, 36], value=12, format_func=lambda m: f"{m} months"))

st.sidebar.markdown("---")
advanced_mode = st.sidebar.toggle(
    "🛠️ Enable Advanced Deal Desk Parameters", value=False,
    help="Start with the simpler founder view. Turn this on to edit every model assumption.",
)

# Engine defaults. Advanced widgets override these; simple mode uses them as-is.
DEFAULTS = {
    # funnel & revenue
    "paid_dau_factor": 0.40, "new_conv": 0.04, "base_conv": 0.005, "churn": 0.05,
    "arpu_pro": 30.0, "arpu_ent": 1000.0, "pro_expansion_pct": 0.10, "fee_pct": 0.029,
    # enterprise contracts
    "ent_term": 12, "ent_renewal_rate": 0.80, "ent_in_term_churn": 0.0, "ent_expansion": 0.015,
    "ent_seats": 20, "ent_prompts": 3000, "impl_fee": 0.0, "impl_cost": 0.0, "impl_rec_months": 1,
    # usage
    "p1_vol": preset["p1_vol"], "p1_month": 3, "p1_width": 1.5, "p2_vol": preset["p2_vol"], "p2_month": 11, "p2_width": 2.0,
    "heavy_share": 0.20, "heavy_mode": "Multiplier", "heavy_mult": 4.0, "pareto_share": 0.80,
    "free_mult": preset["free_mult"], "in_tok": 1000, "out_tok": 400, "in_grow": 0.03, "out_grow": 0.0,
    # routing & token pricing
    "front_mix": preset["front_mix"], "f_in": 2.50, "f_out": 10.00, "s_in": 0.15, "s_out": 0.60,
    "cache_rate": preset["cache_rate"], "cache_ratio": 0.20, "batch_share": 0.20, "batch_ratio": 0.50, "overhead": 0.0,
    # infrastructure & credits
    "base_infra": preset["infra"], "step_type": "DAU Threshold", "step_thresh": 20000.0, "step_cost": 1500.0,
    "vec_cost": 0.05, "ai_credits": preset["ai_credits"], "cloud_credits": preset["cloud_credits"],
    "ai_expiry": 0, "cloud_expiry": 0, "invoice_lag": 1, "ai_elig": 1.0, "cloud_elig": 1.0,
    # scenario hooks (1.0 = unchanged)
    "usage_mult": 1.0, "price_mult": 1.0,
    # cost treatment, fully loaded view and AI cost path (defaults reproduce v2.5 headline numbers)
    "free_treatment": "COGS", "infra_alloc": "paid", "other_cogs": [],
    # traffic mix: trial share of free traffic, separate free-tier routing (None = same as main), internal traffic
    "trial_share": 0.25, "front_mix_free": None, "internal_pct": 0.0, "internal_treatment": "RD",
    "eff_mom": 0.0, "eff_start": 13, "eff_cap": 0.50, "price_step_pct": 0.0, "price_step_month": 13,
}
cfg = dict(DEFAULTS)

st.sidebar.header("Key assumptions")
st.sidebar.caption("Five high-impact inputs. Open More assumptions for the rest.")
_key_left, _key_right = st.sidebar.columns(2, gap="small")
with _key_left:
    starting_dau = st.number_input(
        "Daily active users", min_value=0, value=preset["dau"], step=500, key=f"dau_{founder_stage}",
        help="DAU in the first forecast month. Growth applies from month 2 onward.",
    )
    starting_ent_accounts = st.number_input("Team / SMB accounts", min_value=0, value=preset["ent"], step=10, key=f"ent_{founder_stage}")
    net_new_ent_contracts = st.number_input(
        "New team deals / month", min_value=0.0, value=float(preset["deals"]), step=0.5, key=f"deals_{founder_stage}",
        help="Sales-led pipeline velocity. Fractional values are allowed.",
    )
with _key_right:
    starting_pro_subs = st.number_input("Pro subscribers", min_value=0, value=preset["pro"], step=50, key=f"pro_{founder_stage}")
    dau_monthly_growth = st.slider(
        "Monthly DAU growth", 0.0, 25.0, float(preset["growth"]), 0.5, key=f"growth_{founder_stage}",
        help="Growth of total daily active users, applied from month 2 onward.",
    ) / 100
    _prompts_help = "Typical paying user's prompts per day. Higher usage means a higher inference bill."
    baseline_prompts = st.slider("Prompts / paid user / day", 1, 20, int(preset["prompts"]), key=f"prompts_{founder_stage}", help=_prompts_help)

with st.sidebar.expander("More assumptions · targets and cost treatment", expanded=False):
    st.header("Targets")
    margin_target = st.slider("Target Gross Margin (%)", 0, 90, 60)
    budget_monthly = st.number_input("Monthly COGS Budget Cap ($; 0 = none)", min_value=0.0, value=0.0, step=1000.0)

    st.header("3. Cost Treatment & AI Cost Path")
    _treat_opts = {"All in COGS": "COGS", "All in sales & marketing": "S&M", "Trials in S&M, freemium in COGS": "SPLIT"}
    _treat = st.radio(
        "Free / trial inference cost is booked to", list(_treat_opts), index=0,
        help="Free and trial traffic earns no revenue. Many companies book its inference cost to sales & marketing (it is acquisition spend); others keep it in COGS, "
             "ideally as its own sub-line; some book trials to S&M and the perpetual free tier to COGS. Either way it is a real cost: this switch only moves it. "
             "Paying customers' margin never carries it. The third option uses the trial share set just below.",
    )
    cfg["free_treatment"] = _treat_opts[_treat]
    cfg["trial_share"] = st.slider("Trials as share of free-tier traffic (%)", 0, 100, 25, 5,
        help="Illustrative placeholder. Share of free-tier prompts that come from time-limited trials of a paid plan (the rest is the perpetual free tier). "
             "Drives the trial vs freemium cost split, and what moves to S&M under 'Trials in S&M, freemium in COGS'. Replace it with your own funnel data.") / 100
    with st.expander("📉 AI cost path (efficiency & vendor pricing)"):
        st.caption("Optional. Leave at 0 for flat token prices. These are planning assumptions: efficiency gains must be delivered, and vendor prices can move either way.")
        cfg["eff_mom"] = st.slider("MoM reduction in AI cost per prompt (%)", 0.0, 10.0, 0.0, 0.5,
                                   help="Right-sizing, model routing, caching, prompt trimming. Applies to AI inference (API) cost only, compounding monthly from the start month.") / 100
        cfg["eff_start"] = int(st.slider("Reduction starts in forecast month", 1, 36, 13,
                                         help="1 = first forecast month. The first reduction lands in this month."))
        cfg["eff_cap"] = st.slider("Maximum cumulative reduction (%)", 0, 90, 50, 5,
                                   help="Floor on the efficiency path: cost per prompt never falls below (1 - cap) of its starting level.") / 100
        cfg["price_step_pct"] = st.slider("One-time vendor price change (%)", -50, 100, 0, 5,
                                          help="A step change in vendor list prices (OpenAI, Anthropic, etc.): negative = price cut, positive = increase. Applies on top of the efficiency path.") / 100
        cfg["price_step_month"] = int(st.slider("Vendor price change takes effect in forecast month", 1, 36, 13))
        st.caption(f"Net AI unit-cost index at month {forecast_horizon}: **x{ai_unit_cost_index(cfg, forecast_horizon):.2f}** (1.00 = today's prices)")


if advanced_mode:
    with st.sidebar.expander("📂 A. Subscriber Funnel & Revenue"):
        cfg["paid_dau_factor"] = st.slider("Paid Subscribers Active Daily (%)", 0, 100, 40, 5) / 100
        cfg["new_conv"] = st.slider("New-User Conversion to Pro (%)", 0.0, 50.0, 4.0, 0.5) / 100
        cfg["base_conv"] = st.slider("Existing Free-Pool Conversion (% / month)", 0.0, 10.0, 0.5, 0.1,
                                     help="Monthly probability that an existing free active user upgrades. Applied to the free pool net of paid actives.") / 100
        cfg["churn"] = st.slider("Monthly Pro Churn (%)", 0.0, 15.0, 5.0, 0.5) / 100
        cfg["arpu_pro"] = st.number_input("Pro Tier Monthly ARPU ($)", min_value=0.0, value=30.0, step=5.0)
        cfg["pro_expansion_pct"] = st.slider("Pro Overage / Expansion Revenue (% of Pro base)", 0.0, 50.0, 10.0, 1.0,
                                             help="Applies to Pro revenue only. Enterprise expansion is modeled per account below, so it is not double counted.") / 100
        cfg["fee_pct"] = st.slider("Payment Processing Fee (% of Revenue)", 0.0, 30.0, 2.9, 0.1) / 100

    with st.sidebar.expander("📂 B. Team / SMB Contract Economics"):
        cfg["arpu_ent"] = st.number_input(
            "Team / Small Business Monthly Value / Account ($)", 
            min_value=0.0, 
            value=1000.0, 
            step=100.0,
            help="Modeled at $1,000/mo ($12k ARR) for a team tier (e.g. 20 seats at $50/seat). Scale higher for large enterprise contracts."
        )
        cfg["ent_term"] = int(st.number_input("Contract Term (months)", min_value=1, max_value=120, value=12, step=1))
        cfg["ent_renewal_rate"] = st.slider("Renewal Rate at Term End (%)", 0, 100, 80, 5,
                                            help="Share of accounts reaching term end that renew for another full term.") / 100
        cfg["ent_in_term_churn"] = st.slider("In-Term Logo Churn (% / month)", 0.0, 10.0, 0.0, 0.5,
                                             help="Committed contracts normally do not churn mid-term. Leave at 0 unless you have termination-for-convenience.") / 100
        cfg["ent_expansion"] = st.slider("Seat / Usage Expansion per Account (% / month)", 0.0, 10.0, 1.5, 0.5,
                                         help="Grows contract value, seats and prompts of an existing account. Does NOT add accounts. Timing assumption: expansion starts immediately and compounds monthly; real expansion usually arrives through contract amendments or seat ramps.") / 100
        cfg["ent_seats"] = int(st.number_input("Contracted Seats per Account (at signing)", min_value=1, value=20, step=5))
        cfg["ent_prompts"] = int(st.number_input("Team / SMB Prompts per Account / Month (at signing)", min_value=0, value=3000, step=500))
        cfg["impl_fee"] = st.number_input("Implementation Fee per New Deal ($)", min_value=0.0, value=0.0, step=500.0)
        cfg["impl_rec_months"] = int(st.number_input("Implementation Fee Recognition Period (months; 1 = upfront)", min_value=1, max_value=36, value=1, step=1))
        cfg["impl_cost"] = st.number_input("Implementation Delivery Cost per New Deal ($)", min_value=0.0, value=0.0, step=500.0)

    with st.sidebar.expander("📂 C. Usage Peaks & Heavy-User Skew"):
        st.markdown("**Peak 1 (calendar month)**")
        cfg["p1_vol"] = st.slider("Peak 1 Prompts / User / Day", 1, 100, int(preset["p1_vol"]), key=f"p1_{founder_stage}",
                                  help="Absolute prompts per active user per day at the peak (not a multiplier). Presets use ~1.7x / ~2.1x their own baseline; keep peaks in proportion to the baseline or free-tier usage will dominate cost.")
        cfg["p1_month"] = st.selectbox("Peak 1 Month", list(range(1, 13)), index=2, format_func=lambda m: calendar.month_name[m])
        cfg["p1_width"] = st.slider("Peak 1 Spread (σ, months)", 0.5, 4.0, 1.5, 0.5)
        st.markdown("**Peak 2 (calendar month)**")
        cfg["p2_vol"] = st.slider("Peak 2 Prompts / User / Day", 1, 100, int(preset["p2_vol"]), key=f"p2_{founder_stage}")
        cfg["p2_month"] = st.selectbox("Peak 2 Month", list(range(1, 13)), index=10, format_func=lambda m: calendar.month_name[m])
        cfg["p2_width"] = st.slider("Peak 2 Spread (σ, months)", 0.5, 4.0, 2.0, 0.5)
        cfg["heavy_share"] = st.slider("Heavy Users as Share of Pro Users (%)", 0, 50, 20, 5) / 100
        cfg["heavy_mode"] = st.selectbox("Heavy-User Calibration", ["Multiplier", "Pareto prompt share"],
                                         help="Multiplier: heavy users prompt N x typical. Pareto: heavy users generate X% of Pro prompts; the multiplier is derived.")
        if cfg["heavy_mode"] == "Multiplier":
            cfg["heavy_mult"] = st.slider("Heavy-User Prompt Multiplier (vs. Typical)", 1.0, 20.0, 4.0, 0.5)
        else:
            cfg["pareto_share"] = st.slider("Prompt Share Generated by Heavy Users (%)", 50, 95, 80, 5) / 100
        cfg["free_mult"] = st.slider("Free Tier Usage Multiplier (%)", 0, 100, int(preset["free_mult"] * 100)) / 100
        cfg["in_tok"] = st.number_input("Initial Input Tokens / Prompt", min_value=0, value=1000, step=100)
        cfg["out_tok"] = st.number_input("Initial Output Tokens / Prompt", min_value=0, value=400, step=50)
        cfg["in_grow"] = st.slider("MoM Input Token Growth (%)", 0.0, 20.0, 3.0, 0.5,
                                   help="Compounds monthly for the whole horizon. Token PRICES are held flat, so cost per prompt rises with this setting.") / 100
        cfg["out_grow"] = st.slider("MoM Output Token Growth (%)", 0.0, 20.0, 0.0, 0.5) / 100

    with st.sidebar.expander("📂 D. Token Routing & Pricing Rules"):
        st.caption("⚠️ Token prices below are illustrative placeholders, not live vendor quotes, and are held flat for the whole forecast (no price decline modeled). Replace with your vendor's current rate card.")
        _price_help = "Illustrative placeholder, not a live vendor quote. Held flat across the horizon; scenarios 8-9 apply a flat % change to every month."
        cfg["front_mix"] = st.slider("Frontier Model Traffic Share (%)", 0, 100, int(preset["front_mix"] * 100)) / 100
        cfg["f_in"] = st.number_input("Frontier Input Price ($ / 1M)", min_value=0.0, value=2.50, step=0.25, help=_price_help)
        cfg["f_out"] = st.number_input("Frontier Output Price ($ / 1M)", min_value=0.0, value=10.00, step=0.50, help=_price_help)
        cfg["s_in"] = st.number_input("Standard Input Price ($ / 1M)", min_value=0.0, value=0.15, step=0.05, format="%.4f", help=_price_help)
        cfg["s_out"] = st.number_input("Standard Output Price ($ / 1M)", min_value=0.0, value=0.60, step=0.05, format="%.4f", help=_price_help)
        cfg["cache_rate"] = st.slider("Eligible Input Tokens Cached (%)", 0, 90, int(preset["cache_rate"] * 100)) / 100
        cfg["cache_ratio"] = st.slider("Cached Price Ratio (% of regular)", 0, 100, 20) / 100
        cfg["batch_share"] = st.slider("Traffic Using Batch Pricing (%)", 0, 80, 20) / 100
        cfg["batch_ratio"] = st.slider("Batch Price Ratio (% of regular)", 0, 100, 50) / 100
        cfg["overhead"] = st.slider("Inference Overhead (%)", 0, 100, 0) / 100

    with st.sidebar.expander("📂 E. Infrastructure & Credit Drawdown"):
        cfg["base_infra"] = st.number_input("Base Fixed Monthly Infra ($)", min_value=0.0, value=float(preset["infra"]), step=500.0, key=f"infra_{founder_stage}")
        cfg["step_type"] = st.selectbox("Step-Cost Trigger Basis", ["DAU Threshold", "Paid Subscribers Threshold", "Monthly Prompts (Millions)"])
        if cfg["step_type"] == "DAU Threshold":
            cfg["step_thresh"] = float(st.number_input("DAU Capacity Block (Users)", min_value=1000, value=20000, step=5000))
        elif cfg["step_type"] == "Paid Subscribers Threshold":
            cfg["step_thresh"] = float(st.number_input("Subscriber Capacity Block (Pro subs + team accounts)", min_value=500, value=5000, step=1000))
        else:
            cfg["step_thresh"] = float(st.number_input("Monthly Prompt Block (Millions)", min_value=1, value=20, step=5)) * 1_000_000
        cfg["step_cost"] = st.number_input("Infra Step-Up Cost ($ per block)", min_value=0.0, value=1500.0, step=250.0)
        cfg["vec_cost"] = st.number_input("Vector DB Cost / 1k Prompts ($)", min_value=0.0, value=0.05, step=0.01)
        cfg["ai_credits"] = st.number_input("Starting AI Vendor Credits ($)", min_value=0.0, value=float(preset["ai_credits"]), step=2500.0, key=f"aicred_{founder_stage}")
        cfg["cloud_credits"] = st.number_input("Starting Cloud Hosting Credits ($)", min_value=0.0, value=float(preset["cloud_credits"]), step=5000.0, key=f"cloudcred_{founder_stage}")
        cfg["ai_expiry"] = int(st.number_input("AI Credits Usable Through Forecast Month (0 = no expiry)", min_value=0, max_value=36, value=0, step=1))
        cfg["cloud_expiry"] = int(st.number_input("Cloud Credits Usable Through Forecast Month (0 = no expiry)", min_value=0, max_value=36, value=0, step=1))
        cfg["invoice_lag"] = int(st.selectbox("Vendor Invoice Payment Lag (months)", [0, 1, 2, 3], index=1,
                                              help="Vendor invoices (API, vector, infra) are paid this many months late. Payment fees and implementation costs are paid immediately."))
        cfg["ai_elig"] = st.slider("AI Credit Eligible Share of API Cost (%)", 0, 100, 100) / 100
        cfg["cloud_elig"] = st.slider("Cloud Credit Eligible Share of Vector + Infra (%)", 0, 100, 100) / 100
        _alloc = st.selectbox("Fixed Infrastructure Attributed to Free Tier", ["All to paid customers (conservative)", "Split by prompt share"], index=0,
                              help="Free-tier API + vector cost always follows prompt share. Fixed hosting is needed to serve paying customers anyway, so by default it stays with them; "
                                   "'Split by prompt share' assigns the free tier its share, which raises paid-customer margin.")
        cfg["infra_alloc"] = "paid" if _alloc.startswith("All") else "prompt"

    with st.sidebar.expander("📂 F. Traffic Mix: Trials, Tiers & Internal"):
        if st.checkbox("Route the free tier on a different model mix", value=False,
                       help="Off: free traffic uses the same frontier / standard routing as paying customers. On: set a separate frontier share for free traffic (for example a cheaper model for free users)."):
            cfg["front_mix_free"] = st.slider("Free-Tier Frontier Model Share (%)", 0, 100, int(cfg["front_mix"] * 100)) / 100
        cfg["internal_pct"] = st.slider("Internal / Non-Customer Traffic (% of customer prompts)", 0, 50, 0, 1,
                                        help="Staff use, evals, QA, staging and synthetic tests. Priced like paying-customer traffic, earns no revenue.") / 100
        _it = st.selectbox("Internal Traffic Cost Booked To", ["R&D / G&A (usual)", "COGS"], index=0,
                           help="Internal and eval traffic is usually R&D or G&A, not COGS. Choose COGS to be conservative.")
        cfg["internal_treatment"] = "RD" if _it.startswith("R&D") else "COGS"

st.sidebar.caption("💬 Feedback, bug reports or ideas? See the **Feedback** tab.")

P = dict(cfg)
P.update({
    "start_year": start_year, "start_month": start_month, "horizon": forecast_horizon,
    "dau": float(starting_dau), "pro": float(starting_pro_subs), "ent": float(starting_ent_accounts),
    "dau_growth": float(dau_monthly_growth), "net_new_ent": float(net_new_ent_contracts),
    "base_prompts": float(baseline_prompts),
})


# ==========================================
# 2. ENGINE
# ==========================================
def validate_params(p):
    errs = []
    if not (1 <= int(p["horizon"]) <= 36):
        errs.append("Forecast horizon must be between 1 and 36 months.")
    nonneg = ["dau", "pro", "ent", "net_new_ent", "arpu_pro", "arpu_ent", "base_prompts", "in_tok", "out_tok",
              "f_in", "f_out", "s_in", "s_out", "base_infra", "step_cost", "vec_cost", "ai_credits", "cloud_credits",
              "ent_prompts", "impl_fee", "impl_cost", "usage_mult", "price_mult", "dau_growth", "ent_expansion",
              "pro_expansion_pct", "fee_pct", "overhead", "in_grow", "out_grow"]
    for k in nonneg:
        if p[k] < 0:
            errs.append(f"'{k}' must be >= 0.")
    for k in ["paid_dau_factor", "new_conv", "base_conv", "churn", "ent_renewal_rate", "ent_in_term_churn", "cache_rate",
              "cache_ratio", "batch_share", "batch_ratio", "front_mix", "ai_elig", "cloud_elig", "free_mult", "heavy_share"]:
        if not (0.0 <= p[k] <= 1.0):
            errs.append(f"'{k}' must be between 0 and 1.")
    if int(p["ent_term"]) < 1:
        errs.append("Team / SMB contract term must be at least 1 month.")
    if int(p["ent_seats"]) < 1:
        errs.append("Team / SMB seats per account must be at least 1.")
    if int(p["impl_rec_months"]) < 1:
        errs.append("Implementation recognition period must be at least 1 month.")
    if p["p1_width"] <= 0 or p["p2_width"] <= 0:
        errs.append("Peak spread (sigma) must be positive.")
    if p["step_thresh"] <= 0:
        errs.append("Infrastructure capacity block must be positive.")
    if p["heavy_mode"] == "Pareto prompt share" and p["pareto_share"] >= 1.0:
        errs.append("Pareto prompt share must be below 100%.")
    if int(p["invoice_lag"]) not in (0, 1, 2, 3):
        errs.append("Invoice lag must be 0-3 months.")
    if p["free_treatment"] not in ("COGS", "S&M", "SPLIT"):
        errs.append("Free-tier cost treatment must be COGS, S&M or SPLIT.")
    if p["internal_treatment"] not in ("RD", "COGS"):
        errs.append("Internal traffic treatment must be RD (R&D / G&A) or COGS.")
    if not (0.0 <= p["trial_share"] <= 1.0):
        errs.append("Trial share of free traffic must be between 0% and 100%.")
    if not (0.0 <= p["internal_pct"] <= 5.0):
        errs.append("Internal traffic must be between 0% and 500% of customer prompts.")
    if p["front_mix_free"] is not None and not (0.0 <= p["front_mix_free"] <= 1.0):
        errs.append("Free-tier frontier share must be between 0% and 100%.")
    if p["infra_alloc"] not in ("paid", "prompt"):
        errs.append("Fixed-infrastructure attribution must be 'paid' or 'prompt'.")
    if not (0.0 <= p["eff_mom"] <= 0.5):
        errs.append("Monthly AI cost reduction must be between 0% and 50%.")
    if not (0.0 <= p["eff_cap"] <= 0.95):
        errs.append("Maximum cumulative AI cost reduction must be between 0% and 95%.")
    if not (-0.95 <= p["price_step_pct"] <= 5.0):
        errs.append("One-time vendor price change must be between -95% and +500%.")
    for k in ("eff_start", "price_step_month"):
        if not (1 <= int(p[k]) <= 36):
            errs.append(f"'{k}' must be a forecast month between 1 and 36.")
    for ln in p["other_cogs"]:
        if ln["basis"] not in OTHER_BASIS:
            errs.append(f"Other COGS line '{ln['name']}': basis must be one of {OTHER_BASIS}.")
        if ln["amount"] < 0 or (ln["basis"] == "% of revenue" and ln["amount"] > 100):
            errs.append(f"Other COGS line '{ln['name']}': amount must be >= 0 (and <= 100 when % of revenue).")
        if int(ln["start"]) < 1 or int(ln["end"]) < 0:
            errs.append(f"Other COGS line '{ln['name']}': start must be >= 1 and end >= 0 (0 = through the horizon).")
        if ln["growth"] < -50 or ln["growth"] > 100:
            errs.append(f"Other COGS line '{ln['name']}': monthly growth must be between -50% and +100%.")
    if errs:
        raise ValueError(" ".join(errs))


def heavy_multiplier(p):
    """Heavy-user prompt multiplier. In Pareto mode it is derived: heavy share h generating prompt share s => m = s(1-h)/(h(1-s))."""
    h = p["heavy_share"]
    if h <= 0 or h >= 1:
        return 1.0
    if p["heavy_mode"] == "Pareto prompt share":
        s = min(max(p["pareto_share"], 0.0), 0.99)
        return max(1.0, s * (1 - h) / (h * (1 - s)))
    return max(1.0, p["heavy_mult"])


def usage_curve(cal_months, base, v1, m1, w1, v2, m2, w2):
    """Prompts / active paid user / day by CALENDAR month. Peaks are never below baseline; distance wraps around the year."""
    cal = np.asarray(cal_months, dtype=float)

    def circ(m):
        d = np.abs(cal - m)
        return np.minimum(d, 12.0 - d)

    c1 = (max(v1, base) - base) * np.exp(-(circ(m1) ** 2) / (2 * w1 ** 2))
    c2 = (max(v2, base) - base) * np.exp(-(circ(m2) ** 2) / (2 * w2 ** 2))
    return base + c1 + c2


def other_cogs_for_month(lines, k, revenue):
    """Sum of user-defined extra COGS lines in forecast month k (1-based). % lines scale with that month's recognized revenue;
    $ lines can grow month over month from their start month. A line applies from `start` to `end` (0 = through the horizon)."""
    total = 0.0
    for ln in lines:
        amt = float(ln["amount"])
        if amt <= 0:
            continue
        start, end = int(ln["start"]), int(ln["end"])
        if k < start or (end > 0 and k > end):
            continue
        if ln["basis"] == "% of revenue":
            total += revenue * amt / 100.0
        else:
            total += amt * (1.0 + float(ln["growth"]) / 100.0) ** (k - start)
    return total


def api_cost_for(p, n_prompts, mix, i):
    """AI inference (API) cost of `n_prompts` prompts in forecast month index i (0-based) at frontier share `mix`.
    The single source of per-prompt AI cost: `simulate` and the Deal Desk both call it, so they cannot drift apart."""
    pm = p["price_mult"]
    f_in, f_out, s_in, s_out = p["f_in"] * pm, p["f_out"] * pm, p["s_in"] * pm, p["s_out"] * pm
    in_tok = p["in_tok"] * ((1 + p["in_grow"]) ** i)
    out_tok = p["out_tok"] * ((1 + p["out_grow"]) ** i)
    unit_idx = ai_unit_cost_index(p, i + 1)
    batch_factor = (1 - p["batch_share"]) + p["batch_share"] * p["batch_ratio"]
    raw_in_m = n_prompts * in_tok / 1_000_000
    raw_out_m = n_prompts * out_tok / 1_000_000
    unc_in_m = raw_in_m * (1 - p["cache_rate"])
    cac_in_m = raw_in_m * p["cache_rate"]
    front_in = (unc_in_m * f_in + cac_in_m * f_in * p["cache_ratio"]) * mix
    std_in = (unc_in_m * s_in + cac_in_m * s_in * p["cache_ratio"]) * (1.0 - mix)
    front_out = raw_out_m * f_out * mix
    std_out = raw_out_m * s_out * (1.0 - mix)
    return (front_in + std_in + front_out + std_out) * batch_factor * (1 + p["overhead"]) * unit_idx


def simulate(p):
    validate_params(p)
    n = int(p["horizon"])
    start_period = pd.Period(freq="M", year=int(p["start_year"]), month=int(p["start_month"]))
    dates = [start_period + i for i in range(n)]
    month_names = [d.strftime("%b %Y") for d in dates]
    month_days = [d.days_in_month for d in dates]
    cal_months = [d.month for d in dates]

    curve = usage_curve(cal_months, p["base_prompts"], p["p1_vol"], p["p1_month"], p["p1_width"],
                        p["p2_vol"], p["p2_month"], p["p2_width"])
    h_mult = heavy_multiplier(p)
    act = p["paid_dau_factor"]
    term = int(p["ent_term"])
    seats = float(p["ent_seats"])
    impl_months = int(p["impl_rec_months"])
    impl_sched = np.zeros(n + impl_months + 1)
    pm = p["price_mult"]
    f_in, f_out, s_in, s_out = p["f_in"] * pm, p["f_out"] * pm, p["s_in"] * pm, p["s_out"] * pm

    # Team / SMB cohorts. Starting accounts are a STAGGERED book: evenly spread over remaining terms 1..term,
    # so renewals arrive every month instead of one cliff. 'end' = month index at whose START the term ends.
    cohorts = []
    if p["ent"] > 0:
        for r in range(1, term + 1):
            cohorts.append({"acc": p["ent"] / term, "idx": 1.0, "end": r})

    ai_left = float(p["ai_credits"])
    cloud_left = float(p["cloud_credits"])
    prev_pro = float(p["pro"])
    dau_val = float(p["dau"])
    cum_billed = 0.0
    cum_recognized = 0.0
    rows = []

    for i in range(n):
        # ---------- team / SMB cohort roll-forward ----------
        beg_acc = sum(c["acc"] for c in cohorts)
        beg_equiv = sum(c["acc"] * c["idx"] for c in cohorts)
        due = 0.0
        non_renewed = 0.0
        renewed_ids = set()  # cohorts that reached term end this month: exempt from in-term churn this month
        for c in cohorts:
            if c["end"] == i:
                renewed_ids.add(id(c))
                due += c["acc"]
                lost = c["acc"] * (1.0 - p["ent_renewal_rate"])
                non_renewed += lost
                c["acc"] -= lost
                c["end"] = i + term
        intrm = 0.0
        if p["ent_in_term_churn"] > 0:
            for c in cohorts:
                if id(c) in renewed_ids:
                    continue  # renewal-month exemption: their only exit this month is the renewal decision
                lost = c["acc"] * p["ent_in_term_churn"]
                c["acc"] -= lost
                intrm += lost
        for c in cohorts:
            c["idx"] *= (1.0 + p["ent_expansion"])
        new_ent = float(p["net_new_ent"])
        if new_ent > 0:
            cohorts.append({"acc": new_ent, "idx": 1.0, "end": i + term})
        cohorts = [c for c in cohorts if c["acc"] > 1e-12]
        end_acc = sum(c["acc"] for c in cohorts)
        end_equiv = sum(c["acc"] * c["idx"] for c in cohorts)
        avg_acc = (beg_acc + end_acc) / 2.0
        avg_equiv = (beg_equiv + end_equiv) / 2.0  # expansion-weighted account equivalents

        # ---------- DAU & Pro funnel ----------
        beg_pro = prev_pro
        if i == 0:
            prev_dau = dau_val
            new_users = 0.0
        else:
            prev_dau = dau_val
            dau_val = prev_dau * (1.0 + p["dau_growth"])
            new_users = max(0.0, dau_val - prev_dau)
        paid_active_beg = beg_pro * act + beg_equiv * seats * act
        free_pool_beg = max(0.0, prev_dau - paid_active_beg)
        conv_new = new_users * p["new_conv"]
        conv_base = free_pool_beg * p["base_conv"]
        conv_cap = free_pool_beg + new_users
        conv_total = conv_new + conv_base
        if conv_total > conv_cap and conv_total > 0:
            scale = conv_cap / conv_total
            conv_new *= scale
            conv_base *= scale
            conv_total = conv_cap
        churned_pro = beg_pro * p["churn"]
        ending_pro = beg_pro + conv_total - churned_pro
        prev_pro = ending_pro
        avg_pro = (beg_pro + ending_pro) / 2.0
        avg_dau = (prev_dau + dau_val) / 2.0

        pro_active = avg_pro * act
        ent_active_seats = avg_equiv * seats * act
        free_dau = max(0.0, avg_dau - pro_active - ent_active_seats)

        # ---------- prompts ----------
        typ_rate = max(0.0, float(curve[i])) * p["usage_mult"]
        typ_prompts = pro_active * (1 - p["heavy_share"]) * typ_rate * month_days[i]
        heavy_prompts = pro_active * p["heavy_share"] * typ_rate * h_mult * month_days[i]
        ent_prompts = avg_equiv * p["ent_prompts"] * p["usage_mult"]
        free_prompts = free_dau * typ_rate * p["free_mult"] * month_days[i]
        paid_prompts_m = typ_prompts + heavy_prompts + ent_prompts
        trial_prompts = free_prompts * p["trial_share"]                 # time-limited trials of paid plans
        freemium_prompts = free_prompts - trial_prompts                 # perpetual free tier
        internal_prompts = (paid_prompts_m + free_prompts) * p["internal_pct"]   # staff, evals, QA, synthetic tests
        prompts_total = paid_prompts_m + free_prompts + internal_prompts

        # ---------- API / vector / infra (priced per traffic class so tiers can route differently) ----------
        in_tok = p["in_tok"] * ((1 + p["in_grow"]) ** i)
        out_tok = p["out_tok"] * ((1 + p["out_grow"]) ** i)
        unit_idx = ai_unit_cost_index(p, i + 1)   # AI cost path: efficiency programme x one-time vendor price change (1.0 = none)
        batch_factor = (1 - p["batch_share"]) + p["batch_share"] * p["batch_ratio"]

        def api_for(n_prompts, mix):
            return api_cost_for(p, n_prompts, mix, i)

        free_mix = p["front_mix"] if p["front_mix_free"] is None else p["front_mix_free"]
        api_paid = api_for(paid_prompts_m, p["front_mix"])
        api_free = api_for(free_prompts, free_mix)
        api_internal = api_for(internal_prompts, p["front_mix"])
        api_cost = api_paid + api_free + api_internal
        vec_per_prompt = p["vec_cost"] / 1000.0
        vector_cost = prompts_total * vec_per_prompt

        ending_paid_accounts = ending_pro + end_acc
        if p["step_type"] == "DAU Threshold":
            trigger = dau_val
        elif p["step_type"] == "Paid Subscribers Threshold":
            trigger = ending_paid_accounts
        else:
            trigger = prompts_total
        thr = p["step_thresh"]
        step_mult = max(0, math.ceil(trigger / thr) - 1) if trigger > thr else 0
        infra_cost = p["base_infra"] + step_mult * p["step_cost"]

        # ---------- revenue ----------
        pro_base_rev = avg_pro * p["arpu_pro"]
        pro_exp_rev = pro_base_rev * p["pro_expansion_pct"]
        ent_rev = avg_equiv * p["arpu_ent"]
        subscription_rev = pro_base_rev + pro_exp_rev + ent_rev
        billed = new_ent * p["impl_fee"]
        if billed > 0:
            impl_sched[i:i + impl_months] += billed / impl_months
        impl_rev = float(impl_sched[i])
        cum_billed += billed
        cum_recognized += impl_rev
        recognized_rev = subscription_rev + impl_rev
        ending_mrr = ending_pro * p["arpu_pro"] * (1 + p["pro_expansion_pct"]) + end_equiv * p["arpu_ent"]

        # ---------- COGS ----------
        payment_fees = recognized_rev * p["fee_pct"]
        impl_cost = new_ent * p["impl_cost"]
        direct_cost = api_cost + vector_cost + infra_cost + payment_fees + impl_cost   # every direct cost, whatever its accounting label
        # Three traffic classes: paying customers, free / trial traffic, internal traffic. Variable cost (API + vector) is priced per class;
        # fixed infrastructure stays with paid customers unless the sidebar says to split it by prompt share.
        free_cost = api_free + free_prompts * vec_per_prompt
        internal_cost = api_internal + internal_prompts * vec_per_prompt
        if p["infra_alloc"] == "prompt":
            sh_free = free_prompts / prompts_total if prompts_total > 0 else 1.0   # a month with no traffic: idle capacity lands in the free / idle bucket
            sh_int = internal_prompts / prompts_total if prompts_total > 0 else 0.0
            free_cost += infra_cost * sh_free
            internal_cost += infra_cost * sh_int
        paid_pool = api_cost + vector_cost + infra_cost - free_cost - internal_cost
        trial_cost = free_cost * p["trial_share"]
        freemium_cost = free_cost - trial_cost
        if p["free_treatment"] == "COGS":
            free_in_cogs = free_cost
        elif p["free_treatment"] == "S&M":
            free_in_cogs = 0.0
        else:  # SPLIT: trials are acquisition spend (S&M); the perpetual free tier stays in COGS
            free_in_cogs = freemium_cost
        free_in_sm = free_cost - free_in_cogs
        internal_in_cogs = internal_cost if p["internal_treatment"] == "COGS" else 0.0
        internal_off_cogs = internal_cost - internal_in_cogs                            # booked to R&D / G&A
        paid_cogs = paid_pool + payment_fees + impl_cost                                # costs of serving paying customers
        cogs = paid_cogs + free_in_cogs + internal_in_cogs                              # reported COGS (after the treatment choices)
        gp = recognized_rev - cogs
        gm = gp / recognized_rev * 100 if recognized_rev > 0 else np.nan
        paid_gp = recognized_rev - paid_cogs
        paid_gm = paid_gp / recognized_rev * 100 if recognized_rev > 0 else np.nan
        other_cogs = other_cogs_for_month(p["other_cogs"], i + 1, recognized_rev)
        fl_cogs = cogs + other_cogs
        fl_gp = recognized_rev - fl_cogs
        fl_gm = fl_gp / recognized_rev * 100 if recognized_rev > 0 else np.nan

        # ---------- credits ----------
        ai_open, cloud_open = ai_left, cloud_left
        ai_expired = cloud_expired = 0.0
        if p["ai_expiry"] > 0 and (i + 1) > p["ai_expiry"] and ai_left > 0:
            ai_expired, ai_left = ai_left, 0.0
        if p["cloud_expiry"] > 0 and (i + 1) > p["cloud_expiry"] and cloud_left > 0:
            cloud_expired, cloud_left = cloud_left, 0.0
        ai_applied = min(ai_left, api_cost * p["ai_elig"])
        ai_left -= ai_applied
        cloud_applied = min(cloud_left, (vector_cost + infra_cost) * p["cloud_elig"])
        cloud_left -= cloud_applied
        credits_applied = ai_applied + cloud_applied
        vendor_payable = max(0.0, api_cost + vector_cost + infra_cost - credits_applied)
        immediate_cash = payment_fees + impl_cost
        cash_payable = vendor_payable + immediate_cash

        rows.append({
            "Month": month_names[i],
            "DAU": dau_val,
            "Beginning Pro Subs": beg_pro,
            "New-User Conversions": conv_new,
            "Free-Pool Conversions": conv_base,
            "Churned Pro Subs": churned_pro,
            "Ending Pro Subs": ending_pro,
            "Beginning Team Accounts": beg_acc,
            "New Team Deals": new_ent,
            "Team Accounts Up For Renewal": due,
            "Team Non-Renewals": non_renewed,
            "Team In-Term Churn": intrm,
            "Ending Team Accounts": end_acc,
            "Team Expansion Index": (end_equiv / end_acc) if end_acc > 0 else 1.0,
            "Ending Paid Accounts": ending_paid_accounts,
            "Average Pro Subs": avg_pro,
            "Average Team Accounts": avg_acc,
            "Typical Pro Prompts": typ_prompts,
            "Heavy Pro Prompts": heavy_prompts,
            "Team / SMB Prompts": ent_prompts,
            "Free Prompts": free_prompts,
            "Trial Prompts": trial_prompts,
            "Freemium Prompts": freemium_prompts,
            "Internal Prompts": internal_prompts,
            "Monthly Prompts": prompts_total,
            "Ending MRR Run-Rate ($)": ending_mrr,
            "Subscription Revenue ($)": subscription_rev,
            "Implementation Fees Billed ($)": billed,
            "Implementation Revenue ($)": impl_rev,
            "Recognized Revenue ($)": recognized_rev,
            "API Cost ($)": api_cost,
            "Vector DB Cost ($)": vector_cost,
            "Fixed Infrastructure ($)": infra_cost,
            "Payment Processing ($)": payment_fees,
            "Implementation Cost ($)": impl_cost,
            "Total Direct Cost ($)": direct_cost,
            "Paid-Customer Inference & Infra ($)": paid_pool,
            "Paid-Customer Prompts": paid_prompts_m,
            "Paid-Customer Variable Cost ($)": api_paid + paid_prompts_m * vec_per_prompt,
            "Free-Tier Inference Cost ($)": free_cost,
            "Trial Inference Cost ($)": trial_cost,
            "Freemium Inference Cost ($)": freemium_cost,
            "Internal Traffic Cost ($)": internal_cost,
            "Internal Cost in COGS ($)": internal_in_cogs,
            "Internal Cost in R&D / G&A ($)": internal_off_cogs,
            "Free-Tier Cost in COGS ($)": free_in_cogs,
            "Free-Tier Cost in S&M ($)": free_in_sm,
            "Paid-Customer COGS ($)": paid_cogs,
            "Paid-Customer Gross Margin (%)": paid_gm,
            "Total Modeled COGS ($)": cogs,
            "Modeled Gross Profit ($)": gp,
            "Modeled Gross Margin (%)": gm,
            "Other COGS ($)": other_cogs,
            "Fully Loaded COGS ($)": fl_cogs,
            "Fully Loaded Gross Profit ($)": fl_gp,
            "Fully Loaded Gross Margin (%)": fl_gm,
            "AI Unit-Cost Index (x)": unit_idx,
            "Opening AI Credits ($)": ai_open,
            "AI Credits Expired ($)": ai_expired,
            "AI Credits Applied ($)": ai_applied,
            "Remaining AI Credits ($)": ai_left,
            "Opening Cloud Credits ($)": cloud_open,
            "Cloud Credits Expired ($)": cloud_expired,
            "Cloud Credits Applied ($)": cloud_applied,
            "Remaining Cloud Credits ($)": cloud_left,
            "Total Credits Applied ($)": credits_applied,
            "Vendor Invoices Payable ($)": vendor_payable,
            "Estimated Cash Payable ($)": cash_payable,
        })

    # ---------- invoice lag: vendor invoices lag; fees and implementation costs are paid immediately ----------
    lag = int(p["invoice_lag"])
    vendor = [r["Vendor Invoices Payable ($)"] for r in rows]
    for i, r in enumerate(rows):
        vendor_paid = vendor[i - lag] if i >= lag else 0.0
        r["Estimated Cash Paid This Month ($)"] = vendor_paid + r["Payment Processing ($)"] + r["Implementation Cost ($)"]
        r["Vendor Invoices Outstanding ($)"] = sum(vendor[max(0, i - lag + 1):i + 1]) if lag > 0 else 0.0
        # Accrual revenue less vendor invoices (net of credits, as incurred) and immediate costs. NOT cash profit:
        # it ignores invoice lag timing and revenue collections.
        r["Revenue Less Net Vendor Payables & Immediate Costs ($)"] = r["Recognized Revenue ($)"] - r["Estimated Cash Payable ($)"]
        r["Deferred Implementation Revenue ($)"] = float(np.sum([x["Implementation Fees Billed ($)"] for x in rows[:i + 1]])
                                                         - np.sum([x["Implementation Revenue ($)"] for x in rows[:i + 1]]))
    return pd.DataFrame(rows)


run_model_simulation = st.cache_data(show_spinner=False)(simulate)


def cost_allocation(df):
    """Paid vs free-tier share of API + vector + infra cost, summed over the horizon.
    Variable cost splits by prompt volume each month; fixed infrastructure follows the sidebar attribution setting.
    Months with no traffic at all carry no variable cost, so nothing vanishes."""
    return float(df["Paid-Customer Inference & Infra ($)"].sum()), float(df["Free-Tier Inference Cost ($)"].sum())


def usage_ceiling(df, p, target_pct):
    """Per-account Pro usage ceiling in the ENDING month, using that month's marginal cost per prompt."""
    row = df.iloc[-1]
    prompts = float(row["Paid-Customer Prompts"])   # price the ceiling on paying-customer traffic only (free / internal may use a cheaper model)
    rev = p["arpu_pro"] * (1 + p["pro_expansion_pct"])
    allowed = rev * (1 - target_pct / 100.0 - p["fee_pct"])
    if prompts <= 0:
        return {"var_cost": 0.0, "loaded_cost": 0.0, "var": math.inf, "loaded": math.inf, "no_headroom": allowed <= 0, "rev": rev, "allowed": allowed}
    var_cost = float(row["Paid-Customer Variable Cost ($)"]) / prompts
    loaded_cost = var_cost + float(row["Fixed Infrastructure ($)"]) / prompts
    if allowed <= 0:
        return {"var_cost": var_cost, "loaded_cost": loaded_cost, "var": 0.0, "loaded": 0.0, "no_headroom": True, "rev": rev, "allowed": allowed}
    return {
        "var_cost": var_cost, "loaded_cost": loaded_cost,
        "var": allowed / var_cost if var_cost > 0 else math.inf,
        "loaded": allowed / loaded_cost if loaded_cost > 0 else math.inf,
        "no_headroom": False, "rev": rev, "allowed": allowed,
    }


# ---------- deal desk engine ----------
DEAL_DEFAULTS = {
    "seats": 100, "list_price": 40.0, "discount": 0.20, "term": 12, "billing": "Monthly",
    "impl_fee": 0.0, "impl_cost": 0.0, "pps": 400.0, "heavy_share": 0.10, "usage_growth": 0.02,
    "front_mix": None, "dedicated_infra": 0.0, "support": 0.0, "alloc_shared_infra": True,
    "included_pps": 0.0, "overage_per_1k": 0.0, "cap_pps": 0.0, "start_month": 1,
}


def validate_deal(d):
    """Return an error string, or None when the deal inputs are usable."""
    if d["seats"] <= 0:
        return "Seats must be greater than zero."
    if int(d["term"]) < 1:
        return "Term must be at least 1 month."
    if not (0.0 <= d["discount"] <= 1.0):
        return "Discount must be between 0% and 100%."
    if d["list_price"] < 0 or d["pps"] < 0 or d["impl_fee"] < 0 or d["impl_cost"] < 0 or d["dedicated_infra"] < 0 or d["support"] < 0:
        return "Prices, usage and costs cannot be negative."
    if not (0.0 <= d["heavy_share"] <= 1.0):
        return "Heavy-user share must be between 0% and 100%."
    return None


def deal_pnl(p, d, df, h_mult):
    """Month-by-month P&L and cash for ONE customer deal, priced with the same per-prompt AI cost as the main model.
    Gross margin here = revenue less payment fees, AI inference, vector DB, allocated shared infrastructure, dedicated costs,
    support and implementation delivery. Overage prompts are billable; prompts above a hard cap are blocked (not served, not billed)."""
    T = int(d["term"])
    mix = p["front_mix"] if d["front_mix"] is None else d["front_mix"]
    net_price = d["list_price"] * (1 - d["discount"])
    seat_factor = (1 - d["heavy_share"]) + d["heavy_share"] * h_mult
    impl_rec = max(1, int(p["impl_rec_months"]))
    lag = int(p["invoice_lag"])
    rows, vend_paid_cash = [], []
    start = int(d["start_month"]) - 1
    for t in range(T):
        k = start + t                                     # forecast month index (drives token growth and the AI cost path)
        asked = d["seats"] * d["pps"] * seat_factor * ((1 + d["usage_growth"]) ** t)
        served = min(asked, d["cap_pps"] * d["seats"]) if d["cap_pps"] > 0 else asked
        blocked = asked - served
        over = max(0.0, served - d["included_pps"] * d["seats"]) if d["overage_per_1k"] > 0 else 0.0
        sub_rev = d["seats"] * net_price
        over_rev = over / 1000.0 * d["overage_per_1k"]
        impl_rev = d["impl_fee"] / impl_rec if t < impl_rec else 0.0
        rev = sub_rev + over_rev + impl_rev
        fees = rev * p["fee_pct"]
        ai = api_cost_for(p, served, mix, k)
        vec = served * p["vec_cost"] / 1000.0
        if d["alloc_shared_infra"]:
            kk = min(k, len(df) - 1)
            tot_pr = float(df["Monthly Prompts"].iloc[kk]) + served
            infra = float(df["Fixed Infrastructure ($)"].iloc[kk]) * (served / tot_pr) if tot_pr > 0 else 0.0
        else:
            infra = 0.0
        ded = d["dedicated_infra"]
        sup = d["support"]
        imp = d["impl_cost"] if t == 0 else 0.0
        cogs = fees + ai + vec + infra + ded + sup + imp
        gp = rev - cogs
        vendor = ai + vec + infra
        rows.append({"Deal month": t + 1, "Prompts served": served, "Prompts blocked by cap": blocked, "Subscription revenue ($)": sub_rev,
                     "Overage revenue ($)": over_rev, "Implementation revenue ($)": impl_rev, "Revenue ($)": rev, "Payment fees ($)": fees,
                     "AI inference ($)": ai, "Vector DB ($)": vec, "Allocated shared infra ($)": infra, "Dedicated infra ($)": ded,
                     "Support ($)": sup, "Implementation delivery ($)": imp, "Total COGS ($)": cogs, "Gross profit ($)": gp,
                     "Gross margin (%)": gp / rev * 100 if rev > 0 else np.nan, "_vendor": vendor})
    out = pd.DataFrame(rows)
    # cash: billing timing, fees on billed amounts, vendor invoices paid `lag` months late, other costs immediate
    cash_in = np.zeros(T)
    sub_total = out["Subscription revenue ($)"].to_numpy()
    if d["billing"] == "Annual upfront":
        for t in range(0, T, 12):
            cash_in[t] += sub_total[t:t + 12].sum()
        cash_in += out["Overage revenue ($)"].to_numpy()
    else:
        cash_in += sub_total + out["Overage revenue ($)"].to_numpy()
    cash_in[0] += d["impl_fee"]                           # implementation fee billed up front
    cash_out = (cash_in * p["fee_pct"] + out["Dedicated infra ($)"].to_numpy() + out["Support ($)"].to_numpy() + out["Implementation delivery ($)"].to_numpy())
    vend = out["_vendor"].to_numpy()
    for t in range(T):
        if t + lag < T:
            cash_out[t + lag] += vend[t]
    out["Cash in ($)"] = cash_in
    out["Cash out ($)"] = cash_out
    out["Cumulative cash ($)"] = np.cumsum(cash_in - cash_out)
    return out.drop(columns=["_vendor"])


def deal_summary(out):
    rev = float(out["Revenue ($)"].sum())
    gp = float(out["Gross profit ($)"].sum())
    cum = out["Cumulative cash ($)"].to_numpy()
    ok_from = None                                       # first month from which cumulative cash stays at or above zero
    for t in range(len(cum)):
        if (cum[t:] >= 0).all():
            ok_from = t + 1
            break
    return {"revenue": rev, "cogs": float(out["Total COGS ($)"].sum()), "gp": gp, "margin": gp / rev * 100 if rev > 0 else np.nan,
            "min_cash": float(cum.min()), "cash_ok_from": ok_from, "blocked": float(out["Prompts blocked by cap"].sum())}


def deal_margin(p, d, df, h_mult):
    s = deal_summary(deal_pnl(p, d, df, h_mult))
    return s["margin"] if not pd.isna(s["margin"]) else -1e9


def _solve_max(f, lo, hi, floor, iters=60):
    """Largest x in [lo, hi] with f(x) >= floor, f falling in x. Returns (x, status) with status 'ok', 'none' (fails even at lo) or 'unbounded'."""
    if f(lo) < floor:
        return None, "none"
    if f(hi) >= floor:
        return hi, "unbounded"
    a, b = lo, hi
    for _ in range(iters):
        m = (a + b) / 2
        if f(m) >= floor:
            a = m
        else:
            b = m
    return a, "ok"


def _solve_min(f, lo, hi, floor, iters=60):
    """Smallest x in [lo, hi] with f(x) >= floor, f rising in x. Returns (x, status): 'ok', 'free' (holds at lo) or 'none' (fails at hi)."""
    if f(hi) < floor:
        return None, "none"
    if f(lo) >= floor:
        return lo, "free"
    a, b = lo, hi
    for _ in range(iters):
        m = (a + b) / 2
        if f(m) >= floor:
            b = m
        else:
            a = m
    return b, "ok"


def deal_breakevens(p, d, df, h_mult, floor_pct):
    """Break-evens on whole-term margin: max prompts per seat, min net price per seat, max discount."""
    res = {}
    use_hi = max(d["pps"], 1.0) * 1000
    mu, st_u = _solve_max(lambda x: deal_margin(p, {**d, "pps": x}, df, h_mult), 0.0, use_hi, floor_pct)
    res["max_pps"] = (mu, st_u)
    price_hi = max(d["list_price"], 1.0) * 1000
    lp, st_p = _solve_min(lambda x: deal_margin(p, {**d, "list_price": x, "discount": 0.0}, df, h_mult), 0.0, price_hi, floor_pct)
    res["min_net_price"] = (lp, st_p)
    if lp is not None and d["list_price"] > 0:
        res["max_discount"] = (max(0.0, 1 - lp / d["list_price"]), st_p)
    else:
        res["max_discount"] = (None, st_p if lp is None else "ok")
    return res


def deal_verdict(margin, floor_pct, band_pp=5.0):
    if pd.isna(margin) or margin < floor_pct:
        return "Escalate"
    if margin < floor_pct + band_pp:
        return "Approve with conditions"
    return "Approve"


def credit_status(df, start_balance, opening_col, expired_col, remaining_col):
    if start_balance <= 0:
        return "No starting balance"
    exp = df[df[expired_col] > 0]
    if len(exp):
        return f"Expired {exp['Month'].iloc[0]} (${exp[expired_col].sum():,.0f} unused)"
    gone = df[(df[remaining_col] <= 0.005) & (df[opening_col] > 0.005)]
    if len(gone):
        return f"Exhausted {gone['Month'].iloc[0]}"
    return f"${df[remaining_col].iloc[-1]:,.0f} remains at horizon end"


# ---------- scenarios: each one moves ONE driver family (or is an explicit combined stress) ----------
SCENARIOS = [
    ("1. Base Case", {}),
    ("2. PLG Growth High (1.25x DAU growth)", {"growth": 1.25}),
    ("3. PLG Growth Low (0.75x DAU growth)", {"growth": 0.75}),
    ("4. Team / SMB Deal Velocity High (1.25x)", {"deals": 1.25}),
    ("5. Team / SMB Deal Velocity Low (0.75x)", {"deals": 0.75}),
    ("6. Prompt Intensity High (1.5x, all segments)", {"usage": 1.5}),
    ("7. Prompt Intensity Low (0.7x, all segments)", {"usage": 0.7}),
    ("8. API Price Inflation (+25%)", {"price": 1.25}),
    ("9. API Price Reduction (-20%)", {"price": 0.8}),
    ("10. Conversion Weakness (0.5x conversion rates)", {"conv": 0.5}),
    ("11. Churn Shock (2x Pro churn, renewal -15pp)", {"churn": 2.0, "renewal_delta": -0.15}),
    ("12. Combined Downside Stress", {"growth": 0.75, "deals": 0.75, "usage": 1.4, "price": 1.2, "conv": 0.75, "churn": 1.5, "renewal_delta": -0.10}),
]


def apply_scenario(p, s):
    q = dict(p)
    q["dau_growth"] = p["dau_growth"] * s.get("growth", 1.0)
    q["net_new_ent"] = p["net_new_ent"] * s.get("deals", 1.0)   # no rounding: fractional deals allowed
    q["usage_mult"] = p["usage_mult"] * s.get("usage", 1.0)     # scales typical, heavy, free AND team/SMB prompts
    q["price_mult"] = p["price_mult"] * s.get("price", 1.0)
    q["new_conv"] = min(1.0, p["new_conv"] * s.get("conv", 1.0))
    q["base_conv"] = min(1.0, p["base_conv"] * s.get("conv", 1.0))
    q["churn"] = min(0.99, p["churn"] * s.get("churn", 1.0))
    q["ent_renewal_rate"] = min(1.0, max(0.0, p["ent_renewal_rate"] + s.get("renewal_delta", 0.0)))
    return q


def totals(d):
    rev = float(d["Recognized Revenue ($)"].sum())
    cogs = float(d["Total Modeled COGS ($)"].sum())
    return {
        "rev": rev, "cogs": cogs, "gp": rev - cogs, "gm": ((rev - cogs) / rev * 100) if rev > 0 else np.nan,
        "paid_cogs": float(d["Paid-Customer COGS ($)"].sum()),
        "paid_gm": ((rev - float(d["Paid-Customer COGS ($)"].sum())) / rev * 100) if rev > 0 else np.nan,
        "free_cost": float(d["Free-Tier Inference Cost ($)"].sum()), "free_sm": float(d["Free-Tier Cost in S&M ($)"].sum()),
        "free_cogs": float(d["Free-Tier Cost in COGS ($)"].sum()),
        "trial_cost": float(d["Trial Inference Cost ($)"].sum()), "freemium_cost": float(d["Freemium Inference Cost ($)"].sum()),
        "internal_cost": float(d["Internal Traffic Cost ($)"].sum()), "internal_cogs": float(d["Internal Cost in COGS ($)"].sum()),
        "internal_off": float(d["Internal Cost in R&D / G&A ($)"].sum()),
        "other": float(d["Other COGS ($)"].sum()), "fl_cogs": float(d["Fully Loaded COGS ($)"].sum()),
        "fl_gm": ((rev - float(d["Fully Loaded COGS ($)"].sum())) / rev * 100) if rev > 0 else np.nan,
        "arr": float(d["Ending MRR Run-Rate ($)"].iloc[-1]) * 12,
        "payable": float(d["Estimated Cash Payable ($)"].sum()),
        "paid": float(d["Estimated Cash Paid This Month ($)"].sum()),
        "outstanding": float(d["Vendor Invoices Outstanding ($)"].iloc[-1]),
    }


# ---------- like-for-like pricing archetypes ----------
def compare_pricing(df, p, r):
    n = len(df)
    h = p["heavy_share"]
    pro_acc = df["Average Pro Subs"].to_numpy()
    ent_acc = df["Average Team Accounts"].to_numpy()
    typ_p = df["Typical Pro Prompts"].to_numpy()
    hvy_p = df["Heavy Pro Prompts"].to_numpy()
    ent_p = df["Team / SMB Prompts"].to_numpy()
    paid_p = typ_p + hvy_p + ent_p
    paid_cost, free_cost = cost_allocation(df)
    account_months = float(pro_acc.sum() + ent_acc.sum())

    def per_acct(prompts, accts):
        return np.divide(prompts, accts, out=np.zeros(n), where=accts > 1e-12)

    def overage(prompts, accts, allowance, price):
        use = per_acct(prompts, accts)
        return float((np.maximum(0.0, use - allowance) * accts).sum() * price)

    # segment-level overage: typical, heavy and team measured separately (no pooling)
    def sub_like(pro_fee, pro_allow, ent_fee, ent_allow, ovg):
        base = float(pro_acc.sum() * pro_fee + ent_acc.sum() * ent_fee)
        over = (overage(typ_p, pro_acc * (1 - h), pro_allow, ovg) + overage(hvy_p, pro_acc * h, pro_allow, ovg)
                + overage(ent_p, ent_acc, ent_allow, ovg))
        return base + over

    basis_per_prompt = "Per paid prompt (Pro + Team / SMB); no seat, account or expansion revenue"
    basis_fee = "Fixed fee per average contracted account + overage per prompt; excludes the expansion uplift"
    revenues = [
        ("1. Pay-as-you-go", float(paid_p.sum() * r["payg_rate"]), "Low (usage-driven)", "Low commitment; revenue volatility", basis_per_prompt),
        ("2. Subscription + allowance", sub_like(r["sub_pro_fee"], r["sub_pro_allow"], r["sub_ent_fee"], r["sub_ent_allow"], r["sub_over"]),
         "High (base fee)", "Allowance / overage boundary friction", basis_fee),
        ("3. Hybrid base + overage", sub_like(r["hyb_pro_fee"], r["hyb_pro_allow"], r["hyb_ent_fee"], r["hyb_ent_allow"], r["hyb_over"]),
         "High (base + usage)", "Requires clear meter and overage UX", basis_fee),
        ("4. Prepaid credit burndown",
         float(paid_p.sum() / r["prepaid_ppc"] * r["prepaid_price"] / max(1e-9, 1.0 - r["prepaid_breakage"])) if r["prepaid_ppc"] > 0 else 0.0,
         "Upfront cash; recognition varies", "Breakage, expiry and replenishment risk",
         "Prepaid price per credit pack, grossed up for breakage; no seat, account or expansion revenue"),
        ("0. Current model (subscriptions, ex-implementation)", float(df["Subscription Revenue ($)"].sum()),
         "High (contracted + subscription)", "Reference row: the revenue model driving the rest of the dashboard",
         "ARPU per average Pro sub and per expansion-weighted team account; INCLUDES the Pro overage/expansion uplift and the team expansion index"),
    ]
    out = []
    for label, rev, pred, risk, basis in revenues:
        fees = rev * p["fee_pct"]
        contrib = rev - paid_cost - fees
        after = contrib - free_cost
        out.append({
            "Pricing Archetype": label,
            "Simulated Revenue ($)": rev,
            "Paid-Usage Cost ($)": paid_cost,
            "Payment Fees ($)": fees,
            "Contribution Margin ($)": contrib,
            "Contribution Margin (%)": contrib / rev * 100 if rev > 0 else np.nan,
            "Free-Tier & Idle Cost ($)": free_cost,
            "Gross Profit After Free-Tier Cost ($)": after,
            "Gross Margin After Free-Tier Cost (%)": after / rev * 100 if rev > 0 else np.nan,
            "Revenue / Account-Month ($)": rev / account_months if account_months > 0 else np.nan,
            "Billing Basis": basis,
            "Revenue Predictability": pred,
            "Key Risk": risk,
        })
    res = pd.DataFrame(out).sort_values("Pricing Archetype").reset_index(drop=True)
    return res


# ---------- automated integrity checks ----------
def run_integrity_checks(df, p, base_totals=None, pricing_df=None):
    checks = []

    def add(name, ok, detail="", applicable=True):
        result = ("PASS" if ok else "FAIL") if applicable else "N/A"
        checks.append({"Check": name, "Result": result, "Detail": detail})

    def close(a, b):
        return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=1e-9, atol=1e-6))

    add("Gross profit = revenue - COGS (every month)", close(df["Modeled Gross Profit ($)"], df["Recognized Revenue ($)"] - df["Total Modeled COGS ($)"]))
    comp = (df["API Cost ($)"] + df["Vector DB Cost ($)"] + df["Fixed Infrastructure ($)"] + df["Payment Processing ($)"] + df["Implementation Cost ($)"])
    add("Total direct cost = API + vector + infra + payment fees + implementation cost", close(df["Total Direct Cost ($)"], comp))
    add("Reported COGS = total direct cost - free-tier cost booked to S&M - internal cost booked to R&D / G&A",
        close(df["Total Modeled COGS ($)"], df["Total Direct Cost ($)"] - df["Free-Tier Cost in S&M ($)"] - df["Internal Cost in R&D / G&A ($)"]))
    add("Paid-customer COGS + free-tier cost + internal traffic cost = total direct cost",
        close(df["Paid-Customer COGS ($)"] + df["Free-Tier Inference Cost ($)"] + df["Internal Traffic Cost ($)"], df["Total Direct Cost ($)"]))
    _fb = df["Free-Tier Cost in COGS ($)"] + df["Free-Tier Cost in S&M ($)"]
    _free_ok = close(_fb, df["Free-Tier Inference Cost ($)"])
    if p["free_treatment"] == "COGS":
        _free_ok = _free_ok and bool((df["Free-Tier Cost in S&M ($)"].abs() < 1e-9).all())
    elif p["free_treatment"] == "S&M":
        _free_ok = _free_ok and bool((df["Free-Tier Cost in COGS ($)"].abs() < 1e-9).all())
    else:
        _free_ok = _free_ok and close(df["Free-Tier Cost in COGS ($)"], df["Freemium Inference Cost ($)"])
    add("Free-tier cost is booked once: COGS part + S&M part = free-tier cost (and matches the treatment)", _free_ok, f"treatment = {p['free_treatment']}")
    add("Trial + freemium = free-tier cost and free prompts",
        close(df["Trial Inference Cost ($)"] + df["Freemium Inference Cost ($)"], df["Free-Tier Inference Cost ($)"])
        and close(df["Trial Prompts"] + df["Freemium Prompts"], df["Free Prompts"]))
    add("Internal traffic cost is booked once: COGS part + R&D / G&A part = internal cost",
        close(df["Internal Cost in COGS ($)"] + df["Internal Cost in R&D / G&A ($)"], df["Internal Traffic Cost ($)"]),
        f"internal traffic = {p['internal_pct'] * 100:.0f}% of customer prompts, booked to {'COGS' if p['internal_treatment'] == 'COGS' else 'R&D / G&A'}")
    add("Fully loaded GP = revenue - reported COGS - other COGS",
        close(df["Fully Loaded Gross Profit ($)"], df["Recognized Revenue ($)"] - df["Total Modeled COGS ($)"] - df["Other COGS ($)"]))
    _rev_gp = df["Recognized Revenue ($)"] - df["Paid-Customer COGS ($)"]
    add("Margin ordering every month: paid-customer GP >= reported GP >= fully loaded GP",
        bool((_rev_gp >= df["Modeled Gross Profit ($)"] - 1e-6).all() and (df["Modeled Gross Profit ($)"] >= df["Fully Loaded Gross Profit ($)"] - 1e-6).all()))
    _floor = (1.0 - p["eff_cap"]) * min(1.0, 1.0 + p["price_step_pct"]) - 1e-9
    _ceil = max(1.0, 1.0 + p["price_step_pct"]) + 1e-9
    add("AI unit-cost index stays between the efficiency floor and the vendor price step",
        bool(((df["AI Unit-Cost Index (x)"] >= _floor) & (df["AI Unit-Cost Index (x)"] <= _ceil)).all()),
        ai_path_text(p))
    add("Revenue = subscriptions + implementation recognized", close(df["Recognized Revenue ($)"], df["Subscription Revenue ($)"] + df["Implementation Revenue ($)"]))
    add("Pro roll-forward: beginning + conversions - churn = ending",
        close(df["Beginning Pro Subs"] + df["New-User Conversions"] + df["Free-Pool Conversions"] - df["Churned Pro Subs"], df["Ending Pro Subs"]))
    add("Pro continuity: beginning(t) = ending(t-1)", close(df["Beginning Pro Subs"].iloc[1:], df["Ending Pro Subs"].iloc[:-1]))
    add("Team roll-forward: beginning + new - non-renewals - in-term churn = ending",
        close(df["Beginning Team Accounts"] + df["New Team Deals"] - df["Team Non-Renewals"] - df["Team In-Term Churn"], df["Ending Team Accounts"]))
    add("Team continuity: beginning(t) = ending(t-1)", close(df["Beginning Team Accounts"].iloc[1:], df["Ending Team Accounts"].iloc[:-1]))
    add("Starting team accounts tie to input", abs(float(df["Beginning Team Accounts"].iloc[0]) - p["ent"]) < 1e-6)
    add("Prompt segments sum to total (paid + free + internal)",
        close(df["Typical Pro Prompts"] + df["Heavy Pro Prompts"] + df["Team / SMB Prompts"] + df["Free Prompts"] + df["Internal Prompts"], df["Monthly Prompts"]))
    for tag, start, o, a, e, rem in [("AI", p["ai_credits"], "Opening AI Credits ($)", "AI Credits Applied ($)", "AI Credits Expired ($)", "Remaining AI Credits ($)"),
                                     ("Cloud", p["cloud_credits"], "Opening Cloud Credits ($)", "Cloud Credits Applied ($)", "Cloud Credits Expired ($)", "Remaining Cloud Credits ($)")]:
        add(f"{tag} credits: opening - expired - applied = remaining", close(df[o] - df[e] - df[a], df[rem]))
        add(f"{tag} credits: opening(t+1) = remaining(t); month 1 = input", close(df[o].iloc[1:], df[rem].iloc[:-1]) and abs(float(df[o].iloc[0]) - start) < 1e-6)
    add("Credits applied never exceed eligible cost",
        bool((df["AI Credits Applied ($)"] <= df["API Cost ($)"] * p["ai_elig"] + 1e-6).all()
             and (df["Cloud Credits Applied ($)"] <= (df["Vector DB Cost ($)"] + df["Fixed Infrastructure ($)"]) * p["cloud_elig"] + 1e-6).all()))
    add("Cash payable = total direct cost - credits applied (independent of the COGS / S&M label)",
        close(df["Estimated Cash Payable ($)"], df["Total Direct Cost ($)"] - df["Total Credits Applied ($)"]))
    add("Cash reconciliation: cash paid + outstanding invoices at horizon = cash payable",
        close(df["Estimated Cash Paid This Month ($)"].sum() + df["Vendor Invoices Outstanding ($)"].iloc[-1], df["Estimated Cash Payable ($)"].sum()),
        f"lag = {int(p['invoice_lag'])} month(s)")
    add("Implementation revenue: billed - recognized = deferred balance",
        close(df["Implementation Fees Billed ($)"].cumsum() - df["Implementation Revenue ($)"].cumsum(), df["Deferred Implementation Revenue ($)"]))
    paid_cost, free_cost = cost_allocation(df)
    pool = float((df["API Cost ($)"] + df["Vector DB Cost ($)"] + df["Fixed Infrastructure ($)"]).sum())
    _internal_total = float(df["Internal Traffic Cost ($)"].sum())
    add("Cost allocation: paid + free + internal traffic cost = API + vector + infra", abs(paid_cost + free_cost + _internal_total - pool) < 1e-4 * max(1.0, pool))
    must_be_nonneg = ["DAU", "Beginning Pro Subs", "Ending Pro Subs", "Ending Team Accounts", "Ending Paid Accounts", "Monthly Prompts",
                      "Free Prompts", "Recognized Revenue ($)", "API Cost ($)", "Vector DB Cost ($)", "Fixed Infrastructure ($)",
                      "Total Modeled COGS ($)", "Total Direct Cost ($)", "Paid-Customer COGS ($)", "Free-Tier Inference Cost ($)",
                      "Other COGS ($)", "Fully Loaded COGS ($)", "Internal Traffic Cost ($)", "Trial Inference Cost ($)", "Freemium Inference Cost ($)",
                      "Remaining AI Credits ($)", "Remaining Cloud Credits ($)", "Estimated Cash Payable ($)",
                      "Estimated Cash Paid This Month ($)", "Vendor Invoices Outstanding ($)", "Deferred Implementation Revenue ($)"]
    add("No negative balances, counts, prompts, costs or revenues", bool((df[must_be_nonneg] >= -1e-9).all().all()))
    # Usage ceiling: at the ceiling, per-account margin must equal the target. Always listed; N/A (never silently skipped)
    # when no finite ceiling exists for these inputs (e.g. zero traffic, or fees leave no headroom at any tested target).
    tested, worst = [], 0.0
    for tgt in (40.0, 60.0, 80.0):
        uc = usage_ceiling(df, p, tgt)
        if math.isfinite(uc["var"]) and not uc["no_headroom"] and uc["var_cost"] > 0:
            implied = (uc["rev"] * (1 - p["fee_pct"]) - uc["var"] * uc["var_cost"]) / uc["rev"] * 100
            tested.append(tgt)
            worst = max(worst, abs(implied - tgt))
    add("Usage ceiling solver: per-account margin at the ceiling equals the target",
        worst < 1e-6, f"tested at targets {tested}; worst gap {worst:.2e}pp" if tested else "no finite ceiling for these inputs",
        applicable=bool(tested))
    if base_totals is not None:
        t = totals(df)
        add("Scenario base case reconciles to monthly detail (revenue, COGS, ARR, cash)",
            all(abs(base_totals[k] - t[k]) < 1e-6 * max(1.0, abs(t[k])) for k in ("rev", "cogs", "arr", "payable", "paid")))
    if pricing_df is not None:
        ref = pricing_df[pricing_df["Pricing Archetype"].str.startswith("0.")]
        add("Pricing reference row revenue = model subscription revenue", close(ref["Simulated Revenue ($)"], [df["Subscription Revenue ($)"].sum()]))
        add("Pricing rows share one cost basis (paid-usage cost identical)", float(pricing_df["Paid-Usage Cost ($)"].nunique()) == 1.0)
    return pd.DataFrame(checks)


# ---------- analysis helpers: sensitivity, goal-seek, model health, CAC / LTV ----------
METRIC_OPTIONS = {
    "Reported gross margin (forecast period)": "gm",
    "Fully loaded gross margin (forecast period)": "fl_gm",
    "Paid-customer gross margin (forecast period)": "paid_gm",
    "Reported gross margin (ending month)": "end_gm",
}


def metric_of(df, key):
    if key == "end_gm":
        return float(df["Modeled Gross Margin (%)"].iloc[-1])
    return float(totals(df)[key])


# (parameter, label, kind): 'frac' inputs are clipped to 0-100%, 'ratio' inputs only to >= 0
SENS_DRIVERS = [
    ("usage_mult", "Prompt intensity (all segments)", "ratio"),
    ("free_mult", "Free-tier usage multiplier", "frac"),
    ("front_mix", "Frontier-model traffic share", "frac"),
    ("cache_rate", "Prompt cache hit rate", "frac"),
    ("price_mult", "Token prices (all vendors)", "ratio"),
    ("in_grow", "Input-token growth per month", "frac"),
    ("base_infra", "Base fixed infrastructure ($)", "ratio"),
    ("arpu_pro", "Pro price (monthly ARPU)", "ratio"),
    ("new_conv", "New-user conversion to Pro", "frac"),
    ("churn", "Pro monthly churn", "frac"),
    ("dau_growth", "DAU monthly growth", "ratio"),
    ("heavy_mult", "Heavy-user prompt multiplier", "ratio"),
]


def _fmt_driver(v, kind):
    return f"{v * 100:.1f}%" if kind == "frac" else f"{v:,.2f}"


def tornado_table(p, pct, metric_key):
    """One-at-a-time sensitivity: move each driver down and up by `pct` (relative) and record the change in the chosen margin metric."""
    base = metric_of(simulate(p), metric_key)
    rows = []
    for key, label, kind in SENS_DRIVERS:
        if key == "heavy_mult" and p["heavy_mode"] != "Multiplier":
            continue
        v0 = float(p[key])
        if v0 == 0:
            continue
        lo, hi = v0 * (1 - pct), v0 * (1 + pct)
        if kind == "frac":
            lo, hi = min(max(lo, 0.0), 1.0), min(max(hi, 0.0), 1.0)
        vals = []
        for v in (lo, hi):
            try:
                vals.append(metric_of(simulate({**p, key: v}), metric_key))
            except ValueError:
                vals.append(np.nan)
        d_lo, d_hi = vals[0] - base, vals[1] - base
        rows.append({"Driver": label, "Base input": _fmt_driver(v0, kind), "Low input": _fmt_driver(lo, kind), "High input": _fmt_driver(hi, kind),
                     "Δ Margin at Low Input (pp)": d_lo, "Δ Margin at High Input (pp)": d_hi,
                     "Δ Swing (pp)": float(np.nanmax(np.abs([d_lo, d_hi]))) if not (np.isnan(d_lo) and np.isnan(d_hi)) else np.nan})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("Δ Swing (pp)", ascending=False, na_position="last").reset_index(drop=True)
    return base, out


# (parameter, label, search low, search high, display kind)
GOAL_LEVERS = [
    ("free_mult", "Free-tier usage multiplier", 0.0, 1.0, "pct"),
    ("front_mix", "Frontier-model traffic share", 0.0, 1.0, "pct"),
    ("cache_rate", "Prompt cache hit rate", 0.0, 0.9, "pct"),
    ("batch_share", "Traffic on batch pricing", 0.0, 0.8, "pct"),
    ("eff_mom", "MoM AI cost reduction (uses the start month and cap set in the sidebar)", 0.0, 0.15, "pct"),
    ("arpu_pro", "Pro price (monthly ARPU, $)", 5.0, 500.0, "usd"),
    ("price_mult", "Token price level (x the entered prices)", 0.1, 3.0, "x"),
]


def goal_seek(p, key, lo, hi, target, metric_key, iters=30):
    """Bisection on one lever to hit a target margin. Assumes the margin moves one way as the lever moves (true for these levers)."""
    def f(x):
        try:
            return metric_of(simulate({**p, key: x}), metric_key)
        except ValueError:
            return np.nan
    f_lo, f_hi, cur = f(lo), f(hi), metric_of(simulate(p), metric_key)
    out = {"current_value": float(p[key]), "current_metric": cur, "f_lo": f_lo, "f_hi": f_hi, "lo": lo, "hi": hi}
    if np.isnan(f_lo) or np.isnan(f_hi):
        return {**out, "status": "undefined"}
    if target < min(f_lo, f_hi) - 1e-9 or target > max(f_lo, f_hi) + 1e-9:
        return {**out, "status": "unreachable"}
    increasing = f_hi >= f_lo
    a, b = lo, hi
    for _ in range(iters):
        m = (a + b) / 2.0
        if (f(m) < target) == increasing:
            a = m
        else:
            b = m
    x = (a + b) / 2.0
    return {**out, "status": "ok", "value": x, "achieved": f(x)}


def lever_curve(p, key, lo, hi, metric_key, n=25):
    """Margin at n evenly spaced values of one lever, for the goal-seek chart."""
    xs = np.linspace(lo, hi, n)
    ys = []
    for x in xs:
        try:
            ys.append(metric_of(simulate({**p, key: float(x)}), metric_key))
        except ValueError:
            ys.append(np.nan)
    return xs, np.asarray(ys, dtype=float)


def usage_curve_for(p):
    """Month labels and the typical-paid-user prompts/day curve exactly as the engine builds it (before usage multipliers)."""
    n = int(p["horizon"])
    sp = pd.Period(freq="M", year=int(p["start_year"]), month=int(p["start_month"]))
    dates = [sp + i for i in range(n)]
    curve = usage_curve([d.month for d in dates], p["base_prompts"], p["p1_vol"], p["p1_month"], p["p1_width"], p["p2_vol"], p["p2_month"], p["p2_width"])
    return [d.strftime("%b %Y") for d in dates], np.asarray(curve, dtype=float)


def health_flags(df, p, t, h_mult):
    """Rule-based sanity checks. Level 'check' = likely an input problem worth reviewing; 'info' = worth knowing, not necessarily wrong."""
    flags = []
    _, curve = usage_curve_for(p)
    base = max(float(p["base_prompts"]), 1e-9)
    if curve.max() / base > 3.0:
        flags.append(("check", "Seasonal peak far above baseline",
                      f"Peak usage reaches {curve.max() / base:.1f}x the baseline ({curve.max():.0f} vs {p['base_prompts']:.0f} prompts / active user / day). "
                      "Peaks are absolute numbers, not multipliers, and they apply to free users too; the presets use about 1.7x-2.1x."))
    cust = float(df["Typical Pro Prompts"].sum() + df["Heavy Pro Prompts"].sum() + df["Team / SMB Prompts"].sum() + df["Free Prompts"].sum())
    free_sh = float(df["Free Prompts"].sum()) / cust if cust > 0 else 0.0
    if free_sh > 0.85:
        flags.append(("info", "Free users dominate traffic",
                      f"Free users generate {free_sh * 100:.0f}% of customer prompts, so margin is very sensitive to the free-tier multiplier, routing and caps (see the Levers tab)."))
    end_dau = float(df["DAU"].iloc[-1])
    if end_dau > 0 and float(df["Ending Pro Subs"].iloc[-1]) / end_dau > 0.25:
        flags.append(("check", "Very high Pro share of DAU",
                      f"Pro subscribers equal {float(df['Ending Pro Subs'].iloc[-1]) / end_dau * 100:.0f}% of ending DAU. Check that DAU and subscriber counts use one definition."))
    tok_idx = (1 + p["in_grow"]) ** max(0, len(df) - 1)
    if tok_idx > 3.0:
        flags.append(("info", "Input tokens compound strongly", f"Input tokens per prompt reach x{tok_idx:.1f} by month {len(df)}. Prices are flat unless you set an AI cost path."))
    end_idx = float(df["AI Unit-Cost Index (x)"].iloc[-1])
    if end_idx < 0.4:
        flags.append(("info", "Aggressive AI cost path", f"The AI cost path takes inference cost per prompt to x{end_idx:.2f} by month {len(df)} (a {(1 - end_idx) * 100:.0f}% reduction). "
                                                       "Efficiency gains must be delivered; consider a lower cap or a later start."))
    if h_mult > 10:
        flags.append(("check", "Very high heavy-user multiplier", f"Heavy Pro users prompt {h_mult:.0f}x a typical user. Confirm against per-user usage logs."))
    if p["free_mult"] > 0.8:
        flags.append(("info", "Free users nearly as active as paid", f"Free users prompt at {p['free_mult'] * 100:.0f}% of a typical paid user's rate."))
    tot_prompts = float(df["Monthly Prompts"].sum())
    if tot_prompts > 0 and float(df["Internal Prompts"].sum()) / tot_prompts > 0.15:
        flags.append(("info", "Large internal traffic share", f"Internal / non-customer traffic is {float(df['Internal Prompts'].sum()) / tot_prompts * 100:.0f}% of all prompts."))
    last = df.iloc[-1]
    if not pd.isna(last["Modeled Gross Margin (%)"]) and last["Modeled Gross Margin (%)"] < 0:
        flags.append(("info", "Negative margin at the horizon", f"Reported gross margin is {last['Modeled Gross Margin (%)']:.0f}% in the ending month: unit economics do not work at the end of the forecast under these inputs."))
    if float(last["Total Modeled COGS ($)"]) > 0 and float(last["Fixed Infrastructure ($)"]) / float(last["Total Direct Cost ($)"]) > 0.6:
        flags.append(("info", "Fixed infrastructure dominates cost", "Over 60% of ending-month direct cost is fixed infrastructure, so margin improves mainly with scale."))
    if not p["other_cogs"]:
        flags.append(("info", "No other COGS lines", "Margins cover AI-direct costs only. Add support, observability and similar lines in the 'Fully loaded margin' panel."))
    return flags


def cac_ltv(df, p, t, sm_monthly, count_free):
    """Simplified unit economics. LTV = monthly ARPU x margin x expected lifetime (capped at 60 months, undiscounted, no expansion beyond the Pro uplift).
    Margin = paid-customer margin after other COGS lines, so free-tier cost is counted once (as acquisition cost), never twice."""
    n_pro = float(df["New-User Conversions"].sum() + df["Free-Pool Conversions"].sum())
    n_team = float(df["New Team Deals"].sum())
    n_new = n_pro + n_team
    H = len(df)
    rev = t["rev"]
    margin = (rev - t["paid_cogs"] - t["other"]) / rev if rev > 0 else np.nan
    arpu_pro = p["arpu_pro"] * (1 + p["pro_expansion_pct"])
    life_pro = min(60.0, 1.0 / p["churn"]) if p["churn"] > 0 else 60.0
    life_team = min(60.0, p["ent_term"] / (1.0 - p["ent_renewal_rate"])) if p["ent_renewal_rate"] < 1.0 else 60.0
    ltv_pro = arpu_pro * margin * life_pro
    ltv_team = p["arpu_ent"] * margin * life_team
    free_cost = t["free_cost"] if count_free else 0.0
    spend = sm_monthly * H
    cac = (spend + free_cost) / n_new if n_new > 0 else np.nan
    cac_ex_free = spend / n_new if n_new > 0 else np.nan
    if n_new > 0:
        ltv = (n_pro * ltv_pro + n_team * ltv_team) / n_new
        gp_month = (n_pro * arpu_pro + n_team * p["arpu_ent"]) * margin / n_new
    else:
        ltv, gp_month = np.nan, np.nan
    return {"n_pro": n_pro, "n_team": n_team, "n_new": n_new, "spend": spend, "free_cost": free_cost, "margin": margin,
            "life_pro": life_pro, "life_team": life_team, "ltv_pro": ltv_pro, "ltv_team": ltv_team, "ltv": ltv, "cac": cac, "cac_ex_free": cac_ex_free,
            "ltv_cac": ltv / cac if (n_new > 0 and cac and cac > 0) else np.nan,
            "payback": cac / gp_month if (n_new > 0 and gp_month and gp_month > 0 and not pd.isna(cac)) else np.nan,
            "gp_month": gp_month}


def scenario_snapshot(df, p, t):
    """The comparable numbers saved for a named scenario."""
    last = df.iloc[-1]
    return {
        "Starting profile / horizon": f"{p.get('_preset', '?')} / {len(df)} mo",
        "Free-tier cost booked to": {"COGS": "COGS", "S&M": "Sales & marketing", "SPLIT": "Trials S&M / freemium COGS"}[p["free_treatment"]],
        "AI cost path": ai_path_text(p, len(df)),
        "Ending ARR ($)": t["arr"], "Forecast revenue ($)": t["rev"],
        "Paid-customer GM (%)": t["paid_gm"], "Reported GM (%)": t["gm"], "Fully loaded GM (%)": t["fl_gm"],
        "Ending-month reported GM (%)": float(last["Modeled Gross Margin (%)"]),
        "Free-tier cost ($)": t["free_cost"], "Other COGS ($)": t["other"], "Reported COGS ($)": t["cogs"],
    }


# ==========================================
# 3. RUN, GUARDS, KPIs & BRIEFING
# ==========================================
_OTHER_COLS = ["Line item", "Basis", "Amount", "From month", "To month (0 = end)", "MoM growth (%)"]
_other_default = pd.DataFrame([
    ["Customer support & success", "% of revenue", 0.0, 1, 0, 0.0],
    ["Observability, security & compliance", "$ / month", 0.0, 1, 0, 0.0],
], columns=_OTHER_COLS)


def parse_other_cogs(frame):
    """Turn the editable table into engine lines. Blank, zero or non-numeric rows are ignored; bad basis falls back to $ / month."""
    lines = []
    for _, r in frame.iterrows():
        amt = pd.to_numeric(r.get("Amount"), errors="coerce")
        if pd.isna(amt) or amt <= 0:
            continue
        name = r.get("Line item")
        name = "Other COGS" if (name is None or pd.isna(name) or not str(name).strip()) else str(name).strip()
        basis = r.get("Basis") if r.get("Basis") in OTHER_BASIS else OTHER_BASIS[0]
        start = pd.to_numeric(r.get("From month"), errors="coerce")
        end = pd.to_numeric(r.get("To month (0 = end)"), errors="coerce")
        growth = pd.to_numeric(r.get("MoM growth (%)"), errors="coerce")
        lines.append({"name": name, "basis": basis, "amount": float(amt),
                      "start": 1 if pd.isna(start) else int(max(1, start)),
                      "end": 0 if pd.isna(end) else int(max(0, end)),
                      "growth": 0.0 if pd.isna(growth) else float(growth)})
    return lines


with st.expander("➕ Fully loaded margin: add other COGS lines on top of AI inference", expanded=False):
    st.caption("The model counts AI inference, vector DB, hosting, payment fees and implementation delivery. Add the costs it does not know about "
               "(support, success, observability, security, compliance, moderation, third-party APIs...) to see a fully loaded gross margin. "
               "Add a row for each line; to model a step-up, add a second row with a later start month. Nothing here changes revenue, cash timing or credits.")
    _other_edit = _stretch(
        st.data_editor, _other_default, hide_index=True, num_rows="dynamic", key="other_cogs_editor",
        column_config={
            "Line item": st.column_config.TextColumn("Line item"),
            "Basis": st.column_config.SelectboxColumn("Basis", options=OTHER_BASIS, required=True,
                                                      help="'$ / month' is a fixed monthly cost; '% of revenue' scales with that month's recognized revenue."),
            "Amount": st.column_config.NumberColumn("Amount ($ or %)", min_value=0.0, step=100.0, format="%.2f",
                                                    help="Dollars per month, or a percentage of revenue (e.g. 5 = 5%) depending on Basis."),
            "From month": st.column_config.NumberColumn("From month", min_value=1, max_value=36, step=1, format="%d",
                                                        help="Forecast month the line starts (1 = first month)."),
            "To month (0 = end)": st.column_config.NumberColumn("To month (0 = end)", min_value=0, max_value=36, step=1, format="%d",
                                                                help="Last forecast month the line applies. 0 = through the end of the horizon."),
            "MoM growth (%)": st.column_config.NumberColumn("MoM growth (%)", min_value=-50.0, max_value=100.0, step=0.5, format="%.1f",
                                                            help="Monthly growth of a $ / month line from its start month. Ignored for % of revenue lines."),
        },
    )
P["other_cogs"] = parse_other_cogs(_other_edit)

_act = P["paid_dau_factor"]
_start_active_paid = starting_pro_subs * _act + starting_ent_accounts * P["ent_seats"] * _act
if _start_active_paid > starting_dau:
    st.warning(f"Active paid users implied by your inputs (about {_start_active_paid:,.0f}: Pro subscribers plus team seats, "
               f"each x {_act * 100:.0f}% daily activity) exceed month-1 DAU ({starting_dau:,.0f}). "
               "Confirm that DAU, Pro subscribers and team accounts share one definition; the free pool is floored at zero.")
if starting_dau == 0 and starting_pro_subs + starting_ent_accounts > 0:
    st.warning("You have paid accounts but zero DAU. Confirm that DAU and paid-account counts use compatible definitions.")
if advanced_mode and (cfg["p1_vol"] < baseline_prompts or cfg["p2_vol"] < baseline_prompts):
    st.sidebar.warning(f"A peak volume is below baseline ({baseline_prompts}); peaks are floored at baseline.")

try:
    df = run_model_simulation(P).copy()
except ValueError as e:
    st.error(f"⚠️ Model Configuration Error: {e}")
    st.stop()
except Exception as e:  # unexpected calculation error: show it plainly instead of a raw traceback
    st.error(f"⚠️ Unexpected calculation error: {type(e).__name__}: {e}")
    st.stop()

df["Budget ($)"] = budget_monthly if budget_monthly > 0 else np.nan
df["Budget Variance ($)"] = (df["Total Modeled COGS ($)"] - budget_monthly) if budget_monthly > 0 else np.nan
_prompts_safe = df["Monthly Prompts"].replace(0, np.nan)
df["API Cost / 1,000 Prompts ($)"] = df["API Cost ($)"] / _prompts_safe * 1000
df["Blended Direct Cost / 1,000 Prompts ($)"] = df["Total Direct Cost ($)"] / _prompts_safe * 1000
_paid_accts_safe = df["Ending Paid Accounts"].replace(0, np.nan)
df["Paid-Customer COGS / Paid Account ($)"] = df["Paid-Customer COGS ($)"] / _paid_accts_safe
df["Incl. Free-Tier Cost / Paid Account ($)"] = df["Total Direct Cost ($)"] / _paid_accts_safe

m_last = df.iloc[-1]
tot = totals(df)
last_arr = tot["arr"]
period_gm = tot["gm"]
last_margin = float(m_last["Modeled Gross Margin (%)"])
H = len(df)
h_mult_used = heavy_multiplier(P)
ceil = usage_ceiling(df, P, margin_target)
paid_cost_alloc, free_cost_alloc = cost_allocation(df)
free_share = float(df["Free Prompts"].sum() / df["Monthly Prompts"].sum()) if df["Monthly Prompts"].sum() > 0 else 0.0


def fmt_prompts(v):
    return "No cap (zero marginal cost)" if not math.isfinite(v) else f"{v:,.0f} prompts/mo"


c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Ending ARR Run-Rate (M{H})", f"${last_arr:,.0f} ARR", f"Forecast-period revenue: ${tot['rev']:,.0f}")
c2.metric("Forecast-Period Gross Margin", "n/a (no revenue)" if pd.isna(period_gm) else f"{period_gm:.1f}%",
          f"Target: {margin_target}%" if pd.isna(period_gm) else f"{period_gm - margin_target:+.1f}pp vs {margin_target}% target")
c3.metric(f"Est. Pro Usage Ceiling @ {margin_target}% GM (M{H})",
          "No headroom" if ceil["no_headroom"] else fmt_prompts(ceil["var"]),
          "Estimate, not a safe limit" if ceil["no_headroom"] else f"Incl. fixed infra: {fmt_prompts(ceil['loaded'])} (estimate)",
          help="Estimated from the ending month's company-wide cost per prompt. Capacity steps, customer mix and real routing can move marginal cost; do not treat it as a guaranteed safe usage limit.")
c4.metric("Heavy-User Cohort", f"{P['heavy_share'] * 100:.0f}% of Pro users", f"{h_mult_used:.1f}x typical prompts")


def fmt_pct(v):
    return "n/a" if pd.isna(v) else f"{v:.1f}%"


_treat_name = {"COGS": "COGS", "S&M": "S&M", "SPLIT": "COGS / S&M split"}[P["free_treatment"]]
d1, d2, d3, d4 = st.columns(4)
d1.metric("Paid-Customer Gross Margin", fmt_pct(tot["paid_gm"]), "Before free-tier cost", delta_color="off",
          help="Revenue less the cost of serving paying customers only (their inference, vector, fixed infra, payment fees, implementation). Free-tier cost is never loaded onto paying customers.")
d2.metric(f"Free-Tier Inference Cost (in {_treat_name})", f"${tot['free_cost']:,.0f}",
          "n/a (no revenue)" if tot["rev"] <= 0 else f"{tot['free_cost'] / tot['rev'] * 100:.0f}% of revenue", delta_color="off",
          help="Inference cost of free / trial traffic over the horizon. The sidebar switch books it to COGS or to sales & marketing.")
d3.metric("Fully Loaded Gross Margin", fmt_pct(tot["fl_gm"]),
          f"After ${tot['other']:,.0f} other COGS" if tot["other"] > 0 else "No other COGS lines added", delta_color="off",
          help="Reported gross margin after the extra COGS lines you add in the 'Fully loaded margin' panel above the KPIs. With no lines it equals the reported margin.")
d4.metric(f"AI Unit-Cost Index (M{H})", f"x{float(m_last['AI Unit-Cost Index (x)']):.2f}",
          "Flat prices" if ai_path_is_flat(P) else "AI cost path active", delta_color="off",
          help="Multiplier on AI inference cost from the AI cost path (efficiency programme and vendor price change). 1.00 = prices as entered.")

if GUIDED:
    def _status(gm):
        if pd.isna(gm):
            return None, ""
        if gm >= margin_target:
            return "On target", "good"
        return ("Close to target", "warn") if gm >= margin_target - 10 else ("Below target", "bad")

    _vals = [("Paid-customer margin", tot["paid_gm"], "Revenue less the cost of serving paying customers only.", df["Paid-Customer Gross Margin (%)"]),
             ("Reported margin", tot["gm"], f"After booking free-tier cost to {_treat_name}.", df["Modeled Gross Margin (%)"]),
             ("Fully loaded margin", tot["fl_gm"], "After the extra monthly costs you add." if tot["other"] > 0 else "Same as reported until you add other COGS lines.", df["Fully Loaded Gross Margin (%)"])]
    with slot_cards:
        _cols = st.columns(3)
        for _c, (_lab, _v, _note, _ser) in zip(_cols, _vals):
            _pl, _pk = _status(_v)
            _dup = (_lab == "Fully loaded margin" and tot["other"] <= 0)
            _c.markdown(kpi_card(_lab, fmt_pct(_v), _note, _pl, _pk, series=None if _dup else _ser,
                                 trend_note="Add other COGS lines in the sidebar to see it move away from reported." if _dup else None), unsafe_allow_html=True)
        _known = [v for _, v, _, _ in _vals if not pd.isna(v)]
        if not pd.isna(tot["paid_gm"]) and not pd.isna(tot["gm"]):
            _paid_reported_gap = tot["paid_gm"] - tot["gm"]
            _direction = "lower" if _paid_reported_gap >= 0 else "higher"
            _context_line = (f"Reported margin is {abs(_paid_reported_gap):.1f} percentage points {_direction} than paid-customer margin. "
                             f"That difference reflects how free-tier costs are classified ({_treat_name}); it does not make those costs disappear.")
            st.markdown(f'<div class="takeaway"><b>What this means:</b> {_context_line}</div>', unsafe_allow_html=True)
        if _known:
            _spread = max(_known) - min(_known)
            st.markdown(f'<div class="takeaway"><b>Takeaway:</b> one business, three answers, {_spread:.1f} points apart. '
                        f'Free-tier inference over the forecast is ${tot["free_cost"]:,.0f}. Moving it between COGS and sales &amp; marketing changes the label, '
                        f'not the bill. These are management measures, not standardized accounting figures. Target: {margin_target}%.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="takeaway"><b>Takeaway:</b> no revenue is recognized in this forecast window yet, so margins are undefined. '
                        'Costs are still being incurred. Try the Early traction profile or a longer horizon.</div>', unsafe_allow_html=True)

    _rev_pa = float(m_last["Recognized Revenue ($)"]) / float(m_last["Ending Paid Accounts"]) if float(m_last["Ending Paid Accounts"]) > 0 else np.nan
    _cogs_pa = float(m_last["Paid-Customer COGS / Paid Account ($)"])
    _k1, _k2 = float(df["API Cost / 1,000 Prompts ($)"].dropna().iloc[0]) if df["API Cost / 1,000 Prompts ($)"].notna().any() else np.nan, float(m_last["API Cost / 1,000 Prompts ($)"])
    with slot_ch1_out:
        _a, _b, _c3 = st.columns(3)
        _a.markdown(kpi_card("AI cost per 1,000 prompts", "n/a" if pd.isna(_k2) else f"${_k2:,.2f}",
                             "No prompts yet" if pd.isna(_k1) else f"Month 1: ${_k1:,.2f}, month {H}: ${_k2:,.2f}"), unsafe_allow_html=True)
        _b.markdown(kpi_card("Cost to serve a paid account", "n/a" if pd.isna(_cogs_pa) else f"${_cogs_pa:,.2f}", f"Per month, month {H}. Paying customers only."), unsafe_allow_html=True)
        _c3.markdown(kpi_card("Revenue per paid account", "n/a" if pd.isna(_rev_pa) else f"${_rev_pa:,.2f}", f"Per month, month {H}."), unsafe_allow_html=True)
        st.markdown(f'<div class="takeaway">Each prompt is about {P["in_tok"]:,} input and {P["out_tok"]:,} output tokens, growing {P["in_grow"] * 100:.0f}% a month on the input side. '
                    f'{P["front_mix"] * 100:.0f}% of traffic goes to the premium model and {P["cache_rate"] * 100:.0f}% of input hits the cache. '
                    f'<b>Token prices here are illustrative placeholders, not live vendor prices:</b> replace them with your rate card.</div>', unsafe_allow_html=True)
        _fc = go.Figure()
        _fc.add_trace(go.Scatter(x=df["Month"], y=df["API Cost / 1,000 Prompts ($)"], name="AI cost / 1,000 prompts ($)", mode="lines+markers",
                                 line=dict(color=tone("#0b2cbc", "#7f97ff"), width=2.5)))
        _fc.add_trace(go.Scatter(x=df["Month"], y=df["Paid-Customer COGS / Paid Account ($)"], name="Cost to serve / paid account ($)", mode="lines",
                                 line=dict(color=tone("#9a5b00", "#f3b25c"), width=2, dash="dot"), yaxis="y2"))
        _fc.update_layout(**CHART_KW, height=260, margin=dict(l=30, r=40, t=10, b=20), legend=dict(orientation="h", y=1.18),
                          yaxis=dict(title="$ per 1,000 prompts"), yaxis2=dict(title="$ per paid account", overlaying="y", side="right"))
        show_chart(_fc)
        _share = free_cost_alloc / (paid_cost_alloc + free_cost_alloc) * 100 if (paid_cost_alloc + free_cost_alloc) > 0 else np.nan
        _g1, _g2 = st.columns(2)
        _g1.markdown(ring_card("Free-tier share of the AI bill", _share, "Share of paid + free inference and infrastructure cost that goes to free users."), unsafe_allow_html=True)
        _g2.markdown(drivers_card("Top cost drivers over the forecast", [
            ("Paying customers (inference + infra)", df["Paid-Customer Inference & Infra ($)"].sum()), ("Free tier", df["Free-Tier Inference Cost ($)"].sum()),
            ("Internal traffic", df["Internal Traffic Cost ($)"].sum()), ("Payment fees", df["Payment Processing ($)"].sum()),
            ("Implementation delivery", df["Implementation Cost ($)"].sum())]), unsafe_allow_html=True)
        st.markdown('<div class="bridge">But most of your users do not pay you. Who is using your AI budget? That is the next chapter, in the Value &amp; Pricing and Revenue &amp; Margin tabs.</div>',
                    unsafe_allow_html=True)

_flags = health_flags(df, P, tot, h_mult_used)
_checks = [f for f in _flags if f[0] == "check"]
if _checks:
    st.warning(f"🩺 {len(_checks)} input check(s) worth reviewing: " + "; ".join(f[1] for f in _checks) + ". Details are in the Audit Register tab (Model Health Flags).")

bullets = []
if pd.isna(period_gm):
    bullets.append("<b>Gross Margin Status:</b> No recognized revenue in the forecast horizon, so margin is undefined. Direct COGS is incurred with no revenue offset.")
elif period_gm < margin_target:
    bullets.append(f"<b>Margin Alert:</b> Forecast-period gross margin <b>{period_gm:.1f}%</b> is below the <b>{margin_target}%</b> target (ending month: "
                   f"{'n/a' if pd.isna(last_margin) else f'{last_margin:.1f}%'}). <b>Action:</b> test a usage cap, overage price or lower-cost routing.")
else:
    bullets.append(f"<b>Margin Compliant:</b> Forecast-period gross margin <b>{period_gm:.1f}%</b> meets the <b>{margin_target}%</b> target "
                   f"(ending month: {'n/a' if pd.isna(last_margin) else f'{last_margin:.1f}%' }).")
breach = df.loc[df["Modeled Gross Margin (%)"] < margin_target, "Month"]
if len(breach):
    bullets.append(f"<b>First Monthly Breach:</b> Monthly margin first falls below target in <b>{breach.iloc[0]}</b> ({len(breach)} of {H} months below target).")
if budget_monthly > 0:
    over = df.loc[df["Budget Variance ($)"] > 0, "Month"]
    if len(over):
        bullets.append(f"<b>Budget Overrun:</b> Monthly COGS exceeds the <b>${budget_monthly:,.0f}</b> cap from <b>{over.iloc[0]}</b> (cumulative variance +${df['Budget Variance ($)'].clip(lower=0).sum():,.0f}).")
    else:
        bullets.append(f"<b>Budget:</b> Monthly COGS stays within the ${budget_monthly:,.0f} cap in all {H} months.")
ai_stat = credit_status(df, P["ai_credits"], "Opening AI Credits ($)", "AI Credits Expired ($)", "Remaining AI Credits ($)")
cloud_stat = credit_status(df, P["cloud_credits"], "Opening Cloud Credits ($)", "Cloud Credits Expired ($)", "Remaining Cloud Credits ($)")
bullets.append(f"<b>Credit Runway:</b> AI credits: {ai_stat}. Cloud credits: {cloud_stat}.")
due_total = float(df["Team Accounts Up For Renewal"].sum())
if due_total > 0:
    bullets.append(f"<b>Team / SMB Renewals:</b> {due_total:,.1f} account-renewals fall due in the horizon; at {P['ent_renewal_rate'] * 100:.0f}% renewal about "
                   f"{df['Team Non-Renewals'].sum():,.1f} accounts are lost.")
if df["Monthly Prompts"].sum() > 0:
    _free_pct_rev = (f"{free_cost_alloc / tot['rev'] * 100:.0f}% of revenue" if tot["rev"] > 0 else "no revenue to compare")
    if P["free_treatment"] == "COGS":
        _treat_txt = f"It is booked to <b>COGS</b>, so reported margin carries it; paid-customer margin is <b>{fmt_pct(tot['paid_gm'])}</b> before it."
    elif P["free_treatment"] == "SPLIT":
        _treat_txt = (f"<b>Trials</b> (≈${tot['trial_cost']:,.0f}) are booked to sales & marketing and the perpetual <b>freemium</b> tier (≈${tot['freemium_cost']:,.0f}) to COGS, "
                      f"giving a reported margin of {fmt_pct(tot['gm'])}; paid-customer margin is {fmt_pct(tot['paid_gm'])}.")
    else:
        _treat_txt = (f"It is booked to <b>sales & marketing</b>, so reported COGS excludes it (reported margin {fmt_pct(tot['gm'])}). It is still a real cost: "
                      "judge it as acquisition spend against free-to-paid conversion.")
    bullets.append(f"<b>Free-Tier Cost-to-Serve:</b> Free users generate {free_share * 100:.0f}% of prompts and ≈${free_cost_alloc:,.0f} of inference cost ({_free_pct_rev}). {_treat_txt}")
if tot["other"] > 0:
    bullets.append(f"<b>Fully Loaded Margin:</b> Your other COGS lines add ${tot['other']:,.0f} over the horizon, taking margin from {fmt_pct(tot['gm'])} (reported) to <b>{fmt_pct(tot['fl_gm'])}</b> fully loaded.")
else:
    bullets.append(f"<b>Fully Loaded Margin:</b> No other COGS lines entered, so margin covers AI-direct costs only. Add support, observability, compliance and similar costs in the "
                   f"'Fully loaded margin' panel to see the fully loaded margin.")
if not ai_path_is_flat(P):
    _flat_df = run_model_simulation({**P, "eff_mom": 0.0, "price_step_pct": 0.0})
    _api_delta = float(df["API Cost ($)"].sum() - _flat_df["API Cost ($)"].sum())
    _gm_flat = totals(_flat_df)["gm"]
    _gm_delta_txt = "" if (pd.isna(period_gm) or pd.isna(_gm_flat)) else f" and moves forecast-period margin by {period_gm - _gm_flat:+.1f}pp"
    bullets.append(f"<b>AI Cost Path:</b> {ai_path_text(P, H)}. Versus flat prices this changes API cost by {'-' if _api_delta < 0 else '+'}${abs(_api_delta):,.0f}{_gm_delta_txt}. "
                   "A planning assumption, not a vendor commitment.")
st.markdown(f"""
<div class="cfo-commentary">
    <h4>💡 Deal Desk & Contract Pricing Governance Briefing</h4>
    <ul>{"".join(f"<li>{b}</li>" for b in bullets)}</ul>
</div>
""", unsafe_allow_html=True)
st.markdown(f'<div class="assumption-flag">⚠️ {token_price_note(P, H)}</div>', unsafe_allow_html=True)


# ==========================================
# 4. DASHBOARD TABS
# ==========================================
tornado_cached = st.cache_data(show_spinner=False)(tornado_table)
goal_seek_cached = st.cache_data(show_spinner=False)(goal_seek)

QA_QUESTIONS = {
    "free2x": "What if free usage doubles?",
    "lever": "Which lever gets me to my target margin?",
    "disc30": "Can I afford a 30% discount?",
    "explain": "Explain the three margins",
}


def qa_answer(qid, p, dframe, t, target):
    """Rule-based what-if answers. Every number comes from the same engine (no AI estimate); returns text, optional bars and optional table."""
    def pct(v):
        return "n/a" if pd.isna(v) else f"{v:.1f}%"
    out = {"q": QA_QUESTIONS[qid], "md": "", "bars": None, "table": None, "changed": ""}
    if qid == "free2x":
        newv = min(1.0, p["free_mult"] * 2)
        t2 = totals(simulate({**p, "free_mult": newv}))
        out["md"] = (f"**Reported margin moves from {pct(t['gm'])} to {pct(t2['gm'])}.** Free-tier inference goes from ${t['free_cost']:,.0f} to ${t2['free_cost']:,.0f}, "
                     f"while paid-customer margin goes from {pct(t['paid_gm'])} to {pct(t2['paid_gm'])}. Free usage is capped at 100% of a paying user's usage.")
        out["bars"] = ("Reported margin (%)", ["Now", "Free usage x2"], [t["gm"], t2["gm"]])
        out["changed"] = f"Free-tier usage multiplier {p['free_mult'] * 100:.0f}% to {newv * 100:.0f}%."
    elif qid == "disc30":
        t2 = totals(simulate({**p, "arpu_pro": p["arpu_pro"] * 0.7, "arpu_ent": p["arpu_ent"] * 0.7}))
        verdict = "stays above" if (not pd.isna(t2["gm"]) and t2["gm"] >= target) else "falls below"
        out["md"] = (f"**With 30% off every price, reported margin moves from {pct(t['gm'])} to {pct(t2['gm'])}, which {verdict} your {target:.0f}% target.** "
                     f"Revenue over the forecast goes from ${t['rev']:,.0f} to ${t2['rev']:,.0f}; paid-customer margin goes from {pct(t['paid_gm'])} to {pct(t2['paid_gm'])}. "
                     "This ignores any extra volume a discount might win, so it is the worst case for margin.")
        out["bars"] = ("Reported margin (%)", ["Now", "30% discount"], [t["gm"], t2["gm"]])
        out["changed"] = f"Pro price ${p['arpu_pro']:,.2f} to ${p['arpu_pro'] * 0.7:,.2f}; Team / SMB price x0.7."
    elif qid == "lever":
        rows = []
        for key, lab, lo, hi, kind in GOAL_LEVERS:
            g = goal_seek_cached(p, key, lo, hi, float(target), "gm")
            fm = (lambda v: f"{v * 100:.1f}%") if kind == "pct" else ((lambda v: f"${v:,.2f}") if kind == "usd" else (lambda v: f"x{v:.2f}"))
            if g["status"] == "ok":
                rows.append({"Lever": lab.split(" (")[0], "Now": fm(g["current_value"]), "Needed alone": fm(g["value"])})
            else:
                rows.append({"Lever": lab.split(" (")[0], "Now": fm(g["current_value"]), "Needed alone": "not reachable on its own"})
        out["table"] = pd.DataFrame(rows)
        out["md"] = (f"**To reach a {target:.0f}% reported margin (now {pct(t['gm'])}), here is what each lever would need to be on its own.** "
                     "Real plans combine several smaller moves. The Levers & Goal-Seek tab lets you explore them.")
        out["changed"] = "One lever at a time; everything else as in the sidebar."
    else:
        out["md"] = (f"**Paid-customer margin ({pct(t['paid_gm'])})** counts only the cost of serving paying customers. **Reported margin ({pct(t['gm'])})** also includes free-tier cost "
                     f"if your policy books it to COGS (${t['free_cost']:,.0f} here). **Fully loaded margin ({pct(t['fl_gm'])})** adds any other monthly costs you enter, such as support and tooling. "
                     "The same inference bill appears in all three. These are management measures, not standardized accounting figures.")
        out["changed"] = "Nothing changed."
    return out

lever_curve_cached = st.cache_data(show_spinner=False)(lever_curve)


if GUIDED:
    with slot_chat:
        st.markdown('<div class="chapter" id="ask"><div class="eyebrow">Ask the model</div><h2>Try a <span class="accent">what-if</span></h2>'
                    '<p class="lede">Tap a question. The answer comes from the same model as everything else, with the settings it changed listed underneath. '
                    'It is a rule-based answer, not an AI estimate.</p></div>', unsafe_allow_html=True)
        with _keyed_container("chips"):
            _qcols = st.columns(len(QA_QUESTIONS))
            for _qc, (_qid, _qlab) in zip(_qcols, QA_QUESTIONS.items()):
                if _qc.button(_qlab, key=f"qa_{_qid}"):
                    st.session_state["qa"] = _qid
        _sel = st.session_state.get("qa")
        if _sel in QA_QUESTIONS:
            _ans = qa_answer(_sel, P, df, tot, float(margin_target))
            with st.chat_message("user"):
                st.markdown(_ans["q"])
            with st.chat_message("assistant"):
                st.markdown(_ans["md"])
                if _ans["bars"]:
                    _bt, _bl, _bv = _ans["bars"]
                    _bf = go.Figure()
                    _bf.add_trace(go.Bar(x=_bl, y=[0 if pd.isna(v) else v for v in _bv], marker_color=[tone("#0b2cbc", "#7f97ff"), tone("#9aa8ec", "#4a5fb5")][:len(_bl)]))
                    _bf.update_layout(**CHART_KW, height=220, margin=dict(l=30, r=20, t=10, b=20), yaxis=dict(title=_bt), showlegend=False)
                    show_chart(_bf)
                if _ans["table"] is not None:
                    show_df(_ans["table"], hide_index=True)
                st.caption(f"Settings changed: {_ans['changed']} Same engine as the rest of the app; token prices are illustrative placeholders.")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab_deal, tab10 = st.tabs([
    "📊 Revenue & Margin", "💳 Credits & Cash", "📋 Monthly Detail",
    "📈 Scenario Matrix", "📑 Audit Register", "🎯 Value & Pricing Strategy",
    "🎛️ Levers & Goal-Seek", "💰 CAC & LTV", "🧪 Compare & Export", "🤝 Deal Desk", "💬 Feedback",
])

with tab1:
    st.subheader("Forecast Recognized Revenue, COGS & Gross Margin")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["Month"], y=df["Recognized Revenue ($)"], name="Recognized Revenue ($)", marker_color="#3182ce"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Total Modeled COGS ($)"], name="Reported COGS ($)", mode="lines+markers", line=dict(color="#e53e3e", width=3)), secondary_y=False)
    if df["Other COGS ($)"].sum() > 0:
        fig.add_trace(go.Scatter(x=df["Month"], y=df["Fully Loaded COGS ($)"], name="Fully Loaded COGS ($)", mode="lines", line=dict(color="#d69e2e", width=2, dash="dash")), secondary_y=False)
    if budget_monthly > 0:
        fig.add_trace(go.Scatter(x=df["Month"], y=[budget_monthly] * H, name="COGS Budget Cap ($)", mode="lines", line=dict(color="#dd6b20", width=1.5, dash="dot")), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Modeled Gross Margin (%)"], name="Reported Gross Margin (%)", mode="lines+markers", line=dict(color="#27965a", width=3, dash="dash")), secondary_y=True)
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Paid-Customer Gross Margin (%)"], name="Paid-Customer GM (%)", mode="lines", line=dict(color=tone("#2b6cb0", "#63b3ed"), width=1.5, dash="dot")), secondary_y=True)
    if df["Other COGS ($)"].sum() > 0:
        fig.add_trace(go.Scatter(x=df["Month"], y=df["Fully Loaded Gross Margin (%)"], name="Fully Loaded GM (%)", mode="lines", line=dict(color="#d69e2e", width=2)), secondary_y=True)
    fig.add_trace(go.Scatter(x=df["Month"], y=[margin_target] * H, name=f"Target Margin ({margin_target}%)", mode="lines", line=dict(color="#a0aec0", width=1.5, dash="dot")), secondary_y=True)
    fig.update_layout(**CHART_KW, height=350, margin=dict(l=30, r=40, t=10, b=20), legend=dict(orientation="h", y=1.12))
    fig.update_yaxes(title_text="USD ($ / Month)", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="Gross Margin (%)", secondary_y=True)
    show_chart(fig)
    st.caption("Gross margin here is a management metric: payment processing fees and implementation delivery cost are included in COGS, and free-tier inference is booked to "
               f"{TREAT_LABEL[P['free_treatment']]} (sidebar 3). API cost uses the illustrative token prices in the sidebar; AI cost path: {ai_path_text(P, H)}. "
               + ("Other COGS lines you entered are shown as the fully loaded view." if tot["other"] > 0 else
                  "Support, observability, compliance and other COGS are not included unless you add lines in the 'Fully loaded margin' panel."))

    st.markdown("#### Margin Bridge: paid-customer → reported → fully loaded")
    _rev_p, _rev_e = tot["rev"], float(m_last["Recognized Revenue ($)"])

    def _pct(v, base):
        return v / base * 100 if base > 0 else np.nan

    _bridge_spec = [
        ("Recognized revenue", tot["rev"], _rev_e),
        ("Less: paid-customer COGS (inference, vector, infra, payment fees, implementation)", tot["paid_cogs"], float(m_last["Paid-Customer COGS ($)"])),
        ("= Paid-customer gross profit", tot["rev"] - tot["paid_cogs"], _rev_e - float(m_last["Paid-Customer COGS ($)"])),
        ("Less: free-tier inference cost booked to COGS", tot["free_cogs"], float(m_last["Free-Tier Cost in COGS ($)"])),
        ("= Gross profit as reported", tot["gp"], float(m_last["Modeled Gross Profit ($)"])),
        ("Less: other COGS (your lines)", tot["other"], float(m_last["Other COGS ($)"])),
        ("= Fully loaded gross profit", tot["rev"] - tot["fl_cogs"], float(m_last["Fully Loaded Gross Profit ($)"])),
        ("Memo: free-tier inference cost booked to sales & marketing", tot["free_sm"], float(m_last["Free-Tier Cost in S&M ($)"])),
    ]
    bridge = pd.DataFrame([{"Line": lab, "Forecast Period ($)": v, "Period, % of Revenue (%)": _pct(v, _rev_p),
                            f"Ending Month M{H} ($)": ve, f"Ending Month, % of Revenue (%)": _pct(ve, _rev_e)} for lab, v, ve in _bridge_spec])
    show_df(style_frame(bridge), hide_index=True)
    st.caption("Costs are shown as positive amounts; lines starting with '=' are subtotals. Free-tier inference is a real cost however it is booked: the sidebar switch only moves it "
               "between COGS and sales & marketing, and paying customers' margin never carries it. Other COGS lines are not part of cash or credit timing.")

    st.markdown("#### Inference Traffic & Cost by Customer Type")
    _pt, _ph, _pm = (float(df[c].sum()) for c in ("Typical Pro Prompts", "Heavy Pro Prompts", "Team / SMB Prompts"))
    _paid_pr = _pt + _ph + _pm
    _paid_pool = float(df["Paid-Customer Inference & Infra ($)"].sum())

    def _paid_cost(x):
        return _paid_pool * x / _paid_pr if _paid_pr > 0 else 0.0

    _ft = P["free_treatment"]
    _book_trial = "COGS" if _ft == "COGS" else "Sales & marketing"
    _book_free = "Sales & marketing" if _ft == "S&M" else "COGS"
    _book_int = "COGS" if P["internal_treatment"] == "COGS" else "R&D / G&A"
    _tr_rows = [
        ("Paying: Pro, typical users", _pt, _paid_cost(_pt), "COGS"),
        ("Paying: Pro, heavy users", _ph, _paid_cost(_ph), "COGS"),
        ("Paying: Team / SMB", _pm, _paid_cost(_pm), "COGS"),
        ("Free trials", float(df["Trial Prompts"].sum()), tot["trial_cost"], _book_trial),
        ("Free tier (perpetual freemium)", float(df["Freemium Prompts"].sum()), tot["freemium_cost"], _book_free),
        ("Internal / non-customer", float(df["Internal Prompts"].sum()), tot["internal_cost"], _book_int),
    ]
    _tp, _tc = sum(r[1] for r in _tr_rows), sum(r[2] for r in _tr_rows)
    traffic = pd.DataFrame(
        [{"Customer Type": lab, "Prompts": pr, "Share of Prompts (%)": pr / _tp * 100 if _tp > 0 else np.nan,
          "Inference & Infra Cost ($)": c, "Share of Cost (%)": c / _tc * 100 if _tc > 0 else np.nan,
          "Cost / 1,000 Prompts ($)": c / pr * 1000 if pr > 0 else np.nan, "Booked To": bk} for lab, pr, c, bk in _tr_rows]
        + [{"Customer Type": "Total", "Prompts": _tp, "Share of Prompts (%)": 100.0 if _tp > 0 else np.nan, "Inference & Infra Cost ($)": _tc,
            "Share of Cost (%)": 100.0 if _tc > 0 else np.nan, "Cost / 1,000 Prompts ($)": _tc / _tp * 1000 if _tp > 0 else np.nan, "Booked To": ""}])
    show_df(style_frame(traffic), hide_index=True)
    st.caption("Cost = API + vector + attributed fixed infrastructure over the horizon. The three paying segments share the paid pool by prompt volume. "
               "By default all fixed infrastructure stays with paying customers (Advanced > E), which is why their cost per 1,000 prompts is higher than the free tier's. "
               "Trial vs freemium needs the trial share set in Advanced > F; internal traffic is 0 unless you set it there.")

    st.markdown("#### Founder Unit-Economics Snapshot (ending month)")
    k1, k2, k3, k4 = st.columns(4)
    end_paid = float(m_last["Ending Paid Accounts"])
    cogs_pp = float(m_last["Paid-Customer COGS ($)"]) / end_paid if end_paid > 0 else 0.0
    rev_pp = float(m_last["Recognized Revenue ($)"]) / end_paid if end_paid > 0 else 0.0
    free_pp = float(m_last["Free-Tier Inference Cost ($)"]) / end_paid if end_paid > 0 else 0.0
    k1.metric("Paid-customer COGS / paid account", f"${cogs_pp:,.2f}")
    k2.metric("Revenue / paid account", f"${rev_pp:,.2f}")
    k3.metric("Paid-customer gross profit / paid account", f"${rev_pp - cogs_pp:,.2f}")
    k4.metric("Memo: free-tier cost / paid account", f"${free_pp:,.2f}")
    st.caption("Company-level averages across Pro subscribers and team/SMB accounts (a team account is a contract, not a user). "
               "Free-tier cost is shown as a memo and is NOT loaded onto paying accounts, so per-account cost reflects what it costs to serve a paying customer.")

    st.markdown("#### Inference Unit Cost Telemetry")
    cu1, cu2 = st.columns(2)
    fu1 = go.Figure()
    fu1.add_trace(go.Scatter(x=df["Month"], y=df["API Cost / 1,000 Prompts ($)"], name="API-only / 1k", mode="lines+markers", line=dict(color=tone("#805ad5", "#b794f4"), width=2.5)))
    fu1.add_trace(go.Scatter(x=df["Month"], y=df["Blended Direct Cost / 1,000 Prompts ($)"], name="Blended direct cost / 1k", mode="lines+markers", line=dict(color="#e53e3e", width=2, dash="dash")))
    fu1.update_layout(title="API Cost vs Blended Direct Cost per 1,000 Prompts ($)", **CHART_KW, height=250, margin=dict(l=30, r=30, t=40, b=20), legend=dict(orientation="h", y=1.18))
    with cu1:
        show_chart(fu1)
    fu2 = go.Figure()
    fu2.add_trace(go.Scatter(x=df["Month"], y=df["Paid-Customer COGS / Paid Account ($)"], mode="lines+markers", name="Paid-customer COGS / paid account", line=dict(color="#319795", width=2.5)))
    fu2.add_trace(go.Scatter(x=df["Month"], y=df["Incl. Free-Tier Cost / Paid Account ($)"], mode="lines", name="Incl. free-tier cost (memo)", line=dict(color="#a0aec0", width=1.5, dash="dash")))
    fu2.update_layout(title="COGS per Paid Account ($)", legend=dict(orientation="h", y=1.18), **CHART_KW, height=250, margin=dict(l=30, r=30, t=40, b=20))
    with cu2:
        show_chart(fu2)

    st.markdown("#### COGS Component Disaggregation")
    cogs_fig = go.Figure()
    for col, nm, colr in [("API Cost ($)", "API Inference", "#4299e1"), ("Vector DB Cost ($)", "Vector DB", "#4fd1c5"),
                          ("Fixed Infrastructure ($)", "Infrastructure Step-Cost", "#ed8936"),
                          ("Payment Processing ($)", "Payment Fees", "#9f7aea"), ("Implementation Cost ($)", "Implementation Delivery", "#a0aec0")]:
        cogs_fig.add_trace(go.Bar(x=df["Month"], y=df[col], name=nm, marker_color=colr))
    if df["Other COGS ($)"].sum() > 0:
        cogs_fig.add_trace(go.Bar(x=df["Month"], y=df["Other COGS ($)"], name="Other COGS (your lines)", marker_color="#d69e2e"))
    cogs_fig.add_trace(go.Scatter(x=df["Month"], y=df["Total Modeled COGS ($)"], name="Reported COGS", mode="lines", line=dict(color=tone("#2d3748", "#e2e8f0"), width=2, dash="dot")))
    cogs_fig.update_layout(**CHART_KW, barmode="stack", height=320, margin=dict(l=30, r=30, t=20, b=20), legend=dict(orientation="h", y=1.12))
    show_chart(cogs_fig)
    st.caption("Bars show every direct cost at full value, regardless of whether free-tier cost is booked to COGS or sales & marketing; the dotted line is reported COGS.")

    st.markdown("#### Team / SMB Account Roll-Forward")
    fe = make_subplots(specs=[[{"secondary_y": True}]])
    fe.add_trace(go.Bar(x=df["Month"], y=df["Ending Team Accounts"], name="Ending Accounts", marker_color="#3182ce"), secondary_y=False)
    fe.add_trace(go.Bar(x=df["Month"], y=-df["Team Non-Renewals"], name="Non-Renewals", marker_color="#e53e3e"), secondary_y=False)
    fe.add_trace(go.Scatter(x=df["Month"], y=df["Team Expansion Index"], name="Expansion Index (x)", mode="lines+markers", line=dict(color="#27965a", width=2)), secondary_y=True)
    fe.update_layout(**CHART_KW, barmode="relative", height=300, margin=dict(l=30, r=40, t=10, b=20), legend=dict(orientation="h", y=1.15))
    show_chart(fe)

with tab2:
    st.subheader("Credit Drawdown Velocity, Cash Payable & Cash Paid")
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Bar(x=df["Month"], y=df["AI Credits Applied ($)"], name="AI Credits Applied ($)", marker_color="#38a169"), secondary_y=False)
    fig2.add_trace(go.Bar(x=df["Month"], y=df["Cloud Credits Applied ($)"], name="Cloud Credits Applied ($)", marker_color="#4fd1c5"), secondary_y=False)
    fig2.add_trace(go.Bar(x=df["Month"], y=df["Estimated Cash Payable ($)"], name="Cash Payable, accrual ($)", marker_color="#dd6b20"), secondary_y=False)
    fig2.add_trace(go.Scatter(x=df["Month"], y=df["Estimated Cash Paid This Month ($)"], name="Estimated Cash Paid ($)", mode="lines+markers", line=dict(color=tone("#805ad5", "#b794f4"), width=2)), secondary_y=False)
    fig2.add_trace(go.Scatter(x=df["Month"], y=df["Remaining AI Credits ($)"], name="Remaining AI Credits ($)", mode="lines", line=dict(color=tone("#2b6cb0", "#63b3ed"), width=2, dash="dash")), secondary_y=True)
    fig2.add_trace(go.Scatter(x=df["Month"], y=df["Remaining Cloud Credits ($)"], name="Remaining Cloud Credits ($)", mode="lines", line=dict(color=tone("#805ad5", "#b794f4"), width=2, dash="dot")), secondary_y=True)
    fig2.update_layout(**CHART_KW, height=380, barmode="stack", margin=dict(l=30, r=40, t=10, b=20), legend=dict(orientation="h", y=1.15))
    fig2.update_yaxes(title_text="Monthly COGS coverage ($)", secondary_y=False, rangemode="tozero")
    fig2.update_yaxes(title_text="Credit balances ($)", secondary_y=True, rangemode="tozero")
    show_chart(fig2)
    st.caption("Credits reduce cash payable but NOT modeled gross profit (underlying unit economics). 'Cash Paid' shifts vendor invoices by the payment lag; payment fees and implementation costs are paid immediately.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AI Credits", ai_stat)
    m2.metric("Cloud Credits", cloud_stat)
    m3.metric("Vendor Invoices Outstanding at Horizon", f"${float(m_last['Vendor Invoices Outstanding ($)']):,.0f}")
    m4.metric("Revenue Less Net Vendor Payables & Immediate Costs (horizon)",
              f"${float(df['Revenue Less Net Vendor Payables & Immediate Costs ($)'].sum()):,.0f}",
              help="Recognized revenue minus estimated cash payable (vendor invoices net of credits, plus payment fees and implementation costs). Not cash flow: invoice timing and customer collections are not modeled.")

with tab3:
    st.subheader("Monthly Roll-Forward Detail")
    show_df(style_frame(df), hide_index=True)
    st.download_button("Download Full Audit CSV", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="ai_inference_deal_desk_model.csv", mime="text/csv")

with tab4:
    st.subheader("📈 Multi-Scenario Comparative Analysis Matrix")
    st.caption("Each scenario moves one driver family unless labeled combined. Prompt intensity scales ALL segments, including team/SMB. Deal counts are not rounded. "
               "These are simplified sensitivities on one operating plan, not independent plans. "
               "Scenarios 8-9 apply a flat price change to every month of the horizon; they are not a price-decline trajectory. "
               f"The AI cost path set in the sidebar ({ai_path_text(P, H)}) and the free-tier treatment ({TREAT_LABEL[P['free_treatment']]}) apply to every scenario, and each scenario's other COGS lines follow the same rules.")
    scen_rows = []
    scen_totals = {}
    for name, spec in SCENARIOS:
        try:
            sd = run_model_simulation(apply_scenario(P, spec))
        except Exception as e:
            st.error(f"Scenario '{name}' failed: {e}")
            st.stop()
        scen_totals[name] = totals(sd)
    base_t = scen_totals[SCENARIOS[0][0]]
    for idx, (name, spec) in enumerate(SCENARIOS, start=1):
        t = scen_totals[name]
        scen_rows.append({
            "ID": idx, "Scenario": name,
            "Revenue ($)": t["rev"], "Δ Revenue vs Base ($)": t["rev"] - base_t["rev"],
            "COGS ($)": t["cogs"], "Δ COGS vs Base ($)": t["cogs"] - base_t["cogs"],
            "Gross Profit ($)": t["gp"], "Gross Margin (%)": t["gm"], "Fully Loaded GM (%)": t["fl_gm"],
            "Δ Margin vs Base (pp)": (t["gm"] - base_t["gm"]) if not (pd.isna(t["gm"]) or pd.isna(base_t["gm"])) else np.nan,
            "Ending ARR ($)": t["arr"],
            "Cash Payable, Horizon ($)": t["payable"], "Cash Paid, Horizon ($)": t["paid"],
            "Δ Cash Payable vs Base ($)": t["payable"] - base_t["payable"],
            "Invoices Outstanding at Horizon ($)": t["outstanding"],
        })
    scen_df = pd.DataFrame(scen_rows)
    show_df(style_frame(scen_df), hide_index=True)

with tab5:
    st.subheader("📑 Assumptions & Audit Register (Data Governance)")
    st.caption("Generated from the live inputs, so values cannot drift from the model. 'Status' is an honest label: nothing here has been verified against actuals by this tool.")
    PRESET_EST = "Founder-profile preset (estimate) — replace with actuals"
    ASSUME = "Model assumption — verify"
    TOKEN_FLAG = "KEY ASSUMPTION — illustrative price (flat unless an AI cost path is set); verify against vendor rate card"
    audit = [
        ("Forecast start / horizon", f"{pd.Period(freq='M', year=start_year, month=start_month).strftime('%b %Y')} / {H} months", "FP&A planning calendar", "Planning choice"),
        ("Starting DAU", f"{starting_dau:,.0f}", "Mixpanel / Segment / PostHog (define DAU consistently)", PRESET_EST),
        ("Starting Pro subscribers", f"{starting_pro_subs:,.0f}", "Billing system (Stripe) active subscriptions", PRESET_EST),
        ("Starting team / SMB accounts", f"{starting_ent_accounts:,.0f}", "Contract register / CRM", PRESET_EST),
        ("Total DAU monthly growth", f"{dau_monthly_growth * 100:.1f}%", "Acquisition model / analytics trend", PRESET_EST),
        ("Net new team / SMB deals / month", f"{net_new_ent_contracts:,.2f}", "Sales pipeline (CRM) weighted forecast", PRESET_EST),
        ("Baseline prompts / active paid user / day", f"{baseline_prompts}", "LLM gateway logs (Helicone / Langfuse)", PRESET_EST),
        ("Target gross margin / COGS budget cap", f"{margin_target}% / " + (f"${budget_monthly:,.0f}" if budget_monthly > 0 else "none"), "Board / finance policy", "Policy input"),
        ("Paid-user daily activity factor", f"{P['paid_dau_factor'] * 100:.0f}%", "Product analytics (paid DAU / paid subs)", ASSUME),
        ("New-user conversion to Pro", f"{P['new_conv'] * 100:.1f}%", "Funnel analytics, cohort-based", ASSUME),
        ("Existing free-pool conversion / month", f"{P['base_conv'] * 100:.1f}%", "Funnel analytics, cohort-based", ASSUME),
        ("Pro monthly churn", f"{P['churn'] * 100:.1f}%", "Billing cohort retention", ASSUME),
        ("Pro ARPU / overage-expansion", f"${P['arpu_pro']:,.2f} / +{P['pro_expansion_pct'] * 100:.0f}%", "Price book / metered billing invoices", ASSUME),
        ("Team / SMB monthly value / account", f"${P['arpu_ent']:,.2f}", "Signed order forms", ASSUME),
        ("Team / SMB term / renewal rate / in-term churn", f"{P['ent_term']} mo / {P['ent_renewal_rate'] * 100:.0f}% / {P['ent_in_term_churn'] * 100:.1f}%", "Contract register; historical renewals", ASSUME),
        ("Team / SMB renewal-month exemption", "Cohorts at term end face the renewal decision only; in-term churn is not also applied to them that month", "Contract terms (termination for cause, insolvency)", ASSUME),
        ("Team / SMB expansion / seats / prompts per account", f"{P['ent_expansion'] * 100:.1f}% mo / {P['ent_seats']} seats / {P['ent_prompts']:,} prompts", "Account usage telemetry", ASSUME),
        ("Implementation fee / recognition / delivery cost", f"${P['impl_fee']:,.0f} / {P['impl_rec_months']} mo / ${P['impl_cost']:,.0f}", "Order forms; revenue-recognition policy; PS cost model", ASSUME),
        ("Payment processing fee", f"{P['fee_pct'] * 100:.1f}%", "Processor merchant statement", ASSUME),
        ("Usage peaks (calendar months)", f"{calendar.month_name[P['p1_month']]} @ {P['p1_vol']} / {calendar.month_name[P['p2_month']]} @ {P['p2_vol']}; σ {P['p1_width']} / {P['p2_width']}", "Seasonality in gateway logs", ASSUME),
        ("Heavy-user share / calibration", f"{P['heavy_share'] * 100:.0f}% / " + (f"{P['heavy_mult']:.1f}x multiplier" if P["heavy_mode"] == "Multiplier" else f"Pareto {P['pareto_share'] * 100:.0f}% of prompts (derived {h_mult_used:.1f}x)"), "Per-user prompt distribution", ASSUME),
        ("Free-tier usage multiplier", f"{P['free_mult'] * 100:.0f}% of typical", "Gateway logs by plan", ASSUME),
        ("Tokens / prompt (in / out) and MoM growth", f"{P['in_tok']:,} / {P['out_tok']:,}; +{P['in_grow'] * 100:.1f}% / +{P['out_grow'] * 100:.1f}% "
                                                      f"(input tokens x{(1 + P['in_grow']) ** max(0, H - 1):.2f} by month {H})", "LLM provider logs", ASSUME),
        ("Frontier routing share and prices", f"{P['front_mix'] * 100:.0f}%; ${P['f_in']:.2f} in / ${P['f_out']:.2f} out per 1M", "Vendor rate card (public, unverified)", TOKEN_FLAG),
        ("Standard model prices", f"${P['s_in']:.4f} in / ${P['s_out']:.4f} out per 1M", "Vendor rate card (public, unverified)", TOKEN_FLAG),
        ("AI cost path (efficiency / vendor price change)",
         ("Flat: no price decline or increase is modeled. Scenarios 8-9 apply one flat % change to every month" if ai_path_is_flat(P) else ai_path_text(P, H)),
         "Vendor price history and commit terms; engineering cost-reduction plan with owners and dates", TOKEN_FLAG),
        ("Free / trial inference cost treatment",
         f"Booked to {TREAT_LABEL[P['free_treatment']]}; fixed infrastructure attributed to free tier: "
         f"{'by prompt share' if P['infra_alloc'] == 'prompt' else 'none (stays with paid customers)'}",
         "Accounting policy (CPA / auditor); investor reporting convention", "Policy choice — apply consistently and disclose"),
        ("Trial share of free-tier traffic", f"{P['trial_share'] * 100:.0f}% trials / {(1 - P['trial_share']) * 100:.0f}% freemium",
         "Funnel analytics: trial starts and trial prompt logs", "Illustrative placeholder — replace with your own funnel data"),
        ("Free-tier model routing", ("Same mix as paying customers" if P["front_mix_free"] is None else f"{P['front_mix_free'] * 100:.0f}% frontier on free-tier traffic"),
         "Gateway routing logs by plan", ASSUME),
        ("Internal / non-customer traffic", f"{P['internal_pct'] * 100:.0f}% of customer prompts, booked to {'COGS' if P['internal_treatment'] == 'COGS' else 'R&D / G&A'}",
         "Gateway logs by API key (staff, eval, staging keys)", ASSUME),
        ("Other COGS lines (fully loaded view)",
         ("; ".join(f"{ln['name']}: " + (f"{ln['amount']:g}% of revenue" if ln["basis"] == "% of revenue" else f"${ln['amount']:,.0f}/mo" + (f" (+{ln['growth']:g}%/mo)" if ln["growth"] else ""))
                    + f" from M{ln['start']}" + (f" to M{ln['end']}" if ln["end"] else "") for ln in P["other_cogs"]) or "None entered"),
         "Chart of accounts; support & success headcount; tooling, security and compliance invoices",
         ASSUME if P["other_cogs"] else "Not entered — fully loaded margin equals reported margin"),
        ("Modeled COGS scope",
         "API inference, vector DB, fixed infrastructure, payment fees, implementation delivery" + (f" plus {len(P['other_cogs'])} user-defined line(s) in the fully loaded view"
                                                                                                    if P["other_cogs"] else ". Excludes support, observability, compliance and other COGS unless you add lines"),
         "Chart of accounts / COGS policy",
         "User-defined lines included in fully loaded margin" if P["other_cogs"] else "Known limitation — modeled margin is higher than a fully loaded margin"),
        ("Prompt caching", f"{P['cache_rate'] * 100:.0f}% hit / {P['cache_ratio'] * 100:.0f}% price", "Gateway cache logs; vendor terms", ASSUME),
        ("Batch routing / inference overhead", f"{P['batch_share'] * 100:.0f}% at {P['batch_ratio'] * 100:.0f}% price / +{P['overhead'] * 100:.0f}%", "Batch API telemetry", ASSUME),
        ("Base infrastructure / step block / step cost", f"${P['base_infra']:,.0f} / {P['step_thresh']:,.0f} ({P['step_type']}) / ${P['step_cost']:,.0f}", "Cloud invoices; capacity plan", PRESET_EST),
        ("Vector DB cost / 1k prompts", f"${P['vec_cost']:.3f}", "Vector vendor invoices", ASSUME),
        ("Starting credits (AI / Cloud)", f"${P['ai_credits']:,.0f} / ${P['cloud_credits']:,.0f}", "Grant agreements and vendor portals", PRESET_EST),
        ("Credit expiry month / eligible share", f"AI {P['ai_expiry'] or 'none'} @ {P['ai_elig'] * 100:.0f}% / Cloud {P['cloud_expiry'] or 'none'} @ {P['cloud_elig'] * 100:.0f}%", "Grant terms", ASSUME),
        ("Vendor invoice payment lag", f"{P['invoice_lag']} month(s)", "AP terms with vendors", ASSUME),
    ]
    show_df(pd.DataFrame(audit, columns=["Model Parameter", "Current Value", "Verify Against (System of Record)", "Status"]), hide_index=True)

    st.markdown("#### Model Integrity Checks")
    st.caption("Reconciliations that run on every recalculation. Any FAIL means the model output should not be used until investigated.")
    # computed here with the pricing frame from tab 6 defaults so checks always exist
    _default_rules = {"payg_rate": 0.008, "sub_pro_fee": 30.0, "sub_pro_allow": 1500, "sub_ent_fee": 1000.0, "sub_ent_allow": 3000, "sub_over": 0.006,
                      "hyb_pro_fee": 20.0, "hyb_pro_allow": 1000, "hyb_ent_fee": 500.0, "hyb_ent_allow": 2000, "hyb_over": 0.008,
                      "prepaid_price": 1.0, "prepaid_ppc": 100.0, "prepaid_breakage": 0.10}
    checks_df = run_integrity_checks(df, P, base_totals=base_t, pricing_df=compare_pricing(df, P, _default_rules))
    n_fail = int((checks_df["Result"] == "FAIL").sum())
    n_na = int((checks_df["Result"] == "N/A").sum())
    n_pass = int((checks_df["Result"] == "PASS").sum())
    _na_txt = f" ({n_na} not applicable to these inputs)" if n_na else ""
    (st.success if n_fail == 0 else st.error)(
        f"{n_pass} of {n_pass + n_fail} applicable integrity checks passed{_na_txt}." if n_fail == 0 else f"{n_fail} integrity check(s) FAILED{_na_txt}.")
    st.caption("These checks confirm the arithmetic is internally consistent. They cannot prove an assumption is right: "
               "a consistently wrong input passes every check. Known-answer tests (hand-calculated cases) live in tests/test_engine.py.")
    show_df(checks_df, hide_index=True)

    st.markdown("#### Model Health Flags")
    st.caption("Rule-based sanity checks on inputs and results. ⚠️ = likely worth reviewing; ℹ️ = worth knowing, not necessarily wrong. Unlike the integrity checks above, these look at plausibility, not arithmetic.")
    for _lvl, _ttl, _txt in _flags:
        st.markdown(f"{'⚠️' if _lvl == 'check' else 'ℹ️'} **{_ttl}.** {_txt}")
    if not _flags:
        st.success("No flags.")
    st.markdown("#### Usage Curve: prompts per active user per day")
    _mn, _cv = usage_curve_for(P)
    uf = go.Figure()
    uf.add_trace(go.Scatter(x=_mn, y=_cv * P["usage_mult"], name="Typical paid user", mode="lines+markers", line=dict(color="#3182ce", width=2.5)))
    uf.add_trace(go.Scatter(x=_mn, y=_cv * P["usage_mult"] * h_mult_used, name=f"Heavy Pro user ({h_mult_used:.1f}x)", mode="lines", line=dict(color="#e53e3e", width=1.5, dash="dash")))
    uf.add_trace(go.Scatter(x=_mn, y=_cv * P["usage_mult"] * P["free_mult"], name=f"Free user ({P['free_mult'] * 100:.0f}% of typical)", mode="lines", line=dict(color="#27965a", width=2)))
    uf.add_trace(go.Scatter(x=_mn, y=[P["base_prompts"]] * len(_mn), name="Baseline", mode="lines", line=dict(color="#a0aec0", width=1.5, dash="dot")))
    uf.update_layout(**CHART_KW, height=300, margin=dict(l=30, r=30, t=10, b=20), legend=dict(orientation="h", y=1.15))
    uf.update_yaxes(title_text="Prompts / user / day", rangemode="tozero")
    show_chart(uf)
    st.caption("The curve the engine uses, by calendar month. Peaks are absolute prompts per day (not multipliers over baseline) and drive the cost of every user, including free users.")

with tab6:
    st.subheader("🎯 Value-Creation Anchor & Hybrid Pricing Strategy")
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.markdown("#### 1. Value-Creation Anchor Calculator")
        value_metric_type = st.selectbox("Primary AI Value Metric", ["Automation (Hours Saved)", "Augmentation (Error Reduction)", "Connection (Network Access)", "Revenue (Tasks / Leads Generated)"])
        if value_metric_type == "Automation (Hours Saved)":
            raw_value = st.number_input("Hours Saved / User / Month", min_value=0.0, value=10.0, step=1.0) * st.number_input("Average User Hourly Wage ($)", min_value=0.0, value=50.0, step=5.0)
        elif value_metric_type == "Augmentation (Error Reduction)":
            raw_value = st.number_input("Errors / Rework Incidents Avoided / User / Month", min_value=0.0, value=5.0, step=1.0) * st.number_input("Average Cost per Error ($)", min_value=0.0, value=40.0, step=5.0)
        elif value_metric_type == "Connection (Network Access)":
            raw_value = st.number_input("Qualified Connections / Introductions / User / Month", min_value=0.0, value=4.0, step=1.0) * st.number_input("Value per Connection ($)", min_value=0.0, value=25.0, step=5.0)
        else:
            raw_value = (st.number_input("Tasks / Leads Generated / User / Month", min_value=0.0, value=8.0, step=1.0)
                         * st.slider("Conversion / Close Rate (%)", 0, 100, 10) / 100
                         * st.number_input("Gross Profit per Conversion ($)", min_value=0.0, value=500.0, step=50.0))
        attribution = st.slider("Share of Value Attributable to the AI Product (%)", 0, 100, 100) / 100
        economic_value = raw_value * attribution
        st.info(f"💡 Economic value created: **${economic_value:,.0f} / user / month**")
        price_basis = st.selectbox("Price Basis", ["Pro (per user)", "Team / SMB (per seat)"])
        price_charged = P["arpu_pro"] if price_basis.startswith("Pro") else P["arpu_ent"] / max(1, P["ent_seats"])
        capture_text = f"{price_charged / economic_value * 100:.1f}%" if economic_value > 0 else "N/A (no value created)"
        roi_text = f"{economic_value / price_charged:.1f}x" if price_charged > 0 else "N/A (price is zero)"
        st.markdown(f"""
        * **Price charged ({price_basis}):** `${price_charged:,.2f} / user / mo`
        * **Value capture ratio:** `{capture_text}` of value created
        * **Customer ROI multiple:** `{roi_text}`
        """)

    with col_v2:
        st.markdown("#### 2. Fixed-Usage Pricing Archetype Comparison")
        st.caption("This is a sensitivity on constant customer behavior, not a forecast of what customers would buy: usage, plan mix and churn are held the same under every price. "
                   "All archetypes are applied to the SAME simulated customers and usage, segment by segment (typical Pro, heavy Pro, team / SMB). "
                   "Cost is the paid-attributable share of the API + vector + infrastructure pool (variable cost by prompt volume; fixed infrastructure per the sidebar attribution setting); payment fees are recomputed on each archetype's revenue. "
                   "Free-tier traffic (and any idle capacity in months with no traffic) earns no revenue, so its cost is shown separately.")
        with st.expander("Edit comparable pricing rules", expanded=False):
            rules = {
                "payg_rate": st.number_input("Pay-as-you-go price per prompt ($)", min_value=0.0, value=0.008, step=0.001, format="%.4f"),
                "sub_pro_fee": st.number_input("Subscription: Pro base fee / account / month ($)", min_value=0.0, value=30.0, step=5.0),
                "sub_pro_allow": st.number_input("Subscription: Pro included prompts / account / month", min_value=0, value=1500, step=250),
                "sub_ent_fee": st.number_input("Subscription: Team / SMB base fee / account / month ($)", min_value=0.0, value=1000.0, step=100.0),
                "sub_ent_allow": st.number_input("Subscription: Team / SMB included prompts / account / month", min_value=0, value=3000, step=250),
                "sub_over": st.number_input("Subscription: overage price per prompt ($)", min_value=0.0, value=0.006, step=0.001, format="%.4f"),
                "hyb_pro_fee": st.number_input("Hybrid: Pro base fee / account / month ($)", min_value=0.0, value=20.0, step=5.0),
                "hyb_pro_allow": st.number_input("Hybrid: Pro included prompts / account / month", min_value=0, value=1000, step=250),
                "hyb_ent_fee": st.number_input("Hybrid: Team / SMB base fee / account / month ($)", min_value=0.0, value=500.0, step=50.0),
                "hyb_ent_allow": st.number_input("Hybrid: Team / SMB included prompts / account / month", min_value=0, value=2000, step=250),
                "hyb_over": st.number_input("Hybrid: overage price per prompt ($)", min_value=0.0, value=0.008, step=0.001, format="%.4f"),
                "prepaid_price": st.number_input("Prepaid price per credit ($)", min_value=0.0, value=1.0, step=0.25),
                "prepaid_ppc": st.number_input("Prompts covered per credit", min_value=0.01, value=100.0, step=10.0),
                "prepaid_breakage": st.slider("Prepaid breakage (% of purchased credits never used)", 0, 50, 10) / 100,
            }
        pricing_df = compare_pricing(df, P, rules)
        show_df(style_frame(pricing_df), hide_index=True)
        st.caption("Billing Basis shows what each row charges for. Rows 1-4 do not include the Pro overage/expansion uplift or the team expansion index that the reference row 0 carries, so row 0 is not a like-for-like benchmark for revenue; compare margins and unit economics with that in mind. "
                   "Prepaid revenue = value of credits purchased (consumed + expired breakage); cash timing and deferred-revenue treatment are not modeled. "
                   "Contribution margin is before free-tier and idle cost; gross margin after free-tier cost is the comparable company-level figure.")
        st.markdown(f"**Estimated Pro usage ceiling at {margin_target}% target margin (month {H}):** "
                    + ("no headroom at this target and fee rate." if ceil["no_headroom"]
                       else f"{fmt_prompts(ceil['var'])} on variable cost (API + vector); {fmt_prompts(ceil['loaded'])} including allocated fixed infrastructure. "
                            f"Current average Pro usage that month: "
                            f"{(float(m_last['Typical Pro Prompts'] + m_last['Heavy Pro Prompts']) / max(1e-9, float(m_last['Average Pro Subs']))):,.0f} prompts per subscriber."))
        st.caption("An estimate from company-wide cost per prompt, not a guaranteed safe limit. Capacity steps, customer mix and real routing can move marginal cost. "
                   "Cost per prompt reflects the illustrative token prices and the AI cost path in the ending month (" + ai_path_text(P, H) + "), so the ceiling moves with them.")

# ---------- tab 7: levers & goal-seek ----------
with tab7:
    st.subheader("🎛️ What Moves Margin? Sensitivity & Goal-Seek")
    lc1, lc2 = st.columns(2)
    sens_metric_label = lc1.selectbox("Margin measure", list(METRIC_OPTIONS), index=0)
    sens_pct = lc2.slider("Move each input by ± (%)", 5, 50, 25, 5) / 100
    sens_key = METRIC_OPTIONS[sens_metric_label]
    base_m, torn = tornado_cached(P, sens_pct, sens_key)
    if pd.isna(base_m) or torn.empty:
        st.info("No margin to analyse: the model has no recognized revenue for these inputs.")
    else:
        _top = torn.iloc[0]
        st.info(f"💡 **Biggest lever:** {_top['Driver']} moves the margin by up to {_top['Δ Swing (pp)']:.1f}pp for a ±{sens_pct * 100:.0f}% change. Base margin: {base_m:.1f}%.")
        _ts = torn.iloc[::-1]
        tf = go.Figure()
        tf.add_trace(go.Bar(y=_ts["Driver"], x=_ts["Δ Margin at Low Input (pp)"], name=f"Input -{sens_pct * 100:.0f}%", orientation="h", marker_color="#3182ce"))
        tf.add_trace(go.Bar(y=_ts["Driver"], x=_ts["Δ Margin at High Input (pp)"], name=f"Input +{sens_pct * 100:.0f}%", orientation="h", marker_color="#e53e3e"))
        tf.update_layout(**CHART_KW, barmode="overlay", height=max(300, 34 * len(_ts) + 90), margin=dict(l=30, r=30, t=10, b=30), legend=dict(orientation="h", y=1.08))
        tf.update_xaxes(title_text="Change in margin (percentage points)")
        show_chart(tf)
        show_df(style_frame(torn), hide_index=True)
        st.caption("One input at a time, everything else held at your current settings: a local sensitivity, not a forecast range. Percent inputs move relative to their own value "
                   "(±25% of a 40% multiplier = 30%-50%) and are kept between 0% and 100%. Includes your free-tier treatment, other COGS lines and AI cost path.")

    st.markdown("#### Goal-Seek: what would it take to hit a target margin?")
    gc1, gc2, gc3 = st.columns(3)
    goal_lever_label = gc1.selectbox("Lever to change", [g[1] for g in GOAL_LEVERS], index=0)
    goal_target = gc2.slider("Target margin (%)", 0, 95, int(margin_target), 1)
    goal_metric_label = gc3.selectbox("Margin to hit", list(METRIC_OPTIONS), index=0)
    _gl = next(g for g in GOAL_LEVERS if g[1] == goal_lever_label)
    _gkey, _glo, _ghi, _gkind = _gl[0], _gl[2], _gl[3], _gl[4]
    _gmk = METRIC_OPTIONS[goal_metric_label]

    def _fmt_lever(v):
        return f"{v * 100:.1f}%" if _gkind == "pct" else (f"${v:,.2f}" if _gkind == "usd" else f"x{v:.2f}")

    gres = goal_seek_cached(P, _gkey, _glo, _ghi, float(goal_target), _gmk)
    if gres["status"] == "ok":
        st.success(f"To reach **{goal_target}%** ({goal_metric_label.lower()}), set **{goal_lever_label.split(' (')[0]}** to **{_fmt_lever(gres['value'])}** "
                   f"(currently {_fmt_lever(gres['current_value'])}). Check: the model gives {gres['achieved']:.1f}% at that value.")
    elif gres["status"] == "unreachable":
        st.warning(f"No value of this lever between {_fmt_lever(_glo)} and {_fmt_lever(_ghi)} reaches {goal_target}%: it moves the margin only from {gres['f_lo']:.1f}% to {gres['f_hi']:.1f}%. "
                   "Try a different lever or combine several (use the table above to pick).")
    else:
        st.info("The margin is undefined for these inputs (no recognized revenue).")
    _xs, _ys = lever_curve_cached(P, _gkey, _glo, _ghi, _gmk)
    if not np.all(np.isnan(_ys)):
        gf = go.Figure()
        gf.add_trace(go.Scatter(x=_xs * (100 if _gkind == "pct" else 1), y=_ys, name="Margin", mode="lines", line=dict(color="#3182ce", width=2.5)))
        gf.add_trace(go.Scatter(x=[_xs[0] * (100 if _gkind == "pct" else 1), _xs[-1] * (100 if _gkind == "pct" else 1)], y=[goal_target, goal_target], name="Target", mode="lines",
                                line=dict(color="#a0aec0", width=1.5, dash="dot")))
        if gres["status"] == "ok":
            gf.add_trace(go.Scatter(x=[gres["value"] * (100 if _gkind == "pct" else 1)], y=[gres["achieved"]], name="Solution", mode="markers", marker=dict(color="#27965a", size=11)))
        gf.update_layout(**CHART_KW, height=300, margin=dict(l=30, r=30, t=10, b=30), legend=dict(orientation="h", y=1.12))
        gf.update_xaxes(title_text=goal_lever_label.split(" (")[0] + (" (%)" if _gkind == "pct" else ""))
        gf.update_yaxes(title_text="Margin (%)")
        show_chart(gf)
    st.caption("Searches one lever at a time with everything else as set in the sidebar. It assumes the margin moves in one direction as the lever moves, which holds for these levers. "
               "Apply a result by setting the matching input in the sidebar (Advanced mode for some).")

# ---------- tab 8: CAC, LTV & payback ----------
with tab8:
    st.subheader("💰 CAC, LTV & Payback (free-tier cost counted as acquisition cost)")
    ac1, ac2 = st.columns(2)
    sm_monthly = ac1.number_input("Monthly sales & marketing spend ($, excluding free-tier inference)", min_value=0.0, value=0.0, step=500.0,
                                  help="Ads, sales salaries and commissions, tools, content. The model has no sales & marketing spend of its own, so enter yours.")
    count_free = ac2.toggle("Count free / trial inference as acquisition cost", value=True,
                            help="Free users are how Pro customers are acquired, so their inference cost is part of the cost of acquiring them, however it is booked in the accounts.")
    cl = cac_ltv(df, P, tot, sm_monthly, count_free)
    if sm_monthly <= 0:
        st.warning("No sales & marketing spend entered: the CAC below contains only free-tier inference, so LTV/CAC and payback look better than they would in practice. Enter your real monthly spend.")
    r1, r2, r3 = st.columns(3)
    r1.metric("New paying customers", f"{cl['n_new']:,.0f}", f"{cl['n_pro']:,.0f} Pro, {cl['n_team']:,.1f} Team / SMB deals", delta_color="off",
              help="Pro conversions plus new team deals over the horizon (existing customers at the start are excluded).")
    r2.metric("Acquisition cost over horizon", f"${cl['spend'] + cl['free_cost']:,.0f}", f"${cl['spend']:,.0f} S&M + ${cl['free_cost']:,.0f} free-tier inference", delta_color="off")
    r3.metric("Blended CAC", "n/a" if pd.isna(cl["cac"]) else f"${cl['cac']:,.2f}",
              "n/a" if pd.isna(cl["cac_ex_free"]) else f"${cl['cac_ex_free']:,.2f} excluding free-tier cost", delta_color="off")
    q1, q2, q3 = st.columns(3)
    _neg = pd.isna(cl["margin"]) or cl["margin"] <= 0
    q1.metric("Blended LTV", "n/m (margin <= 0)" if _neg else f"${cl['ltv']:,.2f}", f"Pro ${cl['ltv_pro']:,.0f} / Team ${cl['ltv_team']:,.0f}" if not _neg else "", delta_color="off",
              help="Monthly price x margin x expected lifetime, averaged over new Pro and team customers. Lifetime is capped at 60 months; not discounted.")
    q2.metric("LTV / CAC", "n/m" if (_neg or pd.isna(cl["ltv_cac"])) else f"{cl['ltv_cac']:.1f}x", "Common rule of thumb: 3x or more", delta_color="off")
    q3.metric("CAC payback", "n/a" if (_neg or pd.isna(cl["payback"])) else f"{cl['payback']:.1f} months", "Common rule of thumb: under 12", delta_color="off")
    _marg_txt = "n/a" if pd.isna(cl["margin"]) else f"{cl['margin'] * 100:.1f}%"
    st.caption(f"Method (simplified): margin used = paid-customer margin after your other COGS lines = {_marg_txt}, "
               f"so free-tier cost is counted once, as acquisition cost, and never also in margin. Pro lifetime = 1 / monthly churn ({cl['life_pro']:.0f} months); "
               f"team lifetime = term / (1 - renewal rate) ({cl['life_team']:.0f} months, capped at 60). Team deals are sales-led and usually cost far more to win than Pro conversions, "
               "so a blended CAC can hide a large gap; treat this as a directional planning view, not an investor-grade calculation.")

# ---------- tab 9: save, compare & export ----------
with tab9:
    st.subheader("🧪 Save, Compare & Export")
    if "saved_scenarios" not in st.session_state:
        st.session_state["saved_scenarios"] = {}
    saved = st.session_state["saved_scenarios"]
    _snap_now = scenario_snapshot(df, {**P, "_preset": founder_stage}, tot)
    sc1, sc2 = st.columns([3, 1])
    scn_name = sc1.text_input("Scenario name", value="", placeholder="e.g. Free tier on a cheaper model")
    if sc2.button("💾 Save current as scenario"):
        _nm = clean_text(scn_name, 60) or f"Scenario {len(saved) + 1}"
        if _nm in ("Metric", "Current (unsaved)"):
            _nm += " (saved)"
        while _nm in saved:
            _nm += " (2)"
        if len(saved) >= 8:
            st.warning("You can keep up to 8 saved scenarios. Clear some to save another.")
        else:
            saved[_nm] = {"snapshot": _snap_now, "assumptions": json.loads(json.dumps(P, default=str))}
            st.success(f"Saved '{_nm}'.")

    def _fmt_snap(key, v):
        if isinstance(v, str):
            return v
        if pd.isna(v):
            return "n/a"
        return f"${v:,.0f}" if key.endswith("($)") else (f"{v:.1f}%" if key.endswith("(%)") else str(v))

    compare_df = pd.DataFrame({"Metric": list(_snap_now)})
    compare_df["Current (unsaved)"] = [_fmt_snap(k, v) for k, v in _snap_now.items()]
    for _nm, _rec in saved.items():
        compare_df[_nm] = [_fmt_snap(k, _rec["snapshot"].get(k, np.nan)) for k in _snap_now]
    show_df(compare_df, hide_index=True)
    if saved and st.button("🗑️ Clear all saved scenarios"):
        saved.clear()
        st.rerun()
    st.caption("Saved scenarios live in this browser session only and are lost when you refresh or close the page. Use the exports below to keep a permanent record. "
               "Loading a saved scenario back into the sidebar is not supported; the exported assumptions list what to re-enter.")

    st.markdown("#### Export")
    _audit_df = pd.DataFrame(audit, columns=["Model Parameter", "Current Value", "Verify Against (System of Record)", "Status"])
    _export = {"app": APP_NAME, "app_version": APP_VERSION, "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "starting_profile": founder_stage,
               "assumptions": json.loads(json.dumps(P, default=str)), "assumptions_register": _audit_df.to_dict("records"),
               "results": {k: (None if (not isinstance(v, str) and pd.isna(v)) else v) for k, v in _snap_now.items()}}
    ex1, ex2, ex3 = st.columns(3)
    ex1.download_button("⬇️ Assumptions & results (JSON)", data=json.dumps(_export, indent=2, default=str).encode("utf-8"),
                        file_name="deal_desk_assumptions.json", mime="application/json")
    ex2.download_button("⬇️ Margin bridge (CSV)", data=safe_frame(bridge).to_csv(index=False).encode("utf-8"), file_name="deal_desk_margin_bridge.csv", mime="text/csv")
    try:
        _sheets = {"Assumptions": _audit_df, "Monthly detail": df, "Margin bridge": bridge, "Traffic by type": traffic, "Scenario compare": compare_df}
        if "torn" in globals() and not torn.empty:
            _sheets["Sensitivity"] = torn
        _bio = io.BytesIO()
        with pd.ExcelWriter(_bio, engine="openpyxl") as _xw:
            for _sn, _fr in _sheets.items():
                safe_frame(_fr).to_excel(_xw, sheet_name=_sn[:31], index=False)
        ex3.download_button("⬇️ Full workbook (Excel)", data=_bio.getvalue(), file_name="deal_desk_model.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        ex3.caption("Excel export needs the `openpyxl` package. Add it to requirements.txt to enable it.")

# ---------- tab: deal desk ----------
with tab_deal:
    st.subheader("🤝 Deal Desk: does this customer deal clear the margin floor?")
    st.caption("Prices one customer deal with the same per-prompt AI cost, token prices, caching, batch and AI cost path as the rest of the model. "
               "Token prices are illustrative placeholders, so treat the verdict as a planning view and confirm against your vendor rate card.")
    dd = dict(DEAL_DEFAULTS)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Commercial**")
        dd["seats"] = int(st.number_input("Seats", min_value=0, value=DEAL_DEFAULTS["seats"], step=10, key="dd_seats"))
        dd["list_price"] = float(st.number_input("List price per seat per month ($)", min_value=0.0, value=DEAL_DEFAULTS["list_price"], step=1.0, key="dd_list"))
        dd["discount"] = st.slider("Discount (%)", 0, 100, int(DEAL_DEFAULTS["discount"] * 100), 1, key="dd_disc") / 100
        dd["term"] = int(st.number_input("Contract term (months)", min_value=1, max_value=60, value=DEAL_DEFAULTS["term"], step=1, key="dd_term"))
        dd["billing"] = st.selectbox("Billing", ["Monthly", "Annual upfront"], key="dd_bill")
        dd["impl_fee"] = float(st.number_input("Implementation fee ($)", min_value=0.0, value=0.0, step=1000.0, key="dd_ifee"))
    with c2:
        st.markdown("**Usage**")
        dd["pps"] = float(st.number_input("Prompts per seat per month (typical user)", min_value=0.0, value=DEAL_DEFAULTS["pps"], step=50.0, key="dd_pps"))
        dd["heavy_share"] = st.slider("Heavy-user share of seats (%)", 0, 100, int(DEAL_DEFAULTS["heavy_share"] * 100), 1, key="dd_heavy",
                                      help=f"Heavy users prompt {h_mult_used:.1f}x the typical rate (the heavy-user multiplier from the main model).") / 100
        dd["usage_growth"] = st.slider("Monthly usage growth (%)", 0.0, 15.0, DEAL_DEFAULTS["usage_growth"] * 100, 0.5, key="dd_grow") / 100
        dd["start_month"] = int(st.number_input("Deal starts in forecast month", min_value=1, max_value=36, value=1, step=1, key="dd_start",
                                                help="Sets token-length growth and where the deal sits on the AI cost path."))
        _own_mix = st.checkbox("Use a different frontier-model share for this deal", value=False, key="dd_mixon")
        if _own_mix:
            dd["front_mix"] = st.slider("Frontier-model share for this deal (%)", 0, 100, int(P["front_mix"] * 100), 1, key="dd_mix") / 100
    with c3:
        st.markdown("**Cost to serve & overage**")
        dd["dedicated_infra"] = float(st.number_input("Dedicated infrastructure ($ / month)", min_value=0.0, value=0.0, step=100.0, key="dd_ded"))
        dd["support"] = float(st.number_input("Support / success cost ($ / month)", min_value=0.0, value=0.0, step=100.0, key="dd_sup"))
        dd["impl_cost"] = float(st.number_input("Implementation delivery cost ($, month 1)", min_value=0.0, value=0.0, step=1000.0, key="dd_icost"))
        dd["alloc_shared_infra"] = st.checkbox("Allocate shared fixed infrastructure by prompt share", value=True, key="dd_alloc",
                                               help="Off: the deal carries no share of the shared platform cost (flatters the deal).")
        _ov = st.checkbox("Usage-based overage and cap", value=False, key="dd_ovon")
        if _ov:
            dd["included_pps"] = float(st.number_input("Included prompts per seat per month", min_value=0.0, value=500.0, step=50.0, key="dd_inc"))
            dd["overage_per_1k"] = float(st.number_input("Overage price per 1,000 prompts ($)", min_value=0.0, value=20.0, step=1.0, key="dd_ovp"))
            dd["cap_pps"] = float(st.number_input("Hard cap on prompts per seat (0 = none)", min_value=0.0, value=0.0, step=100.0, key="dd_cap"))
    pc1, pc2 = st.columns(2)
    floor_pct = float(pc1.number_input("Margin floor (%)", min_value=0.0, max_value=100.0, value=float(margin_target), step=1.0, key="dd_floor",
                                       help="Defaults to the sidebar target gross margin."))
    band_pp = float(pc2.number_input("'Approve with conditions' band above the floor (percentage points)", min_value=0.0, max_value=30.0, value=5.0, step=1.0, key="dd_band"))

    _derr = validate_deal(dd)
    if _derr:
        st.error(_derr)
    else:
        dpnl = deal_pnl(P, dd, df, h_mult_used)
        dsum = deal_summary(dpnl)
        verdict = deal_verdict(dsum["margin"], floor_pct, band_pp)
        icon = {"Approve": "✅", "Approve with conditions": "🟡", "Escalate": "🔴"}[verdict]
        st.markdown(f"### {icon} {verdict}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Deal gross margin (whole term)", "n/a" if pd.isna(dsum["margin"]) else f"{dsum['margin']:.1f}%", f"floor {floor_pct:.0f}%", delta_color="off")
        m2.metric("Revenue / gross profit", f"${dsum['revenue']:,.0f}", f"${dsum['gp']:,.0f} gross profit", delta_color="off")
        m3.metric("Lowest cumulative cash", f"${dsum['min_cash']:,.0f}", "cash positive from month " + (str(dsum["cash_ok_from"]) if dsum["cash_ok_from"] else "n/a (not within term)"), delta_color="off")
        m4.metric("Main model paid-customer GM", "n/a" if pd.isna(tot["paid_gm"]) else f"{tot['paid_gm']:.1f}%",
                  ("deal dilutes the plan" if (not pd.isna(dsum["margin"]) and not pd.isna(tot["paid_gm"]) and dsum["margin"] < tot["paid_gm"]) else "deal is at or above plan"), delta_color="off")

        _cost_parts = {"Payment fees": dpnl["Payment fees ($)"].sum(), "AI inference": dpnl["AI inference ($)"].sum(), "Vector DB": dpnl["Vector DB ($)"].sum(),
                       "Shared infra": dpnl["Allocated shared infra ($)"].sum(), "Dedicated infra": dpnl["Dedicated infra ($)"].sum(),
                       "Support": dpnl["Support ($)"].sum(), "Implementation delivery": dpnl["Implementation delivery ($)"].sum()}
        _big = max(_cost_parts, key=_cost_parts.get)
        st.caption(f"Biggest cost driver: {_big} ({_cost_parts[_big] / dsum['cogs'] * 100:.0f}% of this deal's COGS)." if dsum["cogs"] > 0 else "No cost modeled for this deal.")
        if dsum["blocked"] > 0:
            st.warning(f"The hard cap blocks {dsum['blocked']:,.0f} requested prompts over the term. Check the customer accepts the cap.")
        if dsum["revenue"] <= 0:
            st.warning("This deal has no revenue, so margin is not meaningful.")
        elif dsum["gp"] < 0:
            st.warning("Gross profit is negative: every month of this deal loses money at these inputs.")
        if dd["start_month"] + dd["term"] - 1 > len(df):
            st.info(f"The deal runs past forecast month {len(df)}. Month-{len(df)} fixed infrastructure is reused after that; token growth and the AI cost path continue.")

        _be = deal_breakevens(P, dd, df, h_mult_used, floor_pct)
        st.markdown("#### Break-evens at the margin floor")
        b1, b2, b3 = st.columns(3)
        _mu, _su = _be["max_pps"]
        b1.metric("Max typical prompts per seat", "none (fails even at zero usage)" if _su == "none" else ("no limit found" if _su == "unbounded" else f"{_mu:,.0f}"),
                  f"asked: {dd['pps']:,.0f}", delta_color="off", help="Highest typical-user usage at which the whole-term margin still meets the floor, holding price and everything else.")
        _mp, _sp = _be["min_net_price"]
        b2.metric("Min net price per seat / month", "not reachable" if _sp == "none" else ("any price" if _sp == "free" else f"${_mp:,.2f}"),
                  f"quoted: ${dd['list_price'] * (1 - dd['discount']):,.2f}", delta_color="off", help="Lowest net price (after discount) at the expected usage that meets the floor.")
        _md, _sd = _be["max_discount"]
        b3.metric("Max discount off list", "n/a" if _md is None else f"{_md * 100:.0f}%", f"quoted: {dd['discount'] * 100:.0f}%", delta_color="off")

        st.markdown("#### What if the customer uses it more than they said?")
        _sens = [("As quoted", dd, P), ("Usage 1.5x", {**dd, "pps": dd["pps"] * 1.5}, P), ("Usage 2x", {**dd, "pps": dd["pps"] * 2}, P),
                 ("Frontier share +30 pts", {**dd, "front_mix": min(1.0, (P["front_mix"] if dd["front_mix"] is None else dd["front_mix"]) + 0.30)}, P),
                 ("Vendor prices +20%", dd, {**P, "price_mult": P["price_mult"] * 1.2})]
        _srows = []
        for _lab, _dd2, _p2 in _sens:
            _s2 = deal_summary(deal_pnl(_p2, _dd2, df, h_mult_used))
            _srows.append({"Case": _lab, "Gross margin (%)": _s2["margin"], "Gross profit ($)": _s2["gp"], "Verdict": deal_verdict(_s2["margin"], floor_pct, band_pp)})
        show_df(pd.DataFrame(_srows).style.format({"Gross margin (%)": "{:.1f}", "Gross profit ($)": "${:,.0f}"}, na_rep="n/a"), hide_index=True)

        figd = make_subplots(specs=[[{"secondary_y": True}]])
        figd.add_trace(go.Bar(x=dpnl["Deal month"], y=dpnl["Revenue ($)"], name="Revenue ($)", marker_color=tone("#3182ce", "#63b3ed")), secondary_y=False)
        figd.add_trace(go.Bar(x=dpnl["Deal month"], y=dpnl["Total COGS ($)"], name="COGS ($)", marker_color=tone("#e53e3e", "#fc8181")), secondary_y=False)
        figd.add_trace(go.Scatter(x=dpnl["Deal month"], y=dpnl["Gross margin (%)"], name="Gross margin (%)", mode="lines+markers",
                                  line=dict(color=tone("#2d3748", "#e2e8f0"), width=2)), secondary_y=True)
        figd.add_trace(go.Scatter(x=dpnl["Deal month"], y=[floor_pct] * len(dpnl), name="Margin floor (%)", mode="lines",
                                  line=dict(color=tone("#805ad5", "#b794f4"), width=2, dash="dash")), secondary_y=True)
        figd.update_layout(barmode="group", **CHART_KW, height=330, margin=dict(l=30, r=40, t=10, b=20), legend=dict(orientation="h", y=1.12))
        figd.update_yaxes(title_text="$", secondary_y=False)
        figd.update_yaxes(title_text="Gross margin (%)", secondary_y=True)
        show_chart(figd)
        with st.expander("Deal P&L and cash by month"):
            show_df(safe_frame(dpnl).style.format("{:,.2f}", subset=[c for c in dpnl.columns if c != "Deal month"], na_rep="n/a"), hide_index=True)
        _dsum_json = {"app": APP_NAME, "app_version": APP_VERSION, "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "deal_inputs": dd, "margin_floor_pct": floor_pct, "verdict": verdict,
                      "results": {k: (None if (not isinstance(v, str) and v is not None and pd.isna(v)) else v) for k, v in dsum.items()},
                      "break_evens": {k: {"value": v[0], "status": v[1]} for k, v in _be.items()},
                      "model_assumptions": {"price_mult": P["price_mult"], "front_mix": P["front_mix"], "cache_rate": P["cache_rate"],
                                            "ai_cost_path": ai_path_text(P, H), "token_prices": "illustrative placeholders, unverified"}}
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Deal summary (JSON)", data=json.dumps(_dsum_json, indent=2, default=str).encode("utf-8"), file_name="deal_desk_summary.json", mime="application/json")
        d2.download_button("⬇️ Deal P&L (CSV)", data=safe_frame(dpnl).to_csv(index=False).encode("utf-8"), file_name="deal_desk_pnl.csv", mime="text/csv")

# ---------- tab 10: feedback ----------
with tab10:
    st.subheader("💬 Feedback")
    st.markdown(FEEDBACK_WELCOME)
    _fb = feedback_config()
    _fallbacks = []
    if _fb["form_url"]:
        _fallbacks.append(f"[open the feedback form]({_fb['form_url']})")
    if _fb["email"]:
        _fallbacks.append(f"[email me](mailto:{_fb['email']})")
    if not _fb["webhook_url"]:
        st.info("The in-app feedback form isn't connected yet." + (f" In the meantime you can {' or '.join(_fallbacks)}." if _fallbacks else ""))
        with st.expander("For the app owner: connect this form"):
            st.markdown("Add this to `.streamlit/secrets.toml` (or the app's Secrets box on Streamlit Community Cloud). The webhook URL stays private and is never shown in the app.")
            st.code('[feedback]\nwebhook_url = "https://script.google.com/macros/s/XXXX/exec"   # or a Formspree / Zapier / Make URL\n'
                    'form_url = "https://forms.gle/XXXX"   # optional fallback link\nemail = "you@example.com"   # optional fallback', language="toml")
            st.markdown("**Free option: a Google Sheet.** Create a sheet with a tab named `Feedback`, open Extensions > Apps Script, paste the example below, "
                        "then Deploy > New deployment > Web app (execute as you, access: anyone) and paste the web app URL as `webhook_url`.")
            st.code('function doPost(e) {\n  var d = JSON.parse(e.postData.contents);\n  SpreadsheetApp.getActiveSpreadsheet().getSheetByName("Feedback")\n'
                    '    .appendRow([new Date(), d.category, d.message, d.email, d.app_version, JSON.stringify(d.settings || {})]);\n'
                    '  return ContentService.createTextOutput(JSON.stringify({ok: true})).setMimeType(ContentService.MimeType.JSON);\n}', language="javascript")
    else:
        st.session_state.setdefault("fb_opened", time.time())
        with st.form("feedback_form", clear_on_submit=False):
            fb_category = st.radio("What would you like to share?", FEEDBACK_CATEGORIES, index=0)
            fb_message = st.text_area("Your message", max_chars=2000, height=150, placeholder="What happened, what you expected, or what you would like to see. Steps to reproduce help for bugs.")
            fb_email = st.text_input("Email (optional, only if you'd like a reply)", max_chars=200)
            fb_attach = st.checkbox("Attach my current model settings (no personal data) so a problem is easier to reproduce", value=True)
            st.caption("Your message is used only to improve this project. Please don't include confidential or personal financial details.")
            fb_submit = st.form_submit_button("Send feedback")
        if fb_submit:
            _err = validate_feedback(fb_category, fb_message, fb_email)
            if _err:
                st.error(_err)
            elif st.session_state.get("fb_count", 0) >= 5:
                st.warning("You've sent several messages in this session, thank you! Please try again later.")
            elif time.time() - st.session_state["fb_opened"] < 2:
                st.success("Thank you for your feedback!")   # too fast to be a person: acknowledge, send nothing
            else:
                _settings = None
                if fb_attach:
                    _settings = {"starting_profile": founder_stage, "horizon_months": H, "advanced_mode": bool(advanced_mode), "free_tier_booked_to": P["free_treatment"],
                                 "fixed_infra_to_free_tier": P["infra_alloc"], "trial_share": P["trial_share"], "free_tier_frontier_mix": P["front_mix_free"],
                                 "internal_traffic_pct": P["internal_pct"], "ai_cost_path": ai_path_text(P, H), "other_cogs_lines": len(P["other_cogs"]),
                                 "reported_gm_pct": None if pd.isna(tot["gm"]) else round(tot["gm"], 1), "ending_arr": round(tot["arr"])}
                _payload = {"submitted_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "app": APP_NAME, "app_version": APP_VERSION,
                            "category": fb_category, "message": safe_cell(clean_text(fb_message, 2000)), "email": safe_cell(clean_text(fb_email, 200)), "settings": _settings}
                _ok, _msg = post_feedback(_fb["webhook_url"], _payload)
                if _ok:
                    st.session_state["fb_count"] = st.session_state.get("fb_count", 0) + 1
                    st.success("Thank you! Your feedback was sent.")
                else:
                    st.error(f"{_msg} Your message was not sent." + (f" You can also {' or '.join(_fallbacks)}." if _fallbacks else ""))

# ==========================================
# 5. FOOTER, CHANGE LOG & LIMITATIONS
# ==========================================
st.markdown("---")
SHOW_CHANGE_LOG = False
CHANGE_LOG_MD = """
**Presentation update (build 5; calculation engine unchanged)**
1. **Founder-led opening.** Introduces Alex and frames the journey around whether growth can become profit.
2. **Business Chart illustration.** Adds an unDraw visual to the guided-story hero; the illustration is decorative and does not feed any calculations.
3. **Clearer chapter transitions.** Chapter 0 explains why three margin views matter; Chapter 1 starts with the cost of one prompt; the reference tabs are framed as the working model.

**UX build 8 changes**
1. **Equal-width stage buttons.** Explicitly made each stage button fill its own equal-width column, with centered descriptions beneath each button. This is presentation-only; calculations are unchanged.

**UX design update (presentation only; calculation engine unchanged)**
1. Removed internal build/version text from the public hero and renamed the sidebar timing section to “Forecast setup.”
2. Compact key-input layout: the five primary growth/usage assumptions remain visible; target and cost-treatment settings are grouped under “More assumptions.”
4. UX revision: clean white canvas, DM Sans interface typography, compact paired sidebar inputs, and target/cost-treatment controls moved into the sidebar More assumptions expander. Forecast calculations intentionally unchanged.
2. Replaced the externally hosted hero image with a self-contained illustration to avoid broken-image dependencies.
3. Added selective editorial italics, equal-height margin cards, and contextual commentary explaining the paid-customer vs reported margin difference.

**UX updates**
1. **Restored full-width startup-stage buttons and preserved story copy.** Kept the three profile choices aligned, restored “Step 1 · Start here,” and gave each profile a persistent explanatory story rather than a short caption.

**v2.9 changes**
1. **Guided story view (default) and Analyst view.** The Guided story opens with a hero, Chapter 0 (pick a profile, see paid-customer, reported and fully loaded margin side by side) and Chapter 1 (what one prompt costs, with the prompts-per-day control). All remaining tabs sit under "Go deeper". Analyst view is the previous layout. Both views use the same engine, so the numbers are identical.
2. **New visual theme** (light grey page, near-white cards, deep navy and one strong blue; matching dark mode). Status pills use separate green, amber and red.
3. **Each setting has one home.** In the guided view the profile picker and the prompts-per-day slider live in the chapters, not the sidebar; in Analyst view they stay in the sidebar.

**v2.8 changes**
1. **Deal Desk tab.** Prices one customer deal (seats, discount, term, billing, usage, optional overage and cap, dedicated cost) with the same per-prompt AI cost as the main model, and returns Approve / Approve with conditions / Escalate against a margin floor.
2. **Break-evens** for maximum prompts per seat, minimum net price and maximum discount, a usage / model-mix / vendor-price stress table, a monthly deal P&L and cash view, and JSON / CSV export.
3. **One AI cost function.** The simulation and the Deal Desk now share `api_cost_for`, so they cannot drift apart. Main-model results are unchanged.

**v2.7 changes**
1. **Traffic mix.** Free-tier traffic can be split into trials and freemium (sidebar 3) and can be routed to a cheaper model mix; internal / eval traffic can be modeled and booked to R&D / G&A or COGS (Advanced > F). A traffic-by-type table shows paid vs trial vs freemium vs internal.
2. **Sensitivity and goal seek.** A tornado shows which assumptions move your chosen metric most; goal seek solves for the lever value that hits a target margin; a usage-curve chart and sanity warnings flag implausible inputs.
3. **CAC, LTV and payback** count free-tier inference as acquisition cost (once), with a simplified, directional method.
4. **Save, compare and export.** Save scenarios in-session, compare them side by side, and export JSON, CSV or an Excel workbook. Text beginning with = + - @ is neutralised so spreadsheets never run it as a formula.
5. **In-app feedback form** (Feedback tab). It sends to a destination you configure in Streamlit secrets; until then it shows a fallback.
6. **Fix:** the per-account usage ceiling now prices paying-customer traffic only, so a cheaper free-tier model no longer understates paid cost per prompt.

**v2.6 changes**
1. **Free / trial inference cost can be booked to COGS or to sales & marketing** (sidebar 3). The default is COGS, so headline numbers are unchanged. Free-tier cost is now its own sub-line, and paying customers' margin never carries it.
2. **Three margins side by side:** paid-customer gross margin, gross margin as reported (after your COGS / S&M choice) and fully loaded gross margin, with a margin bridge on the Revenue & Margin tab.
3. **Other COGS lines for a fully loaded margin.** An editable table above the KPIs takes $ / month or % of revenue lines with start month, end month and monthly growth. Nothing is added by default.
4. **AI cost path (sidebar 3).** A MoM reduction in AI cost per prompt from a chosen month (with a cumulative cap) and a one-time vendor price change (cut or increase). Defaults are off, so prices stay flat unless you set it.
5. **Per-account cost fixed.** "COGS per paid account" now uses paid-customer COGS only; the version that included free-tier cost is shown as a memo. Blended cost per 1,000 prompts uses total direct cost.
6. **Fixed infrastructure** now stays with paid customers by default (Advanced > E lets you split it by prompt share, the v2.5 behaviour). Variable free-tier cost still follows prompt share. This changes the Value & Pricing tab's free-tier figures versus v2.5; headline gross margin is unchanged.
7. **Dark mode.** Colours now follow the active Streamlit theme (or the operating-system setting when the theme is not reported), and charts switch to a dark template.
8. **New integrity checks** cover the cost split, free-tier booking, fully loaded reconciliation, margin ordering and the AI cost path.

**Known limitations**
- **Token prices are illustrative placeholders.** They are flat unless you set an AI cost path; the path is a planning assumption, not a vendor commitment, and applies to AI inference (API) cost only, not vector DB or hosting. Scenarios 8-9 apply one flat % change to every month.
- **Input tokens per prompt grow monthly,** which pushes cost per prompt up unless offset by an efficiency path.
- **Cache and batch discounts** are vendor-dependent and assumed to stay available at the configured levels.
- **Other COGS lines are display-only:** they change the fully loaded margin but not cash payable, credits or the usage ceiling.
- **Accounting treatment is a policy choice.** The COGS / S&M switch is a presentation of the same cost; confirm your policy with your accountant or auditor.
- **Margins exclude any cost you do not enter,** so modeled margin is higher than a fully loaded margin unless you add lines.

**Material changes from v1**
1. **Team / SMB engine rebuilt as cohorts.** Accounts, renewals, non-renewals, in-term churn and expansion are tracked per cohort.
2. **Expansion is no longer double counted.** Team / SMB expansion grows contract value, seats and prompts of existing accounts.
"""
if SHOW_CHANGE_LOG:
    with st.expander("Change log & limitations (v2.9)"):
        st.markdown(CHANGE_LOG_MD)
else:
    _hidden = html.escape(CHANGE_LOG_MD.strip()).replace("\n\n", "<br>").replace("\n", "<br>")
    st.markdown(f'<div id="change-log-and-limitations" aria-hidden="true" style="display:none">{_hidden}</div>', unsafe_allow_html=True)

# Always-visible disclaimers (independent of SHOW_CHANGE_LOG). Built from live inputs so they cannot quote stale values.
_in_mult = (1 + P["in_grow"]) ** max(0, H - 1)
st.markdown(f"""
<div class="assumption-flag"><b>Key assumptions & limitations</b>
<ul>
<li>{token_price_note(P, H)}</li>
<li><b>Token volume:</b> input tokens per prompt grow {P['in_grow'] * 100:.1f}% per month (x{_in_mult:.2f} by month {H}) and output tokens {P['out_grow'] * 100:.1f}% per month, against {'flat prices, so cost per prompt rises over the horizon' if ai_path_is_flat(P) else 'the AI cost path above, which offsets or adds to that growth'}.</li>
<li><b>Discounts:</b> caching ({P['cache_rate'] * 100:.0f}% of eligible input at {P['cache_ratio'] * 100:.0f}% of list price) and batch routing ({P['batch_share'] * 100:.0f}% of traffic at {P['batch_ratio'] * 100:.0f}% of list price) depend on vendor terms and are assumed to stay available throughout.</li>
<li><b>Price scenarios:</b> scenarios 8-9 apply one flat % price change to every month; for a gradual path use the AI cost path in the sidebar.</li>
<li><b>Cost treatment:</b> free / trial inference cost is booked to {TREAT_LABEL[P['free_treatment']]} (a policy choice; confirm with your accountant or auditor). Paying customers' margin never carries it.</li>
<li><b>Fully loaded margin:</b> {('includes ' + str(len(P['other_cogs'])) + ' user-defined COGS line(s); they do not affect cash timing or credits.') if P['other_cogs'] else 'no other COGS lines entered, so margins cover AI-direct costs only.'}</li>
<li><b>Presets:</b> founder presets are planning estimates, not benchmarks. Replace them with actuals from your billing system and LLM gateway logs.</li>
</ul></div>
""", unsafe_allow_html=True)

st.caption("© 2026. The AI Unit Economics Deal Desk. All rights reserved.")
