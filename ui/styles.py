"""
web_demo/ui/styles.py
---------------------
Self-contained RTL theme for the public demo. Same visual identity as the
desktop product (navy / cream / gold) but independent from the desktop UI.
"""

from __future__ import annotations

import streamlit as st

NAVY = "#0f2038"
NAVY_SOFT = "#16304f"
CREAM = "#f5f1e6"
CREAM_CARD = "#fbf9f2"
GOLD = "#c9a24b"
GREEN = "#2e7d5b"
RED = "#b3403a"
AMBER = "#c9862b"
INK = "#1c2733"
MUTED = "#6b7684"

_CSS = f"""
<style>
:root {{
  --navy: {NAVY}; --navy-soft: {NAVY_SOFT}; --cream: {CREAM};
  --cream-card: {CREAM_CARD}; --gold: {GOLD}; --green: {GREEN};
  --red: {RED}; --amber: {AMBER}; --ink: {INK}; --muted: {MUTED};
}}

html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {{
  direction: rtl;
  font-family: "IBM Plex Sans Arabic", "Segoe UI", Tahoma, sans-serif;
}}
.stApp {{ background: var(--cream); color: var(--ink); }}

/* Sidebar */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, var(--navy) 0%, var(--navy-soft) 100%);
}}
[data-testid="stSidebar"] * {{ color: #eaf0f7 !important; }}
[data-testid="stSidebar"] .stButton > button {{
  background: transparent; border: 1px solid rgba(201,162,75,0.35);
  color: #eaf0f7 !important; text-align: right; border-radius: 10px;
  padding: 0.55rem 0.9rem; width: 100%;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
  border-color: var(--gold); background: rgba(201,162,75,0.12);
}}
.nav-active > button {{
  background: rgba(201,162,75,0.20) !important;
  border-color: var(--gold) !important; font-weight: 700;
}}

/* Brand */
.brand-title {{ font-size: 1.5rem; font-weight: 800; color: var(--gold) !important; margin: 0; }}
.brand-tag {{ font-size: 0.82rem; color: #cdd7e4 !important; margin-top: 0.2rem; line-height: 1.5; }}
.brand-logo {{
  width: 46px; height: 46px; border-radius: 12px; background: var(--gold);
  display:flex; align-items:center; justify-content:center;
  color: var(--navy) !important; font-weight: 900; font-size: 1.3rem;
}}

/* Demo badge */
.demo-badge {{
  display:inline-block; background: rgba(201,162,75,0.18); color: var(--gold) !important;
  border: 1px solid rgba(201,162,75,0.4); border-radius: 999px;
  padding: 0.15rem 0.7rem; font-size: 0.75rem; font-weight: 700; margin-top: 0.5rem;
}}

/* Privacy box */
.privacy-box {{
  border: 1px solid rgba(201,162,75,0.3); border-radius: 12px;
  padding: 0.85rem 1rem; font-size: 0.86rem; line-height: 1.75; color: var(--ink);
  background: var(--cream-card); margin: 0.6rem 0 1rem 0;
}}
.privacy-box.side {{ color: #cdd7e4 !important; background: rgba(0,0,0,0.15); font-size: 0.8rem; }}

/* Stat cards */
.stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.9rem; margin-bottom: 0.5rem; }}
.stat-card {{
  background: var(--cream-card); border: 1px solid rgba(15,32,56,0.10);
  border-radius: 14px; padding: 1rem 1.1rem; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}}
.stat-value {{ font-size: 1.7rem; font-weight: 800; color: var(--navy); }}
.stat-label {{ font-size: 0.85rem; color: var(--muted); margin-top: 0.2rem; }}

/* Document + source cards */
.doc-card {{
  background: var(--cream-card); border: 1px solid rgba(15,32,56,0.10);
  border-right: 4px solid var(--gold); border-radius: 14px;
  padding: 1rem 1.1rem; margin-bottom: 0.8rem;
}}
.doc-title {{ font-size: 1.05rem; font-weight: 700; color: var(--navy); }}
.doc-meta {{ font-size: 0.85rem; color: var(--muted); margin-top: 0.35rem; }}
.src-card {{
  background: var(--cream-card); border: 1px solid rgba(15,32,56,0.10);
  border-radius: 12px; padding: 0.8rem 1rem; margin-bottom: 0.6rem;
}}
.src-head {{ font-weight: 700; color: var(--navy); }}
.src-score {{ float: left; color: var(--gold); font-weight: 700; }}
/* Links out to a URL source's original page */
.src-link, .doc-link {{
  font-size: 0.82rem; font-weight: 600; color: var(--navy);
  text-decoration: underline; text-underline-offset: 3px;
}}
.doc-link {{ display: inline-block; margin-top: 0.4rem; }}

/* Badges */
.badge {{ display:inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.78rem; font-weight: 700; }}
.badge-ready {{ background: rgba(46,125,91,0.15); color: var(--green); }}
.badge-ocr {{ background: rgba(201,134,43,0.15); color: var(--amber); }}
.badge-error {{ background: rgba(179,64,58,0.15); color: var(--red); }}

/* Headings */
.page-title {{ font-size: 1.6rem; font-weight: 800; color: var(--navy); margin-bottom: 0.2rem; }}
.page-sub {{ color: var(--muted); margin-bottom: 1.1rem; }}
.hero {{
  background: linear-gradient(135deg, var(--navy) 0%, var(--navy-soft) 100%);
  color: #eaf0f7; border-radius: 18px; padding: 1.8rem 2rem; margin-bottom: 1.2rem;
}}
.hero h1 {{ color: var(--gold); margin: 0 0 0.3rem 0; font-size: 1.9rem; }}
.hero p {{ color: #cdd7e4; margin: 0; line-height: 1.8; }}

/* Streamlit Community Cloud header icons (GitHub / Edit / Star / Share).
   The host injects these via SET_TOOLBAR_ITEMS into the app header toolbar.
   Selectors target stable data-testid/class hooks from Streamlit 1.61+. */
[data-testid="stToolbarActions"],
[data-testid="stToolbarActionButton"],
.stToolbarActions,
.stToolbarActionButton {{
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
  max-height: 0 !important;
  overflow: hidden !important;
  pointer-events: none !important;
  opacity: 0 !important;
}}

/* Fallback: hide the entire top-right toolbar strip (not the app sidebar). */
[data-testid="stHeader"] [data-testid="stToolbar"] {{
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
  min-height: 0 !important;
  overflow: hidden !important;
  pointer-events: none !important;
}}

[data-testid="stHeader"] {{
  height: auto !important;
  min-height: 0 !important;
  background: transparent !important;
}}
</style>
"""


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
