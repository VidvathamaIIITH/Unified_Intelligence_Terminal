"""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║    ████████╗ █████╗      ████████╗██╗  ██╗███████╗                   ║
║    ╚══██╔══╝██╔══██╗     ╚══██╔══╝██║  ██║██╔════╝                   ║
║       ██║   ╚██████║        ██║   ███████║█████╗                     ║
║       ██║    ╚═══██║        ██║   ██╔══██║██╔══╝                     ║
║       ██║    █████╔╝        ██║   ██║  ██║███████╗                   ║
║       ╚═╝    ╚════╝         ╚═╝   ╚═╝  ╚═╝╚══════╝                   ║
║                                                                      ║
║              THE UNIFIED INTELLIGENCE TERMINAL                       ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

Main Streamlit application — Presentation-grade UI.
"""

import os
import time
import math
from datetime import datetime

import streamlit as st
import pandas as pd

# ─── Load .env if available ────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Local Modules ────────────────────────────────────────────────
from data_engine import (
    fetch_news_articles,
    fetch_tech_articles,
    fetch_finance_articles,
    fetch_all_articles,
    tag_finance_articles,
    get_feed_stats,
)
from intelligence import (
    enrich_news_articles,
    enrich_tech_articles,
    classify_batch,
    summarize_article,
    calculate_sentiment_score,
    synthesize_cross_domain,
    NEWS_CATEGORIES,
    TECH_KEYWORDS,
)

# ═══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="T9 — Unified Intelligence Terminal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
#  MASTER CSS — HYPER-POLISHED DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
    <style>
    /* ══════════════════════════════════════════════════════════════
       T9 DESIGN SYSTEM v2 — PRESENTATION GRADE
       ══════════════════════════════════════════════════════════════ */

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg-primary: #06080F;
        --bg-secondary: #0C1120;
        --bg-tertiary: #111827;
        --bg-card: rgba(17, 24, 39, 0.75);
        --bg-card-solid: #131B2E;
        --border-subtle: rgba(99, 179, 237, 0.06);
        --border-active: rgba(99, 179, 237, 0.2);
        --border-glow: rgba(56, 189, 248, 0.35);
        --text-primary: #F1F5F9;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --text-dim: #475569;
        --accent-sky: #38BDF8;
        --accent-cyan: #22D3EE;
        --accent-teal: #2DD4BF;
        --accent-emerald: #34D399;
        --accent-gold: #FBBF24;
        --accent-amber: #F59E0B;
        --accent-rose: #FB7185;
        --accent-red: #EF4444;
        --accent-violet: #A78BFA;
        --accent-purple: #C084FC;
        --accent-indigo: #818CF8;
        --gradient-brand: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        --gradient-warm: linear-gradient(135deg, #FBBF24 0%, #F59E0B 100%);
        --gradient-success: linear-gradient(135deg, #34D399 0%, #2DD4BF 100%);
        --gradient-danger: linear-gradient(135deg, #FB7185 0%, #EF4444 100%);
        --gradient-cool: linear-gradient(135deg, #38BDF8 0%, #22D3EE 100%);
        --gradient-glass: linear-gradient(135deg, rgba(17, 24, 39, 0.8), rgba(6, 8, 15, 0.9));
        --glass-bg: rgba(17, 24, 39, 0.6);
        --glass-border: rgba(99, 179, 237, 0.08);
        --shadow-sm: 0 2px 8px rgba(0,0,0,0.2);
        --shadow-md: 0 4px 20px rgba(0,0,0,0.3);
        --shadow-lg: 0 8px 40px rgba(0,0,0,0.4);
        --shadow-glow-sky: 0 0 30px rgba(56, 189, 248, 0.08);
        --shadow-glow-emerald: 0 0 30px rgba(52, 211, 153, 0.08);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 24px;
        --ease: cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ─── Reset & Global ───────────────────────────────────────── */
    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary);
    }
    .stApp > header { background: transparent !important; }
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
    .stDeployButton { display: none !important; }
    .block-container { padding-top: 2rem !important; max-width: 1400px !important; }

    /* ─── Sidebar ──────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0C1120 0%, #060A14 100%) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: var(--text-secondary) !important;
        font-family: 'Inter', sans-serif !important;
    }
    section[data-testid="stSidebar"] .stRadio > label {
        display: none !important;
    }
    section[data-testid="stSidebar"] .stRadio > div {
        gap: 2px !important;
    }
    section[data-testid="stSidebar"] .stRadio > div > label {
        background: transparent !important;
        border: 1px solid transparent !important;
        border-radius: var(--radius-md) !important;
        padding: 12px 16px !important;
        margin: 0 !important;
        transition: all 0.25s var(--ease) !important;
        cursor: pointer !important;
    }
    section[data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(56, 189, 248, 0.04) !important;
        border-color: var(--border-subtle) !important;
    }
    section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
    section[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
        background: rgba(56, 189, 248, 0.08) !important;
        border-color: rgba(56, 189, 248, 0.2) !important;
    }

    /* ─── Scrollbar ────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(56, 189, 248, 0.12); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(56, 189, 248, 0.25); }

    /* ─── Hero / Page Header ───────────────────────────────────── */
    .t9-hero {
        position: relative;
        padding: 40px 48px;
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.04) 0%, rgba(129, 140, 248, 0.04) 50%, rgba(192, 132, 252, 0.03) 100%);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-xl);
        margin-bottom: 32px;
        overflow: hidden;
    }
    .t9-hero::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(ellipse at 20% 50%, rgba(56, 189, 248, 0.06) 0%, transparent 60%),
                    radial-gradient(ellipse at 80% 50%, rgba(192, 132, 252, 0.05) 0%, transparent 60%);
        pointer-events: none;
    }
    .t9-hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: var(--accent-sky);
        margin-bottom: 16px;
        position: relative;
    }
    .t9-hero-title {
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.15;
        margin-bottom: 10px;
        position: relative;
    }
    .t9-hero-subtitle {
        font-size: 1rem;
        color: var(--text-muted);
        line-height: 1.6;
        max-width: 640px;
        position: relative;
    }

    /* ─── Live Indicator ───────────────────────────────────────── */
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        background: rgba(52, 211, 153, 0.08);
        border: 1px solid rgba(52, 211, 153, 0.15);
        border-radius: 50px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--accent-emerald);
        letter-spacing: 1px;
    }
    .live-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--accent-emerald);
        animation: livePulse 2s ease-in-out infinite;
    }
    @keyframes livePulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.5); }
        50% { opacity: 0.6; box-shadow: 0 0 0 6px rgba(52, 211, 153, 0); }
    }

    /* ─── Stat Card ────────────────────────────────────────────── */
    .stat-card {
        background: var(--gradient-glass);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-lg);
        padding: 24px;
        text-align: center;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.3s var(--ease);
        position: relative;
        overflow: hidden;
    }
    .stat-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        border-radius: 2px 2px 0 0;
        opacity: 0;
        transition: opacity 0.3s var(--ease);
    }
    .stat-card:hover {
        border-color: var(--border-active);
        transform: translateY(-2px);
        box-shadow: var(--shadow-glow-sky);
    }
    .stat-card:hover::after { opacity: 1; }
    .stat-card-sky::after { background: var(--gradient-cool); }
    .stat-card-emerald::after { background: var(--gradient-success); }
    .stat-card-gold::after { background: var(--gradient-warm); }
    .stat-card-violet::after { background: linear-gradient(135deg, var(--accent-violet), var(--accent-purple)); }
    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.4rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .stat-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* ─── Section Header ───────────────────────────────────────── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--border-subtle);
    }
    .section-icon {
        width: 36px; height: 36px;
        border-radius: var(--radius-sm);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
    }
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    .section-badge {
        margin-left: auto;
        padding: 4px 12px;
        border-radius: 50px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 1px;
    }

    /* ─── Article Card ─────────────────────────────────────────── */
    .article-card {
        background: var(--bg-card);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-lg);
        padding: 22px 26px;
        margin-bottom: 12px;
        backdrop-filter: blur(8px);
        transition: all 0.3s var(--ease);
        position: relative;
    }
    .article-card:hover {
        border-color: var(--border-active);
        box-shadow: var(--shadow-glow-sky);
        transform: translateY(-1px);
    }
    .article-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 10px;
    }
    .article-title {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1.5;
        margin: 0;
    }
    .article-title a {
        color: inherit !important;
        text-decoration: none !important;
        transition: color 0.2s var(--ease);
    }
    .article-title a:hover { color: var(--accent-sky) !important; }
    .article-meta {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid rgba(255,255,255,0.03);
    }
    .article-source {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 500;
        color: var(--text-dim);
        letter-spacing: 0.5px;
    }

    /* ─── Badge ────────────────────────────────────────────────── */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 12px;
        border-radius: 50px;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        text-transform: uppercase;
        white-space: nowrap;
        transition: all 0.2s var(--ease);
    }
    .badge-sky { background: rgba(56, 189, 248, 0.1); color: var(--accent-sky); border: 1px solid rgba(56, 189, 248, 0.2); }
    .badge-emerald { background: rgba(52, 211, 153, 0.1); color: var(--accent-emerald); border: 1px solid rgba(52, 211, 153, 0.2); }
    .badge-gold { background: rgba(251, 191, 36, 0.1); color: var(--accent-gold); border: 1px solid rgba(251, 191, 36, 0.2); }
    .badge-rose { background: rgba(251, 113, 133, 0.1); color: var(--accent-rose); border: 1px solid rgba(251, 113, 133, 0.2); }
    .badge-violet { background: rgba(167, 139, 250, 0.1); color: var(--accent-violet); border: 1px solid rgba(167, 139, 250, 0.2); }
    .badge-indigo { background: rgba(129, 140, 248, 0.1); color: var(--accent-indigo); border: 1px solid rgba(129, 140, 248, 0.2); }
    .badge-teal { background: rgba(45, 212, 191, 0.1); color: var(--accent-teal); border: 1px solid rgba(45, 212, 191, 0.2); }
    .badge-amber { background: rgba(245, 158, 11, 0.1); color: var(--accent-amber); border: 1px solid rgba(245, 158, 11, 0.2); }
    .badge-red { background: rgba(239, 68, 68, 0.1); color: var(--accent-red); border: 1px solid rgba(239, 68, 68, 0.2); }
    .badge-purple { background: rgba(192, 132, 252, 0.1); color: var(--accent-purple); border: 1px solid rgba(192, 132, 252, 0.2); }
    .badge-cyan { background: rgba(34, 211, 238, 0.1); color: var(--accent-cyan); border: 1px solid rgba(34, 211, 238, 0.2); }
    .badge-default { background: rgba(148, 163, 184, 0.08); color: var(--text-muted); border: 1px solid rgba(148, 163, 184, 0.15); }

    /* ─── Ticker Badge ─────────────────────────────────────────── */
    .ticker-tag {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        background: rgba(56, 189, 248, 0.08);
        color: var(--accent-sky);
        border: 1px solid rgba(56, 189, 248, 0.18);
        margin: 2px 3px 2px 0;
    }

    /* ─── AI Summary ───────────────────────────────────────────── */
    .ai-summary {
        font-size: 0.88rem;
        color: var(--text-secondary);
        line-height: 1.7;
        padding: 16px 20px;
        background: rgba(56, 189, 248, 0.03);
        border-left: 3px solid var(--accent-sky);
        border-radius: 0 var(--radius-md) var(--radius-md) 0;
        margin-top: 14px;
    }
    .ai-summary-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        font-weight: 600;
        color: var(--accent-sky);
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    /* ─── Confidence Bar ───────────────────────────────────────── */
    .conf-bar {
        width: 100%;
        height: 3px;
        background: rgba(255, 255, 255, 0.04);
        border-radius: 2px;
        overflow: hidden;
        margin-top: 12px;
    }
    .conf-fill {
        height: 100%;
        border-radius: 2px;
        transition: width 0.8s var(--ease);
    }

    /* ─── Sentiment Ring ───────────────────────────────────────── */
    .sentiment-ring-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 16px 0;
    }
    .sentiment-ring-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1;
    }
    .sentiment-ring-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 8px;
    }

    /* ─── Synthesis Panel ──────────────────────────────────────── */
    .synthesis-panel {
        background: linear-gradient(160deg, rgba(17, 24, 39, 0.9), rgba(6, 8, 15, 0.95));
        border: 1px solid rgba(129, 140, 248, 0.1);
        border-radius: var(--radius-xl);
        padding: 36px;
        position: relative;
        overflow: hidden;
    }
    .synthesis-panel::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: var(--gradient-brand);
    }
    .synthesis-panel h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        margin-top: 28px !important;
        margin-bottom: 10px !important;
    }
    .synthesis-panel p, .synthesis-panel li {
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
        line-height: 1.75 !important;
    }
    .synthesis-panel strong {
        color: var(--accent-sky) !important;
    }

    /* ─── Sentiment Bars ───────────────────────────────────────── */
    .sentiment-bar-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 10px 0;
    }
    .sentiment-bar-label {
        font-size: 0.75rem;
        font-weight: 500;
        width: 80px;
        text-align: right;
        flex-shrink: 0;
    }
    .sentiment-bar-track {
        flex: 1;
        height: 6px;
        background: rgba(255,255,255,0.04);
        border-radius: 3px;
        overflow: hidden;
    }
    .sentiment-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 1s var(--ease);
    }
    .sentiment-bar-pct {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        width: 48px;
        text-align: right;
        flex-shrink: 0;
    }

    /* ─── Ticker Strip ─────────────────────────────────────────── */
    .ticker-strip {
        overflow: hidden;
        white-space: nowrap;
        padding: 14px 24px;
        background: rgba(12, 17, 32, 0.8);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-lg);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-muted);
        position: relative;
    }
    .ticker-strip::before, .ticker-strip::after {
        content: '';
        position: absolute;
        top: 0; bottom: 0;
        width: 60px;
        z-index: 2;
        pointer-events: none;
    }
    .ticker-strip::before { left: 0; background: linear-gradient(90deg, rgba(6,8,15,1) 0%, transparent 100%); }
    .ticker-strip::after { right: 0; background: linear-gradient(90deg, transparent 0%, rgba(6,8,15,1) 100%); }
    .ticker-scroll {
        display: inline-block;
        animation: tickerScroll 80s linear infinite;
    }
    .ticker-divider { color: rgba(56, 189, 248, 0.2); margin: 0 24px; }
    @keyframes tickerScroll {
        0% { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }

    /* ─── Divider ──────────────────────────────────────────────── */
    .t9-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, var(--glass-border) 50%, transparent 100%);
        margin: 28px 0;
    }

    /* ─── Streamlit Overrides ──────────────────────────────────── */
    .stSelectbox label, .stMultiSelect label, .stCheckbox label {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
    }
    .stExpander {
        background: var(--bg-card-solid) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--radius-md) !important;
    }
    div[data-testid="stExpander"] details summary {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }
    div[data-testid="stExpander"] details summary:hover {
        color: var(--accent-sky) !important;
    }

    /* ─── Sidebar Status Styles ────────────────────────────────── */
    .sidebar-logo {
        text-align: center;
        padding: 24px 0 16px 0;
    }
    .sidebar-logo-text {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        background: var(--gradient-brand);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sidebar-logo-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.58rem;
        color: var(--text-dim);
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 4px;
    }
    .sidebar-section-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        font-weight: 600;
        color: var(--text-dim);
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--border-subtle);
    }
    .sidebar-stat {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 6px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-secondary);
    }
    .sidebar-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .sidebar-stat-value {
        margin-left: auto;
        font-weight: 600;
        color: var(--text-primary);
    }
    .sidebar-footer {
        font-size: 0.6rem;
        color: var(--text-dim);
        text-align: center;
        padding: 16px 0;
        border-top: 1px solid var(--border-subtle);
        line-height: 1.8;
    }

    /* ─── Button Override ──────────────────────────────────────── */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--border-active) !important;
        background: rgba(56, 189, 248, 0.06) !important;
        color: var(--accent-sky) !important;
        transition: all 0.25s var(--ease) !important;
        font-size: 0.82rem !important;
    }
    .stButton > button:hover {
        background: rgba(56, 189, 248, 0.12) !important;
        border-color: var(--accent-sky) !important;
        box-shadow: var(--shadow-glow-sky) !important;
    }

    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  UI HELPER COMPONENTS
# ═══════════════════════════════════════════════════════════════════

CATEGORY_BADGE_MAP = {
    "politics": "badge-violet",
    "sports": "badge-emerald",
    "business": "badge-gold",
    "technology": "badge-sky",
    "entertainment": "badge-amber",
    "science": "badge-cyan",
    "health": "badge-rose",
    "education": "badge-teal",
    "environment": "badge-emerald",
    "international affairs": "badge-indigo",
    "crime & law": "badge-red",
    "artificial intelligence": "badge-violet",
    "funding & investment": "badge-gold",
    "semiconductors & chips": "badge-sky",
    "startups & entrepreneurship": "badge-emerald",
    "cloud computing": "badge-indigo",
    "cybersecurity": "badge-rose",
    "blockchain & crypto": "badge-amber",
    "electric vehicles": "badge-teal",
    "space technology": "badge-purple",
    "software development": "badge-cyan",
    "finance": "badge-gold",
}

CATEGORY_ICON_MAP = {
    "politics": "🏛️", "sports": "⚽", "business": "💼", "technology": "💻",
    "entertainment": "🎬", "science": "🔬", "health": "🏥", "education": "📚",
    "environment": "🌿", "international affairs": "🌍", "crime & law": "⚖️",
    "artificial intelligence": "🤖", "funding & investment": "💰",
    "semiconductors & chips": "🔧", "startups & entrepreneurship": "🚀",
    "cloud computing": "☁️", "cybersecurity": "🔒", "blockchain & crypto": "⛓️",
    "electric vehicles": "🔋", "space technology": "🛰️", "software development": "⌨️",
    "finance": "📊",
}


def _badge_class(category: str) -> str:
    return CATEGORY_BADGE_MAP.get(category.lower(), "badge-default")


def _badge_icon(category: str) -> str:
    return CATEGORY_ICON_MAP.get(category.lower(), "📄")


def render_hero(tag: str, title: str, subtitle: str):
    now = datetime.now().strftime("%H:%M IST · %d %b %Y")
    st.markdown(f"""
    <div class="t9-hero">
        <div class="t9-hero-tag">
            <span>{tag}</span>
            <span style="margin-left: 20px;">
                <span class="live-indicator">
                    <span class="live-dot"></span>
                    LIVE · {now}
                </span>
            </span>
        </div>
        <div class="t9-hero-title">{title}</div>
        <div class="t9-hero-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def render_stat(value, label, color_class="stat-card-sky", text_color="var(--accent-sky)"):
    st.markdown(f"""
    <div class="stat-card {color_class}">
        <div class="stat-value" style="color: {text_color};">{value}</div>
        <div class="stat-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(icon, title, badge_text="", badge_class="badge-sky"):
    badge_html = f'<span class="badge {badge_class} section-badge">{badge_text}</span>' if badge_text else ""
    st.markdown(f"""
    <div class="section-header">
        <span style="font-size: 1.3rem;">{icon}</span>
        <span class="section-title">{title}</span>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


def render_article(title, source, category, link="#", tickers="", confidence=0.0, idx=0):
    badge = _badge_class(category)
    icon = _badge_icon(category)

    ticker_html = ""
    if tickers and tickers != "—":
        for t in tickers.split(","):
            t = t.strip()
            if t:
                ticker_html += f'<span class="ticker-tag">{t}</span>'

    conf_pct = max(confidence * 100, 5)
    conf_color = "var(--accent-emerald)" if confidence > 0.7 else "var(--accent-gold)" if confidence > 0.4 else "var(--accent-rose)"

    st.markdown(f"""
    <div class="article-card">
        <div class="article-card-header">
            <div>
                <span class="badge {badge}">{icon} {category}</span>
                {ticker_html}
            </div>
        </div>
        <p class="article-title"><a href="{link}" target="_blank" rel="noopener">{title}</a></p>
        <div class="article-meta">
            <span class="article-source">📡 {source}</span>
            <span class="article-source" style="margin-left: auto;">Confidence: {confidence:.0%}</span>
        </div>
        <div class="conf-bar">
            <div class="conf-fill" style="width:{conf_pct}%; background:{conf_color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sentiment_ring(score, label):
    if score >= 65:
        color = "var(--accent-emerald)"
        ring_color = "#34D399"
        glow = "rgba(52, 211, 153, 0.15)"
    elif score >= 45:
        color = "var(--accent-gold)"
        ring_color = "#FBBF24"
        glow = "rgba(251, 191, 36, 0.15)"
    else:
        color = "var(--accent-rose)"
        ring_color = "#FB7185"
        glow = "rgba(251, 113, 133, 0.15)"

    # SVG ring gauge
    radius = 80
    circumference = 2 * math.pi * radius
    progress = (score / 100) * circumference

    st.markdown(f"""
    <div class="sentiment-ring-container">
        <svg width="200" height="200" viewBox="0 0 200 200" style="filter: drop-shadow(0 0 20px {glow});">
            <!-- Background ring -->
            <circle cx="100" cy="100" r="{radius}" fill="none"
                    stroke="rgba(255,255,255,0.04)" stroke-width="8" />
            <!-- Progress ring -->
            <circle cx="100" cy="100" r="{radius}" fill="none"
                    stroke="{ring_color}" stroke-width="8"
                    stroke-linecap="round"
                    stroke-dasharray="{circumference}"
                    stroke-dashoffset="{circumference - progress}"
                    transform="rotate(-90 100 100)"
                    style="transition: stroke-dashoffset 1.5s cubic-bezier(0.4, 0, 0.2, 1);" />
            <!-- Center glow -->
            <circle cx="100" cy="100" r="60" fill="rgba(6,8,15,0.6)" />
        </svg>
        <div style="position: relative; margin-top: -145px; text-align: center; z-index: 2;">
            <div class="sentiment-ring-value" style="color: {color};">{score}</div>
            <div class="sentiment-ring-label" style="color: {color};">{label}</div>
        </div>
        <div style="height: 30px;"></div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
        <div class="sidebar-logo">
            <div class="sidebar-logo-text">T9</div>
            <div class="sidebar-logo-sub">Unified Intelligence Terminal</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="t9-divider"></div>', unsafe_allow_html=True)

        # Navigation
        page = st.radio(
            "NAV",
            ["🧠  Command Center", "📰  T9-News", "💻  T9-Tech", "📈  T9-Stocks"],
            label_visibility="collapsed",
        )

        st.markdown('<div class="t9-divider"></div>', unsafe_allow_html=True)

        # System Status
        st.markdown('<div class="sidebar-section-title">Data Pipeline</div>', unsafe_allow_html=True)

        try:
            stats = get_feed_stats()
            st.markdown(f"""
            <div>
                <div class="sidebar-stat">
                    <div class="sidebar-dot" style="background: var(--accent-emerald);"></div>
                    News Feed
                    <span class="sidebar-stat-value">{stats['news_count']}</span>
                </div>
                <div class="sidebar-stat">
                    <div class="sidebar-dot" style="background: var(--accent-sky);"></div>
                    Tech Feed
                    <span class="sidebar-stat-value">{stats['tech_count']}</span>
                </div>
                <div class="sidebar-stat">
                    <div class="sidebar-dot" style="background: var(--accent-gold);"></div>
                    Finance Feed
                    <span class="sidebar-stat-value">{stats['finance_count']}</span>
                </div>
                <div class="sidebar-stat" style="margin-top: 6px; padding-top: 8px; border-top: 1px solid var(--border-subtle);">
                    <div class="sidebar-dot" style="background: var(--accent-violet);"></div>
                    Total Ingested
                    <span class="sidebar-stat-value" style="color: var(--accent-violet);">{stats['total_count']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.markdown('<div class="sidebar-stat"><span style="color: var(--accent-rose);">⚠ Feed status unavailable</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="t9-divider"></div>', unsafe_allow_html=True)

        # API Status
        st.markdown('<div class="sidebar-section-title">Connections</div>', unsafe_allow_html=True)

        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        try:
            gemini_key = gemini_key or st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass

        is_connected = gemini_key and gemini_key != "paste_your_gemini_api_key_here"
        gemini_status = "Online" if is_connected else "Offline"
        gemini_color = "var(--accent-emerald)" if is_connected else "var(--accent-rose)"

        st.markdown(f"""
        <div>
            <div class="sidebar-stat">
                <div class="sidebar-dot" style="background: {gemini_color};"></div>
                Gemini API
                <span class="sidebar-stat-value" style="color: {gemini_color};">{gemini_status}</span>
            </div>
            <div class="sidebar-stat">
                <div class="sidebar-dot" style="background: var(--accent-emerald);"></div>
                BART Model
                <span class="sidebar-stat-value" style="color: var(--accent-emerald);">Local</span>
            </div>
            <div class="sidebar-stat">
                <div class="sidebar-dot" style="background: var(--accent-emerald);"></div>
                NSE Master
                <span class="sidebar-stat-value" style="color: var(--accent-emerald);">167 cos</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="t9-divider"></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-footer">
            T9 Intelligence Terminal v1.0<br>
            Streamlit · BART · Gemini · pandas<br>
            © 2025 — SMAI Project
        </div>
        """, unsafe_allow_html=True)

    return page


# ═══════════════════════════════════════════════════════════════════
#  PAGE: COMMAND CENTER (THE BRAIN)
# ═══════════════════════════════════════════════════════════════════

def page_command_center():
    render_hero(
        "T9 COMMAND CENTER · PHASE B",
        "🧠 The Brain",
        "Real-time cross-domain intelligence synthesis — connecting signals across news, technology, and financial markets to surface actionable insights."
    )

    # Fetch data
    with st.spinner("⚡ Ingesting live data streams from 8 sources..."):
        all_articles = fetch_all_articles()
        news_df = fetch_news_articles()
        tech_df = fetch_tech_articles()
        finance_df = fetch_finance_articles()

    if all_articles.empty:
        st.warning("📡 No data available. RSS feeds may be temporarily unreachable. Please refresh the page.")
        return

    # ─── Stats Row ─────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_stat(len(all_articles), "Total Ingested", "stat-card-violet", "var(--accent-violet)")
    with c2:
        render_stat(len(news_df), "News Signals", "stat-card-emerald", "var(--accent-emerald)")
    with c3:
        render_stat(len(tech_df), "Tech Signals", "stat-card-sky", "var(--accent-sky)")
    with c4:
        render_stat(len(finance_df), "Market Signals", "stat-card-gold", "var(--accent-gold)")

    st.markdown('<div class="t9-divider"></div>', unsafe_allow_html=True)

    # ─── Sentiment + Synthesis ─────────────────────────────────
    col_left, col_right = st.columns([2, 5])

    with col_left:
        render_section_header("📊", "Market Sentiment", f"{len(all_articles)} articles", "badge-emerald")

        with st.spinner("🧠 Computing sentiment..."):
            sentiment = calculate_sentiment_score(all_articles)

        render_sentiment_ring(sentiment["score"], sentiment["label"])

        # Breakdown bars
        st.markdown(f"""
        <div style="padding: 0 8px;">
            <div class="sentiment-bar-row">
                <span class="sentiment-bar-label" style="color: var(--accent-emerald);">▲ Positive</span>
                <div class="sentiment-bar-track">
                    <div class="sentiment-bar-fill" style="width:{sentiment['positive_pct']}%; background: var(--accent-emerald);"></div>
                </div>
                <span class="sentiment-bar-pct" style="color: var(--accent-emerald);">{sentiment['positive_pct']}%</span>
            </div>
            <div class="sentiment-bar-row">
                <span class="sentiment-bar-label" style="color: var(--text-dim);">● Neutral</span>
                <div class="sentiment-bar-track">
                    <div class="sentiment-bar-fill" style="width:{sentiment['neutral_pct']}%; background: var(--text-dim);"></div>
                </div>
                <span class="sentiment-bar-pct" style="color: var(--text-dim);">{sentiment['neutral_pct']}%</span>
            </div>
            <div class="sentiment-bar-row">
                <span class="sentiment-bar-label" style="color: var(--accent-rose);">▼ Negative</span>
                <div class="sentiment-bar-track">
                    <div class="sentiment-bar-fill" style="width:{sentiment['negative_pct']}%; background: var(--accent-rose);"></div>
                </div>
                <span class="sentiment-bar-pct" style="color: var(--accent-rose);">{sentiment['negative_pct']}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        render_section_header("🔗", "Cross-Domain Intelligence Synthesis", "POWERED BY GEMINI", "badge-violet")

        with st.spinner("🧠 T9 Brain is synthesizing cross-domain insights..."):
            news_headlines = news_df["title"].tolist()[:10] if not news_df.empty else ["No news data"]
            tech_headlines = tech_df["title"].tolist()[:10] if not tech_df.empty else ["No tech data"]
            finance_headlines = finance_df["title"].tolist()[:10] if not finance_df.empty else ["No finance data"]

            synthesis = synthesize_cross_domain(
                news_headlines, tech_headlines, finance_headlines,
                sentiment_score=sentiment["score"]
            )

        st.markdown(f'<div class="synthesis-panel">{synthesis}</div>', unsafe_allow_html=True)

    st.markdown('<div class="t9-divider"></div>', unsafe_allow_html=True)

    # ─── Ticker Strip ──────────────────────────────────────────
    render_section_header("📡", "Live Headline Feed")
    top_headlines = all_articles["title"].tolist()[:20]
    items = '<span class="ticker-divider">│</span>'.join(top_headlines)
    st.markdown(f"""
    <div class="ticker-strip">
        <div class="ticker-scroll">{items}<span class="ticker-divider">│</span>{items}</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: T9-NEWS
# ═══════════════════════════════════════════════════════════════════

def page_news():
    render_hero(
        "T9-NEWS · PHASE A",
        "📰 News Intelligence",
        "Live Indian news from The Hindu, NDTV & Indian Express — automatically classified into 11 categories by BART zero-shot NLP."
    )

    with st.spinner("📡 Fetching news feeds..."):
        news_df = fetch_news_articles()

    if news_df.empty:
        st.warning("📡 No news articles available. RSS feeds may be temporarily unreachable.")
        return

    with st.spinner("🧠 Classifying articles with BART zero-shot..."):
        news_df = enrich_news_articles(news_df)

    # Filter + Stats
    col_filter, col_s1, col_s2, col_s3 = st.columns([2, 1, 1, 1])
    with col_filter:
        categories = ["All Categories"] + sorted(news_df["classified_category"].unique().tolist())
        selected = st.selectbox("Filter by Category", categories, key="news_filter", label_visibility="collapsed")

    filtered = news_df if selected == "All Categories" else news_df[news_df["classified_category"] == selected]

    with col_s1:
        render_stat(len(filtered), "Showing", "stat-card-emerald", "var(--accent-emerald)")
    with col_s2:
        render_stat(filtered["source"].nunique() if not filtered.empty else 0, "Sources", "stat-card-sky", "var(--accent-sky)")
    with col_s3:
        avg_c = f"{filtered['classification_confidence'].mean():.0%}" if not filtered.empty else "0%"
        render_stat(avg_c, "Avg Confidence", "stat-card-gold", "var(--accent-gold)")

    st.markdown('<div class="t9-divider"></div>', unsafe_allow_html=True)

    # Articles
    for idx, row in filtered.iterrows():
        with st.expander(f"{_badge_icon(row['classified_category'])}  {row['title'][:100]}{'…' if len(row['title'])>100 else ''}", expanded=False):
            render_article(
                title=row["title"], source=row["source"],
                category=row["classified_category"], link=row["link"],
                confidence=row["classification_confidence"], idx=idx,
            )
            if st.button("🤖 Generate AI Summary", key=f"sum_n_{idx}"):
                with st.spinner("Generating..."):
                    summary = summarize_article(row["title"], row["summary_raw"])
                st.markdown(f"""
                <div class="ai-summary">
                    <div class="ai-summary-label">AI-Generated Summary</div>
                    {summary}
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: T9-TECH
# ═══════════════════════════════════════════════════════════════════

def page_tech():
    render_hero(
        "T9-TECH · PHASE A",
        "💻 Tech Intelligence",
        "Global technology news from TechCrunch, The Verge & YourStory — filtered by AI, Funding, Semiconductors, Startups & more."
    )

    with st.spinner("📡 Fetching tech feeds..."):
        tech_df = fetch_tech_articles()

    if tech_df.empty:
        st.warning("📡 No tech articles available. RSS feeds may be temporarily unreachable.")
        return

    with st.spinner("🧠 Classifying tech topics..."):
        tech_df = enrich_tech_articles(tech_df)

    # Filter + Stats
    col_filter, col_s1, col_s2, col_s3 = st.columns([2, 1, 1, 1])
    with col_filter:
        topics = ["All Topics"] + sorted(tech_df["tech_topic"].unique().tolist())
        selected = st.selectbox("Filter by Topic", topics, key="tech_filter", label_visibility="collapsed")

    filtered = tech_df if selected == "All Topics" else tech_df[tech_df["tech_topic"] == selected]

    with col_s1:
        render_stat(len(filtered), "Showing", "stat-card-sky", "var(--accent-sky)")
    with col_s2:
        render_stat(filtered["source"].nunique() if not filtered.empty else 0, "Sources", "stat-card-emerald", "var(--accent-emerald)")
    with col_s3:
        avg_c = f"{filtered['topic_confidence'].mean():.0%}" if not filtered.empty else "0%"
        render_stat(avg_c, "Topic Confidence", "stat-card-gold", "var(--accent-gold)")

    st.markdown('<div class="t9-divider"></div>', unsafe_allow_html=True)

    for idx, row in filtered.iterrows():
        with st.expander(f"{_badge_icon(row['tech_topic'])}  {row['title'][:100]}{'…' if len(row['title'])>100 else ''}", expanded=False):
            render_article(
                title=row["title"], source=row["source"],
                category=row["tech_topic"], link=row["link"],
                confidence=row["topic_confidence"], idx=idx,
            )
            if st.button("🤖 Generate AI Summary", key=f"sum_t_{idx}"):
                with st.spinner("Generating..."):
                    summary = summarize_article(row["title"], row["summary_raw"])
                st.markdown(f"""
                <div class="ai-summary">
                    <div class="ai-summary-label">AI-Generated Summary</div>
                    {summary}
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: T9-STOCKS
# ═══════════════════════════════════════════════════════════════════

def page_stocks():
    render_hero(
        "T9-STOCKS · PHASE A",
        "📈 Market Intelligence",
        "Finance news from MoneyControl & Economic Times — tagged with NSE ticker symbols via AI-powered fuzzy matching against 167 companies."
    )

    with st.spinner("📡 Fetching finance feeds..."):
        finance_df = fetch_finance_articles()

    if finance_df.empty:
        st.warning("📡 No finance articles available. RSS feeds may be temporarily unreachable.")
        return

    with st.spinner("🏷️ Matching companies to NSE tickers..."):
        finance_df = tag_finance_articles(finance_df)

    tagged_count = len(finance_df[finance_df["tickers"] != "—"])

    # Stats
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        render_stat(len(finance_df), "Finance Articles", "stat-card-gold", "var(--accent-gold)")
    with col_s2:
        render_stat(tagged_count, "Ticker-Tagged", "stat-card-sky", "var(--accent-sky)")
    with col_s3:
        pct = f"{tagged_count / len(finance_df) * 100:.0f}%" if len(finance_df) > 0 else "0%"
        render_stat(pct, "Match Rate", "stat-card-emerald", "var(--accent-emerald)")

    st.markdown('<div class="t9-divider"></div>', unsafe_allow_html=True)

    show_tagged = st.checkbox("Show only ticker-tagged articles", value=False, key="stock_filter")
    display_df = finance_df[finance_df["tickers"] != "—"] if show_tagged else finance_df

    for idx, row in display_df.iterrows():
        with st.expander(f"📈  {row['title'][:100]}{'…' if len(row['title'])>100 else ''}", expanded=False):
            render_article(
                title=row["title"], source=row["source"],
                category="Finance", link=row["link"],
                tickers=row.get("tickers", ""), confidence=0.8, idx=idx,
            )

            ticker_details = row.get("ticker_details", [])
            if ticker_details:
                st.markdown("**Matched Tickers:**")
                for td in ticker_details:
                    icon = "🟢" if td["confidence"] >= 85 else "🟡" if td["confidence"] >= 70 else "🟠"
                    st.markdown(f'{icon} **`{td["ticker"]}`** — {td["company"]} ({td["confidence"]}% confidence)')

            if st.button("🤖 Generate AI Summary", key=f"sum_f_{idx}"):
                with st.spinner("Generating..."):
                    summary = summarize_article(row["title"], row["summary_raw"])
                st.markdown(f"""
                <div class="ai-summary">
                    <div class="ai-summary-label">AI-Generated Summary</div>
                    {summary}
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    inject_css()
    page = render_sidebar()

    if "Command Center" in page:
        page_command_center()
    elif "T9-News" in page:
        page_news()
    elif "T9-Tech" in page:
        page_tech()
    elif "T9-Stocks" in page:
        page_stocks()


if __name__ == "__main__":
    main()
