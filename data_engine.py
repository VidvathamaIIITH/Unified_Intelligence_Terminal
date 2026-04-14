"""
╔══════════════════════════════════════════════════════════════════╗
║  T9 — DATA ENGINE                                               ║
║  RSS Ingestion · NSE Ticker Mapping · Data Normalization        ║
╚══════════════════════════════════════════════════════════════════╝

Handles all data acquisition:
  • Live RSS feed parsing from News, Tech, and Finance sources
  • NSE securities fuzzy matching for ticker tagging
  • DataFrame normalization and caching
"""

import os
import re
import html
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import pandas as pd
import feedparser
from fuzzywuzzy import fuzz, process

import streamlit as st

# ─── Logging ───────────────────────────────────────────────────────
logger = logging.getLogger("T9.DataEngine")
logger.setLevel(logging.INFO)

# ─── RSS Feed Registry ────────────────────────────────────────────
RSS_FEEDS = {
    "news": {
        "The Hindu": "https://www.thehindu.com/feeder/default.rss",
        "NDTV": "https://feeds.feedburner.com/ndtvnews-top-stories",
        "Indian Express": "https://indianexpress.com/feed/",
    },
    "tech": {
        "TechCrunch": "https://techcrunch.com/feed/",
        "The Verge": "https://www.theverge.com/rss/index.xml",
        "YourStory": "https://yourstory.com/feed",
    },
    "finance": {
        "MoneyControl": "https://www.moneycontrol.com/rss/latestnews.xml",
        "Economic Times": "https://economictimes.indiatimes.com/rssfeedstopstories.cms",
    },
}

MAX_ARTICLES_PER_SOURCE = 10  # 10 per source × 3 sources = 30 per category

# ─── NSE Data Path ────────────────────────────────────────────────
NSE_CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "nse_securities.csv")


# ═══════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _clean_html(raw_text: str) -> str:
    """Strip HTML tags and decode entities from RSS content."""
    if not raw_text:
        return ""
    clean = re.sub(r"<[^>]+>", "", raw_text)
    clean = html.unescape(clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:500]  # Cap at 500 chars for raw summary


def _parse_date(entry) -> str:
    """Extract and normalize publication date from RSS entry."""
    for field in ["published", "updated", "created"]:
        val = getattr(entry, field, None) or entry.get(field)
        if val:
            return val
    return datetime.now().strftime("%a, %d %b %Y %H:%M:%S")


def _safe_get(entry, field: str, default: str = "") -> str:
    """Safely extract a field from an RSS entry."""
    return getattr(entry, field, None) or entry.get(field, default) or default


# ═══════════════════════════════════════════════════════════════════
#  RSS FEED FETCHING
# ═══════════════════════════════════════════════════════════════════

def _fetch_single_feed(source_name: str, feed_url: str, category: str, max_items: int = MAX_ARTICLES_PER_SOURCE) -> List[Dict]:
    """
    Parse a single RSS feed and return normalized article dicts.
    Handles timeouts and malformed feeds gracefully.
    """
    articles = []
    try:
        feed = feedparser.parse(feed_url)
        
        if feed.bozo and not feed.entries:
            logger.warning(f"[{source_name}] Feed returned bozo error: {feed.bozo_exception}")
            return articles
        
        for entry in feed.entries[:max_items]:
            # Extract summary from multiple possible fields
            raw_summary = (
                _safe_get(entry, "summary") or
                _safe_get(entry, "description") or
                _safe_get(entry, "content", [{}])[0].get("value", "") if isinstance(_safe_get(entry, "content", []), list) else ""
            )
            
            articles.append({
                "title": _clean_html(_safe_get(entry, "title", "Untitled")),
                "source": source_name,
                "category": category,
                "link": _safe_get(entry, "link", "#"),
                "published": _parse_date(entry),
                "summary_raw": _clean_html(raw_summary),
            })
    except Exception as e:
        logger.error(f"[{source_name}] Failed to fetch feed: {e}")
    
    return articles


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news_articles() -> pd.DataFrame:
    """Fetch top 30 general news articles from The Hindu, NDTV, Indian Express."""
    all_articles = []
    for source, url in RSS_FEEDS["news"].items():
        all_articles.extend(_fetch_single_feed(source, url, "General News"))
    
    if not all_articles:
        return pd.DataFrame(columns=["title", "source", "category", "link", "published", "summary_raw"])
    
    return pd.DataFrame(all_articles)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_tech_articles() -> pd.DataFrame:
    """Fetch top 30 tech articles from TechCrunch, The Verge, YourStory."""
    all_articles = []
    for source, url in RSS_FEEDS["tech"].items():
        all_articles.extend(_fetch_single_feed(source, url, "Technology"))
    
    if not all_articles:
        return pd.DataFrame(columns=["title", "source", "category", "link", "published", "summary_raw"])
    
    return pd.DataFrame(all_articles)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_finance_articles() -> pd.DataFrame:
    """Fetch top 30 finance articles from MoneyControl, Economic Times."""
    all_articles = []
    for source, url in RSS_FEEDS["finance"].items():
        all_articles.extend(
            _fetch_single_feed(source, url, "Finance", max_items=15)
        )
    
    if not all_articles:
        return pd.DataFrame(columns=["title", "source", "category", "link", "published", "summary_raw"])
    
    return pd.DataFrame(all_articles)


def fetch_all_articles() -> pd.DataFrame:
    """Fetch and combine all articles from all categories."""
    news = fetch_news_articles()
    tech = fetch_tech_articles()
    finance = fetch_finance_articles()
    
    combined = pd.concat([news, tech, finance], ignore_index=True)
    return combined


# ═══════════════════════════════════════════════════════════════════
#  NSE TICKER FUZZY MATCHING
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_nse_master() -> pd.DataFrame:
    """Load the NSE securities master list from CSV."""
    try:
        df = pd.read_csv(NSE_CSV_PATH)
        df = df.dropna(subset=["Symbol", "Company Name"])
        df["Company Name Lower"] = df["Company Name"].str.lower().str.strip()
        df["Symbol"] = df["Symbol"].str.strip().str.upper()
        return df
    except FileNotFoundError:
        logger.error(f"NSE CSV not found at: {NSE_CSV_PATH}")
        return pd.DataFrame(columns=["Symbol", "Company Name", "Series", "ISIN", "Industry", "Company Name Lower"])
    except Exception as e:
        logger.error(f"Failed to load NSE CSV: {e}")
        return pd.DataFrame(columns=["Symbol", "Company Name", "Series", "ISIN", "Industry", "Company Name Lower"])


def _extract_potential_entities(text: str) -> List[str]:
    """
    Extract potential company name fragments from a headline.
    Looks for capitalized multi-word phrases and known patterns.
    """
    entities = []
    
    # Pattern 1: Capitalized sequences (2+ words)
    cap_pattern = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text)
    entities.extend(cap_pattern)
    
    # Pattern 2: All-caps words (likely tickers)
    ticker_pattern = re.findall(r'\b([A-Z]{2,10})\b', text)
    # Filter out common English all-caps words
    common_words = {"THE", "AND", "FOR", "THIS", "THAT", "WITH", "FROM", "HAVE",
                    "INTO", "OVER", "AFTER", "NEW", "TOP", "BIG", "ALL", "BUT",
                    "NOT", "HOW", "WHY", "GDP", "IPO", "CEO", "CFO", "RBI", "RSS",
                    "BSE", "NSE", "FII", "DII", "AGM", "EMI", "GST", "IPL"}
    entities.extend([t for t in ticker_pattern if t not in common_words and len(t) >= 3])
    
    # Pattern 3: Known Indian company short names
    known_shorts = [
        "Reliance", "Tata", "Infosys", "Wipro", "HCL", "Adani", "Bajaj",
        "Mahindra", "Maruti", "Suzuki", "Airtel", "Jio", "HDFC", "ICICI",
        "Kotak", "Axis", "Titan", "Asian Paints", "Sun Pharma", "Cipla",
        "Zomato", "Paytm", "Nykaa", "Vedanta", "JSW", "Coal India",
        "SBI", "PNB", "ITC", "DLF", "Apollo", "Dabur", "Godrej",
        "Havells", "Indigo", "Marico", "Siemens", "ONGC", "NTPC",
        "GAIL", "IRCTC", "BPCL", "IOC", "HAL", "BEL", "MRF",
        "Nestle", "Britannia", "Colgate", "Whirlpool", "Hero", "TVS",
        "Eicher", "Escorts", "Dixon", "Polycab", "Voltas", "Crompton",
        "Yes Bank", "Federal Bank", "Canara Bank", "Union Bank",
        "Nifty", "Sensex", "Bank Nifty",
    ]
    for name in known_shorts:
        if name.lower() in text.lower():
            entities.append(name)
    
    return list(set(entities))


def match_ticker(headline: str, nse_df: pd.DataFrame, threshold: int = 65) -> List[Tuple[str, str, int]]:
    """
    Match a finance headline against the NSE master list.
    
    Returns:
        List of (ticker, company_name, confidence_score) tuples.
    """
    if nse_df.empty:
        return []
    
    entities = _extract_potential_entities(headline)
    if not entities:
        return []
    
    matches = []
    seen_tickers = set()
    
    company_names = nse_df["Company Name Lower"].tolist()
    symbols = nse_df["Symbol"].tolist()
    
    for entity in entities:
        entity_lower = entity.lower().strip()
        
        # Direct ticker match first
        if entity.upper() in symbols:
            idx = symbols.index(entity.upper())
            ticker = nse_df.iloc[idx]["Symbol"]
            name = nse_df.iloc[idx]["Company Name"]
            if ticker not in seen_tickers:
                matches.append((ticker, name, 100))
                seen_tickers.add(ticker)
            continue
        
        # Fuzzy match against company names
        result = process.extractOne(
            entity_lower,
            company_names,
            scorer=fuzz.token_set_ratio,
            score_cutoff=threshold
        )
        
        if result:
            matched_name, score, idx = result
            ticker = nse_df.iloc[idx]["Symbol"]
            name = nse_df.iloc[idx]["Company Name"]
            if ticker not in seen_tickers:
                matches.append((ticker, name, score))
                seen_tickers.add(ticker)
    
    # Sort by confidence descending
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches[:5]  # Max 5 tickers per headline


def tag_finance_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich finance articles DataFrame with NSE ticker tags.
    Adds columns: 'tickers', 'ticker_details'
    """
    if df.empty:
        return df
    
    nse_df = load_nse_master()
    
    ticker_tags = []
    ticker_details = []
    
    for _, row in df.iterrows():
        matches = match_ticker(row["title"], nse_df)
        if matches:
            tags = [f"[NSE:{m[0]}]" for m in matches]
            details = [{"ticker": m[0], "company": m[1], "confidence": m[2]} for m in matches]
        else:
            tags = []
            details = []
        ticker_tags.append(", ".join(tags) if tags else "—")
        ticker_details.append(details)
    
    df = df.copy()
    df["tickers"] = ticker_tags
    df["ticker_details"] = ticker_details
    
    return df


# ═══════════════════════════════════════════════════════════════════
#  DATA STATISTICS
# ═══════════════════════════════════════════════════════════════════

def get_feed_stats() -> Dict:
    """Return statistics about the fetched data."""
    news = fetch_news_articles()
    tech = fetch_tech_articles()
    finance = fetch_finance_articles()
    
    return {
        "news_count": len(news),
        "tech_count": len(tech),
        "finance_count": len(finance),
        "total_count": len(news) + len(tech) + len(finance),
        "news_sources": news["source"].nunique() if not news.empty else 0,
        "tech_sources": tech["source"].nunique() if not tech.empty else 0,
        "finance_sources": finance["source"].nunique() if not finance.empty else 0,
    }
