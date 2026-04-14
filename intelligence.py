"""
╔══════════════════════════════════════════════════════════════════╗
║  T9 — INTELLIGENCE ENGINE                                       ║
║  Zero-Shot Classification · LLM Summarization · Trend Synthesis ║
╚══════════════════════════════════════════════════════════════════╝

Core NLP pipeline:
  • BART Zero-Shot Classification for article categorization
  • Gemini 1.5 Flash for summarization and cross-domain synthesis
  • Sentiment scoring across all domains
"""

import os
import logging
from typing import List, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

# ─── Logging ───────────────────────────────────────────────────────
logger = logging.getLogger("T9.Intelligence")
logger.setLevel(logging.INFO)

# ─── Category Labels ──────────────────────────────────────────────
NEWS_CATEGORIES = [
    "Politics", "Sports", "Business", "Technology",
    "Entertainment", "Science", "Health", "Education",
    "Environment", "International Affairs", "Crime & Law",
]

TECH_KEYWORDS = [
    "Artificial Intelligence", "Funding & Investment",
    "Semiconductors & Chips", "Startups & Entrepreneurship",
    "Cloud Computing", "Cybersecurity", "Blockchain & Crypto",
    "Electric Vehicles", "Space Technology", "Software Development",
]

SENTIMENT_LABELS = ["Positive", "Negative", "Neutral"]


# ═══════════════════════════════════════════════════════════════════
#  BART ZERO-SHOT CLASSIFIER
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _load_classifier():
    """
    Load the BART-Large-MNLI zero-shot classification pipeline.
    Uses @st.cache_resource to load the model only once across reruns.
    """
    try:
        from transformers import pipeline
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,  # CPU (use 0 for GPU if available)
        )
        logger.info("✓ BART classifier loaded successfully")
        return classifier
    except Exception as e:
        logger.error(f"✗ Failed to load BART classifier: {e}")
        return None


def classify_article(text: str, candidate_labels: List[str] = None) -> Tuple[str, float]:
    """
    Classify a single article using zero-shot classification.
    
    Args:
        text: Article headline or text to classify.
        candidate_labels: List of category labels. Defaults to NEWS_CATEGORIES.
    
    Returns:
        Tuple of (top_label, confidence_score)
    """
    if candidate_labels is None:
        candidate_labels = NEWS_CATEGORIES
    
    classifier = _load_classifier()
    if classifier is None:
        return ("Uncategorized", 0.0)
    
    try:
        text_truncated = text[:512]  # BART max input length
        result = classifier(
            text_truncated,
            candidate_labels=candidate_labels,
            multi_label=False,
        )
        return (result["labels"][0], round(result["scores"][0], 4))
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return ("Uncategorized", 0.0)


def classify_batch(texts: List[str], candidate_labels: List[str] = None, batch_size: int = 8) -> List[Tuple[str, float]]:
    """
    Classify a batch of articles efficiently.
    
    Args:
        texts: List of article headlines/text.
        candidate_labels: Category labels.
        batch_size: Number of articles to process at once.
    
    Returns:
        List of (label, confidence) tuples.
    """
    if candidate_labels is None:
        candidate_labels = NEWS_CATEGORIES
    
    classifier = _load_classifier()
    if classifier is None:
        return [("Uncategorized", 0.0)] * len(texts)
    
    results = []
    try:
        truncated = [t[:512] for t in texts]
        
        for i in range(0, len(truncated), batch_size):
            batch = truncated[i:i + batch_size]
            batch_results = classifier(
                batch,
                candidate_labels=candidate_labels,
                multi_label=False,
            )
            
            if isinstance(batch_results, dict):
                batch_results = [batch_results]
            
            for r in batch_results:
                results.append((r["labels"][0], round(r["scores"][0], 4)))
    except Exception as e:
        logger.error(f"Batch classification failed: {e}")
        results.extend([("Uncategorized", 0.0)] * (len(texts) - len(results)))
    
    return results


# ═══════════════════════════════════════════════════════════════════
#  GEMINI 1.5 FLASH — LLM ENGINE
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _init_gemini():
    """
    Initialize the Google Generative AI client.
    Reads API key from environment variable or Streamlit secrets.
    """
    try:
        import google.generativeai as genai
        
        # Priority: st.secrets > .env > os.environ
        api_key = None
        
        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass
        
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.environ.get("GEMINI_API_KEY")
            except ImportError:
                pass
        
        if not api_key or api_key == "paste_your_gemini_api_key_here":
            logger.warning("⚠ GEMINI_API_KEY not configured. Summaries will be unavailable.")
            return None
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        logger.info("✓ Gemini 1.5 Flash initialized successfully")
        return model
    except Exception as e:
        logger.error(f"✗ Failed to initialize Gemini: {e}")
        return None


def summarize_article(title: str, raw_text: str) -> str:
    """
    Generate a strict 3-line summary using Gemini 1.5 Flash.
    
    Args:
        title: Article headline.
        raw_text: Raw article body/description.
    
    Returns:
        3-line summary string, or fallback message.
    """
    model = _init_gemini()
    if model is None:
        return "⚠ AI Summary unavailable — configure GEMINI_API_KEY"
    
    prompt = f"""You are a concise intelligence analyst. Summarize the following news article in EXACTLY 3 short lines.
Each line should be a complete, informative sentence. No bullet points, no numbering.
Focus on: WHO, WHAT, WHY/IMPACT.

HEADLINE: {title}

ARTICLE EXCERPT: {raw_text[:800]}

YOUR 3-LINE SUMMARY:"""
    
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 200,
                "top_p": 0.8,
            },
        )
        
        if response and response.text:
            summary = response.text.strip()
            # Ensure we don't return more than 3 lines
            lines = [l.strip() for l in summary.split("\n") if l.strip()]
            return "\n".join(lines[:3])
        
        return "Summary generation returned empty response."
    except Exception as e:
        logger.error(f"Summarization failed for '{title[:50]}...': {e}")
        return f"⚠ Summary unavailable — {str(e)[:80]}"


def summarize_batch(articles: List[Dict], max_articles: int = 10) -> List[str]:
    """
    Summarize multiple articles. Rate-limited to prevent API abuse.
    
    Args:
        articles: List of dicts with 'title' and 'summary_raw' keys.
        max_articles: Maximum number to summarize (prevents excessive API calls).
    
    Returns:
        List of summary strings.
    """
    summaries = []
    for article in articles[:max_articles]:
        summary = summarize_article(
            article.get("title", ""),
            article.get("summary_raw", ""),
        )
        summaries.append(summary)
    
    # Fill remaining with placeholder
    remaining = len(articles) - len(summaries)
    if remaining > 0:
        summaries.extend(["Expand article for AI summary..."] * remaining)
    
    return summaries


# ═══════════════════════════════════════════════════════════════════
#  MARKET SENTIMENT SCORING
# ═══════════════════════════════════════════════════════════════════

def calculate_sentiment_score(articles_df: pd.DataFrame) -> Dict:
    """
    Calculate a high-level Market Sentiment Score from all fetched articles.
    
    Uses BART zero-shot to classify each headline as Positive/Negative/Neutral,
    then computes a weighted score on a 0-100 scale.
    
    Returns:
        Dict with 'score', 'label', 'positive_pct', 'negative_pct', 'neutral_pct',
        'breakdown' (list of per-article sentiments).
    """
    if articles_df.empty:
        return {
            "score": 50.0,
            "label": "Neutral",
            "positive_pct": 0,
            "negative_pct": 0,
            "neutral_pct": 100,
            "breakdown": [],
            "total_analyzed": 0,
        }
    
    headlines = articles_df["title"].tolist()
    sentiments = classify_batch(headlines, SENTIMENT_LABELS)
    
    positive_count = sum(1 for s, _ in sentiments if s == "Positive")
    negative_count = sum(1 for s, _ in sentiments if s == "Negative")
    neutral_count = sum(1 for s, _ in sentiments if s == "Neutral")
    total = len(sentiments)
    
    # Score: 0–100 where 50 is neutral
    # Formula: 50 + (positive_pct - negative_pct) * 0.5
    positive_pct = (positive_count / total) * 100 if total > 0 else 0
    negative_pct = (negative_count / total) * 100 if total > 0 else 0
    neutral_pct = (neutral_count / total) * 100 if total > 0 else 100
    
    score = 50 + (positive_pct - negative_pct) * 0.5
    score = max(0, min(100, score))  # Clamp to 0-100
    
    if score >= 65:
        label = "Bullish"
    elif score >= 55:
        label = "Mildly Bullish"
    elif score >= 45:
        label = "Neutral"
    elif score >= 35:
        label = "Mildly Bearish"
    else:
        label = "Bearish"
    
    breakdown = [
        {"headline": h, "sentiment": s, "confidence": c}
        for h, (s, c) in zip(headlines, sentiments)
    ]
    
    return {
        "score": round(score, 1),
        "label": label,
        "positive_pct": round(positive_pct, 1),
        "negative_pct": round(negative_pct, 1),
        "neutral_pct": round(neutral_pct, 1),
        "breakdown": breakdown,
        "total_analyzed": total,
    }


# ═══════════════════════════════════════════════════════════════════
#  CROSS-DOMAIN SYNTHESIS (THE BRAIN)
# ═══════════════════════════════════════════════════════════════════

def synthesize_cross_domain(
    news_headlines: List[str],
    tech_headlines: List[str],
    finance_headlines: List[str],
    sentiment_score: float = 50.0,
) -> str:
    """
    The Brain: Cross-domain trend synthesis using Gemini 1.5 Flash.
    
    Analyzes top headlines from all 3 domains simultaneously to identify
    ripple effects and interconnections.
    
    Args:
        news_headlines: Top 10 general news headlines.
        tech_headlines: Top 10 tech headlines.
        finance_headlines: Top 10 finance headlines.
        sentiment_score: Current market sentiment score (0-100).
    
    Returns:
        Structured markdown analysis string.
    """
    model = _init_gemini()
    if model is None:
        return """## ⚠ Cross-Domain Analysis Unavailable
        
Configure your `GEMINI_API_KEY` in the `.env` file to enable the AI-powered trend synthesis engine.

**Setup:** Copy `.env.example` → `.env` and add your key from [Google AI Studio](https://aistudio.google.com/apikey)."""
    
    news_block = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(news_headlines[:10]))
    tech_block = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(tech_headlines[:10]))
    finance_block = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(finance_headlines[:10]))
    
    prompt = f"""You are "T9 Brain" — a world-class intelligence analyst working for an elite institutional investor. 
You have access to real-time data from three domains. The current Market Sentiment Score is {sentiment_score}/100.

═══ TODAY'S GENERAL NEWS (India) ═══
{news_block}

═══ TODAY'S TECH NEWS (Global) ═══
{tech_block}

═══ TODAY'S FINANCE & MARKET NEWS ═══
{finance_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Produce a structured intelligence briefing with these EXACT sections (use markdown headers):

### 🌐 Macro Environment Snapshot
A 2-sentence assessment of the overall landscape.

### 🔗 Cross-Domain Ripple Effects
Identify 3-4 specific connections between news, tech, and markets. How is general news affecting tech? How might tech shifts impact Indian stock market sectors? Be specific — name companies, sectors, and policies.

### 📊 Sector Impact Matrix
A brief analysis of which NSE sectors (IT, Banking, Pharma, Auto, Energy, FMCG) are most likely to be affected today and WHY.

### ⚡ Key Risk Signals
2-3 risk factors or warning signs visible across the data.

### 🎯 Strategic Takeaway
One bold, actionable insight for today — stated in a single powerful sentence.

Be concise, data-driven, and avoid generic platitudes. Reference specific headlines to support your analysis."""

    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.5,
                "max_output_tokens": 1500,
                "top_p": 0.9,
            },
        )
        
        if response and response.text:
            return response.text.strip()
        
        return "Cross-domain synthesis returned an empty response. Please retry."
    except Exception as e:
        logger.error(f"Cross-domain synthesis failed: {e}")
        return f"""## ⚠ Synthesis Error

The AI engine encountered an error during cross-domain analysis:

```
{str(e)[:200]}
```

**Troubleshooting:**
- Verify your `GEMINI_API_KEY` is valid
- Check your API quota at [Google AI Studio](https://aistudio.google.com/)
- The free tier allows 15 requests/minute"""


# ═══════════════════════════════════════════════════════════════════
#  ARTICLE ENRICHMENT PIPELINE
# ═══════════════════════════════════════════════════════════════════

def enrich_news_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich news articles with zero-shot classification labels.
    Adds 'classified_category' and 'classification_confidence' columns.
    """
    if df.empty:
        return df
    
    df = df.copy()
    results = classify_batch(df["title"].tolist(), NEWS_CATEGORIES)
    df["classified_category"] = [r[0] for r in results]
    df["classification_confidence"] = [r[1] for r in results]
    
    return df


def enrich_tech_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich tech articles with keyword/topic classification.
    """
    if df.empty:
        return df
    
    df = df.copy()
    results = classify_batch(df["title"].tolist(), TECH_KEYWORDS)
    df["tech_topic"] = [r[0] for r in results]
    df["topic_confidence"] = [r[1] for r in results]
    
    return df


# ═══════════════════════════════════════════════════════════════════
#  DIAGNOSTIC / EVAL UTILITIES
# ═══════════════════════════════════════════════════════════════════

def run_classification_eval(eval_df: pd.DataFrame, text_col: str = "headline_text", label_col: str = "category") -> Dict:
    """
    Evaluate classification accuracy against a labeled dataset (e.g., Kaggle India Headlines).
    
    Args:
        eval_df: DataFrame with text and ground-truth label columns.
        text_col: Column name for headline text.
        label_col: Column name for ground-truth category.
    
    Returns:
        Dict with accuracy, per-class metrics, and confusion data.
    """
    if eval_df.empty:
        return {"accuracy": 0.0, "total": 0, "details": []}
    
    sample = eval_df.head(100)  # Limit for compute budget
    predictions = classify_batch(sample[text_col].tolist(), NEWS_CATEGORIES)
    
    correct = 0
    details = []
    for i, (pred_label, pred_conf) in enumerate(predictions):
        true_label = sample.iloc[i][label_col]
        is_correct = pred_label.lower() == str(true_label).lower()
        if is_correct:
            correct += 1
        details.append({
            "text": sample.iloc[i][text_col][:80],
            "true_label": true_label,
            "predicted": pred_label,
            "confidence": pred_conf,
            "correct": is_correct,
        })
    
    return {
        "accuracy": round(correct / len(predictions) * 100, 2) if predictions else 0.0,
        "total": len(predictions),
        "correct": correct,
        "details": details,
    }
