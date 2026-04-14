<div align="center">

#  T9 — The Unified Intelligence Terminal

**Bridging the gap between news, technology, and markets with AI-powered cross-domain analysis**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-BART--MNLI-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/facebook/bart-large-mnli)
[![Gemini](https://img.shields.io/badge/Google-Gemini_1.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

*An AI-powered intelligence dashboard that automates the ingestion, classification, and summarization of live data from Indian News, Global Tech, and the NSE Stock Market — using Zero-Shot NLP and Generative AI to identify cross-domain "ripple effects."*

</div>

---

##  Table of Contents

- [The Problem](#the-problem)
- [The Solution](#the-solution)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [How It Works](#how-it-works)
- [License](#license)

---

## The Problem

In an era of information fragmentation, professionals and investors struggle to synthesize insights across domains. News, technology shifts, and market movements are deeply interconnected — yet they are consumed in silos:

-  A political decision in Delhi can tank pharma stocks the same afternoon
-  A semiconductor shortage announced in Taiwan impacts auto manufacturers on the NSE
-  Startup funding data from TechCrunch connects to venture capital sentiment in India

**No single platform connects these dots in real-time.** Existing tools either cover one domain deeply or sacrifice depth for breadth. The result: missed connections, delayed reactions, and information overload.

---

##  The Solution

**T9 — The Unified Intelligence Terminal** automates the entire insight pipeline:

1. **Ingest** → Live RSS feeds from 8+ sources across 3 domains
2. **Classify** → Zero-shot NLP (BART-Large-MNLI) categorizes every article without training data
3. **Enrich** → Fuzzy matching tags finance articles with NSE ticker symbols
4. **Summarize** → Gemini 1.5 Flash generates strict 3-line intelligence briefings
5. **Synthesize** → Cross-domain AI analysis identifies how news → tech → markets

The result is a single dashboard where a professional can see, in under 30 seconds, what matters today across all three domains.

---

##  Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      RSS FEEDS (LIVE)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │  News    │  │  Tech    │  │ Finance  │                    │
│  │ Hindu    │  │ TechCrch │  │ MoneyCtl │                    │
│  │ NDTV     │  │ Verge    │  │ EconTims │                    │
│  │ IndExp   │  │ YourStry │  │          │                    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                   │
│       │              │              │                         │
│       └──────────────┼──────────────┘                         │
│                      ▼                                        │
│            ┌──────────────────┐                               │
│            │  data_engine.py  │──── NSE CSV ── Fuzzy Match    │
│            └────────┬─────────┘                               │
│                     ▼                                         │
│            ┌──────────────────┐                               │
│            │ intelligence.py  │                               │
│            │  ┌────────────┐  │                               │
│            │  │ BART MNLI  │  │ ◄── Zero-Shot Classification │
│            │  └────────────┘  │                               │
│            │  ┌────────────┐  │                               │
│            │  │Gemini Flash│  │ ◄── Summarization + Synthesis │
│            │  └────────────┘  │                               │
│            └────────┬─────────┘                               │
│                     ▼                                         │
│            ┌──────────────────┐                               │
│            │     app.py       │                               │
│            │  Streamlit UI    │ ◄── Bloomberg Terminal Theme  │
│            └──────────────────┘                               │
└──────────────────────────────────────────────────────────────┘
```

---

##  Features

### Phase A — Independent Terminals

| Terminal | Description |
|----------|-------------|
| ** T9-News** | Indian news from The Hindu, NDTV, Indian Express — classified into 11 categories (Politics, Sports, Business, etc.) using BART zero-shot |
| ** T9-Tech** | Global tech news from TechCrunch, The Verge, YourStory — filtered by topics (AI, Funding, Semiconductors, Startups) |
| ** T9-Stocks** | Finance news tagged with NSE ticker symbols via AI fuzzy matching against 170+ company master list |

### Phase B — The Brain (Command Center)

| Feature | Description |
|---------|-------------|
| **Sentiment Gauge** | Real-time 0-100 market sentiment score computed from all ingested articles |
| **Cross-Domain Synthesis** | AI-generated briefing connecting news ↔ tech ↔ markets with sector impact analysis |
| **Live Headline Feed** | Scrolling ticker of latest headlines across all domains |

### Design

-  **Bloomberg Terminal × Dark Luxury** aesthetic
-  Glassmorphic cards with subtle glow effects
-  Animated sentiment gauge with color-coded states
-  Color-coded category badges
-  Confidence bars for classification transparency
-  Monospace data displays for financial precision

---

##  Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| UI | Streamlit | Dashboard framework |
| Data Ingestion | feedparser | Live RSS parsing |
| Classification | BART-Large-MNLI (HuggingFace) | Zero-shot article categorization |
| Summarization | Gemini 1.5 Flash | 3-line summaries + cross-domain synthesis |
| Ticker Matching | fuzzywuzzy + python-Levenshtein | NSE ticker tagging |
| Data | pandas | DataFrame manipulation |
| Config | python-dotenv | Environment variable management |

---

##  Quick Start

### Prerequisites

- **Python 3.9+** installed
- **pip** (Python package manager)
- A **Google Gemini API Key** (free tier: [Get it here](https://aistudio.google.com/apikey))

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/t9-intelligence-terminal.git
cd t9-intelligence-terminal
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate          # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure API Key

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder with your actual Gemini API key:

```
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

> **Note:** The BART model runs locally and does NOT require an API key. It will be downloaded automatically on first run (~1.6GB).

### Step 5: Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` 🎉

---

## ⚙️ Configuration

### API Keys

| Key | Required | Source | Where Used |
|-----|----------|--------|-----------|
| `GEMINI_API_KEY` | Yes (for summaries) | [Google AI Studio](https://aistudio.google.com/apikey) | `intelligence.py` — Gemini client initialization |
| BART Model | No key needed | Auto-downloaded from HuggingFace | `intelligence.py` — Local inference |

### How API Keys Are Read (Priority Order)

1. **Streamlit Secrets** (`st.secrets["GEMINI_API_KEY"]`) — for cloud deployment
2. **`.env` file** — via `python-dotenv` for local development
3. **OS Environment Variable** — `export GEMINI_API_KEY=...`

### Without Gemini Key

The app is **fully functional** without a Gemini key:
-  RSS ingestion works
-  BART classification works
-  NSE ticker matching works
-  AI summaries show "configure GEMINI_API_KEY" message
-  Cross-domain synthesis shows setup instructions

---

##  Deployment

### Streamlit Community Cloud

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → Select your repo → Set `app.py` as the main file
4. In **Advanced Settings** → **Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```
5. Deploy!

### HuggingFace Spaces

1. Create a new Space (Streamlit SDK)
2. Upload all project files
3. Add `GEMINI_API_KEY` to Space Secrets
4. The BART model will auto-download on first load

> ** Memory Note:** The BART model requires ~2GB RAM. Streamlit Cloud free tier has 1GB limit. For production use, consider HuggingFace Spaces with a CPU+ or GPU instance.

---

##  Project Structure

```
t9-intelligence-terminal/
├── app.py                  # Main Streamlit UI — routing, layout, styling
├── data_engine.py          # RSS fetching, NSE fuzzy matching, data normalization
├── intelligence.py         # BART classification, Gemini summarization, sentiment
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── .gitignore              # Git ignore rules
├── .streamlit/
│   └── config.toml         # Streamlit dark theme configuration
├── data/
│   └── nse_securities.csv  # NSE company master list (170+ companies)
├── report/
│   ├── TECHNICAL_REPORT.md # Full technical report
│   └── PITCH_SLIDE.md      # One-slide pitch content
└── README.md               # This file
```

---

##  Data Sources

### Live RSS Feeds

| Domain | Source | Feed URL |
|--------|--------|----------|
| News | The Hindu | `thehindu.com/feeder/default.rss` |
| News | NDTV | `feeds.feedburner.com/ndtvnews-top-stories` |
| News | Indian Express | `indianexpress.com/feed/` |
| Tech | TechCrunch | `techcrunch.com/feed/` |
| Tech | The Verge | `theverge.com/rss/index.xml` |
| Tech | YourStory | `yourstory.com/feed` |
| Finance | MoneyControl | `moneycontrol.com/rss/latestnews.xml` |
| Finance | Economic Times | `economictimes.indiatimes.com/rssfeedstopstories.cms` |

### Static Datasets

| Dataset | Purpose |
|---------|---------|
| **NSE Securities Master** (included) | 170+ company names/tickers for fuzzy matching |
| **Kaggle India Headlines** (optional) | 21 years of headlines for classification accuracy evaluation |

---

##  How It Works

### 1. Zero-Shot Classification

Using `facebook/bart-large-mnli` (trained on Multi-NLI), we classify articles into categories **without any domain-specific training data**. The model treats classification as a natural language inference task:

> *"Is this article about {category}?"* → Entailment score

### 2. NSE Ticker Matching

1. Extract potential company entities from headlines (capitalized phrases, known short names, all-caps tickers)
2. Attempt direct ticker match against the NSE CSV
3. Fall back to fuzzy matching using `token_set_ratio` with a 65% confidence threshold
4. Tag articles with `[NSE:TICKER]` badges

### 3. Cross-Domain Synthesis

The "Brain" sends the top 10 headlines from each domain to Gemini 1.5 Flash with a structured prompt requesting:
- Macro environment snapshot
- Cross-domain ripple effects
- Sector impact matrix
- Key risk signals
- Strategic takeaway

---

##  License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with  by T9 Intelligence Systems**

*Transforming information noise into actionable intelligence*

</div>
