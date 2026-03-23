# AutoResearch AI

**Autonomous marketing intelligence agent — paste any brand URL, get a full competitor report in under 60 seconds.**

Built with LangGraph, Groq, Tavily, Chroma, LangSmith, and Langfuse. Three features, all powered by real web research — no hallucinated data, no fake placeholders.

---

## Features

### 🔍 Brand Research
Paste one URL → the agent identifies the brand, runs parallel competitor research across the web, Reddit, news, and pricing pages, self-evaluates data quality, loops until sufficient, and generates a structured 8-section intelligence report.

### ⚔️ Compare Brands
Paste up to 3 URLs → all brands are researched **simultaneously** using Python threading (total time = slowest single brand, not sum of all). Produces side-by-side reports plus a cross-brand LLM summary — shared vulnerabilities, common rivals, and a head-to-head verdict no single-brand tool can produce.

### 🔍 Creative Decoder
Paste any competitor's landing page → agent deep-scrapes creative elements (headlines, CTAs, pricing, trust signals), auto-discovers `/pricing` and `/features` pages, falls back to Tavily for JS-rendered sites, and produces a 7-section creative intelligence report with a 5-dimension scorecard and one-sentence verdict.

---

## How It Works

### Brand Research — LangGraph Flow

```
START
  │
  ▼
identify_brand ── Scrapes provided URL, verifies via LLM
  │               Extracts: industry, description, logo
  │               Never guesses a URL from a name
  ▼
search ────────── 4 parallel Tavily streams:
  │               competitors · Reddit · news · pricing
  ▼
scrape ─────────── BeautifulSoup scrape of every URL found
  │
  ▼
check_sufficiency ─ LLM evaluates data quality
  │                 YES → proceed  |  NO + iter < 3 → loop
  ▼
generate_report ─── 8-section markdown intelligence report
  │                 brand_industry injected to prevent hallucination
  ▼
store_memory ────── Persist to Chroma (research_memory collection)
  │
END
```

### Compare Brands — Parallel Threading

```
Main thread spawns N background threads simultaneously
  Thread 0 → full research graph for brand A
  Thread 1 → full research graph for brand B
  Thread 2 → full research graph for brand C

Main thread polls brand_status dict every 0.5s
→ updates 3 live columns in the UI independently

All threads join → render:
  Brand cards · Stats comparison table
  Full reports side by side · Cross-brand LLM summary
```

### Creative Decoder — 5-Node Graph

```
START
  │
  ▼
scrape_creative ─── BeautifulSoup: headlines, CTAs, prices, alts
  │                 JS fallback: Tavily if word_count < 200
  │                 Auto-discovers /pricing + /features (3s timeout)
  │                 Labels content: [HOMEPAGE] [PRICING] [FEATURES]
  ▼
analyse_creative ── LLM Call 1: 7-section qualitative report
  │                 Industry benchmark injected (SaaS/AI/Fintech/etc)
  ▼
score_creative ──── LLM Call 2: JSON scorecard with reasons
  │                 Evidence-based rules enforced in Python
  │                 (no trust found → score < 5, hard override)
  ▼
verdict_creative ── LLM Call 3: ≤25 word synthesis
  │                 "strength — weakness + how to exploit it"
  ▼
store_creative ──── Persist to Chroma (creative_memory collection)
  │
END
```

---

## Report Outputs

### Brand Research — 8 Sections
1. **Top Competitors Identified** — 4–6 real competitors, one-line descriptions
2. **Competitor Analysis** — Positioning, strengths, weaknesses, target audience
3. **Competitor Scoring** — Feature · Pricing · Market Presence · Growth (1–10 table)
4. **SWOT Analysis** — Strengths / Weaknesses / Opportunities / Threats
5. **Reddit Sentiment** — Real user opinions, direct quotes where available
6. **Market Positioning Gaps** — Opportunities competitors are missing
7. **Key Insights** — 7 bullet-point findings
8. **Recommended Actions** — 5 specific steps to win market share

### Creative Decoder — 7 Sections + Scorecard
1. **Creative Strategy Overview** — Core message, emotion/pain point targeted
2. **Headline Analysis** — Each headline with psychological trigger label
3. **CTA Analysis** — Conversion pressure rated soft / medium / aggressive
4. **Tone & Messaging** — Voice archetype, repeating words, benefit vs feature focus
5. **Pricing Strategy** — Presentation, anchoring techniques, freemium signals
6. **Weaknesses & Gaps** — What's missing vs industry benchmark
7. **How to Beat Them** — 5 specific "Instead of X, do Y — because Z" recommendations

**Scorecard:** Clarity · Emotional Impact · CTA Effectiveness · Trust Signals · Overall (each with evidence-based reason)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph — StateGraph |
| LLM | Groq — llama-3.3-70b-versatile |
| Web search | Tavily Search API |
| Web scraping | BeautifulSoup + requests |
| Vector memory | Chroma DB (local, persistent) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Prompt management | Langfuse — live editing without redeployment |
| LLM tracing | LangSmith + Langfuse |
| UI | Streamlit |
| LLM framework | LangChain |

---

## Quick Start

**1. Clone and install**

```bash
git clone https://github.com/BharadwajRaghunathan/autoresearch-ai.git
cd autoresearch-ai
pip install -r requirements.txt
```

**2. Configure environment**

```bash
cp .env.example .env
# Fill in your API keys
```

**3. Run**

```bash
streamlit run app.py
```

Open `http://localhost:8501` and paste any brand URL.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | [console.groq.com](https://console.groq.com) |
| `TAVILY_API_KEY` | Yes | [app.tavily.com](https://app.tavily.com) |
| `LANGCHAIN_API_KEY` | Yes | [smith.langchain.com](https://smith.langchain.com) |
| `LANGCHAIN_TRACING_V2` | Yes | Set to `true` |
| `LANGCHAIN_PROJECT` | Yes | Your LangSmith project name |
| `LANGFUSE_PUBLIC_KEY` | Yes | [cloud.langfuse.com](https://cloud.langfuse.com) |
| `LANGFUSE_SECRET_KEY` | Yes | Langfuse secret key |
| `LANGFUSE_BASE_URL` | Yes | `https://cloud.langfuse.com` |

**Streamlit Cloud:** Add all keys in App Settings → Secrets (TOML format). The app reads `st.secrets` automatically before initialising any API clients.

---

## Project Structure

```
autoresearch-ai/
├── app.py           # Streamlit UI — 3 tabs, animated progress, HTML renderers
├── agent.py         # LangGraph graphs — ResearchState + CreativeState
├── chains.py        # Groq LLM + Langfuse callback + prompt management
├── tools.py         # Tavily search + BeautifulSoup + creative extractor
├── memory.py        # Chroma — research_memory + creative_memory collections
├── requirements.txt
├── .env.example
└── chroma_db/       # Auto-created on first run
```

---

## Observability

Every run is fully traced across two platforms.

**LangSmith** — automatic when `LANGCHAIN_TRACING_V2=true`. Each LangGraph node appears as a span with input state, output state, and token usage.

**Langfuse** — secondary trace + live prompt management. All 6 prompts are managed in Langfuse UI and fetched at runtime:

| Prompt name | Node | Purpose |
|---|---|---|
| `verify-brand` | identify_brand | Confirm a webpage belongs to the target brand |
| `check-sufficiency` | check_sufficiency | Loop or proceed to report generation |
| `generate-report` | generate_report | 8-section competitor intelligence report |
| `analyse-creative` | analyse_creative | 7-section creative strategy analysis |
| `score-creative` | score_creative | Evidence-based JSON scorecard with reasons |
| `verdict-creative` | verdict_creative | One-sentence strength + weakness verdict |

Edit any prompt in Langfuse UI — the change is live on the next run. If Langfuse is unreachable, every prompt has an inline fallback in `agent.py`.

---

## Key Engineering Decisions

**URL-first identity** — The agent never guesses a URL from a brand name. It scrapes the provided URL directly, or runs scored Tavily candidate searches and LLM-verifies each result. This eliminates the "wrong company" problem.

**Industry-locked prompts** — `brand_industry` is extracted from real HTML and injected into every report prompt. The LLM is explicitly blocked from listing competitors outside that industry.

**Parallel execution everywhere** — Search node: 4 Tavily streams via `ThreadPoolExecutor`. Compare tab: N full research graphs via `threading.Thread`. Total time is bounded by the slowest single operation, not their sum.

**Evidence-enforced scoring** — The Creative Decoder scores are grounded in extracted data. If no trust signals were found in the HTML, the trust score is hard-capped at 4 in Python — regardless of what the LLM returns.

**Self-correcting research loop** — After scraping, the LLM evaluates its own data quality and decides whether to loop. Capped at 3 iterations to prevent runaway loops on sparse topics.

**JS-rendered site fallback** — When BeautifulSoup returns fewer than 200 words, the scraper switches to Tavily (which holds pre-rendered page snapshots). Content is labeled by source so the LLM knows what came from where.

**Memory resilience** — Chroma is initialised inside a try/except at import time. A failure makes save/retrieve no-ops — the agent continues normally, only cross-session memory is lost.

**No fake progress** — All UI timing comes from `time.time()` at each LangGraph node completion event. No `sleep()` calls, no simulated progress.

---

## Author

**Bharadwaj R**
- GitHub: [@BharadwajRaghunathan](https://github.com/BharadwajRaghunathan)
- Email: bharadwaj.r2112@gmail.com

---

## License

MIT
