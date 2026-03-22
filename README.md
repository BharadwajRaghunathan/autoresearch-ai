# AutoResearch AI

An autonomous marketing intelligence agent that researches any brand's competitive landscape from a single URL. Paste a brand URL, and the agent scrapes the website, runs parallel web research, evaluates data quality, loops until sufficient, and delivers a structured 8-section competitor report.

---

## Why This Exists

Manual competitor research is slow, inconsistent, and often biased toward whatever shows up on the first page of Google. AutoResearch AI turns that into a repeatable, observable process — every run is traced end-to-end, every LLM prompt is versioned in Langfuse, and every report is stored in a vector database for future recall.

---

## How It Works

```
User pastes brand URL
        │
        ▼
┌─────────────────────┐
│   identify_brand    │  Scrapes the URL, verifies it via LLM,
│                     │  extracts: industry, description, logo
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│       search        │  4 parallel Tavily streams:
│                     │  competitors · Reddit · news · pricing
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│       scrape        │  BeautifulSoup scrape of every
│                     │  competitor URL found
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  check_sufficiency  │  LLM self-evaluation: YES → proceed
│                     │  NO + iterations < 3 → loop back
└────────┬────────────┘
         │  (loop up to 3×)
         ▼
┌─────────────────────┐
│   generate_report   │  8-section markdown intelligence report
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    store_memory     │  Persist to Chroma for future recall
└─────────────────────┘
```

---

## Report Output

Each run produces a structured 8-section report:

1. **Top Competitors Identified** — 4–6 real competitors with one-line descriptions
2. **Competitor Analysis** — Positioning, strengths, weaknesses, target audience per competitor
3. **Competitor Scoring** — Feature · Pricing · Market Presence · Growth scored 1–10
4. **SWOT Analysis** — Strengths / Weaknesses / Opportunities / Threats for the input brand
5. **Reddit Sentiment** — What real users say, with direct quotes where available
6. **Market Positioning Gaps** — Specific opportunities competitors are missing
7. **Key Insights** — 7 bullet-point findings
8. **Recommended Actions** — 5 actionable steps to win market share

---

## Tech Stack

| Layer | Tool |
|---|---|
| Agent orchestration | LangGraph (StateGraph) |
| LLM | Groq — llama-3.3-70b-versatile |
| Web search | Tavily Search API |
| Web scraping | BeautifulSoup + requests |
| Vector memory | Chroma DB (local, persistent) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Prompt management | Langfuse |
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
# Fill in your API keys — see Environment Variables below
```

**3. Run**

```bash
streamlit run app.py
```

Open `http://localhost:8501`, paste any brand URL (e.g. `https://notion.so`), and click **Run Research**.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in the values:

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key — [console.groq.com](https://console.groq.com) |
| `TAVILY_API_KEY` | Yes | Tavily Search API key — [app.tavily.com](https://app.tavily.com) |
| `LANGCHAIN_API_KEY` | Yes | LangSmith API key — [smith.langchain.com](https://smith.langchain.com) |
| `LANGCHAIN_TRACING_V2` | Yes | Set to `true` to enable LangSmith tracing |
| `LANGCHAIN_PROJECT` | Yes | LangSmith project name (e.g. `autoresearch-ai`) |
| `LANGFUSE_PUBLIC_KEY` | Yes | Langfuse public key — [cloud.langfuse.com](https://cloud.langfuse.com) |
| `LANGFUSE_SECRET_KEY` | Yes | Langfuse secret key |
| `LANGFUSE_BASE_URL` | Yes | Langfuse host (default: `https://cloud.langfuse.com`) |

---

## Project Structure

```
autoresearch-ai/
├── app.py            # Streamlit UI — animated progress, report display
├── agent.py          # LangGraph state machine — all 6 nodes
├── chains.py         # Groq LLM setup + Langfuse prompt management
├── tools.py          # Tavily search + BeautifulSoup scraper
├── memory.py         # Chroma vector DB — save and retrieve reports
├── requirements.txt  # Python dependencies
├── .env.example      # Environment variable template
└── chroma_db/        # Auto-created — local Chroma database
```

---

## Observability

Every research run is fully traced across two platforms:

**LangSmith** — automatic agent traces when `LANGCHAIN_TRACING_V2=true`. Each LangGraph node appears as a span, showing input state, output state, and token usage.

**Langfuse** — secondary trace + live prompt management. Three prompts are managed in Langfuse UI and fetched at runtime:

| Prompt name | Used in node | Purpose |
|---|---|---|
| `verify-brand` | identify_brand | Confirm a webpage belongs to the target brand |
| `check-sufficiency` | check_sufficiency | Decide whether to loop or proceed to report |
| `generate-report` | generate_report | Produce the 8-section intelligence report |

Prompts can be edited in the Langfuse UI without redeploying. If a prompt is missing, the agent falls back to the inline version in `agent.py`.

---

## Key Engineering Decisions

**URL-first identity, not brand name guessing**
The agent never assumes a URL from a brand name. It either scrapes the user-provided URL directly or runs scored Tavily searches and verifies candidates with the LLM. This prevents the "wrong company" problem entirely.

**Industry-locked prompts**
`brand_industry` (e.g. "AI Marketing Technology") is extracted from the real website and injected into every report prompt. The LLM is explicitly instructed not to list competitors outside that industry.

**Parallel data collection**
The search node runs four Tavily streams simultaneously using `ThreadPoolExecutor`. Total search time is bounded by the slowest single query, not the sum of all four.

**Self-correcting loop**
After scraping, the LLM evaluates its own data quality. If insufficient, the agent loops back for another search pass — up to three iterations. The cap prevents runaway loops on sparse topics.

**Memory resilience**
Chroma initialisation is wrapped in a try/except at import time. If Chroma fails (e.g. dependency conflict), `save_research` becomes a no-op and the agent continues normally. Only cross-session recall is affected.

**No fake progress**
All UI timing comes from `time.time()` at each node completion event from LangGraph's `stream()`. There are no `sleep()` calls or simulated delays.

---

## Author

**Bharadwaj R**
- **GitHub**: [@BharadwajRaghunathan](https://github.com/BharadwajRaghunathan)
- **Email**: bharadwaj.r2112@gmail.com

---

## License

MIT
