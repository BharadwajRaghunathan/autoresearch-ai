# /deploy-check

Pre-deployment checklist for pushing AutoResearch AI to Streamlit Cloud.

## Instructions

Run through every item below and flag anything that will break on Cloud.

### 1. Secrets injection (critical)
- [ ] `app.py` injects `st.secrets` into `os.environ` at module level, BEFORE any import from `agent`, `chains`, `tools`, or `memory`
- [ ] All API keys read via `os.environ.get(...)` in `chains.py` and `tools.py` — no hardcoded keys anywhere
- [ ] Keys required: `GROQ_API_KEY`, `TAVILY_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGCHAIN_API_KEY`
- [ ] Optional: `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT`, `LANGFUSE_HOST`

### 2. File paths
- [ ] Chroma uses a relative path `./chroma_db/` — Cloud filesystem is ephemeral, memory becomes no-op on restart (this is expected and handled)
- [ ] No hardcoded Windows paths (`D:\\`, `C:\\`) anywhere in the codebase
- [ ] No `.env` file dependency — `load_dotenv()` is safe to call (returns False silently if no .env)

### 3. requirements.txt
- [ ] All imports used in the app are listed: `streamlit`, `langgraph`, `langchain`, `langchain-groq`, `langchain-community`, `langchain-chroma`, `langfuse`, `tavily-python`, `beautifulsoup4`, `chromadb`, `langsmith`
- [ ] No dev-only packages (pytest, black, etc.) unless needed at runtime

### 4. Langfuse prompts
- [ ] All 6 prompts exist in Langfuse UI: `verify-brand`, `check-sufficiency`, `generate-report`, `analyse-creative`, `score-creative`, `verdict-creative`
- [ ] If Langfuse is unreachable, inline fallbacks in `agent.py` will activate — verify fallback constants are present

### 5. Groq free-tier limits
- [ ] Total scraped content per creative analysis ≤ 2400 chars (`_PAGE_CONTENT_CAP` enforcement in `tools.py`)
- [ ] Research graph scrapes capped — `scraped_content` trimmed before LLM call
- [ ] Model is `llama-3.3-70b-versatile` — confirm it's still available on Groq free tier

### 6. Threading (Compare tab)
- [ ] Background threads NEVER call any `st.*` function
- [ ] `brand_status` dict is the only shared state between threads and main thread
- [ ] `time.sleep(0.5)` polling loop has a timeout guard so UI doesn't hang forever

### 7. Git hygiene
- [ ] `.env` is in `.gitignore`
- [ ] `chroma_db/` is in `.gitignore`
- [ ] `.claude/` is in `.gitignore`
- [ ] No secrets committed — run `git log --all -p | grep -i "api_key"` to verify

### 8. Streamlit config
- [ ] `streamlit/config.toml` not required but if present, check for local-only settings
- [ ] App entry point in Streamlit Cloud is set to `app.py`

## Example usage
`/deploy-check — Verify everything is ready before pushing to Streamlit Cloud`
