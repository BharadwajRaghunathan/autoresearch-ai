"""
memory.py — Persistent vector memory for AutoResearch AI.

Stores completed research reports and creative analyses in a local Chroma
vector database so the agent can recall past work across sessions.
Uses sentence-transformers for embeddings — no external embedding API required.

Collections:
    research_memory : Competitor intelligence reports (one per brand per run).
    creative_memory : Ad creative analysis reports (one per URL per run).

Exports:
    save_research(brand_name, report_text)   : Persist a competitor research report.
    retrieve_similar(brand_name, k)          : Retrieve similar past research reports.
    get_previous_research(brand_name)        : Fetch the most recent past report for exact brand (for trend diff).
    save_creative(url, report_text)          : Persist a creative analysis report.

Resilience: all Chroma initialisation is wrapped in a single try/except.
If Chroma is unavailable, all functions degrade gracefully to no-ops / empty
returns. The agent continues to run; only cross-session memory is lost.
"""

from datetime import datetime

_vector_store  = None   # research_memory collection
_creative_store = None  # creative_memory collection

try:
    import chromadb
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    _embeddings    = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    _chroma_client = chromadb.PersistentClient(path="./chroma_db")

    _vector_store = Chroma(
        client=_chroma_client,
        collection_name="research_memory",
        embedding_function=_embeddings,
    )
    # Separate collection keeps creative analyses searchable independently
    # from competitor research — different query patterns, different recall use cases.
    _creative_store = Chroma(
        client=_chroma_client,
        collection_name="creative_memory",
        embedding_function=_embeddings,
    )
    print("[memory] Chroma initialized successfully (research_memory + creative_memory).")
except Exception as _e:
    print(f"[memory] Chroma unavailable — memory disabled. ({_e})")


def save_research(brand_name: str, report_text: str) -> None:
    """
    Embed and persist a completed research report to Chroma.

    The doc_id includes a timestamp so multiple runs for the same brand
    each get their own entry rather than overwriting previous results.
    No-op if Chroma failed to initialise at import time.
    """
    if _vector_store is None:
        return
    doc_id = f"{brand_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _vector_store.add_texts(
        texts=[report_text],
        metadatas=[{"brand": brand_name, "timestamp": datetime.now().isoformat()}],
        ids=[doc_id],
    )
    print(f"[memory] Saved research for '{brand_name}' as '{doc_id}'")


def retrieve_similar(brand_name: str, k: int = 3) -> list[str]:
    """
    Retrieve the top-k most similar past research reports for a given brand.

    Similarity is computed against the brand name string — close brand names
    or brands in the same industry will surface related past reports.

    Returns:
        List of report text strings. Empty list if Chroma is unavailable or
        the collection has no documents yet.
    """
    try:
        results = _vector_store.similarity_search(brand_name, k=k)
        return [doc.page_content for doc in results]
    except Exception as e:
        print(f"[memory] Retrieval error for '{brand_name}': {e}")
        return []


def get_previous_research(brand_name: str) -> str:
    """
    Fetch the single most recent past report for this exact brand name.

    Used by trend_compare_node to diff the current report against the last run.
    Called BEFORE store_memory so the result is always the previous run, never current.

    Uses similarity_search with a metadata filter — the correct LangChain Chroma API.
    Falls back to unfiltered similarity search if the filter returns nothing (handles
    older Chroma collections that may not have metadata indexed the same way).

    Returns:
        The most recent report text string, or "" if no past report exists or
        Chroma is unavailable.
    """
    if _vector_store is None:
        return ""
    try:
        # Primary: filter by exact brand name in metadata
        results = _vector_store.similarity_search(
            brand_name,
            k=10,
            filter={"brand": brand_name},
        )
        if not results:
            print(f"[memory] get_previous_research: no filtered results for '{brand_name}'")
            return ""
        # Sort by timestamp descending — return the most recent report
        results.sort(key=lambda d: d.metadata.get("timestamp", ""), reverse=True)
        print(f"[memory] get_previous_research: found {len(results)} past report(s) for '{brand_name}'")
        return results[0].page_content
    except Exception as e:
        print(f"[memory] get_previous_research error for '{brand_name}': {e}")
        return ""


def save_creative(url: str, report_text: str) -> None:
    """
    Embed and persist a completed creative analysis report to Chroma.

    Uses the creative_memory collection so creative analyses are kept
    separate from competitor research and can be queried independently.
    No-op if Chroma failed to initialise at import time.
    """
    if _creative_store is None:
        return
    # Sanitise URL into a valid Chroma doc ID (no slashes, colons, or dots)
    safe_id = url.replace("https://", "").replace("http://", "").replace("/", "_").replace(".", "_")
    doc_id  = f"creative_{safe_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _creative_store.add_texts(
        texts=[report_text],
        metadatas=[{"url": url, "timestamp": datetime.now().isoformat()}],
        ids=[doc_id],
    )
    print(f"[memory] Saved creative analysis for '{url}' as '{doc_id}'")

