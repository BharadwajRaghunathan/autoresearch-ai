"""
memory.py — Persistent vector memory for AutoResearch AI.

Stores completed research reports in a local Chroma vector database so the
agent can recall past work across sessions. Uses sentence-transformers for
embeddings — no external embedding API key required.

Exports:
    save_research(brand_name, report_text) : Embed and persist a report.
    retrieve_similar(brand_name, k)        : Retrieve the k most similar past reports.

Resilience: all Chroma initialisation is wrapped in a try/except. If Chroma
is unavailable (e.g. conflicting package versions), both functions degrade
gracefully — save_research becomes a no-op and retrieve_similar returns [].
The agent continues to run; only cross-session memory is lost.
"""

from datetime import datetime

_vector_store = None

try:
    import chromadb
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    _chroma_client = chromadb.PersistentClient(path="./chroma_db")
    _vector_store = Chroma(
        client=_chroma_client,
        collection_name="research_memory",
        embedding_function=_embeddings,
    )
    print("[memory] Chroma initialized successfully.")
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

