"""
tools.py — Web research tools used by the AutoResearch AI agent.

Tools (LangChain @tool — callable by the agent graph):
    search_competitors    : Tavily web search across 3 parallel queries.
    scrape_website        : BeautifulSoup page scraper (title + description + body).

Plain functions (called directly from agent nodes):
    search_reddit         : Reddit-specific sentiment search via Tavily.
    search_news           : Recent news and funding headlines.
    search_pricing        : Competitor pricing page search.
    extract_brand_identity: Scrape a homepage for brand name and logo URL.

All Tavily calls go through _tavily_search(), a single internal helper that
normalises the response shape and handles errors without crashing the agent.
"""

import os
import re
import requests
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from tavily import TavilyClient
from langchain_core.tools import tool

load_dotenv()

_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ─────────────────────────────────────────────
# INTERNAL HELPER
# ─────────────────────────────────────────────
def _tavily_search(query: str, max_results: int = 3) -> list[dict]:
    """
    Run a single Tavily search and return a normalised list of result dicts.

    Uses TavilyClient directly (not the LangChain wrapper) because the wrapper
    returns strings in some versions, making .get() calls fail downstream.

    Returns:
        List of {title, url, content} dicts. Empty list on any error.
    """
    try:
        response = _client.search(query, max_results=max_results)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "") or r.get("snippet", ""),
            }
            for r in response.get("results", [])
            if r.get("url")
        ]
    except Exception as e:
        print(f"[tavily] Error on '{query}': {e}")
        return []


# ─────────────────────────────────────────────
# TOOL 1: Competitor web search (parallel)
# ─────────────────────────────────────────────
@tool
def search_competitors(brand_name: str) -> list[dict]:
    """
    Search the web for competitors of a given brand.

    Runs 3 differently-phrased queries in parallel so the result set covers
    both direct comparisons and broader alternatives. Deduplicates by URL.

    Returns:
        List of {title, url, content} dicts (deduplicated).
    """
    queries = [
        f"{brand_name} top competitors analysis",
        f"{brand_name} vs alternatives",
        f"best alternatives to {brand_name}",
    ]
    results, seen_urls = [], set()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_tavily_search, q) for q in queries]
        for future in as_completed(futures):
            for hit in future.result():
                if hit["url"] not in seen_urls:
                    seen_urls.add(hit["url"])
                    results.append(hit)
    print(f"[search_competitors] Found {len(results)} results for '{brand_name}'")
    return results


# ─────────────────────────────────────────────
# TOOL 2: Website scraper
# ─────────────────────────────────────────────
@tool
def scrape_website(url: str) -> str:
    """
    Scrape a webpage and return its title, meta description, and body text.

    Body text is capped at 1000 characters to keep LLM prompts from bloating.
    Returns an empty string on any network or parse error — callers must handle
    empty returns rather than crashing.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, timeout=5, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string.strip() if soup.title else ""
        meta = soup.find("meta", attrs={"name": "description"})
        description = meta["content"].strip() if meta and meta.get("content") else ""
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
        body_text = " ".join(paragraphs)[:1000]
        return f"TITLE: {title}\nDESCRIPTION: {description}\nCONTENT: {body_text}"
    except Exception as e:
        print(f"[scrape] Failed {url}: {e}")
        return ""


# ─────────────────────────────────────────────
# TOOL 3: Reddit sentiment (plain function — called directly from agent nodes)
# ─────────────────────────────────────────────
def search_reddit(brand_name: str) -> list[dict]:
    """
    Search Reddit for honest user opinions about the brand and its competitors.

    Uses site:reddit.com filter in Tavily queries so results are genuine Reddit
    threads rather than aggregator pages that syndicate Reddit content.

    Returns:
        List of {title, url, content} dicts filtered to reddit.com URLs only.
    """
    queries = [
        f"site:reddit.com {brand_name} competitors review",
        f"site:reddit.com {brand_name} alternatives honest opinion",
    ]
    results, seen = [], set()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_tavily_search, q, 3) for q in queries]
        for future in as_completed(futures):
            for hit in future.result():
                if hit["url"] not in seen and "reddit.com" in hit["url"]:
                    seen.add(hit["url"])
                    results.append(hit)
    print(f"[search_reddit] Found {len(results)} Reddit results for '{brand_name}'")
    return results


# ─────────────────────────────────────────────
# TOOL 4: News search (plain function)
# ─────────────────────────────────────────────
def search_news(brand_name: str) -> list[dict]:
    """
    Search for recent news about the brand and competitor market.

    Two queries cover both competitive news and funding/growth signals,
    which are useful for the "Market Positioning Gaps" report section.

    Returns:
        List of {title, url, content} dicts (deduplicated).
    """
    queries = [
        f"{brand_name} competitors news 2025 2026",
        f"{brand_name} market funding growth 2026",
    ]
    results, seen = [], set()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_tavily_search, q, 3) for q in queries]
        for future in as_completed(futures):
            for hit in future.result():
                if hit["url"] not in seen:
                    seen.add(hit["url"])
                    results.append(hit)
    print(f"[search_news] Found {len(results)} news results for '{brand_name}'")
    return results


# ─────────────────────────────────────────────
# TOOL 5: Pricing search (plain function)
# ─────────────────────────────────────────────
def search_pricing(brand_name: str) -> list[dict]:
    """
    Search for pricing information of the brand's competitors.

    Returns:
        List of {title, url, content} dicts from pricing/plans pages.
    """
    results = _tavily_search(f"{brand_name} competitor pricing plans cost per month 2025", max_results=4)
    print(f"[search_pricing] Found {len(results)} pricing results for '{brand_name}'")
    return results


# ─────────────────────────────────────────────
# TOOL 6: Brand identity extractor
# ─────────────────────────────────────────────
def extract_brand_identity(url: str) -> dict:
    """
    Scrape a brand homepage and extract its name and logo URL.

    Extraction priority:
        name : og:site_name → <title> (first segment before |/–) → first <h1>
        logo : og:image → <img> whose src/alt/class contains "logo"

    Why not use a headless browser: BeautifulSoup + requests is fast and free.
    Most modern sites include Open Graph tags, which provide clean data without
    parsing dynamic JS content.

    Returns:
        dict with keys "name" (str) and "logo" (str). Empty strings if not found.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, timeout=8, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Brand name
        name = ""
        og_site = soup.find("meta", property="og:site_name")
        if og_site and og_site.get("content"):
            name = og_site["content"].strip()
        if not name and soup.title and soup.title.string:
            raw = soup.title.string.strip()
            name = re.split(r'\s*[\|\-–—]\s*', raw)[0].strip()
        if not name:
            h1 = soup.find("h1")
            if h1:
                name = h1.get_text(strip=True)

        # Brand logo
        logo = ""
        og_img = soup.find("meta", property="og:image")
        if og_img and og_img.get("content"):
            logo = og_img["content"].strip()
        if not logo:
            for img in soup.find_all("img"):
                src = img.get("src", "")
                alt = img.get("alt", "").lower()
                cls = " ".join(img.get("class", [])).lower()
                if "logo" in src.lower() or "logo" in alt or "logo" in cls:
                    logo = src
                    if logo.startswith("/"):
                        logo = urljoin(url, logo)
                    break

        print(f"[extract_brand_identity] name='{name}' logo={bool(logo)}")
        return {"name": name, "logo": logo}

    except Exception as e:
        print(f"[extract_brand_identity] Failed {url}: {e}")
        return {"name": "", "logo": ""}

