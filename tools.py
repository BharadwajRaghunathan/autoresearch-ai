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
    scrape_creative_page  : Deep creative extraction — headlines, CTAs, prices, alts.

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


# ─────────────────────────────────────────────
# TOOL 7: Deep creative extractor (plain function — called from creative graph)
# ─────────────────────────────────────────────
def scrape_creative_page(url: str) -> dict:
    """
    Deep-scrape a landing page and extract all creative signal elements.

    Why a separate function from scrape_website: the competitor-research scraper
    caps content at 1000 chars for LLM prompt economy. Creative analysis needs
    the full headline/CTA/price structure, not just body text.

    Extracts:
        headlines    : All h1/h2/h3 text nodes.
        ctas         : Button text + anchor text where class/id signals CTA intent.
        meta_title   : <title> tag content.
        meta_description: og:description or meta[name=description] content.
        image_alts   : Non-empty alt attributes from all <img> tags.
        price_mentions: Currency-prefixed strings matched via regex.
        raw_content  : Body paragraph text, capped at 3000 chars.
        word_count   : Approximate visible word count.

    Returns:
        dict with the keys above. All list fields are empty lists on failure.
    """
    # Currency + number pattern covering $, £, ₹, € and common suffixes
    _PRICE_RE = re.compile(
        r'[$£₹€]\s*\d[\d,\.]*(?:\s*(?:/mo|/month|/year|/yr|USD|GBP|INR|EUR))?'
        r'|\d[\d,\.]*\s*(?:USD|GBP|INR|EUR)',
        re.IGNORECASE,
    )
    # Class/id keywords that strongly indicate a CTA element
    _CTA_SIGNALS = {"btn", "button", "cta", "call-to-action", "signup", "sign-up",
                    "getstarted", "get-started", "tryfree", "try-free", "download"}

    # Pricing-page plan tier words — short h2/h3 tags containing these are
    # plan names, not marketing copy, and should be separated out.
    _PLAN_TIER_WORDS = {
        "pro", "hobby", "enterprise", "teams", "team", "basic", "starter",
        "plus", "ultra", "free", "business", "growth", "scale", "premium",
        "standard", "advanced", "lite", "essential", "essentials", "annual",
        "monthly", "unlimited", "individual", "personal",
    }
    # If a short heading contains an action verb it is marketing copy, not a plan name
    _ACTION_WORDS = {
        "get", "start", "build", "create", "launch", "grow", "scale", "win",
        "make", "boost", "increase", "improve", "drive", "turn", "ship",
    }

    def _is_plan_name(text: str) -> bool:
        """Return True if the heading looks like a pricing tier label rather than copy."""
        words = text.lower().split()
        if len(words) >= 4:
            return False  # 4+ words → almost certainly real marketing copy
        if any(w in _ACTION_WORDS for w in words):
            return False  # contains an action verb → keep as headline
        return any(w in _PLAN_TIER_WORDS for w in words)

    # Cap per-page content at 800 chars so 3 pages fit comfortably in Groq's
    # free-tier token budget (3 × 800 = 2400 chars total going into the LLM).
    _PAGE_CONTENT_CAP = 800

    empty = {
        "headlines": [], "plan_names": [], "ctas": [], "meta_title": "",
        "meta_description": "", "image_alts": [], "price_mentions": [],
        "raw_content": "", "word_count": 0,
        "pages_scraped": [], "used_tavily_fallback": False,
    }

    def _page_text(soup_obj) -> str:
        """Extract paragraph text from a BeautifulSoup object, capped at _PAGE_CONTENT_CAP."""
        paras = [p.get_text(separator=" ", strip=True) for p in soup_obj.find_all("p") if p.get_text(strip=True)]
        return " ".join(paras)[:_PAGE_CONTENT_CAP]

    try:
        req_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response    = requests.get(url, timeout=8, headers=req_headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # ── Headlines — h1 always kept; h2/h3 filtered for plan names ──
        headlines, plan_names = [], []
        for tag in soup.find_all(["h1", "h2", "h3"]):
            text = tag.get_text(strip=True)
            if not text:
                continue
            if tag.name == "h1" or not _is_plan_name(text):
                headlines.append(text)
            else:
                plan_names.append(text)
        headlines  = headlines[:25]
        plan_names = list(dict.fromkeys(plan_names))[:15]

        # ── CTAs — buttons + anchors with CTA-signal classes/ids ──
        ctas = []
        seen_ctas: set = set()
        for tag in soup.find_all(["button", "a"]):
            text = tag.get_text(strip=True)
            if not text or len(text) > 80 or text in seen_ctas:
                continue
            tag_classes = " ".join(tag.get("class", [])).lower()
            tag_id      = (tag.get("id") or "").lower()
            tag_href    = (tag.get("href") or "").lower()
            is_cta = (
                any(sig in tag_classes for sig in _CTA_SIGNALS)
                or any(sig in tag_id for sig in _CTA_SIGNALS)
                or (tag.name == "button")
                or ("#" in tag_href and tag_href not in ("#", "#0"))
            )
            if is_cta:
                seen_ctas.add(text)
                ctas.append(text)
        ctas = ctas[:20]

        # ── Meta tags ──
        title_tag = soup.find("title")
        meta_title = title_tag.get_text(strip=True) if title_tag else ""

        og_desc  = soup.find("meta", property="og:description")
        std_desc = soup.find("meta", attrs={"name": "description"})
        meta_description = ""
        if og_desc and og_desc.get("content"):
            meta_description = og_desc["content"].strip()
        elif std_desc and std_desc.get("content"):
            meta_description = std_desc["content"].strip()

        # ── Image alt texts ──
        image_alts = [
            img["alt"].strip()
            for img in soup.find_all("img")
            if img.get("alt") and img["alt"].strip()
        ][:20]

        # ── Main page content + word count ──
        main_text  = _page_text(soup)
        word_count = len(" ".join(
            p.get_text(separator=" ", strip=True)
            for p in soup.find_all("p") if p.get_text(strip=True)
        ).split())

        # ── Price mentions from main page ──
        page_text_for_prices = soup.get_text(separator=" ")
        price_mentions = list(dict.fromkeys(_PRICE_RE.findall(page_text_for_prices)))[:15]

        # ── JS-rendered fallback ─────────────────────────────────────────────
        # If the static scrape returns very few words the page almost certainly
        # renders content via JavaScript. Tavily has already fetched a rendered
        # snapshot, so we use its content as a supplement.
        used_tavily_fallback = False
        tavily_text = ""
        if word_count < 200:
            used_tavily_fallback = True
            from urllib.parse import urlparse as _up
            domain = _up(url).netloc or url
            tv_results = _tavily_search(f"site:{domain} features pricing product overview", 3)
            if tv_results:
                tavily_text = "\n".join(r["content"] for r in tv_results if r.get("content"))[:800]
            print(f"[scrape_creative] JS-rendered ({word_count} words) — using Tavily fallback for {domain}")

        # ── Multi-page auto-discovery ────────────────────────────────────────
        # Scrape /pricing and /features relative to the root domain.
        # Skip /about — rarely contains creative elements worth analysing.
        # Each page is capped at _PAGE_CONTENT_CAP chars to stay within token budget.
        from urllib.parse import urlparse as _up2
        parsed   = _up2(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        # Determine which path the user already provided so we don't double-scrape it
        provided_path = parsed.path.rstrip("/").lower()

        extra_pages = [("/pricing", "PRICING PAGE"), ("/features", "FEATURES PAGE")]
        pages_scraped = ["homepage"]
        content_sections = [f"[HOMEPAGE]\n{main_text}"]
        if tavily_text:
            content_sections.append(f"[TAVILY CONTENT]\n{tavily_text}")

        for suffix, label in extra_pages:
            # Skip if user already provided this exact page
            if provided_path == suffix or provided_path == suffix + "/":
                continue
            page_url = base_url + suffix
            try:
                page_resp = requests.get(page_url, timeout=3, headers=req_headers)
                if page_resp.status_code == 200:
                    page_soup = BeautifulSoup(page_resp.text, "html.parser")
                    page_text = _page_text(page_soup)
                    if page_text and len(page_text.split()) > 20:
                        content_sections.append(f"[{label}]\n{page_text}")
                        pages_scraped.append(suffix.lstrip("/"))
                        # Collect extra price mentions from pricing page
                        extra_prices = _PRICE_RE.findall(page_soup.get_text(separator=" "))
                        for p in extra_prices:
                            if p not in price_mentions:
                                price_mentions.append(p)
                        price_mentions = price_mentions[:15]
            except Exception:
                pass  # 404 or timeout — silently skip

        # Cap total combined content at 2400 chars
        raw_content = "\n\n".join(content_sections)[:2400]
        total_words = word_count + sum(
            len(sec.split()) for sec in content_sections[1:]
        )

        print(
            f"[scrape_creative] {url} — "
            f"{len(headlines)} headlines, {len(plan_names)} plan names filtered, "
            f"{len(ctas)} CTAs, {len(price_mentions)} prices, "
            f"{total_words} words, pages={pages_scraped}, tavily={used_tavily_fallback}"
        )
        return {
            "headlines":            headlines,
            "plan_names":           plan_names,
            "ctas":                 ctas,
            "meta_title":           meta_title,
            "meta_description":     meta_description,
            "image_alts":           image_alts,
            "price_mentions":       price_mentions,
            "raw_content":          raw_content,
            "word_count":           total_words,
            "pages_scraped":        pages_scraped,
            "used_tavily_fallback": used_tavily_fallback,
        }

    except Exception as e:
        print(f"[scrape_creative] Failed {url}: {e}")
        return empty

