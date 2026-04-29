"""
RAG Data Downloader  (v2 — fixed)
===================================
Fixes from v1 run:
  - Tier 7: Semantic Scholar 429 → exponential backoff + longer sleep
  - Tier 8 EN: REST summary API fails on long titles → use extracts API only
  - Tier 1: Rada/President/KMU RSS URLs corrected
  - Tier 3: Think-tank RSS URLs corrected (WordPress /feed/ pattern)

Usage:
    pip install requests feedparser beautifulsoup4 lxml tqdm
    python rag_downloader.py                         # all tiers
    python rag_downloader.py --tiers 1 3 7 8        # specific tiers only
    python rag_downloader.py --tiers 7 --sleep 5    # tier 7 only, slow

Output — rag_data/<tier>/<tier>.jsonl, one JSON object per line:
    { id, source, tier, url, title, text, date, topic, lang }
"""

import re
import json
import time
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import requests
from bs4 import BeautifulSoup
import feedparser
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TOPICS_FILE            = "topics_us.txt"
OUTPUT_DIR             = Path("rag_data")
LOG_FILE               = "rag_downloader.log"
SLEEP                  = 1.2    # seconds between requests (override with --sleep)
MAX_PER_SOURCE         = 200    # cap per individual source feed
REQUEST_TIMEOUT        = 15
SEMANTIC_SCHOLAR_SLEEP = 3.0    # their free tier: ~1 req/sec, use 3s to be safe

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; RAG-research-bot/1.0; "
        "UCU university research; contact: student@ucu.edu.ua)"
    )
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_topics(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def doc_id(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:12]


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def save_docs(docs: list[dict], path: Path):
    if not docs:
        log.info(f"  No docs to save → {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    log.info(f"  Saved {len(docs)} docs → {path}")


def get(url: str, retries: int = 3, backoff: float = 2.0, **kwargs) -> Optional[requests.Response]:
    """GET with retry + exponential backoff on 429/5xx."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, **kwargs)
            if r.status_code == 429:
                wait = backoff * (2 ** attempt)
                log.warning(f"  429 rate-limit on {url[:60]}… waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError as e:
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
            else:
                log.warning(f"  GET failed {url[:80]}: {e}")
        except Exception as e:
            log.warning(f"  GET error {url[:80]}: {e}")
            return None
    return None


def match_topic(text: str, topics: list[str]) -> str:
    text_lower = text.lower()
    for topic in topics:
        kws = re.findall(r"[а-яіїєґa-z]{4,}", topic.lower())
        if any(kw in text_lower for kw in kws[:3]):
            return topic
    return ""


def rss_docs(feed_url: str, source: str, tier: int, lang: str,
             topics: list[str], limit: int, ukraine_filter: bool = False,
             fetch_full: bool = False) -> list[dict]:
    """Generic RSS → list[doc]. fetch_full=True scrapes the article page."""
    feed = feedparser.parse(feed_url)
    entries = feed.entries
    if ukraine_filter:
        entries = [e for e in entries
                   if "ukrain" in (e.get("title", "") + e.get("summary", "")).lower()]
    docs = []
    for entry in tqdm(entries[:limit], desc=f"{source[:28]}", leave=False):
        url     = entry.get("link", "")
        title   = entry.get("title", "")
        summary = re.sub(r"<[^>]+>", "", entry.get("summary", ""))
        date    = (entry.get("published", "") or "")[:10]
        if not url:
            continue
        if fetch_full:
            r = get(url)
            text = clean_html(r.text) if r else summary
            if len(text) < 80:
                text = summary
        else:
            text = summary
        if len(text) < 30:
            continue
        docs.append({
            "id":     doc_id(url),
            "source": source,
            "tier":   tier,
            "url":    url,
            "title":  title,
            "text":   text[:7000],
            "date":   date,
            "topic":  match_topic(title + " " + summary, topics),
            "lang":   lang,
        })
        time.sleep(SLEEP)
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# TIER 1 — OFFICIAL UKRAINIAN SOURCES
# ─────────────────────────────────────────────────────────────────────────────

def download_tier1(topics: list[str], out_dir: Path):
    log.info(f"\n{'='*60}\nTIER 1 — Official Ukrainian Sources\n{'='*60}")
    docs = []

    # Верховна Рада — zakon.rada.gov.ua has RSS for recent laws
    rada_feeds = [
        ("https://zakon.rada.gov.ua/rss/zakon.rss",     "Верховна Рада — закони"),
        ("https://zakon.rada.gov.ua/rss/postanova.rss", "Верховна Рада — постанови"),
    ]
    for url, name in rada_feeds:
        log.info(f"  {name}...")
        docs += rss_docs(url, name, 1, "uk", topics, MAX_PER_SOURCE, fetch_full=False)

    # President — офіційний сайт president.gov.ua
    log.info("  Президент України...")
    docs += rss_docs(
        "https://www.president.gov.ua/news/rss",
        "Президент України", 1, "uk", topics, MAX_PER_SOURCE, fetch_full=False,
    )

    # Cabinet of Ministers
    log.info("  Кабінет Міністрів...")
    docs += rss_docs(
        "https://www.kmu.gov.ua/en/news/rss",
        "Кабінет Міністрів України", 1, "uk", topics, MAX_PER_SOURCE, fetch_full=False,
    )

    # MFA
    log.info("  МЗС України...")
    docs += rss_docs(
        "https://www.mfa.gov.ua/rss",
        "МЗС України", 1, "uk", topics, MAX_PER_SOURCE, fetch_full=False,
    )

    # Fallback: scrape Rada search for key topic-related laws
    # Uses the open Rada legislation search (no auth needed)
    log.info("  Rada search API (key topics)...")
    rada_queries = [
        "воєнний стан", "мобілізація", "антикорупція", "децентралізація",
        "мовна політика", "санкції", "збройні сили",
    ]
    rada_search = "https://zakon.rada.gov.ua/laws/main/find"
    for q in rada_queries:
        params = {"find": q, "lang": "uk"}
        r = get(rada_search, params=params)
        if not r:
            continue
        soup = BeautifulSoup(r.text, "lxml")
        for a in soup.select("div.law-list a[href]")[:5]:
            href  = a["href"]
            link  = f"https://zakon.rada.gov.ua{href}" if href.startswith("/") else href
            title = a.get_text(strip=True)
            r2    = get(link)
            text  = clean_html(r2.text) if r2 else ""
            if len(text) < 100:
                continue
            docs.append({
                "id":     doc_id(link),
                "source": "Верховна Рада — пошук",
                "tier":   1,
                "url":    link,
                "title":  title,
                "text":   text[:7000],
                "date":   "",
                "topic":  match_topic(q + " " + title, topics),
                "lang":   "uk",
            })
            time.sleep(SLEEP)

    save_docs(docs, out_dir / "tier1_official_ua.jsonl")
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# TIER 2 — INTERNATIONAL ORGANISATIONS  (kept from v1, working)
# ─────────────────────────────────────────────────────────────────────────────

def download_tier2(topics: list[str], out_dir: Path):
    log.info(f"\n{'='*60}\nTIER 2 — International Organisations\n{'='*60}")
    docs = []

    feeds = [
        # (rss_url, name, lang, ukraine_filter)
        ("https://news.un.org/feed/subscribe/en/news/region/europe/feed/rss.xml",
         "UN News — Europe", "en", True),
        ("https://www.unhcr.org/rss/news.xml",
         "UNHCR", "en", True),
        ("https://www.consilium.europa.eu/en/press/press-releases/rss/",
         "EU Council", "en", True),
        ("https://ec.europa.eu/commission/presscorner/rss/en_IP.xml",
         "European Commission", "en", True),
        ("https://www.nato.int/rss/feeds/newsroom.xml",
         "NATO", "en", True),
        ("https://www.icc-cpi.int/rss/pressreleases-en",
         "ICC", "en", True),
        ("https://www.osce.org/feeds/news",
         "OSCE", "en", True),
    ]

    for feed_url, name, lang, uf in feeds:
        log.info(f"  {name}...")
        docs += rss_docs(feed_url, name, 2, lang, topics, MAX_PER_SOURCE // 2,
                         ukraine_filter=uf, fetch_full=False)

    save_docs(docs, out_dir / "tier2_international.jsonl")
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# TIER 3 — WESTERN THINK TANKS  (fixed RSS URLs)
# ─────────────────────────────────────────────────────────────────────────────

def download_tier3(topics: list[str], out_dir: Path):
    log.info(f"\n{'='*60}\nTIER 3 — Western Think Tanks\n{'='*60}")
    docs = []

    feeds = [
        # Carnegie — main feed, filter Ukraine
        ("https://carnegieendowment.org/publications/feed",
         "Carnegie Endowment", "en", True),
        # Chatham House — Ukraine tag
        ("https://www.chathamhouse.org/ukraine/feed",
         "Chatham House", "en", False),
        ("https://www.chathamhouse.org/feed",
         "Chatham House", "en", True),
        # Atlantic Council — Ukraine section has its own feed
        ("https://www.atlanticcouncil.org/blogs/ukrainealert/feed/",
         "Atlantic Council — UkraineAlert", "en", False),
        ("https://www.atlanticcouncil.org/feed/",
         "Atlantic Council", "en", True),
        # Brookings
        ("https://www.brookings.edu/feed/",
         "Brookings Institution", "en", True),
        # RAND
        ("https://www.rand.org/pubs/rss/research_report.xml",
         "RAND Corporation", "en", True),
        ("https://www.rand.org/pubs/rss/periodical.xml",
         "RAND Corporation", "en", True),
        # Wilson Center — strong on post-Soviet/Ukraine
        ("https://www.wilsoncenter.org/feed",
         "Wilson Center", "en", True),
        # ECFR — European Council on Foreign Relations
        ("https://ecfr.eu/feed/",
         "ECFR", "en", True),
        # IISS
        ("https://www.iiss.org/feeds/commentary",
         "IISS", "en", True),
    ]

    for feed_url, name, lang, uf in feeds:
        log.info(f"  {name}...")
        docs += rss_docs(feed_url, name, 3, lang, topics,
                         MAX_PER_SOURCE // 4, ukraine_filter=uf, fetch_full=False)

    save_docs(docs, out_dir / "tier3_think_tanks.jsonl")
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# TIER 4 — UKRAINIAN THINK TANKS  (working from v1, kept + expanded)
# ─────────────────────────────────────────────────────────────────────────────

def download_tier4(topics: list[str], out_dir: Path):
    log.info(f"\n{'='*60}\nTIER 4 — Ukrainian Think Tanks\n{'='*60}")
    docs = []

    feeds = [
        ("https://razumkov.org.ua/feed",           "Центр Разумкова",               "uk", False),
        ("https://voxukraine.org/feed/",            "VoxUkraine",                    "uk", False),
        ("https://voxukraine.org/en/feed/",         "VoxUkraine EN",                 "en", False),
        ("https://texty.org.ua/feed/",              "Texty.org.ua",                  "uk", False),
        ("https://uifuture.org/feed/",              "Укр. інститут майбутнього",     "uk", False),
        ("https://www.kas.de/en/web/ukraine/rss",   "KAS Ukraine",                   "en", False),
        ("https://ua.boell.org/rss.xml",            "Heinrich Böll Stiftung Ukraine","uk", False),
    ]

    for feed_url, name, lang, uf in feeds:
        log.info(f"  {name}...")
        docs += rss_docs(feed_url, name, 4, lang, topics,
                         MAX_PER_SOURCE // 2, ukraine_filter=uf, fetch_full=False)

    save_docs(docs, out_dir / "tier4_ua_think_tanks.jsonl")
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# TIER 5 — RELIABLE MEDIA  (working from v1, kept)
# ─────────────────────────────────────────────────────────────────────────────

def download_tier5(topics: list[str], out_dir: Path):
    log.info(f"\n{'='*60}\nTIER 5 — Reliable Media\n{'='*60}")
    docs = []

    feeds = [
        ("https://www.pravda.com.ua/rss/",                          "Українська правда",    "uk", False),
        ("https://www.pravda.com.ua/rss/view_news/",                "Укр. правда — новини", "uk", False),
        ("https://suspilne.media/rss/all.rss",                      "Суспільне",            "uk", False),
        ("https://liga.net/rss/all.rss",                            "Ліга.net",             "uk", False),
        ("https://feeds.bbci.co.uk/ukrainian/rss.xml",              "BBC Ukraine",          "uk", False),
        ("https://feeds.bbci.co.uk/news/world/europe/rss.xml",      "BBC World Europe",     "en", True),
        ("https://www.theguardian.com/world/ukraine/rss",           "The Guardian Ukraine", "en", False),
        ("https://rss.nytimes.com/services/xml/rss/nyt/Europe.xml", "NYT Europe",           "en", True),
        ("https://www.rferl.org/api/zrqmkiouuv",                    "Radio Free Europe",    "en", True),
        ("https://www.aljazeera.com/xml/rss/all.xml",               "Al Jazeera",           "en", True),
    ]

    for feed_url, name, lang, uf in feeds:
        log.info(f"  {name}...")
        docs += rss_docs(feed_url, name, 5, lang, topics,
                         MAX_PER_SOURCE // 4, ukraine_filter=uf, fetch_full=False)

    save_docs(docs, out_dir / "tier5_media.jsonl")
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# TIER 6 — FACT-CHECKERS  (working from v1, kept)
# ─────────────────────────────────────────────────────────────────────────────

def download_tier6(topics: list[str], out_dir: Path):
    log.info(f"\n{'='*60}\nTIER 6 — Fact-Checkers\n{'='*60}")
    docs = []

    feeds = [
        ("https://www.stopfake.org/en/feed/",            "StopFake EN", "en", False),
        ("https://www.stopfake.org/uk/feed/",            "StopFake UA", "uk", False),
        ("https://voxukraine.org/voxcheck/feed/",        "VoxCheck",    "uk", False),
        ("https://voxukraine.org/en/voxcheck/feed/",     "VoxCheck EN", "en", False),
        ("https://detector.media/feed/",                 "Детектор медіа", "uk", True),
        ("https://fact-check.org.ua/feed/",              "Fact Check UA",  "uk", False),
    ]

    for feed_url, name, lang, uf in feeds:
        log.info(f"  {name}...")
        docs += rss_docs(feed_url, name, 6, lang, topics,
                         MAX_PER_SOURCE, ukraine_filter=uf, fetch_full=False)

    save_docs(docs, out_dir / "tier6_factcheck.jsonl")
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# TIER 7 — ACADEMIC via Semantic Scholar  (FIXED: backoff + longer sleep)
# ─────────────────────────────────────────────────────────────────────────────

def download_tier7(topics: list[str], out_dir: Path):
    """
    Semantic Scholar free API — max 1 req/sec unauthenticated.
    Get a free API key at https://www.semanticscholar.org/product/api
    and set env var SEMANTIC_SCHOLAR_KEY to lift rate limits (10 req/sec).
    """
    import os
    log.info(f"\n{'='*60}\nTIER 7 — Academic (Semantic Scholar)\n{'='*60}")

    api_key = os.getenv("SEMANTIC_SCHOLAR_KEY", "")
    if api_key:
        log.info("  Using Semantic Scholar API key (higher rate limit)")
    else:
        log.info("  No API key — using free tier (slow: 1 req/3s). Set SEMANTIC_SCHOLAR_KEY to speed up.")

    headers = dict(HEADERS)
    if api_key:
        headers["x-api-key"] = api_key

    queries = [
        "Holodomor famine Ukraine 1932 1933 Soviet",
        "Ukrainian Soviet Socialist Republic collectivization",
        "Ukrainian independence 1991 referendum post-Soviet",
        "Orange Revolution Ukraine 2004 electoral fraud",
        "Euromaidan Ukraine 2014 protest revolution dignity",
        "Crimea annexation Russia 2014 international law",
        "Donbas war Ukraine Russia 2014 2015 Minsk",
        "Russia Ukraine war 2022 invasion",
        "NATO enlargement Eastern Europe security",
        "Budapest Memorandum 1994 Ukraine nuclear disarmament",
        "Ukraine EU association agreement DCFTA",
        "Holodomor genocide recognition memory politics",
        "OUN UPA Ukrainian nationalism World War II",
        "Soviet Russification language policy Ukraine",
        "Ukraine decentralization local governance reform",
        "NABU anticorruption bureau Ukraine",
        "EU sanctions Russia Ukraine war",
        "ICC war crimes Ukraine prosecution",
        "Ukrainian diaspora lobbying Western governments",
        "Chornobyl Chernobyl 1986 political consequences glasnost",
        "Ukrainian Helsinki group human rights Soviet dissidents",
        "Black Sea security geopolitics Ukraine Russia",
        "grain corridor initiative Ukraine Black Sea 2022",
        "Ukrainian refugees displacement Europe 2022",
        "war reparations Ukraine Russia international mechanism",
        "Zelensky wartime governance Ukraine democracy",
        "Ukraine martial law constitutional rights wartime",
        "Minsk agreements Ukraine Donbas ceasefire",
        "Budapest memorandum security guarantees failure",
        "Ukrainian language law 2019 state language policy",
    ]

    api_base = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields   = "title,abstract,year,authors,externalIds,openAccessPdf,url"
    docs     = []
    seen     = set()

    for query in tqdm(queries, desc="Semantic Scholar"):
        params = {"query": query, "fields": fields, "limit": 10}
        # Retry loop with backoff specifically for 429
        for attempt in range(5):
            try:
                r = requests.get(api_base, params=params,
                                 headers=headers, timeout=REQUEST_TIMEOUT)
                if r.status_code == 429:
                    wait = SEMANTIC_SCHOLAR_SLEEP * (3 ** attempt)
                    log.warning(f"  429 on query '{query[:40]}' — wait {wait:.0f}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                log.warning(f"  Semantic Scholar error ({query[:40]}): {e}")
                data = {}
                break

        for paper in data.get("data", []):
            title    = paper.get("title", "") or ""
            abstract = paper.get("abstract", "") or ""
            year     = str(paper.get("year", ""))
            url      = paper.get("url", "") or ""
            pdf_url  = (paper.get("openAccessPdf") or {}).get("url", "")
            key      = doc_id(title + year)
            if key in seen or len(abstract) < 40:
                continue
            seen.add(key)
            text = abstract
            if pdf_url:
                text += f"\n\n[Open access PDF available: {pdf_url}]"
            docs.append({
                "id":     key,
                "source": "Semantic Scholar",
                "tier":   7,
                "url":    url,
                "title":  title,
                "text":   text[:6000],
                "date":   year,
                "topic":  match_topic(query + " " + title, topics),
                "lang":   "en",
            })

        # Always wait between queries — free tier is strict
        sleep_time = SEMANTIC_SCHOLAR_SLEEP if not api_key else 0.15
        time.sleep(sleep_time)

    save_docs(docs, out_dir / "tier7_academic.jsonl")
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# TIER 8 — WIKIPEDIA  (FIXED: drop REST summary, use extracts API only)
# ─────────────────────────────────────────────────────────────────────────────

def download_tier8(topics: list[str], out_dir: Path):
    log.info(f"\n{'='*60}\nTIER 8 — Wikipedia\n{'='*60}")

    def _wiki_for_lang(lang: str) -> list[dict]:
        api = f"https://{lang}.wikipedia.org/w/api.php"
        docs = []
        seen_titles: set[str] = set()

        for topic in tqdm(topics, desc=f"Wikipedia-{lang}"):
            # Step 1: search — get candidate titles
            r = get(api, params={
                "action": "query", "list": "search",
                "srsearch": topic, "srlimit": 3,
                "srprop": "snippet", "format": "json", "utf8": 1,
            })
            if not r:
                continue
            hits = r.json().get("query", {}).get("search", [])
            if not hits:
                time.sleep(SLEEP * 0.3)
                continue

            # Step 2: fetch full text via extracts API (no REST, avoids 400/500)
            titles_to_fetch = []
            for h in hits:
                t = h["title"]
                if t not in seen_titles:
                    titles_to_fetch.append(t)
                    seen_titles.add(t)

            if not titles_to_fetch:
                continue

            # Batch: up to 3 titles in one request
            r2 = get(api, params={
                "action": "query",
                "prop": "extracts|info",
                "explaintext": True,
                "exsectionformat": "plain",
                "titles": "|".join(titles_to_fetch),
                "inprop": "url",
                "format": "json",
                "utf8": 1,
            })
            if not r2:
                time.sleep(SLEEP)
                continue

            pages = r2.json().get("query", {}).get("pages", {})
            for page in pages.values():
                title   = page.get("title", "")
                extract = (page.get("extract", "") or "").strip()
                url     = page.get("fullurl", f"https://{lang}.wikipedia.org/wiki/{title}")
                if len(extract) < 100:
                    continue
                docs.append({
                    "id":     doc_id(title + lang),
                    "source": f"Wikipedia {lang.upper()}",
                    "tier":   8,
                    "url":    url,
                    "title":  title,
                    "text":   extract[:8000],
                    "date":   "",
                    "topic":  topic,
                    "lang":   lang,
                })

            time.sleep(SLEEP * 0.5)

        return docs

    for lang in ["uk", "en"]:
        log.info(f"\n  Language: {lang.upper()}")
        docs = _wiki_for_lang(lang)
        save_docs(docs, out_dir / f"tier8_wikipedia_{lang}.jsonl")

    return []


# ─────────────────────────────────────────────────────────────────────────────
# STATS & MANIFEST
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(output_dir: Path):
    log.info(f"\n{'='*60}\nDOWNLOAD SUMMARY\n{'='*60}")
    total = 0
    rows  = []
    for p in sorted(output_dir.rglob("*.jsonl")):
        with open(p, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        total += n
        rows.append((p.name, n))
        log.info(f"  {p.name:<48} {n:>5} docs")
    log.info(f"\n  TOTAL: {total} documents")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_docs": total,
        "files": {name: n for name, n in rows},
        "schema": {
            "id":     "md5 hash of url/title",
            "source": "source name",
            "tier":   "1–8 per trust hierarchy",
            "url":    "original url",
            "title":  "document title",
            "text":   "plain text, max 7–8k chars",
            "date":   "YYYY-MM-DD or year string",
            "topic":  "best-matching topic from topics_us.txt",
            "lang":   "uk or en",
        },
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info(f"  Manifest → {output_dir}/manifest.json")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global SLEEP, MAX_PER_SOURCE, SEMANTIC_SCHOLAR_SLEEP

    p = argparse.ArgumentParser(description="RAG data downloader (v2)")
    p.add_argument("--tiers", nargs="+", type=int, default=[1,2,3,4,5,6,7,8],
                   help="Tiers to download (default: all). E.g. --tiers 1 3 7")
    p.add_argument("--topics", default=TOPICS_FILE)
    p.add_argument("--output", default=str(OUTPUT_DIR))
    p.add_argument("--sleep",  type=float, default=SLEEP,
                   help=f"Seconds between requests (default {SLEEP})")
    p.add_argument("--max-articles", type=int, default=MAX_PER_SOURCE,
                   help=f"Max articles per source (default {MAX_PER_SOURCE})")
    args = p.parse_args()

    SLEEP            = args.sleep
    MAX_PER_SOURCE   = args.max_articles
    SEMANTIC_SCHOLAR_SLEEP = max(3.0, args.sleep * 2)

    out    = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    topics = load_topics(args.topics)

    log.info(f"Topics: {len(topics)} | Tiers: {args.tiers} | Output: {out.resolve()}")
    log.info(f"Sleep: {SLEEP}s | Max per source: {MAX_PER_SOURCE}")

    tier_map = {
        1: ("Official UA",       lambda: download_tier1(topics, out / "tier1_official_ua")),
        2: ("International",     lambda: download_tier2(topics, out / "tier2_international")),
        3: ("Think Tanks",       lambda: download_tier3(topics, out / "tier3_think_tanks")),
        4: ("UA Think Tanks",    lambda: download_tier4(topics, out / "tier4_ua_think_tanks")),
        5: ("Media",             lambda: download_tier5(topics, out / "tier5_media")),
        6: ("Fact-checkers",     lambda: download_tier6(topics, out / "tier6_factcheck")),
        7: ("Academic",          lambda: download_tier7(topics, out / "tier7_academic")),
        8: ("Wikipedia",         lambda: download_tier8(topics, out / "tier8_wikipedia")),
    }

    for n in sorted(args.tiers):
        if n not in tier_map:
            continue
        name, fn = tier_map[n]
        log.info(f"\n>>> Tier {n}: {name}")
        try:
            fn()
        except KeyboardInterrupt:
            log.info("Interrupted — saving progress so far...")
            break
        except Exception as e:
            log.error(f"Tier {n} failed: {e}", exc_info=True)

    print_stats(out)
    log.info("Done.")


if __name__ == "__main__":
    main()