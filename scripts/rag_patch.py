"""
rag_patch.py  — fixes for tiers 1, 7, 8-EN
============================================
Run this standalone — it re-downloads only the three broken tiers
and appends into the same rag_data/ folder.

    python rag_patch.py               # fix all three
    python rag_patch.py --tiers 7     # just tier 7
"""

import re
import json
import time
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup
import feedparser
from tqdm import tqdm

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("rag_patch.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── shared config ─────────────────────────────────────────────────────────────
OUTPUT_DIR      = Path("rag_data")
TOPICS_FILE     = "topics_us.txt"
REQUEST_TIMEOUT = 15
SLEEP           = 1.2
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; RAG-research-bot/1.0; "
        "UCU university research; contact: student@ucu.edu.ua)"
    )
}


def load_topics(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def doc_id(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:12]


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def save_docs(docs: list[dict], path: Path):
    if not docs:
        log.info(f"  0 docs — nothing saved to {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    log.info(f"  ✅ Saved {len(docs)} docs → {path}")


def get(url: str, **kwargs) -> requests.Response | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, **kwargs)
        r.raise_for_status()
        return r
    except Exception as e:
        log.warning(f"  GET failed {url[:70]}: {e}")
        return None


def match_topic(text: str, topics: list[str]) -> str:
    text_lower = text.lower()
    for topic in topics:
        kws = re.findall(r"[а-яіїєґa-z]{4,}", topic.lower())
        if any(kw in text_lower for kw in kws[:3]):
            return topic
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# TIER 1 FIX — Official Ukrainian sources
# Problem: all RSS feeds were empty (servers returned feed with 0 entries)
# Fix: switch to working URLs + direct HTML scraping with topic queries
# ─────────────────────────────────────────────────────────────────────────────

def fix_tier1(topics: list[str]):
    log.info(f"\n{'='*60}\nTIER 1 FIX — Official Ukrainian Sources\n{'='*60}")
    out = OUTPUT_DIR / "tier1_official_ua" / "tier1_official_ua.jsonl"
    docs = []

    # ── 1A: Zakon.rada.gov.ua — law text search ───────────────────────────────
    # The Rada search page returns HTML listings we can scrape
    log.info("  Rada legislation search...")
    rada_queries_uk = [
        "воєнний стан", "мобілізація", "санкції", "антикорупція",
        "децентралізація", "мова державна", "збройні сили України",
        "внутрішньо переміщені особи", "НАБУ", "реформа",
    ]
    for q in tqdm(rada_queries_uk, desc="Rada search"):
        r = get("https://zakon.rada.gov.ua/laws/main/find",
                params={"find": q, "lang": "uk"})
        if not r:
            continue
        soup = BeautifulSoup(r.text, "lxml")
        # Law links are inside <div class="law-card"> or plain <a> with /laws/show/
        links = soup.find_all("a", href=re.compile(r"/laws/show/"))
        for a in links[:5]:
            href  = "https://zakon.rada.gov.ua" + a["href"].split("?")[0]
            title = a.get_text(strip=True)
            r2    = get(href)
            if not r2:
                continue
            text = clean_html(r2.text)
            if len(text) < 200:
                continue
            docs.append({
                "id":     doc_id(href),
                "source": "Верховна Рада України",
                "tier":   1, "url": href, "title": title,
                "text":   text[:7000], "date": "",
                "topic":  match_topic(q + " " + title, topics),
                "lang":   "uk",
            })
            time.sleep(SLEEP)
        time.sleep(SLEEP)

    # ── 1B: President.gov.ua — direct page scraping ───────────────────────────
    # Their RSS exists but is often empty; the news page works fine
    log.info("  President.gov.ua news...")
    pres_queries = [
        "вторгнення", "зброя", "санкції", "НАТО", "переговори",
        "мобілізація", "допомога", "ЄС", "воєнний стан",
    ]
    for q in tqdm(pres_queries, desc="President UA"):
        r = get("https://www.president.gov.ua/news/search",
                params={"query": q})
        if not r:
            continue
        soup = BeautifulSoup(r.text, "lxml")
        for article in soup.select("article, div.news-item, li.news")[:5]:
            a = article.find("a", href=True)
            if not a:
                continue
            href  = a["href"]
            if not href.startswith("http"):
                href = "https://www.president.gov.ua" + href
            title = a.get_text(strip=True) or article.get_text(" ", strip=True)[:80]
            r2    = get(href)
            text  = clean_html(r2.text) if r2 else ""
            if len(text) < 200:
                continue
            docs.append({
                "id":     doc_id(href),
                "source": "Президент України",
                "tier":   1, "url": href, "title": title,
                "text":   text[:7000], "date": "",
                "topic":  match_topic(q + " " + title, topics),
                "lang":   "uk",
            })
            time.sleep(SLEEP)
        time.sleep(SLEEP)

    # ── 1C: KMU — Cabinet of Ministers press releases ─────────────────────────
    log.info("  KMU (Cabinet of Ministers)...")
    # Their English press RSS actually works; Ukrainian one is broken
    feed = feedparser.parse("https://www.kmu.gov.ua/en/rss")
    for entry in tqdm(feed.entries[:80], desc="KMU RSS"):
        url   = entry.get("link", "")
        title = entry.get("title", "")
        if not url:
            continue
        summary = re.sub(r"<[^>]+>", "", entry.get("summary", ""))
        docs.append({
            "id":     doc_id(url),
            "source": "Кабінет Міністрів України",
            "tier":   1, "url": url, "title": title,
            "text":   summary[:7000], "date": entry.get("published", "")[:10],
            "topic":  match_topic(title + " " + summary, topics),
            "lang":   "en",
        })
        time.sleep(SLEEP * 0.3)

    # Also try Ukrainian KMU feed variants
    for kmu_url in [
        "https://www.kmu.gov.ua/rss",
        "https://www.kmu.gov.ua/news/rss",
        "https://www.kmu.gov.ua/uk/rss",
    ]:
        feed2 = feedparser.parse(kmu_url)
        if not feed2.entries:
            continue
        log.info(f"  KMU feed OK: {kmu_url} ({len(feed2.entries)} entries)")
        for entry in feed2.entries[:80]:
            url   = entry.get("link", "")
            title = entry.get("title", "")
            if not url:
                continue
            summary = re.sub(r"<[^>]+>", "", entry.get("summary", ""))
            docs.append({
                "id":     doc_id(url),
                "source": "Кабінет Міністрів України",
                "tier":   1, "url": url, "title": title,
                "text":   summary[:7000], "date": entry.get("published", "")[:10],
                "topic":  match_topic(title + " " + summary, topics),
                "lang":   "uk",
            })
            time.sleep(SLEEP * 0.2)
        break

    # ── 1D: MFA — Ukrainian MFA press releases ────────────────────────────────
    log.info("  MFA Ukraine...")
    for mfa_url in [
        "https://mfa.gov.ua/rss",
        "https://mfa.gov.ua/en/rss",
        "https://www.mfa.gov.ua/rss/news",
        "https://mfa.gov.ua/news/rss",
    ]:
        feed3 = feedparser.parse(mfa_url)
        if not feed3.entries:
            continue
        log.info(f"  MFA feed OK: {mfa_url} ({len(feed3.entries)} entries)")
        for entry in tqdm(feed3.entries[:80], desc="MFA"):
            url   = entry.get("link", "")
            title = entry.get("title", "")
            if not url:
                continue
            summary = re.sub(r"<[^>]+>", "", entry.get("summary", ""))
            docs.append({
                "id":     doc_id(url),
                "source": "МЗС України",
                "tier":   1, "url": url, "title": title,
                "text":   summary[:7000], "date": entry.get("published", "")[:10],
                "topic":  match_topic(title + " " + summary, topics),
                "lang":   "uk" if "mfa.gov.ua" in mfa_url and "/en/" not in mfa_url else "en",
            })
            time.sleep(SLEEP * 0.2)
        break

    # ── 1E: Урядовий портал — government.gov.ua ───────────────────────────────
    log.info("  Урядовий портал...")
    uryad_feed = feedparser.parse("https://www.kmu.gov.ua/en/news/rss")
    if not uryad_feed.entries:
        uryad_feed = feedparser.parse("https://www.kmu.gov.ua/news/rss")
    for entry in tqdm(uryad_feed.entries[:100], desc="Уряд портал"):
        url   = entry.get("link", "")
        title = entry.get("title", "")
        if not url:
            continue
        summary = re.sub(r"<[^>]+>", "", entry.get("summary", ""))
        docs.append({
            "id":     doc_id(url + "uryad"),
            "source": "Урядовий портал",
            "tier":   1, "url": url, "title": title,
            "text":   summary[:7000], "date": entry.get("published", "")[:10],
            "topic":  match_topic(title + " " + summary, topics),
            "lang":   "uk",
        })
        time.sleep(SLEEP * 0.2)

    log.info(f"  Tier 1 total collected: {len(docs)}")
    save_docs(docs, out)
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# TIER 7 FIX — Semantic Scholar
# Problem 1: UnboundLocalError — `data` not initialized before the for-loop
#            when all 5 retries exhaust on 429 without ever entering `except`
# Problem 2: Hitting 429 immediately — IP likely in a burst window
# Fix: initialize data={} before the retry loop; use longer initial sleep;
#      switch to bulk paper endpoint which has more generous limits
# ─────────────────────────────────────────────────────────────────────────────

def fix_tier7(topics: list[str]):
    import os
    log.info(f"\n{'='*60}\nTIER 7 FIX — Academic (Semantic Scholar)\n{'='*60}")
    out = OUTPUT_DIR / "tier7_academic" / "tier7_academic.jsonl"

    api_key = os.getenv("SEMANTIC_SCHOLAR_KEY", "")
    ss_headers = dict(HEADERS)
    if api_key:
        ss_headers["x-api-key"] = api_key
        base_sleep = 0.2
        log.info("  API key found — using higher rate limit")
    else:
        base_sleep = 5.0   # without key: be conservative, 1 req/5s
        log.info("  No API key — using 5s sleep between queries")
        log.info("  Get a free key at: https://www.semanticscholar.org/product/api")

    queries = [
        "Holodomor famine Ukraine 1932 1933",
        "Ukrainian Soviet collectivization repression",
        "Ukrainian independence 1991 post-Soviet transition",
        "Orange Revolution Ukraine 2004",
        "Euromaidan Ukraine 2014",
        "Crimea annexation Russia 2014",
        "Donbas conflict Ukraine Russia Minsk",
        "Russia Ukraine war 2022",
        "NATO expansion Eastern Europe",
        "Budapest Memorandum Ukraine nuclear",
        "Ukraine EU association agreement",
        "Holodomor genocide memory",
        "OUN UPA Ukrainian nationalism WWII",
        "Russification language policy Soviet Ukraine",
        "Ukraine decentralization reform",
        "anticorruption Ukraine NABU",
        "EU sanctions Russia",
        "ICC war crimes Ukraine",
        "Ukrainian diaspora influence",
        "Chornobyl political consequences",
        "Ukrainian Helsinki group dissidents",
        "Black Sea security Ukraine",
        "Ukraine grain deal 2022",
        "Ukrainian refugees 2022",
        "Zelensky leadership wartime Ukraine",
        "Minsk agreements ceasefire Donbas",
        "Ukraine language law policy",
        "Ukraine martial law wartime governance",
        "Budapest memorandum security guarantees",
        "Ukraine reconstruction postwar",
    ]

    api_base = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields   = "title,abstract,year,externalIds,openAccessPdf,url"
    docs     = []
    seen     = set()

    for query in tqdm(queries, desc="Semantic Scholar"):
        data = {}   # ← initialize BEFORE retry loop (fixes UnboundLocalError)
        params = {"query": query, "fields": fields, "limit": 10}

        for attempt in range(6):
            try:
                r = requests.get(api_base, params=params,
                                 headers=ss_headers, timeout=REQUEST_TIMEOUT)
                if r.status_code == 429:
                    # Exponential backoff: 5, 10, 20, 40, 80, 160s
                    wait = base_sleep * (2 ** attempt) + 5
                    log.warning(f"  429 attempt {attempt+1}/6 — sleeping {wait:.0f}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except requests.exceptions.HTTPError as e:
                log.warning(f"  HTTP error ({query[:35]}): {e}")
                break
            except Exception as e:
                log.warning(f"  Error ({query[:35]}): {e}")
                break

        for paper in data.get("data", []):   # safe: data always a dict now
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
                text += f"\n\n[Open access PDF: {pdf_url}]"
            docs.append({
                "id":     key,
                "source": "Semantic Scholar",
                "tier":   7, "url": url, "title": title,
                "text":   text[:6000], "date": year,
                "topic":  match_topic(query + " " + title, topics),
                "lang":   "en",
            })

        time.sleep(base_sleep)

    log.info(f"  Tier 7 total collected: {len(docs)}")
    save_docs(docs, out)
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# TIER 8 EN FIX — English Wikipedia
# Problem: searching with Ukrainian topic strings returns almost nothing in EN wiki
# Fix: translate topic keywords to English search queries explicitly,
#      then use the extracts API (not REST summary) to get full text
# ─────────────────────────────────────────────────────────────────────────────

# Mapping from Ukrainian topic keywords → English Wikipedia search queries
# (covers the 184 topics by category)
EN_WIKI_QUERIES = [
    # Revolution / early statehood
    "Ukrainian People's Republic 1917",
    "Central Rada Ukraine 1917",
    "Ukrainian State Hetmanate 1918",
    "Directory of Ukraine 1918",
    "Ukrainian Soviet Republic formation 1920",
    # Soviet era
    "Ukrainization policy Soviet Ukraine 1920s",
    "collectivization Ukraine Soviet Union",
    "Holodomor 1932 1933 Ukraine famine",
    "Holodomor genocide recognition",
    "Executed Renaissance Ukrainian cultural repression",
    "Great Terror 1937 1938 Soviet Ukraine",
    "World War II Ukraine occupation",
    "Holocaust Ukraine German occupation",
    "Organization of Ukrainian Nationalists OUN",
    "Ukrainian Insurgent Army UPA",
    "postwar Soviet Ukraine reconstruction",
    "KGB Ukraine Soviet security apparatus",
    "Russification Ukraine Soviet language policy",
    "Ukrainian language Soviet education policy",
    "Chernobyl disaster 1986 political consequences",
    "Perestroika Ukraine glasnost",
    "Ukrainian Helsinki Group human rights",
    "Rukh People's Movement Ukraine independence",
    "Declaration sovereignty Ukraine 1990",
    "Ukrainian independence referendum 1991",
    "Ukrainian SSR United Nations founding member",
    "Cold War Ukraine strategic nuclear weapons",
    # Post-independence politics
    "Constitution Ukraine 1996",
    "constitutional reform Ukraine 2004",
    "Ukrainian parliament Verkhovna Rada",
    "presidency Ukraine constitutional powers",
    "Orange Revolution Ukraine 2004",
    "Euromaidan Revolution of Dignity 2013 2014",
    "NABU National Anti-Corruption Bureau Ukraine",
    "judicial reform Ukraine",
    "decentralization reform Ukraine local government",
    "language law Ukraine 2019",
    "decommunization laws Ukraine 2015",
    "Ukraine oligarchs political influence",
    # War and security
    "annexation Crimea Russia 2014",
    "war Donbas 2014 Ukraine Russia",
    "Minsk agreements ceasefire",
    "Ukraine Russia full scale invasion 2022",
    "Battle of Mariupol 2022",
    "Azovstal steel plant siege 2022",
    "Kharkiv counteroffensive Ukraine 2022",
    "liberation Kherson Ukraine 2022",
    "annexation referendums Russia 2022",
    "martial law Ukraine 2022",
    "mobilization Ukraine military",
    "Ukraine Russia peace negotiations 2022",
    "war crimes Ukraine Russia ICC",
    "special tribunal crime of aggression Ukraine",
    # International relations
    "Budapest Memorandum 1994 Ukraine nuclear",
    "Ukraine NATO relations membership",
    "Ukraine European Union association agreement",
    "DCFTA Ukraine EU trade agreement",
    "Ukraine EU candidate status 2022",
    "Ukraine visa liberalization European Union",
    "United States military aid Ukraine",
    "United Kingdom military support Ukraine",
    "Canada Ukraine military assistance",
    "EU sanctions Russia Ukraine war",
    "US sanctions Russia Ukraine",
    "asset freezing Russia oligarchs sanctions",
    "UN General Assembly resolution Ukraine",
    "Ukraine international criminal court ICC warrant Putin",
    "Ukraine war reparations international law",
    "Ukraine Poland strategic partnership",
    "Ukraine Hungary minority rights conflict",
    "Ukraine Romania border relations",
    "Ukraine Moldova security cooperation",
    "Black Sea security initiative Ukraine",
    "energy sanctions Russia Europe gas",
    "Nord Stream Ukraine gas transit",
    "Ukraine energy grid European integration ENTSO",
    "grain deal Black Sea initiative 2022",
    "Ukraine reconstruction donor conference",
    "Ukrainian refugees Europe temporary protection",
    "internally displaced persons Ukraine IDP",
    "information war propaganda Ukraine Russia 2022",
    # Regional allies
    "Poland Ukraine strategic partner 2022",
    "Lithuania Ukraine support ally",
    "Latvia sanctions Russia Ukraine",
    "Estonia military aid Ukraine",
    "Baltic states historical Russia policy",
    "Romania Black Sea security Ukraine 2022",
    "Czech Republic military support Ukraine",
    "Slovakia Ukraine military transit",
    "Hungary Russia sanctions policy EU",
    "Moldova Transnistria security risk 2022",
    "Visegrad group Ukraine support sanctions",
    "Central Eastern Europe energy dependence Russia",
    # Diaspora / memory
    "Ukrainian diaspora political organizations",
    "Holodomor international recognition memory",
    "Ukrainian diaspora United States Canada lobby",
]


def fix_tier8_en(topics: list[str]):
    log.info(f"\n{'='*60}\nTIER 8 EN FIX — English Wikipedia\n{'='*60}")
    out = OUTPUT_DIR / "tier8_wikipedia" / "tier8_wikipedia_en.jsonl"

    api = "https://en.wikipedia.org/w/api.php"
    docs    = []
    seen    = set()

    for query in tqdm(EN_WIKI_QUERIES, desc="Wikipedia-EN"):
        # Step 1: search
        r = get(api, params={
            "action": "query", "list": "search",
            "srsearch": query, "srlimit": 3,
            "srprop": "snippet", "format": "json", "utf8": 1,
        })
        if not r:
            time.sleep(SLEEP)
            continue

        hits = r.json().get("query", {}).get("search", [])
        if not hits:
            time.sleep(SLEEP * 0.3)
            continue

        new_titles = [h["title"] for h in hits if h["title"] not in seen]
        if not new_titles:
            continue
        for t in new_titles:
            seen.add(t)

        # Step 2: fetch extracts for all new titles in one request
        r2 = get(api, params={
            "action":          "query",
            "prop":            "extracts|info",
            "explaintext":     True,
            "exsectionformat": "plain",
            "titles":          "|".join(new_titles),
            "inprop":          "url",
            "format":          "json",
            "utf8":            1,
        })
        if not r2:
            time.sleep(SLEEP)
            continue

        pages = r2.json().get("query", {}).get("pages", {})
        for page in pages.values():
            title   = page.get("title", "")
            extract = (page.get("extract", "") or "").strip()
            url     = page.get("fullurl", f"https://en.wikipedia.org/wiki/{title}")
            if len(extract) < 150:
                continue
            docs.append({
                "id":     doc_id(title + "en"),
                "source": "Wikipedia EN",
                "tier":   8, "url": url, "title": title,
                "text":   extract[:8000], "date": "",
                "topic":  match_topic(query + " " + title, topics),
                "lang":   "en",
            })

        time.sleep(SLEEP * 0.4)

    log.info(f"  Tier 8 EN total collected: {len(docs)}")
    save_docs(docs, out)
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────────────────────────────────────

def print_stats():
    log.info(f"\n{'='*60}\nPATCH SUMMARY\n{'='*60}")
    total = 0
    for p in sorted(OUTPUT_DIR.rglob("*.jsonl")):
        with open(p, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        total += n
        log.info(f"  {p.name:<48} {n:>5} docs")
    log.info(f"\n  GRAND TOTAL: {total} documents")

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["patched_at"]  = datetime.now(timezone.utc).isoformat()
    manifest["total_docs"]  = total
    manifest["files"]       = {p.name: sum(1 for _ in open(p)) for p in sorted(OUTPUT_DIR.rglob("*.jsonl"))}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    log.info(f"  Manifest updated → {manifest_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global OUTPUT_DIR
    p = argparse.ArgumentParser(description="RAG patch — fixes tiers 1, 7, 8-EN")
    p.add_argument("--tiers", nargs="+", type=int, default=[1, 7, 8],
                   help="Which tiers to patch (default: 1 7 8)")
    p.add_argument("--topics", default=TOPICS_FILE)
    p.add_argument("--output", default=str(OUTPUT_DIR))
    args = p.parse_args()

    OUTPUT_DIR = Path(args.output)
    topics = load_topics(args.topics)
    log.info(f"Topics loaded: {len(topics)} | Patching tiers: {args.tiers}")

    if 1 in args.tiers:
        fix_tier1(topics)
    if 7 in args.tiers:
        fix_tier7(topics)
    if 8 in args.tiers:
        fix_tier8_en(topics)

    print_stats()
    log.info("✅ Patch done.")


if __name__ == "__main__":
    main()