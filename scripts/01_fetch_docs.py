"""
Fetch PostgreSQL documentation HTML to local disk.

Why fetch and freeze rather than query the live site: the corpus must be a
fixed snapshot. If pages change between an ingest run and an evaluation run,
the numbers stop being comparable -- and comparability is the entire point of
the eval harness.

The chapter list is split into two groups, and the split is deliberate.

EVAL_CHAPTERS are the chapters the golden dataset draws its questions from.
They are dense in identifiers (work_mem, pg_stat_activity, GIN,
autovacuum_vacuum_scale_factor), which is precisely where dense vector
retrieval underperforms and lexical search earns its place. A corpus that exercises the
failure mode hybrid search exists to fix is a design choice, not an accident.

DISTRACTOR_CHAPTERS are never queried. They exist to make retrieval hard:
topics that share vocabulary with the queried chapters -- locking, WAL,
tsvector, partitioning -- without containing the answers. A small index makes
retrieval artificially easy and makes reranking look worthless, because there
is nothing to rerank past. Distractors restore the difficulty a
production-scale index would have, without diluting the coverage of the
question set, which stays concentrated on five chapters.

A manifest records which chapter and group each page came from. Section pages
have no filename convention that reveals their parent chapter, so provenance
has to be captured at fetch time or it is lost.
"""

import json
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://www.postgresql.org/docs/17/"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "html"
MANIFEST_PATH = ROOT / "data" / "raw" / "manifest.json"

# Chapters the golden dataset queries. Identifier-dense, and a domain where
# retrieval results can be checked by eye when verifying the judges' labels.
EVAL_CHAPTERS = [
    "runtime-config.html",     # 19. Server Configuration
    "performance-tips.html",   # 14. Performance Tips
    "indexes.html",            # 11. Indexes
    "maintenance.html",        # 25. Routine Database Maintenance
    "monitoring.html",         # 28. Monitoring Database Activity
]

# Never queried. Present to compete for the top-k slots.
DISTRACTOR_CHAPTERS = [
    "mvcc.html",               # 13. Concurrency Control -- isolation, locking
    "wal.html",                # 30. Reliability and the WAL
    "high-availability.html",  # 27. Replication and failover
    "queries.html",            # 7.  Queries -- joins, CTEs, window functions
    "ddl.html",                # 5.  DDL -- tables, constraints, partitioning
    "textsearch.html",         # 12. Full Text Search -- tsvector, GIN
    "backup.html",             # 26. Backup and Restore
    "functions.html",          # 9.  Functions and Operators -- huge, many pages
    "datatype.html",           # 8.  Data Types
    "client-authentication.html",  # 21. Client Authentication -- pg_hba, SSL
    "sql-syntax.html",         # 4.  SQL Syntax
    "runtime.html",            # 18. Server Setup and Operation
    "logical-replication.html",  # 29. Logical Replication
    "parallel-query.html",     # 15. Parallel Query
    "partitioning.html",       # 5.11 -- may redirect into ddl, harmless if so
    "explicit-locking.html",   # 13.3 -- lock modes, deadlocks
    "storage.html",            # 65. Database Physical Storage
    "planner-optimizer.html",  # 52. Overview of PostgreSQL Internals
]

HEADERS = {
    # Identify the client honestly. Anonymous scrapers get blocked, and
    # accurate identification is the minimum courtesy when hitting a project's
    # documentation.
    "User-Agent": "retrieval-lab/1.0 (personal study project)"
}

REQUEST_DELAY = 0.5  # seconds between requests


def fetch(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"  failed: {url} -- {type(e).__name__}")
        return None


def find_section_links(chapter_html: str, chapter_url: str) -> list[str]:
    """
    Extract section page URLs from a chapter's table of contents.

    The PostgreSQL docs put the TOC in a div with class "toc" (or "sect1toc"
    on some pages). Links there are relative, so they are joined against the
    chapter URL.
    """
    soup = BeautifulSoup(chapter_html, "lxml")

    links: list[str] = []
    for toc in soup.select("div.toc, div.sect1toc, dl.toc"):
        for a in toc.select("a[href]"):
            href = a["href"]
            # Skip same-page anchors and anything off-site.
            if href.startswith("#") or href.startswith("http"):
                continue
            if not href.endswith(".html"):
                continue
            links.append(urljoin(chapter_url, href))

    # dict.fromkeys dedupes while preserving document order; a set would not.
    return list(dict.fromkeys(links))


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # page filename -> {chapter, group, url}
    manifest: dict[str, dict[str, str]] = {}

    for chapter, group in [
        *((c, "eval") for c in EVAL_CHAPTERS),
        *((c, "distractor") for c in DISTRACTOR_CHAPTERS),
    ]:
        url = urljoin(BASE, chapter)
        print(f"chapter: {chapter:26} [{group}]")

        html = fetch(url)
        time.sleep(REQUEST_DELAY)
        if html is None:
            continue

        # Keep the chapter page itself; its intro text is often substantive.
        pages = [url, *find_section_links(html, url)]
        print(f"  {len(pages) - 1} section pages")

        for page_url in pages:
            name = page_url.rstrip("/").split("/")[-1]
            # First chapter wins on the rare page linked from two chapters.
            manifest.setdefault(
                name, {"chapter": chapter, "group": group, "url": page_url}
            )

    print(f"\n{len(manifest)} pages to fetch\n")

    saved = 0
    for i, (name, info) in enumerate(manifest.items(), 1):
        path = RAW_DIR / name

        # Caching makes the script safe to re-run after an interruption, and
        # keeps previously fetched chapters from being re-downloaded when the
        # chapter list grows.
        if path.exists():
            print(f"[{i}/{len(manifest)}] cached  {name}")
            saved += 1
            continue

        html = fetch(info["url"])
        time.sleep(REQUEST_DELAY)
        if html is None:
            continue

        path.write_text(html, encoding="utf-8")
        print(f"[{i}/{len(manifest)}] saved   {name}  ({len(html) // 1024}KB)")
        saved += 1

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    n_eval = sum(1 for v in manifest.values() if v["group"] == "eval")
    print(f"\n{saved} pages in {RAW_DIR}")
    print(f"manifest -> {MANIFEST_PATH}")
    print(f"  {n_eval} eval pages, {len(manifest) - n_eval} distractor pages")


main()