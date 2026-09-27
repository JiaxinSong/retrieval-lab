# 01 — Corpus

The retrieval corpus: PostgreSQL 17 documentation, 224 pages across 21
chapters, split into an evaluated subset and a distractor set.

`scripts/01_fetch_docs.py` → `data/raw/html/` + `data/raw/manifest.json`

---

## What was collected

| | Pages | Chapters |
| --- | --- | --- |
| **Eval** | 49 | 5 |
| **Distractor** | 175 | 16 |
| **Total** | **224** | **21** |

### Eval chapters — the golden dataset queries these

| Chapter | Pages | Content |
| --- | --- | --- |
| 19. Server Configuration | 19 | Every runtime parameter: connections, resource consumption, WAL, replication, query planning, logging, statistics, autovacuum, client defaults, lock management, compatibility, error handling |
| 11. Indexes | 13 | B-tree, Hash, GiST, SP-GiST, GIN, BRIN; multicolumn, unique, partial, expression indexes; operator classes, collations, index-only scans, bitmap scans |
| 28. Monitoring Database Activity | 7 | The `pg_stat_*` views, lock monitoring, `ps` output, progress reporting, dynamic tracing |
| 14. Performance Tips | 6 | `EXPLAIN` output, planner statistics, explicit joins, populating a database, non-durable settings, disk usage |
| 25. Routine Database Maintenance | 4 | Routine vacuuming, dead tuples, transaction ID wraparound, reindexing, log file maintenance |

### Distractor chapters — never queried

| Chapter | Pages | Why it competes |
| --- | --- | --- |
| 9. Functions and Operators | 32 | Pure identifiers — function and operator names throughout |
| 8. Data Types | 22 | Broad vocabulary, overlaps queries and indexing |
| 5. DDL | 16 | Tables, constraints, partitioning; overlaps indexing vocabulary |
| 21. Client Authentication | 16 | `pg_hba.conf`, SSL; overlaps connection settings |
| 18. Server Setup and Operation | 13 | Looks like Server Configuration by name, different content |
| 29. Logical Replication | 13 | Overlaps replication settings |
| 12. Full Text Search | 12 | `tsvector` and GIN indexes — direct overlap with the Indexes chapter |
| 7. Queries | 9 | Joins, CTEs, window functions; overlaps query planning |
| 13. Concurrency Control | 8 | Isolation levels, explicit locking, deadlocks — overlaps lock monitoring |
| 65. Database Physical Storage | 8 | Page layout, TOAST; overlaps disk usage and VACUUM |
| 30. Reliability and the WAL | 7 | Same subject as `runtime-config-wal`, written differently |
| 27. Replication and Failover | 5 | Overlaps replication settings |
| 15. Parallel Query | 5 | Overlaps query planning and cost constants |
| 26. Backup and Restore | 4 | Overlaps maintenance |
| 4. SQL Syntax | 4 | Large vocabulary surface |
| 52. Internals Overview | 1 | Planner architecture; overlaps performance tips |

Page counts are from `manifest.json`. Two entries in `DISTRACTOR_CHAPTERS`,
`partitioning.html` and `explicit-locking.html`, contribute no pages under
their own name: their sections are already reached through the DDL and
Concurrency Control tables of contents, which come earlier in the list, and the
manifest keeps the first chapter that links a page. The content is in the
corpus either way; it is counted under the parent chapter.

Third-level TOC entries (`19.4.1 Memory`, `19.4.2 Disk`, …) are **not**
separate pages — they are in-page anchors on their parent file.
`runtime-config-resource.html` alone contains six subsections. Chunking splits
on those `div.sect2` boundaries, so no content is missed.

---

## Why an eval / distractor split

The evaluation set only asks questions about the five eval chapters. The other
sixteen chapters are never the source of a question.

**A small index makes retrieval artificially easy.** With 355 chunks, a query
about `work_mem` competes against three or four chunks that mention it. The
top-5 is almost guaranteed to contain the answer, reranking has nothing to
rerank past, and the resulting numbers say more about the corpus size than the
architecture.

Distractors restore the difficulty a production-scale index would have.
Current ratio:

| | Chunks |
| --- | --- |
| Eval | 339 |
| Distractor | 954 |
| **Total** | **1,293** |

Nearly 3:1 against.

**The distractors are chosen for overlap, not volume.** `textsearch.html`
discusses GIN indexes; `explicit-locking.html` discusses lock modes while
`monitoring-locks.html` discusses observing them; `wal.html` and
`runtime-config-wal.html` cover the same subject from different angles. These
genuinely enter the candidate pool for eval queries. Padding with unrelated
content would inflate the chunk count without adding difficulty.

**The split keeps evaluation coverage intact.** Expanding the corpus without
expanding the question set normally makes metrics less trustworthy — the same
questions spread across 21 chapters would be too thin. Here the question set
still covers the same five chapters at the same density; only the difficulty
rose.

The split also does concrete work at question-generation time
(`05-questions.md`): candidates are sampled only from eval chunks. For ten of
eighty candidates, a distractor outranked every eval chunk in retrieval. Those
were first going to be discarded, but that filter ranked questions with the
system under test, which biases the test set, so they are kept. The judges
decide instead: if the distractor genuinely answers the question, it becomes
part of the ground truth rather than counting as an intrusion.

Interview framing: *the corpus includes chapters the evaluation set doesn't
query. Those are deliberate distractors — full-text search, MVCC, explicit
locking — topics that share vocabulary with the queried chapters without
containing the answers. Without them, retrieval looks better than it is and
reranking looks worthless.*

---

## Why these five eval chapters

Server Configuration, Indexes, Monitoring, Performance Tips, Routine
Maintenance. Three criteria decided it, and a fourth condition turned out to be
attached to the first.

### 1. Identifier density — the corpus has to exercise the failure mode

These chapters are saturated with exact tokens: `work_mem`, `shared_buffers`,
`autovacuum_vacuum_scale_factor`, `pg_stat_activity`, `effective_cache_size`,
`maintenance_work_mem`, `GIN`, `BRIN`.

Those tokens are precisely where dense vector retrieval underperforms. They
appear rarely in embedding-model training data, so the model has no stable
representation for them. Measured:

| Pair | Cosine similarity |
| --- | --- |
| `"cancel my subscription"` vs `"how to terminate your plan"` | ~0.54 |
| `"work_mem"` vs `"memory allocated per sort operation"` | ~0.29 |
| `"error 42P01"` vs `"relation does not exist"` | ~0.17 |

Paraphrase scores high; identifiers and error codes score low despite meaning
the same thing. Lexical matching handles exactly those cases — and fails
exactly where embeddings succeed, on zero word overlap.

**A corpus that exercises the failure mode hybrid retrieval exists to fix is a
design choice, not an accident.** On pure prose, lexical and vector search
return near-identical results, RRF fusion has nothing to contribute, and the
whole architecture would be unjustifiable — it would look sophisticated and
demonstrate nothing.

### 2. A domain where retrieval results can be judged

Whether a result is correct has to be checkable by eye. The golden dataset is
labelled by LLM judges (`05-questions.md`), but the judges are only trusted
after passing calibration cases verified by hand, and a fixed sample of
accepted questions is printed for a human spot check. Both require knowing the
answer. That is impossible in
a domain you do not know: presented with five plausible-looking chunks about
the C extension interface or collation behaviour, I could not tell which one
answers the question, and neither the calibration cases nor the spot check
would mean anything.

PostgreSQL configuration, indexing, and monitoring are familiar from backend
work.

**Distractor chapters need this property much less.** No question, calibration
case, or spot check is drawn from them; the judges label a distractor chunk
only when it lands in a question's pool. That is what makes it safe to include
something like `functions.html` — far outside the evaluated subject matter,
and useful precisely because it is large and vocabulary-dense.

### 3. Scope that keeps the question set meaningful

A few dozen questions across five chapters gives real coverage. Spread across
all 21 chapters of the corpus, the same set would be too thin to trust — it
would need several times as many questions, and several times the judging
cost.

Scale is also not this project's claim. Indexing the full manual would not
change the answer to "why hybrid retrieval", "how did you evaluate it", or "how
did you pick chunk size". It would only make those answers harder to verify.

### The condition attached to criterion 1

Choosing the corpus for identifier density presumes something that was not
checked at the time: **that the lexical retriever can match an identifier
whole.**

It cannot. Postgres's text-search parser splits on underscores before any
dictionary runs, so `work_mem` is indexed as `work` + `mem`. Worse,
`maintenance_work_mem` indexes as `mainten` + `work` + `mem` — two of three
lexemes shared with `work_mem`, making the two parameters nearly
indistinguishable — and `mem` alone matches every passage about memory, which
is what dense retrieval already does well.

So the lexical path had no comparative advantage on exactly the content the
corpus was selected to provide. Fixing it meant adding a third retriever over a
separate identifier column (`04-retrieval.md`, Bug 3).

Without that retriever, the rationale in criterion 1 would have been **sound in
theory and inert in practice** — the corpus would have exercised a failure mode
the system had no machinery to handle, and the resulting numbers would have
said "hybrid retrieval does not help much here" for the wrong reason.

Worth stating plainly because the reasoning was correct and the outcome was
still nearly wrong. A corpus choice justified by a capability is only as good
as the verification that the capability exists.

---

## How it was collected

### Approach

Fetch once, write to disk, never query the live site during ingest or
evaluation.

**The corpus must be a frozen snapshot.** If pages change between an ingest run
and an evaluation run, the numbers stop being comparable — and comparability is
the entire point of the eval harness. Reproducible evaluation requires a fixed
corpus, not a live data source.

### Why scraping rather than a packaged download

Two alternatives were tried first:

| Option | Outcome |
| --- | --- |
| Official PDF (`postgresql-17-A4.pdf`) | Rejected. PDF extraction loses the section hierarchy that chunk boundaries and metadata depend on, and mangles code blocks and parameter tables — the identifier-dense content this corpus was chosen for |
| Source tarball (`postgresql-17.11.tar.gz`) | Contains only DocBook sources and XSL stylesheets (`stylesheet-html-common.xsl` and friends), no prebuilt HTML. Building it requires `xsltproc` plus the DocBook DTD |

The live HTML is DocBook-generated, so its structure is regular and
machine-readable: `div.sect1` / `div.sect2` for sections, `h2`/`h3` for
headings, `pre.programlisting` for code. That structure is what makes
section-aligned chunking possible, so the HTML is worth the scrape.

### Implementation

Two passes:

1. **Discover.** Fetch each chapter landing page, parse its table of contents
   (`div.toc`, `div.sect1toc`, `dl.toc`), and collect the relative section
   links joined against the chapter URL. Same-page anchors and off-site links
   are skipped. Deduplication uses `dict.fromkeys` rather than a set, to
   preserve document order.

2. **Fetch.** Download each page to `data/raw/html/<name>.html`.

### The manifest

`data/raw/manifest.json` maps each page filename to its chapter and group:

```json
{
  "runtime-config-resource.html": {
    "chapter": "runtime-config.html",
    "group": "eval",
    "url": "https://www.postgresql.org/docs/17/runtime-config-resource.html"
  }
}
```

**Provenance has to be captured at fetch time or it is lost.** Section pages
have no filename convention revealing their parent chapter —
`monitoring-stats.html` gives no clue that it belongs to chapter 28, and
`config-setting.html` none that it belongs to chapter 19. Without the manifest
there is no way to tell an eval chunk from a distractor after the fact.

The chunking step reads this and tags every chunk with its group, which is what
lets question generation sample only eval chunks and lets evaluation report how
often a distractor displaced a correct chunk.

### Operational details

- **Honest user agent** (`retrieval-lab/1.0 (personal study project)`).
  Anonymous scrapers get blocked, and accurate identification is the minimum
  courtesy when hitting a project's docs.
- **0.5s delay** between requests.
- **Cache check before fetch** — a page already on disk is skipped. This makes
  the script safe to re-run after an interruption, and means growing the
  chapter list only downloads what is new. Both corpus expansions re-used every
  previously fetched page.
- **Per-page error handling** — a failed request logs and continues rather than
  aborting the run.

### Layout

```
retrieval-lab/
├── data/
│   ├── raw/
│   │   ├── html/          # 224 fetched pages, frozen
│   │   └── manifest.json  # page -> chapter, group
│   └── processed/
│       └── chunks.json
├── eval/                  # candidates, pools, golden set, judge report
├── scripts/
│   ├── notes/             # these step docs (01-corpus.md ...)
│   ├── 01_fetch_docs.py
│   ├── 02_chunk_docs.py
│   ├── 03_ingest.py
│   ├── 04_hybrid_search.py
│   ├── 05_generate_questions.py
│   └── 06_build_golden.py
├── retrieval.py           # shared retrieval module
├── .env.example
└── README.md
```

`data/raw/` is gitignored — the fetch script is the reproducible artifact, not
several hundred HTML files.

---

## Extending the corpus

Add chapter filenames to `EVAL_CHAPTERS` or `DISTRACTOR_CHAPTERS` and re-run
both scripts. Cached pages are skipped; nothing in retrieval or evaluation
changes.

Adding to `EVAL_CHAPTERS` requires extending the golden dataset to match, or
coverage thins and the metrics stop being trustworthy. Adding to
`DISTRACTOR_CHAPTERS` does not — that is the point of the split.
