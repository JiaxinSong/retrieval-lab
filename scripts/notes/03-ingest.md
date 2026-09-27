# 03 — Ingest

Embedding 1,293 chunks and loading them into pgvector with a vector index and
a lexical index.

`scripts/03_ingest.py` → `retrieval_lab_chunks` table

---

## Results

| Metric | Value |
| --- | --- |
| Rows loaded | 1,293 — 339 eval, 954 distractor |
| Rows missing an embedding | 0 |
| Rows missing a tsvector | 0 |
| HNSW index used by the planner | yes |
| Embedding cost | $0.0104 |

Three retrieval surfaces per row:

| Column | Type | Serves |
| --- | --- | --- |
| `embedding` | `vector(1536)` | Dense semantic search, HNSW |
| `content_tsv` | `tsvector` | Stemmed prose search, GIN |
| `identifiers` | `text[]` | Exact snake_case match, GIN |

The planner choosing the HNSW index at only 1,293 rows is worth noting. At
small row counts Postgres often decides a sequential scan is cheaper, so
`false` there would have been normal rather than a failure. That it picked the
index means HNSW's cost estimate already beats a full scan at this size.

---

## Schema

```sql
CREATE TABLE retrieval_lab_chunks (
    id          SERIAL PRIMARY KEY,
    content     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}',
    embedding   vector(1536),
    identifiers TEXT[] NOT NULL DEFAULT '{}',

    content_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', content)
    ) STORED,

    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### `content_tsv` as a generated column

Lexical full-text search (step 04) reads this column, but nothing in the
ingest path maintains it — Postgres derives it from `content` on every write.

**Keeping the derivation in the database makes drift impossible.** An
application-maintained search vector can fall out of sync with its text: an
`UPDATE content` that forgets to recompute `content_tsv` leaves the lexical
index silently pointing at stale terms. A generated column cannot get this
wrong.

Defining it now rather than in step 04 also means the lexical path works
without a later migration — adding a `tsvector` column and backfilling it over
an existing table is a much more annoying operation than declaring it up front.

### `identifiers` as a plain column

Why this column exists takes three steps: where the problem occurs, why it
reports nothing, and what it costs.

### Where the problem occurs

The `content_tsv` column above is the site of it:

```sql
content_tsv tsvector GENERATED ALWAYS AS (
    to_tsvector('english', content)      -- <- the split happens here
) STORED
```

`to_tsvector` runs two stages internally:

```
input:    "work_mem controls memory"
             |
             v  stage 1: parser -- tokenization
tokens:   [work] [_] [mem] [controls] [memory]
             |
             v  stage 2: dictionary -- stemming, stop words
lexemes:  'work' 'mem' 'control' 'memori'
```

**The underscore is treated as a separator in stage 1.** That is parser
behaviour, not dictionary behaviour — which is why no text-search
configuration fixes it:

```sql
SELECT to_tsvector('english', 'work_mem and pg_stat_activity and maintenance_work_mem');
-- 'activ':6 'mainten':8 'mem':2,10 'pg':4 'stat':5 'work':1,9

SELECT to_tsvector('simple', 'work_mem and pg_stat_activity');
-- 'activity':6 'and':3 'mem':2 'pg':4 'stat':5 'work':1
```

`'simple'` does no stemming at all — note `activity` unstemmed, `and` retained
as a stop word — and splits the identifier anyway. Stage 2 is the only stage a
configuration controls.

This is the default being reasonable for the wrong data. Postgres's full-text
search is built for natural-language prose, where underscores are rare and
treating them as punctuation is correct. In technical documentation they are
part of the identifier, and the default inverts the meaning of the text.

### Why the bug is silent

Nothing in the ingest path reports it. `verify()` checks
`content_tsv IS NULL` and finds zero rows — the column populated fine. The
GIN index built, queries run, results come back.

They are simply the wrong results: a search for `work_mem` matches every
passage about memory, because that is what `mem` means once the identifier is
gone. There is no error, no empty result, no latency symptom. A structural
check ("is the column populated?") passes; only reading the actual lexemes
reveals the problem.

It surfaced two steps later, in the document-frequency filter's diagnostic
output (`04-retrieval.md`, Bug 3), which printed the query's lexemes and made
`work_mem -> ['work', 'mem']` visible. Without that diagnostic it would
plausibly have gone unnoticed until evaluation metrics came back lower than
expected — with no obvious place to look.

### Why this column has to exist

The damage is worse than losing exactness:

- `maintenance_work_mem` indexes as `mainten` + `work` + `mem`, sharing two of
  three lexemes with `work_mem`. The two parameters are nearly
  indistinguishable lexically.
- `mem` alone matches every passage about memory — which is exactly what dense
  retrieval already does well.

**So the lexical path had lost its only comparative advantage, on precisely
the content this corpus was selected for.** The corpus was chosen for
identifier density (see `01-corpus.md`); without whole-identifier matching,
that choice paid off for nothing.

### The fix

Extract identifiers with a regex, outside the `to_tsvector` pipeline
entirely:

```python
IDENTIFIER_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+")

def extract_identifiers(content: str) -> list[str]:
    return sorted({m.group().lower() for m in IDENTIFIER_RE.finditer(content)})
```

This is a bypass, not a correction. `to_tsvector`'s tokenization cannot be
changed — there is no configuration option for underscore handling — so the
only route is a second surface that never goes through it.

Verified against a known case before trusting it:

```sql
SELECT DISTINCT unnest(regexp_matches(
  'Set work_mem and maintenance_work_mem, check pg_stat_activity and autovacuum_vacuum_scale_factor',
  '[a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+', 'g'));
-- maintenance_work_mem
-- work_mem
-- autovacuum_vacuum_scale_factor
-- pg_stat_activity
```

**Why a plain column rather than generated.** Generated columns cannot contain
subqueries, and aggregating `regexp_matches` into an array requires one. A
trigger would work, but it puts the extraction rule somewhere harder to find
than the ingest script that owns it. The cost is that this column *can* drift
from `content` — unlike `content_tsv` — so the rebuild-not-migrate strategy
matters more than it would otherwise.

Identifiers are lowercased at write time. The docs are inconsistent —
`SHARED_BUFFERS` in a heading, `shared_buffers` in prose — so matching has to
be case-insensitive on both sides, and normalizing once at ingest is cheaper
than at every query.

### Four indexes

```sql
CREATE INDEX ... USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ... USING gin  (content_tsv);
CREATE INDEX ... USING gin  (identifiers);
CREATE INDEX ... USING gin  (metadata jsonb_path_ops);
```

**HNSW with `vector_cosine_ops`.** The opclass must match the distance
operator used at query time: `vector_cosine_ops` serves `<=>`,
`vector_l2_ops` serves `<->`. A mismatch is the worst kind of bug — the index
exists, the query returns correct results, and the planner quietly ignores the
index and scans the table. No error, just latency.

**GIN over `content_tsv`.** The inverted index `ts_rank` reads. Same structure
PostgreSQL's own full-text search uses.

**GIN over `identifiers`.** Supports the `&&` array-overlap operator, which is
how the identifier retriever finds candidates.

**GIN over `metadata` with `jsonb_path_ops`.** The eval harness filters on
`metadata->>'group'` and `metadata->>'chapter'`. `jsonb_path_ops` is smaller
and faster than the default `jsonb_ops` for containment queries, which is all
this needs — it indexes paths-to-values rather than every key and value
separately.

### Why HNSW rather than IVFFlat

HNSW requires no training pass, so the index can be created on an empty table
and handles incremental inserts naturally. IVFFlat computes its cluster
centroids once at build time, so it must be built on representative data — and
rebuilt when the corpus shifts.

For a corpus that grows, HNSW is the practical choice. The cost is higher
memory use and slower index build, neither of which matters at this scale.

### Why the indexes are created before the load

Building HNSW after inserting would be faster: bulk graph construction beats
per-insert maintenance. The difference is irrelevant at 1,293 rows, and the
simpler code path is worth more than the seconds saved. At a hundred thousand
rows this would flip.

---

## Ingest strategy

### Full rebuild, not incremental

```python
await conn.execute(f"TRUNCATE {TABLE} RESTART IDENTITY")
```

Chunking parameters are an open experiment — chunk size 256 / 512 / 1024 is a
Tier 2 comparison. When chunk boundaries shift, every chunk on a page changes,
so there is no stable key to reconcile an incremental update against.
Attempting one leaves orphaned rows from the previous configuration, which
then compete in retrieval as invisible duplicates.

Rebuilding 1,293 rows takes about 40 seconds, most of it embedding API
latency. Not worth the complexity of getting incremental right.

`RESTART IDENTITY` resets the sequence, so chunk IDs are stable across runs
with identical input — which matters because the golden dataset stores ground
truth *by ID*. Without it, re-ingesting would silently invalidate every
ground-truth label: the harness would keep running and report Recall@5 of zero,
indistinguishable from a retrieval regression.

This handles re-ingestion but not re-chunking. Changing `MAX_TOKENS` produces
different chunks with different ids, which is why the golden set also stores
section id and a content prefix for every ground-truth chunk
(`05-questions.md`).

### Batched embedding

```python
EMBED_BATCH = 100   # the API accepts up to 2048
```

Batching is not optional at this volume: one request with 100 texts versus 100
requests with one text each is an order of magnitude difference in wall time.

The batch is deliberately well under the API limit. A failed request costs the
whole batch, so smaller batches localize the damage — and combined with the
retry below, a transient failure re-sends 100 chunks rather than 2,000.

### Retry with backoff

```python
for attempt in range(MAX_RETRIES):
    try:
        return await client.embeddings.create(...)
    except Exception:
        await asyncio.sleep(2 ** attempt)
```

Rate limits and transient 5xx responses are routine at this volume. Without a
retry, one failure aborts the run — and since the script truncates first, that
means re-embedding everything and paying for it again.

### `executemany` over a loop

```python
await conn.executemany(
    f"INSERT INTO {TABLE} (content, metadata, embedding) VALUES ($1, $2, $3::vector)",
    rows,
)
```

One round trip per batch instead of one per row. The `$1 $2 $3` placeholders
are asyncpg's syntax (not `?`, not `%s`), and being parameterized they also
close off SQL injection.

### The pgvector string cast

```python
def to_pgvector(embedding: list[float]) -> str:
    return "[" + ",".join(map(str, embedding)) + "]"
```

`vector` is a type the pgvector extension adds, not a Postgres native, so
asyncpg has no codec for it. The value travels as a string literal and is cast
with `::vector` in the SQL. Slightly ugly, unavoidable without registering a
custom codec that would not earn its complexity here.

---

## Term statistics

A second table, built after the load:

```sql
CREATE TABLE retrieval_lab_term_stats AS
SELECT word, ndoc, nentry
FROM ts_stat('SELECT content_tsv FROM retrieval_lab_chunks');

CREATE INDEX ON retrieval_lab_term_stats (word);
```

`ndoc` is the number of chunks containing each lexeme. Nothing new is
computed — `ts_stat` reads the `content_tsv` column that already exists.

### Why it is needed

**Postgres's `ts_rank` has no IDF component.** It scores by term frequency and
position within the document, so a query term appearing in 40% of the corpus
contributes as much to the score as one appearing in 0.3%.

That is survivable with AND semantics, where every term must match. It is not
survivable with OR — and OR is required, because AND returns nothing on
natural-language questions (see `04-retrieval.md`). With OR, a common stem
matches almost everything and drags the ranking with it.

This was measured, not anticipated. For the query *"how do I see which queries
are blocked by pg_locks"*:

| Retrieval | Rank of the correct chunk |
| --- | --- |
| vector only | 1st |
| RRF fused | 4th |

RRF made it worse. The lexical retriever had ranked on `queri` — a stem in
roughly 40% of a PostgreSQL corpus — putting a distractor section about
session-information functions at the top. RRF then saw that chunk in both
result lists and applied its cross-retriever agreement bonus, promoting it to
1st.

**RRF rewards agreement, and agreement is only evidence when both retrievers
are ranking on signal.** That is the general lesson: fusion does not filter
noise, it amplifies whatever the inputs agree on.

### Why frequency filtering rather than real BM25

Real BM25 handles this with inverse document frequency — continuous weighting
by term rarity. Dropping terms above a frequency threshold is blunt by
comparison: binary keep-or-drop instead of weighting.

The trade is deliberate. Implementing BM25 properly means reimplementing the
scorer on top of `ts_stat` output — feasible, but roughly double the work for
a refinement of the same idea. The threshold approach addresses the specific
observed failure, and the threshold itself becomes a parameter the eval
harness tunes rather than a number picked by guess.

Being able to state that distinction accurately matters more than the
sophistication: *"I used Postgres's `ts_rank`, which is TF-IDF-family and has
no IDF term, and substituted document-frequency filtering for the missing
half"* is a true sentence. *"I implemented BM25"* would not be.

### Where it runs

At the end of ingest, after the insert loop — it depends on `content_tsv`
being populated, so it cannot be part of schema creation.

The script also prints the most frequent lexemes, which is what the threshold
gets calibrated against:

```
most frequent:
  queri                    512 chunks  (39.6%)
  set                      445 chunks  (34.4%)
  ...
```

---

## Verification

The script checks four things after loading, and each corresponds to a failure
mode that produces no error message.

| Check | Failure it catches |
| --- | --- |
| Row count and group split | The insert silently dropped a batch |
| `embedding IS NULL` count | A row landed without its vector |
| `content_tsv IS NULL` count | The generated column did not populate — lexical search would return nothing |
| Empty `identifiers` count, mean, max | The regex matched nothing, or matched everything |
| Most common identifiers | The regex caught real parameter names rather than incidental underscore text |
| `EXPLAIN` mentions the HNSW index | The index exists but the planner ignores it |

The identifier checks exist because a regex failure is invisible in
aggregate. An empty array on every row and a plausible-looking array on every
row both load without error; only looking at what was extracted distinguishes
them:

```sql
SELECT ident, COUNT(*) FROM retrieval_lab_chunks, unnest(identifiers) AS ident
GROUP BY 1 ORDER BY 2 DESC LIMIT 8;
```

The `EXPLAIN` check is the one worth keeping. An unused vector index is the
canonical silent failure in this kind of system: results stay correct, latency
grows with corpus size, and nothing in the logs says why.

```python
plan = await conn.fetch(f"""
    EXPLAIN (COSTS OFF)
    SELECT id FROM {TABLE}
    ORDER BY embedding <=> (SELECT embedding FROM {TABLE} LIMIT 1)
    LIMIT 5
""")
used = f"{TABLE}_embedding_idx" in "\n".join(r["QUERY PLAN"] for r in plan)
```

---

## The `ORDER BY` trap

This is the detail most worth carrying forward, and it does not live in the
ingest script — it constrains every query written against this table.

```sql
-- Uses the HNSW index
SELECT id, 1 - (embedding <=> $1::vector) AS similarity
FROM retrieval_lab_chunks
ORDER BY embedding <=> $1::vector
LIMIT 5;

-- Does NOT use the index
SELECT id, 1 - (embedding <=> $1::vector) AS similarity
FROM retrieval_lab_chunks
ORDER BY similarity DESC
LIMIT 5;
```

The planner only recognizes the index-scan pattern in the form
`ORDER BY <indexed column> <operator> <constant>`. Sorting on a SELECT alias
forces it to compute the distance for every row first, then sort — a
sequential scan wearing the right answer.

Both queries return identical results. One is O(log n), the other O(n). The
only signal is latency, so the distance expression gets repeated in `ORDER BY`
even though it reads as duplication.

---

## First retrieval results

A sanity check before building the eval harness — five queries, vector search
only, counting how many distractor chunks appear in the top-10.

| Query | Distractors in top-10 |
| --- | --- |
| Is `work_mem` a per-connection limit? | 3 |
| Why would autovacuum fall behind on a large table? | 0 |
| Which index type should I use for full-text search? | **8** |
| How do I find which queries are waiting on locks? | 3 |
| What does `shared_buffers` control? | 0 |

Mean 2.8 — the distractors are competing, which is what they are for.

### What this revealed

**The full-text-search question was invalid, not the retrieval.** All five top
results came from chapter 12, a distractor chapter — which contains a section
titled *"12.9. Preferred Index Types for Text Search."* That section genuinely
is the best answer to the question. Retrieval worked perfectly; the question
was drawing on a chapter the question set does not cover.

This first produced a rule for the golden dataset: if a distractor outranks
every eval chunk, discard the question. The rule was later dropped
(`05-questions.md`). It filters questions by how the system under test ranks
them, which biases the test set toward what the system already does well. The
golden set handles the case differently: distractor chunks in a question's
pool are judged like any other, and one judged a correct answer becomes part
of the ground truth rather than an intrusion.

It also corrects a framing. The eval/distractor split is not a judgment about
which chapters matter — it is a statement about where the question set draws
from. Those are related but not equivalent, and conflating them is exactly how
the invalid question got written.

**Scores are low and poorly separated.** Top-1 similarity ranged 0.575-0.665.
On the lock query the top five spanned 0.575 down to 0.510 — a 0.065 spread in
which `19.17. Developer Options` outranked `19.12. Lock Management`, which is
plainly wrong.

Dense retrieval has lost discriminative power in that band. This is precisely
what a cross-encoder reranker addresses: it encodes query and document
*together*, so it can judge whether a passage answers the question rather than
whether it sits in the same semantic neighbourhood.

**One section monopolizes the top-k.** `19.4.1. Memory` occupied four of five
slots on two different queries — the same section split across multiple chunks.
Effective diversity in the top-5 is far lower than it appears, which inflates
apparent coverage while narrowing actual coverage.

### Why none of these numbers mean anything yet

`0.602` is neither good nor bad in isolation. There is no ground truth, so
there is no way to tell whether the top-ranked `19.4.1. Memory` chunk is *the
right one* — that section has eight chunks covering a dozen parameters. The
`work_mem` description starts at the end of chunk 1030 and continues into 1031,
and the sentence that answers the question — *"each operation generally being
allowed to use as much memory as this value specifies"* — appears in both,
through the 64-token overlap. Two chunks are correct; six are not; the metadata
is identical across all eight.

Which is the whole argument for the golden dataset. These observations are
hypotheses about where retrieval is weak; the eval harness is what turns them
into measurements.

The three weaknesses above are also, precisely, what hybrid search and
reranking exist to fix. Their presence at this stage is the expected outcome —
if dense retrieval were already clean, the next three steps would have no
reason to exist.
