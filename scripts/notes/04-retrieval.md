# 04 — Hybrid Retrieval

Three retrievers in parallel, fused with Reciprocal Rank Fusion, then capped
for section diversity.

`retrieval.py` (module) · `scripts/04_hybrid_search.py` (demo)

All retrieval logic lives in `retrieval.py` at the project root; the script
only exercises it and prints diagnostics. That split matters for the eval
harness specifically — if the demo and the evaluation held separate copies,
drift between them would surface as a configuration difference between
measured rows, and the comparison would be meaningless.

---

## The architecture

```
query
  ├─→ vector      (HNSW over embeddings)      → top-20
  ├─→ lexical     (GIN over tsvector, OR)     → top-20
  └─→ identifier  (GIN over text[], exact)    → top-20
                    ↓
              RRF fusion
                    ↓
     cross-encoder rerank (Cohere, top-40)
                    ↓
           section cap (2/section)
                    ↓
                 top-5
```

The order of the last two stages was originally reversed — see *Cap before or
after reranking* below.

All three run concurrently via `asyncio.gather` — independent I/O, and
awaiting them in sequence would add the embedding round trip to the other two
for nothing.

---

## Why three retrievers

**The design started with two.** Dense vector search plus lexical search is
the standard hybrid setup, and it is what the first working version
implemented. The third retriever was added after measurement showed the
lexical path could not match identifiers at all — Postgres's text-search
parser splits on underscores, so `work_mem` never reaches the index as
`work_mem`.

That mattered more than it would on another corpus. This one was chosen
specifically for identifier density (`01-corpus.md`), so the gap sat exactly
where the corpus was supposed to demonstrate value. The full diagnosis is Bug
3 below; this section describes where the architecture ended up.

Each retriever fails where the others work.

**Dense vector search** handles paraphrase. A query sharing no words with the
answer still matches, because the comparison happens in semantic space. It is
unreliable on rare exact tokens — `work_mem`, `pg_stat_activity`, error codes
— because those appear too rarely in embedding training data for the model to
hold a stable representation.

| Pair | Cosine similarity |
| --- | --- |
| `"cancel my subscription"` vs `"how to terminate your plan"` | ~0.54 |
| `"work_mem"` vs `"memory allocated per sort operation"` | ~0.29 |
| `"error 42P01"` vs `"relation does not exist"` | ~0.17 |

**Lexical search** matches prose after stemming, so `updates` finds `update`.
It fails completely on zero word overlap — and, as it turns out, it cannot see
identifiers at all.

**Identifier search** exists because of that last point. It matches
snake_case tokens whole, which is the one thing neither of the others can do.

### Candidate-set overlap

How much fusion has to contribute, measured across the demo queries:

| Query | vector | lexical | identifier | in all | union |
| --- | --- | --- | --- | --- | --- |
| `work_mem` | 20 | 19 | 10 | 6 | 31 |
| `work_mem` vs `maintenance_work_mem` | 20 | 20 | 16 | 8 | 32 |
| `Is work_mem a per-connection limit?` | 20 | 20 | 10 | 5 | 33 |
| `why does the database slow down after many updates` | 20 | 20 | **0** | 0 | **40** |
| `what does autovacuum_vacuum_scale_factor control` | 20 | 20 | 2 | 2 | 27 |
| `preventing transaction id wraparound` | 20 | 20 | 0 | 8 | 32 |

The paraphrase query is the extreme case: zero overlap between vector and
lexical, no identifiers at all, 40 chunks in the union. The retrievers found
entirely disjoint sets. That is the case fusion exists for.

Note the identifier column is 0 on two queries. RRF needs no adjustment for a
retriever that returns nothing — it sums over whatever contributed.

---

## Terminology: this is not BM25

Postgres's `ts_rank` is a TF-IDF variant, not BM25. It lacks three things BM25
has:

| BM25 feature | In `ts_rank`? |
| --- | --- |
| Term-frequency saturation | No |
| Document-length normalization | No |
| **IDF weighting** | **No** |

The third omission caused a measured regression (Bug 2 below).

Stating this accurately is worth more than claiming otherwise. *"I used
Postgres's `ts_rank`, which is TF-IDF-family with no IDF term, and substituted
document-frequency filtering for the missing half"* is true. *"I implemented
BM25"* would not be.

---

## Bug 1 — AND semantics killed the lexical path

The first implementation used `websearch_to_tsquery`, which combines terms
with AND:

```sql
SELECT websearch_to_tsquery('english', 'Is work_mem a per-connection limit?');
-- 'work_mem' & 'per' & 'connect' & 'limit'
```

Three of five test queries returned **zero rows**. A chunk has to contain every
term to match, and natural-language phrasing almost guarantees one is absent.

The lexical retriever was dead on anything phrased as a sentence — which is
most real queries.

### Fix: OR semantics

```python
tsquery = " | ".join(terms)
```

OR is also what BM25 actually does. It scores by how many query terms a
document contains and how rare those terms are; it never requires all of them.

Terms come from `to_tsvector`, not a whitespace split:

```sql
SELECT lexeme FROM unnest(to_tsvector('english', $1)) AS t(lexeme)
```

**This matters.** Stemming and stop-word removal must use the same `'english'`
configuration the indexed column was built with. Splitting on whitespace would
leave `updates` unstemmed, and it would never match the indexed `updat`.

The terms are then joined with `|` into a tsquery — how that string reaches
Postgres turned out to matter too; see Bug 4.

---

## Bug 2 — OR semantics broke RRF

Fixing AND exposed something worse. For *"how do I see which queries are
blocked by pg_locks"*:

| Retrieval | Rank of the correct chunk |
| --- | --- |
| vector only | **1st** |
| RRF fused | **4th** |

Fusion made it worse.

### Root cause

The query tokenizes to `'see' | 'queri' | 'block' | 'pg_lock'`.

`queri` appears in roughly 40% of a PostgreSQL corpus. With no IDF weighting
it contributes as much to `ts_rank` as a term appearing in 0.3%. The lexical
top-5 filled with sections that merely mention "query" often.

Then RRF did what RRF does: it saw a distractor chunk in *both* result lists
and applied its cross-retriever agreement bonus.

**RRF rewards agreement, and agreement is only evidence when every retriever
is ranking on signal.** Fusion does not filter noise — it amplifies whatever
the inputs agree on. This is the single most useful thing this project has
taught me about hybrid retrieval, and it is in no tutorial.

### Fix: document-frequency filtering

`ts_stat` materializes per-lexeme document frequency at ingest
(see `03-ingest.md`). Query terms above a threshold are dropped:

```python
MAX_DOC_FRACTION = 0.15   # drop lexemes in >15% of chunks
```

```sql
SELECT t.lexeme, COALESCE(s.ndoc, 0) AS ndoc
FROM unnest(to_tsvector('english', $1)) AS t(lexeme)
LEFT JOIN retrieval_lab_term_stats s ON s.word = t.lexeme
```

One round trip does tokenization and filtering together. `LEFT JOIN` with
`COALESCE(ndoc, 0)` keeps terms absent from the corpus — a typo has no stats
row, and keeping it is harmless since it matches nothing.

**Fallback when filtering removes everything:**

```python
terms = kept or dropped
```

A query made entirely of common words — *"how do I set this up"* — would
otherwise return nothing, which is worse than a noisy result.

Observed on the demo queries:

| Query | kept | dropped |
| --- | --- | --- |
| `work_mem` | `mem` | `work` |
| `why does the database slow down after many updates` | `mani`, `slow`, `updat` | `databas` |
| `how do I see which queries are blocked by pg_locks` | `lock`, `block` | `see`, `queri`, `pg` |

### Why not implement real BM25

Continuous IDF weighting is the correct fix; threshold filtering is binary
keep-or-drop. The trade is deliberate: proper BM25 means reimplementing the
scorer over `ts_stat` output, roughly double the work for a refinement of the
same idea.

The threshold also becomes a parameter the eval harness tunes — 0.10 / 0.15 /
0.25 — rather than a number picked by guess. That is a better position than an
untuned exact implementation.

---

## Bug 3 — the lexical path cannot see identifiers

The `kept`/`dropped` table above contains the real discovery:

```
'work_mem'   →  kept: ['mem'],  dropped: ['work']
'pg_locks'   →  kept: ['lock'], dropped: ['pg']
```

`work_mem` never reaches the index as `work_mem`. **Postgres's text-search
parser splits on underscores at the tokenizer level**, before any dictionary
or stemming:

```sql
SELECT to_tsvector('english', 'work_mem and pg_stat_activity and maintenance_work_mem');
-- 'activ':6 'mainten':8 'mem':2,10 'pg':4 'stat':5 'work':1,9

SELECT to_tsvector('simple', 'work_mem and pg_stat_activity');
-- 'activity':6 'and':3 'mem':2 'pg':4 'stat':5 'work':1
```

`'simple'` does no stemming and splits it anyway — the split is upstream of the
dictionary, so no configuration change fixes it.

### Why this is worse than it looks

Note `'mem':2,10` in the first output: `work_mem` and `maintenance_work_mem`
both contributed. They share two of three lexemes and are nearly
indistinguishable lexically.

And `mem` on its own matches every passage about memory — which is precisely
what dense retrieval already does well.

**The lexical retriever had lost its only comparative advantage, on exactly
the content this corpus was selected for.** The corpus was chosen for
identifier density (`01-corpus.md`). Without whole-identifier matching, that
choice paid off for nothing, and df filtering made it worse by dropping the
common half of each split identifier.

### Fix: a third retriever

Identifiers are extracted with a regex at ingest and stored in a `text[]`
column with a GIN index (`03-ingest.md`). The retriever matches on array
overlap:

```sql
SELECT id, content, metadata,
       cardinality(ARRAY(
           SELECT unnest(identifiers) INTERSECT SELECT unnest($1::text[])
       ))::float AS matched,
       COALESCE((
           SELECT SUM(regexp_count(content, q.ident, 1, 'i'))
           FROM unnest($1::text[]) AS q(ident)
       ), 0) AS mentions
FROM retrieval_lab_chunks
WHERE identifiers && $1::text[]
ORDER BY matched DESC, mentions DESC
```

The same regex runs on both sides — ingest and query. If they diverge, the
query extracts identifiers the index does not contain.

### The tiebreak was wrong the first time

Most queries carry a single identifier, so every match ties on `matched` and
**the tiebreak decides the entire ordering.**

The first version broke the tie by *total identifier count ascending*, on the
theory that a focused chunk should beat a summary table listing forty
parameters. It inverted the ranking instead. For `pg_locks`:

```
1 D  13.2.3. Serializable Isolation Level    ← mentions pg_locks once
2 D  13.3. Explicit Locking
3    27.3. Viewing Locks                     ← the section documenting it
```

The section that documents `pg_locks` naturally also mentions
`pg_stat_activity` and `pg_blocking_pids`, so it carried more identifiers and
was penalized. A section citing the view in passing carried one and won.

**Mention frequency is the signal that actually separates them.** A section
about `pg_locks` names it a dozen times; a section citing it names it once.
After the fix, the identifier top-5 for `work_mem` is entirely eval chunks
where it previously included `18.4.4. Linux Memory Overcommit` — a section
that mentions `work_mem` once.

`regexp_count` requires PostgreSQL 15+.

### …and wrong the second time

The mention count was a substring count. `work_mem` is a substring of
`maintenance_work_mem` and `autovacuum_work_mem`, so chunks documenting those
parameters were credited with `work_mem` mentions they do not contain:

| Chunk | Substring `work_mem` | Whole-word `work_mem` |
| --- | --- | --- |
| 1030 | 2 | 2 |
| 1031 | 10 | 8 |
| 1032 | 5 | **1** |

Chunk 1032 — the tail of `maintenance_work_mem` plus `autovacuum_work_mem` —
was scored five times for a term it uses once.

Fix: count on word boundaries. In Postgres regex `\m` and `\M` mark the start
and end of a word, and underscore counts as a word character, so
`\mwork_mem\M` does not match inside `maintenance_work_mem`:

```sql
SUM(regexp_count(content, '\m' || q.ident || '\M', 1, 'i'))
```

**Same failure class as Bug 3.** The tokenizer treated an identifier as a
sequence of words; the tiebreak treated it as a sequence of characters. Both
fail for the same reason — an identifier is a unit, and every layer that
touches one has to treat it as a unit.

---

## Bug 4 — the query was stemmed twice

Found in a review pass, after retrieval had been declared frozen. The freeze
covers tuning; a defect with no measurement needed to justify fixing it is a
different thing.

The lexical path stems query terms once, correctly, via `to_tsvector`. It then
passed the resulting lexemes to `to_tsquery('english', ...)` — which stems them
again. That would be harmless if stemming were idempotent. Snowball English,
the stemmer behind Postgres's `'english'` configuration, is not:

| Word | Stemmed once (matches the index) | Stemmed twice (what the query used) |
| --- | --- | --- |
| database | `databas` | `databa` |
| increase | `increas` | `increa` |
| decrease | `decreas` | `decrea` |
| release | `releas` | `relea` |
| update | `updat` | `updat` |

Any stem ending in a vowel plus `s` loses the `s` on a second pass. The index
holds `databas`; the query asked for `databa`, which exists nowhere.

Measured against a local Postgres instance loaded through the real ingest
script:

```
'database'   old path: to_tsquery -> 'databa'    matches 0
             new path: literal    -> 'databas'   matches 3
```

**Same failure signature as every other bug in this layer**: no error, no empty
result set when other terms still match, just a common word silently missing
from every query that contains it. *Increase*, *decrease*, *database*, and
*release* are ordinary vocabulary in a database manual.

### Fix: build the tsquery as a literal

The lexemes are already in index form, so they must reach the index unchanged.
The query is built as a tsquery literal and cast, never passed through a text
search parser again:

```python
tsquery = " | ".join("'" + term.replace("\\", "\\\\").replace("'", "''") + "'" for term in terms)
# ... WHERE content_tsv @@ $1::tsquery
```

Quote characters are doubled and backslashes escaped, per tsquery input syntax.

### The pattern across Bugs 3 and 4

Both come from a value crossing a boundary between two layers that normalize
text differently. In Bug 3, the tokenizer disagreed with the data about what a
word is. In Bug 4, the query path normalized something that had already been
normalized. The general rule: **normalize exactly once, and pass normalized
values across boundaries in a form the other side will not re-interpret.**

---

## RRF

```
score(doc) = Σ  1 / (k + rank_in_retriever_i)
```

A chunk ranked 2nd by one retriever and 5th by another:

```
1/(60+2) + 1/(60+5) = 0.01613 + 0.01538 = 0.03151
```

A chunk ranked 1st by one retriever alone:

```
1/(60+1) = 0.01639
```

**Two moderate signals that agree outrank one strong signal standing alone.** A
chunk absent from a list contributes nothing from it — there is no penalty
term, which is why the accumulator is a dict and why going from two retrievers
to three required no change to the formula:

```python
def fuse_rrf(*hit_lists: list[Hit], k: int = RRF_K) -> list[Hit]:
```

### Why not weighted score fusion

```
final = 0.7 × cosine + 0.3 × ts_rank        # don't
```

**Incompatible scales.** Cosine similarity is bounded in [0, 1]. `ts_rank` has
no upper bound. The identifier retriever's `matched` is a small integer. Adding
them directly lets whichever has the largest range dominate.

**Unstable normalization.** Min-max normalization depends on the current result
set, so the same chunk gets a different normalized score depending on what else
came back. Scores stop being comparable across queries.

**Different distribution shapes.** Cosine similarities cluster in a narrow band
— 0.50 to 0.67 across the top-5 on this corpus. `ts_rank` is long-tailed.
Normalizing both to [0, 1] stretches them differently, so the weights no longer
mean what they appear to.

RRF sidesteps all three by using only rank position. One hyperparameter instead
of per-corpus weight tuning — and with three retrievers, instead of two weights
to tune jointly.

### What `k` does

| k | Rank 1 | Rank 2 | Effect |
| --- | --- | --- | --- |
| 1 | 0.500 | 0.333 | Collapses to "whatever one retriever put first" |
| 60 | 0.0164 | 0.0161 | Near-equal; agreement dominates |

High `k` flattens the head of each list, so being in several lists matters more
than being first in one. That is the behaviour worth having — and the reason
noisy inputs were so damaging before df filtering.

---

## Section monopoly

For *"preventing transaction id wraparound"*, the top-5 was five chunks from
the same section:

```
1  24.1.5. Preventing Transaction ID Wraparound Failures
2  24.1.5. Preventing Transaction ID Wraparound Failures
3  24.1.5. Preventing Transaction ID Wraparound Failures
4  24.1.5. Preventing Transaction ID Wraparound Failures
5  24.1.5. Preventing Transaction ID Wraparound Failures
```

Both retrievers ranked all five highly, so RRF had nothing to fuse — the
candidate sets were the same content. Effective diversity was one section wide
while appearing five results deep.

**Reranking does not fix this.** All five chunks genuinely are relevant, so a
cross-encoder scores them all highly and the monopoly survives. It has to be
handled structurally, at fusion time.

```python
MAX_PER_SECTION = 2
```

After the cap:

```
1  24.1.5. Preventing Transaction ID Wraparound Failures
2  24.1.5. Preventing Transaction ID Wraparound Failures
3  19.10. Automatic Vacuuming                    ← previously crowded out
4  13.2. Transaction Isolation
5  19.11.1. Statement Behavior
```

`19.10. Automatic Vacuuming` is genuinely relevant and was invisible before.

### The trade-off

The cap trades recall *within* a section for coverage *across* sections. Right
when an answer needs two sections combined; wrong when an answer is spread
across five chunks of one section.

Which case dominates is empirical, hence the knob. Chunks with no `section_id`
fall back to their own id as the grouping key, so each counts as its own
section rather than all sharing `""`.

### A section is not always a topic

The cap assumes one section means one subject. That holds for
`24.1.5. Preventing Transaction ID Wraparound Failures`. It does not hold for
`19.4.1. Memory`, which is a reference list of a dozen independent parameters
— `shared_buffers`, `temp_buffers`, `work_mem`, `maintenance_work_mem`,
`autovacuum_work_mem` — split across eight chunks. Capping that section at two
chunks treats twelve unrelated answers as duplicates of one another.

Configuration-reference chapters are shaped this way throughout. It is the
strongest reason to expect the cap to hurt on this corpus, and one more thing
the eval harness has to settle.

---

## Current state

Distractor intrusion in the top-5, across the demo queries:

| Query | Distractors in top-5 |
| --- | --- |
| `work_mem` | 0 |
| `work_mem` vs `maintenance_work_mem` | 0 |
| `Is work_mem a per-connection limit?` | 1 (5th) |
| `why does the database slow down after many updates` | 2 |
| `what does autovacuum_vacuum_scale_factor control` | 0 |
| `preventing transaction id wraparound` | 1 (4th) |

Four clean, two with intrusion in the back half.

---

## Parameters

| Parameter | Value | Status |
| --- | --- | --- |
| `RECALL_K` | 20 | Candidates per retriever before fusion. To be swept |
| `RRF_K` | 60 | Convention. To be swept: 20 / 60 / 100 |
| `MAX_PER_SECTION` | 2 | Found by observation. To be swept |
| `MAX_DOC_FRACTION` | 0.15 | Found by observation. To be swept: 0.10 / 0.15 / 0.25 |
| `RERANK_TOP_N` | 5 | Matches the evaluation depth (Recall@5) |
| `RERANK_POOL` | 40 | Fused candidates sent to the reranker, before the cap. Cohere bills per search up to 100 documents, so a larger pool costs nothing |
| `RERANK_MODEL` | `rerank-v3.5` | Cohere's current model; not compared against alternatives |

Every one of these was set by looking at six queries. That is enough to catch
outright bugs and not enough to claim a value is right — which is what the
golden dataset and eval harness are for.

`MAX_PER_SECTION` is the one most likely to be wrong. Its justification rests
on a single query where one section filled the top-5, and reference sections
like 19.4.1 contradict its core assumption (above). If evaluation shows it
costs recall, it belongs at 0.

---

## What the demo queries probe

| Query | Probes |
| --- | --- |
| `work_mem` | Bare identifier — identifier search should dominate |
| `difference between work_mem and maintenance_work_mem` | Two identifiers the tsvector path cannot distinguish |
| `Is work_mem a per-connection limit?` | Identifier inside a sentence — the case AND semantics killed |
| `why does the database slow down after many updates` | Pure paraphrase, zero identifiers — vector's case |
| `what does autovacuum_vacuum_scale_factor control` | Long identifier, topic the distractors do not cover |
| `preventing transaction id wraparound` | A section that splits into many similar chunks |

A lock-related query sat in the fifth slot for several iterations and was
removed. Chapter 13 (Explicit Locking, a distractor) answers *"how do I see
which queries are blocked"* about as well as the monitoring chapter does — the
query was testing corpus design, not retrieval. Keeping it would have meant
chasing a ranking that could not be fixed by ranking changes.

That is the same lesson as the invalid full-text-search question in
`03-ingest.md`: when every retriever returns distractors, suspect the question
before the retriever.

---

## Reranking

The three retrievers all compare *representations*: a vector against vectors,
query lexemes against indexed lexemes, query identifiers against stored
identifiers. **None of them reads the passage to judge whether it answers the
question.**

That limit is visible in the fused output. For *"Is work_mem a per-connection
limit?"*, two chunks from section 19.4.1 appear in the top-5 with identical
metadata — same heading, same section id, same URL. Section 19.4.1 is eight
chunks covering a dozen parameters; the answer lives in two of them (1030 and
1031, where the `work_mem` description crosses a chunk boundary), and the
others document `shared_buffers`, `maintenance_work_mem`,
`autovacuum_work_mem` and so on. Nothing upstream can separate them, because
the difference is in the prose.

A cross-encoder concatenates query and passage into a single input, so
attention runs across both: "per-connection limit" in the query can attend to
"each operation" and "several running sessions" in the passage. That is
reading comprehension, not similarity.

### Why the other three retrievers cannot do this

The chunks differ in content, so it seems they should be separable. They are
not, because **none of the three retrievers reads content** — each compares a
compressed representation of it.

**Vector search** compresses a passage into 1,536 numbers and compares those.
Two chunks from the same section both reduce to "this is about memory
configuration", and the query lands in the same neighbourhood. Compression is
lossy: the specific fact that a limit applies per operation rather than per
connection survives only as a faint direction in the vector, swamped by the
much stronger topical signal. The score gap between a chunk that answers the
question and one that does not can be 0.02.

**Lexical search** compares lexeme sets. Chunks from one section share
`mem`, `memori`, `set`, `configur` — near-total overlap. `ts_rank` scores on
term frequency, which does not distinguish answering from merely discussing.

**Identifier search** compares identifier arrays. Both chunks contain
`work_mem`, so both score `matched = 1`, and the ranking falls to mention
frequency — a proxy, not the thing itself.

A cross-encoder skips the compression step entirely:

```
"Is work_mem a per-connection limit? [SEP] ...each operation generally being
 allowed to use as much memory as this value specifies. Also, several running
 sessions could be doing such operations concurrently..."
```

Attention runs across the join, so "per-connection" in the query attends
directly to "several running sessions" in the passage. That is reading, not
similarity.

The analogy that holds: the first three retrievers find a book by its index and
table of contents — they know it is about memory configuration, not what page
47 actually says. The cross-encoder reads page 47. Accurate, and impossible to
precompute, which is the next section.

### Why it runs after recall, not instead of it

| | Candidates | Per-item cost | Precomputable |
| --- | --- | --- | --- |
| Bi-encoder recall | 1,293 → 20 | negligible | yes — document vectors stored once |
| Cross-encoder rerank | 20 → 5 | high | **no** — every (query, passage) pair |

A bi-encoder embeds documents once at ingest; a query then compares against an
index. A cross-encoder must run a full forward pass for every pair, so nothing
survives between queries. Running it over 1,293 chunks would mean 1,293
inferences per query.

**The bi-encoder is cheap enough to scan everything; the cross-encoder is
expensive enough to handle twenty. That division is the entire argument for
two-stage retrieval.**

### Observed effect

Measured with the original ordering (cap, then rerank), before the reorder
described below:

| Query | Top-1 rerank score | Positions changed |
| --- | --- | --- |
| `preventing transaction id wraparound` | 0.897 | 4/5 |
| `difference between work_mem and maintenance_work_mem` | 0.814 | 4/5 |
| `what does autovacuum_vacuum_scale_factor control` | 0.710 | 1/5 |
| `work_mem` | 0.709 | 2/5 |
| `Is work_mem a per-connection limit?` | 0.624 | 4/5 |
| `why does the database slow down after many updates` | **0.405** | 5/5 |

On the wraparound query, RRF had placed a distractor at rank 4; reranking
pushed it out of the top-5 entirely and replaced it with two genuinely relevant
sections. On the `work_mem` per-connection query, the two same-section chunks
swapped order and a distractor at rank 5 was displaced.

### The score is a recall signal

The last row is the interesting one. Reranking changed 5 of 5 positions, and
the top score was 0.405 with three distractors still in the top-5. **The
reranker did its job on a candidate pool that contained no good answer.**

That gives a cheap diagnostic: a low top-1 rerank score means recall failed,
not that ranking failed. Reranking cannot recover a chunk that was never
retrieved. The agentic layer uses this directly — a score below threshold
triggers query rewriting and a second retrieval pass rather than answering
from bad context.

### Managed rather than self-hosted

An open-source cross-encoder (`bge-reranker-base`) would need PyTorch, a model
download, and local inference latency management. It would not be more
accurate. Reranking is a commoditized capability, and the build-vs-buy call is
easy when the thing you would build is the same as the thing you would buy.

### Failure is non-fatal

```python
except Exception:
    return hits[:top_n]
```

On an API error the fused order is returned unchanged. Degrading to
hybrid-only retrieval beats failing the query — and the eval harness measures
exactly how much that degradation costs, so the fallback's quality is a known
quantity rather than an assumption.

---

## Cap before or after reranking

The pipeline originally ran fuse → cap → rerank. It now runs
fuse → rerank → cap.

**The argument.** The cap decides which chunks of a section survive. Under the
original order it made that decision on RRF rank — and within one section, RRF
rank is the least reliable signal available, because same-section chunks share
heading, section id, and URL. That indistinguishability is the entire reason
reranking exists. Capping first meant discarding candidates, by the one signal
that cannot tell them apart, before the one component that can.

Reranking first means the cap trims a list already ordered by relevance. The
reranker sees up to 40 fused candidates rather than the capped remainder.

**What the argument does not establish.** Whether the new order measurably
performs better. Both are kept as configurations —

```python
"hybrid+rerank":     fuse → rerank → cap     # current
"hybrid+cap+rerank": fuse → cap → rerank     # original
```

— and the eval harness decides. The reason for keeping the old one is the
next section.

---

## A false alarm, and what it changed

Worth recording in full, because the mistake is instructive and it directly
shaped the ground-truth design.

### What appeared to happen

To test whether the cap was deleting correct answers, a check asked one
question: *is chunk 1030 in the final top-5?* Chunk 1030 had been identified as
the chunk documenting `work_mem`. For the query `work_mem`:

```
fused top-8     : [1031, 1044, 1032, 1030, 1043, ...]
after cap top-5 : [1031, 1044, 1032, 1043, 822]
1030 survives cap: False
```

That looked like proof: the cap kept two other chunks of section 19.4.1 and
deleted the one that answers the question. The reorder above was written in
response.

### What actually happened

Chunk 1031 was assumed to be about `maintenance_work_mem`. That came from a
diagnostic that printed each chunk starting at the first match of
`'work_mem ('` — which in 1031 matched inside `maintenance_work_mem (`. The
text *before* that point was never printed.

Counting whole-word mentions settled it:

| Chunk | Whole-word `work_mem` | Actually contains |
| --- | --- | --- |
| 1030 | 2 | `temp_buffers`, then the start of the `work_mem` description |
| 1031 | **8** | The rest of the `work_mem` description, `hash_mem_multiplier`, the start of `maintenance_work_mem` |
| 1032 | 1 | The end of `maintenance_work_mem`, `autovacuum_work_mem` |

The `work_mem` description crosses the chunk boundary between 1030 and 1031.
The 64-token overlap puts its key sentence — *"each operation generally being
allowed to use as much memory as this value specifies"* — in **both**.

So an answer-bearing chunk was in the top-5 under both orderings, for both test
queries. **The system was right; the check was wrong**, because it recognized
one correct chunk and there were two.

### What it changed

**Ground truth is a set, not an id.** Question generation (`05-questions.md`)
records one source chunk per question. That chunk is a candidate, not the
answer — overlap routinely puts the same sentence in two chunks, and the
ground truth has to capture every chunk that answers.

**Judges see full text, not previews.** A truncated diagnostic produced this
mistake. The golden-set judges read each chunk whole, and every question's pool
includes the source chunk's same-section neighbours regardless of retrieval,
since overlap makes those the most likely second answer.

**Retrieval is frozen until measured.** Two conclusions in a row — "the cap
deletes the right answer", then the reorder written to fix it — rested on
reading result lists against an ad-hoc standard. Further changes wait for the
golden dataset.

The word-boundary fix to the tiebreak (above) came out of the same
investigation and stands on its own: substring counting is wrong regardless.
It just was not why 1031 ranked first. 1031 ranks first because it contains
more of the `work_mem` description than 1030 does.

---

## A shared-object display bug

Worth recording because it looks like a data bug and is not.

The demo initially printed rerank scores on the `vector only` rows, which
should be impossible — reranking runs three stages later.

`fuse_rrf` merges duplicate chunks into **one** `Hit` instance rather than
copying:

```python
if hit.id in merged:
    existing = merged[hit.id]
    existing.ranks.update(hit.ranks)
```

So a chunk found by all three retrievers is one object referenced from four
lists. Writing `rerank_score` on it makes the score appear everywhere that
object is held.

The data was correct throughout; only the display was misleading. The fix was
ordering — print the pre-rerank stages before reranking runs — rather than
copying objects, since sharing is what makes the `v#3 l#1 i#2` provenance
summary possible in the first place.

---

## Still open

**Retrieval is frozen.** Every known bug is fixed. What remains are choices
that reading result lists cannot settle — the last few iterations demonstrated
that twice:

- **Cap order** — rerank-then-cap versus cap-then-rerank
- **Cap value** — `MAX_PER_SECTION` at 2 or 0, given reference sections like
  19.4.1 violate its core assumption
- **Four tuned-by-eye parameters** — `RECALL_K`, `RRF_K`, `MAX_DOC_FRACTION`,
  `MAX_PER_SECTION`

Seven configurations are registered for the harness:

| Config | Pipeline |
| --- | --- |
| `vector` | dense only |
| `lexical` | tsvector only |
| `identifier` | identifier array only |
| `hybrid` | fuse → cap |
| `hybrid+rerank` | fuse → rerank → cap |
| `hybrid+cap+rerank` | fuse → cap → rerank |
| `hybrid+rerank+nocap` | fuse → rerank, no cap |

The last one was added after the evaluation design was settled. The cap keeps
at most two chunks per section, so an answer that needs three chunks of one
section loses one. Without a configuration that skips the cap, the evaluation
could compare cap orders but never say whether the cap itself helps.

**One ranking error worth watching.** Under the current order, the top result
for *"Is work_mem a per-connection limit?"* is chunk 1032 — the passage saying
autovacuum may allocate up to `autovacuum_max_workers` times
`maintenance_work_mem`. "Memory multiplied by a count of things" is
semantically close to the question and answers it about a different parameter.
A plausible-but-wrong top-1 is the failure mode most likely to mislead a
generator, and the harness should show whether it recurs.

Steps 06-07 replace the reading with numbers.