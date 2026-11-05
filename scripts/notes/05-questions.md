# 05 — Questions and Ground Truth

Producing golden-dataset questions, and the problems that surfaced along the
way. Labelling which chunks answer them is step 06 (`06-golden-set.md`).

`scripts/05_generate_questions.py` → `eval/candidates.json`

---

## Why a golden dataset is unavoidable

Everything measured so far has been read off result lists by eye. That was
enough to catch several real bugs, and it cannot answer the question the project
now needs answered.

Concretely: for *"Is work_mem a per-connection limit?"*, the reranked top-5
contains two chunks from section 19.4.1, with identical heading, section id,
and URL. Reranking moved one of them from 2nd to 1st. **Was that the right
move?** There is no way to tell by looking — section 19.4.1 splits into eight
chunks covering a dozen parameters, the answer sits in two of them, and nothing
in the output says which two.

The same gap blocks a more important question. Three weaknesses were observed
in earlier steps: low similarity scores, poor separation in the mid-ranks, and
one section monopolizing the top-k. Each could be a *ranking* problem, which
reranking fixes, or a *recall* problem, which it cannot. One number separates
them:

| Recall@5 | MRR | Diagnosis |
| --- | --- | --- |
| high | low | Right chunks retrieved, wrong order — reranking is the answer |
| low | — | Right chunks never retrieved — reranking cannot help |

Cosine similarity is the system's *output*. Ground truth is the standard that
says whether the output is correct. The first exists already; the second has
to be built.

---

## Approach: reverse-engineer questions from chunks

Sample eval chunks, ask an LLM what question each one answers, and the chunk
becomes a ground-truth candidate.

The alternative — writing every question by hand and then hunting for its
answers in 1,293 chunks — is slower and worse. Working backwards guarantees
every question has an answer in the corpus, and guarantees you know which one.

Only eval chunks are sampled, which is what the `group` tag from ingest is
for. The 954 distractor chunks are never the source of a question — though one
can still turn out to answer it, which the judges handle
(`06-golden-set.md`).

---

## Problem 1 — the questions were too easy

First run, 80 candidates:

```
ok                 67
leakage             5
distractor_wins     5
source_not_found    3

of the 67 usable candidates, the source chunk ranked
  1st:      55     (82%)
  mean:     1.2
```

82% of questions retrieved their answer at rank 1.

**That reads like a strong retrieval system and is actually a broken
evaluation set.** With the correct answer first on four questions out of five,
Recall@5 would land near 1.0 for every configuration — vector-only included —
and the comparison table would be a row of near-identical numbers proving
nothing. An evaluation set composed of questions the system already answers
cannot distinguish between systems.

### Root cause

The problem is structural, not a prompt-wording failure. Generating a question
while looking at a passage produces a question whose *concepts* mirror that
passage one-to-one, even when reusing its wording is explicitly forbidden.
Retrieval then hits it immediately, because the question was reverse-derived
from the thing being retrieved.

Real users do not phrase questions the way the document answering them does.
They use their own vocabulary, and they often describe a symptom rather than
naming a mechanism.

### Fix: a paraphrase pass without the source in context

A second LLM call rewrites each question in an engineer's voice. The critical
detail:

**The paraphrase prompt never receives the source passage.** A model that
cannot see the document cannot echo it. Withholding context is the mechanism;
instructing the model not to copy is not, as the first run demonstrated.

```
Rewrite a search query the way a working engineer would actually type it.

- The vocabulary is the engineer's, not the manual's.
- Where natural, describe the symptom rather than naming the mechanism.
  "queries got slow after a big delete" rather than "what is the effect of
  dead tuples on scan performance".
- It stays answerable by the same passage.
- It stays short. Real search queries are under fifteen words.
```

### Result

```
                    before  after
ok                      67     64
leakage                  5      1
distractor_wins          5     10
source_not_found         3      5

rank 1:                 55     43    (82% -> 67%)
rank 2:                  ~      18
rank 3:                  ~       1
rank 5:                  ~       1
rank 6+:                 ~       1
mean:                  1.2    1.45
```

Three things moved:

**Leakage dropped from 5 to 1.** The rewrite washes out copied phrasing as a
side effect — good evidence the pass does what it claims.

**The distribution spread.** Rank 2 went from a handful to 18. Mean rose from
1.2 to 1.45.

**`distractor_wins` doubled, from 5 to 10.** Worth reading correctly: it is not
a regression. Engineer-voice questions use general vocabulary, which reaches
chapters the manual's own wording would not. At the time these 10 were
excluded; they are now kept, for the reason in *Selection* below.

### What the paraphrase pass could not do

67% at rank 1 is better than 82% and still easy.

The reason is visible in a rejected example:

```
was: "What is the maximum delay for cost-based vacuuming?"
now: "How long can vacuuming take in PostgreSQL?"
```

Completely different words, same question. **Paraphrasing changes vocabulary;
it cannot change structure.** The question still has exactly one answer, in
exactly one chunk, in exactly one section.

Genuinely hard questions need a different shape:

- the answer spans two chunks
- several candidate chunks share a section and only the prose separates them
- the answer does not exist

None of those can be generated from a single chunk, because each is defined by
a relationship the generator never sees.

---

## Problem 2 — questions that cannot measure anything

Three checks, each addressing a distinct way a question can look fine and be
useless.

Only the first still removes questions. The other two consult the retrieval
system being evaluated, and letting that system decide which questions it is
tested on biases the test — see *Selection* below. They are recorded in
`candidates.json` as diagnostics and not acted on.

### Leakage

The question reuses a distinctive phrase from its source chunk:

```
Q: Which file do I need to start the server in standby mode?
   leaked: 'to start the server in'
```

Retrieval then matches on the copied phrase rather than on meaning, and every
configuration scores well regardless of quality.

Detection is a 5-gram overlap between question and source. Five consecutive
words is almost never coincidence in a question this short. Single shared words
are expected and fine — a question about `work_mem` has to say `work_mem`.

This is the most common failure mode in LLM-generated evaluation sets, and it
is invisible unless checked, because a leaked question looks perfectly natural.

### Distractor wins

A distractor chunk outranks every eval chunk:

```
Q: how can I see disk space used by a specific table in PostgreSQL?
   distractor at 1, eval at 2
```

The original reading: **the question is invalid, not the retrieval.** It draws
on a chapter the question set does not cover, so the designated "correct"
answer is not the best answer.

The check exists because of a mistake made earlier by hand. A test query
about full-text search index types returned five distractor chunks, and
chapter 12 turns out to contain a section titled *"12.9. Preferred Index Types
for Text Search"* — which genuinely is the best answer. Retrieval was perfect;
the question was wrong.

That incident also corrected a framing error. The eval/distractor split is not
a judgment about which chapters matter. **It is a statement about where the
question set draws from**, and the two are related but not equivalent.
Conflating them is exactly how the invalid question got written.

The check no longer discards anything. A distractor that genuinely answers the
question is not a reason to throw the question away; it is a second correct
answer. The judges label every chunk in the pool, distractors included, and a
distractor judged `yes` joins the ground truth rather than counting as an
intrusion.

### Source not found

The source chunk does not appear in the top-10:

```
Q: How long can vacuuming take in PostgreSQL?
   was: What is the maximum delay for cost-based vacuuming?
```

Either the question is too general to have one answer, or the chunk is not
really what answers it. Here it is the former — the rewrite broadened
"maximum delay for cost-based vacuuming" into something several sections could
address.

### Why the rejects get printed

The script prints two examples from each flagged category. Filters that are
trusted rather than checked will silently discard good questions, and a
filter's threshold — 5-gram, top-10 — is a guess until its output has been
read.

---

## Ground truth stored three ways

```json
{
  "source_chunk_id": 142,
  "source_section_id": "RUNTIME-CONFIG-RESOURCE-MEMORY",
  "source_content": "# 19.4. Resource Consumption > 19.4.1. Memory\n\nwork_mem..."
}
```

The id is what the eval harness uses. The other two exist because **chunk ids
are only stable while the chunking configuration is.** `golden.json` applies
the same idea to every ground-truth chunk: `ground_truth_detail` stores its id,
section id, group, and the first 150 characters of its body.

Tier 2 compares chunk sizes of 256, 512, and 1024. Changing that renumbers
every chunk, and an annotation set keyed only on ids would not break visibly —
it would keep running and report a Recall@5 of zero, looking exactly like a
retrieval regression. The section id and content prefix allow re-resolving
ground truth against a re-chunked corpus.

Ingest's `TRUNCATE ... RESTART IDENTITY` is the other half of this: ids stay
stable across re-runs with identical input. That handles re-ingestion; it does
not handle re-chunking.

---

## Ground truth is a set, not an id

`source_chunk_id` is where a question *came from*. It is not necessarily the
only chunk that answers it, and treating it as the answer produced a false
alarm during retrieval testing (`04-retrieval.md`, *A false alarm*).

The `work_mem` description crosses the boundary between chunks 1030 and 1031.
The 64-token overlap puts its key sentence in both. A check that recognized
only 1030 reported a retrieval failure when an answer-bearing chunk was in the
top-5 all along.

So the ground truth records every chunk that answers — and has to distinguish
two situations that both look like "two chunks":

| Each chunk alone answers? | Situation | Match rule | Recall@5 |
| --- | --- | --- | --- |
| Yes, several do | Duplication — overlap, or the same fact stated twice | `any` | 1 if any ground-truth chunk is in the top-5 |
| No; only combined | True multi-hop | `all` | fraction of ground-truth chunks in the top-5 |

```json
{"question": "Is work_mem a per-connection limit?",
 "ground_truth": [1030, 1031], "match": "any"}
```

Conflating them inflates scores: a multi-hop question scored with `any` counts
as solved when half its evidence is retrieved.

### How to tell which is which

For each candidate chunk: *if this were the only passage the model saw, could
it answer?* There is no shortcut, and in particular **the reranker cannot make
this call.** Cohere scores each (question, chunk) pair independently, so it
measures relevance, not sufficiency. Two halves of a multi-hop answer each score
as moderately relevant, and nothing in the scores reveals that they are
complementary.

LLM judges make this call here, one passage per call, and see passages
together only for the combination check (`06-golden-set.md`). The
golden dataset is the standard itself, so the judges are validated before they
build it: they must pass hand-verified calibration cases, or the run stops.

How easy this is to get wrong: *"Does raising maintenance_work_mem give
autovacuum workers more memory?"* was drafted as a multi-hop question needing
chunks 1031 and 1032. Reading 1032 settles it alone — *"It defaults to -1,
indicating that the value of maintenance_work_mem should be used instead."* The
second chunk adds detail, not necessary evidence. Classified by impression it
was multi-hop; classified by reading it is not.

---

## Reproducibility

```python
rows = await pool.fetch("""
    SELECT id, content, metadata
    FROM retrieval_lab_chunks
    WHERE metadata->>'group' = 'eval'
    ORDER BY id
""")
...
random.seed(42)
sample = random.sample(chunks, N_SAMPLE)
```

Re-running re-examines the same 80 chunks rather than drawing fresh ones.
Without the seed, a second run produces a different candidate set and the two
cannot be compared — which matters precisely when comparing a configuration
change, the situation where this script gets re-run at all.

**The seed alone is not enough.** `random.sample` picks positions in a list, so
the same seed over a list in a different order picks different chunks. Without
`ORDER BY`, Postgres does not guarantee row order; it usually returns rows in
physical order, which matches id order here only because ingest inserts
sequentially and nothing updates the table afterwards. The first version
relied on that by accident. `ORDER BY id` makes the input order part of the
contract, so the seed actually reproduces the sample.

The existing `candidates.json` was generated before this fix. It was not
regenerated: the fix changes what a future run is guaranteed to do, not what
the past run produced.

---

## Selection — and the bias in the first plan

The first plan stratified the 64 usable candidates by the hybrid pipeline's
own rank for the answer: 8 from rank 1, 12 from ranks 2-3, 2 from rank 4+. It
was designed to make the set harder. It would have made the evaluation
meaningless.

**Choosing questions by how the evaluated system performs on them fixes that
system's score in advance.** A "hard" question was defined as one where hybrid
ranks the answer 6th or worse — so hybrid misses every one by construction. An
"easy" one was rank 1 — so hybrid hits every one by construction. Hybrid's
Recall@5 on those 22 questions would have been decided at selection time, while
the configurations it is compared against were never selected on. The
comparison table would have been tilted before a single query ran.

Two of the Problem 2 checks carried the same flaw. `source_not_found` discards
questions whose answer hybrid failed to retrieve — which removes precisely the
questions the system gets wrong, and inflates its score.

The rule that fixes all of it: **nothing that decides the test set may depend
on the system being tested.**

| Check | Consults retrieval? | Now |
| --- | --- | --- |
| Leakage | No — compares text only | Removes automatically |
| Distractor wins | Yes | Recorded as a diagnostic, not used |
| Source not found | Yes | Recorded as a diagnostic, not used |
| Rank-based difficulty | Yes | Dropped |

### What replaced it

**Every generated candidate**, minus leakage and near-duplicates of development
queries. There is no fixed count and no sampling step: all of them go to the
judges, and the ones the judges cannot settle are discarded. The question
count is whatever survives, and the report states it.

A fixed target (30, then 22) was dropped for two reasons. Picking a subset is
another place for a choice to leak in, and once labelling is automated, the
cost of judging every candidate is small — a few dollars.

**8 hand-written questions**, none touching a section inspected during
development:

| Type | Count | Purpose |
| --- | --- | --- |
| Same-section | 2 | Only prose separates candidates — tests reranking specifically |
| Multi-hop, parallel | 2 | Independent sub-questions — query decomposition can handle them |
| Multi-hop, sequential | 2 | Second lookup depends on the first — only iterative retrieval can |
| Unanswerable | 2 | Hallucination control |

The cost of random selection is a set skewed easy — roughly two-thirds of
generated questions retrieve their answer at rank 1. That is accepted: it
reflects the real distribution, and discrimination comes from the hand-written
cases rather than from filtering the generated ones by the system's own
performance.

### Development queries are excluded

The eleven development queries shaped retrieval decisions — the rerank
reorder and both tiebreak fixes were made looking at their results. Testing on
them is testing on the training set, with an unknowable inflation. Candidates
that near-duplicate any development query (word-set Jaccard ≥ 0.5) are
excluded automatically, and `discarded.json` lists each one with the query it
matched.

### Pooling

The judges see a pool of candidate chunks per question, not the whole corpus.
How the pool is built — and why it must not depend on retrieval alone — is in
`06-golden-set.md`.

### What each hand-written type measures

The generated questions all have one shape, because of how they were made: one
question, one answering chunk, answer exists. Four capabilities cannot be
tested with that shape at all, and each gets two questions — two because a
single question's outcome can be luck, and more than two measures nothing
further.

| Type | What it measures | Why it cannot be generated |
| --- | --- | --- |
| Same-section | Whether reranking earns its cost | The generator sees one chunk and cannot know which chunks share metadata |
| Multi-hop, parallel | Whether one retrieval covers two independent facts | The generator sees one chunk |
| Multi-hop, sequential | Whether the agentic layer earns its cost | Requires designing a dependency between two lookups |
| Unanswerable | Hallucination control | A question derived from a chunk always has an answer |

**Same-section** questions are the direct test of reranking. Section
`27.2.3. pg_stat_activity` is fourteen chunks with identical heading, section
id, and URL; only one explains what `idle in transaction` means. Every signal
available to the three retrievers is the same across those fourteen
(`04-retrieval.md`, *Why the other three retrievers cannot do this*). If
reranking does not visibly help here, it does not justify its latency and cost
anywhere.

**Unanswerable** questions are drawn from PostgreSQL's neighbourhood — Citus
for sharding, pgpool-II for query caching — rather than from unrelated
software. Proximity is what makes them hard: retrieval returns semantically
plausible chunks about sharding and connection pooling, which is exactly the
situation that tempts a generator into assembling a confident wrong answer.
"How do I configure Redis clustering?" would be trivially refused and would
measure nothing.

### Why multi-hop splits two ways

- **Parallel** — *"difference between log_min_duration_statement and
  log_statement"*: two independent lookups, so decomposing the question up
  front works.
- **Sequential** — *"index-only scans aren't happening — what maintenance
  affects that?"*: the first lookup establishes that index-only scans depend on
  the visibility map; only then does the second lookup, for what maintains the
  visibility map, become writable. Decomposition cannot write that query in
  advance.

Genuinely sequential questions are hard to write in reference documentation,
where a setting's name and its default usually share a chunk. A question's
`type` is its intended shape, not a verdict: if the judges find one chunk
that answers it alone, it is labelled `any`, whatever it was meant to be. The
report counts how many multi-hop questions survive as `all`.

Single-shot retrieval should lose on both multi-hop kinds. The agentic layer
should recover the sequential ones specifically — nothing else can, and that
gap is the evidence the agentic layer earns its cost.

### These questions also calibrate the agentic trigger

Deciding *when* to decompose or re-retrieve is itself a design question, and
there are three ways to answer it:

| Approach | Cost | Limitation |
| --- | --- | --- |
| Always decompose | An LLM call on every query | Most questions are single-fact; the decomposition returns one sub-question and the call is wasted |
| Ask an LLM whether to decompose | An LLM call on every query | Predicts difficulty before seeing any evidence |
| Trigger on a weak retrieval signal | Free — the signal already exists | Needs a threshold, which needs measuring |

The third is available because reranking already produces one. Observed top-1
rerank scores ranged from 0.897 on a well-covered query down to 0.405 on one
where the answer was not in the corpus at all. A low score means no single
passage covers the question — whether because the question needs several
facts or because retrieval failed. Both call for the same response, so the two
do not have to be told apart.

It also handles sequential multi-hop naturally: the second query is written
*after* seeing the first retrieval, so the model has the term it needs
(`visibility map`) instead of having to guess it while decomposing up front.

**Whether the threshold exists is an empirical question this dataset answers.**
Plotting top-1 rerank score for the multi-hop questions against the
single-fact ones shows whether a line separates them. If it does not, the
signal is insufficient and the trigger has to be an explicit LLM judgment
instead — which is a finding, not a failure.

---

## Labelling the ground truth

Step 06 labels which chunks answer each question, with LLM judges from two
vendors, calibration on hand-verified cases, and Cohen's kappa. See
`06-golden-set.md`.

---

## Status

`candidates.json` is used as-is. Its question text, source ids, and leakage
verdicts are independent of retrieval, and a check against the database found
all 80 source ids still point at the same content. Its rank fields are stale
and unused. Labelling status is in `06-golden-set.md`.
