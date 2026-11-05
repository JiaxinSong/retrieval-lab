# 06 — Golden Set

Labelling the ground truth: for every question, which chunks answer it. Done
entirely by LLM judges, with calibration and agreement checks standing in for
manual annotation.

`scripts/06_build_golden.py` → `eval/pools.json`, `eval/golden.json`,
`eval/discarded.json`, `eval/judge_report.json`, `eval/judgments.jsonl`

Questions come from step 05 (`05-questions.md`). Step 05 writes the questions;
this step decides what counts as a correct answer to each one.

---

## Pipeline

```
candidates.json (80) + hand-written questions (8)
  -> exclude leakage and near-duplicates of development queries
  -> build a pool of candidate chunks per question
  -> calibrate the judges on hand-verified cases; stop if they fail
  -> Judge A and Judge B label every (question, chunk) pair independently
  -> reconcile per chunk; a referee settles direct contradictions
  -> apply question-level rules; discard what cannot be settled
  -> golden.json, discarded.json, judge_report.json
```

---

## Walkthrough: one question, end to end

Think of it as marking an exam. A question has about 33 candidate answer
sheets — the chunks in its pool. Two markers, Judge A (`gpt-5-mini`) and
Judge B (`claude-haiku-4-5`), each read **every sheet separately** and write
one of three labels on it:

- **yes** — this sheet answers the question completely
- **partial** — it answers part of it
- **no** — it does not answer it

The markers work independently and never see each other's labels.

Take one question, *"Is work_mem a per-connection limit?"*, and pretend its
pool has only three chunks.

**Step 1 — each marker labels each chunk.** One API call shows one marker one
question and one chunk, nothing else. Three chunks × two markers = six calls.

| Chunk | Content | Judge A | Judge B |
| --- | --- | --- | --- |
| 1030 | The full `work_mem` definition | yes | yes |
| 1031 | The second half of that definition | partial | yes |
| 1032 | `autovacuum_work_mem` | no | no |

**Step 2 — merge the two labels into one verdict per chunk.** When the markers
differ, the stricter label wins. When one says yes and the other says no — a
direct contradiction, not a difference in strictness — a third marker
(Judge C, the referee) decides.

| Chunk | A | B | Final |
| --- | --- | --- | --- |
| 1030 | yes | yes | **yes** |
| 1031 | partial | yes | **partial** |
| 1032 | no | no | **no** |

**Step 3 — set the question's answer key.** Chunk 1030 is `yes`, so the ground
truth is {1030} with match rule `any`. In step 07, any configuration that puts
1030 in its top 5 gets this question right.

If no chunk were `yes` but two or three were `partial`, both markers would see
those chunks together and say whether they answer the question as a set. If
both agree, the question becomes multi-hop with match rule `all`. Otherwise,
and in every case the rules below cannot settle, the question is discarded.

**The whole step repeats this 87 times:**

```
87 questions
  └─ ~33 chunks each (2,849 pairs in total)
       └─ each pair labelled once by Judge A and once by Judge B (~5,700 calls)
            └─ merged into one verdict per chunk
                 └─ each question gets an answer key, or is discarded
```

Calibration runs exactly the same process first, on questions whose answers
were checked by hand, to test the markers before trusting them. The
calibration progress line counts the same thing: `74/74` is 37
(question, chunk) pairs × 2 judges.

### What one call looks like

Both judges get the same system prompt (the labelling rules) and the same user
message:

```
Question: Is work_mem a per-connection limit?

Passage:
# 19.4. Resource Consumption > 19.4.1. Memory

 memory used for temporary buffers ... work_mem ( integer ) # Sets the base
 maximum amount of memory to be used by a query operation ...
```

Each returns a fixed structure — the label can only be one of the three
values:

```json
{
  "answer": "No. It applies per query operation; several operations and sessions can each use it, so total memory can be many times the value.",
  "label": "yes",
  "reason": "The passage defines the limit as per operation and explains how it multiplies."
}
```

The two vendors enforce the structure differently with the same effect:
OpenAI through structured outputs (`.parse` with a Pydantic model), Anthropic
through a forced call to a `record` tool whose parameters are these fields.
Every result is written to `eval/judgments.jsonl` the moment it returns.

Since both judges see identical input, a disagreement between them reflects a
difference in judgement, not in what they were shown.

---

## Pooling — which chunks the judges see

Judging all 1,293 chunks for every question is too expensive, so each question
gets a pool of candidates. The judges label only what is in the pool, so
**anything not in the pool can never become ground truth.** What goes in the
pool decides what the test set can measure.

A pool has three sources:

| Source | Applies to | Depends on the system under test? |
| --- | --- | --- |
| Union of the three retrievers' top 20 | Every question | Yes |
| The source chunk and its same-section neighbours (id ± 1) | Generated questions | No |
| Keyword matches on seed terms (`ILIKE`) | Hand-written questions | No |

### Why the retrievers' top 20

Every configuration under evaluation draws its results from the same three
top-20 lists, so any configuration's top 5 is inside the pool by construction:
nothing a configuration returns goes unjudged. Pool depth equals `RECALL_K`
for that reason. An earlier version pooled at 10, and the reranker
demonstrably lifts chunks from rank 17 into the top 5 — those would have been
scored wrong because nobody looked at them. Verified by running all six
configurations against a local Postgres instance. Building the pool needs no
reranker calls.

### Why retrieval alone is not enough

If the pool came only from the retrievers, a question whose answer they all
miss would get no `yes` label and be discarded. The questions retrieval fails
on would silently leave the test set, and the decision about which ones leave
would be made by the system being tested. That is the self-selection bias
from step 05 entering by a different door.

The other two sources exist to close that door. Each adds the known answer to
the pool **independently of retrieval**, so a question the system fails on
stays in the test set and every configuration scores zero on it — which is
the failure the evaluation is supposed to expose.

- **Generated questions** were written from a source chunk, so the answer's
  location is known. The source chunk goes in, plus its same-section
  neighbours, because the 64-token overlap often puts the same sentence in the
  next chunk.
- **Hand-written questions** have no source chunk. Seed terms take its place.

### Seed terms are not the lexical retriever

The lexical retriever is already a keyword search, so seed-term matching can
look redundant. It is not. The two differ in what they search for and in whose
side they are on.

**They search for different words.** The lexical retriever searches for the
words in the question. Seed terms are written by someone who knows the
answer, and the decisive ones usually do not appear in the question at all:

| Question | Words in the question | Seed terms |
| --- | --- | --- |
| hw6 — index-only scans aren't happening, what maintenance affects that? | index-only scans, maintenance | **`visibility map`**, index-only scan |
| hw5 — row estimates are far off, what helps the planner? | row estimates, planner | actual rows, **`default_statistics_target`** |
| hw2 — what do the two numbers in `cost=0.00..458.00` mean? | cost=0.00..458.00 | **`start-up cost`**, cost=0.00 |

No lexical search over hw6 can find `visibility map`, because the question
never says it. Only someone who already knows the answer knows to look for it.

**They play different roles.** The three retrievers are the system under
test — the students sitting the exam. Seed-term matching is the examiner's
tool for compiling the answer key. An answer key compiled only from the
students' answers would never contain an answer no student found.

| | Lexical retriever | Seed-term matching |
| --- | --- | --- |
| Searches for | Words in the question | Words chosen by someone who knows the answer |
| Method | Full-text search, common terms dropped, top 20 only | Plain `ILIKE` substring match, every hit |
| Results used for | Ranking and scoring | The judges' pool only; never scored |

For the unanswerable questions, seed terms work in the other direction. The
seed `Citus` guarantees that if the docs did mention Citus, the judges would
see it and label it `yes`, and the question would be discarded as answerable
rather than wrongly accepted as having no answer.

---

## Labels

Each judge sees one question and one passage per call and answers: could a
reader given only this passage answer completely? It first writes the answer
that passage supports, then chooses a label for it (see *The first
calibration run failed* below for why).

| Label | Meaning |
| --- | --- |
| `yes` | The passage alone answers the question completely |
| `partial` | The passage holds part of the answer |
| `no` | It does not help, even if it is on a related topic |

One passage per call is deliberate. Shown several at once, a judge blends them,
and "answers on its own" stops being what is measured.

---

## Two judges, two vendors, one referee

| Role | Model |
| --- | --- |
| Judge A | `gpt-5-mini` (reasoning effort `low`) |
| Judge B | `claude-haiku-4-5` |
| Judge C (referee) | `claude-sonnet-5` |

Two runs of one model disagree only through sampling noise. Two vendors
disagree when their standards differ, which is the disagreement worth
catching.

Neither judge runs at temperature 0. The Anthropic judges cannot: the
current Anthropic SDK no longer accepts a `temperature` parameter, and the
first real run failed preflight with `TypeError: AsyncMessages.create() got an
unexpected keyword argument 'temperature'`. The OpenAI judge is a reasoning
model, which takes a reasoning effort instead of a temperature. All run at
their default sampling. This matters less than it sounds. Every label is cached the
moment it is made, so a rerun reuses it rather than sampling again, and
Cohen's kappa between the judges measures whatever variance remains.
Preflight caught the error in seconds, before any labelling call was made.

Reconciliation keeps the **weaker** label: yes/partial becomes partial,
partial/no becomes no. Only a direct contradiction — yes against no — goes to
Judge C, whose label is final. Keeping the weaker label makes ground truth
conservative. That lowers every configuration's recall a little and in the
same direction, which leaves the comparison between configurations intact.

---

## Question-level rules

| Situation | Outcome |
| --- | --- |
| Unanswerable type, no `yes` | Accept, match `none` |
| Unanswerable type, any `yes` | Discard — the question has an answer |
| At least one `yes` | Accept, ground truth = every `yes`, match `any` |
| No `yes`, no `partial` | Discard — nothing answers it |
| No `yes`, exactly one `partial` | Discard — only half an answer exists |
| No `yes`, two or three `partial` | Combination check |
| No `yes`, four or more `partial` | Discard — too complex to label reliably |

**Combination check:** both judges see the partial passages together and report
whether they answer completely and which are necessary. Accepted as `all` only
if both say complete and name the same necessary set of two or more.

Overlap can make interchangeable passages — the same sentence in two adjacent
chunks — so "A or A′, plus B" is possible. The judges will usually disagree on
which of A and A′ is necessary, and the question is discarded. That trades a
few multi-hop questions for a simple schema; the report counts the survivors so
the cost is visible.

---

## Multi-hop questions: parallel and sequential

A multi-hop question needs two or more chunks together; no single chunk
answers it completely. In the rules above that is the `all` match, reached
through the combination check. The hand-written set has two kinds.

**Parallel** — the pieces are independent. hw3 asks for the difference
between `log_min_duration_statement` and `log_statement`. Each parameter is
described in its own chunk, and finding one does not depend on the other: both
names are already in the question, so both lookups can run at the same time.

**Sequential** — the second lookup depends on the first. hw6 asks why
index-only scans aren't happening. The first lookup establishes that index-only
scans depend on the visibility map; only then can the second lookup, for what
maintains the visibility map, be written. The question never mentions the
visibility map, so that query cannot be written in advance.

### How the test set knows which kind a question is

By design, then confirmed. The intended kind is written into each
hand-written question's `type`, and the judges verify it: each piece should be
labelled `partial` alone, and the combination check should agree that the
pieces together are complete. If the judges find one chunk that answers alone,
the question becomes `any`, whatever it was meant to be. The type is a
hypothesis; the labels are the verdict.

### How a system would know at query time

It doesn't, in advance — and the two kinds reveal themselves differently.

**Parallel is visible in the question.** Everything needed is already in the
text: two named parameters, a "difference between A and B" structure. A
decomposition step can split it before retrieval:

```
"difference between A and B"  ->  ["what is A", "what is B"]  ->  retrieve both concurrently
```

Often even that is unnecessary: with both names in the query, a single
retrieval may return both chunks in the top 5.

**Sequential is only visible after a first retrieval.** Nothing in the question
points to the visibility map. A system does not decide up front that a
question needs two steps; it runs a loop:

1. Retrieve. The result says index-only scans depend on the visibility map.
2. Judge whether that is enough. It explains the dependency but not what
   maintenance affects it: incomplete.
3. Rewrite the query with what was just learned — "what maintains the
   visibility map" — and retrieve again.
4. Stop when the evidence is sufficient, or at a turn cap that bounds cost.

The decision to continue is made after seeing evidence, one step at a time.
The system never knows the total number of steps.

**So a real system does not classify first.** It decomposes obvious parallel
questions cheaply up front, and checks sufficiency after every retrieval. The
planned sufficiency signal is the rerank top-1 score: a low score means no
single passage covers the question. Whether a threshold on that score
separates multi-hop from single-fact questions is exactly what these
hand-written questions will measure (`05-questions.md`, *These questions also
calibrate the agentic trigger*). The loop itself is step 08.

---

## Calibration: the judges are tested before they label

Before any real labelling, the judges run on cases whose answers were verified
by reading the source. Any failure stops the run.

| Case | Required |
| --- | --- |
| `work_mem` definition chunk, for *"Is work_mem a per-connection limit?"* | `yes` |
| The next chunk, which repeats the key sentence through the overlap | not `no` |
| The `autovacuum_work_mem` chunk, same question | not `yes` |
| *"How do I enable TimescaleDB compression on a hypertable?"* against its pool | no `yes` anywhere |

The continuation chunk only has to be "not no": it opens mid-sentence with
"this value", so a strict judge may reasonably call it partial. Calibration
questions are never part of the evaluation set.

### The first calibration run failed — and the judge was wrong, not the case

The first real run stopped at calibration:

```
FAIL  #1030 work_mem definition          A=yes B=partial -> partial
pass  #1031 overlap continuation         A=partial B=yes -> partial
pass  #1032 autovacuum_work_mem          A=no B=no -> no
pass  unanswerable: 34 chunks, 0 judged yes, 0 missing
```

A failed calibration case has two possible causes: the judge is wrong, or the
hand-verified expectation is. Relaxing the case until it passes would make
calibration meaningless, so the first step was to re-read the source.

Chunk 1030 contains the complete definition: `work_mem` "sets the base
maximum amount of memory to be used by a query operation", "a complex query
might perform several sort and hash operations at the same time, with each
operation generally being allowed to use as much memory as this value
specifies", and "several running sessions could be doing such operations
concurrently. Therefore, the total memory used could be many times the value
of `work_mem`". The answer to "Is work_mem a per-connection limit?" — no, it
applies per operation, and operations and sessions multiply it — is all there.
The expectation was right.

The cached reasons showed what went wrong:

- **Haiku, 1030 → partial:** "explains that work_mem is a per-operation limit
  … but does not explicitly state whether it is a per-connection limit." It
  wanted the question's own words. The passage answers by correcting the
  question's premise, and the judge did not count that as an answer.
- **Haiku, 1031 → yes:** "explicitly indicating that work_mem is a
  per-session/per-connection limit." The right label for the wrong reason: it
  read the passage as confirming the premise.
- **gpt-4o-mini, 1030 → yes:** "it implies that this limit applies per
  session." Also the right label for the wrong reason.

Both judges misread the substance, and one of them penalised a correct passage
for not echoing the question. This matters beyond one case: the paraphrase
pass in step 05 deliberately writes questions in an engineer's vocabulary,
often with a mistaken premise, so a judge that demands the question's words
would systematically under-label exactly those questions.

**Fix (prompt v2).** Two changes to the chunk judge:

- **Answer first.** The schema now has an `answer` field before `label`: the
  judge writes the answer a reader would give from this passage alone, then
  labels it. The label is grounded in what the passage says, and the answer is
  stored in the cache, so a wrong label can be traced to a wrong reading.
- **Premise rule.** The prompt states that a passage answers a question when
  it lets the reader give the correct answer, whether or not it uses the
  question's words — including when the correct answer rejects the question's
  premise. The example in the prompt uses a made-up limit, not `work_mem`, so
  the fix does not teach the judges the calibration answer.

The calibration expectations were not changed. The other three cases guard
against the fix overshooting: if the new prompt made judges lenient, the
`autovacuum_work_mem` chunk or a TimescaleDB chunk would start getting `yes`,
and calibration would fail again. `PROMPT_VERSION` moved to `v2`, so no
judgment made under the old prompt is reused.

### The second run failed too — and the stored answers exposed the judge

With prompt v2, Haiku passed: it labelled the definition `yes` and answered
"No, work_mem is not a per-connection limit. It is the base maximum amount of
memory for a single query operation…". `gpt-4o-mini` failed on the
continuation chunk:

```
pass  #1030 work_mem definition          A=yes B=yes -> yes
FAIL  #1031 overlap continuation         A=no B=partial -> no
```

The new `answer` field showed why, and showed something worse:

- **1031 → no:** "the passage does not specify whether work_mem is a
  per-connection limit." The chunk names `work_mem` and says several sessions
  can run such operations concurrently, each using that much memory. At worst
  that is `partial`.
- **1030 → yes**, answering: "Yes, work_mem is a per-connection limit as it
  sets the maximum amount of memory to be used by a query operation within
  each session." The label was right. The answer was the opposite of the text.

That second point is the important one. **A label can be right for the wrong
reason, and a calibration check on labels alone would have passed it.** The
judge was not applying a strict standard differently from Haiku; it was not
understanding the passages. No prompt wording fixes reading comprehension, so
the judge was replaced: Judge A is now `gpt-5-mini`, a reasoning model run
at reasoning effort `low`.

Calibration now checks substance as well as labels. For the `work_mem`
definition chunk, each judge's stored `answer` must not open with "yes" —
the correct answer is "no", so an answer affirming the premise has misread the
passage whatever label it chose. The check is deliberately narrow: it catches
exactly the failure observed, without trying to grade free text.

---

## Report

`judge_report.json` records the confusion matrix between Judge A and Judge B,
Cohen's kappa, how many contradictions went to the referee, discard reasons,
and multi-hop survivors. The script also prints a fixed random sample of ten
accepted questions with their ground-truth chunks, for an optional human
glance of about five minutes.

---

## Operation

- Every judgment is cached in `eval/judgments.jsonl`: one JSON object per
  line, appended the moment a call succeeds. On start-up the file is loaded
  into a dictionary; before every API call the key is looked up, and a hit
  returns the stored label without calling the model. An interrupted run loses
  at most the calls in flight, and a failed call is never written, so the next
  run retries it. A run after a retrieval change judges only chunks new to the
  pool.
- The key is `prompt version | model | question id | chunk id | text hash`,
  for example `v2|gpt-5-mini|g00|5|efb664addd5fded9`. Each part invalidates
  the cache for a different reason: a new prompt version or model means the
  old label was made under different conditions. The hash covers the exact
  question and chunk text the judge saw.
- **Why the hash is needed.** The first version keyed on question id and chunk
  id alone. Both are positions, not content: `g00` is whichever candidate
  comes first in `candidates.json`, and a chunk id points at different text
  after re-chunking. Regenerating questions or re-chunking would have reused
  judgments made for a different question or passage — silently, with no
  error, producing wrong ground truth. Caught in review before the first real
  run, so no cached judgment was affected.
- Plain API calls with client-side rate limiting, bounded concurrency, and SDK
  retries with exponential backoff. An authentication error aborts the run
  rather than retrying.
- `--dry-run` counts calls and spends nothing; `--reset` rebuilds pools and
  keeps the cache; `--clear-cache` discards every judgment.

---

## What this is not

A human-verified benchmark. It is an LLM-judged evaluation set, suitable for
comparing retrieval configurations against each other. One bias is worth
stating: the judges' sense of "answers the question" may resemble a
cross-encoder's sense of relevance, which could flatter the rerank
configurations. Cross-vendor judging reduces that; it does not remove it.

Two more limits surfaced in the real run (see *Results*): the referee comes
from the same vendor as Judge B, and agreement between the judges does not
protect against a question that is itself ambiguous.

---

## Results

*This section describes the first run (87 questions). A second pass added
five multi-hop questions; its results are under* Second pass: results *below,
and the final set is 84 questions.*

**In one sentence:** both judges passed calibration, then labelled every
candidate chunk for 87 questions; 81 questions came out with a trustworthy
answer key and 6 were discarded because they could not be settled. One judge
was much more lenient than the other, and the reconciliation rules kept it
from lowering the standard.

What the step produces is an **answer key**, `eval/golden.json`: for each
accepted question, the chunks that answer it. Step 07 scores each retrieval
configuration against it — a configuration that puts a ground-truth chunk in
its top 5 gets that question right.

The full run completed on the third attempt at calibration (see
*Calibration*), with every judgment present. `pending 0` means no question is
missing a judgment because of a failed call; a nonzero count would mean
rerunning to fill the gaps.

```
81 questions -> eval/golden.json
6 discarded -> eval/discarded.json
accepted 81, discarded 6, pending 0
```

| | Value |
| --- | --- |
| Questions after exclusions | 87 (79 generated, 8 hand-written) |
| Accepted | **81** — 75 generated, 2 same-section, 2 sequential multi-hop, 2 unanswerable |
| Discarded | 6 — 3 "no chunk answers it", 2 "judges did not agree on a combination", 1 "more than 3 partial pieces" |
| Match rules | 79 `any`, 2 `none`, **0 `all`** |
| (question, chunk) pairs judged | 2,849 per judge, 5,698 calls |
| Cohen's kappa, Judge A vs Judge B | **0.589** |
| Yes/no contradictions sent to the referee | 70 → 68 `no`, 1 `yes`, 1 `partial` |
| Ground-truth chunks from distractor chapters | 13 |
| Wall time | 63 minutes, Anthropic rate limit bound |

### The judges disagree in one direction

```
                   no  partial      yes     (Judge B, Haiku)
        no       2352       15        5
   partial        125       48       10
       yes         65       70      159
(Judge A, gpt-5-mini)
```

**How to read it.** Rows are Judge A's label, columns Judge B's; each cell
counts the pairs where A said the row label and B the column label.

- The **diagonal** (2,352 / 48 / 159) is where the judges agree. Most pairs
  are there, and most of those are `no`: a typical pool has a few chunks that
  answer the question and many that don't.
- **Below the diagonal**, A gave a higher label than B: A `yes` / B `no` 65
  times, A `yes` / B `partial` 70, A `partial` / B `no` 125. **Above** it,
  the reverse, is rare: 5, 10, 15.

So the disagreement is almost all in one direction. Judge A says `yes` 294
times, Judge B 174. Judge A is the lenient one.

**Kappa** measures agreement after removing what two raters would agree on by
chance: 0 is chance level, 1 is perfect agreement. 0.589 is conventionally
"moderate". It is not higher mainly because of that one-directional gap in
strictness, not because the judges disagree at random.

This is the case the reconciliation rules exist for. Keeping the weaker label
means the lenient judge cannot pull the standard down: A `yes` / B `partial`
becomes `partial`, and every yes/no contradiction goes to the referee, who
sided with `no` 68 times out of 70.

**One caveat to state plainly.** The referee (`claude-sonnet-5`) and Judge B
(`claude-haiku-4-5`) are both Anthropic models. The referee agreeing with
Judge B almost every time is consistent with Judge B being right, but also
with two models from one vendor sharing a standard. The data here cannot tell
those apart. A referee from a third vendor would.

### No genuine multi-hop question survived (first run)

*Superseded by the second pass below: three parallel multi-hop questions
written from the chunk text were accepted as `all`.*

The report line `multi-hop surviving: 2 of 4` counts questions *designed* as
multi-hop that were accepted, whatever rule they ended with. The match rules
tell the real story: zero questions are `all`.

- Both **sequential** questions were accepted as `any`: the judges found a
  single chunk that answers each one alone. The design notes anticipated this:
  in reference documentation, a setting's name and its effect usually share a
  chunk, so a genuinely sequential question is hard to write.
- Both **parallel** questions were discarded, most likely the two "judges did
  not agree on a combination" — the interchangeable-overlap case described
  under *Question-level rules*.

Consequence: the set measures single-fact retrieval, same-section
disambiguation, and unanswerable questions, but not multi-hop retrieval. The
evaluation in step 07 is unaffected. The agentic layer in step 08 is: its
trigger was to be calibrated by comparing rerank scores on multi-hop against
single-fact questions, and there are no multi-hop questions to compare.
Adding a few more is cheap — a rerun judges only the new questions; every
existing judgment comes from the cache.

### Spot check: one false positive both judges agreed on

The run prints ten accepted questions with their ground truth. Two looked
wrong; the stored answers settled both.

**g50 — correct.** *"What does it mean if idle_session_timeout is zero?"* The
printed preview of chunk 935 showed text about a different parameter, but the
preview is only the first 200 characters. Both judges quoted the passage
stating that zero disables `idle_session_timeout`. The preview misled; the
label is right.

**g21 — a false positive.** *"When do I see non-zero I/O time for columns?"*
Ground truth includes 768 and 769 (`pg_stat_io`, where timing columns stay
zero unless `track_io_timing` is on), which are right, and 759
(`pg_stat_replication`), which is not. Both judges labelled 759 `yes`, reading
"I/O time" as replication lag time:

> gpt-5-mini: "You will see non-zero I/O times while new WAL is being
> generated … pg_stat_replication reports the time taken to write, flush and
> replay…"

The judges answered the question as written. The question is the problem:
the paraphrase pass broadened it until "non-zero time for columns" fits two
unrelated views. **Two judges agreeing is not the same as two judges being
right** — cross-vendor judging catches differences in standards, not
ambiguity in the question both of them read.

The label was left as is. Editing individual labels by hand would break the
rule that ground truth is produced by the documented process, not adjusted
case by case. The effect is bounded: g21 is an `any` question that already
has correct chunks in its ground truth, so the error can only credit a
configuration that retrieves 759 — at most one question in 81. That is below
the "about two questions" threshold step 07 uses before drawing a conclusion
from a difference.

---

## Second pass: multi-hop questions written from the data

The first run left no `all` question, so nothing tested retrieval that needs
two chunks together — and the agentic layer in step 08 had nothing to prove
itself on. The four original multi-hop questions failed because they were
written from memory of the docs: the sequential ones turned out answerable by
one chunk, and the parallel ones split their answer across overlapping chunks
the judges could not agree on.

The second pass reversed the order: **find where the answers are, then write
the question.**

1. **Co-occurrence check.** For 16 candidate pairs of settings, count the
   chunks that mention each one and the chunks that mention both, using
   word-boundary matching. A pair is usable only if no chunk mentions both —
   otherwise one chunk can probably answer alone. Five pairs qualified:
   `shared_buffers` / `effective_cache_size`, `wal_buffers` / `max_wal_size`,
   `random_page_cost` / `effective_io_concurrency`, `work_mem` /
   `temp_file_limit`, `log_checkpoints` / `checkpoint_warning`.
2. **Read the defining chunks.** For each setting, the chunk containing its
   definition (`name ( type )`), read in full.
3. **Word the question around what is in exactly one chunk.** Three of the
   ten definitions cross a chunk boundary, so part of the answer appears in
   two overlapping chunks — the case where the judges disagree about which
   chunk is necessary and the question is discarded. For those, the question
   asks for the **default value**, which appears in only one of the two.
   `work_mem`'s default (4MB) is in chunk 1030, not 1031; `log_checkpoints`'
   default (on) is in 989, not 988; `checkpoint_warning`'s default (30s) is in
   1059, not 1058. A question about `random_page_cost` and SSDs was dropped
   for the same reason: the relevant sentence falls past the chunk's end.

| qid | Question | Expected chunks |
| --- | --- | --- |
| hw9 | How much RAM should I give shared_buffers, and what does effective_cache_size actually do? | 1028 + 1008 |
| hw10 | What do wal_buffers and max_wal_size each control, and what are their defaults? | 1055 + 1059 |
| hw11 | What do random_page_cost and effective_io_concurrency each control, and what are their defaults? | 1006 + 1042 |
| hw12 | Sorts keep spilling to temp files: what controls how much memory a sort gets before it spills, what is its default, and how do I cap the temp file space a process can use? | 1030 + 1036 |
| hw13 | Is log_checkpoints on by default, and by default how close together must checkpoints be before checkpoint_warning logs a warning? | 989 + 1059 |

Expected chunks are notes for checking afterwards, not labels: the judges
label these questions exactly like every other, and a question they settle
differently is accepted or discarded by the same rules. Seed terms are the
exact definition strings (`shared_buffers ( integer )`), so the defining
chunks are in each pool regardless of retrieval.

These are all **parallel** questions. Writing a sequential one — where the
second lookup depends on a term only the first chunk reveals — needs a pair
where chunk A names B without explaining it. The co-occurrence check found
none among these candidates; `log_lock_waits` / `deadlock_timeout` came
closest but both terms appear together in two chunks.

---

### Second pass: results

```
accepted 84, discarded 8, pending 0
accepted by match: {'any': 79, 'none': 2, 'all': 3}
Cohen's kappa: 0.592   (3013 pairs)
```

Rebuilding the pools for 92 questions left every existing pool unchanged:
only the 164 pairs of the five new questions were uncached, so only they were
judged (about four minutes, a few dimes).

| qid | Outcome | Ground truth | Expected |
| --- | --- | --- | --- |
| hw9 | accepted, `all` | 1008 + 1028 | 1028 + 1008 ✓ |
| hw11 | accepted, `all` | 1006 + 1042 | 1006 + 1042 ✓ |
| hw13 | accepted, `all` | 989 + 1059 | 989 + 1059 ✓ |
| hw10 | discarded: more than 3 partial pieces | — | 1055 + 1059 |
| hw12 | discarded: judges did not agree on a combination | — | 1030 + 1036 |

**Every accepted question got exactly the chunks predicted from reading the
text.** The judges labelled blind to those predictions, so the agreement is
a check on both: the method found real two-chunk answers, and the judges
recognised them.

The two discards fail in the two ways the method was trying to avoid:

- **hw10**: `max_wal_size` is mentioned in 11 chunks (the co-occurrence
  check), most of them in passing. Too many were labelled `partial`, and four
  or more partials is discarded as too complex to label reliably. The co-occurrence check ruled out chunks that name *both*
  settings, but not chunks that partly explain *one*.
- **hw12**: asking for `work_mem`'s default was meant to make chunk 1030 the
  only necessary one, but the judges still disagreed on the necessary set. The
  `work_mem` description straddling 1030 and 1031 defeated the workaround.

## Status

Done. `eval/golden.json` (84 questions: 79 `any`, 3 `all`, 2 `none`) is the
input to step 07.

Open items, none blocking:

- **No sequential multi-hop question exists.** The two written as sequential
  were answerable by one chunk, and the co-occurrence check found no pair
  where one chunk names a setting without explaining it. The three `all`
  questions are parallel: both settings are named in the question, so a
  single retrieval can find both halves.
- A third-vendor referee would remove the same-vendor caveat.