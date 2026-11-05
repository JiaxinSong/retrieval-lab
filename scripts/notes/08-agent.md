# 08 — Answers and the Agentic Layer

Turning retrieved passages into a cited answer, and retrieving again when the
first retrieval looks weak.

`scripts/08_agent.py` → `eval/answers.json` (default threshold),
`eval/answers_threshold_1.0.json` (agentic loop forced on every question);
answers cached in `eval/agent_runs.jsonl`

```bash
python scripts/08_agent.py --ask "Is work_mem a per-connection limit?"   # one question, with sources
python scripts/08_agent.py                                                # the golden set
python scripts/08_agent.py --threshold 1.0                                # force the agentic loop
```

---

## How one question is answered

```
question
  -> hybrid + rerank, top 5                     (best configuration from step 07)
  -> top rerank score >= 0.5 ?
       yes: one model call; the model may only call `answer`
       no:  the model also gets `search_docs` and is told the first search
            looked weak; it may rewrite the query and search again,
            up to 3 searches in total, then must answer
  -> answer text + cited passage numbers + answerable flag
  -> citations mapped back to chunk ids and URLs
```

The model is `gpt-5-mini` at reasoning effort `low`, the same model and
setting as Judge A in step 06.

### Why a score gate instead of always letting the model decide

The simplest agent gives the model a search tool on every question and lets
it decide. That pays for a decision on every question, including the large
majority where the first retrieval was fine.

Step 07 showed the reranker's top score already carries that signal: every
answerable question scored at least 0.511, both unanswerable ones scored
below (0.360 and 0.481). So the gate is free — the score already exists — and
keeps the common case at one model call. The agentic loop runs only where the
evidence is weak, which is where another search could help.

The three ways to decide when to retrieve again, from `05-questions.md`:

| Approach | Cost | Limitation |
| --- | --- | --- |
| Always decompose or re-search | An extra decision on every query | Wasted on single-fact questions |
| Ask an LLM up front | An extra call on every query | Predicts difficulty before seeing evidence |
| **Gate on the rerank score** | Free | Needs a threshold, which needs measuring |

### Why the search tool and a hard cap

Retrieving again is a tool the model calls with a query it writes. That is
what lets it handle a sequential question: after the first search, it can use
a term it just learned ("visibility map") that the original question never
mentioned. A query rewritten before seeing any evidence could not.

A model told "search again if needed" can keep rephrasing. The cap of three
searches bounds cost and latency. When the cap is reached, the search tool is
removed from the call and the answer tool is forced.

### Why the answer is a tool call

The model answers through an `answer` tool with a fixed schema:

| Field | Purpose |
| --- | --- |
| `answer` | The text, with `[n]` citations |
| `citations` | The passage numbers relied on |
| `answerable` | `false` when the passages do not answer the question |

A schema makes the output checkable. Citation numbers are mapped back to
chunk ids and URLs, so every claim traces to a page; a number that matches no
passage the model was shown is counted as an invalid citation. `answerable:
false` is how the model refuses instead of filling a gap from its own
knowledge, which matters most on the near-domain unanswerable questions
(Citus, pgpool-II) where plausible-looking passages invite a confident wrong
answer.

**A schema in the prompt is a request; strict mode is a guarantee.** The
first real run crashed on the ninth question: the model called `answer` with
`answer` and `citations` but no `answerable`. The tool definitions now set
`"strict": true`, which makes the API enforce the schema, so a required field
cannot be missing. As a second line, a malformed answer gets one retry: the
model is told which field was wrong and must call `answer` again, with the
search tool withdrawn. A second failure stops the run. Answers already cached
from the first nine questions were valid and are reused.

### Passages keep their numbers across searches

Every passage shown to the model is numbered once. If a later search returns
a chunk already shown, it is reported as "already shown as [3]" rather than
renumbered. Otherwise the same chunk could be cited under two numbers, and a
citation would no longer identify one passage.

### The same guards as step 07

Retrieval goes through the same pacing and the same check for the rerank
step's silent fallback: a result without rerank scores is retried, never
used. For the golden-set run, the first retrieval of each question is read
from step 07's cache (ids and scores, with text fetched from the database),
so the rerank calls step 07 already made are not repeated.

---

## What the golden-set run measures

The golden set labels chunks, not answers, so the run measures what it can
score without new judging:

| Measure | Meaning |
| --- | --- |
| Path | How many questions took the direct path and how many the agentic one, and how many searches the agentic ones used |
| Found first / found in any search | For answerable questions, whether a ground-truth chunk was among the passages after the first search, and after all searches |
| Agentic recovered | Agentic answerable questions where a later search found a ground-truth chunk the first one missed |
| Cites ground truth | Answerable questions whose citations include a ground-truth chunk |
| False refusals | Answerable questions answered with `answerable: false` |
| Correct refusals | Unanswerable questions answered with `answerable: false` |
| Invalid / unjudged citations | Citation numbers that match no passage; citations to chunks outside the judged pool (possible after a rewritten search) |

**Not measured: whether each answer is correct and complete.** That needs an
answer-level judge, which step 09 adds (`09-answer-judge.md`). Citing a ground-truth chunk
is evidence the answer is grounded in the right passage, not proof the answer
says the right thing.

### The threshold was read off the test set

The 0.5 threshold comes from step 07's scores on these same questions, so
on this set it sends almost every answerable question down the direct path by
construction. A threshold tuned on the data it is evaluated on says nothing
about how it generalises.

Two runs separate the questions:

- **Default threshold (0.5).** What the deployed gate would do. Mostly
  measures citations and refusals.
- **`--threshold 1.0`.** Forces the agentic loop on every question. Measures
  the loop itself: does a model that may search again find ground truth the
  first search missed, and does offering it that option make it refuse
  answerable questions or waste searches on questions it could already
  answer?

Choosing a threshold properly needs held-out questions, including multi-hop
ones, which this golden set does not have (`06-golden-set.md`).

---

## Results

Two runs over the 84 golden questions (82 answerable, 2 unanswerable),
including the three parallel multi-hop questions added in step 06's second
pass. For those, "ground truth found" and "citations include ground truth"
require **every** required chunk, not just one.

| | Default gate (0.5) | Loop forced (1.0) |
| --- | --- | --- |
| Questions on the agentic path | 2 | 84 |
| Questions where the model searched again | 1 | **8** (mean 1.1 searches) |
| Ground truth among passages, first search | 80 / 82 | 80 / 82 |
| Ground truth among passages, any search | 80 / 82 | **81 / 82** |
| Citations include the ground truth | 80 | 81 |
| Answerable questions refused | **0** | **0** |
| Unanswerable questions refused | **2 / 2** | **2 / 2** |
| Invalid citation numbers | 0 | 0 |
| Citations to unjudged chunks | 0 | 3 |

### Answers are grounded, and refusals are right

With the default gate, every question whose first retrieval contained its
ground truth (80 of 82) got an answer citing it. No citation
pointed at a passage the model wasn't shown. No answerable question was
refused, and both near-domain unanswerable questions — Citus sharding and
pgpool-II query caching, the kind most likely to draw a confident invented
answer — were refused.

Both unanswerable questions took the agentic path, as the gate intends; one
searched a second time before refusing, the other refused after one search.
Every answerable question took the direct path. That is the circularity the
design section warns about: the threshold was read off these questions.

### Multi-hop answers cite both halves

The three parallel multi-hop questions each need two chunks. In every case
the first retrieval already contained both, the answer took the direct path,
and **the answer cited both required chunks**:

| qid | Question (short) | Cited |
| --- | --- | --- |
| hw9 | shared_buffers sizing + what effective_cache_size does | 1028, 1008 |
| hw11 | random_page_cost + effective_io_concurrency, with defaults | 1006, 1042 |
| hw13 | log_checkpoints default + checkpoint_warning default | 989, 1059 |

A model given two passages that each answer half a question used both and
said which half came from where. That is the behaviour the citation schema
was built to make checkable. It also means these questions never exercised
the agentic loop: both halves were retrieved on the first try, because both
settings are named in the question.

### The forced loop: it doesn't hurt, and it rarely has anything to fix

Offered a second search on every question, the model used it on only eight.
On the other 76 it judged the first passages sufficient and answered. Giving
it the option cost one extra search on about one question in ten, and caused
no refusals of answerable questions.

It recovered **one** question the first retrieval had missed — below the
two-question threshold, so not evidence that the loop helps in general. The
reason is the test set, not the loop: first retrieval already finds 80 of 82,
leaving two questions to recover, and the only multi-hop questions are
parallel ones that a single retrieval already answers. What the loop exists
for — a second search that depends on what the first one found — has no
question in this set to test it.

### The one it recovered is the sequential question

The recovered question is **hw5**, one of the two hand-written sequential
multi-hop questions:

```
search 1: EXPLAIN ANALYZE shows row estimates far from the actual rows --
          what setting helps the planner estimate better?
search 2: default_statistics_target planner estimate better ANALYZE
          statistics target row estimation EXPLAIN ANALYZE
```

The second query names the setting the question never mentions, and the
second search found the ground-truth chunk the first one missed.

**But it was not a second hop.** None of the five passages from the first
search mentions `default_statistics_target` (checked against the database:
chunks 1236, 1246, 1251, 1253, 1254, all from 14.1 *Using EXPLAIN*). The model
did not learn the term from the evidence; it supplied it from its own
knowledge of PostgreSQL. That is **query expansion** — rewriting the user's
words into the docs' words — not multi-hop retrieval, where the second query
depends on something the first search revealed.

It is still a legitimate use of the search tool, and the useful one here. But
the claim has to match the mechanism: the loop recovered a question by
rewriting the query, not by chaining evidence. Whether it can chain evidence
is untested.

### What the rewrites look like

Six of the eight second searches follow one pattern: **the model turned a
natural-language question into an identifier-rich query.** The other two
broadened a vague question with domain terms (maintenance: "vacuum backup log
rotation autovacuum") or restated it (Citus).

| Question (paraphrased) | Second query added |
| --- | --- |
| Disk space used by one table | `pg_table_size`, `pg_total_relation_size` |
| `arg0` in the buffer-sync-done probe | the probe name and `arg0` alone |
| Writing while an index is built | `CREATE INDEX CONCURRENTLY` |
| Default syslog facility | `syslog_facility`, `LOCAL0` |
| What `WALInitSync` means | `WalInitSync` with its exact casing, `WalInitWrite` |
| Row estimates far off | `default_statistics_target` |

Step 07 found that the lexical and identifier retrievers are weak on these
everyday-language questions — fusion without reranking lost to vector search
alone. The agent's rewrites move queries toward exactly the vocabulary those
retrievers are good at. The run is too small to measure the effect, but it
suggests the three retrievers and the agentic loop are complementary: the
loop can translate a user's words into the docs' words.

### Three unjudged citations

After a rewritten search, the model can see chunks that were never in the
judged pool. Three such citations occurred, on g45 (index builds and writes)
and hw7 (Citus, where the model refused but cited a related passage). They
are counted separately, not as wrong: nobody has labelled them.

### What these runs do not show

- **Whether the answers are correct.** Citing a ground-truth chunk shows the
  answer rests on the right passage, not that it says the right thing. An
  answer-level judge is added in step 09 (`09-answer-judge.md`).
- **Whether the loop helps on sequential multi-hop questions.** The parallel
  ones never needed it, and no sequential question survived labelling
  (`06-golden-set.md`). One recovered question is an example, not a
  measurement.
- **Whether 0.5 is a good threshold.** It was chosen on this data.
