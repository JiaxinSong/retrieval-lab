# 07 — Evaluation

Scoring every retrieval configuration against the golden set from step 06.

`scripts/07_evaluate.py` → `eval/results.json`, `eval/results_summary.json`
(retrieval runs cached in `eval/runs.jsonl`)

---

## What is compared

Nine configurations, all defined in `retrieval.CONFIGS` and all returning a top 5:

| Configuration | Pipeline |
| --- | --- |
| `vector` | pgvector HNSW search alone |
| `lexical` | Postgres full-text search alone, with document-frequency filtering |
| `identifier` | Exact identifier matching alone |
| `hybrid` | All three → RRF → section cap |
| `hybrid+rerank` | All three → RRF → Cohere rerank over the top 40 → section cap |
| `hybrid+cap+rerank` | All three → RRF → section cap → Cohere rerank (the original order) |
| `hybrid+rerank+nocap` | All three → RRF → Cohere rerank over the top 40, **no** section cap |
| `hybrid+nocap` | All three → RRF, **no** section cap (added after the first run) |
| `vector+rerank` | Vector top 20 → Cohere rerank → section cap (added after the first run) |

**Why the no-cap configuration exists.** The cap keeps at most two chunks per
section. That stops one long section from filling all five slots, but an
answer that needs three chunks of one section loses one. The other
configurations can compare the two cap orders; only a configuration without
the cap can say whether the cap itself helps or hurts.

Every configuration is cut to the same top 5. Comparing a top-20 vector list
with a top-5 reranked list would measure list length, not retrieval quality.

Each comparison answers one design question. Each pair was meant to differ
in exactly one thing; `vector` versus `hybrid` turned out to differ in two
(see *Results*):

| Comparison | Question |
| --- | --- |
| `vector` vs `hybrid` | Does fusing three retrievers beat vector search alone? |
| `hybrid` vs `hybrid+rerank` | Does reranking earn its cost? |
| `hybrid+cap+rerank` vs `hybrid+rerank` | Does rerank-then-cap beat cap-then-rerank? |
| `hybrid+rerank+nocap` vs `hybrid+rerank` | Does the section cap help after reranking? |
| `vector` vs `hybrid+nocap` | Does fusion beat vector search alone, with no cap on either? |
| `hybrid+nocap` vs `hybrid` | Does the section cap help without reranking? |
| `vector` vs `vector+rerank` | Does reranking help vector search alone? |
| `vector+rerank` vs `hybrid+rerank` | With a reranker, do the lexical and identifier retrievers still add anything? |

The last four were added after the first run. That run showed `hybrid` scoring
below `vector`, but the two differ in both fusion and the section cap;
`hybrid+nocap` separates them. And once reranking proved to be the stage that
pays, the obvious question was whether the reranker needs three retrievers to
feed it at all; `vector+rerank` answers that. Its candidates are the vector
top 20 — the depth the judged pool was built from — so every chunk it can
return has a label.
| `lexical` vs `identifier` | How do the two exact-match retrievers compare? |

---

## Metrics

All on the top 5, over answerable questions.

| Metric | Definition | What it diagnoses |
| --- | --- | --- |
| **Recall@5** | `any`: 1 if any ground-truth chunk is in the top 5. `all`: the fraction of ground-truth chunks found | Was the answer retrieved at all — a recall problem |
| **MRR@5** | 1 / rank of the first ground-truth chunk; 0 if none in the top 5 | How high the answer ranks — a ranking problem |
| **Hit@1** | 1 if the first result is a ground-truth chunk | What a user reading only the top result gets |
| **Intrusions** | Distractor-chapter chunks in the top 5 that are not ground truth | Whether the distractors compete, and whether reranking pushes them out |
| **Sections/q** | Distinct document sections in the top 5 | How varied the top 5 is |
| **3+ same** | Questions where one section fills three or more of the five slots | The monopoly the section cap exists to prevent; always 0 with the cap on |

Read together: high recall with low MRR means the answer is found but
misordered, which reranking can fix; low recall means it was never retrieved,
which no reranker can fix.

A distractor that the judges labelled correct is ground truth, so it is not
counted as an intrusion — 13 ground-truth chunks come from distractor chapters.

**Unanswerable questions** (`none`, 2 of them) have no recall to measure.
They are reported separately: the top rerank score each reranked
configuration gives them, next to the same score on answerable questions.
That is the first look at whether a low score could trigger a retry in the
agentic layer.

---

## Three checks that keep the numbers honest

### 1. Every returned chunk must have been judged

Step 06 built each question's pool from the three retrievers' top 20 — the
same lists every configuration draws from — so every chunk a configuration
returns should have a label. A returned chunk outside the pool would count as
wrong without anyone having looked at it.

The harness counts these per configuration. The expected number is **zero**.
Anything else means the retrieval and the pool have drifted apart, for example
a retrieval change after step 06, and the affected questions are listed in
`results.json`.

### 2. A silent rerank fallback must not be scored

`retrieval.rerank()` returns the fused order when the Cohere call fails. For a
live query that is the right behaviour: degrade, don't fail. For an evaluation
it is a trap. A rate-limited run would score plain hybrid retrieval under the
`hybrid+rerank` label, the numbers would look plausible, and reranking would
appear to add nothing.

The harness treats a reranked result without rerank scores as a failure:
retry with backoff (15, 30, 60 seconds), and stop the run if it keeps failing.
Completed runs are cached, so stopping costs nothing. Rerank calls are also
spaced (6.5 seconds by default, about 9 per minute) so a trial Cohere key stays
under its rate limit; `--cohere-interval` lowers that for a production key.

### 3. Differences smaller than two questions are noise

With 79 answerable questions, one question is more than one percentage point
of Recall@5. A mean difference of 0.013 sounds like a result and is a single
question.

So comparisons are reported as **question counts**: on how many questions
configuration B beats A, and the reverse. A net difference below two questions
is labelled noise. This is also why one known false positive in the golden set
(g21, see `06-golden-set.md`) cannot change a conclusion on its own.

---

## Operation

```bash
python scripts/07_evaluate.py                       # all nine configurations
python scripts/07_evaluate.py --configs vector hybrid
python scripts/07_evaluate.py --fresh               # ignore cached runs
python scripts/07_evaluate.py --cohere-interval 0.2 # production Cohere key
```

Cost: one embedding call per question for each configuration that uses vector
search, and one Cohere rerank call per question for each reranked
configuration — about 240 rerank calls, roughly 26 minutes at the default
spacing. Everything else is local Postgres.

Runs are cached by configuration, question id, and a hash of the question
text, for the same reason as the judgment cache in step 06: ids are positions,
the hash is the content.

---

## Results

84 questions (82 answerable, 2 unanswerable), nine configurations, top 5.
The set includes three parallel multi-hop questions (`all`) added in a second
labelling pass (`06-golden-set.md`); for those, Recall@5 is the fraction of
required chunks found.

| Configuration | Recall@5 | MRR@5 | Hit@1 | Intrusions/q | Sections/q | 3+ same | Unjudged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `vector` | 0.915 | 0.807 | 0.720 | 0.76 | 3.09 | 36 | 0 |
| `lexical` | 0.768 | 0.648 | 0.561 | 1.28 | 3.50 | 26 | 0 |
| `identifier` | 0.433 | 0.359 | 0.305 | 0.18 | 1.06 | 6 | 0 |
| `hybrid` | 0.872 | 0.751 | 0.646 | 0.99 | 3.78 | 0 | 0 |
| `hybrid+nocap` | 0.884 | 0.753 | 0.646 | 0.90 | 3.21 | 36 | 0 |
| `vector+rerank` | 0.945 | 0.913 | 0.890 | 0.72 | 3.66 | 0 | 0 |
| **`hybrid+rerank`** | **0.976** | **0.929** | **0.902** | 0.73 | 3.72 | 0 | 0 |
| `hybrid+cap+rerank` | 0.939 | 0.895 | 0.866 | 0.70 | 3.83 | 0 | 0 |
| `hybrid+rerank+nocap` | 0.963 | 0.925 | 0.902 | 0.57 | 3.15 | 31 | 0 |

Head-to-head on Recall@5, counted in questions (B better / A better):

| A → B | B better | A better | Net | Verdict |
| --- | --- | --- | --- | --- |
| `hybrid` → `hybrid+rerank` | 9 | 0 | +9 | **reranking helps** |
| `vector+rerank` → `hybrid+rerank` | 3 | 0 | +3 | **extra retrievers help, with a reranker** |
| `vector` → `vector+rerank` | 4 | 1 | +3 | reranking helps vector search too |
| `hybrid+cap+rerank` → `hybrid+rerank` | 3 | 0 | +3 | rerank-then-cap better |
| `vector` → `hybrid+nocap` | 3 | 5 | −2 | **fusion alone hurts** (at the threshold) |
| `vector` → `hybrid` (fusion + cap) | 3 | 6 | −3 | vector better |
| `hybrid+nocap` → `hybrid` | 0 | 1 | −1 | noise |
| `hybrid+rerank+nocap` → `hybrid+rerank` | 1 | 0 | +1 | noise |
| `lexical` → `identifier` | 5 | 32 | −27 | lexical better |

Recall@5 by question type (selected configurations):

| Type | n | `vector` | `identifier` | `hybrid` | `vector+rerank` | `hybrid+rerank` |
| --- | --- | --- | --- | --- | --- | --- |
| generated | 75 | 0.93 | 0.44 | 0.92 | 0.97 | 0.99 |
| multi-hop, parallel | 3 | 0.67 | 0.83 | 0.83 | 0.83 | **1.00** |
| multi-hop, sequential | 2 | 0.50 | 0.00 | 0.00 | 0.00 | 0.50 |
| same-section | 2 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 |

**Unjudged: 0 for every configuration.** Every chunk any configuration
returned had been labelled in step 06, so every score rests on a judgment,
not a default.

### The story in four lines

1. **Reranking is the stage that pays**: +9 questions over plain hybrid, none
   lost; Hit@1 from 0.646 to 0.902.
2. **Fusion without reranking hurts**: three retrievers fused by RRF lose to
   vector search alone (−2), and the section cap is not the cause (±1).
3. **Fusion with reranking helps**: `hybrid+rerank` beats `vector+rerank` by
   three questions and loses none, and is the only configuration that finds
   every required chunk of every multi-hop question.
4. **The section cap is about diversity, not recall**: removing it changes
   recall by one question but lets one section fill three or more of the five
   slots on 31 of 82 questions.

### Reranking is the stage that pays

The largest effect in the table. Against plain hybrid, reranking gains nine
questions and loses none, and the top result becomes right for nine
questions in ten instead of six and a half. The best configuration overall is
`hybrid+rerank`: Recall@5 0.976, MRR@5 0.929, Hit@1 0.902.

Reranking helps vector search alone too (+3), but the gain over plain hybrid
is three times larger. That asymmetry is the most interesting result here,
and the next two sections explain it.

### Fusion without reranking hurts — and the cap is not why

`hybrid` and `vector` differ in two things: fusion and the section cap.
`hybrid+nocap` separates them. Without the cap, fusion still loses to vector
search alone (−2, at the threshold); the cap on its own moves one question,
which is noise. **The drop is fusion's.** (On the first, 81-question run it
was −3; the three multi-hop questions, where fusion does well, narrowed it.)

The likely mechanism is the one documented in `04-retrieval.md`: RRF trusts
every list equally, and on this question set two of the three lists are weak.
`lexical` has the most distractor intrusions of any configuration (1.28 per
question), and `identifier` only fires on questions that name an identifier.
The generated questions were deliberately paraphrased into everyday language
so they would not echo the docs' wording, which is where embeddings are
strongest and exact-match retrieval weakest. Fused with equal weight, the weak
lists push noise into the top 5.

### Fusion with reranking helps

Once a cross-encoder reads the candidates, the picture flips.
`hybrid+rerank` beats `vector+rerank` by three questions, with no question
going the other way; MRR@5 0.929 against 0.913, Hit@1 0.902 against 0.890.

This is consistent with fusion's real contribution being **candidates, not
order**. The lexical and identifier retrievers surface chunks that vector
search ranks outside its top 20; fusion gets them into the pool the reranker
reads, and RRF's poor ordering no longer matters because the reranker
reorders everything. Without a reranker, the same extra candidates arrive with
noise attached and the noise wins. (Candidate-pool recall — whether the answer
is anywhere among the reranked candidates — was not measured directly, so
this is the most plausible reading, not a demonstrated one.)

**The multi-hop questions show where the extra retrievers earn their place.**
They name two settings each (`shared_buffers` and `effective_cache_size`, for
example). Vector search finds two thirds of the required chunks; the
identifier retriever, last on everything else, finds 0.83 — exact names are
exactly what it matches. Only `hybrid+rerank` finds every required chunk of
all three. Three questions are an illustration, not a measurement, but they
point the same way as the overall +3: when a query names identifiers, fused
candidates plus a reranker beat any single retriever.

### The section cap: diversity, not recall

| | Recall effect of the cap | Questions where one section fills 3+ slots |
| --- | --- | --- |
| Without reranking (`hybrid` vs `hybrid+nocap`) | −1 (noise) | 0 with cap, 36 without |
| With reranking (`hybrid+rerank` vs `hybrid+rerank+nocap`) | +1 (noise) | 0 with cap, 31 without |

The cap neither helps nor hurts recall measurably. What it changes is the
shape of the top 5: without it, more than a third of questions get three or
more chunks from one section. Whether that diversity matters is a question
for answer generation, not retrieval recall.

**Order matters, though.** Capping before reranking loses three questions
against reranking first. The two same-section questions show the mechanism:
`hybrid+cap+rerank` misses both, every reranked configuration that caps
afterwards (or not at all) gets both. When the cap runs first, it picks which
chunks of a section survive by RRF order, which cannot tell same-section
chunks apart; the right one is cut before the reranker sees it. Two questions
make this an illustration, not proof.

### `identifier` is a specialist

It loses to `lexical` on 32 questions because it returns nothing for
questions that don't name an identifier, and wins on five. On the multi-hop
questions, which name two settings each, it outscores vector search. Its value
is inside fusion, on the queries that name what they want.

### The rerank score as a retry signal

| | n | Top-1 rerank score (`hybrid+rerank`) |
| --- | --- | --- |
| Answerable | 82 | min 0.511, mean 0.796, max 0.931 |
| Unanswerable | 2 | 0.360, 0.481 |

Both unanswerable questions score below every answerable one, so on this data
a threshold around 0.5 would separate them. Encouraging for the agentic
layer's trigger, and too thin to rely on: two unanswerable questions, and a
margin of 0.03 between the highest of them and the lowest answerable score.

### Limits of these numbers

- **Small n.** 82 answerable questions. Several conclusions sit at or just
  above the two-question threshold; the per-type rows have two or three
  questions each and are illustrations.
- **Judges may favour the reranker.** An LLM judge's sense of "answers the
  question" may resemble a cross-encoder's sense of relevance
  (`06-golden-set.md`, *What this is not*). Cross-vendor judging reduces this;
  it does not remove it. The +9 is large enough that it is unlikely to be an
  artefact, but its exact size is uncertain.
- **One known false positive** in the labels (g21) can move one question at
  most.
- **The question mix favours vector search.** Paraphrased, everyday-language
  questions are where embeddings are strongest. A query log with more
  identifiers and error codes could change both fusion results.
- **No sequential multi-hop question.** The three `all` questions are
  parallel: both settings are named, so one good retrieval finds both halves.
  Nothing here measures retrieval that has to learn a term before it can
  search for it.