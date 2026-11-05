# Interview Q&A — retrieval-lab

Each question has two parts. **Key points** are the short version to remember. **Answer** is the version to say out loud, about 30–60 seconds each.

Never quote a number that is not in this file or the step docs. The first follow-up is always "how did you measure that?".

---

## Numbers to remember

| What | Value |
| --- | --- |
| Corpus | PostgreSQL 17 docs, 224 pages, 21 chapters |
| Chunks | 1,293 (339 eval, 954 distractor) |
| Chunk size | Section-aligned, 512-token ceiling, 64-token overlap; min 35, max 522, mean 401 |
| Embedding | `text-embedding-3-small`, 1536 dims; whole corpus cost about $0.01 |
| Retrieval | 3 retrievers × top 20 → RRF (k=60) → Cohere rerank over top 40 → section cap 2 → top 5 |
| Cosine, paraphrase pair | ~0.54 |
| Cosine, `work_mem` vs its own definition | ~0.29 |
| Cosine, `error 42P01` vs "relation does not exist" | ~0.17 |
| Generated questions at rank 1 | 82% before paraphrasing, 67% after |
| Distractors in vector top-10 (5-query spot check) | mean 2.8 |
| Rerank top-1 score | 0.897 on a well-covered query, 0.405 when the answer was absent |
| Judges | `gpt-5-mini` + `claude-haiku-4-5`, `claude-sonnet-5` as referee |
| Golden set | 84 questions accepted of 92 (79 `any`, 3 multi-hop `all`, 2 `none`) |
| Judge agreement | Cohen's kappa 0.592; 70 yes/no contradictions, referee sided `no` 68 times |
| Multi-hop questions written from the chunk text | 5 written, 3 accepted — each with exactly the chunks predicted |
| Best configuration | `hybrid+rerank`: Recall@5 0.976, MRR@5 0.929, Hit@1 0.902 |
| Vector alone | Recall@5 0.915, MRR@5 0.807, Hit@1 0.720 |
| Vector + rerank | Recall@5 0.945, MRR@5 0.913, Hit@1 0.890 |
| Fusion without rerank, no cap | Recall@5 0.884 — worse than vector alone (net −2 questions) |
| Reranking vs hybrid | +9 questions, 0 lost; Hit@1 0.646 → 0.902 |
| Hybrid + rerank vs vector + rerank | +3 questions, 0 lost; only config to find every multi-hop chunk |
| Rerank top-1, unanswerable vs answerable | 0.36, 0.48 vs min 0.51 |
| Answers citing ground truth | 80 of 80 questions where retrieval found it; every multi-hop answer cited both chunks; 0 invalid citations |
| Refusals | 2 / 2 unanswerable refused; 0 answerable refused |
| Agentic loop forced on all 84 | searched again on 8; recovered 1 (hw5) — by query expansion from the model's own knowledge, not a learned second hop |
| Answer-judge calibration | 14 errors planted by code (6 changed values, 6 invented sentences, 2 half-removed multi-hop answers); both judges 20 / 20 after the fixes |
| Answer faithfulness | 0 of 82 answers unfaithful by both judges; 9 flagged by one (small general-knowledge additions) |
| Answer correctness | 69 of 82 correct by both judges; 4 flagged by both → read: 3 real (2 from retrieval misses), 1 judge error |
| Answer-judge kappa | 0.25 correctness, −0.05 faithfulness — flags are rare and don't overlap; raw agreement 70 / 82 and 73 / 82 |

---

## Problems found and how they were solved

For "What challenges did you run into?", "Tell me about a bug", "What went wrong?", or "What would you do differently?". Each row is one story: the problem, how it surfaced, the fix, and where the full answer is.

### Priority

| Mark | Meaning |
| --- | --- |
| ★★★ | Must know cold. Each teaches something general that interviewers probe for; tell these first |
| ★★ | Good second stories, or follow-ups when the interviewer digs into one area |
| ★ | Only if asked about that exact area; true, but small or narrow |

The eight ★★★ stories:

| Story | What it shows |
| --- | --- |
| `work_mem` split into `work` + `mem` | Verifying that a capability exists before relying on it |
| No IDF, and RRF amplifying the noise | Knowing BM25 versus `ts_rank`, and how fusion fails |
| 82% of questions answered at rank 1 | A test set that can't discriminate measures nothing |
| Self-selection bias | The tested system must not choose its own test |
| The false alarm: one answer in two chunks | Checking the check; why ground truth is a set |
| A judge right for the wrong reason | Not trusting LLM output, even when the label looks correct |
| Multi-hop questions rewritten from the data | Find where the answer is before writing the question; predictions matched the judges exactly |
| Calibrating a judge with planted errors | Testing a judge without hand labels; a judge that skimmed past the last sentence |

### Which ones to tell

Pick two or three ★★★ stories from different categories, so the answer shows range:

1. **A retrieval bug** — the underscore split or the missing IDF. Concrete, technical, easy to follow. Double stemming (★★) is a good backup.
2. **An evaluation-design problem** — the 82% too-easy test set, or self-selection bias. Shows you question your own measurements.
3. **An LLM-judge problem** — the right label for the wrong reason, or the planted-error calibration that caught a judge skimming. Shows you don't trust model output blindly.

Shape every story the same way: **what I saw → why it happened → what I changed → what it taught me.** Lead with the symptom, not the fix.

### Data and chunking

| Priority | Problem | How it surfaced | Fix | Detail |
| --- | --- | --- | --- | --- |
| ★★ | Section intros silently dropped: selecting only subsections skipped the conceptual "why" text before them | Auditing what the chunker discarded, not reading code | A pass that keeps sect1 intro text | Q5, `02-chunking.md` |
| ★ | Headings duplicated in the body, plus `#` permalinks and non-breaking spaces; doubled term frequency for title words | Reading sample chunks before paying for embeddings | `normalize()` and dropping headings from the body; 6% of tokens were pure duplication | `02-chunking.md` |
| ★★ | Corpus too small: with a few hundred chunks retrieval was artificially easy and reranking looked useless | Thinking about what the numbers would mean | 16 distractor chapters chosen for vocabulary overlap | Q26, `01-corpus.md` |

### Retrieval

| Priority | Problem | How it surfaced | Fix | Detail |
| --- | --- | --- | --- | --- |
| ★ | AND semantics: sentence-style queries returned zero rows (3 of 5 test queries) | Lexical results were empty | OR semantics | `04-retrieval.md` |
| ★★★ | OR without IDF: common words drowned rare ones; RRF amplified the noise and pushed the right chunk from 1st to 4th | Comparing vector-only with fused results | Document-frequency filtering via `ts_stat` | Q10 |
| ★★★ | Postgres splits `work_mem` into `work` + `mem`, so lexical search could not match identifiers — the reason the corpus was chosen | Inspecting the indexed lexemes | A third retriever over an exact identifier column | Q9 |
| ★ | Identifier tiebreak counted substrings, crediting `maintenance_work_mem` with `work_mem` mentions | Results that ranked the wrong parameter first | Word-boundary matching | `04-retrieval.md` |
| ★★ | Double stemming: `database` → `databas` → `databa`, matching nothing; no error | Review pass; confirmed on a real Postgres 17 | Build the tsquery as a literal and cast it | Q16 |
| ★★ | `ORDER BY` an alias skips the HNSW index: correct results, silently slow | Checking the query plan | Repeat the distance expression in `ORDER BY` | Q16, `03-ingest.md` |
| ★★ | One long section filled the whole top 5 | Reading result lists | Section cap; then moved it after reranking so the cap trims a list ordered by actual relevance | Q14 |

### Evaluation design

| Priority | Problem | How it surfaced | Fix | Detail |
| --- | --- | --- | --- | --- |
| ★★★ | Test set too easy: 82% of generated questions retrieved their answer at rank 1 | Rank distribution of the first run | Paraphrase pass that never sees the source passage → 67% | Q20 |
| ★★★ | Self-selection bias: the plan picked questions by the tested system's own ranks, fixing its score in advance | Reasoning about the selection plan before running it | Rule: nothing that decides the test set may depend on the system under test | Q21 |
| ★★★ | False alarm: a check keyed on one chunk id reported a retrieval failure when the system was right (overlap put the answer in two chunks) | Re-reading the full chunk text | Ground truth is a set with an `any` / `all` match rule | Q23 |
| ★★ | Pool too shallow: depth 10, but the reranker lifts chunks from rank 17 into the top 5 | Tracing where reranked results came from | Pool depth equals the retrieval depth, so every top 5 is judged | Q24 |
| ★★ | Hard hand-written questions would silently drop out if the pool came only from retrieval | Asking what happens when all retrievers miss | Seed terms written with knowledge of the answer, independent of retrieval | `06-golden-set.md` |
| ★★ | A comparison meant to isolate fusion also differed in the section cap, so its result (hybrid below vector) couldn't be attributed | Reading the first run's results against the design | Added a configuration without the cap: the drop was fusion's. Then added vector + rerank to test whether the reranker needs three retrievers | `07-evaluation.md` |
| ★★ | The agentic gate's threshold (0.5) was read off the same questions it would be tested on, so it routed every answerable question to the direct path by construction | Noticed before the run | A second run with the loop forced on every question, to measure the loop independently of the gate | `08-agent.md` |
| ★★★ | Multi-hop questions written from memory all failed labelling; rewritten from the data, the predicted chunks were exactly what the judges chose | First run: 0 `all` questions | Co-occurrence check across chunks, read the defining chunks, word each question around facts in exactly one chunk; 3 of 5 accepted, each with the predicted chunk pair | `06-golden-set.md` |
| ★★ | No genuine multi-hop question survived: sequential ones turned out answerable by one chunk, parallel ones were discarded | The match rules in the results: 0 `all` | Documented as a gap; the set measures single-fact retrieval. More multi-hop questions before the agentic layer | `06-golden-set.md` |
| ★ | A rule discarding questions where a distractor ranked first was itself biased | Same self-selection reasoning | Judge distractors like any chunk; one that answers becomes ground truth | Q26 |

### LLM judges

| Priority | Problem | How it surfaced | Fix | Detail |
| --- | --- | --- | --- | --- |
| ★★ | Both judges agreed on a wrong label: an ambiguous paraphrased question ("non-zero I/O time") also fit replication lag times | Spot check of accepted questions; the stored answers showed both judges reading it the same wrong way | Left unedited (no hand-fixing labels); bounded to one `any` question that also has correct chunks. Lesson: agreement is not correctness | `06-golden-set.md` |
| ★★ | A judge demanded the question's own words, marking a correct passage `partial` because it answered by correcting the premise | Calibration failed; re-read the source to confirm the expectation was right | Answer-first schema and a premise rule in the prompt; no relaxing of the test | Q28 |
| ★★★ | A judge gave the right label for the wrong reason: "yes", while answering the opposite of the text | Calibration failed again; the stored answer showed the misreading | Replaced the judge model; calibration now checks the answer's substance, not just the label | Q28 |
| ★ | Temperature 0 unavailable: the new Anthropic SDK rejects the parameter; the replacement OpenAI judge is a reasoning model | Preflight failed with a `TypeError` before any spending | Reproducibility from the cache instead of sampling settings | Q29 |
| ★ | The referee model reaches end-of-life in a month, which would break any later rerun | Deprecation warning on the first run | Switched to a current model before it was ever called | `06-golden-set.md` |
| ★★★ | No hand labels to calibrate the answer judges, and Haiku passed answers containing an invented sentence ("added at the request of the Debian packaging team") as fully supported | Calibration on errors planted by code: 2 of 6 invented sentences missed | Prompt v2: one line per sentence, "supported by [n]" or "NOT SUPPORTED", before the verdict → 6 of 6 | Q40, `09-answer-judge.md` |
| ★★ | A planted error that wasn't an error: after "removing half" of a multi-hop answer, both judges still called it correct — the removed fact survived in another sentence | Both judges' reasons quoted the fact that was supposedly gone | Keep only sentences citing exactly the remaining chunk; the bar stayed at 90%. Same lesson as the false alarm: check the check | Q40, `09-answer-judge.md` |
| ★★ | The correctness judge marked down a true fact the answer took from a cited, non-reference passage, despite the prompt saying not to | Reading the four answers both judges flagged, against the chunks | Counted as a judge error, not an answer error; fix (show the judge the cited passages too) listed as next step | Q39, `09-answer-judge.md` |
| ★ | Answer-judge kappa near zero although raw agreement was 89% | Faithfulness kappa −0.05 | Rare flags make kappa collapse; report raw agreement and flag overlap beside it, and treat only shared flags as findings | Q39 |

### Engineering

| Priority | Problem | How it surfaced | Fix | Detail |
| --- | --- | --- | --- | --- |
| ★★ | The answer tool's schema was only a request: the model once omitted a required field and crashed the run on question 10 | First real run of step 08 | `strict: true` on the tool definitions, so the API enforces the schema; plus one retry with the error reported | `08-agent.md` |
| ★★ | Judgment cache keyed on positions (question id, chunk id), so regenerated questions would silently reuse other questions' labels | Code review before the first real run | Hash of the exact question and chunk text in the key | Q30 |
| ★ | Seeded sampling without `ORDER BY`: same seed, different row order, different sample | Code review | `ORDER BY id` | Q30 |
| ★ | A demo printed rerank scores on rows from before reranking, which looked like a data bug | Output that should have been impossible | Shared objects were mutated later; print each stage before the next runs | `04-retrieval.md` |

### A ready answer

> The most interesting problems were the silent ones. Postgres's text parser splits `work_mem` into `work` and `mem`, so my lexical search couldn't match identifiers — the very thing I chose the corpus for — and nothing errored; I added a third retriever for exact identifiers. On the evaluation side, my first test set was too easy: 82% of questions retrieved their answer first, so every configuration would have scored near perfect. I rewrote the questions without showing the model the source and brought that to 67%. And my LLM judges failed calibration twice — once a judge gave the right label while stating the opposite of what the passage says. I only caught that because I store each judge's answer, not just its label, and I replaced that model. The common thread is that none of these crashed; they just quietly made the numbers mean less.

---

# Part 1 — Overview

## Q1. Walk me through the project.

**Key points**
- Retrieval system built from scratch in plain Python, no LangChain, over the PostgreSQL docs
- Three retrievers → RRF fusion → Cohere rerank → section cap
- The emphasis is evaluation: question generation, hand-written hard cases, LLM-judged ground truth, no self-selection bias

**Answer**

I built a retrieval system from scratch over the PostgreSQL documentation, in plain Python with no LangChain, because I wanted to see what each stage actually contributes.

Three retrievers run in parallel: vector search in pgvector, Postgres full-text search, and an exact-match retriever for identifiers like `work_mem`. Their results are merged with Reciprocal Rank Fusion, reranked with a Cohere cross-encoder, and capped so one long section can't fill the top five.

The part I spent the most care on is evaluation. I generated questions from the docs, rewrote them so they don't copy the source wording, added hand-written hard cases like multi-hop and unanswerable questions, and had two LLM judges from different vendors label the ground truth. I made sure the system under test never decides which questions go into the test set, because that would bias the score.

---

## Q2. Why build it from scratch instead of using LangChain or LlamaIndex?

**Key points**
- Frameworks hide the layers with the most signal: chunking, fusion, prompt construction
- Writing them myself means every decision needs a stated reason
- Concrete payoff: several bugs only surfaced because I owned the code (underscore split, double stemming)
- Not anti-framework; in production I might use one

**Answer**

Frameworks hide exactly the layers I wanted to understand: chunking, fusion, and how the query reaches the database. Writing them myself meant every decision had to be explicit.

It paid off. Several real bugs only surfaced because I owned that code. For example, Postgres's text parser splits `work_mem` into `work` and `mem`, and my lexical query was being stemmed twice. A framework would have hidden both behind a default.

I'm not against frameworks. In production, with a team, I'd weigh them on maintenance cost. For learning what actually moves retrieval quality, building it directly was the better choice.

---

## Q3. Why the PostgreSQL docs as the corpus?

**Key points**
- Identifier-dense (`work_mem`, `pg_stat_activity`), which is vector search's weak spot → shows why hybrid matters
- Backend background means I can judge whether an answer is right
- Regular structure (DocBook HTML) allows section-aligned chunking
- Public, permissively licensed

**Answer**

Three reasons. First, it's dense with exact identifiers like `work_mem` and `pg_stat_activity`, which is exactly where vector search is weak. I measured it: a paraphrase pair scores about 0.54 cosine, but `work_mem` against its own definition only 0.29. A corpus of plain prose wouldn't show why hybrid retrieval matters.

Second, I know the domain from backend work, so I can tell whether a retrieved chunk actually answers a question. That matters for building ground truth.

Third, the HTML is generated from DocBook, so the section structure is regular and I could chunk on real topic boundaries.

---

## Q4. Why pgvector instead of a dedicated vector database?

**Key points**
- One database holds vectors, a full-text index, an identifier array, and JSONB metadata
- All three retrievers live in one SQL system; no second store to keep in sync
- Small corpus (1,293 chunks); HNSW is more than enough
- Consider a dedicated vector DB at tens of millions of vectors or for managed scaling

**Answer**

All three of my retrievers need different index types over the same rows: an HNSW index for vectors, a GIN index on a `tsvector` for full-text search, and a GIN index on a text array for identifiers. Postgres holds all of them in one table, so there's no second store to keep in sync and no consistency problem between them.

At this scale, about 1,300 chunks, HNSW in pgvector is more than enough. I'd look at a dedicated vector database if the corpus grew into tens of millions of vectors, or if I needed features like built-in sharding or managed scaling. The decision is about operational simplicity, not about pgvector being the fastest option.

---

# Part 2 — Chunking

## Q5. How did you chunk the documents?

**Key points**
- Not fixed-size. Split on DocBook subsections (`div.sect2`); one section = one parameter or concept
- 512 is a ceiling: a section that fits stays whole; only sections over 512 fall back to windows inside the section, 64-token overlap
- Mean of 401 shows most sections fit without splitting
- Also added sect1 intros (the conceptual "why" content), which were dropped at first

**Answer**

I chunk on the document's own section boundaries. The docs mark every subsection with a `div.sect2`, and each one is usually one parameter or one concept, so a section becomes one chunk.

512 tokens is a ceiling, not a target. If a section fits, it stays whole. Only sections longer than that fall back to 512-token windows with 64 tokens of overlap, and those windows never cross into another section. The mean chunk size is 401 tokens, which shows most sections fit without splitting.

One bug I caught by auditing: selecting only subsections silently dropped the intro text that sits before the first subsection. Those intros are the conceptual overviews, the "why" content, so I added a pass to keep them.

---

## Q6. Why not semantic chunking?

**Key points**
- Semantic chunking *guesses* topic boundaries in unstructured text; these docs already mark them
- Adjacent sections are too similar: `19.4.1 Memory` is one memory parameter after another, so similarity never drops
- Identifiers embed poorly (0.29), so the similarity signal is weakest on exactly this content
- Costs an embedding per sentence and a threshold to tune; a new threshold renumbers every chunk
- Right tool for unstructured text: transcripts, chat logs

**Answer**

Semantic chunking infers topic boundaries from drops in embedding similarity between sentences. That's useful when text has no structure, but here the authors already marked the boundaries: each DocBook subsection is one parameter or concept. An explicit boundary beats an inferred one.

It would actually struggle on this corpus. In the memory settings section, `work_mem`, `hash_mem_multiplier`, and `maintenance_work_mem` follow one another, and their embeddings are all close, so similarity never drops enough to split them. Identifiers also embed poorly, so the similarity signal is weakest on exactly the content this project cares about.

It also costs an embedding per sentence and a threshold to tune, and changing the threshold renumbers every chunk. I'd use it for unstructured text like transcripts.

---

## Q7. Why 512 tokens and 64 overlap?

**Key points**
- Honest answer: 512 is a common starting point, not tuned
- Rationale: small enough to stay on one topic, large enough to hold a complete answer
- Overlap verified in practice: the key `work_mem` sentence crosses two chunks and appears in both
- Next step: measure 256 / 1024; the evaluation harness already supports it

**Answer**

Honestly, 512 is a starting point, not a tuned value. It's small enough that a chunk usually covers one topic and large enough to hold a complete answer.

The overlap I can justify with a real case: the description of `work_mem` crosses the boundary between two chunks, and the 64-token overlap puts its key sentence in both. Without it, that sentence could have been cut in half.

Measuring chunk size is my next step: run the same evaluation at 256, 512, and 1024. The harness already supports it. The one complication is that changing chunk size renumbers every chunk, so I store ground truth by section id and content as well as by chunk id.

---

## Q8. What is the heading chain prefix?

**Key points**
- Each chunk is prefixed with its position in the docs, e.g. `# 19.4. Resource Consumption > 19.4.1. Memory`
- A chunk read in isolation loses its topic; the prefix restores it before embedding
- Zero-cost version of Anthropic's Contextual Retrieval (which uses an LLM call per chunk)
- Cost: about 5% of the token window

**Answer**

Every chunk is prefixed with its heading chain, like "Resource Consumption > Memory", before it's embedded. A chunk lifted out of its document loses the topic it belongs to. Text about `work_mem` with no heading is just a paragraph about memory; with the chain, the vector also encodes that it's about server configuration.

It's a zero-cost version of Anthropic's Contextual Retrieval, which uses an LLM call per chunk to write that context. Here the document's own hierarchy supplies it. The cost is roughly 5% of the token window. Comparing it against the LLM-generated version is a measurement I'd like to run.

---

# Part 3 — Retrieval

## Q9. Why three retrievers? Why not just vector search?

**Key points**
- Vector search handles paraphrase, not exact identifiers or error codes (0.29, 0.17)
- Lexical search is the opposite: exact words, but no match without word overlap
- Third retriever: Postgres splits `work_mem` into `work` + `mem`, so lexical search can't match whole identifiers
- The three fail on different queries; fused, they cover each other

**Answer**

Each retriever fails in a different place. Vector search handles paraphrase well but is weak on exact identifiers and error codes: `work_mem` against its own definition scores only 0.29 cosine. Lexical search is the opposite: it matches exact words but fails when the question shares no words with the answer.

The third retriever exists because of a bug I found. Postgres's text parser splits on underscores, so `work_mem` is indexed as `work` and `mem`, and `maintenance_work_mem` shares two of those three pieces. The lexical side couldn't match identifiers at all, which was the whole reason I chose this corpus. So I extract snake_case identifiers into a separate text-array column and match them exactly.

Because the three fail on different queries, fusing them covers more than any one alone.

---

## Q10. Is your lexical search BM25?

**Key points**
- No. Postgres `ts_rank` is a TF-IDF variant: no IDF, no term-frequency saturation, no length normalization
- Without IDF, common terms drown rare ones under OR semantics, and RRF amplifies the noise (correct chunk fell from 1st to 4th)
- Fix: `ts_stat` gives each term's document frequency; terms in more than 15% of chunks are dropped
- A blunt substitute (keep-or-drop, no weighting), chosen deliberately

**Answer**

No, and I'm careful to say so. Postgres's `ts_rank` has no inverse document frequency, so with OR semantics a common stem like `queri` counts as much as a rare one like `pg_locks`. I measured the damage: on one query, fusion pushed the correct chunk from first to fourth, because the noisy lexical list was voting for the wrong chunks.

My fix is document-frequency filtering. I compute how many chunks contain each term using `ts_stat`, and drop query terms that appear in more than 15% of the corpus. It's a blunt substitute for IDF, keep-or-drop instead of a continuous weight, but it fixes the measured failure without reimplementing the scorer. I'd rather say "I implemented the missing half of BM25" than claim I implemented BM25.

---

## Q11. Why RRF instead of a weighted sum of scores?

**Key points**
- The three score scales are incompatible: cosine distance, `ts_rank`, match count
- Normalization is unstable: every query has a different score distribution
- RRF uses ranks only: `1/(k + rank)`, k = 60
- One hyperparameter, no weights to tune
- Weakness: if one list is noise, RRF fuses the noise too (why df filtering was added)

**Answer**

The three retrievers produce scores on incompatible scales: cosine distance, `ts_rank`, and a match count. Adding them needs normalization, and normalization is unstable because each query has a different score distribution.

RRF ignores scores and uses only ranks: each list contributes one over k plus rank, with k at 60. A chunk ranked well by several retrievers rises; there's one parameter and nothing to tune per query.

Its weakness is that it trusts every list equally. If one retriever outputs noise, RRF fuses the noise too. That's exactly what happened with the lexical side before I added document-frequency filtering.

---

## Q12. How does the lexical retriever turn a question into a query?

**Key points**
- The keywords come from the question itself; no LLM involved
- `to_tsvector('english', question)`: split into words, lowercase, drop stop words, stem (`connection` → `connect`), and split on underscores (`work_mem` → `work` + `mem`)
- Drop terms that appear in more than 15% of chunks (document frequency computed at ingest) — a stand-in for the IDF that `ts_rank` lacks
- If every term was dropped, fall back to the unfiltered terms rather than return nothing
- Join with OR, not AND: AND returned zero rows for 3 of 5 sentence-style test queries
- Every chunk was processed the same way at ingest (`content_tsv`), so both sides speak the same lexemes; `ts_rank` orders the matches

**Answer**

The keywords come straight from the question; there's no model involved. Postgres runs the question through the same English text-search configuration used to index the chunks: it splits it into words, lowercases them, drops stop words like "is" and "a", and stems the rest, so "connection" becomes "connect". It also splits on underscores, which is why `work_mem` becomes `work` and `mem`, and why I needed a separate identifier retriever.

Next I drop any term that appears in more than 15% of chunks. I computed those document frequencies at ingest. It's a blunt substitute for IDF, which Postgres's ranking doesn't have. If that removes everything, I fall back to the unfiltered terms.

The remaining terms are joined with OR. My first version used AND, and three of five natural-language test queries returned nothing, because no single chunk contained every word. The query runs against a column where every chunk was processed by exactly the same rules, and `ts_rank` orders the matches.

---

## Q13. What does reranking do? Why not rerank the whole corpus?

**Key points**
- The first three retrievers compare separate representations of question and document (bi-encoder or term match)
- A cross-encoder reads question and passage together, so it judges whether the passage answers the question
- Slow: one model call per (question, passage) pair, nothing precomputable
- So recall first, rerank second: top 20 from each retriever → RRF → top 40 to Cohere

**Answer**

My three retrievers compare representations: the question and the document are encoded or tokenized separately, then matched. A cross-encoder reads the question and the passage together, so it can judge whether the passage actually answers the question rather than just being about the same topic.

The cost is that nothing can be precomputed: every question-passage pair is a model call. Scoring the whole corpus per query is infeasible, so reranking always comes after recall, never instead of it. I take the top 20 from each retriever, fuse them, and rerank the top 40 fused candidates.

There's a useful side effect: the top rerank score is a confidence signal. It was 0.897 on a well-covered question and 0.405 on one whose answer wasn't in the corpus. I plan to use that as the trigger for an agentic retry.

---

## Q14. What is the section cap, and why does it come after reranking?

**Key points**
- A long section splits into chunks with identical heading, section id, URL, and similar scores
- Observed: five chunks of one section filled the whole top 5
- Reranking can't fix it: all five genuinely are relevant
- Cap: at most 2 chunks per section
- Order: rerank first, then cap. Capping first lets the RRF rank choose survivors, and RRF can't separate same-section chunks
- Both orders kept as configurations; the evaluation decides

**Answer**

A long section splits into many chunks with the same heading, section id, and URL, and they tend to score alike. I saw five chunks from one section take the entire top five. Reranking doesn't fix that, because all five genuinely are relevant. So I cap each section at two chunks.

The order matters. If the cap runs first, it decides which two chunks survive using the fused rank, which is the one signal that can't tell same-section chunks apart. If reranking runs first, the cap trims a list the cross-encoder has already ordered by actually reading the text.

That's the argument from principle. I kept both orders as configurations so the evaluation decides, rather than my reasoning.

---

## Q15. If an answer needs three chunks from one section, doesn't the cap lose one?

**Key points**
- Yes — the cap trades depth within a section for coverage across sections
- Current impact on the test set is small: no `all` question survived, and an `any` question needs just one chunk; with rerank before the cap, the two most relevant chunks of a section survive
- Real users can hit it: "which settings in the memory section affect sorting?" spans several chunks of one section
- Alternatives: raise the cap, add neighbouring chunks (id ± 1), MMR (penalise near-duplicates instead of a hard count), give generation a longer list than the 5 being evaluated, or let the agentic layer retrieve again
- Measured rather than argued: a configuration runs reranking with no cap at all — recall changes by one question (noise), but one section fills 3+ slots on 30 of 79 questions

**Answer**

Yes, and that's a real trade-off, not a free win. The cap keeps at most two chunks per section, so it stops one long section from filling all five slots, but an answer spread across three chunks of one section loses one.

On my test set the impact is small: no genuine multi-hop question survived labelling, and for a question that any one correct chunk answers, reranking first means the two most relevant chunks of a section are the ones kept. A real user could still hit it, though, with a question that spans a section listing a dozen related settings.

There are softer alternatives: a higher cap, pulling in neighbouring chunks, or MMR, which penalises near-duplicates instead of counting. But rather than argue it, I added a configuration with no cap at all. Comparing it with the capped version shows directly whether the cap costs recall, and how often one section would otherwise take three or more of the five slots.

---

## Q16. Tell me about a bug you found.

**Key points**
- **Double stemming**: query terms were stemmed correctly, then passed to `to_tsquery('english', ...)` and stemmed again. Snowball isn't idempotent: `database` → `databas` → `databa`, which matches nothing. Fix: build a tsquery literal and cast with `::tsquery`
- **ORDER BY trap**: sorting on an alias makes pgvector skip the HNSW index; results stay correct but slow, no error
- **Underscore split**: see Q9
- Common thread: silent bugs that lower quality without raising errors

**Answer**

My favourite is double stemming, because it was completely silent. My lexical search stemmed query terms once, correctly, and then passed them to `to_tsquery` with the English config, which stemmed them again. The Snowball stemmer isn't idempotent: "database" becomes "databas", and stemming again gives "databa", which matches nothing in the index. Common words like "database", "increase", and "release" just stopped matching.

There was no error. Retrieval quietly got worse. I confirmed it against a real Postgres 17 instance and fixed it by building the tsquery as a literal and casting it, so the terms aren't processed twice.

The lesson I took from this and the underscore bug: retrieval bugs rarely crash. They lower quality, and you only find them by inspecting what each stage actually produced.

---

# Part 4 — Evaluation

## Q17. How do you evaluate the system?

**Key points**
- Retrieval is evaluated on its own; generation isn't built yet
- Golden dataset: questions + the set of chunks that answer each
- Metrics: Recall@5 (correct chunk in top 5), MRR (rank of first correct chunk), distractor intrusion
- Nine configurations: each retriever alone, hybrid with and without the section cap, hybrid+rerank, hybrid+cap-then-rerank, hybrid+rerank without the cap, vector+rerank
- Purpose: compare configurations, not produce an absolute score

**Answer**

I evaluate retrieval separately from generation. If retrieval misses the answer, no prompt can fix it, so I want that number on its own.

The golden dataset is a set of questions, each with the set of chunks that answer it. I measure Recall@5, whether a correct chunk reaches the top five; MRR, how high the first correct chunk ranks; and distractor intrusion, how often a chunk from an unrelated chapter takes a slot.

I run nine configurations: each retriever alone, hybrid fusion with and without the section cap, hybrid with reranking in both orders relative to the cap, reranking with no cap, and vector search with reranking, which asks whether the reranker needs three retrievers to feed it. The goal is comparing configurations against each other, so I can say what each stage contributes, rather than producing one absolute score.

---

## Q18. Which configurations do you compare, and why compare them in pairs?

**Key points**
- Nine configurations, all cut to the same top 5: `vector`, `lexical`, `identifier`, `hybrid` (fuse + cap), `hybrid+nocap`, `hybrid+rerank` (fuse → rerank → cap), `hybrid+cap+rerank` (the original order), `hybrid+rerank+nocap`, `vector+rerank`
- Each pair differs in exactly one thing, so a difference can only come from that thing — a controlled comparison
- The first run had one pair that didn't: `vector` vs `hybrid` also differed in the section cap. Fixed by adding `hybrid+nocap` — the drop turned out to be fusion's, not the cap's
- After reranking proved decisive, added `vector+rerank` to ask whether the reranker needs three retrievers at all
- Reported as question counts: "B better on 8, A better on 0", not just averages

**Answer**

I compare nine configurations, all returning the same top five, because comparing a twenty-item list with a five-item list would measure list length.

The useful comparisons are pairs that differ in exactly one thing. My first run had a pair that didn't: vector search against hybrid, where hybrid also applied the section cap. Hybrid scored lower, and I couldn't say whether fusion or the cap caused it. So I added hybrid without the cap. It still lost to vector search, which pinned the drop on fusion. Then, once reranking turned out to be the stage that pays, I added vector search with reranking, to ask whether the reranker needs the other two retrievers at all.

And I report each comparison per question: on how many questions B beats A, and the reverse. An average hides that; a count makes it obvious when a difference is two questions out of eighty.

---

## Q19. How do you keep the evaluation numbers honest?

**Key points**
- **Unjudged results:** every returned chunk should have been judged, because the pool came from the same top-20 lists; the count must be 0, otherwise retrieval and pool have drifted
- **Silent rerank fallback:** on a Cohere error the pipeline returns fused order instead of failing — right for users, a trap for evaluation (hybrid scored under the rerank label). Every reranked result must carry a rerank score; missing scores → retry with backoff, then stop
- **Small differences are noise:** with ~80 questions one question is over a percentage point; a net difference under 2 questions is labelled noise
- Plus: cached runs so an interruption costs nothing, and rerank calls spaced to stay under the rate limit

**Answer**

Three checks. First, every chunk a configuration returns should already have a label, because the judging pool was built from the same top-20 lists every configuration draws from. The harness counts unjudged results, and the expected number is zero. Anything else means the retrieval changed after labelling.

Second, my rerank step falls back to the fused order if the Cohere call fails. For a live query that's the right design: degrade instead of erroring. In an evaluation it's a trap, because a rate-limited run would score plain hybrid retrieval under the rerank label and nothing would look wrong. So every reranked result must carry a rerank score; if it doesn't, the harness retries with backoff and stops rather than score it.

Third, with about eighty questions, one question is more than a percentage point. I report differences as question counts and treat anything under two questions as noise. That also bounds the effect of the one false positive I found in the labels.

---

## Q20. How did you build the questions?

**Key points**
- Reverse-engineer questions from chunks: sample a chunk, ask an LLM what question it answers
- Problem: too easy. 82% retrieved their answer at rank 1 → every configuration near perfect, comparison meaningless
- Cause: writing while looking at the passage mirrors its concepts one to one
- Fix: a paraphrase pass that **never sees the source passage**, in an engineer's voice → 67%
- Leakage check: drop questions sharing 5 consecutive words with the source
- Paraphrasing can't change structure → hand-written hard cases: multi-hop, same-section, unanswerable

**Answer**

I reverse-engineered questions from chunks: sample a chunk, ask an LLM what question it answers. That guarantees every question has an answer and I know where it is.

The first run had a problem: 82% of questions retrieved their answer at rank 1. That sounds great but it's a broken test set. Every configuration would score near perfect and the comparison would show nothing. The cause is structural: a question written while looking at the passage mirrors its concepts one to one.

So I added a paraphrase pass that rewrites each question in an engineer's voice without seeing the source passage. A model can't copy what it can't see. That brought rank-1 down to 67%. I also drop questions that share five consecutive words with the source.

Paraphrasing changes vocabulary but not structure, so I hand-wrote the shapes generation can't produce: multi-hop questions, questions where several chunks share a section, and questions with no answer in the corpus.

---

## Q21. What is self-selection bias, and how did you avoid it?

**Key points**
- First plan: pick questions by hybrid's own rank (some at rank 1, some at 6+) to make the set harder
- Problem: that fixes hybrid's score at selection time, while other configurations were never selected on → comparison tilted in advance
- Rule: **nothing that decides the test set may depend on the system under test**
- Now only leakage (text-only) removes questions; retrieval-based checks are recorded, never used
- Development queries are excluded too (they're the training set)

**Answer**

My first plan was to make the test set harder by choosing questions based on where hybrid retrieval ranked the answer: some at rank one, some at rank six or worse. Then I realized that fixes hybrid's score at selection time. Questions where it ranks the answer sixth are misses by construction. And the other configurations weren't selected on at all, so the comparison is tilted before a single query runs.

The rule I adopted: nothing that decides the test set may depend on the system being tested. Now the only filter that removes questions is leakage, which compares text only. Checks that consult retrieval are recorded as diagnostics but never used for selection. I also excluded every query I'd looked at while developing retrieval, since testing on them is testing on the training set.

---

## Q22. How do you know your ground-truth labels are correct?

**Key points**
- Two judges from different vendors (OpenAI + Anthropic) independently label each (question, chunk): yes / partial / no
- Different vendors: one model run twice differs only by noise; vendors differ when their standards do
- Keep the weaker label; a direct yes-vs-no contradiction goes to a stronger referee model
- Judges must pass hand-verified calibration cases first, or the run stops
- Inter-judge agreement reported as Cohen's kappa
- Be upfront: LLM-labelled, not a human benchmark; fit for comparing configurations

**Answer**

Two LLM judges from different vendors, OpenAI and Anthropic, label every question-chunk pair independently as yes, partial, or no, reading one passage at a time. Different vendors matter: two runs of the same model only disagree by noise, while two vendors disagree when their standards differ, which is the disagreement worth catching.

When they disagree, I keep the weaker label, so the ground truth is conservative. A direct yes-versus-no contradiction goes to a third, stronger model.

Before any real labelling, the judges must pass calibration cases whose answers I verified by reading the source; any failure stops the run. I also report inter-judge agreement as Cohen's kappa.

Calibration has already earned its place: it stopped the run twice on real judge errors, which I can walk through.

I'm explicit that this is an LLM-judged set, not a human benchmark. It's suited to comparing configurations against each other, which is its purpose.

---

## Q23. Why is ground truth a set of chunks, not one chunk?

**Key points**
- Keying ground truth on one chunk id once produced a false retrieval failure
- Cause: the `work_mem` description crosses two chunks; the overlap puts the key sentence in both, and the system found the other one
- Two kinds of "two chunks" mean opposite things:
  - Each chunk answers alone → duplication (overlap) → `any`: one hit counts
  - Only together → true multi-hop → `all`: every piece needed
- Conflating them inflates scores

**Answer**

I learned this by getting it wrong. I tested whether the section cap was deleting correct answers by checking for one specific chunk id, and the check said yes. But the answering sentence crossed a chunk boundary, and through the overlap it appeared in two chunks. The other one was in the top five the whole time. The system was right; my check was wrong.

So ground truth is a set with a match rule. If several chunks each answer on their own, that's duplication, scored as "any": one hit counts. If no single chunk answers but several together do, that's true multi-hop, scored as "all". Mixing them up inflates scores, because a multi-hop question scored as "any" counts as solved with half its evidence.

---

## Q24. How do you decide which chunks to judge for each question? (pooling)

**Key points**
- Judging all 1,293 chunks per question is too expensive
- Pool: union of the three retrievers' top 20 + the source chunk and its same-section neighbours (generated questions) + keyword matches on seed terms written with knowledge of the answer (hand-written questions)
- The last two don't depend on retrieval: if every retriever misses the answer, it is still judged, so the question stays in the test set and every configuration scores zero on it instead of the question silently disappearing
- Seed terms are not the lexical retriever: they search for words that aren't in the question (`visibility map` for "index-only scans aren't happening")
- Why 20: every configuration draws from those top-20 lists, so any top 5 is inside the pool; nothing returned goes unjudged
- An early pool depth of 10 was too shallow: the reranker lifted a chunk from rank 17 into the top 5
- This is TREC-style pooling

**Answer**

Judging every chunk for every question is too expensive, so I use pooling, the approach TREC evaluations use. Each question's pool is the union of all three retrievers' top 20, plus answers added independently of retrieval: for generated questions, the source chunk and its neighbours in the same section; for hand-written ones, chunks matching seed terms I wrote knowing the answer, often words the question never uses, like "visibility map" for a question about index-only scans.

That second part matters. If the pool came only from retrieval, a question every retriever fails on would get no correct label and be discarded, so the hardest questions would silently leave the test set, chosen by the system under test. With the answer guaranteed to be judged, the question stays and every configuration scores zero on it, which is exactly the failure the evaluation should show.

The depth is chosen deliberately. Every configuration I evaluate draws from those same top-20 lists, so any configuration's top five is guaranteed to be inside the pool. Nothing a configuration returns goes unjudged. An earlier version pooled at 10, and I found the reranker lifting a chunk from rank 17 into the top five, which would have been scored wrong just because nobody looked at it.

---

## Q25. Recall@5 versus MRR — what does each tell you?

**Key points**
- Recall@5: did a correct chunk reach the top 5 → recall problem
- MRR: how high the first correct chunk ranks → ranking problem
- Read together:
  - High recall, low MRR → found but misordered → reranking helps
  - Low recall → never retrieved → reranking can't help; fix recall upstream
- For `all` multi-hop questions, recall is the fraction of required chunks found

**Answer**

They answer different questions. Recall@5 asks whether a correct chunk made it into the top five at all. MRR asks how high the first correct chunk ranks.

Together they diagnose the problem. High recall with low MRR means the right chunks were found but in the wrong order, which is what reranking fixes. Low recall means the right chunks were never retrieved, and no reranker can recover them; that's a recall problem to fix upstream, in chunking or the retrievers. For multi-hop questions scored as "all", recall is the fraction of required chunks found.

---

## Q26. What are distractors, and why add them?

**Key points**
- 21 chapters in the corpus, questions drawn from 5; the other 16 are distractors
- A small corpus makes retrieval look easy and reranking look useless (nothing to rerank past)
- Distractors chosen for vocabulary overlap: full-text search also covers GIN indexes, locking overlaps lock monitoring
- Spot check: 2.8 distractors on average in the vector top 10 → they really compete
- A distractor that genuinely answers is labelled correct by the judges, not counted as intrusion

**Answer**

Questions come from five chapters, but the corpus has 21. The other 16 are distractors: they compete for top-k slots but are never the source of a question.

A small index makes retrieval artificially easy. With a few hundred chunks, the answer is almost always in the top five, reranking has nothing to rerank past, and the numbers describe the corpus size more than the architecture.

I chose distractors for vocabulary overlap, not volume. The full-text search chapter also discusses GIN indexes; the locking chapter overlaps with lock monitoring. A spot check found about three distractors in the vector top ten on average, so they really compete. And if a distractor genuinely answers a question, the judges label it correct rather than counting it as an intrusion.

---

## Q27. What are the risks of using an LLM as a judge?

**Key points**
- A judge's sense of "answers the question" may resemble a cross-encoder's relevance → may flatter rerank configurations
- Cross-vendor judging reduces this, doesn't remove it
- Position and length bias: avoided by one passage per call
- A judge can give the right label for the wrong reason (seen in practice, Q28); storing its answer makes that visible
- Guards: calibration cases, kappa, weaker label, spot check
- Not an absolute quality measure

**Answer**

The main one I worry about is shared bias. A judge's sense of "this answers the question" might resemble a cross-encoder's sense of relevance, which could flatter the reranking configurations, the very thing I'm trying to measure. Using judges from two vendors reduces that; it doesn't remove it, and I say so in the README.

Other known issues are position and length bias when a judge sees several passages at once. I avoid that by giving each judge one passage per call.

A subtler risk is a judge giving the right label for the wrong reason. I've seen it happen: a judge labelled a passage correctly while stating the opposite of what it says. That's why each judgment stores the judge's answer, not just its label.

The guards I have are calibration cases verified by hand, inter-judge agreement, keeping the weaker label, and a spot check of accepted questions. With those, I trust it for comparing configurations, not as an absolute quality measure.

---

## Q28. Did the judge calibration ever catch anything?

**Key points**
- Yes, it stopped the run twice, and both times the problem was real
- First run: Haiku labelled the `work_mem` definition `partial` because the passage never says the words "per-connection". It answers by correcting the question's premise (it's per operation), and the judge didn't count that
- Before changing anything, re-read the source: the hand-verified expectation was right, so the case was not relaxed
- Fix 1: an `answer` field before `label` (the judge writes the answer, then labels it), plus a rule that correcting a wrong premise counts as answering. Prompt version bumped so old judgments aren't reused
- Second run: gpt-4o-mini labelled the definition `yes` while answering "Yes, work_mem is a per-connection limit" — the opposite of the text. Right label, wrong reason
- Fix 2: no prompt fixes reading comprehension, so the judge was replaced with a stronger model; calibration now also checks the answer's substance, not just the label
- The other calibration cases guard against overcorrection: if the judges became lenient, the wrong-parameter chunk or an unanswerable question would start getting `yes`

**Answer**

Yes, twice, and both were real problems. Before any labelling, the judges have to pass cases where I verified the answer myself. One is "Is work_mem a per-connection limit?" against the chunk that defines `work_mem`.

The first run failed because Haiku labelled that chunk partial. Its reason was that the passage "does not explicitly state whether it is a per-connection limit." The passage says the limit applies per query operation and multiplies across operations and sessions. It answers the question by correcting its premise, and the judge wanted the question's own words. Before touching anything I re-read the source to check whether my expectation was wrong. It wasn't, so I fixed the judge, not the test: the judge now writes its answer before choosing a label, and the prompt says that correcting a wrong premise counts as answering. That matters beyond this one case, because my question generator deliberately writes questions in a user's vocabulary, often with a mistaken premise.

The second run failed on the OpenAI judge, and the new answer field showed something worse. It labelled the definition "yes" while answering "Yes, work_mem is a per-connection limit", the opposite of the text. The label was right by accident. A check on labels alone would have passed it. No prompt fixes reading comprehension, so I replaced it with a stronger model and added a check on the substance of the answer, not just the label.

What I took from it: calibration only works if a failure makes you investigate rather than adjust the threshold. And storing the judge's reasoning is what made the failures diagnosable.

---

## Q29. How do you make LLM judging reproducible?

**Key points**
- Every judgment is cached the moment it returns, in an append-only JSONL file; reruns read the cache instead of resampling
- Temperature 0 turned out not to be available: the current Anthropic SDK rejects the parameter, and the OpenAI judge is a reasoning model that takes a reasoning effort instead
- So reproducibility comes from the cache, not from sampling settings; any remaining variance shows up in the inter-judge agreement
- Cache key: prompt version, model, question id, chunk id, and a hash of the exact question and chunk text
- Each part invalidates for a reason: a new prompt or model means the old label was made under different conditions; the hash catches changed text

**Answer**

The obvious answer is temperature zero, and it turned out not to be available. The current Anthropic SDK no longer accepts a temperature parameter, which my first run hit as a TypeError at the preflight check, and the OpenAI judge I ended up with is a reasoning model that takes a reasoning effort instead.

So reproducibility comes from caching. Every judgment is written to an append-only file the moment it returns, and a rerun reads it back instead of calling the model. That also makes the run resumable: an interruption loses only the calls in flight, and a failed call is never written, so the next run retries it.

The cache key is the prompt version, the model, the question and chunk ids, and a hash of the exact text the judge saw. Each part is there for a reason: a new prompt or a new model means the old label was made under different conditions, and the hash means changed text is always judged again.

---

## Q30. Tell me about a silent bug in your evaluation code.

**Key points**
- **Cache keyed on positions.** The first cache key was prompt version + model + question id + chunk id. Question ids are positions (`g00` is whichever candidate comes first); chunk ids point at different text after re-chunking. Regenerating questions would have silently reused judgments made for different questions. Fix: add a hash of the question and chunk text. Caught in review before the first real run
- **Seeded sampling that wasn't reproducible.** The question generator seeded `random.sample`, but the SQL had no `ORDER BY`. `random.sample` picks positions, so the same seed over rows in a different order picks different chunks. It worked by accident because rows came back in insert order. Fix: `ORDER BY id`
- Same class as the retrieval bugs: no error, just a promise (correct labels, reproducible sample) that wasn't actually kept

**Answer**

Two, both in the same family: something that looked like a guarantee but wasn't one.

The first was the judgment cache. I keyed it on prompt version, model, question id, and chunk id. But those ids are positions, not content. Question "g00" is just whichever candidate comes first in the file, and a chunk id points at different text after re-chunking. If I'd regenerated the questions, the cache would have handed back judgments made for different questions, with no error, and the ground truth would have been quietly wrong. I added a hash of the exact question and chunk text to the key. I caught it reading the code before the first real run, so nothing was affected.

The second was in question sampling. I seeded the random sample for reproducibility, but the SQL feeding it had no ORDER BY. Postgres doesn't guarantee row order, and `random.sample` picks positions, so the same seed over a different order picks different chunks. It worked only because rows happened to come back in insert order. Adding ORDER BY made the guarantee real.

The lesson is the same as with the retrieval bugs: evaluation code fails silently. It doesn't crash; it just stops measuring what you think it measures.

---

## Q31. What were the results?

**Key points**
- 84 questions (82 answerable, 3 of them multi-hop), nine configurations, top 5; every returned chunk had been judged (0 unjudged)
- Best: hybrid + rerank — Recall@5 0.976, MRR 0.929, Hit@1 0.902; vector alone 0.915 / 0.720
- Reranking is the stage that pays: +9 questions over hybrid, 0 lost; Hit@1 0.646 → 0.902
- Fusion *without* reranking hurts: −2 against vector alone, and the cap isn't the cause (±1)
- Fusion *with* reranking helps: hybrid+rerank beats vector+rerank by 3, loses 0, and is the only configuration that finds every chunk of every multi-hop question — fusion supplies candidates, the reranker supplies order
- Rerank before cap beats cap before rerank by 3; the cap itself is about diversity (31 of 82 questions get 3+ chunks from one section without it), not recall
- Answers: every answer whose retrieval found the ground truth cited it, multi-hop answers cited both halves, no answerable question refused, both unanswerable ones refused
- Answer quality: no answer unfaithful by both judges; 69 of 82 correct by both; of 3 confirmed problems, 2 trace back to the only 2 retrieval misses

**Answer**

On 82 answerable questions, the best configuration was hybrid retrieval with reranking: Recall@5 0.976 and Hit@1 0.902, against 0.915 and 0.720 for vector search alone. Every chunk any configuration returned had been judged.

The interesting part is how that breaks down. Reranking is the stage that pays: nine questions better than plain hybrid, none worse. But fusing three retrievers without a reranker was worse than vector search alone. My questions are paraphrased into everyday language, where embeddings are strongest, and RRF gives the noisy lexical list equal weight. With a reranker, the picture flips: hybrid plus rerank beats vector plus rerank by three questions and loses none. So fusion's real contribution is candidates, not ordering. On the multi-hop questions, which name two settings each, hybrid plus rerank was the only configuration that found both halves every time.

On the answer side, every answer whose retrieval found the right chunk cited it, the multi-hop answers cited both chunks they needed, and the model refused exactly the two questions the docs can't answer and none of the others. Judging the answer text itself, no answer was flagged as unfaithful by both judges, and 69 of 82 were correct by both. Of the three answers I confirmed as wrong or incomplete, two were the only two questions where retrieval had missed the ground truth, so retrieval is still where the errors start.

---

# Part 5 — Beyond retrieval

## Q32. How would you handle multi-hop questions?

**Key points**
- Parallel multi-hop (two independent facts): decompose up front, retrieve each
- Sequential multi-hop (step two depends on step one): needs iterative retrieval; can't be decomposed in advance
  - Example: index-only scans not happening → learn they depend on the visibility map → then search what maintains it
- Trigger: rerank top-1 score; low = no single passage covers the question → retrieve again
- Whether a clean threshold exists is what the evaluation data will show
- Built: a rerank-score gate at 0.5 and a `search_docs` tool, up to 3 searches
- Parallel: three multi-hop questions written from the chunk text; hybrid + rerank found both halves of all three on the first search, and every answer cited both
- Sequential: on the one sequential question first retrieval missed, the second query named the missing setting and found it — but the term came from the model's own knowledge, not the first search's passages: query expansion, not a true second hop

**Answer**

There are two kinds, and they need different handling. Parallel multi-hop, like "what's the difference between `log_min_duration_statement` and `log_statement`", is two independent lookups, so decomposing the question up front works.

Sequential multi-hop is harder. "Index-only scans aren't happening, what maintenance affects that?" First you learn that index-only scans depend on the visibility map; only then can you search for what maintains the visibility map. You can't write the second query in advance, so it needs iterative retrieval.

For deciding when to retry, I use the top rerank score as a gate: below 0.5, the model gets a search tool and may rewrite the query, up to three searches. On my one sequential question that first retrieval missed, the second query named `default_statistics_target`, which the question never mentions, and found the answer. But I checked: none of the first search's passages mentioned that setting, so the model supplied it from its own knowledge. That's query expansion, not a true second hop. I did add parallel multi-hop questions, and retrieval handled them on the first search: both settings are named in the question, so one good retrieval finds both halves, and the answers cited both. A sequential question that survives labelling is still missing, and that's the case the loop is for.

---

## Q33. Did the agentic layer help?

**Key points**
- Default gate (0.5): only the 2 unanswerable questions took the agentic path — the threshold was read off this same data, so that's by construction
- Forced the loop on all 84 to test the loop itself: the model searched again on only 8, refused nothing it shouldn't, recovered 1 question
- 1 question is below my 2-question noise threshold, so: no evidence it helps in general, clear evidence it doesn't hurt
- Why so little: first retrieval already finds 80 of 82, and the multi-hop questions that survived are parallel ones a single search already answers
- Interesting pattern: 6 of 8 rewrites turned everyday language into identifiers (`pg_table_size`, `syslog_facility`, `default_statistics_target`) — exactly what the weak lexical and identifier retrievers need

**Answer**

Honestly, on my test set it barely had a chance to. With the default gate, only the two unanswerable questions took the agentic path, and that's partly by construction: I picked the 0.5 threshold from these same questions' scores.

So I ran it a second time with the loop forced on every question, to test the loop rather than the gate. Offered a second search everywhere, the model used it on only eight of 84, didn't refuse a single answerable question, and recovered one question first retrieval had missed. That one was a sequential question: the second query named the planner setting the question never mentioned. When I checked, the first search's passages never mentioned it either, so the model brought that term from its own knowledge. Useful, but it's query expansion, not chaining evidence across searches. But one question is below the noise threshold I use, so what I can claim is that the loop doesn't hurt, not that it helps. The reason is the test set: first retrieval already finds 80 of 82, and the multi-hop questions that survived labelling are parallel ones, where both settings are named and one search finds both halves.

The most interesting thing was what the rewrites looked like. Six of eight turned a natural-language question into identifiers, like table-size function names or a setting name. That's exactly the vocabulary my lexical and identifier retrievers are good at, and the ones that underperformed on everyday-language questions. So the loop and the hybrid retrievers look complementary. Proving it needs harder questions.

---

## Q34. How would you handle questions the corpus can't answer?

**Key points**
- Hand-wrote unanswerable test questions from the adjacent domain (Citus, pgpool-II), not unrelated topics (Redis is trivially refused)
- Near-domain questions retrieve plausible chunks, which is what tempts a model to invent an answer
- Signal: low rerank top-1 score (0.405 vs 0.897)
- In generation: the answer tool has an `answerable` flag, and the prompt says to answer only from the passages and say when they don't contain the answer
- Result: both near-domain questions refused; none of the 79 answerable ones refused

**Answer**

I wrote unanswerable test questions on purpose, and drew them from PostgreSQL's neighbourhood: Citus for sharding, pgpool-II for query caching. Asking about Redis would be trivially refused and measure nothing. Near-domain questions are hard because retrieval returns plausible chunks about sharding or connection pooling, and that's exactly the situation that tempts a model into a confident wrong answer.

The rerank score helps here too: it was 0.405 on a question whose answer wasn't in the corpus, versus 0.897 on a well-covered one. In generation, the model answers through a tool with an `answerable` flag and is told to use only the passages. Both near-domain questions were refused, and none of the 79 answerable ones was, so the refusal isn't coming from a model that refuses everything.

---

## Q35. What are the latency and cost of a query?

**Key points**
- The three retrievers run concurrently with `asyncio.gather`, not in sequence
- The slowest step is the Cohere rerank network call
- No end-to-end benchmark yet; don't quote milliseconds
- Cost: embedding the whole corpus ≈ $0.01; each query = one embedding call + one rerank call

**Answer**

The three retrievers run concurrently with `asyncio.gather`, so retrieval latency is roughly the slowest of the three rather than their sum. The Postgres queries are fast at this scale; the dominant cost per query is the Cohere rerank call over the network. I haven't run a proper end-to-end latency benchmark, so I won't quote a number.

On cost, embedding the whole corpus was about a cent. Each query costs one embedding call and one rerank call. If latency mattered, the first things I'd look at are skipping rerank when the fused top results already agree, and caching embeddings for repeated queries.

---

## Q36. How would this scale to millions of documents in production?

**Key points**
- HNSW lives mostly in memory; tune `m`, `ef_construction`, `ef_search`
- Partition by tenant or topic; at larger scale, consider a dedicated vector DB
- Incremental ingest: process only changed documents, not a full rebuild
- Chunk id stability: currently full rebuild + RESTART IDENTITY; production needs content-based ids
- Evaluation: the golden set grows with the corpus and runs as a regression suite

**Answer**

A few things would change. HNSW indexes live mostly in memory, so I'd tune its parameters for the recall-latency trade-off and partition by tenant or topic. At tens of millions of vectors, a dedicated vector database becomes worth its operational cost.

Ingest would become incremental. Right now I rebuild the whole table and reset ids so chunk ids stay stable for the golden set. In production I'd use content-based ids so a document update doesn't renumber everything.

And evaluation would run continuously: the golden set grows with the corpus, and every retrieval change runs against it as a regression test, the same way the six configurations are compared now.

---

## Q37. What would you do differently, or do next?

**Key points**
- Next: multi-hop questions that survive labelling (to actually test the agentic loop); a correctness judge that also sees the cited passages; then chunk size (256 / 512 / 1024) and Contextual Retrieval
- Should have verified earlier: the corpus choice rested on lexical search matching identifiers, which it couldn't
- Lesson: build evaluation before tuning retrieval; reading result lists by eye led to a wrong conclusion (the false alarm)

**Answer**

The biggest lesson is to build evaluation before tuning retrieval. Twice I drew conclusions by reading result lists, and once I was simply wrong: the false alarm where my check recognised only one of two correct chunks. After that I froze retrieval until I could measure it.

I'd also verify assumptions earlier. I chose this corpus because it's full of identifiers, assuming lexical search could match them. It couldn't, because of the underscore split, and I only found out later. The reasoning was right, but the capability behind it didn't exist yet.

Next steps, in order: multi-hop questions that survive labelling, because the agentic loop is built but barely tested; a fix to the answer judge I found when reading its flags — the correctness judge only sees the reference chunks, so it marks down true facts the answer took from other cited passages; then chunk size and LLM-generated context.

---

## Q38. What was the hardest part?

**Key points**
- Not the code; making the evaluation trustworthy
- Examples: too-easy questions (82%), self-selection bias, ground truth as a set, pool depth
- Each was a trap where everything looks fine but the numbers would mean nothing

**Answer**

Not the retrieval code. The hardest part was making the evaluation trustworthy, because every mistake there looks fine on the surface.

A test set where 82% of answers rank first looks like a great system. Choosing questions by the system's own ranks looks like making the test harder. Keying ground truth on one chunk id looks precise. Each of those would have produced numbers that mean nothing, and none of them throws an error. Most of my time went into finding those traps and designing around them.

---

# Part 6 — Answer quality

## Q39. How do you know the answers are right? Do they hallucinate?

**Key points**
- Citations aren't enough: an answer can cite the right chunk and still say "8MB" where it says 4MB
- Two checks per answer, two judges (gpt-5-mini, which also wrote the answers, and claude-haiku-4-5 to offset self-preference):
  - **Faithfulness**: answer vs the passages it cites — is every claim supported? This is the hallucination check
  - **Correctness**: answer vs the ground-truth chunks — correct, incomplete, or incorrect
- Weaker verdict wins; only flags both judges share are treated as findings
- Results on 82 answers: 0 unfaithful by both judges; 69 correct by both; 4 flagged by both
- Read those 4 against the chunks: 3 real, 1 judge error
  - hw5, g28: incomplete — the only 2 questions where retrieval missed the ground truth; the model answered confidently from what it had, without saying what was missing
  - g58: a reasoning error on a vague question (one bullet argued the opposite case)
  - g25: judge error — a true fact from a cited non-reference passage, marked down
- Kappa is low (0.25, −0.05): rare flags make kappa collapse; raw agreement 70 / 82 and 73 / 82

**Answer**

Citations alone don't tell you. An answer can cite exactly the right chunk and still misread it, say "the default is 8MB" next to a passage that says 4MB. So I added a second judging step over the answer text, with two checks. Faithfulness compares the answer to the passages it cites and asks whether every claim is supported; that's the hallucination check. Correctness compares it to the ground-truth chunks and grades it correct, incomplete, or incorrect.

The answers were written by gpt-5-mini, and a model grading its own output tends to be lenient, so the second judge is Claude Haiku, and the weaker verdict wins.

No answer was flagged as unfaithful by both judges, and 69 of 82 were correct by both. Four were flagged by both, and I read those against the chunks. Three were real. Two were incomplete, and they were exactly the two questions where retrieval had missed the ground truth: the model answered confidently from whatever it had, without saying anything was missing. The third was a reasoning error on a vague question. The fourth was the judge's mistake: it marked down a true fact the answer took from a cited passage that just wasn't one of the reference chunks.

So the takeaway is that the errors start in retrieval, and the next thing I'd improve is making the model say when its passages only answer part of the question.

---

## Q40. How did you calibrate the answer judges without hand labels?

**Key points**
- No known-correct answers to test against, so: **plant known errors in copies of real answers, by code**
  - Change a value the reference states (4MB → 8MB) → must be unfaithful and not correct
  - Append an invented sentence with a citation ("added at the request of the Debian packaging team [1]"), used only if "Debian" appears in none of the cited passages → must be unfaithful
  - Remove half of a multi-hop answer → must not be correct
- Code made the error, so the expected verdict is certain without anyone reading it; each judge must catch 90% or the run stops
- Caught: Haiku passed 2 of 6 invented sentences as supported — it judged the answer as a whole and skimmed the last line. Fix: one line per sentence before the verdict → 6 of 6
- Also caught my own bug: a "half-removed" answer still contained the removed fact in another sentence; both judges said correct, and their reasons quoted the fact. The test was wrong, not the judges
- Limit: proves the judges catch clear errors, not that they never flag correct answers

**Answer**

For the retrieval labels I calibrated on cases I'd checked by hand. For answers I wanted something that didn't need a human, so I planted errors with code. I took copies of real answers and broke them in ways I could control: changed a value the reference states, like 4MB to 8MB; appended an invented sentence with a citation, only if its keyword appeared nowhere in the cited passages; or removed half of a multi-hop answer. Because the code made the error, the expected verdict is certain, and each judge has to catch 90% before the real run.

It caught two things. First, Haiku passed two answers with the invented sentence as fully supported. It was judging the answer as a whole and skimming past the last line. I changed the prompt so the judge writes one line per sentence, "supported by passage n" or "not supported", before giving a verdict, and it then caught all six. That's the same fix as my retrieval judges: make the model show what it read before it decides.

Second, it caught a bug of mine. One "half-removed" answer was still judged correct by both judges, and their reasons quoted the very fact I thought I'd removed. It had survived in another sentence. So the judges were right and my test was wrong; I fixed the removal rule rather than lowering the bar.

The limit is that planted errors only test one direction. They show the judges catch clear errors, not that they never flag a correct answer. For that side I only have the two judges' agreement, which is why I read the cases both of them flagged.

