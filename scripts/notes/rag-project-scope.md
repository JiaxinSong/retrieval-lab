# RAG Project — Scope Plan

Three effort tiers for the same project. Each tier is a working, interview-ready
deliverable; later tiers add depth rather than replacing earlier work.

No frontend, no web API in any tier. Those are excluded deliberately —
interviewers ask about retrieval design and evaluation, not interface code.

**Priority key**

| Mark | Meaning |
| --- | --- |
| **MUST** | The project is not credible in an interview without this |
| **SHOULD** | Clear differentiator; highest return per hour |
| **NICE** | Real value, but "I know it, I chose not to prioritize it" is an acceptable answer |

---

## Stack (all tiers)

| Layer | Choice |
| --- | --- |
| Storage | PostgreSQL 17 + pgvector, local Docker |
| Embedding | OpenAI `text-embedding-3-small` (1536d) |
| Rerank | Cohere `rerank-v3.5` |
| Generation | `gpt-4o-mini` |
| Evaluation judges | `gpt-4o-mini` + `claude-haiku-4-5`, `claude-sonnet-4-5` as referee |
| Corpus | PostgreSQL official documentation |
| Framework | None — raw SDKs and SQL |

Rationale for no framework: LangChain abstracts away exactly the layers that
carry interview signal — chunking strategy, retrieval fusion, prompt
construction. Writing them directly means every decision is explicit and
defensible.

---

## Tier 1 — 9 hours

The minimum version that stands up to questioning.

| Item | Priority | Time | Status |
| --- | --- | --- | --- |
| Scrape PostgreSQL docs, eval + distractor split, 224 pages | **MUST** | 1.0h | done |
| Section-aligned chunking with heading chains, 1,293 chunks | **MUST** | 1.0h | done |
| Batched embedding → pgvector, HNSW + GIN indexes | **MUST** | 1.0h | done |
| Lexical retrieval via Postgres FTS, OR semantics | **MUST** | 0.5h | done |
| Document-frequency filtering (substitute for missing IDF) | *unplanned* | 0.5h | done |
| Identifier retriever over a `text[]` column | *unplanned* | 0.5h | done |
| RRF fusion, three retrievers | **MUST** | 1.0h | done |
| Section-diversity cap | *unplanned* | 0.3h | done |
| Cohere cross-encoder reranking to top-5 | **MUST** | 1.0h | done |
| Extract `retrieval.py` as a shared module | *unplanned* | 0.3h | done |
| Question generation with filtering | **MUST** | 0.8h | done |
| Paraphrase pass to raise question difficulty | *unplanned* | 0.3h | done |
| Word-boundary fix to the identifier tiebreak | *unplanned* | 0.2h | done |
| Rerank-before-cap reorder, both orders kept as configs | *unplanned* | 0.3h | done |
| Hand-written hard cases (parallel/sequential multi-hop, same-section, unanswerable) | *unplanned* | 0.5h | done |
| Selection redesigned to remove self-selection bias; pooling at `RECALL_K` | *unplanned* | 0.5h | done |
| Double-stemming fix in the lexical query | *unplanned* | 0.2h | done |
| Pipeline verified end-to-end against a local Postgres instance | *unplanned* | 0.3h | done |
| Golden set: two cross-vendor LLM judges + referee, calibration, set-valued any/all ground truth | **MUST** | 1.5h | built, not yet run |
| Eval harness: Recall@5, MRR, distractor intrusion | **MUST** | 1.0h | |
| Agentic layer: tool calling, relevance grading, query rewrite, retry | **MUST** | 2.0h | |

Items marked *unplanned* were added after measurement or review exposed
problems the original scope did not anticipate. Retrieval items are documented
in `04-retrieval.md`, question and ground-truth items in `05-questions.md`:

- **Document-frequency filtering** — `ts_rank` has no IDF component, so with
  OR semantics a stem in 40% of the corpus drowned out rare terms, and RRF
  amplified the noise rather than filtering it
- **Identifier retriever** — Postgres's text-search parser splits on
  underscores, so the lexical path could not match `work_mem` whole, which was
  the one capability the corpus was chosen to exercise
- **Section cap** — a long section splits into chunks that all score alike, and
  five of them monopolized one top-5; reranking cannot fix this because all
  five genuinely are relevant
- **Paraphrase pass** — 82% of generated questions retrieved their answer at
  rank 1, which would have made every configuration score near the ceiling and
  the comparison meaningless; rewriting each question without the source
  passage in context brought it to 67%
- **Hand-written hard cases** — paraphrasing changes vocabulary but not
  structure, so multi-hop, same-section, and unanswerable questions have to be
  written by hand; the unanswerable ones cannot be generated at all, since a
  generator working backwards from a chunk always produces a question with an
  answer

- **Word-boundary counting** — the identifier tiebreak counted substrings, so
  `maintenance_work_mem` was credited with `work_mem` mentions; same failure
  class as the tokenizer split
- **Rerank before cap** — the cap was choosing which same-section chunks survive
  by RRF rank, the one signal that cannot separate them; reordered on
  principle, with the original order kept for measurement

One item is not a feature but a correction, and arguably the most useful:

- **Set-valued ground truth** — a check keyed on a single chunk id reported a
  retrieval failure when the system was right; overlap had put the answer in
  two chunks. Ground truth is now a set with an explicit `any` / `all` rule,
  and retrieval is frozen until the harness can measure instead of eyeball

Two more came from a review pass rather than from symptoms:

- **Double stemming** — query lexemes were stemmed once correctly, then passed
  to `to_tsquery('english', ...)` and stemmed again; Snowball English is not
  idempotent, so `database`, `increase`, `release` silently stopped matching.
  Confirmed against a real Postgres instance: zero matches before, three after
- **Self-selection bias** — the planned question selection stratified by the
  evaluated system's own ranks, which would have fixed its score before
  measuring it; selection is now random by chapter, and pooling depth matches
  what every configuration can return

One planned item changed shape:

- **Automated ground truth** — manual annotation of 30 queries was replaced by
  two LLM judges from different vendors labelling every (question, chunk) pair
  in the pool, with a referee for yes/no conflicts, calibration on
  hand-verified cases, and Cohen's kappa reported. The question count is no
  longer fixed: every question that survives the judging rules is kept, and
  ambiguous ones are discarded rather than hand-fixed

The unplanned work is the part worth talking about in an interview. The planned
items are what a tutorial produces.

**What this gives you**

- Every layer of the retrieval architecture, explained from first principles
- One set of real evaluation numbers
- Agentic RAG, which most job descriptions now ask about

**Known weakness**

Only one evaluation run. Asked "how did you pick 512 tokens?", the honest
answer is "it's a common starting point" — which is weak. Tier 2 fixes exactly
this.

Four parameters are currently set by eye rather than measurement:
`RECALL_K`, `RRF_K`, `MAX_PER_SECTION`, `MAX_DOC_FRACTION`. Reading result
lists was enough to catch several outright bugs, and is not enough to claim any
of these values is right.

---

## Tier 2 — 15 hours (+6h)

The highest-value tier. Do this before applying if at all possible.

| Item | Priority | Time |
| --- | --- | --- |
| Parameter matrix: chunk size 256 / 512 / 1024 | **SHOULD** | 2.5h |
| Parameter matrix: rerank on/off, vector-only baseline | **SHOULD** | 1.5h |
| Contextual Retrieval — LLM-generated chunk context prepended before embedding | **SHOULD** | 2.0h |

**Contextual Retrieval, concretely**

Before embedding, prepend a one-line statement of where the chunk sits in the
document:

```
before: "Revenue reached $4.2 billion, a 15% increase."
after:  "From the Financial Highlights section of the 2025 annual report.
         Revenue reached $4.2 billion, a 15% increase."
```

Costs one LLM call per chunk at ingest time; prompt caching keeps it cheap.
Anthropic's published result is a substantial drop in retrieval failures. Few
candidates implement it, so it reads as depth rather than tutorial-following.

**What this unlocks in interviews**

- "How did you pick chunk size?" → three measured configurations, with the
  tradeoff between recall and token cost
- "Was reranking worth it?" → a number, not an opinion
- "What have you read recently?" → a technique you implemented, not just read

---

## Tier 3 — 40 hours (+25h)

Diminishing returns. Worth it only for specific target roles.

| Item | Priority | Time |
| --- | --- | --- |
| Full corpus (~8K chunks): tables, code blocks, cross-references | **NICE** | 5h |
| Extended parameter matrix: RRF `k`, recall `top_k`, embedding dim truncation | **NICE** | 6h |
| Human spot-check of LLM ground-truth labels, measured agreement with the judges | **NICE** | 3h |
| LLM-as-judge for groundedness and hallucination rate | **NICE** | 4h |
| Agentic: query decomposition for multi-hop, conversational query rewriting | **SHOULD if the role names agentic work** | 5h |

**When Tier 3 is justified**

The agentic items move to **SHOULD** if the job description explicitly asks for
agentic or multi-agent systems. Everything else here is genuinely optional —
"I know the technique and chose not to prioritize it" is a complete answer, and
sometimes a better one than having built it without reason.

---

## Recommended path

1. Build Tier 1. Get real numbers on the board.
2. Add Tier 2. This is where the project becomes distinctive.
3. **Start applying at Tier 2.** Do not wait for Tier 3 — polish the project and
   send applications in parallel.
4. Pull Tier 3 items in only when a specific role calls for them.

---

## Resume bullets

Fill the bracketed figures after the eval harness runs. Do not estimate them —
the first follow-up question is always how they were measured.

**RAG Retrieval Pipeline** — Python, PostgreSQL/pgvector, OpenAI, Cohere

- Built a retrieval-augmented generation pipeline from scratch without
  LangChain, indexing 1,293 section-aligned chunks of PostgreSQL documentation
  (512 tokens / 64 overlap) with heading-chain context prefixes
- Implemented three-way hybrid retrieval — HNSW vector search, Postgres
  full-text search with document-frequency filtering, and exact identifier
  matching — fused via Reciprocal Rank Fusion, then Cohere cross-encoder
  reranking to top-5
- Built an automated golden dataset with cross-vendor LLM judges, calibration,
  and set-valued ground truth, showing hybrid retrieval improved Recall@5 from
  **[X]** to **[Y]** over vector-only, with reranking adding **[Z]**
- Extended into agentic RAG with tool calling — the model grades retrieval
  relevance, rewrites queries, and retries, with a turn cap to bound cost

Pre-numbers version of the third bullet, safe to use immediately:

- Built an automated golden dataset and evaluation harness measuring Recall@5
  and MRR across vector-only, hybrid, and hybrid-plus-reranking configurations

The agentic bullet is only usable once step 08 is built.

**Skills additions:** `pgvector`, `hybrid search`, `reranking`,
`retrieval evaluation`

---

## Interview material to capture while building

Write these down as they happen — they are harder to reconstruct later than the
code is.

| Observation | Where it comes up |
| --- | --- |
| Cosine scores for paraphrase pairs vs error-code pairs | "Why hybrid search?" — concrete failure mode beats generalities |
| Why `ORDER BY embedding <=> $1` and not `ORDER BY similarity` | Index usage; a silent performance trap |
| Why RRF instead of weighted score fusion | Incompatible score scales, unstable normalization, one hyperparameter |
| Why rerank after recall, never instead of it | Cross-encoders cannot precompute; full-corpus scoring is infeasible |
| Why chunks stream before tokens | Retrieval finishes in ms, generation in seconds; sources before conclusions |
| What the model did with an out-of-corpus question | Hallucination control and the value of a similarity floor |
| The `temperature` parameter the Anthropic streaming helper rejects | A real abstraction-layer tradeoff: drop tuning knobs, raise on correctness params |
