# retrieval-lab — instructions for Claude Code

Three-way hybrid retrieval over the PostgreSQL 17 docs (vector + full-text + identifier → RRF → Cohere rerank → section cap) with an LLM-judged evaluation pipeline. Plain Python, no LangChain. Portfolio project for AI engineering job applications: every design decision must be explainable in an interview.

Read `README.md` first, then the step docs in `scripts/notes/` for whatever step you are touching.

## Rules

- **Code is English only.** No Chinese in code, comments, docstrings, prompts, or log output. Explanations to the user in chat are in Chinese.
- **Keep docs in sync with code.** When a script changes behaviour, update the matching doc in `scripts/notes/` (`0N-*.md`) in the same change. Each doc highlights the problems found and how they were solved.
- **Table names include the project name** (`retrieval_lab_chunks`, `retrieval_lab_term_stats`).
- **Never fabricate.** Do not quote PostgreSQL documentation from memory; query the chunk text in the database first. Do not estimate metrics; numbers come from running code.
- **Explain before changing.** Say what you plan to change and why before editing, and keep changes small enough to review. The user needs to understand every line.
- **Ask before spending money.** Anything that calls OpenAI, Anthropic, or Cohere at scale (re-ingest, `06_build_golden.py` without `--dry-run`, evaluation over all configs) needs a go-ahead and a cost estimate first.
- Prefer clean, explicit code over clever one-liners.

## Environment

- Python 3.12 venv at `.venv/` (`source .venv/bin/activate`). Run scripts from the project root.
- Postgres 17 + pgvector in Docker: container `pgvector`, database `ragdb`. Inspect with `docker exec -it pgvector psql -U postgres -d ragdb`.
- API keys and `DATABASE_URL` are in `.env` (loaded with `load_dotenv(find_dotenv())`). Never print or commit them.

## Layout

- `retrieval.py` — shared retrieval module; all retrieval logic lives here. `CONFIGS` holds the nine configurations under evaluation.
- `scripts/01_fetch_docs.py` … `09_judge_answers.py` — one script per pipeline step.
- `scripts/notes/` — design doc per step (`01-corpus.md` … `09-answer-judge.md`), plus `interview-qa.md` and `rag-project-scope.md`.
- `data/processed/chunks.json` — frozen corpus, committed. `data/raw/html/` is gitignored.
- `eval/` — `candidates.json` now; `pools.json`, `golden.json`, `discarded.json`, `judge_report.json`, `judgments.jsonl` after step 06.

## Invariants — do not break

- **Chunk ids** are assigned by `03_ingest.py` in `chunks.json` order (`TRUNCATE ... RESTART IDENTITY`, sequential inserts). Ground truth in `eval/` is keyed by these ids. Re-chunking changes ids and invalidates `eval/`.
- **pgvector ordering**: `ORDER BY embedding <=> $1` must repeat the distance expression; ordering by an alias skips the HNSW index.
- **Lexical query**: the tsquery is built as a literal and cast with `::tsquery`. Passing stemmed lexemes to `to_tsquery('english', ...)` stems them twice (Snowball is not idempotent).
- **Identifiers**: the full-text parser splits on underscores, which is why the identifier retriever exists. Match identifiers with word boundaries (`\m ... \M`), not substrings.
- **Rerank before cap** in `hybrid_rerank_search`; the cap-first order is kept as a separate config for comparison.
- **No self-selection bias**: nothing that decides the test set may depend on the system under test. Only leakage removes generated questions.
- **Ground truth is a set** with match rule `any`, `all`, or `none`.

## Status

- Done: 01 fetch, 02 chunk (1,293 chunks), 03 ingest, 04 retrieval (frozen), 05 question generation (`eval/candidates.json`, 80 candidates), 06 golden set (`eval/golden.json`: 84 questions, 79 `any`, 3 `all`, 2 `none`; kappa 0.592; the 3 `all` questions were added in a second pass written from the chunk text). Details in `scripts/notes/06-golden-set.md`, *Results*.
- Known gap: no sequential multi-hop question survived labelling; the 3 `all` questions are parallel.
- Done: 07 evaluation (`eval/results_summary.json`): best is `hybrid+rerank`, Recall@5 0.976 on 84 questions; fusion without rerank loses to vector alone; see `scripts/notes/07-evaluation.md`, *Results*.
- Done: 08 answers and agentic layer (`eval/answers.json`, `eval/answers_threshold_1.0.json`): 80/80 grounded citations (multi-hop answers cite both chunks), 2/2 correct refusals, 0 false refusals; forced loop recovered 1 question (hw5, by query expansion). See `scripts/notes/08-agent.md`, *Results*.
- Done: 09 answer judging (`eval/answer_eval.json`): calibrated on code-planted errors (both judges 20/20 after prompt v2); 0 answers unfaithful by both judges; 69/82 correct by both; 4 flagged by both, read: 3 real (hw5, g28 from retrieval misses; g58 reasoning error), 1 judge error (g25). See `scripts/notes/09-answer-judge.md`, *Results*.
- Remaining, optional: a sequential multi-hop question that survives labelling; a correctness judge that also sees the cited passages (g25 failure mode).
- Deadline: the user leaves the US in early November, after which OpenAI / Anthropic / Cohere may be unreachable. API-dependent steps (06–09) are done.

## Next steps

1. ~~Run 06.~~ Done.
2. **After 07, update docs with real numbers**: `README.md` (add a Results section), `scripts/notes/rag-project-scope.md`, and the *pending* entries in `scripts/notes/interview-qa.md`.
3. ~~Build 07 evaluation harness~~ Done. (`scripts/07_evaluate.py`), over every config in `retrieval.CONFIGS`:
   - Recall@5: `any` → 1 if any ground-truth chunk is in the top 5; `all` → fraction of ground-truth chunks in the top 5.
   - MRR: reciprocal rank of the first ground-truth chunk.
   - Distractor intrusion: distractor chunks in the top 5, excluding distractors that are part of the ground truth.
   - Unanswerable questions (`none`) reported separately, not averaged into recall.
   - Check that every top-5 chunk was judged (in the question's pool); the expected count of unjudged chunks is 0. Report any.
   - Do not draw conclusions from differences of fewer than about 2 questions.
   - Write a doc `scripts/notes/07-evaluation.md`.
4. **Before 08: add parallel multi-hop questions** to `HANDWRITTEN` in `06_build_golden.py`, written so no single chunk answers them, and rerun 06 (only new questions are judged; the rest come from the cache).
5. **Answer generation** with inline chunk citations, then **08 agentic layer**: tool calling, the rerank top-1 score as the trigger to rewrite the query and retrieve again, with a turn cap.
