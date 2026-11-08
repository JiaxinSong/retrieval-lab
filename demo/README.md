# Demos

Saved terminal output from five runs, so the pipeline can be seen without API keys. Each file was produced with `| tee demo/<file>.txt`; rerunning them needs `OPENAI_API_KEY` and `COHERE_API_KEY` and costs a few cents.

| File | What it demonstrates |
| --- | --- |
| `04_hybrid_search.txt` | Every retrieval stage, side by side, for six queries |
| `ask_work_mem.txt` | A question with a false premise, answered by correcting it |
| `ask_multihop.txt` | A two-part question, each half answered from a different section and cited |
| `ask_unanswerable.txt` | A question the docs can't answer: low score, a second search, then a refusal |
| `ask_agentic.txt` | The agentic loop forced on: the model rewrites the query and answers from the second search |

---

## 1. `04_hybrid_search.txt` — every retrieval stage

```bash
python scripts/04_hybrid_search.py
```

**What it shows.** For each query, the top 5 from each retriever on its own, then after RRF fusion, after the section cap, and after reranking. It is the whole pipeline in `scripts/notes/04-retrieval.md`, made visible.

**How to read a line.**

```
    4 D v#14 l#4                    18.4.3. Resource Limits
    1   v#4 l#5 i#7        0.6241   19.4.1. Memory
```

- First number: position in this list
- `D`: the chunk is from a distractor chapter (never a question source)
- `v#14 l#4 i#7`: where the chunk ranked in the vector, lexical, and identifier lists; a missing tag means that retriever didn't return it
- The decimal in the last block: the Cohere rerank score
- At the top of each query: `lexemes kept / dropped` is the lexical query after document-frequency filtering; `identifiers` is what the identifier retriever searched for

**What to look for, query by query.**

| Query | What it demonstrates |
| --- | --- |
| `work_mem` | The underscore split: the lexical query is just `mem` (`work` was dropped as too common), so only the identifier retriever searches for `work_mem` as a whole |
| `difference between work_mem and maintenance_work_mem` | The two parameters share the lexeme `mem`; lexical search alone can barely tell them apart |
| `Is work_mem a per-connection limit?` | Distractors (`18.4.1`, `18.4.3`) reach the vector and lexical top 5; reranking reorders 4 of 5 positions |
| `why does the database slow down after many updates` | Vector and lexical find completely different chunks (40 in the union, 0 in common) and no identifiers. The top rerank score is only **0.40** and the first result is a distractor: a recall failure, and the low score is the signal |
| `what does autovacuum_vacuum_scale_factor control` | A query retrieval already handles: reranking changes only 1 of 5 positions |
| `preventing transaction id wraparound` | Section domination: all five vector results come from one section, `24.1.5`. The cap makes room for `19.10. Automatic Vacuuming`; the top rerank score is **0.90** |

The two scores, 0.90 on a well-covered question and 0.40 on one with no good answer, are the evidence behind using the rerank score as the agentic trigger.

---

## 2. `ask_work_mem.txt` — correcting a false premise

```bash
python scripts/08_agent.py --ask "Is work_mem a per-connection limit?"
```

**What it shows.** The question assumes something wrong. The answer starts with "No" and explains the real rule: the limit applies per sort or hash operation, and operations, sessions, and parallel workers can each use it.

**What to look for.**

- `top score 0.62` is above the 0.5 gate, so `path: direct`: one model call, no search tool
- Every claim carries a `[n]` citation, and `Sources` maps each number to a section and a URL
- This is the behaviour step 06's judges were calibrated to accept: a passage answers the question by correcting its premise, not by repeating its words (`scripts/notes/06-golden-set.md`, *The first calibration failed*)

---

## 3. `ask_multihop.txt` — two halves, two sections

```bash
python scripts/08_agent.py --ask "How much RAM should I give shared_buffers, and what does effective_cache_size actually do?"
```

**What it shows.** One of the parallel multi-hop questions from the golden set (hw9). No single chunk answers both parts: the sizing guidance for `shared_buffers` is in `19.4.1. Memory`, and what `effective_cache_size` does is in `19.7.2. Planner Cost Constants`.

**What to look for.**

- `[1]` supports the first half and `[2]` the second; each source is a different section
- The first retrieval already found both (`top score 0.59`, `path: direct`), because both setting names are in the question. This is why the parallel multi-hop questions never needed the agentic loop (`scripts/notes/08-agent.md`)

---

## 4. `ask_unanswerable.txt` — refusing instead of inventing

```bash
python scripts/08_agent.py --ask "How do I shard a table across nodes with Citus?"
```

**What it shows.** Citus is a PostgreSQL extension, but the PostgreSQL manual doesn't document it. Retrieval returns plausible chunks about partitioning, the kind of near miss that invites a confident made-up answer.

**What to look for.**

- `top score 0.12`, below 0.5, so the model gets the search tool: `path: agentic`
- It searches again with a rewritten query; the score only reaches 0.20
- It answers `answerable: False` and says the documentation doesn't cover it

**A known flaw this demo exposes.** The refusal cites all nine passages it read (`[1]` to `[9]`), although none supports an answer. A refusal should carry no citations. The fix is a rule in the answer tool, "no citations when `answerable` is false"; it isn't in place yet, because changing it means rerunning step 08.

---

## 5. `ask_agentic.txt` — the agentic loop

```bash
python scripts/08_agent.py --ask "EXPLAIN ANALYZE shows row estimates far from the actual rows -- what setting helps the planner estimate better?" --threshold 1.0
```

**What it shows.** `--threshold 1.0` gives the model the search tool on every question, whatever the score. This is hw5, the one question the forced loop recovered in step 08.

**What to look for.**

- The first search scored 0.76; at the default gate this would have gone down the direct path
- The model chose to search again, and its rewritten query adds the vocabulary of the docs: `default_statistics_target`, `SET STATISTICS`
- The answer cites `[6][7][9]`, numbers beyond the first search's five, so they come from the second search. Passage numbers continue across searches, so each number still identifies one passage
- Interpreting it: in step 08's golden-set run, the first search's passages were checked and none mentioned `default_statistics_target`, so the model brought the term from its own knowledge. That is query expansion, not a second hop that depends on the first search's evidence (`scripts/notes/08-agent.md`, *The one it recovered*)

---

## Which demo for which interview question

| Question | Demo |
| --- | --- |
| "Why three retrievers?" / "Why an identifier retriever?" | 1, the `work_mem` query |
| "What does reranking add?" | 1, the wraparound and slow-down queries (0.90 vs 0.40) |
| "How do you handle hallucination?" | 4 |
| "Can it answer multi-part questions?" | 3 |
| "What does the agent actually do?" | 5 |
| "Show me it working" | 2, then 3 |
