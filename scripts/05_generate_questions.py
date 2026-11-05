"""
Generate golden-dataset candidates by reverse-engineering questions from
chunks, paraphrasing them away from the source wording, then filtering out the
ones that cannot measure anything.

Generation is easy: sample eval chunks, ask an LLM what question each one
answers, and the chunk becomes a ground-truth candidate. Everything after that
is what makes the dataset worth having.

THE PARAPHRASE PASS

A first version skipped it, and 82% of the resulting questions retrieved their
source chunk at rank 1. That sounds like a good retrieval system and is
actually a broken evaluation set: with Recall@5 near 1.0 for every
configuration, the comparison between them measures nothing.

The cause is structural. Generating a question while looking at a passage
produces a question whose *concepts* mirror that passage one-to-one, even when
the wording is forbidden from overlapping. Real users do not phrase things the
way the document that answers them does -- they use their own vocabulary, and
often describe a symptom rather than naming a mechanism.

So a second pass rewrites each question in an engineer's voice, without the
source passage in context. It cannot copy what it cannot see.

THE CHECKS

Three ways a question can look fine and measure nothing. Only the first --
leakage -- is used to exclude questions. The other two consult the retrieval
system under evaluation, and letting that system choose which questions it is
tested on biases the test. Their verdicts are recorded here as diagnostics,
not acted on:

  leakage     The question reuses a distinctive phrase from its source chunk.
              Retrieval then matches on that phrase rather than on meaning,
              and every configuration scores well regardless of quality. This
              is the most common failure in LLM-generated evaluation sets,
              because the model is looking at the passage while it writes.

  distractor  A distractor chunk outranks every eval chunk. The question may
              be drawing on a chapter the question set does not cover. Found
              the hard way: a question about full-text search index types
              returned five distractor chunks, and chapter 12 genuinely has a
              section titled "Preferred Index Types for Text Search".
              Retrieval was right. The golden-set judges label distractor
              chunks too, so a distractor that answers becomes ground truth
              instead of a reason to drop the question.

  unfound     The source chunk does not appear in the top-10 at all. Either
              the question is too general to have one answer, or the chunk is
              not really what answers it.

Output is a candidate file, not a finished dataset: 06_build_golden.py labels
the ground truth with LLM judges. The rank fields (source_rank,
first_eval_rank, first_distractor_rank) are diagnostics only and are not used
for selection -- stratifying by the evaluated system's own ranks would fix its
score in advance.
"""

import asyncio
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import AsyncOpenAI  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from retrieval import connect, hybrid_search  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "eval" / "candidates.json"

MODEL = "gpt-4o-mini"
N_SAMPLE = 80          # chunks to generate questions from
TOP_K_CHECK = 10       # retrieval depth used by the filters
LEAK_NGRAM = 5         # word-window size for the leakage check
CONCURRENCY = 8

client = AsyncOpenAI()


# ---------------------------------------------------------------- prompts

GENERATE_SYSTEM = """You write evaluation questions for a documentation search system.

Given a passage from the PostgreSQL manual, write ONE question that the passage
answers.

Rules:
- Write it the way an engineer would type it into a search box, not the way a
  textbook would phrase it.
- Do NOT reuse distinctive phrases from the passage. If the passage says
  "applies per operation, not per query", do not write "does it apply per
  operation" -- write "is this limit per connection?". Reusing the wording
  makes the question trivially retrievable and measures nothing.
- The question must have a specific answer, not a general topic. Bad: "What is
  autovacuum?". Good: "What fraction of dead rows triggers autovacuum?".
- If the passage is too thin to support a specific question -- a stub, a
  cross-reference, a list of links -- return an empty string.
"""

# Deliberately does NOT receive the source passage. A model that cannot see
# the document cannot echo it, which is the entire point of this pass.
PARAPHRASE_SYSTEM = """Rewrite a search query the way a working engineer would actually type it.

You will be given a question about PostgreSQL. Rewrite it so that:
- The vocabulary is the engineer's, not the manual's. Prefer the words someone
  would use before they have found the documentation.
- Where natural, describe the symptom rather than naming the mechanism.
  "queries got slow after a big delete" rather than "what is the effect of
  dead tuples on scan performance".
- It stays answerable by the same passage. Do not broaden the topic or change
  what is being asked.
- It stays short. Real search queries are under fifteen words.

Return only the rewritten question.
"""


class GeneratedQuestion(BaseModel):
    question: str = Field(description="The question, or empty string if the passage is unusable")
    reasoning: str = Field(description="One sentence: what in the passage answers it")


class Paraphrase(BaseModel):
    question: str = Field(description="The rewritten question")


# ---------------------------------------------------------------- leakage

def normalize_words(text: str) -> list[str]:
    """Lowercase word tokens, for the leakage n-gram comparison."""
    return re.findall(r"[a-z0-9_]+", text.lower())


def leaked_phrase(question: str, chunk: str, n: int = LEAK_NGRAM) -> str | None:
    """
    Return the first n-gram the question shares with its source chunk.

    An overlapping phrase of five or more words is almost never coincidence in
    a question this short -- it means the model copied from the passage it was
    looking at. Retrieval would then match on the copied phrase, and every
    configuration would score well regardless of how good it is.

    Single shared words are expected and fine: a question about work_mem has to
    say work_mem.
    """
    q_words = normalize_words(question)
    c_words = normalize_words(chunk)

    if len(q_words) < n:
        return None

    chunk_ngrams = {" ".join(c_words[i : i + n]) for i in range(len(c_words) - n + 1)}

    for i in range(len(q_words) - n + 1):
        gram = " ".join(q_words[i : i + n])
        if gram in chunk_ngrams:
            return gram
    return None


# ---------------------------------------------------------------- generation

async def generate_one(chunk: dict) -> dict | None:
    """Ask the model for one question about one chunk."""
    try:
        resp = await client.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": GENERATE_SYSTEM},
                {"role": "user", "content": chunk["content"]},
            ],
            response_format=GeneratedQuestion,
            # Some variety: identical phrasing across 80 questions would make
            # the set narrower than its size suggests.
            temperature=0.7,
        )
    except Exception as e:
        print(f"  generation failed for chunk {chunk['id']}: {type(e).__name__}")
        return None

    parsed = resp.choices[0].message.parsed
    if not parsed.question.strip():
        return None

    return {
        "question": parsed.question.strip(),
        "reasoning": parsed.reasoning.strip(),
        "source_chunk_id": chunk["id"],
        "source_heading": chunk["metadata"].get("heading", ""),
        "source_section_id": chunk["metadata"].get("section_id", ""),
        "source_content": chunk["content"],
    }


async def paraphrase_one(cand: dict) -> dict:
    """
    Rewrite the question without the source passage in context.

    Keeps the original under `question_original` so the effect of this pass is
    measurable -- comparing source_rank before and after is what tells you
    whether it did anything.
    """
    try:
        resp = await client.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": PARAPHRASE_SYSTEM},
                {"role": "user", "content": cand["question"]},
            ],
            response_format=Paraphrase,
            temperature=0.8,
        )
        rewritten = resp.choices[0].message.parsed.question.strip()
    except Exception as e:
        print(f"  paraphrase failed for chunk {cand['source_chunk_id']}: {type(e).__name__}")
        rewritten = ""

    return {
        **cand,
        "question_original": cand["question"],
        # Fall back to the original rather than dropping the candidate: a
        # failed rewrite is a less useful question, not an unusable one.
        "question": rewritten or cand["question"],
    }


# ---------------------------------------------------------------- filtering

async def evaluate_candidate(pool, cand: dict) -> dict:
    """
    Run retrieval and record what the filters need to decide.

    Ground truth is stored three ways -- chunk id, section id, and the content
    itself. The id is what the eval harness uses; the other two exist because
    chunk ids are only stable while the chunking configuration is. Comparing
    chunk sizes (a Tier 2 experiment) renumbers everything, and an annotation
    set keyed only on ids would silently become meaningless rather than
    visibly break.
    """
    hits = await hybrid_search(pool, cand["question"])
    top = hits[:TOP_K_CHECK]

    ids = [h.id for h in top]
    source_rank = (
        ids.index(cand["source_chunk_id"]) + 1 if cand["source_chunk_id"] in ids else None
    )

    # Rank of the best eval chunk and the best distractor, for the second filter.
    first_eval = next((i for i, h in enumerate(top, 1) if h.group == "eval"), None)
    first_distractor = next((i for i, h in enumerate(top, 1) if h.group == "distractor"), None)

    leak = leaked_phrase(cand["question"], cand["source_content"])

    if leak:
        verdict = "leakage"
    elif first_distractor is not None and (first_eval is None or first_distractor < first_eval):
        verdict = "distractor_wins"
    elif source_rank is None:
        verdict = "source_not_found"
    else:
        verdict = "ok"

    return {
        **cand,
        "verdict": verdict,
        "leaked_phrase": leak,
        "source_rank": source_rank,
        "first_eval_rank": first_eval,
        "first_distractor_rank": first_distractor,
        "retrieved": [
            {
                "id": h.id,
                "group": h.group,
                "heading": h.heading,
                "section_id": h.section_id,
                "preview": h.content[:200],
            }
            for h in top
        ],
    }


# ---------------------------------------------------------------- report

def report(results: list[dict]) -> None:
    counts: dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    print(f"\n{len(results)} candidates -> {OUT_PATH}\n")
    for verdict in ("ok", "leakage", "distractor_wins", "source_not_found"):
        print(f"  {verdict:18} {counts.get(verdict, 0)}")

    ok = [r for r in results if r["verdict"] == "ok"]
    if not ok:
        return

    ranks = [r["source_rank"] for r in ok]
    dist = {r: sum(1 for x in ranks if x == r) for r in range(1, 6)}
    beyond = sum(1 for x in ranks if x > 5)

    print(f"\nsource-chunk rank among the {len(ok)} usable candidates:")
    for rank, n in dist.items():
        bar = "#" * n
        print(f"  rank {rank}:  {n:3}  {bar}")
    print(f"  rank 6+: {beyond:3}")
    print(f"  mean:    {sum(ranks) / len(ranks):.2f}")

    at_one = dist[1] / len(ok)
    print(f"\n{at_one:.0%} rank 1st.")
    if at_one > 0.7:
        # A set the system already answers cannot distinguish configurations:
        # every one of them scores near the ceiling.
        print("  Above ~70% the set is too easy to separate configurations.")
        print("  Hand-written hard cases (multi-hop, unanswerable, same-section)")
        print("  matter more than raw count here.")

    # A sample of what was thrown away, so the filters can be checked rather
    # than trusted.
    for verdict in ("leakage", "distractor_wins", "source_not_found"):
        examples = [r for r in results if r["verdict"] == verdict][:2]
        if not examples:
            continue
        print(f"\n--- flagged: {verdict} ---")
        for r in examples:
            print(f"  Q: {r['question']}")
            if r.get("question_original") and r["question_original"] != r["question"]:
                print(f"     was: {r['question_original']}")
            if r["leaked_phrase"]:
                print(f"     leaked: {r['leaked_phrase']!r}")
            if verdict == "distractor_wins":
                print(f"     distractor at {r['first_distractor_rank']}, "
                      f"eval at {r['first_eval_rank']}")


def report_paraphrase_effect(results: list[dict]) -> None:
    """
    Did the paraphrase pass actually make the set harder?

    If ranks did not move, the rewrite is cosmetic and the difficulty problem
    is unsolved -- worth knowing before paying for the judging run.
    """
    moved = [r for r in results
             if r["verdict"] == "ok" and r.get("question_original") != r["question"]]
    if not moved:
        return

    print(f"\nparaphrase rewrote {len(moved)} of the usable candidates")
    print("  (compare the rank distribution above against a run with")
    print("   PARAPHRASE = False to see the effect)")


# ---------------------------------------------------------------- main

PARAPHRASE = True


async def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pool = await connect()

    try:
        rows = await pool.fetch(
            """
            SELECT id, content, metadata
            FROM retrieval_lab_chunks
            WHERE metadata->>'group' = 'eval'
            ORDER BY id
            """
        )
        chunks = [
            {"id": r["id"], "content": r["content"], "metadata": json.loads(r["metadata"])}
            for r in rows
        ]

        # Seeded so the sample is reproducible. Re-running should re-examine
        # the same chunks rather than a fresh draw, or two runs are not
        # comparable. The seed only works because ORDER BY id fixes the input
        # order: random.sample picks positions, not rows.
        random.seed(42)
        sample = random.sample(chunks, min(N_SAMPLE, len(chunks)))
        print(f"{len(chunks)} eval chunks, sampling {len(sample)}\n")

        # Bounded concurrency: 80 unthrottled requests will hit a rate limit.
        sem = asyncio.Semaphore(CONCURRENCY)

        async def bounded(fn, arg):
            async with sem:
                return await fn(arg)

        print("generating questions...")
        generated = [
            c for c in await asyncio.gather(*[bounded(generate_one, c) for c in sample]) if c
        ]
        print(f"  {len(generated)} generated "
              f"({len(sample) - len(generated)} passages rejected as too thin)\n")

        if PARAPHRASE:
            print("paraphrasing away from source wording...")
            generated = await asyncio.gather(
                *[bounded(paraphrase_one, c) for c in generated]
            )
            changed = sum(1 for c in generated if c["question"] != c["question_original"])
            print(f"  {changed}/{len(generated)} rewritten\n")

        print("running retrieval for filters...")
        results = []
        for i, cand in enumerate(generated, 1):
            results.append(await evaluate_candidate(pool, cand))
            if i % 10 == 0:
                print(f"  {i}/{len(generated)}")

        OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

        report(results)
        if PARAPHRASE:
            report_paraphrase_effect(results)

    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())