"""
Build the golden dataset with LLM judges -- no manual annotation.

Pipeline:

    candidates.json + hand-written questions
      -> exclude leakage and near-duplicates of development queries
      -> pool candidate chunks per question
      -> calibrate the judges on hand-verified cases; stop if they fail
      -> Judge A and Judge B label every (question, chunk) pair independently
      -> reconcile per chunk; a third judge settles direct contradictions
      -> apply question-level rules; discard what cannot be settled
      -> golden.json, discarded.json, judge_report.json

WHAT THE LABELS MEAN

Each judge sees one question and one passage, never several at once, and
answers: could a reader given only this passage answer completely?

    yes      the passage alone answers the question completely
    partial  the passage holds part of the answer
    no       it does not help, even if it is on a related topic

One passage per call is deliberate. Shown several passages together, a judge
blends them, and "answers on its own" stops being what is measured.

WHY TWO JUDGES FROM DIFFERENT VENDORS

Two runs of one model disagree only through sampling noise. Two vendors
disagree when their standards differ, which is the disagreement worth
catching. Neither vendor's judge runs at temperature 0: the OpenAI judge is
a reasoning model, which takes a reasoning effort instead of a temperature,
and the current Anthropic SDK no longer accepts a temperature parameter. That costs less than it sounds: every label is cached
the moment it is made, so a rerun reuses it rather than sampling again, and
the reported inter-judge agreement measures whatever variance remains.

RECONCILIATION

Labels are ordered yes > partial > no, and the weaker of the two is kept:
yes/partial becomes partial, partial/no becomes no. Only yes against no -- a
direct contradiction, not a difference of strictness -- goes to Judge C, a
stronger model, whose label is final.

Keeping the weaker label makes ground truth conservative: an answer the judges
disagree about being complete does not count as complete. That lowers every
configuration's recall a little and in the same direction, which leaves the
comparison between configurations -- the purpose of this dataset -- intact.

QUESTION-LEVEL RULES

    unanswerable type, no yes        accept, match "none"
    unanswerable type, any yes       discard: the question has an answer
    at least one yes                 accept, ground truth = every yes, "any"
    no yes, no partial               discard: nothing answers it
    no yes, exactly one partial      discard: only half an answer exists
    no yes, two or three partials    combination check (below)
    no yes, four or more partials    discard: too complex to label reliably

Combination check: both judges see the partial passages together and report
whether they answer completely and which are necessary. Accepted as "all" only
if both say complete and name the same necessary set of two or more.

Overlap can make interchangeable passages -- the same sentence in two adjacent
chunks -- so "A or A', plus B" is possible. The judges will usually disagree on
which of A and A' is necessary, and the question is discarded. That trades a
few multi-hop questions for a simple schema; the report counts how many
multi-hop questions survive so the cost is visible.

WHY THE POOL IS WHAT IT IS

Candidates are the union of the three retrievers' top RECALL_K, plus the source
chunk and its same-section neighbours regardless of retrieval, plus keyword
matches for hand-written questions. Every configuration under evaluation draws
from the retrievers' top RECALL_K, so each configuration's top-5 is a subset of
the pool by construction: nothing a configuration returns goes unjudged. No
reranker calls are needed.

Distractor chunks in the pool are judged like any other. One judged "yes" is a
correct answer -- the eval/distractor split existed to bound manual annotation,
and that constraint is gone.

WHAT THIS IS NOT

A human-verified benchmark. It is an LLM-judged evaluation set, suitable for
comparing retrieval configurations against each other. One bias is worth
stating: the judges' sense of "answers the question" may resemble a
cross-encoder's sense of relevance, which could flatter the rerank
configurations. Cross-vendor judging reduces that; it does not remove it.

OPERATION

Every judgment is cached in eval/judgments.jsonl, keyed by prompt version,
model, question id, chunk id, and a hash of the exact question and chunk text. Interrupted runs resume without paying twice, and
a run after a retrieval change judges only chunks that are new to the pool.

    python3 scripts/06_build_golden.py                 # full run
    python3 scripts/06_build_golden.py --dry-run       # count calls, spend nothing
    python3 scripts/06_build_golden.py --reset         # rebuild pools, keep cache
    python3 scripts/06_build_golden.py --clear-cache   # discard all judgments
"""

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Literal

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval import (  # noqa: E402
    RECALL_K,
    TABLE,
    connect,
    identifier_search,
    lexical_search,
    vector_search,
)

EVAL_DIR = ROOT / "eval"
CANDIDATES_PATH = EVAL_DIR / "candidates.json"
POOLS_PATH = EVAL_DIR / "pools.json"
CACHE_PATH = EVAL_DIR / "judgments.jsonl"
GOLDEN_PATH = EVAL_DIR / "golden.json"
DISCARDED_PATH = EVAL_DIR / "discarded.json"
REPORT_PATH = EVAL_DIR / "judge_report.json"

# ---------------------------------------------------------------- judges

# gpt-4o-mini was the first choice. Calibration showed it misreading passages:
# it labelled the work_mem definition "yes" while answering that work_mem IS
# a per-connection limit (the opposite of the text), and labelled the
# continuation chunk "no". A stronger reader replaced it.
JUDGE_A = ("openai", "gpt-5-mini")
JUDGE_B = ("anthropic", "claude-haiku-4-5-20251001")
# Referee, contradictions only. claude-sonnet-4-5 was the first choice but
# reaches end-of-life on 2026-11-30, which would break any later rerun.
JUDGE_C = ("anthropic", "claude-sonnet-5")

# Bump when a prompt changes. It is part of every cache key, so judgments made
# under an old prompt are never reused under a new one.
PROMPT_VERSION = "v2"

# Client-side pacing, kept just under typical tier-1 limits. Raise the
# Anthropic figure if your account tier allows more
# (console.anthropic.com/settings/limits).
OPENAI_RPM = 400
OPENAI_CONCURRENCY = 16
ANTHROPIC_RPM = 45
ANTHROPIC_CONCURRENCY = 4
SDK_MAX_RETRIES = 8        # the SDKs back off exponentially and honour retry-after

MAX_PARTIALS_FOR_COMBO = 3
SPOT_CHECK_N = 10
SEED = 7

# ---------------------------------------------------------------- selection inputs

SEED_TERM_LIMIT = 8
DEMO_OVERLAP = 0.5

# Every query run while developing retrieval. Decisions were made by reading
# their results, so none of them -- or anything close -- may be in the test set.
DEMO_QUERIES = [
    "work_mem",
    "difference between work_mem and maintenance_work_mem",
    "Is work_mem a per-connection limit?",
    "why does the database slow down after many updates",
    "what does autovacuum_vacuum_scale_factor control",
    "preventing transaction id wraparound",
    "how do I see which queries are blocked by pg_locks",
    "Which index type should I use for full-text search?",
    "What does shared_buffers control?",
    "How do I find which queries are waiting on locks?",
    "Why would autovacuum fall behind on a large table?",
]

# Hand-written hard cases. None touches a section inspected during development.
# `type` is intended shape, not a verdict: the judges decide the match rule.
HANDWRITTEN = [
    {"qid": "hw1", "type": "same_section",
     "question": "In pg_stat_activity, what does a state of 'idle in transaction' mean?",
     "seed_terms": ["idle in transaction"]},
    {"qid": "hw2", "type": "same_section",
     "question": "In an EXPLAIN plan, what do the two numbers in cost=0.00..458.00 mean?",
     "seed_terms": ["start-up cost", "cost=0.00"]},
    {"qid": "hw3", "type": "multi_hop_parallel",
     "question": "What's the difference between log_min_duration_statement and log_statement?",
     "seed_terms": ["log_min_duration_statement", "log_statement ("]},
    {"qid": "hw4", "type": "multi_hop_parallel",
     "question": "When would I use a partial index versus an index on an expression?",
     "seed_terms": ["partial index", "Indexes on Expressions"]},
    {"qid": "hw5", "type": "multi_hop_sequential",
     "question": "EXPLAIN ANALYZE shows row estimates far from the actual rows -- what setting helps the planner estimate better?",
     "seed_terms": ["actual rows", "default_statistics_target"]},
    {"qid": "hw6", "type": "multi_hop_sequential",
     "question": "Index-only scans aren't happening even with a covering index -- what maintenance affects whether they can be used?",
     "seed_terms": ["visibility map", "index-only scan"]},
    {"qid": "hw7", "type": "unanswerable",
     "question": "How do I shard tables across multiple servers with Citus?",
     "seed_terms": ["Citus", "shard"]},
    {"qid": "hw8", "type": "unanswerable",
     "question": "How do I enable query result caching in pgpool-II?",
     "seed_terms": ["pgpool", "query cache"]},

    # Parallel multi-hop, added after the first run left no `all` question.
    # Each was checked against the chunk text before being written: the two
    # halves of the answer sit in different chunks, and no chunk names both
    # settings. Where a definition crosses a chunk boundary (work_mem,
    # log_checkpoints, checkpoint_warning), the question asks for the default,
    # which appears in only one of the two overlapping chunks -- so exactly
    # one set of chunks is necessary and the judges' combination check has a
    # single right answer. Expected chunks are noted for later checking only;
    # the judges decide.
    {"qid": "hw9", "type": "multi_hop_parallel",   # expect 1028 + 1008
     "question": "How much RAM should I give shared_buffers, and what does effective_cache_size actually do?",
     "seed_terms": ["shared_buffers ( integer )", "effective_cache_size ( integer )"]},
    {"qid": "hw10", "type": "multi_hop_parallel",  # expect 1055 + 1059
     "question": "What do wal_buffers and max_wal_size each control, and what are their defaults?",
     "seed_terms": ["wal_buffers ( integer )", "max_wal_size ( integer )"]},
    {"qid": "hw11", "type": "multi_hop_parallel",  # expect 1006 + 1042
     "question": "What do random_page_cost and effective_io_concurrency each control, and what are their defaults?",
     "seed_terms": ["random_page_cost ( floating point )", "effective_io_concurrency ( integer )"]},
    {"qid": "hw12", "type": "multi_hop_parallel",  # expect 1030 + 1036
     "question": "Sorts keep spilling to temp files: what controls how much memory a sort gets before it spills, "
                 "what is its default, and how do I cap the temp file space a process can use?",
     "seed_terms": ["work_mem ( integer )", "temp_file_limit ( integer )"]},
    {"qid": "hw13", "type": "multi_hop_parallel",  # expect 989 + 1059
     "question": "Is log_checkpoints on by default, and by default how close together must checkpoints be "
                 "before checkpoint_warning logs a warning?",
     "seed_terms": ["log_checkpoints ( boolean )", "checkpoint_warning ( integer )"]},
]

# Calibration: cases whose answers were verified by reading the source. Never
# part of the evaluation set.
#
# work_mem: the definition chunk states the limit is per operation and that
# concurrent sessions multiply it -- a complete answer. The next chunk repeats
# that sentence through the overlap, but opens mid-sentence with "this value",
# so a strict judge may reasonably call it partial; it only has to be not "no".
# The autovacuum_work_mem chunk answers a different parameter and must not be
# "yes".
CAL_WORK_MEM_QUESTION = "Is work_mem a per-connection limit?"
CAL_DEFINITION_MARK = "Sets the base maximum amount of memory to be used by a query operation"
CAL_CONTINUATION_MARK = "operation generally being allowed"
CAL_OTHER_PARAM_MARK = "autovacuum_work_mem ( integer )"
# Near-domain and absent from the manual: the pool will contain plausible
# chunks (partitioning, TOAST compression), and none may be judged "yes".
CAL_UNANSWERABLE_QUESTION = "How do I enable TimescaleDB compression on a hypertable?"

STOPWORDS = {
    "the", "and", "for", "what", "does", "how", "why", "which", "when", "with",
    "can", "are", "is", "a", "an", "of", "in", "on", "to", "do", "i", "my",
    "it", "this", "that", "be", "or", "should", "after", "many",
}

# ---------------------------------------------------------------- prompts

CHUNK_SYSTEM = """You judge whether a passage from the PostgreSQL manual answers a question.

Judge the passage alone. Do not use outside knowledge, and do not assume other passages exist.

First write the answer a reader would give using only this passage, in one or two sentences. If the passage gives no answer, write "no answer". Then choose the label for that answer.

A passage answers the question when it lets the reader give the correct answer, whether or not it uses the question's words. That includes questions whose premise is wrong: if a question asks "is this limit per user?" and the passage says the limit applies per request, the passage answers it ("no, it applies per request").

Labels:
- "yes": a reader given only this passage could answer the question correctly and completely.
- "partial": the passage contains information the answer needs, but not all of it.
- "no": the passage does not help answer the question, even if it is on a related topic.

Be strict about "yes". Being on the right topic, mentioning the same setting, or answering a neighbouring question is not enough. If a complete answer needs information this passage does not contain, the label is "partial" or "no".

Passages were cut from longer pages and may begin or end mid-sentence. The first line names the section the passage came from. That is fine as long as the needed information is present.

Give a one-sentence reason."""

COMBO_SYSTEM = """You judge whether several passages from the PostgreSQL manual, taken together, answer a question.

Use only the passages. Do not use outside knowledge.

Report:
- complete: true if a reader given only these passages could answer the question correctly and completely.
- necessary: the labels (such as "P1") of the passages required for a complete answer. Leave out a passage that only repeats information another passage already provides, or adds detail the answer does not need.

Give a one-sentence reason."""


class ChunkJudgment(BaseModel):
    # Field order matters: the judge writes the answer before choosing a
    # label, so the label is grounded in what the passage actually says.
    answer: str
    label: Literal["yes", "partial", "no"]
    reason: str


class ComboJudgment(BaseModel):
    complete: bool
    necessary: list[str]
    reason: str


RANK = {"no": 0, "partial": 1, "yes": 2}

# ---------------------------------------------------------------- infrastructure

openai_client = AsyncOpenAI(max_retries=SDK_MAX_RETRIES)
anthropic_client = AsyncAnthropic(max_retries=SDK_MAX_RETRIES)

# Errors that retrying cannot fix. On seeing one, stop issuing calls rather
# than burning through thousands of doomed requests.
FATAL_MARKERS = ("insufficient_quota", "credit balance", "credit_balance",
                 "billing", "invalid x-api-key", "incorrect api key",
                 "could not resolve authentication", "authentication_error",
                 "api_key client option must be set",
                 "not_found_error", "model_not_found")
ABORT = asyncio.Event()
ERRORS: Counter = Counter()


class RateLimiter:
    """Space request starts evenly so the provider's per-minute limit is never
    hit in bursts. Holding the lock while sleeping is intentional: it
    serializes starts, not the requests themselves."""

    def __init__(self, rpm: int) -> None:
        self.interval = 60.0 / rpm
        self.lock = asyncio.Lock()
        self.next_start = 0.0

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            if self.next_start > now:
                await asyncio.sleep(self.next_start - now)
            self.next_start = max(now, self.next_start) + self.interval


LIMITS: dict[str, tuple[RateLimiter, asyncio.Semaphore]] = {}


def init_limits() -> None:
    LIMITS["openai"] = (RateLimiter(OPENAI_RPM), asyncio.Semaphore(OPENAI_CONCURRENCY))
    LIMITS["anthropic"] = (RateLimiter(ANTHROPIC_RPM), asyncio.Semaphore(ANTHROPIC_CONCURRENCY))


class Cache:
    """Append-only JSONL store of judgments."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: dict[str, dict] = {}
        self.lock = asyncio.Lock()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.records[rec["key"]] = rec

    def get(self, key: str) -> dict | None:
        return self.records.get(key)

    async def put(self, key: str, record: dict) -> None:
        record = {"key": key, **record}
        async with self.lock:
            self.records[key] = record
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


CACHE: Cache


async def call_judge(judge: tuple[str, str], system: str, user: str,
                     schema: type[BaseModel]) -> BaseModel | None:
    """One structured judgment. Returns None on failure; the caller leaves the
    item uncached so the next run retries it."""
    if ABORT.is_set():
        return None
    vendor, model = judge
    limiter, sem = LIMITS[vendor]
    async with sem:
        await limiter.wait()
        if ABORT.is_set():
            return None
        try:
            if vendor == "openai":
                resp = await openai_client.chat.completions.parse(
                    model=model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    response_format=schema,
                    **openai_sampling(model),
                )
                parsed = resp.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("no parsed output (refusal?)")
                return parsed

            # Anthropic: a forced tool call is the reliable way to get
            # schema-shaped output.
            # No temperature argument: the current SDK rejects it with a
            # TypeError. Determinism comes from the cache instead.
            resp = await anthropic_client.messages.create(
                model=model,
                max_tokens=400,
                system=system,
                tools=[{"name": "record", "description": "Record the judgment.",
                        "input_schema": schema.model_json_schema()}],
                tool_choice={"type": "tool", "name": "record"},
                messages=[{"role": "user", "content": user}],
            )
            block = next(b for b in resp.content if b.type == "tool_use")
            return schema.model_validate(block.input)

        except Exception as e:  # noqa: BLE001 -- every failure is recorded, none is fatal to the run
            text = f"{type(e).__name__}: {e}".lower()
            if any(m in text for m in FATAL_MARKERS):
                if not ABORT.is_set():
                    print(f"\nFATAL ({model}): {e}\nStopping; nothing further will be sent.")
                ABORT.set()
            ERRORS[model] += 1
            if ERRORS[model] <= 3:
                print(f"  {model} call failed: {type(e).__name__}: {str(e)[:160]}")
            return None


def openai_sampling(model: str) -> dict:
    """Sampling arguments the model accepts.

    Reasoning models (the gpt-5 and o-series) reject temperature; they take a
    reasoning effort instead. "low" is enough to read one passage carefully
    without paying for long hidden reasoning on thousands of calls.
    """
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        return {"reasoning_effort": "low"}
    return {"temperature": 0}


def fingerprint(*texts: str) -> str:
    """Short hash of the exact text a judge saw.

    qid and chunk id are positions, not content: g00 is whatever candidate
    happens to come first, and a chunk id points at different text after
    re-chunking. Keying on them alone would silently reuse a judgment made for
    a different question or passage. The fingerprint changes whenever the text
    does, so a changed question or chunk is judged again.
    """
    digest = hashlib.sha256("\x1f".join(texts).encode("utf-8")).hexdigest()
    return digest[:16]


def chunk_key(judge, qid: str, cid: int, question: str, content: str) -> str:
    return f"{PROMPT_VERSION}|{judge[1]}|{qid}|{cid}|{fingerprint(question, content)}"


def combo_key(judge, qid: str, cids: list[int], question: str, contents: list[str]) -> str:
    ids = ",".join(map(str, cids))
    return f"{PROMPT_VERSION}|{judge[1]}|{qid}|combo|{ids}|{fingerprint(question, *contents)}"


async def judge_chunk(judge, qid: str, question: str, cid: int, content: str) -> str | None:
    key = chunk_key(judge, qid, cid, question, content)
    hit = CACHE.get(key)
    if hit:
        return hit["label"]
    result = await call_judge(judge, CHUNK_SYSTEM,
                              f"Question: {question}\n\nPassage:\n{content}", ChunkJudgment)
    if result is None:
        return None
    await CACHE.put(key, {"qid": qid, "chunk_id": cid, "model": judge[1],
                          "answer": result.answer, "label": result.label,
                          "reason": result.reason})
    return result.label


async def judge_combo(judge, qid: str, question: str, cids: list[int],
                      contents: list[str]) -> tuple[bool, list[int]] | None:
    key = combo_key(judge, qid, cids, question, contents)
    hit = CACHE.get(key)
    if hit:
        return hit["complete"], hit["necessary_ids"]
    labels = [f"P{i}" for i in range(1, len(cids) + 1)]
    body = "\n\n".join(f"[{lab}]\n{text}" for lab, text in zip(labels, contents))
    result = await call_judge(judge, COMBO_SYSTEM,
                              f"Question: {question}\n\n{body}", ComboJudgment)
    if result is None:
        return None
    by_label = dict(zip(labels, cids))
    necessary = sorted({by_label[n] for n in result.necessary if n in by_label})
    await CACHE.put(key, {"qid": qid, "chunk_ids": cids, "model": judge[1],
                          "complete": result.complete, "necessary_ids": necessary,
                          "reason": result.reason})
    return result.complete, necessary


class Progress:
    def __init__(self, total: int, label: str) -> None:
        self.total, self.label, self.done = total, label, 0
        self.start = time.monotonic()
        self.step = max(25, total // 20)   # ~20 lines on the main run, 1-2 on small ones

    def tick(self) -> None:
        self.done += 1
        if self.done % self.step == 0 or self.done == self.total:
            elapsed = time.monotonic() - self.start
            rate = self.done / elapsed if elapsed else 0
            eta = (self.total - self.done) / rate / 60 if rate else 0
            print(f"  {self.label}: {self.done}/{self.total}  "
                  f"{elapsed / 60:.1f} min elapsed, ~{eta:.0f} min left")


# ---------------------------------------------------------------- judging and reconciling

async def label_items(items: list[tuple[str, str, int, str]], label: str) -> dict:
    """
    Judge A and Judge B on every (qid, question, chunk_id, content), then
    reconcile, sending direct contradictions to Judge C.

    Returns {(qid, cid): {"a", "b", "c", "final"}}; "final" is None when a
    call failed and the item needs a rerun.
    """
    uncached = sum(1 for j in (JUDGE_A, JUDGE_B) for qid, q, cid, text in items
                   if not CACHE.get(chunk_key(j, qid, cid, q, text)))
    progress = Progress(max(uncached, 1), label)

    async def one(judge, qid, question, cid, content):
        cached = CACHE.get(chunk_key(judge, qid, cid, question, content)) is not None
        result = await judge_chunk(judge, qid, question, cid, content)
        if not cached:
            progress.tick()
        return result

    tasks = [one(j, qid, q, cid, text) for qid, q, cid, text in items for j in (JUDGE_A, JUDGE_B)]
    labels = await asyncio.gather(*tasks)

    out: dict = {}
    for n, (qid, q, cid, text) in enumerate(items):
        a, b = labels[2 * n], labels[2 * n + 1]
        out[(qid, cid)] = {"a": a, "b": b, "c": None, "final": None,
                           "question": q, "content": text}

    # Referee: only yes against no.
    conflicts = [k for k, v in out.items()
                 if v["a"] and v["b"] and {v["a"], v["b"]} == {"yes", "no"}]
    if conflicts:
        print(f"  {label}: {len(conflicts)} yes/no contradictions -> referee")
        c_labels = await asyncio.gather(*[
            judge_chunk(JUDGE_C, qid, out[(qid, cid)]["question"], cid, out[(qid, cid)]["content"])
            for qid, cid in conflicts])
        for k, c in zip(conflicts, c_labels):
            out[k]["c"] = c

    for k, v in out.items():
        if v["a"] is None or v["b"] is None:
            continue
        if {v["a"], v["b"]} == {"yes", "no"}:
            v["final"] = v["c"]                     # None if the referee failed
        else:
            v["final"] = min(v["a"], v["b"], key=RANK.get)
    return out


async def decide(q: dict, labels: dict) -> dict:
    """Apply the question-level rules. Returns a status record."""
    finals = {cid: labels[(q["qid"], cid)]["final"] for cid in q["pool"]}
    if any(v is None for v in finals.values()):
        return {"status": "pending", "reason": "judgment missing; rerun to fill"}

    yes = sorted(c for c, v in finals.items() if v == "yes")
    partial = sorted(c for c, v in finals.items() if v == "partial")

    if q["type"] == "unanswerable":
        if yes:
            return {"status": "discard", "reason": "unanswerable question has an answer",
                    "warn": True, "yes": yes}
        return {"status": "accept", "match": "none", "ground_truth": []}

    if yes:
        return {"status": "accept", "match": "any", "ground_truth": yes}
    if not partial:
        return {"status": "discard", "reason": "no chunk answers it"}
    if len(partial) == 1:
        return {"status": "discard", "reason": "only a partial answer exists"}
    if len(partial) > MAX_PARTIALS_FOR_COMBO:
        return {"status": "discard", "reason": f"more than {MAX_PARTIALS_FOR_COMBO} partial pieces"}

    contents = [labels[(q["qid"], c)]["content"] for c in partial]
    ra, rb = await asyncio.gather(
        judge_combo(JUDGE_A, q["qid"], q["question"], partial, contents),
        judge_combo(JUDGE_B, q["qid"], q["question"], partial, contents),
    )
    if ra is None or rb is None:
        return {"status": "pending", "reason": "combination check missing; rerun to fill"}
    (ca, na), (cb, nb) = ra, rb
    if ca and cb and na == nb and len(na) >= 2:
        return {"status": "accept", "match": "all", "ground_truth": na}
    return {"status": "discard", "reason": "judges did not agree on a combination"}


# ---------------------------------------------------------------- questions and pools

def word_set(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in STOPWORDS}


def demo_overlap(question: str) -> str | None:
    q = word_set(question)
    for demo in DEMO_QUERIES:
        d = word_set(demo)
        if q and d and len(q & d) / len(q | d) >= DEMO_OVERLAP:
            return demo
    return None


async def neighbours(pool, chunk_id: int, section_id: str) -> list[int]:
    rows = await pool.fetch(
        f"SELECT id FROM {TABLE} WHERE metadata->>'section_id' = $1 "
        f"AND id BETWEEN $2 - 1 AND $2 + 1",
        section_id, chunk_id)
    return [r["id"] for r in rows]


async def keyword_matches(pool, term: str) -> list[int]:
    rows = await pool.fetch(
        f"SELECT id FROM {TABLE} WHERE content ILIKE '%' || $1 || '%' ORDER BY id LIMIT $2",
        term, SEED_TERM_LIMIT)
    return [r["id"] for r in rows]


async def build_pool(pool, question: str, source_id: int | None,
                     source_section: str | None, seed_terms: list[str]) -> list[int]:
    v, lx, ix = await asyncio.gather(
        vector_search(pool, question, RECALL_K),
        lexical_search(pool, question, RECALL_K),
        identifier_search(pool, question, RECALL_K),
    )
    ids = {h.id for h in (*v, *lx, *ix)}
    if source_id is not None:
        ids.add(source_id)
        if source_section:
            ids.update(await neighbours(pool, source_id, source_section))
    for term in seed_terms:
        ids.update(await keyword_matches(pool, term))
    return sorted(ids)


async def fetch_chunks(pool, ids) -> dict[str, dict]:
    rows = await pool.fetch(
        f"SELECT id, content, metadata FROM {TABLE} WHERE id = ANY($1::int[])", list(ids))
    return {str(r["id"]): {"content": r["content"], **json.loads(r["metadata"])} for r in rows}


async def build_pools(pool) -> dict:
    candidates = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    questions: dict[str, dict] = {}
    excluded = []

    for n, c in enumerate(candidates):
        if c.get("verdict") == "leakage":
            excluded.append((c["question"], "leakage"))
            continue
        demo = demo_overlap(c["question"])
        if demo:
            excluded.append((c["question"], f"near-duplicate of dev query {demo!r}"))
            continue
        chapter = await pool.fetchval(
            f"SELECT metadata->>'chapter' FROM {TABLE} WHERE id = $1", c["source_chunk_id"])
        questions[f"g{n:02d}"] = {
            "qid": f"g{n:02d}", "type": "generated", "question": c["question"],
            "chapter": chapter, "source_chunk_id": c["source_chunk_id"],
            "source_section_id": c.get("source_section_id", ""), "seed_terms": [],
        }
    for h in HANDWRITTEN:
        questions[h["qid"]] = {**h, "chapter": None, "source_chunk_id": None,
                               "source_section_id": None}

    print(f"{len(candidates)} candidates + {len(HANDWRITTEN)} hand-written; "
          f"{len(excluded)} excluded")
    for q, why in excluded:
        print(f"  excluded ({why}): {q}")

    print(f"\npooling {len(questions)} questions...")
    all_ids: set[int] = set()
    for q in questions.values():
        q["pool"] = await build_pool(pool, q["question"], q["source_chunk_id"],
                                     q["source_section_id"], q["seed_terms"])
        all_ids.update(q["pool"])
    sizes = [len(q["pool"]) for q in questions.values()]
    print(f"  pool size: min {min(sizes)}, max {max(sizes)}, mean {sum(sizes) / len(sizes):.1f}")

    snapshot = {"questions": questions, "chunks": await fetch_chunks(pool, all_ids)}
    POOLS_PATH.write_text(json.dumps(snapshot, indent=1), encoding="utf-8")
    return snapshot


# ---------------------------------------------------------------- calibration

async def calibrate(pool) -> bool:
    """Run the judges on cases verified by hand. Any failure stops the run:
    judges that get these wrong would produce a ground truth not worth having."""
    print("\ncalibrating judges on hand-verified cases...")

    async def find(mark: str, heading: str | None = None) -> int | None:
        sql = f"SELECT id FROM {TABLE} WHERE content LIKE '%' || $1 || '%'"
        args = [mark]
        if heading:
            sql += " AND metadata->>'heading' = $2"
            args.append(heading)
        return await pool.fetchval(sql + " ORDER BY id LIMIT 1", *args)

    definition = await find(CAL_DEFINITION_MARK, "19.4.1. Memory")
    other_param = await find(CAL_OTHER_PARAM_MARK)
    continuation = None
    if definition:
        row = await pool.fetchrow(f"SELECT content, metadata->>'section_id' AS s FROM {TABLE} WHERE id = $1",
                                  definition + 1)
        sec = await pool.fetchval(f"SELECT metadata->>'section_id' FROM {TABLE} WHERE id = $1", definition)
        if row and row["s"] == sec and CAL_CONTINUATION_MARK in row["content"]:
            continuation = definition + 1

    if not (definition and continuation and other_param):
        print(f"  could not locate calibration chunks (definition={definition}, "
              f"continuation={continuation}, other={other_param}).")
        print("  Has the corpus been re-chunked? Rerun with --skip-calibration only if you know why.")
        return False

    cal_ids = [definition, continuation, other_param]
    texts = await fetch_chunks(pool, cal_ids)
    items = [("cal_work_mem", CAL_WORK_MEM_QUESTION, cid, texts[str(cid)]["content"]) for cid in cal_ids]

    un_pool = await build_pool(pool, CAL_UNANSWERABLE_QUESTION, None, None, ["TimescaleDB", "hypertable"])
    un_texts = await fetch_chunks(pool, un_pool)
    items += [("cal_unanswerable", CAL_UNANSWERABLE_QUESTION, cid, un_texts[str(cid)]["content"])
              for cid in un_pool]

    labels = await label_items(items, "calibration")
    if ABORT.is_set():
        return False

    ok = True

    def show(cid, name, want):
        nonlocal ok
        v = labels[("cal_work_mem", cid)]
        passed = v["final"] is not None and want(v["final"])
        ok &= passed
        print(f"  {'pass' if passed else 'FAIL'}  #{cid} {name:28} "
              f"A={v['a']} B={v['b']}" + (f" C={v['c']}" if v["c"] else "") + f" -> {v['final']}")

    show(definition, "work_mem definition", lambda f: f == "yes")
    show(continuation, "overlap continuation", lambda f: f != "no")
    show(other_param, "autovacuum_work_mem", lambda f: f != "yes")

    # A label can be right for the wrong reason: gpt-4o-mini once labelled the
    # definition "yes" while answering that work_mem IS per-connection. Check
    # the substance too. The correct answer is "no"; an answer that opens with
    # "yes" has misread the passage, whatever label it chose.
    def_text = texts[str(definition)]["content"]
    for judge in (JUDGE_A, JUDGE_B):
        rec = CACHE.get(chunk_key(judge, "cal_work_mem", definition,
                                  CAL_WORK_MEM_QUESTION, def_text)) or {}
        answer = rec.get("answer", "")
        passed = bool(answer) and not answer.strip().lower().startswith("yes")
        ok &= passed
        print(f"  {'pass' if passed else 'FAIL'}  {judge[1]} answer on #{definition}: {answer[:110]!r}")

    un = [v for (qid, _), v in labels.items() if qid == "cal_unanswerable"]
    un_yes = [v for v in un if v["final"] == "yes"]
    un_missing = [v for v in un if v["final"] is None]
    passed = not un_yes and not un_missing
    ok &= passed
    print(f"  {'pass' if passed else 'FAIL'}  unanswerable: {len(un)} chunks, "
          f"{len(un_yes)} judged yes, {len(un_missing)} missing")
    for v in un_yes:
        print(f"        wrongly yes: {v['content'][:100]!r}")

    if not ok:
        print("\ncalibration FAILED -- judges disagree with hand-verified answers. "
              "Stopping before the main run.")
    return ok


# ---------------------------------------------------------------- reporting

def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in RANK) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def report(snapshot: dict, labels: dict, results: dict) -> None:
    questions, chunks = snapshot["questions"], snapshot["chunks"]

    pairs = [(v["a"], v["b"]) for v in labels.values() if v["a"] and v["b"]]
    matrix = {a: {b: 0 for b in RANK} for a in RANK}
    for a, b in pairs:
        matrix[a][b] += 1
    kappa = cohen_kappa(pairs)
    referred = [v for v in labels.values() if v["c"] is not None]

    accepted = {qid: r for qid, r in results.items() if r["status"] == "accept"}
    discarded = {qid: r for qid, r in results.items() if r["status"] == "discard"}
    pending = {qid: r for qid, r in results.items() if r["status"] == "pending"}

    gt_distractors = sum(1 for r in accepted.values() for c in r["ground_truth"]
                         if chunks[str(c)].get("group") == "distractor")

    print("\n" + "=" * 70)
    print("JUDGE AGREEMENT (chunk labels, A rows x B columns)")
    print("            " + "".join(f"{b:>9}" for b in RANK))
    for a in RANK:
        print(f"  {a:>8}  " + "".join(f"{matrix[a][b]:>9}" for b in RANK))
    print(f"  Cohen's kappa: {kappa:.3f}   ({len(pairs)} pairs)")
    print(f"  yes/no contradictions sent to referee: {len(referred)}  "
          f"-> {dict(Counter(v['c'] for v in referred))}")

    print("\nQUESTIONS")
    print(f"  accepted {len(accepted)}, discarded {len(discarded)}, pending {len(pending)}")
    by_type = Counter(questions[q]["type"] for q in accepted)
    by_match = Counter(r["match"] for r in accepted.values())
    print(f"  accepted by type:  {dict(by_type)}")
    print(f"  accepted by match: {dict(by_match)}")
    multi = [q for q in accepted if questions[q]["type"].startswith("multi_hop")]
    print(f"  multi-hop surviving: {len(multi)} of "
          f"{sum(1 for q in questions.values() if q['type'].startswith('multi_hop'))}")
    print(f"  ground-truth chunks from distractor chapters: {gt_distractors}")

    print("\n  discard reasons:")
    for reason, n in Counter(r["reason"] for r in discarded.values()).most_common():
        print(f"    {n:3}  {reason}")
    warns = [q for q, r in discarded.items() if r.get("warn")]
    for q in warns:
        print(f"  WARNING {q}: {questions[q]['question']!r} was meant to be unanswerable "
              f"but chunks {discarded[q]['yes']} were judged yes")

    if pending or any(ERRORS.values()):
        print(f"\n  INCOMPLETE: {len(pending)} questions pending, call errors {dict(ERRORS)}.")
        print("  Rerun the same command; cached judgments are reused, only gaps are filled.")

    REPORT_PATH.write_text(json.dumps({
        "prompt_version": PROMPT_VERSION,
        "judges": {"a": JUDGE_A[1], "b": JUDGE_B[1], "c": JUDGE_C[1]},
        "confusion_a_by_b": matrix, "kappa": kappa, "referred": len(referred),
        "accepted": len(accepted), "discarded": len(discarded), "pending": len(pending),
        "accepted_by_type": by_type, "accepted_by_match": by_match,
        "discard_reasons": Counter(r["reason"] for r in discarded.values()),
        "multi_hop_surviving": len(multi), "gt_distractor_chunks": gt_distractors,
        "errors": dict(ERRORS),
    }, indent=2), encoding="utf-8")

    # Optional human glance: a fixed random sample of accepted questions.
    sample = random.Random(SEED).sample(sorted(accepted), min(SPOT_CHECK_N, len(accepted)))
    print(f"\nSPOT CHECK ({len(sample)} accepted, fixed sample) -- optional, ~5 minutes")
    for qid in sample:
        q, r = questions[qid], accepted[qid]
        print(f"\n  [{qid}] {q['question']}   ({r['match']})")
        for c in r["ground_truth"][:3]:
            body = chunks[str(c)]["content"].split("\n\n", 1)[-1]
            print(f"      #{c} {chunks[str(c)].get('heading', '')[:50]}")
            print(f"         {' '.join(body.split())[:220]}")
        if not r["ground_truth"]:
            print("      (no answer expected)")


def write_outputs(snapshot: dict, results: dict) -> None:
    questions, chunks = snapshot["questions"], snapshot["chunks"]
    golden, discarded = [], []
    for qid in sorted(results):
        q, r = questions[qid], results[qid]
        if r["status"] == "accept":
            golden.append({
                "qid": qid, "question": q["question"], "type": q["type"],
                "chapter": q["chapter"], "match": r["match"],
                "ground_truth": r["ground_truth"],
                # Stored three ways: id for the harness; section id and a body
                # prefix to re-resolve ground truth if chunking ever changes.
                "ground_truth_detail": [
                    {"id": c, "section_id": chunks[str(c)].get("section_id", ""),
                     "group": chunks[str(c)].get("group", ""),
                     "content_head": chunks[str(c)]["content"].split("\n\n", 1)[-1][:150]}
                    for c in r["ground_truth"]],
                "source_chunk_id": q["source_chunk_id"],
            })
        elif r["status"] == "discard":
            discarded.append({"qid": qid, "question": q["question"],
                              "type": q["type"], "reason": r["reason"]})
    GOLDEN_PATH.write_text(json.dumps(golden, indent=2), encoding="utf-8")
    DISCARDED_PATH.write_text(json.dumps(discarded, indent=2), encoding="utf-8")
    print(f"\n{len(golden)} questions -> {GOLDEN_PATH}")
    print(f"{len(discarded)} discarded -> {DISCARDED_PATH}")


# ---------------------------------------------------------------- main

async def preflight() -> bool:
    """One tiny call per model, so a wrong model name or empty balance fails in
    seconds rather than after thousands of queued requests."""
    print("preflight: checking each judge model responds...")
    ok = True
    for judge in (JUDGE_A, JUDGE_B, JUDGE_C):
        r = await call_judge(judge, CHUNK_SYSTEM,
                             "Question: What is 2+2?\n\nPassage:\n# Arithmetic\n\n2+2 equals 4.",
                             ChunkJudgment)
        print(f"  {judge[1]:30} {'ok' if r else 'FAILED'}")
        ok &= r is not None
    return ok


async def run(args) -> None:
    global CACHE
    EVAL_DIR.mkdir(exist_ok=True)
    if args.clear_cache:
        CACHE_PATH.unlink(missing_ok=True)
    if args.reset:
        POOLS_PATH.unlink(missing_ok=True)
    CACHE = Cache(CACHE_PATH)
    init_limits()

    db = await connect()
    try:
        if POOLS_PATH.exists():
            snapshot = json.loads(POOLS_PATH.read_text(encoding="utf-8"))
            print(f"using existing pools ({len(snapshot['questions'])} questions); --reset to rebuild")
        else:
            snapshot = await build_pools(db)

        questions, chunks = snapshot["questions"], snapshot["chunks"]
        items = [(qid, q["question"], cid, chunks[str(cid)]["content"])
                 for qid, q in questions.items() for cid in q["pool"]]
        todo_a = sum(1 for qid, q, cid, text in items
                     if not CACHE.get(chunk_key(JUDGE_A, qid, cid, q, text)))
        todo_b = sum(1 for qid, q, cid, text in items
                     if not CACHE.get(chunk_key(JUDGE_B, qid, cid, q, text)))
        print(f"\n{len(items)} (question, chunk) pairs; uncached: "
              f"A {todo_a}, B {todo_b}; estimated time ~{todo_b / ANTHROPIC_RPM:.0f} min "
              f"(Anthropic-bound)")

        if args.dry_run:
            return
        if not await preflight():
            print("preflight failed; fix keys, balance, or model names and rerun.")
            return
        if not args.skip_calibration and not await calibrate(db):
            return
    finally:
        await db.close()

    print("\njudging all pairs...")
    labels = await label_items(items, "main")

    results = {}
    for qid, q in questions.items():
        results[qid] = await decide(q, labels)

    write_outputs(snapshot, results)
    report(snapshot, labels, results)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="count calls, spend nothing")
    ap.add_argument("--reset", action="store_true", help="rebuild pools (keeps judgment cache)")
    ap.add_argument("--clear-cache", action="store_true", help="discard all cached judgments")
    ap.add_argument("--skip-calibration", action="store_true")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()