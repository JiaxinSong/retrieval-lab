"""
Answer questions from the PostgreSQL docs with citations, and retrieve again
when the first retrieval looks weak.

    python3 scripts/08_agent.py --ask "Is work_mem a per-connection limit?"
    python3 scripts/08_agent.py                 # run the golden set, write eval/answers.json
    python3 scripts/08_agent.py --limit 10      # first 10 golden questions only
    python3 scripts/08_agent.py --threshold 1.0 # force the agentic loop on every question

HOW ONE QUESTION IS ANSWERED

1. Retrieve with the best configuration from step 07 (hybrid + rerank, top 5).
2. Read the top rerank score. Step 07 found every answerable question scoring
   at least 0.511 and both unanswerable ones below it (0.360, 0.481), so the
   score is a cheap signal of whether the evidence covers the question.
3. Strong evidence (score >= RETRY_THRESHOLD): one model call. The model may
   only call the `answer` tool. No retry machinery, no extra cost.
4. Weak evidence: the model also gets a `search_docs` tool and is told the
   first search looked weak. It decides whether the passages are enough; if
   not, it rewrites the query and searches again, up to MAX_SEARCHES in total,
   then answers -- or says the documentation does not answer the question.

Why a score gate instead of always letting the model decide: an always-on
agent pays an extra decision on every question, and step 07 showed the score
already separates strong from weak retrieval on this data. The gate keeps the
common case at one call and spends the agentic loop only where it can help.

Why a hard cap on searches: a model told "search again if needed" can keep
rephrasing forever. MAX_SEARCHES bounds both cost and latency.

ANSWERS

The model answers through a tool with a fixed schema -- the answer text, the
passage numbers it cites, and whether the passages answer the question at
all. Citations are passage numbers the model was shown, mapped back to chunk
ids, so every claim can be traced to a URL. `answerable: false` is how the
model refuses instead of filling a gap from its own knowledge.

WHAT THE GOLDEN-SET RUN MEASURES

Retrieval-side effects, which the golden set can score without new judging:

- how many questions take the agentic path, and how many searches they use;
- for answerable questions on that path, whether the extra searches found a
  ground-truth chunk the first search missed;
- whether the model's citations include a ground-truth chunk -- for a
  multi-hop ("all") question, every ground-truth chunk;
- whether unanswerable questions are refused, and answerable ones are not.

THE THRESHOLD WAS READ OFF THE TEST SET

RETRY_THRESHOLD comes from step 07's rerank scores on these same 81
questions, so on this set it routes almost every answerable question to the
direct path by construction. That says nothing about how the threshold
generalises. Running with --threshold 1.0 forces the agentic loop on every
question and shows what the loop itself does: whether a model that may search
again finds anything the first search missed, and whether giving it that
option makes it refuse answerable questions.

It does not judge whether each answer is correct and complete. That needs an
answer-level judge, which this step does not add. A citation that points at a
chunk outside the judged pool -- possible after a rewritten search -- is
counted separately as unjudged, not as wrong.
"""

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openai import AsyncOpenAI  # noqa: E402
from pydantic import BaseModel, Field, ValidationError  # noqa: E402

from retrieval import CONFIGS, TABLE, connect  # noqa: E402

EVAL_DIR = ROOT / "eval"
GOLDEN_PATH = EVAL_DIR / "golden.json"
POOLS_PATH = EVAL_DIR / "pools.json"
RUNS_PATH = EVAL_DIR / "runs.jsonl"            # step 07's retrieval cache
AGENT_CACHE_PATH = EVAL_DIR / "agent_runs.jsonl"
ANSWERS_PATH = EVAL_DIR / "answers.json"

MODEL = "gpt-5-mini"
PROMPT_VERSION = "v1"
RETRIEVAL_CONFIG = "hybrid+rerank"
RETRY_THRESHOLD = 0.5      # top rerank score below this takes the agentic path
MAX_SEARCHES = 3           # including the first one
MAX_TURNS = MAX_SEARCHES + 2
COHERE_INTERVAL = 6.5      # seconds between rerank calls (trial-key safe)

client = AsyncOpenAI()


# ---------------------------------------------------------------- prompts and tools

SYSTEM = """You answer questions about PostgreSQL using only the numbered passages you are given from the PostgreSQL 17 documentation.

Rules:
- Use only the passages. Do not add facts from your own knowledge.
- Cite the passages that support each claim with their numbers in square brackets, for example [2] or [1][3].
- If the passages do not contain the answer, set answerable to false and say that the documentation provided does not answer the question. A partial answer is fine when you say what is missing.
- Keep the answer short: a few sentences, the way an experienced engineer would answer a colleague.
- Finish by calling the answer tool."""

WEAK_EVIDENCE_NOTE = """The first search returned weak matches (top relevance score {score:.2f}). Before answering, check whether the passages actually answer the question. If they do not, call search_docs with a rewritten query: use different words, use a term you found in the passages, or target the part of the question that is still unanswered. You can search {remaining} more time(s). If the passages still do not answer it, answer with answerable set to false."""

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Search the PostgreSQL documentation and get the top passages for a query.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A search query, different from earlier ones."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "answer",
        "description": "Give the final answer.",
        # Strict mode makes the API enforce the schema. Without it, the first
        # real run got an answer with `answerable` missing.
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "The answer, with [n] citations."},
                "citations": {"type": "array", "items": {"type": "integer"},
                              "description": "Numbers of the passages the answer relies on."},
                "answerable": {"type": "boolean",
                               "description": "False if the passages do not answer the question."},
            },
            "required": ["answer", "citations", "answerable"],
            "additionalProperties": False,
        },
    },
}


class Answer(BaseModel):
    answer: str
    citations: list[int] = Field(default_factory=list)
    answerable: bool


def openai_sampling(model: str) -> dict:
    """Reasoning models (gpt-5, o-series) reject temperature and take a
    reasoning effort instead; the same rule as step 06."""
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        return {"reasoning_effort": "low"}
    return {"temperature": 0}


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- retrieval

class Searcher:
    """hybrid + rerank, with the same guards as step 07: Cohere calls are
    spaced, and a result without rerank scores (the pipeline's silent
    fallback) is retried rather than used."""

    def __init__(self, pool, interval: float) -> None:
        self.pool = pool
        self.interval = interval
        self.last = 0.0
        self.calls = 0

    async def search(self, query: str) -> list[dict]:
        for wait in (0, 15, 30, 60):
            if wait:
                print(f"    rerank fell back to fused order; retrying in {wait}s")
                await asyncio.sleep(wait)
            delay = self.last + self.interval - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self.last = time.monotonic()
            self.calls += 1
            hits = await CONFIGS[RETRIEVAL_CONFIG](self.pool, query)
            if all(h.rerank_score is not None for h in hits):
                return [{"id": h.id, "heading": h.heading, "url": h.metadata.get("url", ""),
                         "content": h.content, "score": h.rerank_score} for h in hits]
        raise RuntimeError(f"rerank kept failing for {query!r}")

    async def from_step07_cache(self, cached_hits: list[dict]) -> list[dict]:
        """Rebuild step 07's cached top 5 (ids and scores only) with content
        from the database, so the golden-set run does not repeat 81 rerank
        calls already made."""
        ids = [h["id"] for h in cached_hits]
        rows = await self.pool.fetch(
            f"SELECT id, content, metadata FROM {TABLE} WHERE id = ANY($1::int[])", ids)
        by_id = {r["id"]: r for r in rows}
        out = []
        for h in cached_hits:
            r = by_id[h["id"]]
            meta = json.loads(r["metadata"])
            out.append({"id": h["id"], "heading": meta.get("heading", ""), "url": meta.get("url", ""),
                        "content": r["content"], "score": h["rerank_score"]})
        return out


# ---------------------------------------------------------------- the agent

class Passages:
    """Every passage shown to the model, numbered once. A chunk returned by a
    second search keeps its first number, so citations stay unambiguous."""

    def __init__(self) -> None:
        self.items: list[dict] = []
        self.number_of: dict[int, int] = {}

    def add(self, hits: list[dict]) -> tuple[list[dict], list[int]]:
        new, repeated = [], []
        for h in hits:
            if h["id"] in self.number_of:
                repeated.append(self.number_of[h["id"]])
                continue
            n = len(self.items) + 1
            self.number_of[h["id"]] = n
            item = {**h, "n": n}
            self.items.append(item)
            new.append(item)
        return new, repeated

    def chunk_id(self, n: int) -> int | None:
        return self.items[n - 1]["id"] if 1 <= n <= len(self.items) else None


def format_passages(items: list[dict]) -> str:
    return "\n\n".join(f"[{p['n']}] (relevance {p['score']:.2f})\n{p['content']}" for p in items)


async def answer_question(question: str, searcher: Searcher,
                          first_hits: list[dict] | None = None,
                          threshold: float = RETRY_THRESHOLD) -> dict:
    passages = Passages()
    hits = first_hits if first_hits is not None else await searcher.search(question)
    passages.add(hits)
    top1 = hits[0]["score"] if hits else 0.0
    agentic = top1 < threshold

    user = f"Question: {question}\n\nPassages:\n{format_passages(passages.items)}"
    if agentic:
        user += "\n\n" + WEAK_EVIDENCE_NOTE.format(score=top1, remaining=MAX_SEARCHES - 1)
    messages: list[dict] = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": user}]

    trace = [{"query": question, "top1": top1, "ids": [h["id"] for h in hits]}]
    searches, llm_calls, malformed_retries = 1, 0, 0
    force_answer = False

    for _ in range(MAX_TURNS):
        can_search = agentic and not force_answer and searches < MAX_SEARCHES
        tools = [ANSWER_TOOL, SEARCH_TOOL] if can_search else [ANSWER_TOOL]
        tool_choice = "required" if can_search else {"type": "function", "function": {"name": "answer"}}

        resp = await client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools, tool_choice=tool_choice,
            **openai_sampling(MODEL))
        llm_calls += 1
        msg = resp.choices[0].message
        calls = msg.tool_calls or []
        assistant = {"role": "assistant", "content": msg.content}
        if calls:
            assistant["tool_calls"] = [c.model_dump() for c in calls]
        messages.append(assistant)
        if not calls:
            # tool_choice should prevent this; nudge rather than loop silently.
            messages.append({"role": "user", "content": "Call the answer tool now."})
            continue

        final = next((c for c in calls if c.function.name == "answer"), None)
        if final is not None:
            try:
                ans = Answer.model_validate_json(final.function.arguments)
            except ValidationError as e:
                # Strict mode should prevent this. If it happens anyway, say
                # what was wrong and ask once more rather than crash the run;
                # every tool call in the message needs a reply first.
                if malformed_retries >= 1:
                    raise RuntimeError(f"malformed answer for {question!r}: {e}") from e
                malformed_retries += 1
                for call in calls:
                    note = (f"Invalid answer: {e.errors()[0]['msg']} for "
                            f"{e.errors()[0]['loc']}. Call answer again with all of "
                            "answer, citations, and answerable."
                            if call is final else "Ignored: answer again with the answer tool.")
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": note})
                force_answer = True      # the retry must be an answer, not a search
                continue
            cited = [passages.chunk_id(n) for n in ans.citations]
            return {
                "answer": ans.answer, "answerable": ans.answerable,
                "citations": ans.citations,
                "cited_ids": [c for c in cited if c is not None],
                "invalid_citations": [n for n, c in zip(ans.citations, cited) if c is None],
                "path": "agentic" if agentic else "direct",
                "searches": searches, "llm_calls": llm_calls, "trace": trace,
                "shown_ids": [p["id"] for p in passages.items],
                "sources": {p["n"]: {"id": p["id"], "heading": p["heading"], "url": p["url"]}
                            for p in passages.items},
            }

        for call in calls:
            if call.function.name != "search_docs":
                continue
            if searches >= MAX_SEARCHES:
                content = "Search limit reached. Answer with the passages you have."
            else:
                query = json.loads(call.function.arguments)["query"]
                hits = await searcher.search(query)
                searches += 1
                new, repeated = passages.add(hits)
                trace.append({"query": query, "top1": hits[0]["score"] if hits else 0.0,
                              "ids": [h["id"] for h in hits]})
                content = format_passages(new) if new else "No new passages."
                if repeated:
                    content += f"\n\nAlso returned, already shown: {', '.join(f'[{n}]' for n in repeated)}."
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

    raise RuntimeError(f"no answer after {MAX_TURNS} turns for {question!r}")


# ---------------------------------------------------------------- golden-set run

class AgentCache:
    def __init__(self, path: Path, enabled: bool) -> None:
        self.path = path
        self.records: dict[str, dict] = {}
        if enabled and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.records[rec["key"]] = rec

    @staticmethod
    def key(qid: str, question: str, threshold: float) -> str:
        return f"{PROMPT_VERSION}|{MODEL}|{threshold}|{MAX_SEARCHES}|{qid}|{fingerprint(question)}"

    def get(self, key: str) -> dict | None:
        rec = self.records.get(key)
        return rec["result"] if rec else None

    def put(self, key: str, result: dict) -> None:
        self.records[key] = {"key": key, "result": result}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "result": result}) + "\n")


def load_step07_runs() -> dict[str, list[dict]]:
    runs = {}
    if RUNS_PATH.exists():
        for line in RUNS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                runs[rec["key"]] = rec["hits"]
    return runs


def score(q: dict, r: dict, pool_ids: set[int]) -> dict:
    gt = set(q["ground_truth"])
    first_ids = set(r["trace"][0]["ids"])
    shown = set(r["shown_ids"])
    cited = set(r["cited_ids"])
    out = {"qid": q["qid"], "type": q["type"], "match": q["match"], "path": r["path"],
           "searches": r["searches"], "top1": r["trace"][0]["top1"],
           "answerable_pred": r["answerable"], "invalid_citations": len(r["invalid_citations"]),
           "unjudged_citations": len(cited - pool_ids)}
    if q["match"] == "none":
        out["correct_refusal"] = not r["answerable"]
        return out
    # An `all` question needs every ground-truth chunk; an `any` question
    # needs one. The first version scored both as "any overlap", which would
    # count a multi-hop answer citing half its evidence as complete.
    if q["match"] == "all":
        covered = lambda ids: gt <= ids
    else:
        covered = lambda ids: bool(gt & ids)
    out.update({
        "found_first": covered(first_ids),
        "found_any_search": covered(shown),
        "cites_ground_truth": covered(cited),
        "false_refusal": not r["answerable"],
    })
    return out


def report(rows: list[dict], searcher_calls: int, threshold: float) -> dict:
    ans = [r for r in rows if r["match"] != "none"]
    una = [r for r in rows if r["match"] == "none"]
    agentic = [r for r in rows if r["path"] == "agentic"]
    agentic_ans = [r for r in agentic if r["match"] != "none"]

    summary = {
        "questions": len(rows),
        "direct": len(rows) - len(agentic),
        "agentic": len(agentic),
        "agentic_searches_mean": (sum(r["searches"] for r in agentic) / len(agentic)) if agentic else 0,
        "answerable": len(ans),
        "found_first": sum(r["found_first"] for r in ans),
        "found_any_search": sum(r["found_any_search"] for r in ans),
        "agentic_answerable": len(agentic_ans),
        "agentic_recovered": sum(1 for r in agentic_ans if r["found_any_search"] and not r["found_first"]),
        "cites_ground_truth": sum(r["cites_ground_truth"] for r in ans),
        "false_refusals": sum(r["false_refusal"] for r in ans),
        "unanswerable": len(una),
        "correct_refusals": sum(r["correct_refusal"] for r in una),
        "invalid_citations": sum(r["invalid_citations"] for r in rows),
        "unjudged_citations": sum(r["unjudged_citations"] for r in rows),
        "rerank_calls_this_run": searcher_calls,
        "threshold": threshold,
    }

    s = summary
    print("\n" + "=" * 70)
    print(f"ANSWERS  ({s['questions']} questions, threshold {threshold}, max {MAX_SEARCHES} searches)\n")
    print(f"  path:                 {s['direct']} direct, {s['agentic']} agentic "
          f"(mean {s['agentic_searches_mean']:.1f} searches)")
    print(f"  answerable questions: {s['answerable']}")
    print(f"    ground truth in first search:   {s['found_first']}")
    print(f"    ground truth in any search:     {s['found_any_search']}")
    print(f"    agentic path recovered:         {s['agentic_recovered']} of {s['agentic_answerable']} "
          "agentic answerable questions")
    print(f"    citations include ground truth: {s['cites_ground_truth']}")
    print(f"    refused although answerable:    {s['false_refusals']}")
    print(f"  unanswerable questions: {s['unanswerable']}, correctly refused: {s['correct_refusals']}")
    print(f"  invalid citation numbers: {s['invalid_citations']}; "
          f"citations to unjudged chunks: {s['unjudged_citations']}")
    return summary


async def run_golden(args) -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if args.limit:
        golden = golden[: args.limit]
    pools = json.loads(POOLS_PATH.read_text(encoding="utf-8"))["questions"]
    step07 = load_step07_runs()
    cache = AgentCache(AGENT_CACHE_PATH, enabled=not args.fresh)

    db = await connect()
    searcher = Searcher(db, args.cohere_interval)
    rows, results = [], {}
    try:
        for n, q in enumerate(golden, 1):
            key = AgentCache.key(q["qid"], q["question"], args.threshold)
            r = cache.get(key)
            if r is None:
                first = None
                cached = step07.get(f"{RETRIEVAL_CONFIG}|{q['qid']}|{fingerprint(q['question'])}")
                if cached:
                    first = await searcher.from_step07_cache(cached)
                r = await answer_question(q["question"], searcher, first, args.threshold)
                cache.put(key, r)
            results[q["qid"]] = r
            rows.append(score(q, r, set(pools.get(q["qid"], {}).get("pool", []))))
            mark = "agentic" if r["path"] == "agentic" else "direct "
            print(f"  [{n:2}/{len(golden)}] {q['qid']:5} {mark} searches={r['searches']} "
                  f"answerable={r['answerable']}")
    finally:
        await db.close()

    summary = report(rows, searcher.calls, args.threshold)
    out_path = ANSWERS_PATH if args.threshold == RETRY_THRESHOLD else \
        EVAL_DIR / f"answers_threshold_{args.threshold}.json"
    out_path.write_text(json.dumps({"summary": summary, "rows": rows, "answers": results},
                                       indent=2), encoding="utf-8")
    print(f"\n-> {out_path}")


async def ask(question: str, interval: float, threshold: float) -> None:
    db = await connect()
    try:
        r = await answer_question(question, Searcher(db, interval), threshold=threshold)
    finally:
        await db.close()
    print(f"\n{r['answer']}\n")
    for step in r["trace"]:
        print(f"  search: {step['query']!r}  top score {step['top1']:.2f}")
    print(f"  path: {r['path']}, answerable: {r['answerable']}\n\nSources:")
    for n in sorted(set(r["citations"])):
        src = r["sources"].get(n)
        if src:
            print(f"  [{n}] {src['heading']} — {src['url']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ask", help="answer one question and print it with sources")
    ap.add_argument("--limit", type=int, help="only the first N golden questions")
    ap.add_argument("--fresh", action="store_true", help="ignore cached answers")
    ap.add_argument("--cohere-interval", type=float, default=COHERE_INTERVAL)
    ap.add_argument("--threshold", type=float, default=RETRY_THRESHOLD,
                    help="top rerank score below which the agentic path runs; 1.0 forces it on every question")
    args = ap.parse_args()
    if args.ask:
        asyncio.run(ask(args.ask, args.cohere_interval, args.threshold))
    else:
        asyncio.run(run_golden(args))


if __name__ == "__main__":
    main()