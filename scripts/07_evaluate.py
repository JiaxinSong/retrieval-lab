"""
Score every retrieval configuration against the golden set.

Input:  eval/golden.json (step 06), eval/pools.json (step 06)
Output: eval/results.json          per configuration, per question: top 5 and metrics
        eval/results_summary.json  the aggregate table
        eval/runs.jsonl            cache of retrieval runs, so a rerun is free

METRICS (top 5, answerable questions only)

  Recall@5     "any": 1 if any ground-truth chunk is in the top 5, else 0.
               "all": the fraction of ground-truth chunks in the top 5.
  MRR@5        1 / rank of the first ground-truth chunk in the top 5; 0 if none.
  Hit@1        1 if the first result is a ground-truth chunk.
  Intrusion    distractor-chapter chunks in the top 5 that are not part of the
               ground truth. A distractor the judges labelled correct is an
               answer, not an intrusion.
  Sections     distinct document sections in the top 5, and how often one
               section fills three or more of the five slots -- what the
               section cap exists to prevent.

Unanswerable questions ("none") have no recall to measure and are reported
separately: for reranked configurations, their top rerank score, next to the
same score on answerable questions. That comparison is the first look at
whether a low score could trigger a retry in the agentic layer.

THREE CHECKS THAT KEEP THE NUMBERS HONEST

1. Unjudged results. Every chunk a configuration returns should have been
   judged in step 06 -- the pool was built from the same top-20 lists every
   configuration draws from. A returned chunk outside the pool is counted as
   wrong without anyone having looked at it. The expected count is zero; any
   other number is printed and means the pool and the retrieval have drifted
   apart (for example, a retrieval change after step 06).

2. Silent rerank fallback. retrieval.rerank() returns the fused order when the
   Cohere call fails, so that a live query degrades instead of failing. In an
   evaluation that behaviour is a trap: a rate-limited run would score plain
   hybrid retrieval under the rerank label, and nothing would look wrong. Here
   a reranked result without rerank scores is treated as a failure, retried
   with backoff, and the run stops if it keeps failing.

3. Differences too small to mean anything. With about 80 questions, one
   question is more than one percentage point. Comparisons are reported as
   question counts -- how many questions configuration A gets right that B
   gets wrong, and the reverse -- and a difference of fewer than
   MIN_MEANINGFUL_DIFF questions is marked as noise.

OPERATION

    python3 scripts/07_evaluate.py                      # all configurations
    python3 scripts/07_evaluate.py --configs vector hybrid
    python3 scripts/07_evaluate.py --fresh              # ignore the run cache
    python3 scripts/07_evaluate.py --cohere-interval 0.2   # production Cohere key

Reranked configurations call Cohere once per question. The default spacing
of COHERE_INTERVAL seconds keeps a trial key under its per-minute limit; with
a production key, lower it.
"""

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval import CONFIGS, RERANK_TOP_N, connect  # noqa: E402

EVAL_DIR = ROOT / "eval"
GOLDEN_PATH = EVAL_DIR / "golden.json"
POOLS_PATH = EVAL_DIR / "pools.json"
RUNS_PATH = EVAL_DIR / "runs.jsonl"
RESULTS_PATH = EVAL_DIR / "results.json"
SUMMARY_PATH = EVAL_DIR / "results_summary.json"

TOP_K = RERANK_TOP_N
RERANK_CONFIGS = {"hybrid+rerank", "hybrid+cap+rerank", "hybrid+rerank+nocap", "vector+rerank"}

COHERE_INTERVAL = 6.5            # seconds between rerank calls (~9/min, trial-key safe)
RERANK_RETRY_WAITS = (15, 30, 60)  # backoff after a silent fallback
MIN_MEANINGFUL_DIFF = 2          # questions; smaller differences are noise

# Pairs worth comparing directly, each answering one design question.
COMPARISONS = [
    ("vector", "hybrid", "Fusion + cap versus vector search alone (differs in two things)"),
    ("vector", "hybrid+nocap", "Does fusing three retrievers beat vector search alone? (no cap on either)"),
    ("hybrid+nocap", "hybrid", "Does the section cap help without reranking?"),
    ("vector", "vector+rerank", "Does reranking help vector search alone?"),
    ("vector+rerank", "hybrid+rerank", "With a reranker, do the lexical and identifier retrievers still add anything?"),
    ("hybrid", "hybrid+rerank", "Does reranking earn its cost?"),
    ("hybrid+cap+rerank", "hybrid+rerank", "Does rerank-then-cap beat cap-then-rerank?"),
    ("hybrid+rerank+nocap", "hybrid+rerank", "Does the section cap help after reranking?"),
    ("lexical", "identifier", "How do the two exact-match retrievers compare?"),
]


class RerankFallback(Exception):
    """A reranked configuration returned results without rerank scores."""


# ---------------------------------------------------------------- run cache

def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class RunCache:
    """Append-only JSONL of retrieval runs, keyed by configuration, question
    id, and a hash of the question text -- the same reasoning as the judgment
    cache in step 06: ids are positions, the hash is the content."""

    def __init__(self, path: Path, enabled: bool) -> None:
        self.path = path
        self.records: dict[str, dict] = {}
        if enabled and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.records[rec["key"]] = rec

    @staticmethod
    def key(config: str, qid: str, question: str) -> str:
        return f"{config}|{qid}|{fingerprint(question)}"

    def get(self, key: str) -> dict | None:
        return self.records.get(key)

    def put(self, key: str, hits: list[dict]) -> None:
        record = {"key": key, "hits": hits}
        self.records[key] = record
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------- retrieval

class CoherePacer:
    """Space rerank calls so a trial key's per-minute limit is never hit."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.last = 0.0

    async def wait(self) -> None:
        delay = self.last + self.interval - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        self.last = time.monotonic()


async def retrieve(pool, config: str, question: str, pacer: CoherePacer) -> list[dict]:
    """Run one configuration on one question. For reranked configurations,
    a result without rerank scores means the Cohere call failed and the
    pipeline fell back to fused order -- retried, never scored."""
    attempts = len(RERANK_RETRY_WAITS) + 1 if config in RERANK_CONFIGS else 1

    for attempt in range(attempts):
        if config in RERANK_CONFIGS:
            await pacer.wait()
        hits = await CONFIGS[config](pool, question)
        top = hits[:TOP_K]

        if config in RERANK_CONFIGS and any(h.rerank_score is None for h in top):
            if attempt < attempts - 1:
                wait = RERANK_RETRY_WAITS[attempt]
                print(f"    rerank fell back to fused order; retrying in {wait}s")
                await asyncio.sleep(wait)
                continue
            raise RerankFallback(f"{config}: rerank kept failing for {question!r}")

        return [{"id": h.id, "group": h.group, "section_id": h.section_id,
                 "rerank_score": h.rerank_score} for h in top]
    raise AssertionError("unreachable")


# ---------------------------------------------------------------- metrics

def score_question(q: dict, hits: list[dict], pool_ids: set[int]) -> dict:
    ids = [h["id"] for h in hits]
    gt = set(q["ground_truth"])
    unjudged = [i for i in ids if i not in pool_ids]

    section_counts = Counter(h["section_id"] or f"__{h['id']}" for h in hits)
    result = {"top": ids, "unjudged": unjudged,
              "top1_rerank": hits[0]["rerank_score"] if hits else None,
              "distinct_sections": len(section_counts),
              "max_same_section": max(section_counts.values(), default=0)}

    if q["match"] == "none":
        return result

    found = [i for i in ids if i in gt]
    if q["match"] == "all":
        recall = len(set(found)) / len(gt)
    else:
        recall = 1.0 if found else 0.0

    first = next((rank for rank, i in enumerate(ids, 1) if i in gt), None)
    intrusions = sum(1 for h in hits if h["group"] == "distractor" and h["id"] not in gt)

    result.update({
        "recall": recall,
        "rr": 1.0 / first if first else 0.0,
        "hit1": 1.0 if ids and ids[0] in gt else 0.0,
        "intrusions": intrusions,
    })
    return result


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarise(golden: list[dict], per_q: dict[str, dict]) -> dict:
    answerable = [q for q in golden if q["match"] != "none"]
    rows = [per_q[q["qid"]] for q in answerable]

    by_type: dict[str, list[float]] = defaultdict(list)
    for q in answerable:
        by_type[q["type"]].append(per_q[q["qid"]]["recall"])

    unanswerable = [per_q[q["qid"]]["top1_rerank"] for q in golden if q["match"] == "none"]
    answerable_top1 = [r["top1_rerank"] for r in rows if r["top1_rerank"] is not None]

    return {
        "n": len(rows),
        "recall@5": mean([r["recall"] for r in rows]),
        "mrr@5": mean([r["rr"] for r in rows]),
        "hit@1": mean([r["hit1"] for r in rows]),
        "intrusions_per_question": mean([r["intrusions"] for r in rows]),
        "questions_with_intrusion": sum(1 for r in rows if r["intrusions"] > 0),
        "distinct_sections_per_question": mean([r["distinct_sections"] for r in rows]),
        "questions_with_3plus_same_section": sum(1 for r in rows if r["max_same_section"] >= 3),
        "unjudged_chunks": sum(len(per_q[q["qid"]]["unjudged"]) for q in golden),
        "recall@5_by_type": {t: {"n": len(v), "recall@5": mean(v)} for t, v in sorted(by_type.items())},
        "top1_rerank_answerable": answerable_top1,
        "top1_rerank_unanswerable": [s for s in unanswerable if s is not None],
    }


def compare(golden: list[dict], results: dict, a: str, b: str, why: str) -> dict:
    """Question-level wins and losses on Recall@5. A question counts as a win
    for B when B scores higher on it than A, and the reverse."""
    wins_b, wins_a = [], []
    for q in golden:
        if q["match"] == "none":
            continue
        ra, rb = results[a][q["qid"]]["recall"], results[b][q["qid"]]["recall"]
        if rb > ra:
            wins_b.append(q["qid"])
        elif ra > rb:
            wins_a.append(q["qid"])
    return {"question": why, "a": a, "b": b, "b_better": wins_b, "a_better": wins_a,
            "net": len(wins_b) - len(wins_a)}


# ---------------------------------------------------------------- report

def print_report(summaries: dict, comparisons: list[dict]) -> None:
    print("\n" + "=" * 78)
    print(f"RESULTS  (top {TOP_K}, {next(iter(summaries.values()))['n']} answerable questions)\n")
    print(f"| {'Configuration':20} | Recall@5 | MRR@5 | Hit@1 | Intrusions/q | Sections/q | 3+ same | Unjudged |")
    print(f"| {'-' * 20} | -------- | ----- | ----- | ------------ | ---------- | ------- | -------- |")
    for name, s in summaries.items():
        print(f"| {name:20} | {s['recall@5']:8.3f} | {s['mrr@5']:5.3f} | {s['hit@1']:5.3f} "
              f"| {s['intrusions_per_question']:12.2f} | {s['distinct_sections_per_question']:10.2f} "
              f"| {s['questions_with_3plus_same_section']:7} | {s['unjudged_chunks']:8} |")
    print("\n  Sections/q: distinct sections in the top 5.  3+ same: questions where one")
    print("  section fills three or more of the five slots (the cap allows at most 2).")

    print("\nRecall@5 by question type:")
    types = sorted({t for s in summaries.values() for t in s["recall@5_by_type"]})
    for t in types:
        n = next(s["recall@5_by_type"][t]["n"] for s in summaries.values() if t in s["recall@5_by_type"])
        cells = "  ".join(f"{name}={s['recall@5_by_type'].get(t, {}).get('recall@5', 0):.2f}"
                          for name, s in summaries.items())
        print(f"  {t} (n={n}): {cells}")

    print("\nHead-to-head (questions where one configuration's Recall@5 is higher):")
    for c in comparisons:
        verdict = ("noise" if abs(c["net"]) < MIN_MEANINGFUL_DIFF
                   else f"{c['b'] if c['net'] > 0 else c['a']} better")
        print(f"  {c['question']}")
        print(f"    {c['b']} better on {len(c['b_better'])}, {c['a']} better on "
              f"{len(c['a_better'])}  -> net {c['net']:+d}  ({verdict})")

    for name in [c for c in summaries if c in RERANK_CONFIGS]:
        s = summaries[name]
        ans, una = s["top1_rerank_answerable"], s["top1_rerank_unanswerable"]
        if ans and una:
            print(f"\nTop-1 rerank score, {name}:")
            print(f"  answerable   n={len(ans)}  min {min(ans):.3f}  mean {mean(ans):.3f}  max {max(ans):.3f}")
            print(f"  unanswerable n={len(una)}  " + ", ".join(f"{x:.3f}" for x in una))

    unjudged = sum(s["unjudged_chunks"] for s in summaries.values())
    if unjudged:
        print(f"\nWARNING: {unjudged} returned chunks were never judged. Those count as "
              "wrong without evidence; see 'unjudged' in eval/results.json.")


# ---------------------------------------------------------------- main

async def run(args) -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    pools = json.loads(POOLS_PATH.read_text(encoding="utf-8"))["questions"]
    pool_ids = {qid: set(q["pool"]) for qid, q in pools.items()}

    configs = args.configs or list(CONFIGS)
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        sys.exit(f"unknown configuration(s): {unknown}; choose from {list(CONFIGS)}")

    cache = RunCache(RUNS_PATH, enabled=not args.fresh)
    pacer = CoherePacer(args.cohere_interval)

    todo_rerank = sum(1 for c in configs if c in RERANK_CONFIGS for q in golden
                      if not cache.get(RunCache.key(c, q["qid"], q["question"])))
    print(f"{len(golden)} questions x {len(configs)} configurations; "
          f"{todo_rerank} uncached rerank calls (~{todo_rerank * args.cohere_interval / 60:.0f} min "
          f"at {args.cohere_interval}s spacing)")

    db = await connect()
    results: dict[str, dict] = {}
    try:
        for config in configs:
            print(f"\n{config}")
            per_q = {}
            for n, q in enumerate(golden, 1):
                key = RunCache.key(config, q["qid"], q["question"])
                cached = cache.get(key)
                if cached:
                    hits = cached["hits"]
                else:
                    hits = await retrieve(db, config, q["question"], pacer)
                    cache.put(key, hits)
                per_q[q["qid"]] = score_question(q, hits, pool_ids.get(q["qid"], set()))
                if n % 20 == 0 or n == len(golden):
                    print(f"  {n}/{len(golden)}")
            results[config] = per_q
    except RerankFallback as e:
        print(f"\nSTOPPED: {e}\nCompleted runs are cached; rerun to continue, "
              "with a larger --cohere-interval if the key is rate limited.")
        return
    finally:
        await db.close()

    summaries = {c: summarise(golden, results[c]) for c in configs}
    comparisons = [compare(golden, results, a, b, why) for a, b, why in COMPARISONS
                   if a in results and b in results]

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps({"top_k": TOP_K, "configs": summaries,
                                        "comparisons": comparisons}, indent=2), encoding="utf-8")
    print_report(summaries, comparisons)
    print(f"\n-> {RESULTS_PATH}\n-> {SUMMARY_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--configs", nargs="+", help=f"subset of {list(CONFIGS)}")
    ap.add_argument("--fresh", action="store_true", help="ignore cached retrieval runs")
    ap.add_argument("--cohere-interval", type=float, default=COHERE_INTERVAL,
                    help="seconds between rerank calls")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()