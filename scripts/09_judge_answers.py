"""
Judge the answers from step 08: are they faithful to the passages they cite,
and are they correct?

    python3 scripts/09_judge_answers.py --dry-run          # build calibration cases, count calls, spend nothing
    python3 scripts/09_judge_answers.py                    # calibrate, then judge eval/answers.json
    python3 scripts/09_judge_answers.py --answers eval/answers_threshold_1.0.json

WHAT STEP 08 COULD NOT MEASURE

Step 08 scored citations: did the answer cite a ground-truth chunk? An answer
can cite the right chunk and still say the wrong thing -- "the default is 8MB"
next to a passage that says 4MB. This step reads the answer text.

TWO CHECKS PER ANSWER

    faithfulness  The answer and the passages it cites. Is every claim
                  supported by the passage it cites? This is the
                  hallucination check: a claim the passages do not make is
                  unsupported even if it happens to be true.
    correctness   The answer and the ground-truth chunks from step 06. Does it
                  answer every part of the question without contradicting
                  them? correct / incomplete / incorrect.

They are separate calls because they compare against different text. An
answer can be faithful but incomplete (it cited one of two needed passages),
or correct but unfaithful (right fact, from the model's own knowledge).

Each judge writes its own reading first -- the unsupported claims, or the
reference answer -- and only then the verdict, the same answer-first order as
step 06.

JUDGES

Judge A is gpt-5-mini, the model that wrote the answers. A model grading its
own output tends to be lenient, which is the reason for Judge B from another
vendor (claude-haiku-4-5). The weaker verdict wins: an answer is faithful only
if both judges say so, and its correctness is the lower of the two grades.
Disagreements are listed, not refereed.

CALIBRATION WITHOUT HAND LABELS

Nobody has hand-checked which answers are right, so there are no known-correct
cases to calibrate on. Instead, code plants known errors in copies of real
answers, and both judges must catch them before the run:

    number      a value from the docs ("4MB") is changed ("8MB")
                -> expected: not faithful, and not correct
    fabricated  a sentence the passages do not contain is appended, with a
                citation -> expected: not faithful
    drop_half   for a multi-hop answer, the sentences citing one of the two
                required chunks are removed -> expected: not correct

The changes are made by code, not by a model, so the expected verdicts are
certain without anyone reading them. Each judge must catch at least
CAL_MIN_CATCH of the planted errors, or the run stops.

What this proves and what it does not: it shows the judges catch clear errors.
It cannot show they never flag a correct answer as wrong -- that would need
known-correct answers. Agreement between the two judges (Cohen's kappa) is
the only check on that side, and the list of disagreements is printed for
reading.

Every judgment is cached in eval/answer_judgments.jsonl, keyed by prompt
version, model, check, item, and a hash of the exact text the judge saw.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anthropic import AsyncAnthropic  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from retrieval import TABLE, connect  # noqa: E402  (also loads .env)

EVAL_DIR = ROOT / "eval"
GOLDEN_PATH = EVAL_DIR / "golden.json"
ANSWERS_PATH = EVAL_DIR / "answers.json"
CACHE_PATH = EVAL_DIR / "answer_judgments.jsonl"

JUDGE_A = ("openai", "gpt-5-mini")                    # also wrote the answers
JUDGE_B = ("anthropic", "claude-haiku-4-5-20251001")
JUDGES = {"A": JUDGE_A, "B": JUDGE_B}

# v1 -> v2: the faithfulness judge now checks the answer sentence by sentence.
# Under v1, Haiku passed two answers with a planted sentence ("added at the
# request of the Debian packaging team") as fully supported -- it judged the
# answer as a whole and skimmed past the last line.
PROMPT_VERSION = "v2"

CAL_PER_TYPE = 6          # planted errors per type (drop_half has at most 3 candidates)
CAL_MIN_CATCH = 0.9       # share of planted errors each judge must catch
SEED = 7

OPENAI_CONCURRENCY = 8
ANTHROPIC_CONCURRENCY = 4
ANTHROPIC_RPM = 45
SDK_MAX_RETRIES = 8

FATAL_MARKERS = ("insufficient_quota", "credit balance", "credit_balance",
                 "billing", "invalid x-api-key", "incorrect api key",
                 "authentication_error", "not_found_error", "model_not_found")

openai_client = AsyncOpenAI(max_retries=SDK_MAX_RETRIES)
anthropic_client = AsyncAnthropic(max_retries=SDK_MAX_RETRIES)


# ---------------------------------------------------------------- prompts

FAITH_SYSTEM = """You check whether an answer is supported by the documentation passages it cites.

The answer was written from numbered passages of the PostgreSQL 17 manual and cites them as [n]. You are given the passages it cites, with the same numbers.

First go through the answer sentence by sentence, in order, and write one line per sentence in checks: the start of the sentence, then "supported by [n]" or "NOT SUPPORTED: <why>". Do not skip any sentence, including the last one.

Then list every claim in the answer that the passages do not support. A claim is unsupported if the passage it cites does not state it or contradicts it, or if it cites nothing and no passage given states it. Use only the passages. Do not use outside knowledge, even if you believe a claim is true.

Rewording, summarising, and simple inferences from what a passage states count as supported. A statement that the documentation does not cover something is not a claim to check.

Then set faithful to true only if the list is empty. Give a one-sentence reason."""

CORRECT_SYSTEM = """You grade an answer to a question about PostgreSQL against reference passages from the PostgreSQL 17 manual that are known to contain the correct answer.

First write the correct answer in one or two sentences, using only the references. Then compare the answer to it and choose a verdict:
- "correct": the answer covers every part of the question, and nothing it says contradicts the references.
- "incomplete": nothing it says contradicts the references, but it leaves part of the question unanswered.
- "incorrect": it states something that contradicts the references (a wrong value, a wrong yes or no, the wrong setting), or it does not answer the question.

Judge only against the references. Do not mark the answer down for extra details the references do not mention; those are checked separately. Ignore the [n] citation markers.

Give a one-sentence reason."""

MATCH_NOTE = {
    "any": "Each reference passage on its own contains the full answer; they may repeat each other.",
    "all": "The references together contain the full answer; each holds a different part of it.",
}


class FaithJudgment(BaseModel):
    # Order matters: every sentence is checked, then the unsupported claims
    # are listed, then the verdict follows from the list.
    checks: list[str]
    unsupported_claims: list[str]
    faithful: bool
    reason: str


class CorrectJudgment(BaseModel):
    reference_answer: str
    verdict: Literal["correct", "incomplete", "incorrect"]
    reason: str


RANK = {"incorrect": 0, "incomplete": 1, "correct": 2}


# ---------------------------------------------------------------- judge calls

def openai_sampling(model: str) -> dict:
    """Reasoning models reject temperature; same rule as steps 06 and 08."""
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        return {"reasoning_effort": "low"}
    return {"temperature": 0}


def fingerprint(*texts: str) -> str:
    return hashlib.sha256("\x1f".join(texts).encode("utf-8")).hexdigest()[:16]


class Judging:
    """Calls, pacing, and the cache, in one place."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self.records: dict[str, dict] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.records[rec["key"]] = rec
        self.lock = asyncio.Lock()
        self.sems = {"openai": asyncio.Semaphore(OPENAI_CONCURRENCY),
                     "anthropic": asyncio.Semaphore(ANTHROPIC_CONCURRENCY)}
        self.anthropic_interval = 60.0 / ANTHROPIC_RPM
        self.anthropic_next = 0.0
        self.pace_lock = asyncio.Lock()
        self.abort = False
        self.failures = 0
        self.new_calls = 0

    async def pace(self, vendor: str) -> None:
        if vendor != "anthropic":
            return
        async with self.pace_lock:
            now = time.monotonic()
            if self.anthropic_next > now:
                await asyncio.sleep(self.anthropic_next - now)
            self.anthropic_next = max(now, self.anthropic_next) + self.anthropic_interval

    async def call(self, judge: tuple[str, str], system: str, user: str,
                   schema: type[BaseModel]) -> BaseModel | None:
        vendor, model = judge
        async with self.sems[vendor]:
            await self.pace(vendor)
            if self.abort:
                return None
            try:
                if vendor == "openai":
                    resp = await openai_client.chat.completions.parse(
                        model=model,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": user}],
                        response_format=schema, **openai_sampling(model))
                    parsed = resp.choices[0].message.parsed
                    if parsed is None:
                        raise ValueError("no parsed output")
                    return parsed
                resp = await anthropic_client.messages.create(
                    model=model, max_tokens=1200, system=system,
                    tools=[{"name": "record", "description": "Record the judgment.",
                            "input_schema": schema.model_json_schema()}],
                    tool_choice={"type": "tool", "name": "record"},
                    messages=[{"role": "user", "content": user}])
                block = next(b for b in resp.content if b.type == "tool_use")
                return schema.model_validate(block.input)
            except Exception as e:  # noqa: BLE001 -- recorded, retried on the next run
                text = f"{type(e).__name__}: {e}".lower()
                if any(m in text for m in FATAL_MARKERS):
                    if not self.abort:
                        print(f"\nFATAL ({model}): {e}\nStopping; nothing further will be sent.")
                    self.abort = True
                self.failures += 1
                if self.failures <= 3:
                    print(f"  {model} call failed: {type(e).__name__}: {str(e)[:160]}")
                return None

    @staticmethod
    def key(judge_name: str, check: str, item: dict) -> str:
        texts = [p["content"] for p in item["cited"]] if check == "faith" else item["refs"]
        return (f"{PROMPT_VERSION}|{JUDGES[judge_name][1]}|{check}|{item['id']}|"
                f"{fingerprint(item['question'], item['answer'], *texts)}")

    async def judge(self, judge_name: str, check: str, item: dict) -> dict | None:
        """One check on one item by one judge. Cached by the exact text seen."""
        judge = JUDGES[judge_name]
        if check == "faith":
            system, schema = FAITH_SYSTEM, FaithJudgment
            user = (f"Question: {item['question']}\n\nAnswer:\n{item['answer']}\n\n"
                    f"Passages it cites:\n{format_cited(item['cited'])}")
        else:
            system, schema = CORRECT_SYSTEM, CorrectJudgment
            user = (f"Question: {item['question']}\n\n{MATCH_NOTE[item['match']]}\n\n"
                    f"References:\n{format_refs(item['refs'])}\n\nAnswer to grade:\n{item['answer']}")
        key = self.key(judge_name, check, item)
        hit = self.records.get(key)
        if hit:
            return hit["result"]
        result = await self.call(judge, system, user, schema)
        if result is None:
            return None
        out = result.model_dump()
        if check == "faith":
            # The verdict must agree with the checks; if not, take the stricter.
            flagged = any("NOT SUPPORTED" in c.upper() for c in out["checks"])
            out["faithful"] = out["faithful"] and not out["unsupported_claims"] and not flagged
        self.new_calls += 1
        async with self.lock:
            self.records[key] = {"key": key, "result": out}
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "result": out}) + "\n")
        return out


def format_cited(cited: list[dict]) -> str:
    return "\n\n".join(f"[{p['n']}]\n{p['content']}" for p in cited)


def format_refs(refs: list[str]) -> str:
    return "\n\n".join(f"[R{i}]\n{text}" for i, text in enumerate(refs, 1))


# ---------------------------------------------------------------- items

async def build_items(answers: dict, golden: list[dict], db) -> list[dict]:
    """One item per answerable question: the answer, the passages it cites
    (with the numbers the model saw), and the ground-truth chunk texts."""
    needed = set()
    for q in golden:
        r = answers.get(q["qid"])
        if q["match"] == "none" or r is None:
            continue
        needed |= set(q["ground_truth"])
        needed |= {s["id"] for s in r["sources"].values()}
    rows = await db.fetch(f"SELECT id, content FROM {TABLE} WHERE id = ANY($1::int[])", list(needed))
    content = {r["id"]: r["content"] for r in rows}

    items = []
    for q in golden:
        r = answers.get(q["qid"])
        if q["match"] == "none" or r is None:
            continue
        sources = {int(n): s for n, s in r["sources"].items()}
        numbers = sorted({n for n in r["citations"] if n in sources}) or sorted(sources)
        items.append({
            "id": q["qid"], "qid": q["qid"], "question": q["question"], "match": q["match"],
            "answer": r["answer"], "answerable_pred": r["answerable"],
            "sources": sources,
            "cited": [{"n": n, "id": sources[n]["id"], "content": content[sources[n]["id"]]}
                      for n in numbers],
            "gt_ids": list(q["ground_truth"]),
            "refs": [content[c] for c in q["ground_truth"]],
        })
    return items


# ---------------------------------------------------------------- planted errors

UNIT = r"(?:kB|MB|GB|TB|ms|s|min|h|d|bytes?|seconds?|minutes?|hours?|milliseconds?|pages?|%)"
NUMBER_WITH_UNIT = re.compile(rf"(?<![\w.\[])(\d+(?:\.\d+)?)\s?{UNIT}(?![A-Za-z])")
DEFAULT_NUMBER = re.compile(r"default(?: value)?(?: is| of|:|=)?\s+(\d+(?:\.\d+)?)(?![\d.]*\])", re.I)
CITATION = re.compile(r"\[(\d+)\]")

# Each claim is checked against the passages by its keyword: the sentence is
# used only if the keyword appears in none of them, so it cannot be supported.
FABRICATED = [
    ("Debian", "This behaviour was added at the request of the Debian packaging team [{n}]."),
    ("Windows", "On Windows the setting is ignored and a fixed value is used instead [{n}]."),
    ("Oracle", "The name and default were chosen to match Oracle's equivalent parameter [{n}]."),
]


def standalone(value: str, text: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(value)}(?![\d])", text) is not None


def changed_value(value: str, reference: str) -> str | None:
    """A different value that the reference text does not contain anywhere."""
    num = float(value)
    decimals = len(value.split(".")[1]) if "." in value else 0
    for new in (num * 2, num * 10, num + 3):
        text = f"{new:.{decimals}f}"      # keep the original format: 4.0 -> 8.0, not 8
        if text != value and not standalone(text, reference):
            return text
    return None


def plant_number(item: dict) -> dict | None:
    """Change a value the ground-truth chunks state, so the changed answer
    contradicts the references, not just some side detail."""
    ground_truth = "\n".join(item["refs"])
    everything = "\n".join(item["refs"] + [p["content"] for p in item["cited"]])
    spans = [(m.start(1), m.end(1)) for m in NUMBER_WITH_UNIT.finditer(item["answer"])]
    spans += [(m.start(1), m.end(1)) for m in DEFAULT_NUMBER.finditer(item["answer"])]
    for start, end in sorted(set(spans)):
        value = item["answer"][start:end]
        if not standalone(value, ground_truth):
            continue
        new = changed_value(value, everything)
        if new is None:
            continue
        answer = item["answer"][:start] + new + item["answer"][end:]
        return {**item, "id": f"{item['qid']}~number", "answer": answer,
                "planted": f"{value} -> {new}", "expect": {"faith": False, "correct": False}}
    return None


def plant_fabricated(item: dict) -> dict | None:
    if not item["cited"]:
        return None
    texts = "\n".join(p["content"] for p in item["cited"]).lower()
    for keyword, sentence in FABRICATED:
        if keyword.lower() not in texts:
            added = sentence.format(n=item["cited"][0]["n"])
            return {**item, "id": f"{item['qid']}~fabricated",
                    "answer": item["answer"].rstrip() + " " + added,
                    "planted": f"+ {added}", "expect": {"faith": False}}
    return None


def plant_drop_half(item: dict) -> dict | None:
    """Multi-hop answers only: keep just the sentences whose citations are
    exactly one of the two required chunks; remove everything else.

    Earlier versions removed only the sentences citing the other required
    chunk. On hw9 both judges still called the result correct, with reasons
    quoting the 25% shared_buffers guidance -- the fact survived in a sentence
    the rule kept (one citing another passage, or none). The planted error
    was not an error. Keeping only sentences sourced from the one remaining
    chunk leaves nothing that can carry the other half.
    """
    if item["match"] != "all" or len(item["gt_ids"]) != 2:
        return None
    sentences = re.split(r"(?<=[.!?])\s+", item["answer"].strip())
    cited_ids = [{item["sources"][int(n)]["id"] for n in CITATION.findall(s)
                  if int(n) in item["sources"]} for s in sentences]
    for keep, drop in (item["gt_ids"], item["gt_ids"][::-1]):
        kept = [s for s, ids in zip(sentences, cited_ids) if ids == {keep}]
        if kept and len(kept) < len(sentences):
            removed = [s for s in sentences if s not in kept]
            return {**item, "id": f"{item['qid']}~drop_half", "answer": " ".join(kept),
                    "planted": f"removed: {' '.join(removed)[:120]}", "expect": {"correct": False}}
    return None


def calibration_cases(items: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    cases = []
    for plant in (plant_number, plant_fabricated, plant_drop_half):
        made = [c for c in (plant(i) for i in items) if c is not None]
        made.sort(key=lambda c: c["id"])
        cases += rng.sample(made, min(CAL_PER_TYPE, len(made)))
    return cases


# ---------------------------------------------------------------- running

def is_pass(check: str, result: dict) -> bool:
    return result["faithful"] if check == "faith" else result["verdict"] == "correct"


async def calibrate(judging: Judging, cases: list[dict]) -> bool:
    jobs = [(c, j, check) for c in cases for j in JUDGES for check in c["expect"]]
    results = await asyncio.gather(*(judging.judge(j, check, c) for c, j, check in jobs))
    if any(r is None for r in results):
        print("\nSome calibration calls failed; rerun to retry them (finished ones are cached).")
        return False

    caught, total = Counter(), Counter()
    misses = []
    for (c, j, check), r in zip(jobs, results):
        total[j] += 1
        if is_pass(check, r) == c["expect"][check]:
            caught[j] += 1
        else:
            misses.append((c, j, check, r))

    print("\nCALIBRATION (planted errors)")
    for c in cases:
        print(f"  {c['id']:18} {c['planted'][:90]}")
    ok = True
    for j in JUDGES:
        rate = caught[j] / total[j] if total[j] else 0.0
        print(f"  Judge {j} ({JUDGES[j][1]}): caught {caught[j]}/{total[j]} ({rate:.0%})")
        ok = ok and rate >= CAL_MIN_CATCH
    for c, j, check, r in misses:
        print(f"  MISSED  {c['id']:18} judge {j} {check}: {r['reason'][:150]}")
    if not ok:
        print(f"\nCalibration FAILED (each judge must catch {CAL_MIN_CATCH:.0%}). Stopping before the main run.")
    return ok


def cohen_kappa(pairs: list[tuple]) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


async def judge_all(judging: Judging, items: list[dict], step08_rows: dict) -> dict | None:
    jobs = [(i, j, check) for i in items for j in JUDGES for check in ("faith", "correct")]
    results = await asyncio.gather(*(judging.judge(j, check, i) for i, j, check in jobs))
    if any(r is None for r in results):
        print("\nSome calls failed; rerun to retry them (finished ones are cached).")
        return None
    got = {(i["id"], j, check): r for (i, j, check), r in zip(jobs, results)}

    rows = []
    for i in items:
        fa, fb = got[(i["id"], "A", "faith")], got[(i["id"], "B", "faith")]
        ca, cb = got[(i["id"], "A", "correct")], got[(i["id"], "B", "correct")]
        rows.append({
            "qid": i["qid"], "match": i["match"],
            "cites_ground_truth": step08_rows.get(i["qid"], {}).get("cites_ground_truth"),
            "faith_a": fa["faithful"], "faith_b": fb["faithful"],
            "faithful": fa["faithful"] and fb["faithful"],
            "correct_a": ca["verdict"], "correct_b": cb["verdict"],
            "correctness": min(ca["verdict"], cb["verdict"], key=RANK.get),
            "unsupported_a": fa["unsupported_claims"], "unsupported_b": fb["unsupported_claims"],
            "reason_correct_a": ca["reason"], "reason_correct_b": cb["reason"],
        })

    n = len(rows)
    summary = {
        "answers": n,
        "faithful": sum(r["faithful"] for r in rows),
        "faithful_a": sum(r["faith_a"] for r in rows),
        "faithful_b": sum(r["faith_b"] for r in rows),
        "correctness": dict(Counter(r["correctness"] for r in rows)),
        "correctness_a": dict(Counter(r["correct_a"] for r in rows)),
        "correctness_b": dict(Counter(r["correct_b"] for r in rows)),
        "kappa_faith": cohen_kappa([(r["faith_a"], r["faith_b"]) for r in rows]),
        "kappa_correct": cohen_kappa([(r["correct_a"], r["correct_b"]) for r in rows]),
        "agree_faith": sum(r["faith_a"] == r["faith_b"] for r in rows),
        "agree_correct": sum(r["correct_a"] == r["correct_b"] for r in rows),
        "multi_hop_correct": sum(r["correctness"] == "correct" for r in rows if r["match"] == "all"),
        "multi_hop": sum(r["match"] == "all" for r in rows),
        "correct_when_cites_gt": sum(r["correctness"] == "correct" for r in rows if r["cites_ground_truth"]),
        "cites_gt": sum(1 for r in rows if r["cites_ground_truth"]),
    }

    s = summary
    print("\n" + "=" * 70)
    print(f"ANSWER JUDGING  ({n} answerable questions; the weaker verdict wins)\n")
    print(f"  faithful:     {s['faithful']}/{n}   (A {s['faithful_a']}, B {s['faithful_b']}; "
          f"agree {s['agree_faith']}/{n}, kappa {s['kappa_faith']:.2f})")
    print(f"  correctness:  {s['correctness']}")
    print(f"                A {s['correctness_a']}")
    print(f"                B {s['correctness_b']}   (agree {s['agree_correct']}/{n}, "
          f"kappa {s['kappa_correct']:.2f})")
    print(f"  multi-hop correct: {s['multi_hop_correct']}/{s['multi_hop']}")
    print(f"  correct among answers citing ground truth: {s['correct_when_cites_gt']}/{s['cites_gt']}")

    flagged = [r for r in rows if not r["faithful"] or r["correctness"] != "correct"]
    if flagged:
        print(f"\nFLAGGED ({len(flagged)}): not faithful by either judge, or not correct by either judge")
        for r in flagged:
            print(f"  {r['qid']:5} faith A={r['faith_a']!s:5} B={r['faith_b']!s:5}  "
                  f"correct A={r['correct_a']:10} B={r['correct_b']}")
            for claim in (r["unsupported_a"] + r["unsupported_b"])[:2]:
                print(f"        unsupported: {claim[:120]}")
            if r["correctness"] != "correct":
                reason = r["reason_correct_a"] if r["correct_a"] != "correct" else r["reason_correct_b"]
                print(f"        correctness: {reason[:140]}")
    return {"summary": summary, "rows": rows}


def show(judging: Judging, items: list[dict], qids: list[str]) -> None:
    """Print answers with their references and the cached judgments, for
    reading the flagged cases. Sends nothing."""
    by_id = {i["id"]: i for i in items}
    for qid in qids:
        item = by_id.get(qid)
        if item is None:
            print(f"\n{qid}: not an answerable golden question")
            continue
        print("\n" + "=" * 70)
        print(f"{qid}  {item['question']}\n\nANSWER:\n{item['answer']}\n")
        print(f"cited chunks: {[p['id'] for p in item['cited']]}   ground truth: {item['gt_ids']}")
        for cid, text in zip(item["gt_ids"], item["refs"]):
            print(f"\nREFERENCE {cid}:\n{text}")
        for j in JUDGES:
            for check in ("faith", "correct"):
                rec = judging.records.get(Judging.key(j, check, item))
                if rec is None:
                    print(f"\n[{j} {check}] not judged yet")
                    continue
                r = rec["result"]
                verdict = r["faithful"] if check == "faith" else r["verdict"]
                print(f"\n[{j} {check}] {verdict}: {r['reason']}")
                for claim in r.get("unsupported_claims", []):
                    print(f"    unsupported: {claim}")
                if check == "correct":
                    print(f"    reference answer: {r['reference_answer']}")


async def main_async(args) -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    answers_file = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    answers = answers_file["answers"]
    step08_rows = {r["qid"]: r for r in answers_file["rows"]}

    db = await connect()
    try:
        items = await build_items(answers, golden, db)
    finally:
        await db.close()
    cases = calibration_cases(items)
    judging = Judging(CACHE_PATH)

    by_type = Counter(c["id"].split("~")[1] for c in cases)
    calls = sum(len(c["expect"]) for c in cases) * len(JUDGES) + len(items) * 2 * len(JUDGES)
    print(f"{len(items)} answers to judge; {len(cases)} calibration cases {dict(by_type)}")
    print(f"up to {calls} judge calls (cached ones are free)")

    if args.show:
        show(judging, items, args.show)
        return

    if args.dry_run:
        print("\nPLANTED ERRORS (dry run, nothing sent)")
        for c in cases:
            print(f"  {c['id']:18} {c['planted'][:110]}")
            if c["id"].endswith("~drop_half"):
                print(f"  {'':18} left: {c['answer']}")
        return

    if not await calibrate(judging, cases):
        return
    if args.calibrate_only:
        return
    result = await judge_all(judging, items, step08_rows)
    if result is None:
        return
    out = EVAL_DIR / Path(args.answers).name.replace("answers", "answer_eval", 1)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nnew judge calls this run: {judging.new_calls}\n-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--answers", default=str(ANSWERS_PATH), help="step 08 output to judge")
    ap.add_argument("--dry-run", action="store_true", help="build calibration cases and count calls only")
    ap.add_argument("--calibrate-only", action="store_true", help="stop after calibration")
    ap.add_argument("--show", nargs="+", metavar="QID",
                    help="print these answers with references and cached judgments; sends nothing")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()