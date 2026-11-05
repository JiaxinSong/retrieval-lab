# 09 — Judging the Answers

Step 08 checked citations. This step checks what the answers say.

`scripts/09_judge_answers.py` → `eval/answer_eval.json`; judgments cached in
`eval/answer_judgments.jsonl`

```bash
python scripts/09_judge_answers.py --dry-run          # show the planted errors, count calls, spend nothing
python scripts/09_judge_answers.py                    # calibrate, then judge eval/answers.json
python scripts/09_judge_answers.py --answers eval/answers_threshold_1.0.json
```

---

## Why citations are not enough

Step 08 counted an answer as grounded when it cited a ground-truth chunk. That
says the answer rests on the right passage. It does not say the answer reads
that passage correctly:

```
Q: What is the default for work_mem?
A: The default is 8MB [1].        <- [1] is the ground-truth chunk, which says 4MB
```

By step 08's measures this answer is a success. This step would mark it
unfaithful and incorrect.

## Two checks per answer

| Check | The judge sees | Question | Verdicts |
| --- | --- | --- | --- |
| **Faithfulness** | The answer and the passages it cites | Is every claim supported by the passage it cites? | faithful / not |
| **Correctness** | The answer and the ground-truth chunks from step 06 | Does it answer every part of the question without contradicting them? | correct / incomplete / incorrect |

They are separate because they compare against different text, and they fail
independently:

- **Faithful but incomplete**: a multi-hop answer that cites one of the two
  required chunks and says only what that one says.
- **Correct but unfaithful**: the right fact, supplied from the model's own
  knowledge rather than the passage it cites. This is a hallucination even when
  it is true: the citation claims a source that does not say it.

Faithfulness is the hallucination check. Correctness is the "did it answer the
question" check. The 2 unanswerable questions are skipped: step 08 already
scored their refusals.

Each judge writes its own reading before the verdict: the list of unsupported
claims, or the correct answer in its own words. This is the same answer-first
order that fixed calibration in step 06 — the verdict follows from what the
judge actually read.

## Two judges, because one of them wrote the answers

Judge A is `gpt-5-mini`, the same model that wrote the answers in step 08. A
model grading its own output tends to be lenient (self-preference bias). Judge
B is `claude-haiku-4-5`, from another vendor.

As in step 06, the weaker verdict wins: an answer is faithful only if both
judges say so, and its correctness is the lower of the two grades. The two
judges' agreement is reported as Cohen's kappa, and every answer either judge
flags is printed with the reason.

There is no referee this time. Disagreements are few enough to read, and the
weaker verdict is the conservative one.

## Calibration without hand labels

Step 06 calibrated its judges on cases verified by reading the docs. That needs
someone to check each case by hand. Here the calibration cases are made by code
instead: **copies of real answers with known errors planted in them.**

| Planted error | How code makes it | Expected verdict |
| --- | --- | --- |
| **number** | A value the ground-truth chunk states ("4MB") is changed to a value no passage contains ("8MB") | not faithful, and not correct |
| **fabricated** | A sentence is appended with a citation, e.g. "This behaviour was added at the request of the Debian packaging team [1]." It is used only if its keyword ("Debian") appears in none of the cited passages | not faithful |
| **drop_half** | For a multi-hop answer, the sentences citing one of the two required chunks are removed | not correct |

Because code made the error, the expected verdict is certain without anyone
reading the case. Six cases per type are sampled; drop_half has only two,
because only two of the three multi-hop answers have a sentence that cites
just one of the required chunks. That is 20 expected verdicts per judge, and
each judge must get at least 90% of them right, or the run stops before the
main pass.

Some details that keep the expected verdicts certain:

- **Numbers are matched with their unit** (`4MB`, `200 ms`, `16 pages`) or after
  "default is". A bare number could be a section number ("19.4") or a version,
  and changing it might not make the answer wrong.
- **The original value must appear in the ground-truth chunk**, so the changed
  answer contradicts the reference, not some side detail.
- **The new value must appear in none of the passages**, so the judge cannot
  find support for it anywhere.

`--dry-run` prints every planted error before anything is spent, and for
drop_half the full answer that is left, so the planted error can be checked by
eye.

### What calibration caught

**A judge that skimmed.** On the first run, Haiku passed two answers carrying
the planted Debian sentence as "all claims supported". It judged each answer
as a whole and read past the last line. The fix (prompt v2) makes the judge
write one line per sentence, in order — "supported by [n]" or "NOT
SUPPORTED" — before listing unsupported claims, and any NOT SUPPORTED line
makes the answer unfaithful. Haiku then caught 6 of 6. It is the same idea as
step 06's answer-first schema: make the judge show what it read before it
decides.

**A planted error that wasn't one.** Both judges called the hw9 drop_half case
correct, on two runs in a row. Their reasons quoted the 25% shared_buffers
guidance — the fact that had supposedly been removed. The rule removed only
the sentences citing the dropped chunk, and the fact survived in a sentence
that cited something else. The judges were right and the test was wrong. The
rule now keeps only sentences whose citations are exactly the remaining
chunk; the hw9 case is reduced to "Default is 4GB [2]." and both judges flag
it. The 90% bar was not lowered for either problem.

| Run | Judge A (gpt-5-mini) | Judge B (claude-haiku-4-5) |
| --- | --- | --- |
| v1 prompt | 19 / 20 (missed: hw9 drop_half) | 17 / 20 (missed: 2 fabricated, hw9 drop_half) |
| v2 prompt | 19 / 20 (missed: hw9 drop_half) | 19 / 20 (missed: hw9 drop_half) |
| v2 prompt, drop_half fixed | **20 / 20** | **20 / 20** |

### What this proves, and what it doesn't

It proves the judges **catch clear errors**: a wrong value, an invented claim,
a missing half.

It does not prove the judges **never flag a correct answer as wrong**. That
would need answers known to be correct, which means reading them by hand. The
only check on that side is the agreement between two judges from different
vendors, and the printed list of flagged answers.

The planted errors are also blunter than real ones. A real error is more often
a subtle misreading than an invented sentence about Debian. Catching planted
errors is a floor, not a guarantee.

---

## Results

The 82 answerable questions from the default-gate run of step 08
(`eval/answers.json`).

| | Both judges | One judge only | Neither |
| --- | --- | --- | --- |
| Flagged **unfaithful** | **0** | 9 (A 3, B 6) | 73 |
| Flagged **not correct** | **4** | 9 | 69 |

With the weaker verdict winning: 73 of 82 faithful; 69 correct, 8
incomplete, 5 incorrect. All three multi-hop answers were correct by both
judges.

### No answer is a hallucination both judges agree on

The faithfulness check found no answer that both judges call unsupported. The
nine single-judge flags are mostly the model adding a small piece of general
knowledge the passage doesn't state: expanding "WAL" to "Write-Ahead Log",
saying vacuum "including autovacuum" maintains the visibility map. Strictly
unsupported, as the prompt defines it; not invented facts.

Together with the calibration result — both judges catch every planted
fabricated sentence — the fair claim is: **no clear hallucination in 82
answers.** It is not "zero unsupported words".

### The four answers both judges flagged, read against the chunks

`--show` prints each answer with its ground-truth chunk text and both judges'
reasons:

| qid | Verdict on reading | What happened |
| --- | --- | --- |
| hw5 | **Real: incomplete** | Retrieval missed every ground-truth chunk. The answer says "run ANALYZE" from the one passage it had, and omits `default_statistics_target`, which the question asks for |
| g28 | **Real: incomplete** | Retrieval missed the ground-truth chunk (memory units B, kB, MB, GB, TB). The answer pieced units together from parameter-specific chunks, missing B and TB |
| g58 | **Real: wrong** | The first bullet, under "a single index is not a good choice when", describes the opposite case: many partial indexes, where one index is the better choice. The answer also omits the main point of its references (a multicolumn index should not be the only index) |
| g25 | **Judge error** | The answer says `logging_collector` is needed for csvlog/jsonlog output. That is stated in a passage it cites (chunk 979), and both faithfulness judges confirmed it. But 979 isn't a ground-truth chunk, so both correctness judges marked the answer down for it — despite the prompt telling them not to |

**Three real problems in 82 answers, and two of them are retrieval misses.**
hw5 and g28 are the only two answerable questions whose retrieval missed the
ground truth (80 of 82 found it, step 07/08). Both answers came out
incomplete, and neither said what was missing, although the prompt allows a
partial answer that says so. The model answered confidently from whatever
passages it had. The one error that isn't retrieval's fault, g58, is a
reasoning error on a vague question.

### Correctness: how many are actually right

69 answers are correct by both judges; 3 are confirmed wrong or incomplete.
The other 10 (9 single-judge flags and g25) are not confirmed either way. Several of the
single-judge reasons show the same mistake as g25 — Haiku marking an answer
"incomplete" because it "adds information not in the references" (g07, g66).
So the honest range is **69 to 79 correct of 82**, with 3 confirmed problems.

### Agreement is low, and why

| Check | Raw agreement | Cohen's kappa |
| --- | --- | --- |
| Faithfulness | 73 / 82 | −0.05 |
| Correctness | 70 / 82 | 0.25 |

Two reasons, one statistical and one real:

- **Rare flags make kappa collapse.** Each judge calls 90%+ of answers fine,
  so chance agreement is already high, and kappa only rewards agreement on
  the rare flags. On faithfulness, the two judges' flags don't overlap at all,
  which drives kappa to zero even at 89% raw agreement. Kappa is reported,
  but with the raw agreement and the overlap beside it.
- **The judges flag different answers.** That part is real: on borderline
  cases (an acronym expansion, an extra detail), the two models draw the line
  in different places. Only the cases both flag are solid signal; the
  single-judge flags are candidates for reading, not findings.

### A design flaw this exposed

The correctness judge sees only the ground-truth chunks. A true fact the
answer took from another retrieved passage looks to it like "something the
references don't say", and the instruction not to mark that down was ignored
at least once by each judge (g25). Giving the correctness judge the cited passages too,
or grading only the parts of the answer that address the question, would
remove that failure mode. Not done here.

### What these results do not show

- **That the judges never flag correct answers.** Calibration only planted
  errors; nothing checks the other direction except the two judges against
  each other.
- **Anything about subtle errors.** The planted errors are blunt. g58 — a real
  error of reasoning, not a wrong number — was caught by both judges, which is
  encouraging, but it is one case.
- **Absolute quality.** 82 questions, one corpus, judges from two vendors.
  The numbers rank failure modes (retrieval misses come first); they are not a
  quality score to quote alone.
