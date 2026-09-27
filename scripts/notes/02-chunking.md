# 02 — Chunking

Converting 224 HTML pages into 1,293 retrieval chunks.

`scripts/02_chunk_docs.py` → `data/processed/chunks.json`

---

## Results

| Metric | Value |
| --- | --- |
| Input | 224 HTML pages, 21 chapters (5 eval, 16 distractor) |
| Output | **1,293 chunks** — 339 eval, 954 distractor |
| Tokens per chunk | min 35, max 522, mean 401 |
| Total tokens | 519,501 |
| Embedding cost | $0.0104 at $0.02 / 1M tokens |

Target was 512 tokens; the mean landing at 401 means most sections fit inside
one chunk without needing a split. That is the intended outcome — a chunk that
matches a document section is more coherent than one ending at an arbitrary
token offset.

The mean held between 401 and 412 across every corpus size (339 → 676 → 1,293
chunks), which is a useful signal: the splitting strategy is stable across
chapters with quite different structure, not tuned to the first five.

---

## Strategy: split on document structure, not token offsets

The corpus is DocBook-generated HTML, which marks every subsection with
`div.sect1` / `div.sect2`. Each of those is one coherent unit: one parameter,
one index type, one concept.

Cutting there produces chunks that are semantically whole. Cutting at token 512
produces chunks that are half a thought, and embeddings represent an incomplete
idea poorly — the vector sits between two topics and matches neither query
well.

Section granularity is selected with a fallback chain:

```python
sections = soup.select("div.sect2") or soup.select("div.sect1") or soup.select("div.refsect1")
```

`sect2` is preferred because it is usually one parameter (`work_mem`,
`shared_buffers`). Pages that only use `sect1` still get handled rather than
skipped.

---

## Heading chains: contextual retrieval for free

Every chunk is prefixed with its position in the document:

```
# 19.4. Resource Consumption > 19.4.1. Memory

work_mem (integer) Sets the base maximum amount of memory to be used by a
query operation before writing to temporary disk files...
```

A chunk lifted out of its document loses the topic it belongs to. Text about
`work_mem` with no heading is just a paragraph about memory; with the heading
chain it is *configuration* text about *resource consumption*. The prefix puts
that back before the text is embedded, so the vector encodes the topic and not
only the sentence.

This is the zero-cost version of Anthropic's Contextual Retrieval. The paid
version generates the context with an LLM call per chunk; here the document's
own hierarchy supplies it. The LLM version is on the Tier 2 list — the
comparison between the two is a measurement to make, not an assumption.

Interview framing: this is an answer to "how did you handle chunk context?"
that is neither "I used fixed-size chunks" nor "I copied Anthropic's blog
post."

The chain is counted against the chunk's token budget like any other text —
roughly 15-30 tokens, about 5% of a 512-token window. For a real example:

```
# 13.4. Data Consistency Checks at the Application Level > 13.4.1. Enforcing Consistency with Serializable Transactions
```

28 tokens, leaving 484 for the body. A worthwhile trade: 5% of the window buys
a chunk that knows which topic it belongs to.

---

## Section intros — a bug found by auditing

The original implementation selected only `div.sect2` when present, which
silently dropped prose sitting in a `sect1` before its first subsection.

An audit across all pages found five sections with substantial intro text:

| Page | Tokens | Content |
| --- | --- | --- |
| `using-explain.html` | 234 | Why choosing the right query plan matters |
| `runtime-config-replication.html` | 204 | What the replication settings control |
| `dynamic-trace.html` | 167 | What dynamic tracing provides |
| `routine-vacuuming.html` | 165 | What vacuuming is, and the autovacuum daemon's role |
| `monitoring-stats.html` | 139 | What the cumulative statistics system collects |

These are the **conceptual overviews** — the text that answers "why" questions.
The per-parameter subsections answer "what does this setting do"; the intro
answers "why would I care." Losing them would have quietly capped the ceiling
on an entire class of query.

The fix adds a pass over `sect1` nodes before the main loop, extracting the
text that remains once subsections are removed:

```python
# Copy before decomposing: mutating s1 in place would destroy the
# sect2 nodes the main loop is about to read.
s1_copy = BeautifulSoup(str(s1), "lxml")
for s2 in s1_copy.select("div.sect2"):
    s2.decompose()
```

**That copy is load-bearing.** Calling `decompose()` on the live tree would
delete the `sect2` nodes the main loop then iterates, and every chunk on the
page would vanish.

`MIN_INTRO_TOKENS = 100` filters the noise case. A page like
`runtime-config-resource.html` has a 64-token "intro" that is just the heading
plus an in-page table of contents. The five real intros start at 139 tokens, so
100 is a clean boundary.

Worth noting how this was found: not by reading the code, but by measuring what
the code discarded. A chunking bug produces no error — just a lower ceiling on
retrieval that no amount of reranking recovers.

---

## Text normalization

Raw extraction from DocBook HTML is dirty in four specific ways. All four were
found by reading sample chunks, and all four were fixed before any embedding
spend — the argument for the JSON-before-database ordering below.

### Permalink markers in headings

The docs render a `#` anchor beside every heading, and `get_text()` pulls it
in:

```
"28.2. Data Checksums #"
```

Left in, it appears in `page_title`, in `heading`, and twice inside the heading
chain — a meaningless high-frequency token added to the full-text index on
every chunk.

### The heading duplicated inside the body

The chain prefix stated the heading, then the body opened with the same heading
again, because the `h3` lives inside the `sect2` div that `clean_text`
extracts:

```
# 13.4. Data Consistency ... > 13.4.1. Enforcing Consistency with Serializable Transactions

13.4.1. Enforcing Consistency with Serializable Transactions

If the Serializable transaction isolation level is used...
```

Two costs. The obvious one is 10-15 wasted tokens per chunk. The one that
matters is that repetition **doubles those terms' frequency in the full-text
index** — a section whose title matches the query gets a lexical boost it did
not earn, which is a ranking bug, not just waste.

`clean_text` now decomposes headings out of the body by default.

### Non-breaking spaces and source indentation

DocBook emits U+00A0 between a section number and its title, and in cross
references:

```
"13.4.1.\u00a0Enforcing Consistency"
"See Section\u00a013.2.3 for performance suggestions."
```

It also carries the source file's indentation into text nodes, so raw output is
littered with newlines and tabs. The original regex was `[ \t]+`, which matches
neither newlines nor U+00A0, so both passed straight through into the embedded
text and the tokenizer.

A single `normalize()` now handles all of it, and `page_title`, `heading`, and
body text all route through it so they cannot drift apart:

```python
text = text.replace("\u00a0", " ").replace("\u200b", "")
text = re.sub(r"\s+", " ", text)
text = re.sub(r"\s*#\s*$", "", text)
```

Paragraph structure is not preserved, and does not need to be:
`get_text(separator=" ")` already discarded it, and neither embeddings nor
full-text search read paragraph breaks. The only newline surviving in a finished chunk is the one
deliberately inserted between the heading chain and the body.

### What this cost

| | Before | After |
| --- | --- | --- |
| Chunks | 1,346 | 1,293 |
| Total tokens | 550,753 | 519,501 |
| Mean tokens | 409 | 401 |

53 chunks disappeared. Those were sections consisting of a heading plus one
short sentence; removing the duplicated heading dropped them below
`MIN_TOKENS`. That is the correct outcome — a chunk that is mostly its own
title will never be the right answer, and only pollutes the candidate pool.

6% of total tokens were pure duplication.

Curly quotes (`\u201c`, `\u201d`) are left alone. They are ordinary
punctuation, the tokenizer handles them, and normalizing them would change the
source text for no retrieval benefit.

---

## Oversized sections

Some sections exceed 512 tokens on their own — `monitoring-stats.html` has
several. Those fall back to token windows with 64-token overlap.

Two details in `split_oversized`:

**The heading prefix is repeated on every piece**, and its token cost is
subtracted from the window budget, plus one token for the `"\n\n"` joiner:

```python
prefix_tokens = count_tokens(prefix) + 1 if prefix else 0
budget = MAX_TOKENS - prefix_tokens
```

Without the subtraction, prefixed pieces would exceed the limit. Without the
repetition, fragments 2..n would be context-free — exactly the problem the
heading chain exists to solve.

**The loop breaks once the window covers the tail:**

```python
if start + budget >= len(tokens):
    break
```

Without this, `range(0, len(tokens), stride)` keeps emitting progressively
shorter chunks that are all subsets of the last full one. Pure duplicate
storage, and duplicates in the candidate pool distort RRF fusion — the same
content appearing twice inflates its apparent cross-retriever agreement.

---

## What is kept and what is stripped

**Stripped:** `div.navheader`, `div.navfooter`, `div.toc`, `div.sect1toc` —
navigation furniture that appears on every page and would otherwise dominate
term frequencies in the full-text index.

**Kept:** code blocks (`pre.programlisting`) and parameter tables.

Keeping code and tables is deliberate. That is where the identifiers live —
`work_mem`, `GIN`, `pg_stat_activity`, `autovacuum_vacuum_scale_factor` — and
those are precisely the tokens lexical search matches well and embeddings
handle badly.
Stripping them would remove the reason this corpus was chosen.

`clean_text` operates on a re-parsed copy so `decompose()` does not mutate the
caller's tree:

```python
node = BeautifulSoup(str(node), "lxml")
```

Same class of bug as the section-intro copy above: without it, the first chunk
of a page strips navigation and subsequent chunks read from an already-modified
tree.

---

## Output format

Each chunk is one object in `chunks.json`. A real example, from a distractor
chapter:

```json
{
  "content": "# 28.2. Data Checksums > 28.2.1. Off-line Enabling of Checksums\n\nThe pg_checksums application can be used to enable or disable data checksums, as well as verify checksums, on an offline cluster.",
  "metadata": {
    "source": "checksums.html",
    "chapter": "wal.html",
    "group": "distractor",
    "page_title": "28.2. Data Checksums",
    "heading": "28.2.1. Off-line Enabling of Checksums",
    "section_id": "CHECKSUMS-OFFLINE-ENABLE-DISABLE",
    "url": "https://www.postgresql.org/docs/17/checksums.html"
  }
}
```

### Fields

| Field | What it is | What it is for |
| --- | --- | --- |
| `content` | The heading chain followed by the section text. This is the **only** field that gets embedded and the only one the LLM ever sees | Retrieval and generation |
| `metadata.source` | Filename of the HTML page this came from | Grouping chunks by page; debugging a bad chunk back to its origin |
| `metadata.chapter` | The **chapter landing page** this section was discovered through, not the section's own file. `checksums.html` sits under `wal.html` because that is the chapter whose TOC links to it | Reporting metrics per chapter; tracing which chapter contributed a distractor |
| `metadata.group` | `eval` or `distractor`, inherited from the chapter | Question generation samples only `eval` chunks; evaluation counts distractor intrusions in top-k |
| `metadata.page_title` | The page's top-level heading (`h1`/`h2`) | First half of the heading chain in `content` |
| `metadata.heading` | This section's own heading | Second half of the heading chain; human-readable label when inspecting retrieval output |
| `metadata.section_id` | The HTML anchor `id` of the section div, e.g. `CHECKSUMS-OFFLINE-ENABLE-DISABLE`. Falls back to `<page>#<n>` when the div has no id | Deep-linking to the exact subsection: `<url>#<section_id>` |
| `metadata.url` | Canonical URL of the page | Verification. A model can cite "Chunk 2", but without a URL nobody can check the claim against the source |

Note the distinction between `source` and `chapter`. `source` is where the
text physically lives; `chapter` is which chapter's table of contents linked to
it. They differ for every section page, and the `chapter` value is what carries
the eval/distractor decision.

`section_id` plus `url` is what makes a citation actionable rather than
decorative — `checksums.html#CHECKSUMS-OFFLINE-ENABLE-DISABLE` lands the reader
on the exact paragraph, not the top of a long page.

Only `content` is embedded. Everything in `metadata` exists for annotation,
evaluation, and verification — none of it influences the vector.

### Group tagging

The `group` field is read from the fetch manifest and has two downstream uses:

- **Question generation** samples source chunks only from `group == "eval"`, so
  every generated question is about one of the five eval chapters
- **Evaluation** can report how often a distractor displaced a correct chunk in
  the top-k — a direct measure of whether the distractors are doing their job,
  and of what reranking is actually fixing

`group` does not exclude distractors from ground truth. The golden-set judges
label every chunk in a question's pool, distractors included, and a distractor
judged a correct answer counts as correct rather than as an intrusion
(`05-questions.md`).

A missing manifest is tolerated rather than fatal — chunking still runs, every
chunk tagged `unknown`, with a warning. That keeps a partial re-run from
failing outright.

---

## Parameters

| Parameter | Value | Reasoning |
| --- | --- | --- |
| `MAX_TOKENS` | 512 | Small enough that a chunk covers one topic; large enough to carry a usable answer. To be measured against 256 and 1024 in Tier 2 — note that changing it renumbers every chunk, so the golden dataset stores ground truth by section id and content as well as by id (`05-questions.md`) |
| `OVERLAP` | 64 | Only applies to oversized sections. Content near a forced boundary appears in both neighbours, so an answer split across the cut survives. Observed working: see *Overlap in practice* below |
| `MIN_TOKENS` | 32 | Below this a chunk is a heading and a sentence fragment — it pollutes the candidate pool without ever being the right answer. Dropped 53 chunks once duplicated headings were removed |
| `MIN_INTRO_TOKENS` | 100 | Separates real section intros (139+) from heading-plus-TOC noise (64) |
| Encoding | `cl100k_base` | The encoding used by `text-embedding-3-*` and `gpt-4o-*`. A different encoding would make every count here wrong |

Chunk size was not tuned — 512 is a starting point, and "512 is a common
default" is a weak interview answer. Tier 2 replaces it with three measured
configurations.

---

## Overlap in practice

`19.4.1. Memory` is a reference list of a dozen parameters, too long for one
chunk, so it splits into eight. The `work_mem` description happens to straddle
the boundary between chunks 1030 and 1031, and the overlap does exactly its
job: the sentence that answers *"is work_mem a per-connection limit?"* —
*"each operation generally being allowed to use as much memory as this value
specifies"* — appears in both.

Without the overlap, the sentence could have been cut in half, leaving neither
chunk able to answer.

The overlap has a consequence downstream that is easy to miss: **one answer,
two correct chunks.** An evaluation keyed on a single chunk id will mark a
retrieval as failed when it returned the other one. That is precisely what
happened during retrieval testing (`04-retrieval.md`, *A false alarm*), and it
is why the golden dataset stores ground truth as a set of chunks with an
explicit `any` / `all` match rule (`05-questions.md`).

It also points at a cheap retrieval improvement for later: when a chunk is
retrieved, fetch its same-section neighbours (id ± 1) as well. That reunites
a description split across a boundary, and costs one indexed lookup.

---

## Why JSON before the database

Chunking and embedding are separate scripts with `chunks.json` between them.

**Inspect before paying.** Embedding is a metered API call. Re-chunking is free;
re-embedding is not. Verifying heading chains, sentence boundaries, and
identifier survival costs nothing at this stage — and both real bugs were caught
here, before any embedding spend: the dropped section intros, and the duplicated
headings inflating full-text term frequencies. Chunking bugs do not raise errors.
They quietly lower the ceiling on retrieval, and no amount of reranking recovers
what was never indexed.

**Cheap parameter sweeps.** Tier 2 compares three chunk sizes. Each run
produces a JSON file whose chunk count and token distribution can be compared
before committing to an embedding run.

**Independent re-runs.** A bug in chunking does not require re-fetching; a bug
in embedding does not require re-chunking. Both corpus expansions re-ran
chunking over cached HTML with no network calls.

---

## Known limitations

**`max` is 522, ten tokens over the nominal limit.** `split_oversized`
subtracts the heading-chain cost from the window budget, including the `"\n\n"`
joiner — and the overflow persists anyway, for a more interesting reason than
arithmetic.

Token boundaries are not preserved across decode-then-concatenate.
`ENCODER.decode(tokens[start:start + budget])` cuts at a token boundary, but the
resulting *string*, once joined to the prefix and re-encoded, can tokenize
differently at the seam. A few characters that were one token in isolation
become two in context, or the reverse.

Not worth fixing. `MAX_TOKENS` is a knob of ours, not an API limit —
`text-embedding-3-small` accepts 8,191 tokens, so 522 is nowhere near a
constraint. Enforcing an exact ceiling would mean re-encoding and trimming in a
loop, which is real complexity for no benefit. The boundary is approximate, and
knowing *why* it is approximate is worth more than making it exact.

**Sub-headings inside a section body are dropped.** `drop_headings` removes
every `h1`-`h5` from the body, which takes out genuine in-section structure
along with the duplicated section title — an admonition heading like
`"Warning: Serializable Transactions and Data Replication"` disappears while its
body text remains.

The warning content survives and reads coherently in context, so the loss is
small. Keeping only the section's own first heading out would preserve these,
at the cost of noticeably more fiddly code for a marginal gain.

**Distractor sufficiency is only spot-checked.** 954 distractor chunks against
339 eval chunks is roughly 3:1, and the distractor chapters were chosen for
vocabulary overlap rather than volume. A five-query check after ingest found a
mean of 2.8 distractors in the vector top-10 (`03-ingest.md`, *First retrieval
results*), so the pressure is real. The full measurement is the distractor
intrusion rate reported by the evaluation harness.

**No LLM-generated chunk context yet.** The heading chain is a structural
substitute. Whether an LLM-written context line beats it, and by how much, is
an open measurement.
