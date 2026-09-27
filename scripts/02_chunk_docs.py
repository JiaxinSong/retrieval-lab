"""
Convert PostgreSQL documentation HTML into retrieval chunks.

Strategy: split on the document's own section boundaries rather than at fixed
token offsets. DocBook-generated HTML marks each subsection with
div.sect1/div.sect2, and each of those is one coherent idea. Cutting there
produces chunks that are semantically whole; cutting at token 512 produces
chunks that are half a thought, which embeddings represent poorly.

Every chunk carries its heading chain as a prefix -- "Resource Consumption >
Memory" ahead of the text about work_mem. This is a no-cost version of
contextual retrieval: a chunk lifted out of its document loses the topic it
belongs to, and the heading chain puts it back before the text is embedded.

Oversized sections still need splitting, and those fall back to token windows
with overlap. The heading prefix is repeated on each piece so no fragment is
context-free.

Each chunk carries its chapter group from the fetch manifest. "eval" chunks
come from chapters the golden dataset queries; "distractor" chunks are never
the source of a question and exist to compete for top-k slots. Tagging them at
chunking time is what lets question generation sample only eval chunks and
lets the evaluation step report how often a distractor displaced a correct
chunk.
"""

import json
import re
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup, Tag

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "html"
MANIFEST_PATH = ROOT / "data" / "raw" / "manifest.json"
OUT_PATH = ROOT / "data" / "processed" / "chunks.json"

# cl100k_base is the encoding used by text-embedding-3-* and gpt-4o-*.
# A different encoding would make every token count here wrong.
ENCODER = tiktoken.get_encoding("cl100k_base")

MAX_TOKENS = 512
OVERLAP = 64
MIN_TOKENS = 32          # below this a chunk is too thin to retrieve usefully
MIN_INTRO_TOKENS = 100   # below this a section "intro" is a heading plus a TOC


def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))


def load_manifest() -> dict[str, dict[str, str]]:
    """
    Page-level provenance written by the fetch script.

    Missing manifest is tolerated rather than fatal: chunking still works,
    every chunk is just tagged "unknown". That keeps a partial re-run from
    failing outright, but the group tags are needed for evaluation, so the
    warning matters.
    """
    if not MANIFEST_PATH.exists():
        print(f"warning: no manifest at {MANIFEST_PATH}; groups will be 'unknown'\n")
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def normalize(text: str) -> str:
    """
    Collapse every run of whitespace into a single space.

    DocBook output carries the source file's indentation into the text nodes,
    so raw extraction is littered with newlines and tabs that mean nothing.
    It also emits U+00A0 between a section number and its title
    ("28.2.1.\u00a0Off-line Enabling"), which the \\s class does not match in
    all engines -- hence the explicit replace first.

    Paragraph structure is not preserved because get_text(separator=" ")
    already discarded it, and neither embeddings nor full-text search read
    paragraph breaks. The only newline that survives in a finished chunk is the one
    deliberately inserted between the heading chain and the body.

    The trailing "#" comes from the permalink anchor the docs render beside
    every heading. Left in, it lands in page_title, in heading, and twice in
    the heading chain -- a meaningless high-frequency token in the full-text
    index.
    """
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*#\s*$", "", text)  # permalink marker at end of headings
    return text.strip()


def clean_text(node: Tag, drop_headings: bool = True) -> str:
    """
    Extract readable text from a DocBook HTML node.

    Navigation furniture is removed; code blocks and tables are kept. Code and
    parameter tables are where the identifiers live -- work_mem, GIN,
    pg_stat_activity -- and those are exactly the tokens lexical search matches
    and embeddings handle badly. Stripping them would remove the reason this
    corpus was chosen.

    Headings are dropped from the body by default because the heading chain
    prefix already states them. Leaving them in wastes 10-15 tokens per chunk
    and, worse, doubles the frequency of those terms in the full-text index -- a
    section whose title matches the query gets a lexical boost it did not earn.
    """
    # Work on a copy so the caller's tree is not mutated by decompose().
    node = BeautifulSoup(str(node), "lxml")

    for junk in node.select("div.navheader, div.navfooter, div.toc, div.sect1toc"):
        junk.decompose()

    if drop_headings:
        for h in node.find_all(["h1", "h2", "h3", "h4", "h5"]):
            h.decompose()

    return normalize(node.get_text(separator=" "))


def heading_of(node: Tag) -> str:
    """First heading inside a section node, or empty string if none."""
    h = node.find(["h1", "h2", "h3", "h4"], recursive=True)
    return normalize(h.get_text()) if h else ""


def split_oversized(text: str, prefix: str) -> list[str]:
    """
    Token-window split for sections that exceed MAX_TOKENS on their own.

    The heading prefix is prepended to every piece and its cost is subtracted
    from the window, so each fragment stays under the limit and none of them
    ends up context-free.
    """
    # The "\n\n" joiner costs a token too; charging for it keeps every piece
    # at or under MAX_TOKENS rather than one over.
    prefix_tokens = count_tokens(prefix) + 1 if prefix else 0
    budget = MAX_TOKENS - prefix_tokens

    if budget <= OVERLAP:
        # Pathological case: the heading chain alone nearly fills the window.
        # Emit the body unprefixed rather than loop forever.
        budget, prefix = MAX_TOKENS, ""

    tokens = ENCODER.encode(text)
    stride = budget - OVERLAP
    pieces: list[str] = []

    for start in range(0, len(tokens), stride):
        body = ENCODER.decode(tokens[start : start + budget])
        pieces.append(f"{prefix}\n\n{body}".strip() if prefix else body)

        # Stop once the window covers the tail. Without this the loop emits
        # progressively shorter chunks that are all subsets of the last one,
        # and duplicates in the candidate pool distort RRF fusion -- the same
        # content appearing twice inflates its apparent cross-retriever
        # agreement.
        if start + budget >= len(tokens):
            break

    return pieces


def chunk_file(path: Path, group: str, chapter: str) -> list[dict]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")

    # The page title is the top of every heading chain on this page.
    title_node = soup.find(["h1", "h2"])
    page_title = normalize(title_node.get_text()) if title_node else path.stem

    # Prefer the finest-grained sections available. sect2 is usually one
    # parameter or one concept; sect1 is the whole page. Falling back through
    # the levels keeps pages that use only one of them from being skipped.
    sections = (
        soup.select("div.sect2")
        or soup.select("div.sect1")
        or soup.select("div.refsect1")
    )

    chunks: list[dict] = []

    def emit(content: str, heading: str, section_id: str) -> None:
        if count_tokens(content) < MIN_TOKENS:
            return
        chunks.append({
            "content": content,
            "metadata": {
                "source": path.name,
                "chapter": chapter,
                "group": group,
                "page_title": page_title,
                "heading": heading,
                "section_id": section_id,
                "url": f"https://www.postgresql.org/docs/17/{path.name}",
            },
        })

    if not sections:
        # No section divs at all -- treat the body as one unit.
        body = soup.find("body") or soup
        text = clean_text(body)
        prefix = f"# {page_title}"
        pieces = (
            split_oversized(text, prefix)
            if count_tokens(text) > MAX_TOKENS
            else [f"{prefix}\n\n{text}"]
        )
        for piece in pieces:
            emit(piece, page_title, path.stem)
        return chunks

    # Section intros: prose sitting in a sect1 before its first sect2. These
    # are the conceptual overviews -- what vacuuming is, why query plans matter
    # -- and they answer the "why" questions that per-parameter subsections do
    # not. Selecting only sect2 would drop them entirely.
    if soup.select("div.sect2"):
        for s1 in soup.select("div.sect1"):
            if not s1.select("div.sect2"):
                continue  # no subsections here; the main loop covers this node

            # Copy before decomposing: mutating s1 in place would destroy the
            # sect2 nodes the main loop is about to read.
            s1_copy = BeautifulSoup(str(s1), "lxml")
            for s2 in s1_copy.select("div.sect2"):
                s2.decompose()

            intro = clean_text(s1_copy)
            if count_tokens(intro) < MIN_INTRO_TOKENS:
                continue

            s1_heading = heading_of(s1)
            chain = f"# {page_title}"
            if s1_heading and s1_heading != page_title:
                chain = f"# {page_title} > {s1_heading}"

            full = f"{chain}\n\n{intro}"
            section_id = s1.get("id") or f"{path.stem}#intro"

            if count_tokens(full) <= MAX_TOKENS:
                emit(full, s1_heading or page_title, section_id)
            else:
                for piece in split_oversized(intro, chain):
                    emit(piece, s1_heading or page_title, section_id)

    for sect in sections:
        heading = heading_of(sect)
        text = clean_text(sect)

        if not text:
            continue

        # The heading chain: page title, then this section's heading. This is
        # what restores topical context to a chunk read in isolation.
        chain = f"# {page_title}" + (
            f" > {heading}" if heading and heading != page_title else ""
        )
        section_id = sect.get("id") or f"{path.stem}#{len(chunks)}"

        full = f"{chain}\n\n{text}"

        if count_tokens(full) <= MAX_TOKENS:
            emit(full, heading or page_title, section_id)
        else:
            for piece in split_oversized(text, chain):
                emit(piece, heading or page_title, section_id)

    return chunks


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    files = sorted(RAW_DIR.glob("*.html"))
    print(f"{len(files)} HTML files\n")

    all_chunks: list[dict] = []
    for path in files:
        info = manifest.get(path.name, {})
        group = info.get("group", "unknown")
        chapter = info.get("chapter", "unknown")

        produced = chunk_file(path, group, chapter)
        all_chunks.extend(produced)
        print(f"  {path.name:42} {len(produced):3} chunks  [{group}]")

    OUT_PATH.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")

    sizes = [count_tokens(c["content"]) for c in all_chunks]
    n_eval = sum(1 for c in all_chunks if c["metadata"]["group"] == "eval")
    n_distractor = sum(1 for c in all_chunks if c["metadata"]["group"] == "distractor")

    print(f"\n{len(all_chunks)} chunks -> {OUT_PATH}")
    print(f"  {n_eval} eval, {n_distractor} distractor")
    print(f"tokens: min {min(sizes)}, max {max(sizes)}, mean {sum(sizes) // len(sizes)}")
    print(f"total:  {sum(sizes):,} tokens")
    print(f"embedding cost at $0.02/1M: ${sum(sizes) * 0.02 / 1_000_000:.4f}")


main()