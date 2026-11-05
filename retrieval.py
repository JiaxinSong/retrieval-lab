"""
Retrieval layer: three retrievers, RRF fusion, section-diversity cap, and
cross-encoder reranking.

A module, not a script -- the demo (scripts/04_hybrid_search.py) and the eval
harness both import from here. Keeping one implementation matters for the
harness specifically: if the demo and the evaluation used separate copies,
any drift between them would show up as a configuration difference between
measured rows, and the comparison would be meaningless.

The retrieval stack, and why each layer exists:

  vector      dense semantic search over HNSW. Handles paraphrase; unreliable
              on rare exact tokens, because those appear too rarely in
              embedding training data for a stable representation.

  lexical     stemmed prose search over a GIN tsvector index, OR semantics
              with document-frequency filtering. Handles literal matches;
              fails on zero word overlap, and cannot see identifiers at all
              (Postgres's parser splits on underscores).

  identifier  exact snake_case match over a text[] column. Exists because of
              that last point -- this corpus was chosen for identifier
              density, so the gap sat exactly where the corpus was meant to
              demonstrate value.

  RRF         rank-based fusion. Rewards cross-retriever agreement, which is
              only evidence when every retriever is ranking on signal -- see
              MAX_DOC_FRACTION.

  section cap structural diversity. A long section splits into chunks that all
              score alike and can monopolize the top-k. Reranking cannot fix
              this because all of them genuinely are relevant.

  rerank      cross-encoder precision. The only layer that reads the passage
              to judge whether it answers the question, rather than comparing
              representations.
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass, field

import asyncpg
import cohere
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI

load_dotenv(find_dotenv())

TABLE = "retrieval_lab_chunks"
STATS_TABLE = "retrieval_lab_term_stats"

EMBED_MODEL = "text-embedding-3-small"
RERANK_MODEL = "rerank-v3.5"

RECALL_K = 20            # candidates each retriever returns before fusion
RRF_K = 60               # RRF smoothing constant
MAX_PER_SECTION = 2      # cap on chunks from one document section; 0 disables
MAX_DOC_FRACTION = 0.15  # drop query lexemes appearing in more than this
                         # share of the corpus
RERANK_TOP_N = 5         # final result size
RERANK_POOL = 40         # fused candidates sent to the reranker, before the
                         # section cap runs. Cohere bills per search up to 100
                         # documents, so a larger pool costs nothing extra

# Must match the pattern the ingest script used. If the two diverge, the query
# extracts identifiers the index does not contain.
IDENTIFIER_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+")

openai_client = AsyncOpenAI()

# Cohere's client is created lazily: importing this module should not require
# a Cohere key when only the retrievers are used.
_cohere_client: cohere.ClientV2 | None = None

# Corpus size, needed to turn MAX_DOC_FRACTION into an absolute ndoc cutoff.
# Cached after the first lookup since it does not change within a run.
_corpus_size: int | None = None


# ---------------------------------------------------------------- data model

@dataclass
class Hit:
    """One retrieved chunk, carrying where each retriever ranked it."""

    id: int
    content: str
    metadata: dict

    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    rrf_score: float = 0.0
    rerank_score: float | None = None

    @property
    def group(self) -> str:
        return self.metadata.get("group", "unknown")

    @property
    def heading(self) -> str:
        return self.metadata.get("heading", "")

    @property
    def section_id(self) -> str:
        return self.metadata.get("section_id", "")

    def rank_summary(self) -> str:
        """Compact provenance, e.g. "v#3 l#1 i#2" -- the diagnostic that
        explains why fusion placed this chunk where it did."""
        initials = {"vector": "v", "lexical": "l", "identifier": "i"}
        return " ".join(
            f"{initials[name]}#{rank}" for name, rank in self.ranks.items()
        )


def to_pgvector(embedding: list[float]) -> str:
    """
    pgvector's type is added by an extension, not native to Postgres, so
    asyncpg has no codec for it. The value travels as a string literal and is
    cast with ::vector in the SQL.
    """
    return "[" + ",".join(map(str, embedding)) + "]"


async def corpus_size(pool: asyncpg.Pool) -> int:
    global _corpus_size
    if _corpus_size is None:
        _corpus_size = await pool.fetchval(f"SELECT COUNT(*) FROM {TABLE}")
    return _corpus_size


def _make_hits(rows, retriever: str, score_key: str) -> list[Hit]:
    """Build Hit objects, recording this retriever's rank and score."""
    return [
        Hit(
            id=r["id"],
            content=r["content"],
            metadata=json.loads(r["metadata"]),
            ranks={retriever: i},
            scores={retriever: float(r[score_key])},
        )
        for i, r in enumerate(rows, start=1)
    ]


# ---------------------------------------------------------------- retrievers

async def vector_search(pool: asyncpg.Pool, query: str, k: int = RECALL_K) -> list[Hit]:
    """
    Dense retrieval over the HNSW index.

    ORDER BY repeats the full distance expression rather than sorting on the
    SELECT alias. The planner only recognizes the index-scan pattern in the
    form `ORDER BY <indexed column> <operator> <constant>`; sorting on an alias
    forces it to compute the distance for every row first. Correct results,
    sequential scan, no error -- the only symptom is latency.
    """
    resp = await openai_client.embeddings.create(model=EMBED_MODEL, input=[query])
    q_vec = to_pgvector(resp.data[0].embedding)

    rows = await pool.fetch(
        f"""
        SELECT id, content, metadata,
               1 - (embedding <=> $1::vector) AS similarity
        FROM {TABLE}
        ORDER BY embedding <=> $1::vector
        LIMIT $2
        """,
        q_vec,
        k,
    )
    return _make_hits(rows, "vector", "similarity")


async def query_lexemes(
    pool: asyncpg.Pool,
    query: str,
    max_doc_fraction: float = MAX_DOC_FRACTION,
) -> tuple[list[str], list[str]]:
    """
    Tokenize a query and split its lexemes into kept and dropped.

    Terms are extracted through to_tsvector rather than split on whitespace, so
    stemming and stop-word removal use the same 'english' configuration the
    indexed column was built with. Splitting manually would leave 'updates'
    unstemmed and it would never match the indexed 'updat'.

    Terms above the document-frequency threshold are dropped. This substitutes
    for the IDF weighting ts_rank does not have: with OR semantics, a stem
    appearing in 40% of the corpus matches almost everything and contributes as
    much to the score as a rare term. RRF then amplifies the noise, because a
    chunk several retrievers rank highly gets an agreement bonus regardless of
    *why* the lexical retriever ranked it.

    LEFT JOIN with COALESCE(ndoc, 0) keeps terms absent from the corpus -- a
    typo has no stats row, and keeping it is harmless since it matches nothing.

    Returns (kept, dropped). The dropped list exists for diagnostics: seeing
    which terms were discarded is how the threshold gets calibrated, and it is
    what exposed the underscore-splitting bug.
    """
    cutoff = int(await corpus_size(pool) * max_doc_fraction)

    rows = await pool.fetch(
        f"""
        SELECT t.lexeme,
               COALESCE(s.ndoc, 0) AS ndoc
        FROM unnest(to_tsvector('english', $1)) AS t(lexeme)
        LEFT JOIN {STATS_TABLE} s ON s.word = t.lexeme
        """,
        query,
    )

    kept = [r["lexeme"] for r in rows if r["ndoc"] <= cutoff]
    dropped = [r["lexeme"] for r in rows if r["ndoc"] > cutoff]
    return kept, dropped


async def lexical_search(
    pool: asyncpg.Pool,
    query: str,
    k: int = RECALL_K,
    max_doc_fraction: float = MAX_DOC_FRACTION,
) -> list[Hit]:
    """
    Lexical retrieval over the GIN index on content_tsv.

    Query terms are combined with OR, not AND. Both websearch_to_tsquery and
    plainto_tsquery default to AND, which fails on natural-language questions:
    "Is work_mem a per-connection limit?" becomes
    'work_mem & per & connect & limit' and returns nothing unless a single
    chunk contains every term. Three of five test queries returned zero rows
    that way -- the lexical path was dead on anything phrased as a sentence.

    OR is also what BM25 actually does. It scores documents by how many query
    terms they contain and how rare those terms are; it never requires all of
    them. The rarity half is missing from ts_rank, which query_lexemes
    compensates for.
    """
    kept, dropped = await query_lexemes(pool, query, max_doc_fraction)

    # Fallback: if filtering removed everything, use the unfiltered terms. A
    # query made entirely of common words ("how do I set this up") would
    # otherwise return nothing at all, which is worse than a noisy result.
    terms = kept or dropped
    if not terms:
        # The query was entirely stop words; there is no lexical signal here.
        return []

    # The terms are already lexemes -- query_lexemes ran them through
    # to_tsvector('english'). They must reach the index unchanged, so the query
    # is built as a tsquery literal and cast, not passed to to_tsquery.
    #
    # to_tsquery('english', ...) would stem them a second time, and Snowball
    # English is not idempotent: 'database' -> 'databas' -> 'databa'. The
    # index holds 'databas'; the twice-stemmed 'databa' matches nothing. Same
    # for increase, decrease, release, purchase -- common words in this
    # corpus, silently dropped from every query that used them, with no error
    # and no empty result to notice.
    #
    # Lexemes are quoted for tsquery input syntax: quote characters doubled,
    # backslashes escaped.
    tsquery = " | ".join(
        "'" + term.replace("\\", "\\\\").replace("'", "''") + "'" for term in terms
    )

    rows = await pool.fetch(
        f"""
        SELECT id, content, metadata,
               ts_rank(content_tsv, $1::tsquery) AS rank
        FROM {TABLE}
        WHERE content_tsv @@ $1::tsquery
        ORDER BY rank DESC
        LIMIT $2
        """,
        tsquery,
        k,
    )
    return _make_hits(rows, "lexical", "rank")


def extract_identifiers(text: str) -> list[str]:
    """Snake_case tokens, lowercased to match the stored form."""
    return sorted({m.group().lower() for m in IDENTIFIER_RE.finditer(text)})


async def identifier_search(
    pool: asyncpg.Pool,
    query: str,
    k: int = RECALL_K,
) -> list[Hit]:
    """
    Exact match over the identifiers array.

    This retriever exists because Postgres's text-search parser splits on
    underscores at the tokenizer level, before any dictionary or stemming
    runs. work_mem indexes as 'work' + 'mem'; no text-search configuration
    changes that, including 'simple'.

    The damage goes beyond losing exactness. maintenance_work_mem indexes as
    'mainten' + 'work' + 'mem', sharing two of three lexemes with work_mem, so
    the two parameters are nearly indistinguishable lexically. And 'mem' alone
    matches every passage about memory -- which is what dense retrieval already
    does well. The lexical path had lost its only comparative advantage on
    exactly the content this corpus was chosen for.

    Ranking is by how many distinct query identifiers a chunk contains, then by
    how often those identifiers appear in its text.

    The second term is doing real work. Most queries carry a single identifier,
    so every match ties on the first term and the tiebreak decides the entire
    ordering. An earlier version broke the tie by total identifier count, on
    the theory that a focused chunk beats a summary table. It inverted the
    ranking instead: for "pg_locks", a section that mentions the view once in
    passing outranked the section documenting it, because the documenting
    section naturally also mentions pg_stat_activity and pg_blocking_pids.

    Mention frequency is the signal that actually separates them. A section
    about pg_locks names it a dozen times; a section citing it names it once.

    Mentions are counted on word boundaries (\\m ... \\M in Postgres regex, where
    underscore counts as a word character). work_mem is a substring of
    maintenance_work_mem and autovacuum_work_mem, so a plain substring count
    inflates chunks that document those instead: chunk 1032 contains
    "work_mem" five times as a substring and once as a word. Same failure class
    as the underscore split in the tokenizer -- treating an identifier as a
    string of characters rather than as a unit.

    Returns empty when the query has no identifiers, which is most natural
    questions. RRF handles a missing retriever without adjustment.
    """
    idents = extract_identifiers(query)
    if not idents:
        return []

    rows = await pool.fetch(
        f"""
        SELECT id, content, metadata,
               cardinality(ARRAY(
                   SELECT unnest(identifiers) INTERSECT SELECT unnest($1::text[])
               ))::float AS matched,
               COALESCE((
                   SELECT SUM(regexp_count(content, '\\m' || q.ident || '\\M', 1, 'i'))
                   FROM unnest($1::text[]) AS q(ident)
               ), 0) AS mentions
        FROM {TABLE}
        WHERE identifiers && $1::text[]
        ORDER BY matched DESC, mentions DESC
        LIMIT $2
        """,
        idents,
        k,
    )
    return _make_hits(rows, "identifier", "matched")


# ---------------------------------------------------------------- fusion

def fuse_rrf(*hit_lists: list[Hit], k: int = RRF_K) -> list[Hit]:
    """
    Reciprocal Rank Fusion over any number of retrievers.

        score(doc) = sum over retrievers of 1 / (k + rank)

    A chunk absent from one list contributes nothing from it. There is no
    penalty term, which is why the accumulator is a dict rather than something
    requiring every rank to exist -- and why adding a third retriever needed no
    change to the formula.

    The consequence is that RRF rewards cross-retriever agreement. A chunk
    ranked 2nd and 5th scores 1/62 + 1/65 = 0.0315, beating a chunk ranked 1st
    by one retriever alone at 1/61 = 0.0164. Two moderate signals that agree
    outrank one strong signal standing by itself.

    That property is also the mechanism behind the noise amplification that
    document-frequency filtering fixes: agreement is only evidence when every
    retriever is ranking on signal. Fusion does not filter noise -- it
    amplifies whatever the inputs agree on.

    k controls how much rank position matters relative to mere presence. At
    k=60 the gap between rank 1 and rank 2 is small, so agreement dominates.
    At k=1, rank 1 scores 0.5 against rank 2's 0.33 and the fusion collapses
    into "whatever one retriever happened to put first."
    """
    merged: dict[int, Hit] = {}

    for hits in hit_lists:
        for hit in hits:
            if hit.id in merged:
                existing = merged[hit.id]
                existing.ranks.update(hit.ranks)
                existing.scores.update(hit.scores)
                target = existing
            else:
                merged[hit.id] = hit
                target = hit

            # Every Hit from _make_hits carries exactly one rank entry.
            (rank,) = hit.ranks.values()
            target.rrf_score += 1 / (k + rank)

    return sorted(merged.values(), key=lambda h: h.rrf_score, reverse=True)


def cap_per_section(hits: list[Hit], max_per_section: int = MAX_PER_SECTION) -> list[Hit]:
    """
    Keep at most N chunks from any one document section.

    A long section splits into several chunks, all of which score similarly on
    the same query. Without a cap they monopolize the top-k: one test query
    returned five chunks from "24.1.5. Preventing Transaction ID Wraparound
    Failures" and nothing else. Effective diversity was one section wide while
    appearing five results deep, and RRF had nothing to contribute because both
    retrievers' candidate sets were the same content.

    Reranking does not fix this, and the reason is worth being precise about. A
    reranker judges relevance; those five chunks genuinely are relevant, so a
    cross-encoder scores them all highly and the monopoly survives. Reranking
    fixes "relevant and irrelevant mixed, in the wrong order." This is
    "entirely relevant, all from one place" -- a diversity problem, not a
    relevance problem, so it has to be handled structurally.

    The cap trades recall within a section for coverage across sections. Right
    when an answer needs two sections combined, wrong when an answer is spread
    across five chunks of one section. Which case dominates is an empirical
    question for the eval harness, hence the knob.
    """
    if max_per_section <= 0:
        return hits

    seen: dict[str, int] = {}
    kept: list[Hit] = []

    for hit in hits:
        # Chunks with no section id are not grouped -- fall back to the id so
        # each counts as its own section rather than all sharing "".
        key = hit.section_id or f"__{hit.id}"
        if seen.get(key, 0) >= max_per_section:
            continue
        seen[key] = seen.get(key, 0) + 1
        kept.append(hit)

    return kept


# ---------------------------------------------------------------- reranking

def _get_cohere() -> cohere.ClientV2:
    global _cohere_client
    if _cohere_client is None:
        _cohere_client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
    return _cohere_client


async def rerank(
    query: str,
    hits: list[Hit],
    top_n: int = RERANK_TOP_N,
    model: str = RERANK_MODEL,
) -> list[Hit]:
    """
    Cross-encoder reranking of a fused candidate list.

    The three retrievers all compare *representations*: a vector against
    vectors, query lexemes against indexed lexemes, query identifiers against
    stored identifiers. None of them reads the passage to judge whether it
    answers the question.

    That limit is visible in the fused output. For "Is work_mem a
    per-connection limit?", three chunks from section 19.4.1 appear in the
    top-5 with identical metadata -- same heading, same section id, same URL.
    Section 19.4.1 is eight chunks covering a dozen parameters. The answer
    ("each operation generally being allowed to use as much memory as this
    value specifies") sits in two of them, 1030 and 1031, where the work_mem
    description crosses a chunk boundary; the rest document other parameters.
    Nothing upstream can separate them, because the difference is in the
    prose.

    A cross-encoder concatenates query and passage into one input, so attention
    runs across both: "per-connection limit" in the query can attend to "each
    operation" and "several running sessions" in the passage. The cost is that
    nothing can
    be precomputed -- every (query, passage) pair needs a full forward pass.
    A bi-encoder stores document vectors once and compares cheaply; that is
    what makes it able to scan the whole corpus and what makes it blind to
    this distinction.

    Hence two stages, and hence reranking runs *after* recall rather than
    instead of it:

        bi-encoder recall:     1,293 -> 20   cheap, precomputed, scans all
        cross-encoder rerank:     20 -> 5    expensive, per-pair, small sets

    Cohere's client is synchronous, so the call goes through asyncio.to_thread
    to keep the event loop free -- it composes with the concurrent retrievers
    rather than blocking them.

    Failure is non-fatal: on an API error the fused order is returned
    unchanged, truncated to top_n. Degrading to hybrid-only retrieval is far
    better than failing the query, and the eval harness measures exactly how
    much is lost when this layer is absent.
    """
    if not hits:
        return []

    documents = [h.content for h in hits]

    try:
        resp = await asyncio.to_thread(
            _get_cohere().rerank,
            model=model,
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
        )
    except Exception as e:
        print(f"  rerank failed ({type(e).__name__}), falling back to fused order")
        return hits[:top_n]

    reranked: list[Hit] = []
    for result in resp.results:
        hit = hits[result.index]
        hit.rerank_score = round(result.relevance_score, 4)
        reranked.append(hit)

    return reranked


# ---------------------------------------------------------------- pipelines

async def hybrid_search(
    pool: asyncpg.Pool,
    query: str,
    recall_k: int = RECALL_K,
    max_per_section: int = MAX_PER_SECTION,
    max_doc_fraction: float = MAX_DOC_FRACTION,
) -> list[Hit]:
    """
    Three retrievers concurrently, then fuse, then cap section monopolies.

    gather matters here: the queries are independent I/O. Awaiting them in
    sequence would add the embedding round trip to the other two for nothing.
    """
    vector_hits, lexical_hits, ident_hits = await asyncio.gather(
        vector_search(pool, query, recall_k),
        lexical_search(pool, query, recall_k, max_doc_fraction),
        identifier_search(pool, query, recall_k),
    )
    fused = fuse_rrf(vector_hits, lexical_hits, ident_hits)
    return cap_per_section(fused, max_per_section)


async def hybrid_rerank_search(
    pool: asyncpg.Pool,
    query: str,
    recall_k: int = RECALL_K,
    max_per_section: int = MAX_PER_SECTION,
    max_doc_fraction: float = MAX_DOC_FRACTION,
    top_n: int = RERANK_TOP_N,
) -> list[Hit]:
    """
    The full stack: fuse, rerank, *then* cap section monopolies.

    The order matters, and the first version had it wrong. It capped before
    reranking, so the decision "which two chunks of this section survive" was
    made on RRF order -- and within one section RRF order is the least
    reliable signal available, because the chunks share heading, section id,
    and URL. That indistinguishability is the whole reason reranking exists.

    The cap also assumes a section is one topic. True for "24.1.5. Preventing
    Transaction ID Wraparound Failures". False for "19.4.1. Memory", which is a
    list of a dozen independent parameters -- shared_buffers, work_mem,
    maintenance_work_mem, autovacuum_work_mem -- split across eight chunks.

    Reranking first means the cap trims a list already ordered by relevance.
    That is the argument from principle. Whether it measurably beats the
    original order is an open question, so both are kept as configurations
    (see CONFIGS) and the eval harness decides.

    An early "measurement" appeared to show the old order deleting the correct
    answer. It did not: the check was keyed on one chunk id, but the key
    sentence of the work_mem description spans a chunk boundary and appears in
    both 1030 and 1031 through the 64-token overlap. An answer-bearing chunk was
    in the top-5 under both orders. The lesson carries into annotation -- ground
    truth is a set of chunks, not a single id.
    """
    vector_hits, lexical_hits, ident_hits = await asyncio.gather(
        vector_search(pool, query, recall_k),
        lexical_search(pool, query, recall_k, max_doc_fraction),
        identifier_search(pool, query, recall_k),
    )
    fused = fuse_rrf(vector_hits, lexical_hits, ident_hits)
    reranked = await rerank(query, fused[:RERANK_POOL], top_n=RERANK_POOL)
    return cap_per_section(reranked, max_per_section)[:top_n]


async def hybrid_cap_then_rerank(
    pool: asyncpg.Pool,
    query: str,
    top_n: int = RERANK_TOP_N,
) -> list[Hit]:
    """
    The original ordering -- cap on RRF order, then rerank the survivors.

    Kept only so the eval harness can measure it against rerank-then-cap
    rather than the choice resting on an argument.
    """
    capped = await hybrid_search(pool, query)
    return await rerank(query, capped, top_n)


# ---------------------------------------------------------------- single-retriever
# Thin wrappers so the eval harness can address every configuration through
# one uniform signature. Each truncates to the same top_n the fused pipelines
# return -- comparing a top-20 vector list against a top-5 reranked list would
# measure list length, not retrieval quality.

async def vector_only(pool: asyncpg.Pool, query: str, top_n: int = RERANK_TOP_N) -> list[Hit]:
    return (await vector_search(pool, query, RECALL_K))[:top_n]


async def lexical_only(pool: asyncpg.Pool, query: str, top_n: int = RERANK_TOP_N) -> list[Hit]:
    return (await lexical_search(pool, query, RECALL_K))[:top_n]


async def identifier_only(pool: asyncpg.Pool, query: str, top_n: int = RERANK_TOP_N) -> list[Hit]:
    return (await identifier_search(pool, query, RECALL_K))[:top_n]


async def hybrid_only(pool: asyncpg.Pool, query: str, top_n: int = RERANK_TOP_N) -> list[Hit]:
    return (await hybrid_search(pool, query))[:top_n]


async def hybrid_plus_rerank(pool: asyncpg.Pool, query: str, top_n: int = RERANK_TOP_N) -> list[Hit]:
    return await hybrid_rerank_search(pool, query, top_n=top_n)


async def hybrid_no_cap(pool: asyncpg.Pool, query: str, top_n: int = RERANK_TOP_N) -> list[Hit]:
    """Fusion without the section cap. Against `vector`, this isolates fusion:
    `hybrid` differs from `vector` in both fusion and the cap, so on its own
    that comparison cannot say which one moved the result."""
    return (await hybrid_search(pool, query, max_per_section=0))[:top_n]


async def vector_rerank(pool: asyncpg.Pool, query: str, top_n: int = RERANK_TOP_N) -> list[Hit]:
    """Vector search alone as the reranker's candidate source.

    Against `hybrid+rerank`, this asks whether the lexical and identifier
    retrievers still add anything once a cross-encoder reads the candidates.
    The candidates are the vector top RECALL_K -- the same depth the judged
    pool was built from, so every chunk this can return has a label. Rerank
    then cap, matching `hybrid+rerank`.
    """
    hits = await vector_search(pool, query, RECALL_K)
    reranked = await rerank(query, hits, top_n=len(hits))
    return cap_per_section(reranked)[:top_n]


async def hybrid_rerank_no_cap(pool: asyncpg.Pool, query: str, top_n: int = RERANK_TOP_N) -> list[Hit]:
    """The full stack without the section cap. The cap trades depth within a
    section for coverage across sections; this configuration measures what
    that trade costs and what it buys."""
    return await hybrid_rerank_search(pool, query, max_per_section=0, top_n=top_n)


# The configurations the eval harness measures. Keeping them in the retrieval
# module rather than the harness means a new configuration is added in one
# place and every report picks it up.
CONFIGS = {
    "vector": vector_only,
    "lexical": lexical_only,
    "identifier": identifier_only,
    "hybrid": hybrid_only,
    "hybrid+rerank": hybrid_plus_rerank,
    "hybrid+cap+rerank": hybrid_cap_then_rerank,
    "hybrid+rerank+nocap": hybrid_rerank_no_cap,
    "hybrid+nocap": hybrid_no_cap,
    "vector+rerank": vector_rerank,
}


async def connect() -> asyncpg.Pool:
    return await asyncpg.create_pool(os.environ["DATABASE_URL"], min_size=2, max_size=10)