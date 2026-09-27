"""
Embed chunks and load them into pgvector.

Reads data/processed/chunks.json, embeds each chunk's content in batches, and
writes rows to retrieval_lab_chunks with three retrieval surfaces:

  embedding    vector(1536)  -- dense semantic search (HNSW)
  content_tsv  tsvector      -- stemmed prose search (GIN)
  identifiers  text[]        -- exact snake_case match (GIN)

Then materializes per-lexeme document frequency into
retrieval_lab_term_stats, which the lexical retriever uses to drop
low-discrimination query terms.

Design decisions worth carrying forward:

- Full rebuild, not incremental. The script truncates before loading. Chunking
  parameters change (chunk size is a Tier 2 experiment), and reconciling an
  incremental update against shifted chunk boundaries is complex and leaves
  stale rows behind. Rebuilding ~1,300 rows takes seconds.

- content_tsv is a generated column, computed by Postgres from content. The
  lexical path needs it, but nothing here has to maintain it -- defining it
  now means no later migration, and the text can never drift apart from its
  search vector.

- identifiers is a plain column populated in Python. Generated columns cannot
  contain subqueries, and aggregating regexp_matches into an array requires
  one. A trigger would work but puts the extraction rule somewhere harder to
  find than the ingest script that owns it.
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path

import asyncpg
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI

load_dotenv(find_dotenv())

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.json"

TABLE = "retrieval_lab_chunks"
STATS_TABLE = "retrieval_lab_term_stats"

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
EMBED_BATCH = 100      # the API accepts up to 2048, but smaller batches
                       # localize the damage when one request fails
MAX_RETRIES = 3

# snake_case identifiers: a word followed by at least one _word group.
#
# Why this column exists at all: Postgres's text-search parser splits on
# underscores, at the tokenizer level, before any dictionary or stemming
# runs. So work_mem indexes as 'work' + 'mem', and no text-search
# configuration changes that -- 'simple' splits it too.
#
# The consequences are worse than losing exactness. maintenance_work_mem
# indexes as 'mainten' + 'work' + 'mem', sharing two of three lexemes with
# work_mem, so the two parameters are nearly indistinguishable lexically. And
# 'mem' alone matches every passage about memory -- which is precisely what
# dense retrieval already does well, meaning the lexical path had lost its
# only comparative advantage.
#
# This corpus was chosen for identifier density. Extracting identifiers
# whole, outside the tsvector pipeline, is what makes that choice pay off.
IDENTIFIER_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+")

client = AsyncOpenAI()


def to_pgvector(embedding: list[float]) -> str:
    """
    pgvector's type is added by an extension, not native to Postgres, so
    asyncpg has no codec for it. The value travels as a string literal and is
    cast with ::vector in the SQL.
    """
    return "[" + ",".join(map(str, embedding)) + "]"


def extract_identifiers(content: str) -> list[str]:
    """
    Pull snake_case identifiers out of chunk text, lowercased and deduped.

    Lowercasing matters because the docs are inconsistent -- SHARED_BUFFERS in
    a heading, shared_buffers in prose. Matching has to be case-insensitive on
    both sides, and normalizing at write time is cheaper than at query time.
    """
    return sorted({m.group().lower() for m in IDENTIFIER_RE.finditer(content)})


async def init_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                id          SERIAL PRIMARY KEY,
                content     TEXT NOT NULL,
                metadata    JSONB NOT NULL DEFAULT '{{}}',
                embedding   vector({EMBED_DIM}),

                -- Exact snake_case identifiers, populated in Python. See
                -- IDENTIFIER_RE for why this cannot live in content_tsv.
                identifiers TEXT[] NOT NULL DEFAULT '{{}}',

                -- Generated column: Postgres derives this from content on
                -- write. Keeping the derivation in the database rather than
                -- the application makes it impossible for the text and its
                -- search vector to drift apart.
                content_tsv tsvector GENERATED ALWAYS AS (
                    to_tsvector('english', content)
                ) STORED,

                created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        # vector_cosine_ops builds the graph for cosine distance, which is what
        # the <=> operator uses. A mismatched opclass leaves the index in place
        # but unused -- queries stay correct and get slow, with no error to
        # tell you.
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {TABLE}_embedding_idx
            ON {TABLE} USING hnsw (embedding vector_cosine_ops)
        """)

        # GIN over the generated tsvector: the inverted index ts_rank reads.
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {TABLE}_tsv_idx
            ON {TABLE} USING gin (content_tsv)
        """)

        # GIN over the identifier array supports the && overlap operator.
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {TABLE}_identifiers_idx
            ON {TABLE} USING gin (identifiers)
        """)

        # The eval harness filters on metadata->>'group' and 'chapter'.
        # jsonb_path_ops is smaller and faster than the default opclass for
        # containment queries, which is all this needs.
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {TABLE}_metadata_idx
            ON {TABLE} USING gin (metadata jsonb_path_ops)
        """)


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed one batch, retrying with exponential backoff.

    Rate limits and transient 5xx responses are routine at this volume.
    Without a retry, one failure aborts the run -- and since the script
    truncates first, that means re-embedding everything and paying again.
    """
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.embeddings.create(model=EMBED_MODEL, input=texts)
            return [item.embedding for item in resp.data]
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            print(f"    {type(e).__name__}, retrying in {wait}s")
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")


async def ingest(pool: asyncpg.Pool, chunks: list[dict]) -> None:
    async with pool.acquire() as conn:
        # RESTART IDENTITY keeps chunk ids stable across runs with identical
        # input. The golden dataset stores ground truth by id, so without this
        # a re-ingest would silently invalidate every annotation.
        await conn.execute(f"TRUNCATE {TABLE} RESTART IDENTITY")

    start = time.monotonic()

    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        embeddings = await embed_batch([c["content"] for c in batch])

        # executemany batches the round trips. A loop of single execute calls
        # is an order of magnitude slower once there are thousands of rows.
        async with pool.acquire() as conn:
            await conn.executemany(
                f"INSERT INTO {TABLE} (content, metadata, embedding, identifiers) "
                "VALUES ($1, $2, $3::vector, $4)",
                [
                    (
                        c["content"],
                        json.dumps(c["metadata"]),
                        to_pgvector(emb),
                        extract_identifiers(c["content"]),
                    )
                    for c, emb in zip(batch, embeddings)
                ],
            )

        done = min(i + EMBED_BATCH, len(chunks))
        print(f"  {done}/{len(chunks)}  ({time.monotonic() - start:.1f}s)")


async def build_term_stats(pool: asyncpg.Pool) -> None:
    """
    Materialize document frequency per lexeme.

    Postgres's ts_rank has no IDF component. It scores by term frequency and
    position, so a query term appearing in 40% of the corpus contributes as
    much as one appearing in 0.3%. Combined with OR semantics in the lexical
    query, common stems ('queri', 'set', 'use') become pure noise -- and RRF
    then amplifies it, because a chunk both retrievers rank highly gets an
    agreement bonus even when the lexical ranking was driven entirely by a
    stopword-like term.

    This was a measured regression, not a theoretical worry: for the query
    "how do I see which queries are blocked by pg_locks", RRF ranked the
    correct chunk 4th where plain vector search had it 1st.

    Real BM25 solves this with inverse document frequency -- continuous
    weighting by term rarity. Dropping terms above a frequency threshold is a
    blunt substitute: binary keep-or-drop instead of weighting. It addresses
    the specific failure without reimplementing the scorer.

    ts_stat scans content_tsv across the whole table, so this is computed once
    at ingest rather than per query.
    """
    async with pool.acquire() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {STATS_TABLE}")
        # ts_stat takes the query as a text argument. Dollar quoting avoids
        # escaping the inner single quotes.
        await conn.execute(f"""
            CREATE TABLE {STATS_TABLE} AS
            SELECT word, ndoc, nentry
            FROM ts_stat($q$SELECT content_tsv FROM {TABLE}$q$)
        """)
        await conn.execute(f"CREATE INDEX ON {STATS_TABLE} (word)")

        total = await conn.fetchval(f"SELECT COUNT(*) FROM {TABLE}")
        distinct = await conn.fetchval(f"SELECT COUNT(*) FROM {STATS_TABLE}")

        # The terms most likely to be doing damage, for threshold calibration.
        top = await conn.fetch(f"""
            SELECT word, ndoc, ROUND(100.0 * ndoc / $1, 1) AS pct
            FROM {STATS_TABLE}
            ORDER BY ndoc DESC
            LIMIT 12
        """, total)

    print(f"\n{distinct} distinct lexemes -> {STATS_TABLE}")
    print("most frequent:")
    for r in top:
        print(f"  {r['word']:22} {r['ndoc']:5} chunks  ({r['pct']}%)")


async def verify(pool: asyncpg.Pool) -> None:
    """
    Confirm the load landed and every retrieval surface is populated.

    Each check corresponds to a failure mode that produces no error message.
    """
    async with pool.acquire() as conn:
        total = await conn.fetchval(f"SELECT COUNT(*) FROM {TABLE}")
        by_group = await conn.fetch(f"""
            SELECT metadata->>'group' AS grp, COUNT(*) AS n
            FROM {TABLE}
            GROUP BY 1
            ORDER BY 2 DESC
        """)

        print(f"\n{total} rows in {TABLE}")
        for row in by_group:
            print(f"  {row['grp']:12} {row['n']}")

        # Every row should carry a vector; a NULL means an insert silently
        # dropped its embedding.
        missing = await conn.fetchval(
            f"SELECT COUNT(*) FROM {TABLE} WHERE embedding IS NULL"
        )
        print(f"\nrows missing an embedding: {missing}")

        # content_tsv is generated, so this checks Postgres populated it --
        # not that our own code did.
        tsv_empty = await conn.fetchval(
            f"SELECT COUNT(*) FROM {TABLE} WHERE content_tsv IS NULL"
        )
        print(f"rows missing a tsvector:   {tsv_empty}")

        ident_stats = await conn.fetchrow(f"""
            SELECT COUNT(*) FILTER (WHERE cardinality(identifiers) = 0) AS empty,
                   ROUND(AVG(cardinality(identifiers)), 1) AS mean,
                   MAX(cardinality(identifiers)) AS max
            FROM {TABLE}
        """)
        print(f"rows with no identifiers:  {ident_stats['empty']}"
              f"  (mean {ident_stats['mean']}, max {ident_stats['max']})")

        # The identifiers that appear in the most chunks -- a sanity check that
        # the regex caught real parameter names rather than incidental
        # underscore text.
        top_idents = await conn.fetch(f"""
            SELECT ident, COUNT(*) AS n
            FROM {TABLE}, unnest(identifiers) AS ident
            GROUP BY 1
            ORDER BY 2 DESC
            LIMIT 8
        """)
        print("most common identifiers:")
        for r in top_idents:
            print(f"  {r['ident']:34} {r['n']:4} chunks")

        plan = await conn.fetch(f"""
            EXPLAIN (COSTS OFF)
            SELECT id FROM {TABLE}
            ORDER BY embedding <=> (SELECT embedding FROM {TABLE} LIMIT 1)
            LIMIT 5
        """)
        plan_text = "\n".join(r["QUERY PLAN"] for r in plan)
        used = f"{TABLE}_embedding_idx" in plan_text
        print(f"\nHNSW index used by the planner: {used}")
        if not used:
            # At small row counts a sequential scan is genuinely cheaper, so
            # this is informational rather than a failure.
            print("  (a sequential scan is expected at this row count)")
            print(plan_text)


async def main() -> None:
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    print(f"{len(chunks)} chunks from {CHUNKS_PATH.name}\n")

    pool = await asyncpg.create_pool(
        os.environ["DATABASE_URL"], min_size=2, max_size=10
    )
    try:
        # DROP rather than migrate: the identifiers column is new, and an
        # ALTER would leave existing rows with an empty array. A full rebuild
        # is the ingest strategy anyway.
        async with pool.acquire() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {TABLE}")

        await init_schema(pool)
        await ingest(pool, chunks)
        # Term stats depend on content_tsv being populated, so this runs after
        # the load, not as part of schema creation.
        await build_term_stats(pool)
        await verify(pool)
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())