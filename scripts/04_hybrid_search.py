"""
Demo script for the retrieval stack. All logic lives in retrieval.py; this
file only exercises it and prints diagnostics.

Run from the project root so the module import resolves:

    python3 scripts/04_hybrid_search.py

The queries are chosen to probe specific failure modes rather than to look
good. Three were added after they exposed bugs, and they stay in the set so
the same bugs would be visible if a future change reintroduced them.
"""

import asyncio
import sys
from pathlib import Path

# retrieval.py sits at the project root, not in scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval import (  # noqa: E402
    MAX_DOC_FRACTION,
    MAX_PER_SECTION,
    RECALL_K,
    RERANK_POOL,
    Hit,
    cap_per_section,
    connect,
    corpus_size,
    extract_identifiers,
    fuse_rrf,
    identifier_search,
    lexical_search,
    query_lexemes,
    rerank,
    vector_search,
)

QUERIES = [
    # Bare identifier -- identifier search should dominate
    "work_mem",
    # Two identifiers the tsvector path cannot distinguish: both index as
    # 'work' + 'mem', one with 'mainten' added
    "difference between work_mem and maintenance_work_mem",
    # Identifier inside a sentence -- the case AND semantics killed, and the
    # case where three chunks from one section have identical metadata
    "Is work_mem a per-connection limit?",
    # Pure paraphrase, zero identifiers -- dense retrieval's case
    "why does the database slow down after many updates",
    # Long identifier, topic the distractor chapters do not cover
    "what does autovacuum_vacuum_scale_factor control",
    # A section that splits into many similar chunks -- the monopoly case
    "preventing transaction id wraparound",
]


def show(label: str, hits: list[Hit], n: int = 5) -> None:
    print(f"  {label}")
    if not hits:
        print("    (no results)\n")
        return

    for i, h in enumerate(hits[:n], 1):
        tag = "D" if h.group == "distractor" else " "
        score = f"{h.rerank_score:.4f}" if h.rerank_score is not None else ""
        print(f"    {i} {tag} {h.rank_summary():18} {score:8} {h.heading[:44]}")
    print()


async def main() -> None:
    pool = await connect()
    try:
        size = await corpus_size(pool)
        cutoff = int(size * MAX_DOC_FRACTION)
        print(f"corpus {size} chunks, df cutoff {cutoff} ({MAX_DOC_FRACTION:.0%})\n")

        for query in QUERIES:
            print("=" * 72)
            print(f"{query!r}\n")

            kept, dropped = await query_lexemes(pool, query)
            print(f"  lexemes kept:    {kept}")
            print(f"  lexemes dropped: {dropped}")
            print(f"  identifiers:     {extract_identifiers(query)}\n")

            vector_hits, lexical_hits, ident_hits = await asyncio.gather(
                vector_search(pool, query, RECALL_K),
                lexical_search(pool, query, RECALL_K),
                identifier_search(pool, query, RECALL_K),
            )
            fused = fuse_rrf(vector_hits, lexical_hits, ident_hits)
            capped = cap_per_section(fused)

            # Print the pre-rerank stages before reranking runs. Hit objects
            # are shared across these lists -- fuse_rrf merges duplicates into
            # one instance rather than copying -- so once rerank writes
            # rerank_score, every list holding that chunk shows the score too.
            # Harmless to the data, misleading in the output: the vector-only
            # rows would appear to carry cross-encoder scores.
            show("vector only", vector_hits)
            show("lexical only", lexical_hits)
            show("identifier only", ident_hits)
            show("RRF fused", fused)
            show(f"hybrid only: + section cap ({MAX_PER_SECTION}/section)", capped)

            # Full stack reranks the *uncapped* pool, then caps. Capping first
            # decided which same-section chunks survive using RRF order -- the
            # one signal that cannot tell them apart.
            reranked = await rerank(query, fused[:RERANK_POOL], top_n=RERANK_POOL)
            final = cap_per_section(reranked)[:5]
            show("full stack: rerank -> section cap", final)
            capped = capped[:5]
            reranked = final

            # Overlap quantifies how much fusion has to contribute. Low
            # overlap means the retrievers are genuinely finding different
            # things; high overlap means they agree and RRF is only
            # reordering the same set.
            sets = {
                "vector": {h.id for h in vector_hits},
                "lexical": {h.id for h in lexical_hits},
                "identifier": {h.id for h in ident_hits},
            }
            nonempty = [s for s in sets.values() if s]
            union = set().union(*nonempty) if nonempty else set()
            all_of = set.intersection(*nonempty) if len(nonempty) > 1 else set()
            counts = ", ".join(f"{len(v)} {k}" for k, v in sets.items())
            print(f"  candidates: {counts}, {len(all_of)} in all, {len(union)} union")

            # How far the reranker moved things. A large reordering means the
            # fused ranking was poorly separated in that band; no movement
            # means fusion had already got it right.
            fused_ids = [h.id for h in capped[:5]]
            rerank_ids = [h.id for h in reranked]
            moved = sum(1 for i, cid in enumerate(rerank_ids)
                        if i >= len(fused_ids) or fused_ids[i] != cid)
            print(f"  rerank changed {moved}/{len(rerank_ids)} top positions")
            print()
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())