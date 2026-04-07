#!/usr/bin/env python3
"""
bench_backends.py — Compare ChromaDB vs PostgreSQL backend for MemPalace

Ingests the same documents into both backends, runs identical search queries,
and compares retrieval quality and performance.
"""

import os
import sys
import time
import shutil
import hashlib
import json
from pathlib import Path
from collections import defaultdict

# Add mempalace to path
sys.path.insert(0, os.path.dirname(__file__))

# ── Config ──
DOCS_DIR = os.path.expanduser("~/Projects/C/clustered_pg/docs")
MEMORY_DIR = os.path.expanduser(
    "~/.claude/projects/-Users-sergey-Projects-C-clustered-pg/memory"
)
PALACE_CHROMA = "/tmp/mempalace_bench_chroma"
PG_DSN = "host=/tmp port=65499 dbname=mempalace_test"

SEARCH_QUERIES = [
    "HNSW index performance benchmarks",
    "GraphRAG multi-hop traversal",
    "zone map pruning",
    "FlashHadamard int16 NEON kernel",
    "pg_dump restore limitation",
    "SQ8 quantization cache",
    "how does sorted heap storage work",
    "IVF-PQ approximate nearest neighbor",
    "concurrent UPDATE and online compact",
    "Qdrant vs pgvector comparison",
]

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
WING = "clustered_pg"


def chunk_file(filepath):
    """Chunk a file into drawer-sized pieces."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return []

    if len(content) < 50:
        return []

    chunks = []
    start = 0
    idx = 0
    while start < len(content):
        end = min(start + CHUNK_SIZE, len(content))
        if end < len(content):
            nl = content.rfind("\n\n", start, end)
            if nl > start + CHUNK_SIZE // 2:
                end = nl
            else:
                nl = content.rfind("\n", start, end)
                if nl > start + CHUNK_SIZE // 2:
                    end = nl
        chunk = content[start:end].strip()
        if len(chunk) >= 50:
            room = detect_room(chunk, filepath.stem)
            drawer_id = f"drawer_{WING}_{room}_{hashlib.md5((str(filepath) + str(idx)).encode()).hexdigest()[:16]}"
            chunks.append(
                {
                    "id": drawer_id,
                    "content": chunk,
                    "wing": WING,
                    "room": room,
                    "source_file": str(filepath),
                    "chunk_index": idx,
                }
            )
            idx += 1
        start = end - CHUNK_OVERLAP if end < len(content) else end

    return chunks


def detect_room(content, filename):
    """Simple room detection based on keywords."""
    text = (content[:2000] + " " + filename).lower()
    scores = {
        "benchmarks": sum(
            1 for w in ["benchmark", "latency", "recall", "throughput", "p50", "ms"]
            if w in text
        ),
        "architecture": sum(
            1
            for w in ["architecture", "design", "storage", "heap", "page", "zone"]
            if w in text
        ),
        "graphrag": sum(
            1
            for w in ["graphrag", "graph_rag", "multi-hop", "traversal", "entity"]
            if w in text
        ),
        "hnsw": sum(
            1
            for w in ["hnsw", "index", "cache", "scan", "sorted_hnsw", "sq8"]
            if w in text
        ),
        "flashhadamard": sum(
            1
            for w in [
                "flashhadamard",
                "hadamard",
                "neon",
                "quantiz",
                "kernel",
                "int16",
            ]
            if w in text
        ),
    }
    if scores:
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best
    return "general"


def collect_files():
    """Collect all .md files from docs and memory dirs."""
    files = []
    for d in [DOCS_DIR, MEMORY_DIR]:
        p = Path(d)
        if p.exists():
            files.extend(p.glob("*.md"))
    return files


def precompute_embeddings(chunks, queries):
    """Pre-compute embeddings for all documents and queries using sentence-transformers.
    This ensures both backends use identical vectors for fair comparison."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    print("  Pre-computing embeddings (sentence-transformers)...")
    t0 = time.time()
    doc_texts = [ch["content"] for ch in chunks]
    doc_embs = model.encode(doc_texts, normalize_embeddings=True, show_progress_bar=False)
    query_embs = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
    t_emb = time.time() - t0
    print(f"    {len(doc_texts)} docs + {len(queries)} queries in {t_emb:.2f}s")
    return doc_embs.tolist(), query_embs.tolist()


EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def ingest_chromadb(chunks, doc_embeddings):
    """Ingest chunks with pre-computed embeddings into ChromaDB (cosine distance)."""
    import chromadb

    if os.path.exists(PALACE_CHROMA):
        shutil.rmtree(PALACE_CHROMA)
    os.makedirs(PALACE_CHROMA)

    client = chromadb.PersistentClient(path=PALACE_CHROMA)
    col = client.create_collection("mempalace_drawers", metadata={"hnsw:space": "cosine"})

    t0 = time.time()
    # Batch insert
    batch = 100
    for i in range(0, len(chunks), batch):
        b = chunks[i : i + batch]
        e = doc_embeddings[i : i + batch]
        col.add(
            documents=[ch["content"] for ch in b],
            ids=[ch["id"] for ch in b],
            embeddings=e,
            metadatas=[
                {
                    "wing": ch["wing"],
                    "room": ch["room"],
                    "source_file": ch["source_file"],
                    "chunk_index": ch["chunk_index"],
                }
                for ch in b
            ],
        )
    t_ingest = time.time() - t0
    return col, t_ingest, col.count()


def ingest_pg(chunks, doc_embeddings):
    """Ingest chunks with pre-computed embeddings into PostgreSQL."""
    from mempalace.pg_backend import PGCollection
    import psycopg2

    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS mempalace_drawers CASCADE")
    conn.close()

    col = PGCollection(PG_DSN)

    t0 = time.time()
    batch = 50
    for i in range(0, len(chunks), batch):
        b = chunks[i : i + batch]
        e = doc_embeddings[i : i + batch]
        col.add(
            documents=[ch["content"] for ch in b],
            ids=[ch["id"] for ch in b],
            embeddings=e,
            metadatas=[
                {
                    "wing": ch["wing"],
                    "room": ch["room"],
                    "source_file": ch["source_file"],
                    "chunk_index": ch["chunk_index"],
                }
                for ch in b
            ],
        )
    t_ingest = time.time() - t0
    return col, t_ingest, col.count()


def search_backend(col, queries, query_embeddings, n_results=5):
    """Run search queries with pre-computed embeddings."""
    results = []
    total_time = 0

    for q, qe in zip(queries, query_embeddings):
        t0 = time.time()
        res = col.query(
            query_embeddings=[qe],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        t_q = time.time() - t0
        total_time += t_q

        hits = []
        if res["ids"] and res["ids"][0]:
            for j, doc_id in enumerate(res["ids"][0]):
                hits.append(
                    {
                        "id": doc_id,
                        "distance": res["distances"][0][j],
                        "similarity": round(1 - res["distances"][0][j], 4),
                        "wing": res["metadatas"][0][j].get("wing", ""),
                        "room": res["metadatas"][0][j].get("room", ""),
                        "snippet": res["documents"][0][j][:120].replace("\n", " "),
                    }
                )
        results.append({"query": q, "time_ms": round(t_q * 1000, 1), "hits": hits})

    return results, total_time


def compare_results(chroma_results, pg_results):
    """Compare search results between backends."""
    print(f"\n{'=' * 80}")
    print("  SEARCH COMPARISON: ChromaDB vs PostgreSQL")
    print(f"{'=' * 80}\n")

    total_overlap = 0
    total_hits = 0

    for cr, pr in zip(chroma_results, pg_results):
        query = cr["query"]
        chroma_ids = set(h["id"] for h in cr["hits"])
        pg_ids = set(h["id"] for h in pr["hits"])
        overlap = len(chroma_ids & pg_ids)
        total_overlap += overlap
        total_hits += max(len(chroma_ids), len(pg_ids))

        print(f'  Q: "{query}"')
        print(
            f'    ChromaDB:    {cr["time_ms"]:6.1f}ms  top={cr["hits"][0]["similarity"]:.4f}  room={cr["hits"][0]["room"]}'
            if cr["hits"]
            else f'    ChromaDB:    {cr["time_ms"]:6.1f}ms  (no results)'
        )
        print(
            f'    PostgreSQL:  {pr["time_ms"]:6.1f}ms  top={pr["hits"][0]["similarity"]:.4f}  room={pr["hits"][0]["room"]}'
            if pr["hits"]
            else f'    PostgreSQL:  {pr["time_ms"]:6.1f}ms  (no results)'
        )
        print(f"    Overlap:     {overlap}/{max(len(chroma_ids), len(pg_ids), 1)} results match")
        print()

    overlap_pct = 100 * total_overlap / total_hits if total_hits else 0
    print(f"  Overall result overlap: {total_overlap}/{total_hits} ({overlap_pct:.1f}%)")


def main():
    print("MemPalace Backend Benchmark")
    print("=" * 50)

    # Collect and chunk files
    files = collect_files()
    print(f"  Files found: {len(files)}")

    all_chunks = []
    for f in files:
        all_chunks.extend(chunk_file(f))
    print(f"  Total chunks: {len(all_chunks)}")

    room_counts = defaultdict(int)
    for ch in all_chunks:
        room_counts[ch["room"]] += 1
    for room, cnt in sorted(room_counts.items(), key=lambda x: -x[1]):
        print(f"    {room:20} {cnt:4} chunks")

    # Pre-compute embeddings (same vectors for both backends)
    print(f"\n{'─' * 50}")
    doc_embs, query_embs = precompute_embeddings(all_chunks, SEARCH_QUERIES)

    # Ingest into ChromaDB
    print("\n  Ingesting into ChromaDB (cosine, pre-computed embeddings)...")
    chroma_col, chroma_t, chroma_n = ingest_chromadb(all_chunks, doc_embs)
    print(f"    {chroma_n} drawers in {chroma_t:.2f}s ({chroma_n / chroma_t:.0f} docs/s)")

    # Ingest into PostgreSQL
    print("\n  Ingesting into PostgreSQL (pre-computed embeddings)...")
    pg_col, pg_t, pg_n = ingest_pg(all_chunks, doc_embs)
    print(f"    {pg_n} drawers in {pg_t:.2f}s ({pg_n / pg_t:.0f} docs/s)")

    # Search both (warm up first)
    print(f"\n{'─' * 50}")
    print(f"  Warming up...")
    search_backend(chroma_col, SEARCH_QUERIES[:1], query_embs[:1])
    search_backend(pg_col, SEARCH_QUERIES[:1], query_embs[:1])

    print(f"  Running {len(SEARCH_QUERIES)} search queries on each backend...")
    chroma_results, chroma_total = search_backend(chroma_col, SEARCH_QUERIES, query_embs)
    pg_results, pg_total = search_backend(pg_col, SEARCH_QUERIES, query_embs)

    # Compare
    compare_results(chroma_results, pg_results)

    # Summary
    print(f"\n{'=' * 80}")
    print("  SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Documents:       {len(all_chunks)}")
    print(f"  Queries:         {len(SEARCH_QUERIES)}")
    print()
    print(f"  Ingest time:     ChromaDB={chroma_t:.2f}s  PG={pg_t:.2f}s  ({pg_t / chroma_t:.1f}x)")
    print(f"  Search total:    ChromaDB={chroma_total * 1000:.1f}ms  PG={pg_total * 1000:.1f}ms")
    print(
        f"  Search per-query: ChromaDB={chroma_total * 1000 / len(SEARCH_QUERIES):.1f}ms  "
        f"PG={pg_total * 1000 / len(SEARCH_QUERIES):.1f}ms"
    )

    # Detect PG backend type
    if pg_col._vec_type == "svec":
        print(f"\n  PG backend: pg_sorted_heap (svec + sorted_hnsw + zone maps)")
    else:
        print(f"\n  PG backend: pgvector (vector + hnsw)")

    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
