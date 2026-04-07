"""
pg_backend.py — PostgreSQL backend for MemPalace
=================================================

Drop-in replacement for ChromaDB using PostgreSQL with pgvector or
pg_sorted_heap for vector storage and search.

Benefits over ChromaDB:
  - ACID transactions (no corruption on crash)
  - SQL access to all memories (JOIN, aggregate, export)
  - Better performance at scale (100K+ drawers)
  - Zone map filtering with pg_sorted_heap (wing/room pruning for free)
  - HNSW index with SQ8 quantization (pg_sorted_heap)

Requires:
  - PostgreSQL 15+ with pgvector OR pg_sorted_heap extension
  - psycopg2
  - sentence-transformers (for embedding generation)

Usage:
  Set in ~/.mempalace/config.json:
    {
      "backend": "postgresql",
      "pg_dsn": "host=localhost port=5432 dbname=mempalace"
    }
"""

import json
import logging

logger = logging.getLogger("mempalace.pg")

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
    return _embedder


def _embed(texts):
    """Embed a list of texts into vectors. Returns list of lists."""
    model = _get_embedder()
    vecs = model.encode(texts, normalize_embeddings=True)
    return vecs.tolist()


def _vec_literal(vec):
    """Convert a vector (list of floats) to PostgreSQL literal string."""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


class PGCollection:
    """
    Drop-in replacement for ChromaDB Collection backed by PostgreSQL.

    Supports both pgvector (vector type + hnsw) and pg_sorted_heap
    (svec type + sorted_hnsw with zone map pruning on wing/room).
    Auto-detects which extension is available, preferring pg_sorted_heap.
    """

    def __init__(self, dsn, table_name="mempalace_drawers"):
        self.dsn = dsn
        self.table = table_name
        self._conn = None
        self._vec_type = None  # 'svec' or 'vector'
        self._am = None  # 'sorted_heap' or 'heap'
        self._index_am = None  # 'sorted_hnsw' or 'hnsw'
        self._setup_done = False

    def _get_conn(self):
        import psycopg2

        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.dsn)
            self._conn.autocommit = True
        return self._conn

    def _ensure_setup(self):
        if self._setup_done:
            return
        conn = self._get_conn()
        cur = conn.cursor()

        # Detect installed extensions
        cur.execute(
            "SELECT extname FROM pg_extension WHERE extname IN ('pg_sorted_heap', 'vector')"
        )
        installed = {row[0] for row in cur.fetchall()}

        if "pg_sorted_heap" in installed:
            self._vec_type = "svec"
            self._am = "sorted_heap"
            self._index_am = "sorted_hnsw"
            logger.info("Using pg_sorted_heap backend (svec + sorted_hnsw + zone maps)")
        elif "vector" in installed:
            self._vec_type = "vector"
            self._am = "heap"
            self._index_am = "hnsw"
            logger.info("Using pgvector backend (vector + hnsw)")
        else:
            # Try to create extensions
            for ext, vt, am, iam in [
                ("pg_sorted_heap", "svec", "sorted_heap", "sorted_hnsw"),
                ("vector", "vector", "heap", "hnsw"),
            ]:
                try:
                    cur.execute(f"CREATE EXTENSION IF NOT EXISTS {ext}")
                    self._vec_type = vt
                    self._am = am
                    self._index_am = iam
                    logger.info(f"Created extension {ext}")
                    break
                except Exception:
                    pass

            if not self._vec_type:
                raise RuntimeError(
                    "PostgreSQL backend requires pgvector or pg_sorted_heap. "
                    "Install: CREATE EXTENSION vector; or CREATE EXTENSION pg_sorted_heap;"
                )

        self._create_table(cur)
        self._setup_done = True

    def _create_table(self, cur):
        """Create table and indexes if they don't exist."""
        cur.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = %s AND table_schema = 'public'",
            (self.table,),
        )
        if cur.fetchone():
            return

        vec_type = f"{self._vec_type}({EMBEDDING_DIM})"
        using = f" USING {self._am}" if self._am != "heap" else ""

        # For sorted_heap: wing, room first for zone map pruning
        if self._am == "sorted_heap":
            cur.execute(
                f"""
                CREATE TABLE {self.table} (
                    wing text NOT NULL DEFAULT '',
                    room text NOT NULL DEFAULT '',
                    id text NOT NULL,
                    document text NOT NULL,
                    embedding {vec_type},
                    metadata jsonb DEFAULT '{{}}'
                ){using}
            """
            )
            # btree index on id for lookups
            cur.execute(
                f"CREATE INDEX {self.table}_id_idx ON {self.table} USING btree (id)"
            )
        else:
            cur.execute(
                f"""
                CREATE TABLE {self.table} (
                    id text PRIMARY KEY,
                    wing text NOT NULL DEFAULT '',
                    room text NOT NULL DEFAULT '',
                    document text NOT NULL,
                    embedding {vec_type},
                    metadata jsonb DEFAULT '{{}}'
                )
            """
            )
            cur.execute(f"CREATE INDEX {self.table}_wing_idx ON {self.table} (wing)")
            cur.execute(f"CREATE INDEX {self.table}_room_idx ON {self.table} (room)")

        # Vector index — skip for small collections, add later via ensure_vector_index()
        # Exact cosine search is fast enough for < 5K rows; avoids HNSW quality
        # issues on very small graphs.
        logger.info(f"Created table {self.table} ({self._am}, {self._vec_type})")

    def ensure_vector_index(self):
        """Create HNSW vector index if it doesn't exist. For large collections (5K+)."""
        self._ensure_setup()
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM pg_indexes WHERE indexname = %s",
            (f"{self.table}_vec_idx",),
        )
        if cur.fetchone():
            return
        count = self.count()
        if count < 5000:
            logger.info(
                f"Skipping HNSW index for {count} rows (exact cosine is fast enough)"
            )
            return
        ops = "svec_cosine_ops" if self._vec_type == "svec" else "vector_cosine_ops"
        cur.execute(
            f"CREATE INDEX {self.table}_vec_idx ON {self.table} "
            f"USING {self._index_am} (embedding {ops})"
        )
        logger.info(f"Created HNSW index ({self._index_am}, {count} rows)")

    def _where_to_sql(self, where):
        """Convert ChromaDB where filter to SQL WHERE clause + params."""
        if not where:
            return "", []

        if "$and" in where:
            parts = []
            params = []
            for cond in where["$and"]:
                c, p = self._where_to_sql(cond)
                if c:
                    parts.append(f"({c})")
                    params.extend(p)
            return " AND ".join(parts), params

        clauses = []
        params = []
        for key, val in where.items():
            if key.startswith("$"):
                continue
            if key in ("wing", "room"):
                clauses.append(f"{key} = %s")
                params.append(str(val))
            else:
                clauses.append(f"metadata->>%s = %s")
                params.extend([key, str(val)])

        return " AND ".join(clauses), params

    def _meta_dict(self, wing, room, metadata_json):
        """Reconstruct ChromaDB-compatible metadata dict."""
        meta = metadata_json if isinstance(metadata_json, dict) else {}
        meta["wing"] = wing
        meta["room"] = room
        return meta

    # ── ChromaDB-compatible interface ──

    def add(self, documents, ids, metadatas=None, embeddings=None):
        """Add documents with embeddings and metadata."""
        self._ensure_setup()
        conn = self._get_conn()
        cur = conn.cursor()

        if embeddings is None:
            embeddings = _embed(documents)

        for i, (doc_id, doc) in enumerate(zip(ids, documents)):
            meta = dict(metadatas[i]) if metadatas and i < len(metadatas) else {}
            wing = meta.pop("wing", "")
            room = meta.pop("room", "")
            emb_str = _vec_literal(embeddings[i])

            if self._am == "sorted_heap":
                # No ON CONFLICT for sorted_heap — check first
                cur.execute(
                    f"SELECT 1 FROM {self.table} WHERE id = %s LIMIT 1", (doc_id,)
                )
                if cur.fetchone():
                    continue
                cur.execute(
                    f"INSERT INTO {self.table} (wing, room, id, document, embedding, metadata) "
                    f"VALUES (%s, %s, %s, %s, %s::{self._vec_type}, %s::jsonb)",
                    (wing, room, doc_id, doc, emb_str, json.dumps(meta)),
                )
            else:
                cur.execute(
                    f"INSERT INTO {self.table} (id, wing, room, document, embedding, metadata) "
                    f"VALUES (%s, %s, %s, %s, %s::{self._vec_type}, %s::jsonb) "
                    f"ON CONFLICT (id) DO NOTHING",
                    (doc_id, wing, room, doc, emb_str, json.dumps(meta)),
                )

    def query(self, query_texts=None, query_embeddings=None, n_results=5, where=None, include=None):
        """Semantic search — returns results in ChromaDB format."""
        self._ensure_setup()
        conn = self._get_conn()
        cur = conn.cursor()

        if query_embeddings:
            query_emb = query_embeddings[0]
        elif query_texts:
            query_emb = _embed(query_texts[:1])[0]
        else:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        emb_str = _vec_literal(query_emb)

        where_sql, where_params = self._where_to_sql(where)
        where_clause = f"WHERE {where_sql}" if where_sql else ""

        cast = f"::{self._vec_type}"
        sql = (
            f"SELECT id, document, wing, room, metadata, "
            f"embedding <=> %s{cast} AS distance "
            f"FROM {self.table} {where_clause} "
            f"ORDER BY embedding <=> %s{cast} "
            f"LIMIT %s"
        )
        params = [emb_str] + where_params + [emb_str, n_results]
        cur.execute(sql, params)
        rows = cur.fetchall()

        result_ids = []
        result_docs = []
        result_metas = []
        result_dists = []
        for row in rows:
            rid, doc, wing, room, meta_json, dist = row
            result_ids.append(rid)
            result_docs.append(doc)
            result_metas.append(self._meta_dict(wing, room, meta_json))
            result_dists.append(float(dist))

        return {
            "ids": [result_ids],
            "documents": [result_docs],
            "metadatas": [result_metas],
            "distances": [result_dists],
        }

    def get(self, ids=None, where=None, limit=None, include=None):
        """Get documents by ID or metadata filter (no semantic search)."""
        self._ensure_setup()
        conn = self._get_conn()
        cur = conn.cursor()

        clauses = []
        params = []

        if ids:
            placeholders = ",".join(["%s"] * len(ids))
            clauses.append(f"id IN ({placeholders})")
            params.extend(ids)

        if where:
            w_sql, w_params = self._where_to_sql(where)
            if w_sql:
                clauses.append(w_sql)
                params.extend(w_params)

        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_clause = f"LIMIT {int(limit)}" if limit else ""

        cur.execute(
            f"SELECT id, document, wing, room, metadata "
            f"FROM {self.table} {where_clause} {limit_clause}",
            params,
        )
        rows = cur.fetchall()

        result = {
            "ids": [r[0] for r in rows],
            "documents": [r[1] for r in rows] if (not include or "documents" in include) else None,
            "metadatas": [self._meta_dict(r[2], r[3], r[4]) for r in rows]
            if (not include or "metadatas" in include)
            else None,
        }
        return result

    def count(self):
        """Return total number of documents."""
        self._ensure_setup()
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {self.table}")
        return cur.fetchone()[0]

    def delete(self, ids):
        """Delete documents by ID."""
        self._ensure_setup()
        conn = self._get_conn()
        cur = conn.cursor()
        placeholders = ",".join(["%s"] * len(ids))
        cur.execute(f"DELETE FROM {self.table} WHERE id IN ({placeholders})", ids)

    def get_or_create(self):
        """Ensure table exists and return self. ChromaDB compat."""
        self._ensure_setup()
        return self


class PGClient:
    """Drop-in replacement for chromadb.PersistentClient."""

    def __init__(self, dsn):
        self.dsn = dsn
        self._collections = {}

    def get_collection(self, name):
        if name not in self._collections:
            col = PGCollection(self.dsn, table_name=name)
            col._ensure_setup()
            self._collections[name] = col
        return self._collections[name]

    def get_or_create_collection(self, name):
        if name not in self._collections:
            self._collections[name] = PGCollection(self.dsn, table_name=name)
        col = self._collections[name]
        col._ensure_setup()
        return col

    def create_collection(self, name):
        return self.get_or_create_collection(name)
