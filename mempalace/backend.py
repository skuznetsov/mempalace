"""
backend.py — Backend factory for MemPalace
============================================

Returns either a ChromaDB or PostgreSQL collection based on config.
All other modules use this instead of importing chromadb directly.
"""

import os


def get_collection(palace_path=None, create=False):
    """
    Return a collection object (ChromaDB or PostgreSQL) based on config.

    The returned object has the same interface regardless of backend:
      - add(documents, ids, metadatas)
      - query(query_texts, n_results, where, include)
      - get(ids, where, limit, include)
      - count()
      - delete(ids)
    """
    from .config import MempalaceConfig

    cfg = MempalaceConfig()

    if cfg.backend == "postgresql":
        from .pg_backend import PGClient

        client = PGClient(cfg.pg_dsn)
        if create:
            return client.get_or_create_collection(cfg.collection_name)
        return client.get_collection(cfg.collection_name)

    # Default: ChromaDB
    import chromadb

    palace = palace_path or cfg.palace_path
    os.makedirs(palace, exist_ok=True)
    client = chromadb.PersistentClient(path=palace)
    if create:
        try:
            return client.get_collection(cfg.collection_name)
        except Exception:
            return client.create_collection(cfg.collection_name)
    return client.get_collection(cfg.collection_name)
