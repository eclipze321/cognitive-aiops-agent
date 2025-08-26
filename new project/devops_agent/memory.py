"""Simple in-process memory & similarity search (placeholder).

Copyright (c) 2025 Patrick Morrison. Licensed under the MIT License.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from typing import List, Tuple

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT,
    content TEXT,
    embedding TEXT
);
CREATE INDEX IF NOT EXISTS idx_memory_kind ON memory(kind);
"""


def _hash_embed(text: str, dims: int = 48) -> List[float]:
    h = hashlib.sha256(text.encode('utf-8')).hexdigest()
    seg = max(1, len(h) // dims)
    vals = []
    for i in range(0, len(h), seg):
        chunk = h[i:i+seg]
        if not chunk:
            continue
        vals.append(int(chunk, 16) % 1000 / 1000.0)
        if len(vals) == dims:
            break
    if len(vals) < dims:
        vals.extend([0.0]*(dims-len(vals)))
    return vals


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)) or 1.0
    nb = math.sqrt(sum(x*x for x in b)) or 1.0
    return dot/(na*nb)


class MemoryStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        cur = self.conn.cursor()
        cur.executescript(SCHEMA)
        self.conn.commit()

    def add(self, kind: str, content: str):
        emb = _hash_embed(content)
        cur = self.conn.cursor()
        cur.execute('INSERT INTO memory (kind, content, embedding) VALUES (?, ?, ?)', (kind, content, json.dumps(emb)))
        self.conn.commit()

    def similar(self, kind: str, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        q_emb = _hash_embed(query)
        cur = self.conn.cursor()
        cur.execute('SELECT content, embedding FROM memory WHERE kind=?', (kind,))
        scored: List[Tuple[str,float]] = []
        for content, emb_json in cur.fetchall():
            try:
                emb = json.loads(emb_json)
            except Exception:
                continue
            score = _cosine(q_emb, emb)
            scored.append((content, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

__all__ = ['MemoryStore']
