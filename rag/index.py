"""Hybrid retrieval: BM25 keyword search + dense vector similarity, merged with RRF.

* BM25 (rank_bm25) keeps exact identifiers (error codes, hostnames, x509, errno=28)
  searchable: the tokenizer keeps joined tokens AND their parts.
* Vector side: by default LSA vectors (TF-IDF -> TruncatedSVD, L2-normalised), which
  are deterministic and need no model download. If EMBEDDER=sentence-transformers is
  set in .env and the package is installed, a small sentence-transformers model is used.
* Reciprocal rank fusion (RRF, k=60) merges the two rankings.
Every document has an id, so answers can cite it.
"""
from __future__ import annotations

import os
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

_TOK = re.compile(r"[a-z0-9]+(?:[._:/=-][a-z0-9]+)*")
RRF_K = 60


def tokenize(text: str) -> list[str]:
    toks = []
    for t in _TOK.findall(text.lower()):
        toks.append(t)
        parts = re.split(r"[._:/=-]", t)
        if len(parts) > 1:
            toks.extend(p for p in parts if p)
    return toks


class _LSAEmbedder:
    name = "lsa-tfidf-svd"

    def __init__(self, dim: int = 128, seed: int = 42):
        self.dim, self.seed = dim, seed

    def fit(self, texts: list[str]):
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                   sublinear_tf=True, min_df=1)
        x = self.vec.fit_transform(texts)
        dim = max(2, min(self.dim, x.shape[1] - 1, x.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=dim, random_state=self.seed).fit(x)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        z = self.svd.transform(self.vec.transform(texts))
        return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


class _STEmbedder:  # optional upgrade, only used if explicitly enabled
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.name, self.m = model, SentenceTransformer(model)

    def fit(self, texts):
        return self

    def encode(self, texts):
        return np.asarray(self.m.encode(list(texts), normalize_embeddings=True))


def make_embedder():
    if os.getenv("EMBEDDER", "lsa").lower() == "sentence-transformers":
        try:
            return _STEmbedder()
        except Exception as e:
            print(f"[rag] sentence-transformers unavailable ({e}); using LSA vectors")
    return _LSAEmbedder()


class HybridIndex:
    """A small in-memory hybrid index over documents {id, text, meta}."""

    def __init__(self, docs: list[dict], name: str = "index"):
        if not docs:
            raise ValueError("cannot build an empty index")
        ids = [d["id"] for d in docs]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate ids in {name}")
        self.name, self.docs = name, docs
        self.pos = {d["id"]: i for i, d in enumerate(docs)}
        texts = [d["text"] for d in docs]
        self.bm25 = BM25Okapi([tokenize(t) or ["_empty_"] for t in texts])
        self.embedder = make_embedder().fit(texts)
        self.emb = self.embedder.encode(texts)

    def __len__(self):
        return len(self.docs)

    def get(self, doc_id: str) -> dict | None:
        i = self.pos.get(doc_id)
        return None if i is None else self.docs[i]

    def search(self, query: str, k: int = 5, mode: str = "hybrid",
               exclude_ids: set[str] | None = None, where: dict | None = None,
               depth: int = 100) -> list[dict]:
        """Return top-k hits with rank details and cosine similarity.

        mode: "bm25" | "vector" | "hybrid".  `where` filters on meta key == value.
        """
        exclude_ids = exclude_ids or set()
        ok = np.array([d["id"] not in exclude_ids and
                       all(d["meta"].get(a) == b for a, b in (where or {}).items())
                       for d in self.docs])
        if not ok.any():
            return []
        qv = self.embedder.encode([query])[0]
        cos = self.emb @ qv
        bm = np.asarray(self.bm25.get_scores(tokenize(query) or ["_empty_"]))
        bm_rank = _ranks(np.where(ok, bm, -np.inf), depth)
        vec_rank = _ranks(np.where(ok, cos, -np.inf), depth)
        if mode == "bm25":
            fused = {i: 1.0 / (RRF_K + r) for i, r in bm_rank.items() if bm[i] > 0}
        elif mode == "vector":
            fused = {i: 1.0 / (RRF_K + r) for i, r in vec_rank.items()}
        elif mode == "hybrid":
            fused = {}
            for i, r in bm_rank.items():
                if bm[i] > 0:
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r)
            for i, r in vec_rank.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        return [{"id": self.docs[i]["id"], "score": round(fused[i], 6),
                 "similarity": round(float(max(cos[i], 0.0)), 4),
                 "bm25": round(float(bm[i]), 3), "bm25_rank": bm_rank.get(i),
                 "vector_rank": vec_rank.get(i), "meta": self.docs[i]["meta"],
                 "text": self.docs[i]["text"]} for i in top]


def _ranks(scores: np.ndarray, depth: int) -> dict[int, int]:
    finite = np.isfinite(scores)
    n = int(min(depth, finite.sum()))
    if n == 0:
        return {}
    idx = np.argpartition(-np.where(finite, scores, -1e18), n - 1)[:n]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return {int(i): r + 1 for r, i in enumerate(idx)}
