"""Embeddings — OpenAI in production, a deterministic FakeEmbedder when TESTING=1.

The fake produces stable pseudo-vectors from token hashes, so similarity search behaves
consistently in tests with no API key and no network.
"""
import hashlib
import math
import os
from typing import List

TESTING = os.getenv("TESTING") == "1"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
_FAKE_DIM = 64


def _fake_embed(text: str) -> List[float]:
    vec = [0.0] * _FAKE_DIM
    for word in text.lower().split():
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % _FAKE_DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def get_embeddings(texts: List[str]) -> List[List[float]]:
    if TESTING:
        return [_fake_embed(t) for t in texts]
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def get_embedding(text: str) -> List[float]:
    return get_embeddings([text])[0]


def cosine(a: List[float], b: List[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (da * db)
