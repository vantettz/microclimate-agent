import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from corpus.hybrid_search import HybridRetriever

_encoder = None
_index = None
_corpus = None
_retriever = None

def _load_resources():
    global _encoder, _index, _corpus
    if _encoder is None:
        _encoder = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    if _index is None:
        _index = faiss.read_index("corpus/index.faiss")
    if _corpus is None:
        with open("corpus/papers.jsonl", "r", encoding="utf-8") as f:
            _corpus = [json.loads(line) for line in f]

def search_literature(query: str, topk: int = 3) -> str:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever.search(query, topk)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        passage = _corpus[idx]
        results.append(
            f"[{rank+1}] 来源：{passage.get('source', '未知')}\n"
            f"{passage.get('content', '')}"
        )

    if not results:
        return "未检索到相关文献"

    return "\n\n".join(results)