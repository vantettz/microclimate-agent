# corpus/hybrid_search.py
import json
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

class HybridRetriever:
    def __init__(self):
        # 密集检索资源
        self.encoder = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.dense_index = faiss.read_index("corpus/index.faiss")
        
        # 加载语料库
        with open("corpus/papers.jsonl", "r", encoding="utf-8") as f:
            self.corpus = [json.loads(line) for line in f]
        
        # 稀疏检索资源（BM25）
        tokenized = [doc["content"].split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized)

    def dense_search(self, query: str, topk: int = 10):
        """密集检索，返回(idx, score)列表"""
        vec = self.encoder.encode([query], normalize_embeddings=True)
        vec = np.array(vec, dtype="float32")
        distances, indices = self.dense_index.search(vec, topk)
        return list(zip(indices[0], distances[0]))

    def sparse_search(self, query: str, topk: int = 10):
        """稀疏检索BM25，返回(idx, score)列表"""
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:topk]
        return [(idx, scores[idx]) for idx in top_indices]

    def rrf_fusion(self, dense_results, sparse_results, k=60):
        """
        RRF融合：每个文档的最终分数 = sum(1/(k + rank))
        k=60是经验值，防止排名靠前的文档得分过高
        """
        scores = {}
        
        for rank, (idx, _) in enumerate(dense_results):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
        
        for rank, (idx, _) in enumerate(sparse_results):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
        
        # 按融合分数排序
        sorted_results = sorted(scores.items(), 
                                key=lambda x: x[1], reverse=True)
        return sorted_results

    def search(self, query: str, topk: int = 3) -> str:
        """混合检索主入口"""
        dense_results  = self.dense_search(query, topk=10)
        sparse_results = self.sparse_search(query, topk=10)
        fused = self.rrf_fusion(dense_results, sparse_results)
        
        results = []
        for idx, score in fused[:topk]:
            if idx == -1 or idx >= len(self.corpus):
                continue
            passage = self.corpus[idx]
            results.append(
                f"[来源：{passage.get('source', '未知')}]\n"
                f"{passage.get('content', '')}"
            )
        
        return "\n\n".join(results) if results else "未检索到相关文献"