"""
混合检索引擎 - BM25关键词检索 + 向量语义检索 + RRF融合 + 重排序
智能知识库Agent的核心检索模块
"""
import numpy as np
from typing import Optional
from collections import defaultdict
import math


class BM25Retriever:
    """BM25关键词检索器 - 精确匹配, 弥补向量检索的不足"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.documents: list[str] = []
        self._tokenized: list[list[str]] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0

    def _tokenize(self, text: str) -> list[str]:
        """中文分词: 按字符粒度切分 (生产环境应用jieba)"""
        tokens = []
        for ch in text:
            if '一' <= ch <= '鿿' or ch.isalnum():
                tokens.append(ch)
        return tokens

    def index(self, documents: list[str]) -> None:
        """构建BM25索引"""
        self.documents = documents
        self._tokenized = [self._tokenize(d) for d in documents]
        doc_lens = [len(t) for t in self._tokenized]
        self._avgdl = np.mean(doc_lens) if doc_lens else 1.0
        n = len(documents)
        df = defaultdict(int)
        for tokens in self._tokenized:
            for w in set(tokens):
                df[w] += 1
        self._idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """BM25检索, 返回 (文档索引, 分数) 列表"""
        if not self.documents:
            return []
        qt = self._tokenize(query)
        scores = []
        for i, dt in enumerate(self._tokenized):
            score = 0.0
            dl = len(dt)
            for w in qt:
                if w in self._idf:
                    tf = dt.count(w)
                    score += self._idf[w] * tf * (self.k1 + 1) / (
                        tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class VectorRetriever:
    """向量检索器 - 语义匹配"""

    def __init__(self, embedding_model=None):
        self._model = embedding_model
        self._embeddings: Optional[np.ndarray] = None
        self._documents: list[str] = []

    def index(self, documents: list[str], embeddings: Optional[np.ndarray] = None) -> None:
        """构建向量索引"""
        self._documents = documents
        if embeddings is not None:
            self._embeddings = embeddings
        elif self._model:
            self._embeddings = self._model.encode(documents, normalize_embeddings=True)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """向量检索"""
        if self._embeddings is None:
            return []
        if self._model:
            qv = self._model.encode([query], normalize_embeddings=True)[0]
        else:
            return []
        scores = np.dot(self._embeddings, qv)
        indices = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in indices if scores[i] > 0]


class RRF:
    """Reciprocal Rank Fusion - 无需调参的排名融合"""

    @staticmethod
    def fuse(rankings: list[list[tuple[int, float]]], k: int = 60, top_n: int = 10) -> list[tuple[int, float]]:
        """融合多个排名列表"""
        scores = defaultdict(float)
        for ranking in rankings:
            for rank, (doc_id, _) in enumerate(ranking):
                scores[doc_id] += 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


class Reranker:
    """重排序器 - 用Embedding模型对候选文档精细排序"""

    def __init__(self, embedding_model=None):
        self._model = embedding_model

    def rerank(self, query: str, candidates: list[tuple[int, str]], top_k: int = 5) -> list[tuple[int, float]]:
        """对候选文档重排序"""
        if not self._model or not candidates:
            return [(cid, 0.0) for cid, _ in candidates[:top_k]]
        qv = self._model.encode([query], normalize_embeddings=True)[0]
        doc_texts = [text for _, text in candidates]
        dvs = self._model.encode(doc_texts, normalize_embeddings=True)
        scores = np.dot(dvs, qv)
        ranked = [(candidates[i][0], float(scores[i])) for i in np.argsort(scores)[::-1]]
        return ranked[:top_k]


class HybridSearcher:
    """混合检索引擎 - 组合BM25+向量+RRF+重排序"""

    def __init__(self, embedding_model=None):
        self.bm25 = BM25Retriever()
        self.vector = VectorRetriever(embedding_model)
        self.reranker = Reranker(embedding_model)

    def index(self, documents: list[str]) -> None:
        """构建完整索引"""
        self.bm25.index(documents)
        self.vector.index(documents)

    def search(self, query: str, top_k: int = 5,
               use_hybrid: bool = True, use_rerank: bool = True) -> list[dict]:
        """混合检索"""
        if use_hybrid:
            bm25_results = self.bm25.search(query, k=top_k * 2)
            vec_results = self.vector.search(query, k=top_k * 2)
            fused = RRF.fuse([bm25_results, vec_results], top_n=top_k * 2)
        else:
            fused = self.vector.search(query, k=top_k * 2)

        if use_rerank and fused:
            candidates = [(doc_id, self.bm25.documents[doc_id]) for doc_id, _ in fused]
            ranked = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            ranked = fused[:top_k]

        return [
            {"doc_id": doc_id, "score": round(score, 4),
             "content": self.bm25.documents[doc_id][:200]}
            for doc_id, score in ranked
        ]

    def self_assess(self, results: list[dict], threshold: float = 0.3) -> dict:
        """自反思评估: 检查检索质量"""
        if not results:
            return {"grade": "F", "action": "rewrite_query"}
        scores = [r["score"] for r in results]
        avg_score = np.mean(scores)
        coverage = np.mean([1.0 if s > threshold else 0.0 for s in scores])
        total = 0.5 * avg_score + 0.5 * coverage
        if total >= 0.6:
            return {"grade": "A", "action": "generate"}
        elif total >= 0.4:
            return {"grade": "B", "action": "generate_with_warning"}
        elif total >= 0.3:
            return {"grade": "C", "action": "expand_search"}
        else:
            return {"grade": "D", "action": "rewrite_query"}
