from .semantic_search import ChunkedSemanticSearch
from .keyword_search import InvertedIndex

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def _min_max_normalize(self, scores: list[float]) -> list[float]:
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if min_score == max_score:
            return [1.0 for _ in scores]

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def weighted_search(self, query: str, alpha: float = 0.5, limit: int = 5):
        expanded_limit = limit * 500

        # 1️⃣ Get raw results
        bm25_results = self._bm25_search(query, expanded_limit)
        semantic_results = self.semantic_search.search_chunks(query, expanded_limit)

        # 2️⃣ Extract scores
        bm25_scores = [r.get("score", 0.0) for r in bm25_results]
        semantic_scores = [r.get("score", 0.0) for r in semantic_results]

        # 3️⃣ Normalize scores
        normalized_bm25 = self._min_max_normalize(bm25_scores)
        normalized_semantic = self._min_max_normalize(semantic_scores)

        # 4️⃣ Combine results by doc ID
        combined = {}

        for result, norm_score in zip(bm25_results, normalized_bm25):
            doc_id = result.get("id")
            if not doc_id:
                continue
            combined[doc_id] = {
                "id": doc_id,
                "title": result.get("title"),
                "description": result.get("description") or result.get("document", ""),
                "bm25": norm_score,
                "semantic": 0.0,
            }

        for result, norm_score in zip(semantic_results, normalized_semantic):
            doc_id = result.get("id")
            if not doc_id:
                continue
            if doc_id not in combined:
                combined[doc_id] = {
                    "id": doc_id,
                    "title": result.get("title"),
                    "description": result.get("description") or result.get("document", ""),
                    "bm25": 0.0,
                    "semantic": norm_score,
                }
            else:
                combined[doc_id]["semantic"] = norm_score

        # 5️⃣ Compute hybrid scores
        for doc_id, data in combined.items():
            hybrid = alpha * data["bm25"] + (1 - alpha) * data["semantic"]
            data["hybrid"] = hybrid

        # 6️⃣ Sort descending by hybrid
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x["hybrid"],
            reverse=True
        )

        # 7️⃣ Return top `limit` results
        return sorted_results[:limit]