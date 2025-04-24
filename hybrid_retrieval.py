class SimpleWeightedHybridRetriever:
    def __init__(self, bm25_retriever, semantic_retriever, alpha=0.5):
        self.bm25_retriever = bm25_retriever
        self.semantic_retriever = semantic_retriever
        self.alpha = alpha
    
    def retrieve(self, query, k=10):
        # Get results from both retrievers (get more to ensure coverage)
        bm25_results = self.bm25_retriever.retrieve(query, k=k*2)
        semantic_results = self.semantic_retriever.retrieve(query, k=k*2)
        
        # Normalize scores
        bm25_max = max([r["score"] for r in bm25_results]) if bm25_results else 1.0
        semantic_max = max([r["score"] for r in semantic_results]) if semantic_results else 1.0
        
        # Create a dictionary of combined scores
        combined_scores = {}
        
        # Add BM25 scores
        for result in bm25_results:
            doc_id = result["id"]
            normalized_score = result["score"] / bm25_max
            combined_scores[doc_id] = {
                "bm25_score": normalized_score,
                "semantic_score": 0,
                "combined_score": self.alpha * normalized_score,
                "data": result
            }
        
        # Add semantic scores
        for result in semantic_results:
            doc_id = result["id"]
            normalized_score = result["score"] / semantic_max
            
            if doc_id in combined_scores:
                combined_scores[doc_id]["semantic_score"] = normalized_score
                combined_scores[doc_id]["combined_score"] += (1-self.alpha) * normalized_score
            else:
                combined_scores[doc_id] = {
                    "bm25_score": 0,
                    "semantic_score": normalized_score, 
                    "combined_score": (1-self.alpha) * normalized_score,
                    "data": result
                }
            
            if doc_id.startswith("python") and "python" in query.lower():
                combined_scores[doc_id]["combined_score"] *= 1.1
            elif doc_id.startswith("java") and "java" in query.lower():
                combined_scores[doc_id]["combined_score"] *= 1.1
        
        # Sort by combined score and get top-k results
        top_doc_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x]["combined_score"], reverse=True)[:k]
        
        results = []
        for doc_id in top_doc_ids:
            result = combined_scores[doc_id]["data"].copy()
            result["bm25_score"] = combined_scores[doc_id]["bm25_score"]
            result["semantic_score"] = combined_scores[doc_id]["semantic_score"] 
            result["combined_score"] = combined_scores[doc_id]["combined_score"]
            results.append(result)
        
        return results

class ReciprocalRankFusionRetriever:
    def __init__(self, bm25_retriever, semantic_retriever, k=60):
        self.bm25_retriever = bm25_retriever
        self.semantic_retriever = semantic_retriever
        self.k = k  # RRF constant
    
    def retrieve(self, query, k=10):
        # Get results from both methods
        bm25_results = self.bm25_retriever.retrieve(query, k=k*3)
        semantic_results = self.semantic_retriever.retrieve(query, k=k*3)
        
        # Create dictionaries for rankings
        bm25_ranks = {result["id"]: 1/(i+self.k) for i, result in enumerate(bm25_results)}
        semantic_ranks = {result["id"]: 1/(i+self.k) for i, result in enumerate(semantic_results)}
        
        # Combine all document IDs
        all_ids = set(list(bm25_ranks.keys()) + list(semantic_ranks.keys()))
        
        # Calculate RRF scores
        rrf_scores = {}
        for doc_id in all_ids:
            rrf_scores[doc_id] = bm25_ranks.get(doc_id, 0) + semantic_ranks.get(doc_id, 0)
        
        # Sort by RRF score
        top_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]
        
        # Get the actual documents
        results = []
        for doc_id in top_doc_ids:
            # Find the document in the original results
            doc = None
            for res in bm25_results:
                if res["id"] == doc_id:
                    doc = res.copy()
                    break
            
            if doc is None:
                for res in semantic_results:
                    if res["id"] == doc_id:
                        doc = res.copy()
                        break
            
            if doc:
                doc["rrf_score"] = rrf_scores[doc_id]
                for res in bm25_results:
                    if res["id"] == doc_id:
                        doc["bm25_score"] = res["score"]
                        break
                for res in semantic_results:
                    if res["id"] == doc_id:
                        doc["semantic_score"] = res["score"]
                        break
                results.append(doc)
        
        return results