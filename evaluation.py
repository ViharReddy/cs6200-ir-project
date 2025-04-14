import numpy as np
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, queries, relevance_judgments):
        """
        Initialize evaluator with queries and relevance judgments.
        
        Args:
            queries: List of query dictionaries with 'id', 'query', 'narrative', etc.
            relevance_judgments: Dictionary mapping query_id -> doc_id -> relevance_score
        """
        self.queries = queries
        self.relevance_judgments = relevance_judgments
    
    def evaluate_retriever(self, retriever, metrics=["precision@5", "precision@10", "mrr", "recall@10"], k=10):
        """
        Evaluate a retriever on all queries using specified metrics.
        
        Args:
            retriever: A retriever object with a retrieve(query, k) method
            metrics: List of metrics to compute
            k: Number of results to retrieve
            
        Returns:
            Dictionary of metric -> average score
        """
        results = {metric: [] for metric in metrics}
        
        for query in self.queries:
            query_id = query["id"]
            query_text = query["query"]
            
            # Get retrieval results
            retrieved_docs = retriever.retrieve(query_text, k=max(10, k))
            
            # Get relevance judgments for this query
            if query_id in self.relevance_judgments:
                query_judgments = self.relevance_judgments[query_id]
                
                # Calculate metrics
                if "precision@5" in metrics:
                    p5 = self._precision_at_k(retrieved_docs, query_judgments, 5)
                    results["precision@5"].append(p5)
                
                if "precision@10" in metrics:
                    p10 = self._precision_at_k(retrieved_docs, query_judgments, 10)
                    results["precision@10"].append(p10)
                
                if "recall@10" in metrics:
                    r10 = self._recall_at_k(retrieved_docs, query_judgments, 10)
                    results["recall@10"].append(r10)
                
                if "mrr" in metrics:
                    mrr = self._mrr(retrieved_docs, query_judgments)
                    results["mrr"].append(mrr)
        
        # Calculate averages
        avg_results = {metric: np.mean(scores) for metric, scores in results.items()}
        return avg_results
    
    def _precision_at_k(self, retrieved_docs, relevance_judgments, k):
        """Calculate precision@k for a single query"""
        if not retrieved_docs:
            return 0.0
        
        k = min(k, len(retrieved_docs))
        relevant_count = 0
        
        for i in range(k):
            doc_id = retrieved_docs[i]["id"]
            if doc_id in relevance_judgments and relevance_judgments[doc_id] > 0:
                relevant_count += 1
        
        return relevant_count / k
    
    def _recall_at_k(self, retrieved_docs, relevance_judgments, k):
        """Calculate recall@k for a single query"""
        relevant_docs = sum(1 for rel in relevance_judgments.values() if rel > 0)
        if relevant_docs == 0:
            return 1.0  # All relevant docs retrieved (there were none)
        
        k = min(k, len(retrieved_docs))
        relevant_retrieved = 0
        
        for i in range(k):
            doc_id = retrieved_docs[i]["id"]
            if doc_id in relevance_judgments and relevance_judgments[doc_id] > 0:
                relevant_retrieved += 1
        
        return relevant_retrieved / relevant_docs
    
    def _mrr(self, retrieved_docs, relevance_judgments):
        """Calculate Mean Reciprocal Rank for a single query"""
        if not retrieved_docs:
            return 0.0
        
        for i, doc in enumerate(retrieved_docs):
            doc_id = doc["id"]
            if doc_id in relevance_judgments and relevance_judgments[doc_id] > 0:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def compare_retrievers(self, retrievers, names=None, metrics=["precision@5", "precision@10", "mrr"]):
        """
        Compare multiple retrievers on all queries.
        
        Args:
            retrievers: List of retriever objects
            names: List of names for the retrievers (for visualization)
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of retriever_name -> metric -> score
        """
        if names is None:
            names = [f"Retriever {i+1}" for i in range(len(retrievers))]
        
        results = {}
        for i, retriever in enumerate(retrievers):
            name = names[i]
            results[name] = self.evaluate_retriever(retriever, metrics)
        
        return results
    
    def plot_comparison(self, comparison_results, metric="precision@10"):
        """Plot comparison of retrievers for a specific metric"""
        names = list(comparison_results.keys())
        scores = [comparison_results[name][metric] for name in names]
        
        plt.figure(figsize=(10, 6))
        plt.bar(names, scores)
        plt.title(f"Comparison of Retrievers ({metric})")
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt