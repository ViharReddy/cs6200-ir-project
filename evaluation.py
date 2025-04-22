# evaluation.py
import json
import numpy as np
import matplotlib.pyplot as plt
from bm25_retrieval import BM25Retriever
from semantic_retrieval_optimized import OptimizedCodeBERTRetriever
from hybrid_retrieval import SimpleWeightedHybridRetriever, ReciprocalRankFusionRetriever

class RetrievalEvaluator:
    def __init__(self, python_data_file, java_data_file, annotations_file="annotations.json"):
        # Load data
        with open(python_data_file, 'r') as f:
            self.python_data = json.load(f)
        
        with open(java_data_file, 'r') as f:
            self.java_data = json.load(f)
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        self.queries = annotations.get("queries", {})
        self.judgments = annotations.get("judgments", {})
        
        # Initialize retrievers
        print("Initializing retrievers...")
        self.bm25_retriever = BM25Retriever(self.python_data, self.java_data)
        self.semantic_retriever = OptimizedCodeBERTRetriever(self.python_data, self.java_data)
        self.hybrid_retriever_03 = SimpleWeightedHybridRetriever(self.bm25_retriever, self.semantic_retriever, alpha=0.3)
        self.hybrid_retriever_05 = SimpleWeightedHybridRetriever(self.bm25_retriever, self.semantic_retriever, alpha=0.5)
        self.hybrid_retriever_07 = SimpleWeightedHybridRetriever(self.bm25_retriever, self.semantic_retriever, alpha=0.7)
        self.rrf_retriever = ReciprocalRankFusionRetriever(self.bm25_retriever, self.semantic_retriever)
        
        self.retrievers = {
            "BM25": self.bm25_retriever,
            "Semantic": self.semantic_retriever,
            "Hybrid (α=0.3)": self.hybrid_retriever_03,
            "Hybrid (α=0.5)": self.hybrid_retriever_05,
            "Hybrid (α=0.7)": self.hybrid_retriever_07,
            "RRF": self.rrf_retriever
        }
        
        print("Evaluator initialized!")
    
    def evaluate_all(self, k=10):
        """Evaluate all retrieval methods on all annotated queries"""
        if not self.queries or not self.judgments:
            print("No annotations available for evaluation.")
            return
        
        # Metrics to calculate
        metrics = ["precision@1", "precision@3", "precision@5", "precision@10", 
                  "recall@10", "mrr", "ndcg@10", "map"]
        
        # Store results for each retriever
        results = {retriever_name: {metric: [] for metric in metrics} for retriever_name in self.retrievers}
        
        # Evaluate each query
        for query_id, query in self.queries.items():
            if query_id not in self.judgments or not self.judgments[query_id]:
                continue  # Skip queries without judgments
            
            query_text = query["query"]
            query_judgments = self.judgments[query_id]
            
            print(f"Evaluating query: {query_text}")
            
            # Evaluate each retriever
            for retriever_name, retriever in self.retrievers.items():
                retrieved_docs = retriever.retrieve(query_text, k=k)
                
                # Calculate metrics
                precision_1 = self._precision_at_k(retrieved_docs, query_judgments, 1)
                precision_3 = self._precision_at_k(retrieved_docs, query_judgments, 3)
                precision_5 = self._precision_at_k(retrieved_docs, query_judgments, 5)
                precision_10 = self._precision_at_k(retrieved_docs, query_judgments, 10)
                recall_10 = self._recall_at_k(retrieved_docs, query_judgments, 10)
                mrr = self._mrr(retrieved_docs, query_judgments)
                ndcg_10 = self._ndcg_at_k(retrieved_docs, query_judgments, 10)
                map_score = self._average_precision(retrieved_docs, query_judgments)
                
                # Add to results
                results[retriever_name]["precision@1"].append(precision_1)
                results[retriever_name]["precision@3"].append(precision_3)
                results[retriever_name]["precision@5"].append(precision_5)
                results[retriever_name]["precision@10"].append(precision_10)
                results[retriever_name]["recall@10"].append(recall_10)
                results[retriever_name]["mrr"].append(mrr)
                results[retriever_name]["ndcg@10"].append(ndcg_10)
                results[retriever_name]["map"].append(map_score)
        
        # Calculate averages
        averages = {}
        for retriever_name in self.retrievers:
            averages[retriever_name] = {}
            for metric in metrics:
                if results[retriever_name][metric]:
                    averages[retriever_name][metric] = np.mean(results[retriever_name][metric])
                else:
                    averages[retriever_name][metric] = 0
        
        self._print_results(averages)
        self._plot_results(averages)
        
        return averages
    
    def evaluate_by_query_type(self, k=10):
        """Evaluate all retrieval methods grouped by query type"""
        if not self.queries or not self.judgments:
            print("No annotations available for evaluation.")
            return
        
        # Group queries by type
        query_types = {}
        for query_id, query in self.queries.items():
            if query_id not in self.judgments or not self.judgments[query_id]:
                continue  # Skip queries without judgments
            
            query_type = query.get("type", "unknown")
            if query_type not in query_types:
                query_types[query_type] = []
            query_types[query_type].append(query_id)
        
        # Evaluate for each query type
        results_by_type = {}
        for query_type, query_ids in query_types.items():
            print(f"\nEvaluating queries of type: {query_type} ({len(query_ids)} queries)")
            
            # Metrics to calculate
            metrics = ["precision@5", "recall@10", "mrr", "map"]
            
            # Store results for each retriever
            results = {retriever_name: {metric: [] for metric in metrics} for retriever_name in self.retrievers}
            
            # Evaluate each query in this type
            for query_id in query_ids:
                query = self.queries[query_id]
                query_text = query["query"]
                query_judgments = self.judgments[query_id]
                
                # Evaluate each retriever
                for retriever_name, retriever in self.retrievers.items():
                    retrieved_docs = retriever.retrieve(query_text, k=k)
                    
                    # Calculate metrics
                    precision_5 = self._precision_at_k(retrieved_docs, query_judgments, 5)
                    recall_10 = self._recall_at_k(retrieved_docs, query_judgments, 10)
                    mrr = self._mrr(retrieved_docs, query_judgments)
                    map_score = self._average_precision(retrieved_docs, query_judgments)
                    
                    # Add to results
                    results[retriever_name]["precision@5"].append(precision_5)
                    results[retriever_name]["recall@10"].append(recall_10)
                    results[retriever_name]["mrr"].append(mrr)
                    results[retriever_name]["map"].append(map_score)
            
            # Calculate averages
            averages = {}
            for retriever_name in self.retrievers:
                averages[retriever_name] = {}
                for metric in metrics:
                    if results[retriever_name][metric]:
                        averages[retriever_name][metric] = np.mean(results[retriever_name][metric])
                    else:
                        averages[retriever_name][metric] = 0
            
            results_by_type[query_type] = averages
            self._print_results_by_type(query_type, averages)
            self._plot_results_by_type(query_type, averages)
        
        return results_by_type
    
    def evaluate_by_language(self, k=10):
        """Evaluate all retrieval methods grouped by preferred language"""
        if not self.queries or not self.judgments:
            print("No annotations available for evaluation.")
            return
        
        # Group queries by language
        languages = {}
        for query_id, query in self.queries.items():
            if query_id not in self.judgments or not self.judgments[query_id]:
                continue  # Skip queries without judgments
            
            language = query.get("language", "any")
            if language not in languages:
                languages[language] = []
            languages[language].append(query_id)
        
        # Evaluate for each language
        results_by_language = {}
        for language, query_ids in languages.items():
            print(f"\nEvaluating queries for language: {language} ({len(query_ids)} queries)")
            
            # Metrics to calculate
            metrics = ["precision@5", "recall@10", "mrr", "map"]
            
            # Store results for each retriever
            results = {retriever_name: {metric: [] for metric in metrics} for retriever_name in self.retrievers}
            
            # Evaluate each query in this language
            for query_id in query_ids:
                query = self.queries[query_id]
                query_text = query["query"]
                query_judgments = self.judgments[query_id]
                
                # Evaluate each retriever
                for retriever_name, retriever in self.retrievers.items():
                    retrieved_docs = retriever.retrieve(query_text, k=k)
                    
                    # Calculate metrics
                    precision_5 = self._precision_at_k(retrieved_docs, query_judgments, 5)
                    recall_10 = self._recall_at_k(retrieved_docs, query_judgments, 10)
                    mrr = self._mrr(retrieved_docs, query_judgments)
                    map_score = self._average_precision(retrieved_docs, query_judgments)
                    
                    # Add to results
                    results[retriever_name]["precision@5"].append(precision_5)
                    results[retriever_name]["recall@10"].append(recall_10)
                    results[retriever_name]["mrr"].append(mrr)
                    results[retriever_name]["map"].append(map_score)
            
            # Calculate averages
            averages = {}
            for retriever_name in self.retrievers:
                averages[retriever_name] = {}
                for metric in metrics:
                    if results[retriever_name][metric]:
                        averages[retriever_name][metric] = np.mean(results[retriever_name][metric])
                    else:
                        averages[retriever_name][metric] = 0
            
            results_by_language[language] = averages
            self._print_results_by_language(language, averages)
            self._plot_results_by_language(language, averages)
        
        return results_by_language
    
    def _precision_at_k(self, retrieved_docs, judgments, k):
        """Calculate precision@k"""
        k = min(k, len(retrieved_docs))
        if k == 0:
            return 0
        
        relevant = 0
        for i in range(k):
            doc_id = retrieved_docs[i]["id"]
            if doc_id in judgments and judgments[doc_id] > 0:  # Any score > 0 is considered relevant
                relevant += 1
        
        return relevant / k
    
    def _recall_at_k(self, retrieved_docs, judgments, k):
        """Calculate recall@k"""
        k = min(k, len(retrieved_docs))
        
        # Count all relevant documents in judgments
        total_relevant = sum(1 for score in judgments.values() if score > 0)
        if total_relevant == 0:
            return 1.0  # Perfect recall if no relevant docs exist
        
        # Count relevant documents in top-k retrieved
        relevant_retrieved = 0
        for i in range(k):
            doc_id = retrieved_docs[i]["id"]
            if doc_id in judgments and judgments[doc_id] > 0:
                relevant_retrieved += 1
        
        return relevant_retrieved / total_relevant
    
    def _mrr(self, retrieved_docs, judgments):
        """Calculate Mean Reciprocal Rank"""
        for i, doc in enumerate(retrieved_docs):
            doc_id = doc["id"]
            if doc_id in judgments and judgments[doc_id] > 0:  # Any score > 0 is considered relevant
                return 1.0 / (i + 1)
        
        return 0.0
    
    def _average_precision(self, retrieved_docs, judgments):
        """Calculate Average Precision"""
        relevant_count = 0
        sum_precision = 0
        
        # Count total relevant documents
        total_relevant = sum(1 for score in judgments.values() if score > 0)
        
        if total_relevant == 0:
            return 0.0
        
        # Calculate precision at each relevant document
        for i, doc in enumerate(retrieved_docs):
            doc_id = doc["id"]
            if doc_id in judgments and judgments[doc_id] > 0:  # Found a relevant document
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                sum_precision += precision_at_i
        
        return sum_precision / total_relevant
    
    def _ndcg_at_k(self, retrieved_docs, judgments, k):
        """Calculate Normalized Discounted Cumulative Gain at k"""
        k = min(k, len(retrieved_docs))
        if k == 0:
            return 0
        
        # Calculate DCG
        dcg = 0
        for i in range(k):
            doc_id = retrieved_docs[i]["id"]
            rel = judgments.get(doc_id, 0)
            # Using the formula: (2^rel - 1) / log2(i+2)
            dcg += (2 ** rel - 1) / np.log2(i + 2)
        
        # Calculate ideal DCG
        rel_scores = sorted([judgments[doc_id] for doc_id in judgments], reverse=True)
        idcg = 0
        for i in range(min(k, len(rel_scores))):
            idcg += (2 ** rel_scores[i] - 1) / np.log2(i + 2)
        
        if idcg == 0:
            return 0
        
        return dcg / idcg
    
    def _print_results(self, averages):
        """Print average results for all queries"""
        print("\nAverage Results Across All Queries:")
        print("="*90)
        print(f"{'Retriever':<15} {'P@1':<8} {'P@3':<8} {'P@5':<8} {'P@10':<8} {'R@10':<8} {'MRR':<8} {'MAP':<8} {'NDCG@10':<8}")
        print("-"*90)
        
        for retriever_name, metrics in averages.items():
            p1 = metrics["precision@1"]
            p3 = metrics["precision@3"]
            p5 = metrics["precision@5"]
            p10 = metrics["precision@10"]
            r10 = metrics["recall@10"]
            mrr = metrics["mrr"]
            map_score = metrics["map"]
            ndcg = metrics["ndcg@10"]
            
            print(f"{retriever_name:<15} {p1:<8.3f} {p3:<8.3f} {p5:<8.3f} {p10:<8.3f} {r10:<8.3f} {mrr:<8.3f} {map_score:<8.3f} {ndcg:<8.3f}")
    
    def _print_results_by_type(self, query_type, averages):
        """Print average results for a specific query type"""
        print(f"\nAverage Results for Query Type: {query_type}")
        print("="*70)
        print(f"{'Retriever':<15} {'P@5':<8} {'R@10':<8} {'MRR':<8} {'MAP':<8}")
        print("-"*70)
        
        for retriever_name, metrics in averages.items():
            p5 = metrics["precision@5"]
            r10 = metrics["recall@10"]
            mrr = metrics["mrr"]
            map_score = metrics["map"]
            
            print(f"{retriever_name:<15} {p5:<8.3f} {r10:<8.3f} {mrr:<8.3f} {map_score:<8.3f}")
    
    def _print_results_by_language(self, language, averages):
        """Print average results for a specific language"""
        print(f"\nAverage Results for Language: {language}")
        print("="*70)
        print(f"{'Retriever':<15} {'P@5':<8} {'R@10':<8} {'MRR':<8} {'MAP':<8}")
        print("-"*70)
        
        for retriever_name, metrics in averages.items():
            p5 = metrics["precision@5"]
            r10 = metrics["recall@10"]
            mrr = metrics["mrr"]
            map_score = metrics["map"]
            
            print(f"{retriever_name:<15} {p5:<8.3f} {r10:<8.3f} {mrr:<8.3f} {map_score:<8.3f}")
    
    def _plot_results(self, averages):
        """Plot average results for all queries"""
        metrics = ["precision@5", "precision@10", "recall@10", "mrr", "map", "ndcg@10"]
        retriever_names = list(averages.keys())
        
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            values = [averages[name][metric] for name in retriever_names]
            plt.bar(retriever_names, values)
            
            # Format metric name for title
            if metric == "map":
                metric_title = "MAP"
            elif metric.startswith("mrr"):
                metric_title = "MRR"
            elif metric.startswith("ndcg"):
                metric_title = metric.upper()
            else:
                metric_title = metric.capitalize()
                
            plt.title(f"{metric_title} Across All Queries")
            plt.ylabel(metric)
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"evaluation_all_{metric.replace('@', '_at_')}.png")
            plt.close()
    
    def _plot_results_by_type(self, query_type, averages):
        """Plot average results for a specific query type"""
        metrics = ["precision@5", "recall@10", "mrr", "map"]
        retriever_names = list(averages.keys())
        
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            values = [averages[name][metric] for name in retriever_names]
            plt.bar(retriever_names, values)
            
            # Format metric name for title
            if metric == "map":
                metric_title = "MAP"
            elif metric.startswith("mrr"):
                metric_title = "MRR"
            else:
                metric_title = metric.capitalize()
                
            plt.title(f"{metric_title} for Query Type: {query_type}")
            plt.ylabel(metric)
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"evaluation_{query_type}_{metric.replace('@', '_at_')}.png")
            plt.close()
    
    def _plot_results_by_language(self, language, averages):
        """Plot average results for a specific language"""
        metrics = ["precision@5", "recall@10", "mrr", "map"]
        retriever_names = list(averages.keys())
        
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            values = [averages[name][metric] for name in retriever_names]
            plt.bar(retriever_names, values)
            
            # Format metric name for title
            if metric == "map":
                metric_title = "MAP"
            elif metric.startswith("mrr"):
                metric_title = "MRR"
            else:
                metric_title = metric.capitalize()
                
            plt.title(f"{metric_title} for Language: {language}")
            plt.ylabel(metric)
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"evaluation_{language}_{metric.replace('@', '_at_')}.png")
            plt.close()
    
    def cleanup(self):
        """Clean up resources"""
        self.bm25_retriever.cleanup()

# Run evaluation
if __name__ == "__main__":
    # Path to your data files
    python_data_file = "preprocessed_codesearchnet_python.json"
    java_data_file = "preprocessed_codesearchnet_java.json"
    annotations_file = "annotations.json"
    
    try:
        # Create evaluator
        evaluator = RetrievalEvaluator(python_data_file, java_data_file, annotations_file)
        
        # Run evaluations
        print("\nEvaluating all queries...")
        evaluator.evaluate_all()
        
        print("\nEvaluating by query type...")
        evaluator.evaluate_by_query_type()
        
        print("\nEvaluating by language preference...")
        evaluator.evaluate_by_language()
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
    
    finally:
        # Clean up
        if 'evaluator' in locals():
            evaluator.cleanup()