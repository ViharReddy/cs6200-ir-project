import json
import numpy as np
import matplotlib.pyplot as plt
from bm25_retrieval import BM25Retriever
from semantic_retrieval_optimized import OptimizedCodeBERTRetriever
from hybrid_retrieval import SimpleWeightedHybridRetriever, ReciprocalRankFusionRetriever
from dataset_loader import load_subset_dataset
from evaluation import Evaluator

def run_evaluation(queries_file="queries.json", judgments_file="relevance_judgments.json",
                  python_data_file="preprocessed_codesearchnet_python.json",
                  java_data_file="preprocessed_codesearchnet_java.json",
                  max_items=10000):
    
    # Load data
    code_data_python, code_data_java = load_subset_dataset(
        python_data_file, java_data_file, max_items_per_language=max_items
    )
    
    # Load queries and judgments
    with open(queries_file, 'r') as f:
        queries = json.load(f)
    
    with open(judgments_file, 'r') as f:
        relevance_judgments = json.load(f)
    
    # Create retrievers
    print("Building retrieval models...")
    bm25_retriever = BM25Retriever(code_data_python, code_data_java)
    semantic_retriever = OptimizedCodeBERTRetriever(code_data_python, code_data_java)
    
    # Create hybrid retrievers
    hybrid_03 = SimpleWeightedHybridRetriever(bm25_retriever, semantic_retriever, alpha=0.3)
    hybrid_05 = SimpleWeightedHybridRetriever(bm25_retriever, semantic_retriever, alpha=0.5)
    hybrid_07 = SimpleWeightedHybridRetriever(bm25_retriever, semantic_retriever, alpha=0.7)
    rrf_retriever = ReciprocalRankFusionRetriever(bm25_retriever, semantic_retriever)
    
    # Initialize evaluator
    evaluator = Evaluator(queries, relevance_judgments)
    
    # Compare retrievers
    retrievers = [
        bm25_retriever,
        semantic_retriever,
        hybrid_03,
        hybrid_05, 
        hybrid_07,
        rrf_retriever
    ]
    
    names = [
        "BM25",
        "CodeBERT",
        "Hybrid (α=0.3)",
        "Hybrid (α=0.5)",
        "Hybrid (α=0.7)",
        "RRF"
    ]
    
    # Evaluate
    print("Running evaluation...")
    results = evaluator.compare_retrievers(retrievers, names)
    
    # Print results
    print("\nEvaluation Results:")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Generate visualizations
    metrics = ["precision@5", "precision@10", "mrr", "recall@10"]
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.bar(names, [results[name][metric] for name in names])
        plt.title(f"Comparison of Retrievers ({metric})")
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"comparison_{metric}.png")
    
    # Clean up
    bm25_retriever.cleanup()
    
    return results

if __name__ == "__main__":
    run_evaluation()