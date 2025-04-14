import json
import pandas as pd
import matplotlib.pyplot as plt
import time
from dataset_loader import load_subset_dataset
from bm25_retrieval import BM25Retriever
from semantic_retrieval_optimized import OptimizedCodeBERTRetriever
from hybrid_retrieval import SimpleWeightedHybridRetriever, ReciprocalRankFusionRetriever

# Load a subset of the preprocessed data
code_data_python, code_data_java = load_subset_dataset(
    "preprocessed_codesearchnet_python.json", 
    "preprocessed_codesearchnet_java.json", 
    max_items_per_language=3000
)

print("Building BM25 index...")
start_time = time.time()
bm25_retriever = BM25Retriever(code_data_python, code_data_java)
print(f"BM25 index built in {time.time() - start_time:.2f} seconds")

print("Building semantic retriever...")
start_time = time.time()
semantic_retriever = OptimizedCodeBERTRetriever(code_data_python, code_data_java)
print(f"Semantic retriever built in {time.time() - start_time:.2f} seconds")

# Create hybrid retrieval models
weighted_hybrid_retriever = SimpleWeightedHybridRetriever(bm25_retriever, semantic_retriever, alpha=0.5)
rrf_retriever = ReciprocalRankFusionRetriever(bm25_retriever, semantic_retriever)

# Test retrievers with a sample query
query = "Binary search implementation in Python"
print("\nBM25 Results:")
for i, result in enumerate(bm25_retriever.retrieve(query, k=5)):
    print(f"{i+1}. [{result['language']}] {result['function_name']} (Score: {result['score']:.4f})")
    print(f"   Docstring: {result['docstring'][:100]}...")

print("\nSemantic Results:")
for i, result in enumerate(semantic_retriever.retrieve(query, k=5)):
    print(f"{i+1}. [{result['language']}] {result['function_name']} (Score: {result['score']:.4f})")
    print(f"   Docstring: {result['docstring'][:100]}...")

print("\nHybrid Results (alpha=0.5):")
for i, result in enumerate(weighted_hybrid_retriever.retrieve(query, k=5)):
    print(f"{i+1}. [{result['language']}] {result['function_name']} (Combined Score: {result['combined_score']:.4f})")
    print(f"   Docstring: {result['docstring'][:100]}...")

print("\nRRF Results:")
for i, result in enumerate(rrf_retriever.retrieve(query, k_results=5)):
    print(f"{i+1}. [{result['language']}] {result['function_name']} (RRF Score: {result['rrf_score']:.4f})")
    print(f"   Docstring: {result['docstring'][:100]}...")

# Clean up resources
bm25_retriever.cleanup()