import time
from dataset_loader import load_subset_dataset
from bm25_retrieval import BM25Retriever
from semantic_retrieval import GraphCodeBERTRetriever
from hybrid_retrieval import SimpleWeightedHybridRetriever, ReciprocalRankFusionRetriever

# Load a subset of the preprocessed data
code_data_python, code_data_java = load_subset_dataset(
    "codesearchnet_python.json", 
    "codesearchnet_java.json", 
    max_items_per_language=1000
)

print("Building BM25 index...")
start_time = time.time()
bm25_retriever = BM25Retriever(code_data_python, code_data_java)
print(f"BM25 index built in {time.time() - start_time:.2f} seconds")

print("Building semantic retriever...")
start_time = time.time()
semantic_retriever = GraphCodeBERTRetriever(code_data_python, code_data_java)
print(f"Semantic retriever built in {time.time() - start_time:.2f} seconds")

# Create hybrid retrieval models
weighted_hybrid_retriever = SimpleWeightedHybridRetriever(bm25_retriever, semantic_retriever, alpha=0.5)
rrf_retriever = ReciprocalRankFusionRetriever(bm25_retriever, semantic_retriever)

# Test retrievers with few sample queries
test_queries = [
    "Binary search implementation in Python",
    "Java HashMap implementation",
    "How to read files in Python",
    "Sort array efficiently in Java",
    "Python async function example"
]
for query in test_queries:
    print(f"\n\nTesting query: {query}")
    print("=" * 60)

    query_start = time.time()
    results_bm25 = bm25_retriever.retrieve(query, k=5)
    query_time = time.time() - query_start
    print(f"\nBM25 Results (retrieved in {query_time:.3f}s):")
    for i, result in enumerate(results_bm25):
        print(f"{i+1}. [{result['language']}] {result['function_name']} (Score: {result['score']:.4f})")
        print(f"   Docstring: {result['docstring'][:100]}...")
        code_preview = result['code'].split('\n')[0][:50] + "..." if result['code'] else "No code"
        print(f"   Code: {code_preview}")

    query_start = time.time()
    results_semantic = semantic_retriever.retrieve(query, k=5)
    query_time = time.time() - query_start
    print(f"\nSemantic Results (retrieved in {query_time:.3f}s):")
    for i, result in enumerate(results_semantic):
        print(f"{i+1}. [{result['language']}] {result['function_name']} (Score: {result['score']:.4f})")
        print(f"   Docstring: {result['docstring'][:100]}...")
        code_preview = result['code'].split('\n')[0][:50] + "..." if result['code'] else "No code"
        print(f"   Code: {code_preview}")

    query_start = time.time()
    results_hybrid = weighted_hybrid_retriever.retrieve(query, k=5)
    query_time = time.time() - query_start
    print(f"\nHybrid Results (alpha=0.5)(retrieved in {query_time:.3f}s):")
    for i, result in enumerate(results_hybrid):
        print(f"{i+1}. [{result['language']}] {result['function_name']} (Combined Score: {result['combined_score']:.4f})")
        print(f"   Docstring: {result['docstring'][:100]}...")
        code_preview = result['code'].split('\n')[0][:50] + "..." if result['code'] else "No code"
        print(f"   Code: {code_preview}")

    query_start = time.time()
    results_rrf = rrf_retriever.retrieve(query, k=5)
    query_time = time.time() - query_start
    print(f"\nRRF Results (retrieved in {query_time:.3f}s):")
    for i, result in enumerate(results_rrf):
        print(f"{i+1}. [{result['language']}] {result['function_name']} (RRF Score: {result['rrf_score']:.4f})")
        print(f"   Docstring: {result['docstring'][:100]}...")
        code_preview = result['code'].split('\n')[0][:50] + "..." if result['code'] else "No code"
        print(f"   Code: {code_preview}")

# Clean up resources
bm25_retriever.cleanup()