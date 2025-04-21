import json
import pandas as pd
from dataset_loader import load_subset_dataset
from semantic_retrieval_optimized import OptimizedCodeBERTRetriever
from bm25_retrieval import BM25Retriever
from hybrid_retrieval import SimpleWeightedHybridRetriever

# Load a subset of the preprocessed data
code_data_python, code_data_java = load_subset_dataset(
    "preprocessed_codesearchnet_python.json", 
    "preprocessed_codesearchnet_java.json", 
    max_items_per_language=10000
)

# Create the retrievers
bm25_retriever = BM25Retriever(code_data_python, code_data_java)
semantic_retriever = OptimizedCodeBERTRetriever(code_data_python, code_data_java)

# Test with specific query about binary search
query = "Binary search implementation in Python"

print(f"\nBM25 Results for: {query}")
print("-" * 60)
binary_search_results_bm25 = bm25_retriever.retrieve(query, k=20)
for i, result in enumerate(binary_search_results_bm25):
    if "binary" in result["code"].lower() or "binary" in result["function_name"].lower():
        print(f"{i+1}. [{result['language']}] {result['function_name']} (Score: {result['score']:.4f})")
        print(f"   Docstring: {result['docstring'][:100]}...")
        # Print first few lines of code
        code_lines = result['code'].split('\n')[:5]
        code_snippet = '\n'.join(code_lines)
        print(f"   Code snippet:\n{code_snippet}")
        print()

# Search for manual matches in the datasets
print("\nManually searching for binary search implementations:")
print("-" * 60)
binary_search_codes = []

for i, item in enumerate(code_data_python):
    if ("binary search" in item["code"].lower() or 
        "binary search" in item["function_name"].lower() or
        "binary search" in item["docstring"].lower()):
        binary_search_codes.append({
            "id": f"python_{item['id']}",
            "language": "python",
            "function_name": item["function_name"],
            "code": item["code"],
            "docstring": item["docstring"]
        })

for i, item in enumerate(code_data_java):
    if ("binary search" in item["code"].lower() or 
        "binary search" in item["function_name"].lower() or
        "binary search" in item["docstring"].lower()):
        binary_search_codes.append({
            "id": f"java_{item['id']}",
            "language": "java",
            "function_name": item["function_name"],
            "code": item["code"],
            "docstring": item["docstring"]
        })

print(f"Found {len(binary_search_codes)} code snippets containing 'binary search'")
for i, code in enumerate(binary_search_codes[:5]):  # Show first 5
    print(f"{i+1}. [{code['language']}] {code['function_name']}")
    print(f"   Docstring: {code['docstring'][:100]}...")
    code_lines = code['code'].split('\n')[:5]
    code_snippet = '\n'.join(code_lines)
    print(f"   Code snippet:\n{code_snippet}")
    print()

# Now let's implement a custom binary search retriever
class CustomBinarySearchRetriever:
    def __init__(self, binary_search_codes):
        self.binary_search_codes = binary_search_codes
    
    def retrieve(self, query, k=10):
        # First filter by language preference
        query_lower = query.lower()
        if "python" in query_lower:
            preferred_language = "python"
        elif "java" in query_lower:
            preferred_language = "java"
        else:
            preferred_language = None
        
        filtered_codes = self.binary_search_codes
        if preferred_language:
            filtered_codes = [code for code in filtered_codes if code["language"] == preferred_language]
        
        # Simple ranking by matching terms in function name and docstring
        for code in filtered_codes:
            score = 0
            if "binary search" in code["function_name"].lower():
                score += 5
            if "binary search" in code["docstring"].lower():
                score += 3
            if "implementation" in code["docstring"].lower():
                score += 2
            code["score"] = score
        
        # Sort by score and return top k
        sorted_codes = sorted(filtered_codes, key=lambda x: x["score"], reverse=True)
        return sorted_codes[:k]

# Test the custom retriever
custom_retriever = CustomBinarySearchRetriever(binary_search_codes)
print(f"\nCustom Retriever Results for: {query}")
print("-" * 60)
custom_results = custom_retriever.retrieve(query, k=5)
for i, result in enumerate(custom_results):
    print(f"{i+1}. [{result['language']}] {result['function_name']} (Score: {result['score']:.1f})")
    print(f"   Docstring: {result['docstring'][:100]}...")
    code_lines = result['code'].split('\n')[:5]
    code_snippet = '\n'.join(code_lines)
    print(f"   Code snippet:\n{code_snippet}")
    print()