# annotation_tool.py
import json
import os
from pathlib import Path

class AnnotationTool:
    def __init__(self, retrievers, retriever_names, queries_file="queries.json", 
                 judgments_file="relevance_judgments.json"):
        self.retrievers = retrievers
        self.retriever_names = retriever_names
        self.queries_file = queries_file
        self.judgments_file = judgments_file
        
        # Load or initialize queries
        if os.path.exists(queries_file):
            with open(queries_file, 'r') as f:
                self.queries = json.load(f)
        else:
            self.queries = []
        
        # Load or initialize relevance judgments
        if os.path.exists(judgments_file):
            with open(judgments_file, 'r') as f:
                self.relevance_judgments = json.load(f)
        else:
            self.relevance_judgments = {}
    
    def add_query(self, query_text, narrative, language=None, query_type=None):
        """Add a new query to the collection"""
        query_id = f"q{len(self.queries) + 1}"
        
        query = {
            "id": query_id,
            "query": query_text,
            "narrative": narrative,
            "language": language,
            "type": query_type
        }
        
        self.queries.append(query)
        self._save_queries()
        
        return query_id
    
    def annotate_query(self, query_id, k=10):
        """Manually annotate results for a query"""
        # Find the query
        query = None
        for q in self.queries:
            if q["id"] == query_id:
                query = q
                break
        
        if query is None:
            print(f"Query {query_id} not found")
            return
        
        print(f"\nAnnotating results for query: {query['query']}")
        print(f"Narrative: {query['narrative']}")
        
        # Initialize judgments for this query if not exist
        if query_id not in self.relevance_judgments:
            self.relevance_judgments[query_id] = {}
        
        # Get results from all retrievers
        all_results = {}
        for i, retriever in enumerate(self.retrievers):
            retriever_name = self.retriever_names[i]
            results = retriever.retrieve(query["query"], k=k)
            
            # Add to all results
            for result in results:
                doc_id = result["id"]
                if doc_id not in all_results:
                    all_results[doc_id] = result
        
        # Sort by document ID for consistent ordering
        doc_ids = sorted(all_results.keys())
        
        # Annotate each result
        for doc_id in doc_ids:
            result = all_results[doc_id]
            
            # Skip if already annotated
            if doc_id in self.relevance_judgments[query_id]:
                continue
            
            print(f"\nDocument ID: {doc_id}")
            print(f"Language: {result['language']}")
            print(f"Function: {result['function_name']}")
            print(f"Docstring: {result['docstring']}")
            print(f"Code Snippet:\n{result['code'][:500]}...")
            
            # Get relevance judgment
            while True:
                try:
                    relevance = int(input("\nRelevance (0=Not relevant, 1=Somewhat relevant, 2=Highly relevant): "))
                    if relevance in [0, 1, 2]:
                        break
                    print("Please enter 0, 1, or 2")
                except ValueError:
                    print("Please enter a number")
            
            # Save judgment
            self.relevance_judgments[query_id][doc_id] = relevance
            self._save_judgments()
            
            # Ask if the user wants to continue
            if len(self.relevance_judgments[query_id]) % 5 == 0:
                cont = input("\nContinue annotating? (y/n): ")
                if cont.lower() != 'y':
                    break
        
        print(f"\nAnnotation completed for query {query_id}")
    
    def get_queries(self):
        """Get all queries"""
        return self.queries
    
    def get_judgments(self):
        """Get all relevance judgments"""
        return self.relevance_judgments
    
    def _save_queries(self):
        """Save queries to file"""
        with open(self.queries_file, 'w') as f:
            json.dump(self.queries, f, indent=2)
    
    def _save_judgments(self):
        """Save judgments to file"""
        with open(self.judgments_file, 'w') as f:
            json.dump(self.relevance_judgments, f, indent=2)

# Example usage:
"""
# In your main script:
annotation_tool = AnnotationTool([bm25_retriever, semantic_retriever, weighted_hybrid_retriever, rrf_retriever],
                                ["BM25", "CodeBERT", "Hybrid", "RRF"])

# Add queries from your examples
annotation_tool.add_query(
    "Binary search implementation in Python",
    "The user is looking for a sample code that is correct and efficient, written in Python.",
    language="python",
    query_type="concept-based"
)

# Annotate results
annotation_tool.annotate_query("q1")
"""