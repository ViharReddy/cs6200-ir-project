# annotation_tool.py
import json
import os
from datetime import datetime
from bm25_retrieval import BM25Retriever
from semantic_retrieval_optimized import OptimizedCodeBERTRetriever
from hybrid_retrieval import SimpleWeightedHybridRetriever, ReciprocalRankFusionRetriever

class AnnotationTool:
    def __init__(self, python_data_file, java_data_file, output_file="annotations.json"):
        # Load data
        with open(python_data_file, 'r') as f:
            self.python_data = json.load(f)
        
        with open(java_data_file, 'r') as f:
            self.java_data = json.load(f)
        
        self.output_file = output_file
        
        # Load existing annotations if available
        self.annotations = {}
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                self.annotations = json.load(f)
        
        # Initialize retrievers
        print("Initializing retrievers...")
        self.bm25_retriever = BM25Retriever(self.python_data, self.java_data)
        self.semantic_retriever = OptimizedCodeBERTRetriever(self.python_data, self.java_data)
        self.hybrid_retriever = SimpleWeightedHybridRetriever(self.bm25_retriever, self.semantic_retriever, alpha=0.5)
        self.rrf_retriever = ReciprocalRankFusionRetriever(self.bm25_retriever, self.semantic_retriever)
        
        print("Annotation tool initialized!")
    
    def add_query(self, query_text, narrative, query_type=None, language=None):
        """Add a new query for annotation"""
        # Generate a unique ID
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        query_id = f"q_{timestamp}"
        
        # Create query object
        query = {
            "id": query_id,
            "query": query_text,
            "narrative": narrative,
            "type": query_type,
            "language": language,
            "timestamp": timestamp
        }
        
        # Store in annotations
        if "queries" not in self.annotations:
            self.annotations["queries"] = {}
        
        self.annotations["queries"][query_id] = query
        self._save_annotations()
        
        print(f"Added query: {query_text} (ID: {query_id})")
        return query_id
    
    def annotate_query(self, query_id, max_results=10):
        """Annotate results for a specific query"""
        if "queries" not in self.annotations or query_id not in self.annotations["queries"]:
            print(f"Query ID {query_id} not found!")
            return
        
        query = self.annotations["queries"][query_id]
        query_text = query["query"]
        
        print("\n" + "="*60)
        print(f"Query: {query_text}")
        print(f"Narrative: {query['narrative']}")
        print("="*60)
        
        # Initialize judgments for this query if not exist
        if "judgments" not in self.annotations:
            self.annotations["judgments"] = {}
        
        if query_id not in self.annotations["judgments"]:
            self.annotations["judgments"][query_id] = {}
        
        # Get results from all retrievers
        print("Retrieving results...")
        bm25_results = self.bm25_retriever.retrieve(query_text, k=max_results)
        semantic_results = self.semantic_retriever.retrieve(query_text, k=max_results)
        hybrid_results = self.hybrid_retriever.retrieve(query_text, k=max_results)
        rrf_results = self.rrf_retriever.retrieve(query_text, k=max_results)
        
        # Combine results and remove duplicates
        all_results = {}
        
        for result in bm25_results:
            doc_id = result["id"]
            all_results[doc_id] = result
            all_results[doc_id]["methods"] = ["BM25"]
        
        for result in semantic_results:
            doc_id = result["id"]
            if doc_id in all_results:
                all_results[doc_id]["methods"].append("Semantic")
            else:
                all_results[doc_id] = result
                all_results[doc_id]["methods"] = ["Semantic"]
        
        for result in hybrid_results:
            doc_id = result["id"]
            if doc_id in all_results:
                all_results[doc_id]["methods"].append("Hybrid")
            else:
                all_results[doc_id] = result
                all_results[doc_id]["methods"] = ["Hybrid"]
        
        for result in rrf_results:
            doc_id = result["id"]
            if doc_id in all_results:
                all_results[doc_id]["methods"].append("RRF")
            else:
                all_results[doc_id] = result
                all_results[doc_id]["methods"] = ["RRF"]
        
        # Sort by the number of methods that retrieved this result
        result_items = list(all_results.items())
        result_items.sort(key=lambda x: len(x[1]["methods"]), reverse=True)
        
        # Annotate each result
        for i, (doc_id, result) in enumerate(result_items):
            # Skip if already annotated
            if doc_id in self.annotations["judgments"][query_id]:
                continue
            
            print("\n" + "-"*60)
            print(f"Result {i+1}/{len(result_items)}: {doc_id}")
            print(f"Retrieved by: {', '.join(result['methods'])}")
            print(f"Language: {result['language']}")
            print(f"Function: {result['function_name']}")
            print(f"Docstring: {result['docstring']}")
            print("-"*20 + " Code " + "-"*20)
            
            # Display code with line numbers (truncated if too long)
            code_lines = result["code"].split("\n")
            if len(code_lines) > 20:
                print("\n".join([f"{i+1}: {line}" for i, line in enumerate(code_lines[:20])]))
                print(f"... (truncated, {len(code_lines)} lines total)")
            else:
                print("\n".join([f"{i+1}: {line}" for i, line in enumerate(code_lines)]))
            
            # Get relevance judgment
            while True:
                try:
                    judgment = input("\nRelevance score (0=Not relevant, 1=Somewhat relevant, 2=Highly relevant): ")
                    judgment = int(judgment)
                    if judgment in [0, 1, 2]:
                        break
                    print("Please enter 0, 1, or 2")
                except ValueError:
                    print("Please enter a number")
            
            # Save judgment
            self.annotations["judgments"][query_id][doc_id] = judgment
            self._save_annotations()
            
            # Ask if user wants to continue
            if i < len(result_items) - 1:  # Not the last result
                cont = input("\nContinue annotating? (y/n): ")
                if cont.lower() != 'y':
                    break
        
        print("\nAnnotation session complete!")
    
    def list_queries(self):
        """List all queries available for annotation"""
        if "queries" not in self.annotations or not self.annotations["queries"]:
            print("No queries have been added yet.")
            return
        
        print("\nAvailable Queries:")
        print("="*80)
        print(f"{'ID':<15} {'Query':<40} {'#Judgments':<10} {'Type':<15}")
        print("-"*80)
        
        for query_id, query in self.annotations["queries"].items():
            num_judgments = 0
            if "judgments" in self.annotations and query_id in self.annotations["judgments"]:
                num_judgments = len(self.annotations["judgments"][query_id])
            
            print(f"{query_id:<15} {query['query'][:40]:<40} {num_judgments:<10} {query.get('type', ''):<15}")
    
    def get_statistics(self):
        """Get statistics about annotation progress"""
        if "queries" not in self.annotations or not self.annotations["queries"]:
            print("No queries have been added yet.")
            return
        
        total_queries = len(self.annotations["queries"])
        annotated_queries = 0
        total_judgments = 0
        
        if "judgments" in self.annotations:
            for query_id in self.annotations["queries"]:
                if query_id in self.annotations["judgments"] and self.annotations["judgments"][query_id]:
                    annotated_queries += 1
                    total_judgments += len(self.annotations["judgments"][query_id])
        
        print("\nAnnotation Statistics:")
        print(f"Total Queries: {total_queries}")
        print(f"Queries with Annotations: {annotated_queries}")
        print(f"Total Relevance Judgments: {total_judgments}")
        print(f"Average Judgments per Annotated Query: {total_judgments/annotated_queries if annotated_queries else 0:.2f}")
    
    def _save_annotations(self):
        """Save annotations to file"""
        with open(self.output_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
    
    def cleanup(self):
        """Clean up resources"""
        self.bm25_retriever.cleanup()

# Interactive annotation session
if __name__ == "__main__":
    # Path to your data files
    python_data_file = "preprocessed_codesearchnet_python.json"
    java_data_file = "preprocessed_codesearchnet_java.json"
    
    # Create tool
    annotator = AnnotationTool(python_data_file, java_data_file)
    
    while True:
        print("\n" + "="*60)
        print("Code Snippet Retrieval Annotation Tool")
        print("="*60)
        print("1. Add new query")
        print("2. Annotate existing query")
        print("3. List available queries")
        print("4. Show annotation statistics")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ")
        
        if choice == "1":
            query = input("Enter query text: ")
            narrative = input("Enter narrative (explain what user is looking for): ")
            
            query_type = input("Enter query type (concept/debugging/optimization): ")
            language = input("Enter preferred language (python/java/any): ")
            
            annotator.add_query(query, narrative, query_type, language)
            
        elif choice == "2":
            annotator.list_queries()
            query_id = input("\nEnter query ID to annotate: ")
            annotator.annotate_query(query_id)
            
        elif choice == "3":
            annotator.list_queries()
            
        elif choice == "4":
            annotator.get_statistics()
            
        elif choice == "5":
            print("Cleaning up resources...")
            annotator.cleanup()
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice, please try again.")