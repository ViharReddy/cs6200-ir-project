import json
import os
import re
import subprocess
import tempfile
from pyserini.index.lucene import LuceneIndexReader
from pyserini.search.lucene import LuceneSearcher

class BM25Retriever:
    def __init__(self, code_data_python, code_data_java):
        self.code_data = {}
        
        # Combine Python and Java data
        for item in code_data_python:
            self.code_data[f"python_{item['id']}"] = {
                "id": f"python_{item['id']}",
                "language": item["language"],
                "function_name": item["function_name"],
                "code": item["code"],
                "docstring": item["docstring"],
                "file_path": item["file_path"],
                "tokenized_function": item.get("tokenized_function", ""),
                "clean_code": item.get("clean_code", ""),
                "indexed_content": item.get("indexed_content", "")
            }
            
        for item in code_data_java:
            self.code_data[f"java_{item['id']}"] = {
                "id": f"java_{item['id']}",
                "language": item["language"],
                "function_name": item["function_name"],
                "code": item["code"],
                "docstring": item["docstring"],
                "file_path": item["file_path"],
                "tokenized_function": item.get("tokenized_function", ""),
                "clean_code": item.get("clean_code", ""),
                "indexed_content": item.get("indexed_content", "")
            }
        
        # Create a temporary directory for indexing
        self.index_dir = tempfile.mkdtemp()
        self.docs_dir = os.path.join(self.index_dir, "docs")
        os.makedirs(self.docs_dir, exist_ok=True)
        
        # Create JSON documents for indexing
        for doc_id, doc in self.code_data.items():
            # Use the enhanced indexed_content field if available
            if "indexed_content" in doc and doc["indexed_content"]:
                content = doc["indexed_content"]
            else:
                # Fallback to the old method
                function_name = doc.get("tokenized_function", doc["function_name"])
                # Give function name more weight by repeating it
                weighted_function = f"{function_name} {function_name} {function_name}"
                language_prefix = f"{doc['language']}_language "
                content = f"{language_prefix}{weighted_function} {doc['docstring']} {doc['code']}"
            
            # Save as JSON document
            json_doc = {
                "id": doc_id,
                "contents": content,
                "language": doc["language"],
                "function_name": doc["function_name"]
            }
            
            with open(os.path.join(self.docs_dir, f"{doc_id}.json"), "w") as f:
                json.dump(json_doc, f)
        
        # Build the index
        result = subprocess.run(
            [
                "python", "-m", "pyserini.index.lucene",
                "-collection", "JsonCollection",
                "-generator", "DefaultLuceneDocumentGenerator",
                "-threads", "4",
                "-input", self.docs_dir,
                "-index", f"{self.index_dir}/index",
                "-storePositions", "-storeDocvectors", "-storeRaw"  # Enable more features
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print("[DEBUG] Indexing stdout:", result.stdout)
        print("[DEBUG] Indexing stderr:", result.stderr)

        if result.returncode != 0:
            raise RuntimeError("Indexing failed. Check above for details.")
        
        # Initialize searcher
        self.searcher = LuceneSearcher(f"{self.index_dir}/index")
        
        # Improve query expansion with synonyms for common programming terms
        self.searcher.set_bm25(0.9, 0.4)  # Use BM25 parameters optimized for code
    
    def preprocess_query(query):
        # Convert to lowercase
        query = query.lower().strip()
        
        # Extract language preference
        language_prefix = ""
        if "python" in query.lower():
            language_prefix = "python_language "
        elif "java" in query.lower():
            language_prefix = "java_language "
        
        # Handle camelCase and snake_case in programming identifiers
        # This helps match functions like "bubbleSort" or "binary_search"
        query = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', query)  # Split camelCase
        query = re.sub(r'_', ' ', query)  # Split snake_case
        
        # Remove special characters but preserve important coding symbols
        query = re.sub(r'[^\w\s\.\(\)\[\]_]', '', query)
        
        return language_prefix + query

    def retrieve(self, query, k=10):
        processed_query = query.lower().strip()
        processed_query = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', processed_query) # split camelCase
        processed_query = re.sub(r'_', ' ', processed_query) # split snake_case
        # Remove special characters but preserve important coding symbols
        processed_query = re.sub(r'[^\w\s\.\(\)\[\]_]', '', processed_query)
        if "python" in processed_query:
            processed_query = "python_language " + processed_query
        elif "java" in processed_query:
            processed_query = "java_language " + processed_query
        
        hits = self.searcher.search(processed_query, k)
        results = []
        
        for hit in hits:
            doc_id = hit.docid
            if doc_id in self.code_data:
                result = self.code_data[doc_id].copy()
                result["score"] = hit.score
                results.append(result)
        
        return results
    
    def cleanup(self):
        # Clean up temporary directories
        import shutil
        shutil.rmtree(self.index_dir)