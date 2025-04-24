#bm25_retrieval.py
import json
import os
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
                "file_path": item["file_path"]
            }
            
        for item in code_data_java:
            self.code_data[f"java_{item['id']}"] = {
                "id": f"java_{item['id']}",
                "language": item["language"],
                "function_name": item["function_name"],
                "code": item["code"],
                "docstring": item["docstring"],
                "file_path": item["file_path"]
            }
        
        # Create a temporary directory for indexing
        self.index_dir = tempfile.mkdtemp()
        self.docs_dir = os.path.join(self.index_dir, "docs")
        os.makedirs(self.docs_dir, exist_ok=True)
        
        # Create JSON documents for indexing
        for doc_id, doc in self.code_data.items():
            # Create content combining function name, docstring, and code
            content = f"{doc['function_name']} {doc['docstring']} {doc['code']}"
            
            # Save as JSON document
            json_doc = {
                "id": doc_id,
                "contents": content,
                "language": doc["language"],
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
                "-index", f"{self.index_dir}/index"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print("[DEBUG] Indexing stdout:", result.stdout)
        print("[DEBUG] Indexing stderr:", result.stderr)

        if result.returncode != 0:
            raise RuntimeError("Indexing failed. Check above for details.")
        
        print(f"[DEBUG] Looking for Lucene index at: {self.index_dir}/index")
        print("[DEBUG] Exists:", os.path.isdir(f"{self.index_dir}/index"))
        print("[DEBUG] Contents:", os.listdir(self.index_dir) if os.path.exists(self.index_dir) else "Directory doesn't exist")
        # Initialize searcher
        self.searcher = LuceneSearcher(f"{self.index_dir}/index")
    
    def retrieve(self, query, k=10):
        hits = self.searcher.search(query, k)
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