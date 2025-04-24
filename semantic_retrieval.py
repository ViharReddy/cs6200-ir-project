import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
import os
import pickle

class GraphCodeBERTRetriever:
    def __init__(self, code_data_python, code_data_java, cache_dir="cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.code_data = {}
        self.embeddings_file = os.path.join(cache_dir, "codebert_embeddings.pkl")
        
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
        
        # Load CodeBERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
        self.model = AutoModel.from_pretrained("microsoft/graphcodebert-base")
        
        # Load or generate embeddings
        if os.path.exists(self.embeddings_file):
            print("Loading cached embeddings...")
            with open(self.embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)
        else:
            print("Generating embeddings (this may take a while)...")
            self.embeddings = self._generate_embeddings()
            
            # Save embeddings for future use
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
    
    def _generate_embeddings(self, batch_size=64):
        """Generate embeddings for all code snippets with larger batch size"""
        embeddings = {}
        doc_ids = list(self.code_data.keys())
        
        # Use larger batches and GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        
        for i in tqdm(range(0, len(doc_ids), batch_size), desc="Generating embeddings"):
            batch_ids = doc_ids[i:i+batch_size]
            batch_texts = []
            
            for doc_id in batch_ids:
                entry = self.code_data[doc_id]
                language = entry['language']
                func_name = entry['function_name']
                docstring = entry['docstring'] or ""
                code = entry['code'] or ""
                code_snippet = code[:300]

                # create a more structured representation
                text = f"language: {language} function: {func_name} description: {docstring} code: {code_snippet}"
                batch_texts.append(text)
            
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=384,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
            
            for idx, doc_id in enumerate(batch_ids):
                embeddings[doc_id] = batch_embeddings[idx]
        
        return embeddings
    
    def _get_query_embedding(self, query):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        query_lower = query.lower()
        if "python" in query_lower:
            language = "python"
        elif "java" in query_lower:
            language = "java"
        else:
            language = "unknown"
        
        structured_query = f"language: {language} query: {query}"

        inputs = self.tokenizer(structured_query, return_tensors="pt", padding=True, truncation=True, max_length=384)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            query_embedding = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()[0]
        
        return query_embedding
    
    def retrieve(self, query, k=10):
        query_embedding = self._get_query_embedding(query)

        query_lower = query.lower()
        preferred_language = None
        if "python" in query_lower:
            preferred_language = "python"
        elif "java" in query_lower:
            preferred_language = "java"
        
        # Calculate cosine similarity with all code snippets
        similarities = {}
        for doc_id, doc_embedding in self.embeddings.items():
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )

            # apply language preference boost only
            if preferred_language and doc_id.startswith(preferred_language):
                similarity *= 1.25 # 25% boost for language match
            similarities[doc_id] = similarity
        
        # Get top-k results
        top_doc_ids = sorted(similarities.keys(), key=lambda x: similarities[x], reverse=True)[:k]
        
        results = []
        for doc_id in top_doc_ids:
            result = self.code_data[doc_id].copy()
            result["score"] = float(similarities[doc_id])
            results.append(result)
        
        return results