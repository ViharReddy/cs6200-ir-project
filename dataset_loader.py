import json
import pandas as pd
import random

def load_subset_dataset(python_path, java_path, max_items_per_language=3000, seed=42):
    """
    Load a subset of the code datasets to improve processing time.
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Load Python data
    with open(python_path, 'r') as f:
        python_data = json.load(f)
    
    # Load Java data
    with open(java_path, 'r') as f:
        java_data = json.load(f)
    
    # Sample a subset of the data
    if len(python_data) > max_items_per_language:
        python_data = random.sample(python_data, max_items_per_language)
    
    if len(java_data) > max_items_per_language:
        java_data = random.sample(java_data, max_items_per_language)
    
    print(f"Loaded {len(python_data)} Python samples and {len(java_data)} Java samples")
    
    return python_data, java_data