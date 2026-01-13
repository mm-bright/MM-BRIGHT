
import os
import json
import numpy as np
from pathlib import Path

class SmartCache:
    """
    Handles robust caching of embeddings with automatic updates for new documents.
    Replaces the repeated 50+ lines of caching logic in retrievers.
    """
    
    def __init__(self, cache_dir, model_id, task, long_context, batch_size, sub_dir='doc_emb'):
        self.cache_dir = Path(cache_dir)
        self.model_id = model_id
        self.task = task
        self.long_context = long_context
        self.batch_size = batch_size
        self.sub_dir = sub_dir
        
        # Define cache paths
        # Define cache paths
        self.cache_dir_path = self.cache_dir / self.sub_dir / self.model_id / self.task / f"long_{self.long_context}_{self.batch_size}"
        self.embeddings_path = self.cache_dir_path / 'embeddings.npy'
        self.mapping_path = self.cache_dir_path / 'doc_id_mapping.json'
        
        self.cache_dir_path.mkdir(parents=True, exist_ok=True)
        
        self.cached_embeddings = {}
        self.doc_id_to_index = {}
        self.is_loaded = False

    def load(self, ignore_cache=False):
        """Load existing cache from disk."""
        if not ignore_cache and self.embeddings_path.exists() and self.mapping_path.exists():
            print("Loading existing cache...")
            try:
                cached_emb_array = np.load(self.embeddings_path, allow_pickle=True)
                with open(self.mapping_path, 'r') as f:
                    self.doc_id_to_index = json.load(f)
                
                # Reconstruct dict map
                for doc_id, idx in self.doc_id_to_index.items():
                    # Boundary check
                    if idx < len(cached_emb_array):
                         self.cached_embeddings[doc_id] = cached_emb_array[idx]
                
                print(f"Loaded {len(self.cached_embeddings)} cached embeddings")
                self.is_loaded = True
            except Exception as e:
                print(f"Error loading cache: {e}. Starting fresh.")
                self.cached_embeddings = {}
                self.doc_id_to_index = {}

    def get_uncached_docs(self, doc_ids, documents):
        """Identify which documents need to be encoded."""
        docs_to_encode = []
        docs_to_encode_ids = []
        
        for doc_id, doc_text in zip(doc_ids, documents):
            if doc_id not in self.cached_embeddings:
                docs_to_encode.append(doc_text)
                docs_to_encode_ids.append(doc_id)
        
        print(f"Documents in corpus: {len(documents)}")
        print(f"Already cached: {len(self.cached_embeddings)}")
        print(f"Need to encode: {len(docs_to_encode)}")
        
        return docs_to_encode, docs_to_encode_ids

    def update(self, new_embeddings, new_ids):
        """Update in-memory cache with new embeddings."""
        # Check for empty/None embeddings properly (numpy arrays can't use truthiness directly)
        if new_embeddings is None or (hasattr(new_embeddings, '__len__') and len(new_embeddings) == 0):
            return
            
        # If list of numpy arrays, concatenate
        if isinstance(new_embeddings, list) and len(new_embeddings) > 0 and isinstance(new_embeddings[0], np.ndarray):
            new_embeddings = np.concatenate(new_embeddings, axis=0)
        
        # If tensor, convert to numpy
        if hasattr(new_embeddings, 'cpu'):
            new_embeddings = new_embeddings.cpu().numpy()
            
        next_index = len(self.cached_embeddings)
        for i, doc_id in enumerate(new_ids):
            self.cached_embeddings[doc_id] = new_embeddings[i]
            self.doc_id_to_index[doc_id] = next_index + i

    def save(self):
        """Save current cache state to disk."""
        print("Saving updated cache...")
        # Sort by index to ensure array order
        sorted_ids = sorted(self.doc_id_to_index.keys(), key=lambda x: self.doc_id_to_index[x])
        all_embeddings = np.array([self.cached_embeddings[doc_id] for doc_id in sorted_ids])
        
        np.save(self.embeddings_path, all_embeddings)
        
        with open(self.mapping_path, 'w') as f:
            json.dump(self.doc_id_to_index, f, indent=2)
        
        print(f"Cache updated: {len(self.cached_embeddings)} total embeddings")

    def get_embeddings_array(self, requested_doc_ids):
        """Get the final numpy/tensor array for the requested docs in order."""
        # This assumes all requested IDs are now in cache (after update)
        embs = [self.cached_embeddings[doc_id] for doc_id in requested_doc_ids]
        return np.array(embs)
