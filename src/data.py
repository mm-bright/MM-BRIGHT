
import os
import re
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from .utils import norm_id, base_key_from_passage_id, base_key_from_image_rel, is_readable_image

class DataLoader:
    def __init__(self, dataset_name="mm-bright/MM-BRIGHT", hf_token=None):
        self.dataset_name = dataset_name
        # Note: HF token usually handled by environment variable or login, but can pass if needed
        
    def load_corpus_texts(self, domain):
        """
        Load corpus documents from HF dataset config 'documents'.
        Returns: doc_ids, documents
        """
        print(f"Loading {domain} corpus from HF ({self.dataset_name})...")
        try:
            # Check if we can filter by 'domain' field if it exists, otherwise filtering is tricky if monolithic.
            # Based on user's prepre.py, data is partitioned by domain now.
            # HF dataset loading for a specific split/subset
            # Assuming "documents" config has splits named by domain or we filter.
            # verify_hf_dataset output showed splits: academia, bitcoin, etc. under 'documents' config?
            # Actually prepre.py creates separate parquets per domain.
            # Verify script showed configs: 'documents', 'examples', etc.
            # And under 'documents', splits are likely the domains.
            ds = load_dataset(self.dataset_name, "documents", split=domain, trust_remote_code=True)
            
            doc_ids = ds['id']
            documents = ds['content']
            
            return doc_ids, documents
        except Exception as e:
            print(f"Error loading corpus for {domain}: {e}")
            return [], []

    def load_queries(self, domain, config_name="examples"):
        """
        Load queries from HF dataset.
        config_name: 'examples' (Task 1/2) or 'examples_multimodal' (Task 3/4)
        """
        print(f"Loading {domain} queries ({config_name}) from HF...")
        try:
            ds = load_dataset(self.dataset_name, config_name, split=domain, trust_remote_code=True)
            
            # Convert dataset rows to lists/maps expected by evaluation
            queries = ds['query']
            query_ids = ds['id']
            
            # Reconstruct maps
            gold_ids_map = {}
            excluded_ids_map = {}
            query_images_map = {} # For multimodal queries
            
            # Using iterate to handle potential missing columns gracefully or varying structure
            for row in ds:
                qid = row['id']
                gold_ids_map[qid] = row['gold_ids']
                
                # exclusions (Task 1/2: negative_ids, Task 3/4: negative_images paths?)
                
                # Handle text negatives
                if 'negative_ids' in row and row['negative_ids']:
                    excluded_ids_map[qid] = row['negative_ids']
                else:
                    excluded_ids_map[qid] = []
                
                # Handle query images (for Task 2/3)
                if 'image_paths' in row:
                    query_images_map[qid] = row['image_paths']
                elif 'positive_images' in row: # Fallback (though positive_images usually means ground truth)
                     # If image_paths is missing, we might be misinterpreting. 
                     # But for now assume fallback logic was for local files where field names differ.
                     query_images_map[qid] = row['positive_images']
                else:
                    query_images_map[qid] = []
            
            return queries, query_ids, gold_ids_map, excluded_ids_map, query_images_map
            
        except Exception as e:
            print(f"Error loading queries for {domain}: {e}")
            return [], [], {}, {}, {}

    def load_image_corpus_paths(self, domain, passage_images_dir):
        """
        Load image paths from local directory (since pixels aren't in HF parquet).
        This mimics the original scripts' behavior of scanning directories.
        """
        passage_images_dir = Path(passage_images_dir)
        domain_dir = passage_images_dir / domain
        
        image_ids = []
        image_paths = []
        
        if not domain_dir.exists():
            return [], []

        # Use utils for consistency
        from .utils import IMG_EXTS, is_readable_image
        
        files = sorted([p for p in domain_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])
        
        valid_files = []
        for p in tqdm(files, desc=f"Scanning {domain} images"):
            if is_readable_image(p):
                valid_files.append(p)
                
        for p in valid_files:
             rel_id = p.relative_to(passage_images_dir).as_posix()
             image_ids.append(rel_id)
             image_paths.append(str(p))
             
        return image_ids, image_paths

    def build_it_it_pairs(self, passage_ids, passage_texts, passage_images_dir, domain):
        """
        Logic from run_task4.py to build (text, image) pairs.
        """
        passage_images_dir = Path(passage_images_dir)
        
        # 1. Index images
        base_to_imgs = {}
        img_rel_to_abs = {}
        
        domain_dir = passage_images_dir / domain
        if domain_dir.exists():
            from .utils import IMG_EXTS, is_readable_image, norm_id, base_key_from_image_rel, base_key_from_passage_id
            
            files = sorted([p for p in domain_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])
            for p in files:
                if is_readable_image(p):
                    rel = norm_id(p.relative_to(passage_images_dir).as_posix())
                    img_rel_to_abs[rel] = str(p)
                    bk = base_key_from_image_rel(rel, domain)
                    if bk:
                        base_to_imgs.setdefault(bk, []).append(rel)
        
        # 2. Build pairs
        pair_ids, pair_texts, pair_img_abs = [], [], []
        
        for pid, txt in zip(passage_ids, passage_texts):
             # NO_IMAGE pair
             pair_ids.append(f"{pid}|||__NO_IMAGE__")
             pair_texts.append(txt)
             pair_img_abs.append(None)
             
             # Image pairs
             from .utils import base_key_from_passage_id
             bk = base_key_from_passage_id(pid)
             for img_rel in base_to_imgs.get(bk, []):
                 if img_rel in img_rel_to_abs:
                     pair_ids.append(f"{pid}|||{img_rel}")
                     pair_texts.append(txt)
                     pair_img_abs.append(img_rel_to_abs[img_rel])
                     
        return pair_ids, pair_texts, pair_img_abs, base_to_imgs, img_rel_to_abs

