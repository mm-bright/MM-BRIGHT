
import os
import io
import re
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image


class DataLoader:
    """
    Loads MM-BRIGHT data from HuggingFace hub.
    
    Key configs on HF:
    - 'documents': Corpus text documents (id, content)
    - 'examples': Task 1/2 queries (id, query, gold_ids, gold_answers, image_paths, ...)
    - 'examples_multimodal': Task 3/4 queries with additional fields 
    - 'document_images': Corpus images as bytes (path, bytes)
    - 'examples_images': Query images as bytes (path, bytes)
    """
    
    def __init__(self, dataset_name="mm-bright/MM-BRIGHT", hf_token=None, local_image_cache=None):
        self.dataset_name = dataset_name
        self.local_image_cache = local_image_cache  # Optional local cache for extracted images
        self._image_cache = {}  # In-memory cache: path -> PIL.Image
        
    def load_corpus_texts(self, domain):
        """
        Load corpus documents from HF dataset config 'documents'.
        Returns: list[str] doc_ids, list[str] documents
        """
        print(f"Loading {domain} corpus from HF ({self.dataset_name})...")
        try:
            ds = load_dataset(self.dataset_name, "documents", split=domain, trust_remote_code=True)
            doc_ids = list(ds['id'])
            documents = list(ds['content'])
            return doc_ids, documents
        except Exception as e:
            print(f"Error loading corpus for {domain}: {e}")
            import traceback
            traceback.print_exc()
            return [], []

    def load_queries(self, domain, config_name="examples"):
        """
        Load queries from HF dataset.
        
        config_name: 
            'examples' (Task 1: text queries)
            'examples_multimodal' (Task 2/3/4: multimodal queries with image_paths)
        
        Returns: queries, query_ids, gold_ids_map, excluded_ids_map, query_images_map
        """
        print(f"Loading {domain} queries ({config_name}) from HF...")
        try:
            # For examples_multimodal, we need to bypass schema enforcement
            # because the parquet has extra columns not in the README features
            if config_name == "examples_multimodal":
                # Load parquet directly via pandas to avoid schema cast error
                import pandas as pd
                from huggingface_hub import hf_hub_download
                
                parquet_path = hf_hub_download(
                    repo_id=self.dataset_name,
                    filename=f"examples_multimodal/{domain}.parquet",
                    repo_type="dataset"
                )
                df = pd.read_parquet(parquet_path)
                
                queries = df['query'].tolist()
                query_ids = df['id'].tolist()
                
                gold_ids_map = {}
                excluded_ids_map = {}
                query_images_map = {}
                positive_images_map = {}  # For Task 3 image retrieval
                negative_images_map = {}  # For Task 3 image retrieval
                
                for idx, row in df.iterrows():
                    qid = row['id']
                    
                    # Handle potential numpy arrays from parquet - convert to list
                    def to_list(val):
                        if val is None:
                            return []
                        if hasattr(val, 'tolist'):  # numpy array
                            return val.tolist()
                        if isinstance(val, list):
                            return val
                        return []
                    
                    # Extract positive_images (for Task 3) - handle dict format
                    def extract_image_paths(val):
                        """Extract image paths from positive_images/negative_images field.
                        Can be list of dicts with 'image_path' or list of strings."""
                        items = to_list(val)
                        paths = []
                        for x in items:
                            if isinstance(x, dict) and 'image_path' in x:
                                paths.append(str(x['image_path']).replace("\\", "/"))
                            elif isinstance(x, str):
                                paths.append(x.replace("\\", "/"))
                        return paths
                    
                    gold_ids_map[qid] = to_list(row.get('gold_ids'))
                    excluded_ids_map[qid] = to_list(row.get('negative_ids'))
                    query_images_map[qid] = to_list(row.get('image_paths'))
                    positive_images_map[qid] = extract_image_paths(row.get('positive_images'))
                    negative_images_map[qid] = extract_image_paths(row.get('negative_images'))
                    
                    # Debug first row
                    if idx == 0:
                        print(f"[DEBUG data.py] Columns available: {list(df.columns)}")
                        print(f"[DEBUG data.py] positive_images raw value: {row.get('positive_images')}")
                
                return queries, query_ids, gold_ids_map, excluded_ids_map, query_images_map, positive_images_map, negative_images_map
            
            else:
                # For 'examples' config, normal loading works fine
                ds = load_dataset(
                    self.dataset_name, 
                    config_name, 
                    split=domain, 
                    trust_remote_code=True,
                )
                
                queries = list(ds['query'])
                query_ids = list(ds['id'])
                
                gold_ids_map = {}
                excluded_ids_map = {}
                query_images_map = {}
                
                for row in ds:
                    qid = row['id']
                    gold_ids_map[qid] = row.get('gold_ids', []) or []
                    excluded_ids_map[qid] = row.get('negative_ids', []) or []
                    img_paths = row.get('image_paths', []) or []
                    query_images_map[qid] = img_paths if img_paths else []
                
                return queries, query_ids, gold_ids_map, excluded_ids_map, query_images_map
            
        except Exception as e:
            print(f"Error loading queries for {domain}: {e}")
            import traceback
            traceback.print_exc()
            return [], [], {}, {}, {}

    def load_corpus_images(self, domain):
        """
        Load corpus/document images from HF 'document_images' config.
        Returns: dict[path] -> PIL.Image
        """
        print(f"Loading {domain} document images from HF...")
        try:
            ds = load_dataset(self.dataset_name, "document_images", split=domain, trust_remote_code=True)
            
            images_map = {}
            for row in tqdm(ds, desc=f"Decoding {domain} document images"):
                path = row['path']
                img_bytes = row['bytes']
                try:
                    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    images_map[path] = img
                except Exception as e:
                    print(f"Warning: Could not decode image {path}: {e}")
                    
            return images_map
        except Exception as e:
            print(f"Error loading document images for {domain}: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def load_query_images(self, domain):
        """
        Load query images from HF 'examples_images' config.
        Returns: dict[path] -> PIL.Image
        """
        print(f"Loading {domain} query images from HF...")
        try:
            ds = load_dataset(self.dataset_name, "examples_images", split=domain, trust_remote_code=True)
            
            images_map = {}
            for row in tqdm(ds, desc=f"Decoding {domain} query images"):
                path = row['path']
                img_bytes = row['bytes']
                try:
                    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    images_map[path] = img
                except Exception as e:
                    print(f"Warning: Could not decode image {path}: {e}")
                    
            return images_map
        except Exception as e:
            print(f"Error loading query images for {domain}: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def get_query_pil_images(self, query_images_map, query_images_data):
        """
        Resolve query image paths to actual PIL images.
        
        Args:
            query_images_map: dict[qid] -> list[path] (from load_queries)
            query_images_data: dict[path] -> PIL.Image (from load_query_images)
        
        Returns: dict[qid] -> list[PIL.Image]
        """
        result = {}
        for qid, paths in query_images_map.items():
            images = []
            for p in paths:
                if p in query_images_data:
                    images.append(query_images_data[p])
                else:
                    print(f"Warning: Query image not found: {p}")
            result[qid] = images
        return result

    def load_corpus_image_paths_from_local(self, domain, passage_images_dir):
        """
        FALLBACK: Load image paths from local directory.
        Use this if you have images stored locally instead of using HF bytes.
        """
        from .utils import IMG_EXTS, is_readable_image
        
        passage_images_dir = Path(passage_images_dir)
        domain_dir = passage_images_dir / domain
        
        image_ids = []
        image_paths = []
        
        if not domain_dir.exists():
            print(f"Warning: Local image directory not found: {domain_dir}")
            return [], []

        files = sorted([p for p in domain_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])
        
        valid_files = []
        for p in tqdm(files, desc=f"Scanning {domain} local images"):
            if is_readable_image(p):
                valid_files.append(p)
                
        for p in valid_files:
            rel_id = p.relative_to(passage_images_dir).as_posix()
            image_ids.append(rel_id)
            image_paths.append(str(p))
             
        return image_ids, image_paths

    def build_it_it_pairs(self, passage_ids, passage_texts, corpus_images_map, domain):
        """
        Build (text, image) pairs for Task 4 (IT→IT retrieval).
        
        Args:
            passage_ids: list of document IDs
            passage_texts: list of document texts
            corpus_images_map: dict[path] -> PIL.Image (from load_corpus_images)
            domain: domain name
            
        Returns: pair_ids, pair_texts, pair_images (PIL or None), base_to_imgs dict
        """
        from .utils import base_key_from_passage_id, base_key_from_image_rel
        
        # Index images by base key (extracted from path)
        base_to_imgs = {}
        for img_path in corpus_images_map.keys():
            bk = base_key_from_image_rel(img_path, domain)
            if bk:
                base_to_imgs.setdefault(bk, []).append(img_path)
        
        # Build pairs
        pair_ids, pair_texts, pair_images = [], [], []
        
        for pid, txt in zip(passage_ids, passage_texts):
            # NO_IMAGE pair (text-only)
            pair_ids.append(f"{pid}|||__NO_IMAGE__")
            pair_texts.append(txt)
            pair_images.append(None)
            
            # Image pairs
            bk = base_key_from_passage_id(pid)
            for img_path in base_to_imgs.get(bk, []):
                if img_path in corpus_images_map:
                    pair_ids.append(f"{pid}|||{img_path}")
                    pair_texts.append(txt)
                    pair_images.append(corpus_images_map[img_path])
                     
        return pair_ids, pair_texts, pair_images, base_to_imgs
