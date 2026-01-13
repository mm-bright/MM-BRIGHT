# multimodal_retrievers.py
"""
Multimodal Retrieval Models for Image+Text Query -> Text Document Retrieval
Each model is implemented based on official documentation and HuggingFace implementations.
"""

import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm, trange
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from src.caching import SmartCache
from src.utils import cut_text, add_instruct_concatenate, last_token_pool

# Add this to your retrievers.py file

from PIL import Image
import requests
from io import BytesIO
import clip  # pip install git+https://github.com/openai/CLIP.git

@torch.no_grad()
def retrieval_clip(queries, query_ids, documents, doc_ids, task, model_id, instructions, cache_dir, excluded_ids, long_context, **kwargs):
    """
    CLIP-based multimodal retrieval: query text + image -> text documents
    """
    from transformers import CLIPProcessor, CLIPModel
    
    # Load CLIP model
    model_name = kwargs.get('model_name', 'openai/clip-vit-large-patch14')
    model = CLIPModel.from_pretrained(model_name).to('cuda')
    processor = CLIPProcessor.from_pretrained(model_name)
    
    batch_size = kwargs.get('batch_size', 8)
    image_dir = Path(kwargs.get('dataset_dir', '.')) / 'images' / task
    
    # ==================== DOCUMENT EMBEDDINGS (TEXT ONLY) ====================
    cache = SmartCache(cache_dir, f'{model_id}_clip', task, long_context, batch_size)
    cache.load()
    
    docs_to_encode, docs_to_encode_ids = cache.get_uncached_docs(doc_ids, documents)
    
    if len(docs_to_encode) > 0:
        print("Encoding new documents (text-only)...")
        new_embeddings = []
        
        for i in trange(0, len(docs_to_encode), batch_size, desc="Encoding documents"):
            batch_texts = docs_to_encode[i:i+batch_size]
            
            inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            outputs = model.get_text_features(**inputs)
            embeddings = F.normalize(outputs, p=2, dim=1).cpu().numpy()
            new_embeddings.append(embeddings)
        
        cache.update(new_embeddings, docs_to_encode_ids)
        cache.save()
    
    doc_emb = cache.get_embeddings_array(doc_ids)
    doc_emb = torch.tensor(doc_emb)
    doc_emb = F.normalize(doc_emb, p=2, dim=1)
    
    # ==================== QUERY EMBEDDINGS (TEXT + IMAGE) ====================
    # ==================== QUERY EMBEDDINGS (TEXT + IMAGE) ====================
    print("\nEncoding queries with images...")
    query_emb = []
    
    query_images_map = kwargs.get('query_images_map', {})
    
    for i in trange(0, len(queries), batch_size, desc="Encoding queries"):
        batch_queries = queries[i:i+batch_size]
        batch_query_ids = query_ids[i:i+batch_size]
        
        batch_images = []
        batch_texts = []
        
        for qid, query_text in zip(batch_query_ids, batch_queries):
            # Get images for this query (now PIL.Image objects from HF)
            images = query_images_map.get(qid, [])
            
            if images:
                first_img = images[0]
                # Check if already PIL Image
                if hasattr(first_img, 'convert'):
                    batch_images.append(first_img.convert('RGB'))
                else:
                    # Fallback for unexpected type
                    batch_images.append(Image.new('RGB', (224, 224), color='white'))
            else:
                # Text-only query (no image)
                batch_images.append(Image.new('RGB', (224, 224), color='white'))
            batch_texts.append(query_text)
        
        # Process multimodal inputs
        inputs = processor(
            text=batch_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Get combined text and image features
        text_features = model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        image_features = model.get_image_features(pixel_values=inputs['pixel_values'])
        
        # Combine features (average fusion)
        combined_features = (text_features + image_features) / 2
        combined_features = F.normalize(combined_features, p=2, dim=1)
        
        query_emb.append(combined_features.cpu())
    
    query_emb = torch.cat(query_emb, dim=0)
    print("query_emb shape:", query_emb.shape)
    
    # Compute scores
    scores = (query_emb @ doc_emb.T) * 100
    scores = scores.tolist()
    
    return get_scores(query_ids=query_ids, doc_ids=doc_ids, scores=scores, excluded_ids=excluded_ids)


@torch.no_grad()
def retrieval_siglip(queries, query_ids, documents, doc_ids, task, model_id, instructions, cache_dir, excluded_ids, long_context, **kwargs):
    """
    SigLIP-based multimodal retrieval (Google's improved CLIP)
    """
    from transformers import AutoProcessor, AutoModel
    
    model_name = kwargs.get('model_name', 'google/siglip-so400m-patch14-384')
    model = AutoModel.from_pretrained(model_name).to('cuda')
    processor = AutoProcessor.from_pretrained(model_name)
    
    batch_size = kwargs.get('batch_size', 8)
    image_dir = Path(kwargs.get('dataset_dir', '.')) / 'images' / task
    
    # ==================== DOCUMENT EMBEDDINGS ====================
    cache = SmartCache(cache_dir, f'{model_id}_siglip', task, long_context, batch_size)
    cache.load()
    
    docs_to_encode, docs_to_encode_ids = cache.get_uncached_docs(doc_ids, documents)
    
    if len(docs_to_encode) > 0:
        print("Encoding new documents...")
        new_embeddings = []
        
        for i in trange(0, len(docs_to_encode), batch_size, desc="Encoding documents"):
            batch_texts = docs_to_encode[i:i+batch_size]
            
            inputs = processor(text=batch_texts, return_tensors="pt", padding="max_length", truncation=True)
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            outputs = model.get_text_features(**inputs)
            embeddings = F.normalize(outputs, p=2, dim=1).cpu().numpy()
            new_embeddings.append(embeddings)
        
        cache.update(new_embeddings, docs_to_encode_ids)
        cache.save()
    
    doc_emb = cache.get_embeddings_array(doc_ids)
    doc_emb = torch.tensor(doc_emb)
    doc_emb = F.normalize(doc_emb, p=2, dim=1)
    
    # ==================== QUERY EMBEDDINGS (MULTIMODAL) ====================
    print("\nEncoding multimodal queries...")
    query_emb = []
    
    # ==================== QUERY EMBEDDINGS (MULTIMODAL) ====================
    print("\nEncoding multimodal queries...")
    query_emb = []
    
    query_images_map = kwargs.get('query_images_map', {})
    
    for i in trange(0, len(queries), batch_size, desc="Encoding queries"):
        batch_queries = queries[i:i+batch_size]
        batch_query_ids = query_ids[i:i+batch_size]
        
        batch_images = []
        batch_texts = []
        
        for qid, query_text in zip(batch_query_ids, batch_queries):
            # Get images for this query (now PIL.Image objects from HF)
            images = query_images_map.get(qid, [])
            
            if images:
                first_img = images[0]
                if hasattr(first_img, 'convert'):
                    batch_images.append(first_img.convert('RGB'))
                else:
                    batch_images.append(Image.new('RGB', (384, 384), color='white'))
            else:
                batch_images.append(Image.new('RGB', (384, 384), color='white'))
            
            batch_texts.append(query_text)
        
        inputs = processor(
            text=batch_texts,
            images=batch_images,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        text_features = model.get_text_features(input_ids=inputs['input_ids']) #, attention_mask=inputs['attention_mask']
        image_features = model.get_image_features(pixel_values=inputs['pixel_values'])
        
        # Weighted fusion (you can adjust weights)
        combined_features = 0.5 * text_features + 0.5 * image_features
        combined_features = F.normalize(combined_features, p=2, dim=1)
        
        query_emb.append(combined_features.cpu())
    
    query_emb = torch.cat(query_emb, dim=0)
    
    scores = (query_emb @ doc_emb.T) * 100
    scores = scores.tolist()
    
    return get_scores(query_ids=query_ids, doc_ids=doc_ids, scores=scores, excluded_ids=excluded_ids)


@torch.no_grad()
def retrieval_jina_clip(queries, query_ids, documents, doc_ids, task, model_id, instructions, cache_dir, excluded_ids, long_context, **kwargs):
    """
    Jina CLIP v1 - optimized for retrieval tasks
    """
    from transformers import AutoModel
    from PIL import Image
    
    model_name = 'jinaai/jina-clip-v1'
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to('cuda')
    
    batch_size = kwargs.get('batch_size', 8)
    image_dir = Path(kwargs.get('dataset_dir', '.')) / 'images' / task
    
    # ==================== DOCUMENT EMBEDDINGS ====================
    cache = SmartCache(cache_dir, 'jina_clip', task, long_context, batch_size)
    cache.load()
    
    docs_to_encode, docs_to_encode_ids = cache.get_uncached_docs(doc_ids, documents)
    
    print(f"Need to encode: {len(docs_to_encode)} documents")
    
    if len(docs_to_encode) > 0:
        print("Encoding new documents...")
        new_embeddings = []
        
        for i in trange(0, len(docs_to_encode), batch_size, desc="Encoding documents"):
            batch_texts = docs_to_encode[i:i+batch_size]
            
            # Jina CLIP has encode_text method
            embeddings = model.encode_text(batch_texts)
            if isinstance(embeddings, np.ndarray):
                new_embeddings.append(embeddings)
            else:
                new_embeddings.append(embeddings.cpu().numpy())

        cache.update(new_embeddings, docs_to_encode_ids)
        cache.save()
    
    doc_emb = cache.get_embeddings_array(doc_ids)
    # Convert numpy to tensor if needed
    if isinstance(doc_emb, np.ndarray):
        doc_emb = torch.from_numpy(doc_emb).float()
    doc_emb = F.normalize(doc_emb, p=2, dim=1)
    
    # ==================== QUERY EMBEDDINGS (MULTIMODAL) ====================
    # ==================== QUERY EMBEDDINGS (MULTIMODAL) ====================
    print("\nEncoding multimodal queries...")
    
    query_images_map = kwargs.get('query_images_map', {})
    
    query_emb = []
    
    for i in trange(0, len(queries), batch_size, desc="Encoding queries"):
        batch_queries = queries[i:i+batch_size]
        batch_query_ids = query_ids[i:i+batch_size]
        
        batch_images = []
        batch_texts = []
        
        for qid, query_text in zip(batch_query_ids, batch_queries):
            # Get images for this query (now PIL.Image objects from HF)
            images = query_images_map.get(qid, [])
            
            if images:
                first_img = images[0]
                if hasattr(first_img, 'convert'):
                    batch_images.append(first_img.convert('RGB'))
                else:
                    batch_images.append(None)
            else:
                batch_images.append(None)
            
            batch_texts.append(query_text)
        
        # Encode text
        text_embeds = model.encode_text(batch_texts)
        # Ensure torch.Tensor
        if isinstance(text_embeds, np.ndarray):
            text_embeds = torch.from_numpy(text_embeds)
        else:
            text_embeds = text_embeds.detach().cpu()
        
        # Encode images (only for queries that have images)
        image_embeds_list = []
        for img in batch_images:
            if img is not None:
                img_embed = model.encode_image([img])
                # img_embed could be np or torch
                if isinstance(img_embed, np.ndarray):
                    img_embed = torch.from_numpy(img_embed)
                else:
                    img_embed = img_embed.detach().cpu()
                # take first in batch: shape (dim,)
                image_embeds_list.append(img_embed[0])
            else:
                # Use zero vector for queries without images (same type/shape as text_embeds[0])
                image_embeds_list.append(torch.zeros_like(text_embeds[0]))
        
        image_embeds = torch.stack(image_embeds_list, dim=0)
        
        # Combine text and image embeddings
        combined = (text_embeds + image_embeds) / 2
        combined = F.normalize(combined, p=2, dim=1)
        
        query_emb.append(combined.cpu())
    
    query_emb = torch.cat(query_emb, dim=0)

    
    scores = (query_emb @ doc_emb.T) * 100
    scores = scores.tolist()
    
    return get_scores(query_ids=query_ids, doc_ids=doc_ids, scores=scores, excluded_ids=excluded_ids)



def load_query_images(query_ids, query_images_map, image_dir=None):
    """Helper function to load images for queries.
    
    query_images_map can now contain either:
    - list of PIL.Image objects (from HF bytes loading)
    - list of file path strings (from local files)
    """
    query_images = []
    images_found = 0
    
    for qid in query_ids:
        images = query_images_map.get(qid, [])
        
        if images:
            first_img = images[0]
            
            # Already a PIL Image
            if hasattr(first_img, 'convert'):
                query_images.append(first_img.convert('RGB') if first_img.mode != 'RGB' else first_img)
                images_found += 1
            # It's a file path string
            elif isinstance(first_img, str):
                try:
                    from pathlib import Path
                    if image_dir:
                        img_path = Path(image_dir) / first_img.split('/')[-1]
                    else:
                        img_path = Path(first_img)
                    if img_path.exists():
                        query_images.append(Image.open(img_path).convert('RGB'))
                        images_found += 1
                    else:
                        print(f"Warning: Image not found: {img_path}")
                        query_images.append(None)
                except Exception as e:
                    print(f"Error loading image: {e}")
                    query_images.append(None)
            else:
                print(f"Warning: Unknown image type for qid {qid}: {type(first_img)}")
                query_images.append(None)
        else:
            query_images.append(None)
    
    print(f"[load_query_images] Loaded {images_found}/{len(query_ids)} images from query_images_map")
    if images_found == 0 and query_images_map:
        # Debug: print sample keys
        sample_keys = list(query_images_map.keys())[:3]
        sample_qids = list(query_ids)[:3]
        print(f"[DEBUG] Sample query_images_map keys: {sample_keys}")
        print(f"[DEBUG] Sample query_ids: {sample_qids}")
    
    return query_images


def get_scores(query_ids, doc_ids, scores, excluded_ids):
    """Convert scores to required format and filter excluded IDs"""
    assert len(scores) == len(query_ids), f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0]) == len(doc_ids), f"{len(scores[0])}, {len(doc_ids)}"
    
    emb_scores = {}
    for query_id, doc_scores in zip(query_ids, scores):
        cur_scores = {}
        for did, s in zip(doc_ids, doc_scores):
            cur_scores[str(did)] = s
        
        # Remove excluded IDs
        for did in set(excluded_ids.get(str(query_id), [])):
            if did != "N/A" and did in cur_scores:
                cur_scores.pop(did)
        
        # Keep top 1000
        cur_scores = sorted(cur_scores.items(), key=lambda x: x[1], reverse=True)[:1000]
        emb_scores[str(query_id)] = {pair[0]: pair[1] for pair in cur_scores}
    
    return emb_scores


# ==================== NOMIC EMBED VISION ====================
@torch.no_grad()
def retrieval_nomic_vision(queries, query_ids, documents, doc_ids, task, model_id, instructions, 
                          cache_dir, excluded_ids, long_context, **kwargs):
    """
    Nomic Embed Vision v1.5 + Text v1.5
    Vision model: nomic-ai/nomic-embed-vision-v1.5 (for images)
    Text model: nomic-ai/nomic-embed-text-v1.5 (for text)
    They share the same embedding space!
    """
    from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
    
    print("\n" + "="*80)
    print("NOMIC EMBED VISION V1.5 + TEXT V1.5")
    print("="*80)
    
    # Load TEXT model for documents and text queries
    text_model_name = 'nomic-ai/nomic-embed-text-v1.5'
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_model = AutoModel.from_pretrained(text_model_name, trust_remote_code=True).to('cuda')
    text_model.eval()
    
    # Load VISION model for images
    vision_model_name = 'nomic-ai/nomic-embed-vision-v1.5'
    vision_processor = AutoImageProcessor.from_pretrained(vision_model_name)
    vision_model = AutoModel.from_pretrained(vision_model_name, trust_remote_code=True).to('cuda')
    vision_model.eval()
    
    batch_size = kwargs.get('batch_size', 8)
    dataset_dir = Path(kwargs.get('dataset_dir', '.'))
    image_dir = dataset_dir / 'images' / task
    
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # ==================== CACHE DOCUMENT EMBEDDINGS ====================
    cache = SmartCache(cache_dir, 'nomic_vision', task, long_context, batch_size)
    cache.load()
    
    docs_to_encode, docs_to_encode_ids = cache.get_uncached_docs(doc_ids, documents)
    
    if len(docs_to_encode) > 0:
        print(f"Encoding {len(docs_to_encode)} new documents with TEXT model...")
        new_embeddings = []
        
        for i in trange(0, len(docs_to_encode), batch_size, desc="Documents"):
            batch_texts = docs_to_encode[i:i+batch_size]
            batch_texts = [text[:8000] for text in batch_texts]
            
            # Use TEXT model for documents
            encoded_input = text_tokenizer(batch_texts, padding=True, truncation=True, 
                                          return_tensors='pt', max_length=8192).to('cuda')
            
            model_output = text_model(**encoded_input)
            embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Nomic uses layer norm + L2 norm
            embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            new_embeddings.append(embeddings.cpu().numpy())
        
        cache.update(new_embeddings, docs_to_encode_ids)
        cache.save()
    
    doc_emb = cache.get_embeddings_array(doc_ids)
    # Convert numpy to tensor if needed
    if isinstance(doc_emb, np.ndarray):
        doc_emb = torch.from_numpy(doc_emb).float()
    doc_emb = F.normalize(doc_emb, p=2, dim=1)
    
    # ==================== ENCODE QUERIES (TEXT + VISION) ====================
    print("\nEncoding multimodal queries...")
    
    query_images_map = kwargs.get('query_images_map', {})
    
    query_emb_list = []
    
    for i in trange(0, len(queries), batch_size, desc="Queries"):
        batch_queries = queries[i:i+batch_size]
        batch_query_ids = query_ids[i:i+batch_size]
        batch_images = load_query_images(batch_query_ids, query_images_map, image_dir)
        
        for query_text, query_image in zip(batch_queries, batch_images):
            # Encode text with prefix
            text_with_prefix = f"search_query: {query_text}"
            encoded_input = text_tokenizer([text_with_prefix], padding=True, truncation=True,
                                          return_tensors='pt', max_length=8192).to('cuda')
            
            model_output = text_model(**encoded_input)
            text_emb = mean_pooling(model_output, encoded_input['attention_mask'])
            text_emb = F.layer_norm(text_emb, normalized_shape=(text_emb.shape[1],))
            text_emb = F.normalize(text_emb, p=2, dim=1)
            
            if query_image is not None:
                # Encode image with VISION model
                image_inputs = vision_processor(query_image, return_tensors="pt").to('cuda')
                img_output = vision_model(**image_inputs)
                
                # Extract image embedding (use first token like in official example)
                img_emb = img_output.last_hidden_state[:, 0]
                img_emb = F.normalize(img_emb, p=2, dim=1)
                
                # Combine text and image embeddings (average fusion)
                combined_emb = (text_emb + img_emb) / 2
                combined_emb = F.normalize(combined_emb, p=2, dim=1)
                query_emb_list.append(combined_emb.cpu())
            else:
                # Text-only query
                query_emb_list.append(text_emb.cpu())
    
    query_emb = torch.cat(query_emb_list, dim=0)
    scores = (query_emb @ doc_emb.T) * 100
    
    return get_scores(query_ids, doc_ids, scores.tolist(), excluded_ids)


# ==================== LLAVA MODELS ====================
@torch.no_grad()
def retrieval_llava(queries, query_ids, documents, doc_ids, task, model_id, instructions,
                   cache_dir, excluded_ids, long_context, **kwargs):
    """
    LLaVA (Large Language and Vision Assistant)
    Supports: llava-1.5-7b-hf, llava-v1.6-mistral-7b-hf
    Official: https://huggingface.co/llava-hf
    
    Note: LLaVA is primarily a generative model, but we extract embeddings from hidden states
    """
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    
    print("\n" + "="*80)
    print(f"LLAVA - {model_id}")
    print("="*80)
    
    model_name = kwargs.get('model_name', 'llava-hf/llava-1.5-7b-hf')
    if model_id == 'llava-1.6':
        model_name = 'llava-hf/llava-v1.6-mistral-7b-hf'
    
    print(f"Loading model: {model_name}")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    
    batch_size = kwargs.get('batch_size', 1)  # LLaVA is large, use small batch
    dataset_dir = Path(kwargs.get('dataset_dir', '.'))
    image_dir = dataset_dir / 'images' / task
    
    # ==================== CACHE DOCUMENT EMBEDDINGS ====================
    cache = SmartCache(cache_dir, f'llava_{model_id}', task, long_context, batch_size)
    cache.load()
    
    docs_to_encode, docs_to_encode_ids = cache.get_uncached_docs(doc_ids, documents)
    
    if len(docs_to_encode) > 0:
        print(f"Encoding {len(docs_to_encode)} new documents...")
        new_embeddings = []
        
        for i in trange(0, len(docs_to_encode), batch_size, desc="Documents"):
            batch_texts = docs_to_encode[i:i+batch_size]
            batch_texts = [f"Document: {text[:2000]}" for text in batch_texts]
            
            inputs = processor(text=batch_texts, return_tensors="pt", padding=True, 
                             truncation=True, max_length=2048).to(model.device)
            
            outputs = model.language_model(**inputs, output_hidden_states=True)
            # Extract embeddings from last hidden state, mean pool
            embeddings = outputs.hidden_states[-1].mean(dim=1)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            new_embeddings.append(embeddings.cpu().numpy())
        
        cache.update(new_embeddings, docs_to_encode_ids)
        cache.save()
    
    doc_emb = cache.get_embeddings_array(doc_ids)
    # Convert numpy to tensor if needed
    if isinstance(doc_emb, np.ndarray):
        doc_emb = torch.from_numpy(doc_emb).float()
    doc_emb = F.normalize(doc_emb, p=2, dim=1)
    
    # ==================== ENCODE QUERIES (TEXT + VISION) ====================
    print("\nEncoding multimodal queries...")
    
    query_images_map = kwargs.get('query_images_map', {})
    
    query_emb = []
    for i in trange(0, len(queries), batch_size, desc="Queries"):
        batch_queries = queries[i:i+batch_size]
        batch_query_ids = query_ids[i:i+batch_size]
        batch_images = load_query_images(batch_query_ids, query_images_map, image_dir)
        
        for query_text, query_image in zip(batch_queries, batch_images):
            if query_image is not None:
                # Multimodal query
                prompt = f"USER: <image>\n{query_text}\nASSISTANT:"
                inputs = processor(text=prompt, images=query_image, return_tensors="pt",
                                 padding=True, truncation=True).to(model.device)
            else:
                # Text-only query
                prompt = f"USER: {query_text}\nASSISTANT:"
                inputs = processor(text=prompt, return_tensors="pt", padding=True,
                                 truncation=True).to(model.device)
            
            outputs = model(**inputs, output_hidden_states=True)
            # Extract from vision tower if image present, otherwise language model
            if query_image is not None and hasattr(outputs, 'vision_tower_hidden_states'):
                embeddings = outputs.vision_tower_hidden_states[-1].mean(dim=1)
            else:
                embeddings = outputs.hidden_states[-1].mean(dim=1)
            
            embeddings = F.normalize(embeddings, p=2, dim=1)
            query_emb.append(embeddings.cpu())
    
    query_emb = torch.cat(query_emb, dim=0).float()  # Ensure float32
    doc_emb = doc_emb.float()  # Ensure float32
    scores = (query_emb @ doc_emb.T) * 100
    
    return get_scores(query_ids, doc_ids, scores.tolist(), excluded_ids)


# ==================== BGE-VL MODELS ====================
@torch.no_grad()
def retrieval_bge_vl(queries, query_ids, documents, doc_ids, task, model_id, instructions,
                    cache_dir, excluded_ids, long_context, **kwargs):
    """
    BGE-VL (Vision-Language) from BAAI
    Uses VERY aggressive character truncation to stay under 77 tokens
    """
    from transformers import AutoModel
    
    print("\n" + "="*80)
    print(f"BGE-VL - {model_id}")
    print("="*80)
    
    model_name = 'BAAI/BGE-VL-base' if model_id == 'bge-vl-base' else 'BAAI/BGE-VL-large'
    
    print(f"Loading model: {model_name}")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.set_processor(model_name)
    model.eval()
    model.to('cuda')
    
    batch_size = kwargs.get('batch_size', 16)
    dataset_dir = Path(kwargs.get('dataset_dir', '.'))
    image_dir = dataset_dir / 'images' / task
    
    def ultra_safe_truncate(text, max_chars=150):
        """
        Ultra-conservative character truncation
        150 chars ≈ 30-40 tokens (very safe for 77 token limit)
        """
        text = text.strip()
        if len(text) > max_chars:
            # Truncate and try to end at word boundary
            text = text[:max_chars]
            # Find last space to avoid cutting mid-word
            last_space = text.rfind(' ')
            if last_space > max_chars * 0.7:  # Only if we don't lose too much
                text = text[:last_space]
        return text.strip()
    
    # ==================== CACHE DOCUMENT EMBEDDINGS ====================
    cache_dir_path = Path(cache_dir) / 'doc_emb' / f'bge_vl_{model_id}' / task / f"long_{long_context}_{batch_size}"
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    
    embeddings_cache_path = cache_dir_path / 'embeddings.npy'
    mapping_cache_path = cache_dir_path / 'doc_id_mapping.json'
    
    cached_embeddings = {}
    doc_id_to_index = {}
    
    if embeddings_cache_path.exists() and mapping_cache_path.exists():
        print("Loading cached document embeddings...")
        cached_emb_array = np.load(embeddings_cache_path, allow_pickle=True)
        with open(mapping_cache_path, 'r') as f:
            doc_id_to_index = json.load(f)
        
        for doc_id, idx in doc_id_to_index.items():
            cached_embeddings[doc_id] = cached_emb_array[idx]
        print(f"✓ Loaded {len(cached_embeddings)} cached embeddings")
    
    docs_to_encode = [(doc_id, doc_text) for doc_id, doc_text in zip(doc_ids, documents)
                      if doc_id not in cached_embeddings]
    
    if docs_to_encode:
        print(f"Encoding {len(docs_to_encode)} new documents...")
        print("⚠️  BGE-VL: Using ultra-conservative 150 char truncation")
        new_embeddings = []
        
        failed_docs = []
        
        for doc_id, doc_text in tqdm(docs_to_encode, desc="Documents"):
            success = False
            
            # Try progressively smaller character limits
            for char_limit in [150, 120, 100, 80, 60, 40]:
                try:
                    truncated_text = ultra_safe_truncate(doc_text, max_chars=char_limit)
                    
                    # Skip if text becomes too short
                    if len(truncated_text.strip()) < 5:
                        truncated_text = doc_text[:40].strip()  # Minimum viable text
                    
                    embedding = model.encode(text=truncated_text)
                    
                    if len(embedding.shape) == 1:
                        embedding = embedding.reshape(1, -1)
                    
                    embedding = F.normalize(torch.tensor(embedding), p=2, dim=1)
                    new_embeddings.append(embedding.cpu().numpy())
                    success = True
                    break
                    
                except RuntimeError as e:
                    if "size of tensor" in str(e):
                        continue  # Try smaller limit
                    else:
                        raise e
            
            if not success:
                # Absolute fallback - just use first 20 chars
                failed_docs.append(doc_id)
                truncated_text = doc_text[:20].strip()
                if len(truncated_text) < 3:
                    truncated_text = "document"  # Last resort
                embedding = model.encode(text=truncated_text)
                if len(embedding.shape) == 1:
                    embedding = embedding.reshape(1, -1)
                embedding = F.normalize(torch.tensor(embedding), p=2, dim=1)
                new_embeddings.append(embedding.cpu().numpy())
        
        if failed_docs:
            print(f"\n⚠️  {len(failed_docs)} documents required extreme truncation (<20 chars)")
        
        if new_embeddings:
            new_embeddings = np.concatenate(new_embeddings, axis=0)
            next_index = len(cached_embeddings)
            
            for i, (doc_id, _) in enumerate(docs_to_encode):
                cached_embeddings[doc_id] = new_embeddings[i]
                doc_id_to_index[doc_id] = next_index + i
            
            all_embeddings = np.array([cached_embeddings[doc_id] for doc_id in
                                      sorted(doc_id_to_index.keys(), key=lambda x: doc_id_to_index[x])])
            np.save(embeddings_cache_path, all_embeddings)
            with open(mapping_cache_path, 'w') as f:
                json.dump(doc_id_to_index, f, indent=2)
            print(f"✓ Cached {len(cached_embeddings)} document embeddings")
    
    doc_emb = torch.tensor([cached_embeddings[doc_id] for doc_id in doc_ids])
    doc_emb = F.normalize(doc_emb, p=2, dim=1)
    
    # ==================== ENCODE QUERIES (TEXT + VISION) ====================
    # Use query_images_map passed via kwargs (PIL images from HF)
    query_images_map = kwargs.get('query_images_map', {})
    
    query_emb_list = []
    failed_queries = []
    
    for i in trange(0, len(queries), desc="Queries"):
        query_text = queries[i]
        query_id = query_ids[i]
        
        # Get images from query_images_map (now PIL.Image objects from HF)
        images = query_images_map.get(query_id, [])
        
        success = False
        
        # Try progressively smaller character limits
        for char_limit in [150, 120, 100, 80, 60, 40, 20]:
            try:
                truncated_query = ultra_safe_truncate(query_text, max_chars=char_limit)
                
                if len(truncated_query.strip()) < 3:
                    truncated_query = query_text[:20].strip()
                
                if images and hasattr(images[0], 'convert'):
                    # Save PIL image to temp file for bge-vl (requires file path)
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        images[0].convert('RGB').save(tmp, format='PNG')
                        tmp_path = tmp.name
                    query_emb = model.encode(images=tmp_path, text=truncated_query)
                    import os
                    os.unlink(tmp_path)  # Clean up
                else:
                    query_emb = model.encode(text=truncated_query)
                
                success = True
                break
                
            except RuntimeError as e:
                if "size of tensor" in str(e):
                    continue
                else:
                    raise e
        
        if not success:
            # Absolute fallback
            failed_queries.append(query_id)
            truncated_query = query_text[:15].strip() if query_text.strip() else "query"
            query_emb = model.encode(text=truncated_query)
        
        # Handle shape
        if len(query_emb.shape) == 1:
            query_emb = query_emb.reshape(1, -1)
        
        query_emb = F.normalize(torch.tensor(query_emb), p=2, dim=1)
        query_emb_list.append(query_emb.cpu())
    
    if failed_queries:
        print(f"\n⚠️  {len(failed_queries)} queries required extreme truncation")
    
    query_emb = torch.cat(query_emb_list, dim=0)
    scores = (query_emb @ doc_emb.T) * 100
    
    return get_scores(query_ids, doc_ids, scores.tolist(), excluded_ids)

def get_scores_qwenvl(query_ids, doc_ids, scores, excluded_ids):
    """Convert scores to required format and filter excluded IDs"""
    # scores is usually 2D: [num_queries][num_docs]
    # Some models may produce 3D: [num_queries][1][num_docs]
    if (
        len(scores) > 0
        and len(scores[0]) == 1
        and isinstance(scores[0][0], (list, tuple))
        and len(scores[0][0]) == len(doc_ids)
    ):
        # Flatten (num_queries, 1, num_docs) -> (num_queries, num_docs)
        scores = [row[0] for row in scores]

    assert len(scores) == len(query_ids), f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0]) == len(doc_ids), f"{len(scores[0])}, {len(doc_ids)}"
    
    emb_scores = {}
    for query_id, doc_scores in zip(query_ids, scores):
        cur_scores = {}
        for did, s in zip(doc_ids, doc_scores):
            cur_scores[str(did)] = s
        
        # Remove excluded IDs
        for did in set(excluded_ids.get(str(query_id), [])):
            if did != "N/A" and did in cur_scores:
                cur_scores.pop(did)
        
        # Keep top 1000
        cur_scores = sorted(cur_scores.items(), key=lambda x: x[1], reverse=True)[:1000]
        emb_scores[str(query_id)] = {pair[0]: pair[1] for pair in cur_scores}
    
    return emb_scores

# ==================== GME-QWEN2-VL MODELS ====================
@torch.no_grad()
def retrieval_gme_qwen2_vl(queries, query_ids, documents, doc_ids, task, model_id, instructions,
                           cache_dir, excluded_ids, long_context, **kwargs):
    """
    GME-Qwen2-VL using sentence_transformers (no transformers version constraint)
    """
    from sentence_transformers import SentenceTransformer
    
    print("\n" + "="*80)
    print(f"GME-QWEN2-VL - {model_id}")
    print("="*80)
    
    model_name = 'Alibaba-NLP/gme-Qwen2-VL-2B-Instruct' if model_id == 'gme-qwen2-vl-2b' else 'Alibaba-NLP/gme-Qwen2-VL-7B-Instruct'
    
    print(f"Loading model: {model_name}")
    
    # Add src/retrievers to path so custom_st can be imported by the model code
    import sys
    import os
    retrievers_path = os.path.join(os.getcwd(), 'src', 'retrievers')
    if retrievers_path not in sys.path:
        sys.path.append(retrievers_path)
        
    gme = SentenceTransformer(model_name, trust_remote_code=True)
    
    batch_size = kwargs.get('batch_size', 8)
    dataset_dir = Path(kwargs.get('dataset_dir', '.'))
    image_dir = dataset_dir / 'images' / task
    
    # ==================== CACHE DOCUMENT EMBEDDINGS ====================
    cache_dir_path = Path(cache_dir) / 'doc_emb' / f'gme_qwen2_vl_{model_id}' / task / f"long_{long_context}_{batch_size}"
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    
    embeddings_cache_path = cache_dir_path / 'embeddings.npy'
    mapping_cache_path = cache_dir_path / 'doc_id_mapping.json'
    
    cached_embeddings = {}
    doc_id_to_index = {}
    
    if embeddings_cache_path.exists() and mapping_cache_path.exists():
        print("Loading cached document embeddings...")
        cached_emb_array = np.load(embeddings_cache_path, allow_pickle=True)
        with open(mapping_cache_path, 'r') as f:
            doc_id_to_index = json.load(f)
        
        for doc_id, idx in doc_id_to_index.items():
            cached_embeddings[doc_id] = cached_emb_array[idx]
        print(f"✓ Loaded {len(cached_embeddings)} cached embeddings")
    
    docs_to_encode = [(doc_id, doc_text) for doc_id, doc_text in zip(doc_ids, documents)
                      if doc_id not in cached_embeddings]
    
    if docs_to_encode:
        print(f"Encoding {len(docs_to_encode)} new documents...")
        new_embeddings = []
        
        for i in trange(0, len(docs_to_encode), batch_size, desc="Documents"):
            batch = docs_to_encode[i:i+batch_size]
            batch_texts = [text[:4000] for _, text in batch]
            
            # sentence_transformers API: just pass strings
            embeddings = gme.encode(batch_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar = False)
            new_embeddings.append(embeddings.cpu().float().numpy())
        
        if new_embeddings:
            new_embeddings = np.concatenate(new_embeddings, axis=0)
            next_index = len(cached_embeddings)
            
            for i, (doc_id, _) in enumerate(docs_to_encode):
                cached_embeddings[doc_id] = new_embeddings[i]
                doc_id_to_index[doc_id] = next_index + i
            
            all_embeddings = np.array([cached_embeddings[doc_id] for doc_id in
                                      sorted(doc_id_to_index.keys(), key=lambda x: doc_id_to_index[x])])
            np.save(embeddings_cache_path, all_embeddings)
            with open(mapping_cache_path, 'w') as f:
                json.dump(doc_id_to_index, f, indent=2)
            print(f"✓ Cached {len(cached_embeddings)} document embeddings")
    
    # doc_emb = np.array([cached_embeddings[doc_id] for doc_id in doc_ids])
    # doc_emb = torch.tensor(doc_emb)
    # doc_emb = F.normalize(doc_emb, p=2, dim=1)
    # Build document embedding matrix from cache
    doc_emb = np.array([cached_embeddings[doc_id] for doc_id in doc_ids])

    # Ensure doc_emb is (num_docs, dim)
    if doc_emb.ndim == 1:
        # Single document: (dim,) -> (1, dim)
        doc_emb = doc_emb[None, :]
    elif doc_emb.ndim == 3 and doc_emb.shape[1] == 1:
        # Old cache style: (num_docs, 1, dim) -> (num_docs, dim)
        doc_emb = doc_emb.reshape(doc_emb.shape[0], -1)

    doc_emb = torch.tensor(doc_emb)
    doc_emb = F.normalize(doc_emb, p=2, dim=1)

    # ==================== ENCODE QUERIES (TEXT + VISION) ====================
    print("\nEncoding multimodal queries...")
    
    # Use query_images_map passed via kwargs (PIL images from HF)
    query_images_map = kwargs.get('query_images_map', {})
    
    query_emb_list = []

    # Process queries individually for multimodal
    for i in trange(len(queries), desc="Queries"):
        query_text = queries[i][:4000]
        query_id = query_ids[i]

        # Get images from query_images_map (now PIL.Image objects from HF)
        images = query_images_map.get(query_id, [])

        tmp_path = None  # Track temp file for cleanup
        if images and hasattr(images[0], 'convert'):
            # Save PIL image to temp file for gme-qwen2-vl (requires file path or PIL)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                images[0].convert('RGB').save(tmp, format='PNG')
                tmp_path = tmp.name
            
            # Multimodal: text + image as dict
            query_input = {"text": query_text, "image": tmp_path}
        else:
            # Text-only query
            query_input = query_text

        # ✅ Always pass a *list* into encode, even for a single query
        emb = gme.encode(
            [query_input],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        
        # Clean up temp file after encoding
        if tmp_path:
            import os
            os.unlink(tmp_path)
            
        # emb shape is typically (1, dim) -> flatten to (dim,)
        if emb.ndim == 2 and emb.size(0) == 1:
            emb_vec = emb[0]
        else:
            emb_vec = emb.view(-1)

        query_emb_list.append(emb_vec.cpu().float())

    # Now every entry is shape (dim,), so this works
    query_emb = torch.stack(query_emb_list, dim=0)

    scores = (query_emb @ doc_emb.T) * 100
    
    return get_scores_qwenvl(query_ids, doc_ids, scores.tolist(), excluded_ids)


# ==================== NVIDIA MM-EMBED ====================
@torch.no_grad()
def retrieval_nvidia_mm_embed(queries, query_ids, documents, doc_ids, task, model_id, instructions,
                              cache_dir, excluded_ids, long_context, **kwargs):
    """NVIDIA MM-Embed multimodal retrieval"""
    import sys
    from pathlib import Path as P
    
    print("\n" + "="*80)
    print("NVIDIA MM-EMBED")
    print("="*80)
    
    model_name = 'nvidia/MM-Embed'
    
    # FIX: Suppress registration error by temporarily catching it
    original_register = None
    try:
        from transformers.models.auto.auto_factory import _BaseAutoModelClass
        original_register = _BaseAutoModelClass.register
        
        def safe_register(cls, config_class, model_class, exist_ok=False):
            """Wrapper that ignores re-registration errors"""
            try:
                return original_register(config_class, model_class, exist_ok=True)
            except (ValueError, AttributeError):
                pass  # Ignore if already registered
        
        _BaseAutoModelClass.register = classmethod(safe_register)
    except:
        pass
    
    try:
        print(f"Loading model: {model_name}")
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to('cuda')
    finally:
        # Restore original register method
        if original_register is not None:
            try:
                _BaseAutoModelClass.register = original_register
            except:
                pass
    
    batch_size = kwargs.get('batch_size', 8)
    dataset_dir = P(kwargs.get('dataset_dir', '.'))
    image_dir = dataset_dir / 'images' / task
    
    # ==================== CACHE DOCUMENT EMBEDDINGS ====================
    cache_dir_path = P(cache_dir) / 'doc_emb' / 'nvidia_mm_embed' / task / f"long_{long_context}_{batch_size}"
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    
    embeddings_cache_path = cache_dir_path / 'embeddings.npy'
    mapping_cache_path = cache_dir_path / 'doc_id_mapping.json'
    
    cached_embeddings = {}
    doc_id_to_index = {}
    
    if embeddings_cache_path.exists() and mapping_cache_path.exists():
        print("Loading cached document embeddings...")
        cached_emb_array = np.load(embeddings_cache_path, allow_pickle=True)
        with open(mapping_cache_path, 'r') as f:
            doc_id_to_index = json.load(f)
        
        for doc_id, idx in doc_id_to_index.items():
            cached_embeddings[doc_id] = cached_emb_array[idx]
        print(f"✓ Loaded {len(cached_embeddings)} cached embeddings")
    
    docs_to_encode = [(doc_id, doc_text) for doc_id, doc_text in zip(doc_ids, documents)
                      if doc_id not in cached_embeddings]
    
    if docs_to_encode:
        print(f"Encoding {len(docs_to_encode)} new documents...")
        new_embeddings = []
        
        for i in trange(0, len(docs_to_encode), batch_size, desc="Documents"):
            batch = docs_to_encode[i:i+batch_size]
            batch_texts = [text[:8000] for _, text in batch]
            
            embeddings = model.encode_text(batch_texts)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            new_embeddings.append(embeddings.cpu().float().numpy())
        
        if new_embeddings:
            new_embeddings = np.concatenate(new_embeddings, axis=0)
            next_index = len(cached_embeddings)
            
            for i, (doc_id, _) in enumerate(docs_to_encode):
                cached_embeddings[doc_id] = new_embeddings[i]
                doc_id_to_index[doc_id] = next_index + i
            
            all_embeddings = np.array([cached_embeddings[doc_id] for doc_id in
                                      sorted(doc_id_to_index.keys(), key=lambda x: doc_id_to_index[x])])
            np.save(embeddings_cache_path, all_embeddings)
            with open(mapping_cache_path, 'w') as f:
                json.dump(doc_id_to_index, f, indent=2)
            print(f"✓ Cached {len(cached_embeddings)} document embeddings")
    
    doc_emb = np.array([cached_embeddings[doc_id] for doc_id in doc_ids])
    doc_emb = torch.tensor(doc_emb)
    doc_emb = F.normalize(doc_emb, p=2, dim=1)
    
    # ==================== ENCODE QUERIES ====================
    print("\nEncoding multimodal queries...")
    query_file = dataset_dir / 'filtered_queries' / task / f"{task}_queries_kept.jsonl"
    query_data_map = {}
    with open(query_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            query_data_map[data['id']] = data
    
    from PIL import Image
    query_emb_list = []
    
    for i in trange(len(queries), desc="Queries"):
        query_text = queries[i][:8000]
        query_id = query_ids[i]
        
        # Load image if available
        query_data = query_data_map.get(query_id, {})
        image_paths = query_data.get('image_paths', [])
        
        if image_paths:
            img_path = "./" / image_dir / image_paths[0].split('/')[-1]
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert('RGB')
                    # Encode text and image separately, then fuse
                    text_emb = model.encode_text([query_text])
                    img_emb = model.encode_image([img])
                    # Weighted fusion (0.6 text, 0.4 image)
                    query_emb = 0.6 * text_emb + 0.4 * img_emb
                except:
                    query_emb = model.encode_text([query_text])
            else:
                query_emb = model.encode_text([query_text])
        else:
            query_emb = model.encode_text([query_text])
        
        query_emb = F.normalize(query_emb, p=2, dim=1)
        query_emb_list.append(query_emb.cpu().float())
    
    query_emb = torch.cat(query_emb_list, dim=0)
    scores = (query_emb @ doc_emb.T) * 100
    
    return get_scores(query_ids, doc_ids, scores.tolist(), excluded_ids)

# ==================== SEED-1.6 MULTIMODAL ====================
@torch.no_grad()
def retrieval_seed_multimodal(queries, query_ids, documents, doc_ids, task, model_id, instructions,
                              cache_dir, excluded_ids, long_context, **kwargs):
    """
    Seed-1.6 Multimodal Model
    Note: This is based on available Seed multimodal models on HuggingFace
    Adjust model_name if specific version is different
    """
    from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
    
    print("\n" + "="*80)
    print("SEED-1.6 MULTIMODAL")
    print("="*80)
    
    # Note: Using SEED-X-I (instruction-tuned) or base model
    # The exact model path may vary - check HuggingFace for latest
    model_name = kwargs.get('model_name', 'AILab-CVC/seed-x-i-8b')  # or 'AILab-CVC/SEED-X'
    
    print(f"Loading model: {model_name}")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    batch_size = kwargs.get('batch_size', 4)
    dataset_dir = Path(kwargs.get('dataset_dir', '.'))
    image_dir = dataset_dir / 'images' / task
    
    # ==================== CACHE DOCUMENT EMBEDDINGS ====================
    cache_dir_path = Path(cache_dir) / 'doc_emb' / 'seed_multimodal' / task / f"long_{long_context}_{batch_size}"
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    
    embeddings_cache_path = cache_dir_path / 'embeddings.npy'
    mapping_cache_path = cache_dir_path / 'doc_id_mapping.json'
    
    cached_embeddings = {}
    doc_id_to_index = {}
    
    if embeddings_cache_path.exists() and mapping_cache_path.exists():
        print("Loading cached document embeddings...")
        cached_emb_array = np.load(embeddings_cache_path, allow_pickle=True)
        with open(mapping_cache_path, 'r') as f:
            doc_id_to_index = json.load(f)
        
        for doc_id, idx in doc_id_to_index.items():
            cached_embeddings[doc_id] = cached_emb_array[idx]
        print(f"✓ Loaded {len(cached_embeddings)} cached embeddings")
    
    docs_to_encode = [(doc_id, doc_text) for doc_id, doc_text in zip(doc_ids, documents)
                      if doc_id not in cached_embeddings]
    
    if docs_to_encode:
        print(f"Encoding {len(docs_to_encode)} new documents...")
        new_embeddings = []
        
        for i in trange(0, len(docs_to_encode), batch_size, desc="Documents"):
            batch = docs_to_encode[i:i+batch_size]
            batch_texts = [text[:2000] for _, text in batch]
            
            inputs = tokenizer(batch_texts, padding=True, truncation=True,
                             return_tensors="pt", max_length=2048).to('cuda')
            
            outputs = model(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1].mean(dim=1)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            new_embeddings.append(embeddings.cpu().numpy())
        
        if new_embeddings:
            new_embeddings = np.concatenate(new_embeddings, axis=0)
            next_index = len(cached_embeddings)
            
            for i, (doc_id, _) in enumerate(docs_to_encode):
                cached_embeddings[doc_id] = new_embeddings[i]
                doc_id_to_index[doc_id] = next_index + i
            
            all_embeddings = np.array([cached_embeddings[doc_id] for doc_id in
                                      sorted(doc_id_to_index.keys(), key=lambda x: doc_id_to_index[x])])
            np.save(embeddings_cache_path, all_embeddings)
            with open(mapping_cache_path, 'w') as f:
                json.dump(doc_id_to_index, f, indent=2)
            print(f"✓ Cached {len(cached_embeddings)} document embeddings")
    
    doc_emb = torch.tensor([cached_embeddings[doc_id] for doc_id in doc_ids])
    doc_emb = F.normalize(doc_emb, p=2, dim=1)
    
    # ==================== ENCODE QUERIES (TEXT + VISION) ====================
    print("\nEncoding multimodal queries...")
    query_file = dataset_dir / 'filtered_queries' / task / f"{task}_queries_kept.jsonl"
    query_data_map = {}
    with open(query_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            query_data_map[data['id']] = data
    
    query_emb = []
    for i in trange(0, len(queries), batch_size, desc="Queries"):
        batch_queries = queries[i:i+batch_size]
        batch_query_ids = query_ids[i:i+batch_size]
        batch_images = load_query_images(batch_query_ids, query_data_map, image_dir)
        
        for query_text, query_image in zip(batch_queries, batch_images):
            if query_image is not None:
                # Process multimodal input
                text_inputs = tokenizer([query_text], return_tensors="pt",
                                      padding=True, truncation=True).to('cuda')
                image_inputs = processor(query_image, return_tensors="pt").to('cuda')
                
                # Get embeddings
                text_outputs = model(**text_inputs, output_hidden_states=True)
                image_outputs = model(pixel_values=image_inputs['pixel_values'], 
                                    output_hidden_states=True)
                
                text_emb = text_outputs.hidden_states[-1].mean(dim=1)
                image_emb = image_outputs.hidden_states[-1].mean(dim=1)
                
                # Fusion
                combined = (text_emb + image_emb) / 2
                combined = F.normalize(combined, p=2, dim=1)
                query_emb.append(combined.cpu())
            else:
                # Text-only
                text_inputs = tokenizer([query_text], return_tensors="pt",
                                      padding=True, truncation=True).to('cuda')
                outputs = model(**text_inputs, output_hidden_states=True)
                embeddings = outputs.hidden_states[-1].mean(dim=1)
                embeddings = F.normalize(embeddings, p=2, dim=1)
                query_emb.append(embeddings.cpu())
    
    query_emb = torch.cat(query_emb, dim=0)
    scores = (query_emb @ doc_emb.T) * 100
    
    return get_scores(query_ids, doc_ids, scores.tolist(), excluded_ids)


# ==================== RETRIEVAL FUNCTIONS DICTIONARY ====================
MULTIMODAL_RETRIEVAL_FUNCS = {
    'nomic-vision': retrieval_nomic_vision,
    # Note: llava-1.5/1.6 removed - they are VLM (generative) models, not embedding models
    'bge-vl-large': retrieval_bge_vl,
    'bge-vl-base': retrieval_bge_vl,
    'gme-qwen2-vl-2b': retrieval_gme_qwen2_vl,
    'gme-qwen2-vl-7b': retrieval_gme_qwen2_vl,
    # 'nvidia-mm-embed': retrieval_nvidia_mm_embed,  # Incompatible: custom config not recognized by AutoModel
    # 'seed-multimodal': retrieval_seed_multimodal,  # Incompatible: repository not found/accessible
}




# ==================== ADD TO RETRIEVAL_FUNCS DICTIONARY ====================
MULTIMODAL_RETRIEVAL_FUNCS['clip'] = retrieval_clip
MULTIMODAL_RETRIEVAL_FUNCS['siglip'] = retrieval_siglip
MULTIMODAL_RETRIEVAL_FUNCS['jina-clip'] = retrieval_jina_clip