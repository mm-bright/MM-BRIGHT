# ==================== IT→I (Image Retrieval) HELPERS ====================
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
from src.utils import cut_text, add_instruct_concatenate, last_token_pool, safe_image_path, ensure_blank

# Add this to your retrievers.py file

from PIL import Image
import requests
from io import BytesIO
import clip  # pip install git+https://github.com/openai/CLIP.git

def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)

def _l2norm_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def _get_query_image_paths(batch_query_ids, query_images_map, image_dir: Path):
    # returns list[Path|None] aligned with batch_query_ids
    out = []
    for qid in batch_query_ids:
        image_paths = query_images_map.get(qid, []) or []
        if image_paths:
            img_name = Path(image_paths[0]).name
            p = image_dir / img_name
            out.append(p if p.exists() else None)
        else:
            out.append(None)
    return out

def _scores_topk_from_embeddings(query_ids, query_emb_np, corpus_ids, corpus_emb_np,
                                 excluded_ids=None, topk=1000, chunk_size=50000, scale=100.0):
    """
    Builds scores dict WITHOUT making full (Q x N) matrix (safe for big image corpora)
    Returns: {qid: {corpus_id: score}}
    """
    excluded_ids = excluded_ids or {}

    # ensure shapes
    if query_emb_np.ndim == 1:
        query_emb_np = query_emb_np[None, :]
    if corpus_emb_np.ndim == 1:
        corpus_emb_np = corpus_emb_np[None, :]

    Q, D = query_emb_np.shape
    N, D2 = corpus_emb_np.shape
    assert D == D2, f"Dim mismatch: query {D} vs corpus {D2}"

    scores_out = {}

    corpus_emb_np = corpus_emb_np.astype(np.float32)

    for qi, qid in enumerate(query_ids):
        q = query_emb_np[qi].astype(np.float32)  # (D,)

        best_scores = np.full((topk,), -1e9, dtype=np.float32)
        best_ids = np.full((topk,), -1, dtype=np.int64)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = corpus_emb_np[start:end]  # (c, D)
            chunk_scores = chunk @ q          # (c,)

            # merge with current best
            merged_scores = np.concatenate([best_scores, chunk_scores], axis=0)
            merged_ids = np.concatenate([best_ids, np.arange(start, end, dtype=np.int64)], axis=0)

            keep = min(topk, merged_scores.shape[0])
            idx = np.argpartition(merged_scores, -keep)[-keep:]
            best_scores = merged_scores[idx]
            best_ids = merged_ids[idx]

        # sort descending
        order = np.argsort(-best_scores)
        best_scores = best_scores[order]
        best_ids = best_ids[order]

        # build dict + apply excluded_ids
        cur = {}
        ex = set(excluded_ids.get(str(qid), [])) if excluded_ids else set()

        for s, idx in zip(best_scores, best_ids):
            if idx < 0:
                continue
            item_id = str(corpus_ids[int(idx)])
            if item_id in ex:
                continue
            cur[item_id] = float(s * scale)

        # keep exactly topk
        if len(cur) > topk:
            cur = dict(list(cur.items())[:topk])

        scores_out[str(qid)] = cur

    return scores_out


# ==================== IT→I: CLIP ====================
@torch.no_grad()
def retrieval_clip_it2i(queries, query_ids, documents, doc_ids, task, model_id, instructions,
                        cache_dir, excluded_ids, long_context, **kwargs):
    import json
    import numpy as np
    import torch
    import torch.nn.functional as F
    from pathlib import Path
    from tqdm import trange
    from PIL import Image, ImageFile
    from transformers import CLIPProcessor, CLIPModel

    ImageFile.LOAD_TRUNCATED_IMAGES = True  # helps with broken jpg/webp

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def safe_rgb_np(path_or_img, fallback_hw=(224, 224)):
        """
        Always return np.uint8 image with shape (H,W,3).
        Never grayscale, never RGBA, never weird dtype.
        """
        H, W = fallback_hw
        try:
            if isinstance(path_or_img, Image.Image):
                im = path_or_img
            else:
                im = Image.open(path_or_img)

            im = im.convert("RGB")
            im.load()  # force decode now
            arr = np.asarray(im)

            # Make absolutely sure it's HxWx3 uint8
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]

            if arr.dtype != np.uint8:
                arr = arr.astype(np.float32)
                mx = float(arr.max()) if arr.size else 0.0
                if mx > 0:
                    arr = arr / mx * 255.0
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            # If still broken, fallback
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"Bad array shape: {arr.shape}")

            return arr
        except Exception:
            return np.full((H, W, 3), 255, dtype=np.uint8)

    model_name = kwargs.get("model_name", "openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    batch_size = int(kwargs.get("batch_size", 8))
    topk = int(kwargs.get("topk", 1000))
    chunk_size = int(kwargs.get("chunk_size", 50000))

    dataset_dir = Path(kwargs.get("dataset_dir", "."))
    image_dir = dataset_dir / "images" / task

    # -------- cache corpus IMAGE embeddings --------
    cache = SmartCache(cache_dir, f"{model_id}_clip_it2i", task, long_context, batch_size, sub_dir='img_emb')
    cache.load()
    
    # documents here are image paths
    to_encode, to_encode_ids = cache.get_uncached_docs(doc_ids, documents)

    if len(to_encode) > 0:
        new_embs = []
        
        # Batching logic from original code, adapted for to_encode list
        for i in trange(0, len(to_encode), batch_size, desc="Encoding corpus images (CLIP)"):
            batch_paths = to_encode[i:i + batch_size]
            # to_encode contains the docs (paths) directly
            img_np = [safe_rgb_np(path, fallback_hw=(224, 224)) for path in batch_paths]

            # Batch preprocess sometimes fails -> fallback per-image
            try:
                inputs = processor(images=img_np, return_tensors="pt")
            except Exception as e:
                fixed = []
                for path, arr_img in zip(batch_paths, img_np):
                    try:
                        _ = processor(images=[arr_img], return_tensors="pt")
                        fixed.append(arr_img)
                    except Exception as e2:
                        print(f"⚠️ CLIP bad image replaced: {path} | {e2}")
                        fixed.append(np.full((224, 224, 3), 255, dtype=np.uint8))
                inputs = processor(images=fixed, return_tensors="pt")

            inputs = {k: v.to(device) for k, v in inputs.items()}
            feats = model.get_image_features(**inputs)
            feats = F.normalize(feats, p=2, dim=1)

            new_embs.append(feats.detach().cpu().numpy())

        cache.update(new_embs, to_encode_ids)
        cache.save()

    corpus_emb = cache.get_embeddings_array(doc_ids)
    corpus_emb = _l2norm_np(corpus_emb)

    # -------- encode QUERY (text + image) --------
    query_images_map = kwargs.get('query_images_map', {})

    q_embs = []
    for i in trange(0, len(queries), batch_size, desc="Encoding queries (CLIP IT)"):
        bq = queries[i:i + batch_size]
        bqid = query_ids[i:i + batch_size]
        qimg_paths = _get_query_image_paths(bqid, query_images_map, image_dir)

        texts = [(t or "") for t in bq]
        img_np = []
        for qp in qimg_paths:
            if qp is None:
                img_np.append(np.full((224, 224, 3), 255, dtype=np.uint8))
            else:
                img_np.append(safe_rgb_np(str(qp), fallback_hw=(224, 224)))

        # If a weird query image still breaks, replace it
        try:
            inputs = processor(text=texts, images=img_np, return_tensors="pt", padding=True, truncation=True)
        except Exception as e:
            fixed = []
            for qp, arr_img in zip(qimg_paths, img_np):
                try:
                    _ = processor(text=["x"], images=[arr_img], return_tensors="pt", padding=True, truncation=True)
                    fixed.append(arr_img)
                except Exception as e2:
                    print(f"⚠️ CLIP bad query image replaced: {qp} | {e2}")
                    fixed.append(np.full((224, 224, 3), 255, dtype=np.uint8))
            inputs = processor(text=texts, images=fixed, return_tensors="pt", padding=True, truncation=True)

        inputs = {k: v.to(device) for k, v in inputs.items()}
        tfeat = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        ifeat = model.get_image_features(pixel_values=inputs["pixel_values"])
        fused = (tfeat + ifeat) / 2.0
        fused = F.normalize(fused, p=2, dim=1)
        q_embs.append(fused.detach().cpu().numpy())

    query_emb = _l2norm_np(np.concatenate(q_embs, axis=0))

    return _scores_topk_from_embeddings(
        query_ids=query_ids,
        query_emb_np=query_emb,
        corpus_ids=doc_ids,
        corpus_emb_np=corpus_emb,
        excluded_ids=excluded_ids,
        topk=topk,
        chunk_size=chunk_size,
        scale=100.0,
    )



# ==================== IT→I: SigLIP ====================
@torch.no_grad()
def retrieval_siglip_it2i(queries, query_ids, documents, doc_ids, task, model_id, instructions,
                          cache_dir, excluded_ids, long_context, **kwargs):
    import json
    import numpy as np
    import torch
    import torch.nn.functional as F
    from pathlib import Path
    from tqdm import trange
    from PIL import Image, ImageFile
    from transformers import AutoProcessor, AutoModel

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def safe_rgb_np(path_or_img, fallback_hw=(384, 384)):
        """Return np.uint8 (H,W,3). If anything weird -> blank white RGB."""
        H, W = fallback_hw
        try:
            if isinstance(path_or_img, Image.Image):
                im = path_or_img
            else:
                im = Image.open(path_or_img)

            im = im.convert("RGB")
            im.load()
            arr = np.asarray(im)

            # ensure HxWx3
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]

            # if channels are still not 3 => invalid (e.g., (1,1,545))
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"Bad image array shape: {arr.shape}")

            if arr.dtype != np.uint8:
                arr = arr.astype(np.float32)
                mx = float(arr.max()) if arr.size else 0.0
                if mx > 0:
                    arr = arr / mx * 255.0
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            return arr
        except Exception:
            return np.full((H, W, 3), 255, dtype=np.uint8)

    model_name = kwargs.get("model_name", "google/siglip-so400m-patch14-384")
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)

    batch_size = int(kwargs.get("batch_size", 8))
    topk = int(kwargs.get("topk", 1000))
    chunk_size = int(kwargs.get("chunk_size", 50000))

    dataset_dir = Path(kwargs.get("dataset_dir", "."))
    image_dir = dataset_dir / "images" / task

    # -------- cache corpus IMAGE embeddings --------
    cache = SmartCache(cache_dir, f"{model_id}_siglip_it2i", task, long_context, batch_size, sub_dir='img_emb')
    cache.load()
    
    to_encode, to_encode_ids = cache.get_uncached_docs(doc_ids, documents)

    if len(to_encode) > 0:
        new_embs = []
        for i in trange(0, len(to_encode), batch_size, desc="Encoding corpus images (SigLIP)"):
            batch_paths = to_encode[i:i + batch_size]
            img_np = [safe_rgb_np(path, fallback_hw=(384, 384)) for path in batch_paths]

            # If batch preprocess fails, validate per-image and replace bad ones
            try:
                inputs = processor(images=img_np, return_tensors="pt")
            except Exception as e:
                fixed = []
                for path, arr_img in zip(batch_paths, img_np):
                    try:
                        _ = processor(images=[arr_img], return_tensors="pt")
                        fixed.append(arr_img)
                    except Exception as e2:
                        print(f"⚠️ SigLIP bad image replaced: {path} | {e2}")
                        fixed.append(np.full((384, 384, 3), 255, dtype=np.uint8))
                inputs = processor(images=fixed, return_tensors="pt")

            inputs = {k: v.to(device) for k, v in inputs.items()}
            if hasattr(model, "get_image_features"):
                feats = model.get_image_features(**inputs)
            else:
                feats = model(**inputs).image_embeds
            feats = F.normalize(feats, p=2, dim=1)

            new_embs.append(feats.detach().cpu().numpy())

        cache.update(new_embs, to_encode_ids)
        cache.save()

    corpus_emb = cache.get_embeddings_array(doc_ids)
    corpus_emb = _l2norm_np(corpus_emb)

    # -------- encode QUERY (text + image) --------
    query_images_map = kwargs.get('query_images_map', {})

    q_embs = []
    for i in trange(0, len(queries), batch_size, desc="Encoding queries (SigLIP IT)"):
        bq = queries[i:i + batch_size]
        bqid = query_ids[i:i + batch_size]
        qimg_paths = _get_query_image_paths(bqid, query_images_map, image_dir)

        texts = [(t or "") for t in bq]
        img_np = []
        for qp in qimg_paths:
            if qp is None:
                img_np.append(np.full((384, 384, 3), 255, dtype=np.uint8))
            else:
                img_np.append(safe_rgb_np(str(qp), fallback_hw=(384, 384)))

        try:
            inputs = processor(text=texts, images=img_np, return_tensors="pt", padding=True, truncation=True)
        except Exception as e:
            fixed = []
            for qp, arr_img in zip(qimg_paths, img_np):
                try:
                    _ = processor(text=["x"], images=[arr_img], return_tensors="pt", padding=True, truncation=True)
                    fixed.append(arr_img)
                except Exception as e2:
                    print(f"⚠️ SigLIP bad query image replaced: {qp} | {e2}")
                    fixed.append(np.full((384, 384, 3), 255, dtype=np.uint8))
            inputs = processor(text=texts, images=fixed, return_tensors="pt", padding=True, truncation=True)

        inputs = {k: v.to(device) for k, v in inputs.items()}

        if hasattr(model, "get_text_features"):
            tfeat = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask", None))
        else:
            tfeat = model(**{k: inputs[k] for k in ["input_ids", "attention_mask"] if k in inputs}).text_embeds

        if hasattr(model, "get_image_features"):
            ifeat = model.get_image_features(pixel_values=inputs["pixel_values"])
        else:
            ifeat = model(**{k: inputs[k] for k in ["pixel_values"] if k in inputs}).image_embeds

        fused = (tfeat + ifeat) / 2.0
        fused = F.normalize(fused, p=2, dim=1)
        q_embs.append(fused.detach().cpu().numpy())

    query_emb = _l2norm_np(np.concatenate(q_embs, axis=0))

    return _scores_topk_from_embeddings(
        query_ids=query_ids,
        query_emb_np=query_emb,
        corpus_ids=doc_ids,
        corpus_emb_np=corpus_emb,
        excluded_ids=excluded_ids,
        topk=topk,
        chunk_size=chunk_size,
        scale=100.0
    )

# ==================== IT→I: Jina-CLIP ====================
@torch.no_grad()
def retrieval_jina_clip_it2i(queries, query_ids, documents, doc_ids, task, model_id, instructions,
                             cache_dir, excluded_ids, long_context, **kwargs):
    from transformers import AutoModel

    model_name = kwargs.get('model_name', 'jinaai/jina-clip-v1')
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to('cuda')
    model.eval()

    batch_size = kwargs.get('batch_size', 8)
    topk = kwargs.get('topk', 1000)
    chunk_size = kwargs.get('chunk_size', 50000)

    dataset_dir = Path(kwargs.get('dataset_dir', '.'))
    image_dir = dataset_dir / 'images' / task

    cache = SmartCache(cache_dir, f'{model_id}_jina_it2i', task, long_context, batch_size, sub_dir='img_emb')
    cache.load()
    
    to_encode, to_encode_ids = cache.get_uncached_docs(doc_ids, documents)

    if len(to_encode) > 0:
        new_embs = []
        for i in trange(0, len(to_encode), batch_size, desc="Encoding corpus images (Jina-CLIP)"):
            batch_paths = to_encode[i:i+batch_size]
            imgs = []
            for path in batch_paths:
                try:
                    imgs.append(Image.open(path).convert("RGB"))
                except:
                    imgs.append(Image.new("RGB", (224, 224), color="white"))

            emb = model.encode_image(imgs)  # may return torch or np
            emb = _to_numpy(emb)
            emb = _l2norm_np(emb)
            new_embs.append(emb)

        cache.update(new_embs, to_encode_ids)
        cache.save()

    corpus_emb = _l2norm_np(cache.get_embeddings_array(doc_ids))

    query_images_map = kwargs.get('query_images_map', {})

    q_embs = []
    for i in trange(0, len(queries), batch_size, desc="Encoding queries (Jina IT)"):
        bq = queries[i:i+batch_size]
        bqid = query_ids[i:i+batch_size]
        qimg_paths = _get_query_image_paths(bqid, query_images_map, image_dir)

        # text embeds
        t = model.encode_text(bq)
        t = _l2norm_np(_to_numpy(t))

        # image embeds (one per query; use zero if missing)
        im_list = []
        for qp in qimg_paths:
            if qp is not None:
                try:
                    img = Image.open(qp).convert("RGB")
                except:
                    img = Image.new("RGB", (224, 224), color="white")
                e = model.encode_image([img])
                e = _to_numpy(e)
                im_list.append(e[0])
            else:
                im_list.append(np.zeros((t.shape[1],), dtype=np.float32))

        im = _l2norm_np(np.stack(im_list, axis=0))
        fused = _l2norm_np((t + im) / 2.0)
        q_embs.append(fused)

    query_emb = np.concatenate(q_embs, axis=0)

    return _scores_topk_from_embeddings(query_ids, query_emb, doc_ids, corpus_emb,
                                        excluded_ids=excluded_ids, topk=topk, chunk_size=chunk_size, scale=100.0)


# ==================== IT→I: Nomic (vision corpus, fused query) ====================
@torch.no_grad()
def retrieval_nomic_it2i(queries, query_ids, documents, doc_ids, task, model_id, instructions,
                         cache_dir, excluded_ids, long_context, **kwargs):
    from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

    text_model_name = 'nomic-ai/nomic-embed-text-v1.5'
    vision_model_name = 'nomic-ai/nomic-embed-vision-v1.5'

    text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_model = AutoModel.from_pretrained(text_model_name, trust_remote_code=True).to('cuda').eval()

    vision_processor = AutoImageProcessor.from_pretrained(vision_model_name)
    vision_model = AutoModel.from_pretrained(vision_model_name, trust_remote_code=True).to('cuda').eval()

    batch_size = kwargs.get('batch_size', 8)
    topk = kwargs.get('topk', 1000)
    chunk_size = kwargs.get('chunk_size', 50000)

    dataset_dir = Path(kwargs.get('dataset_dir', '.'))
    image_dir = dataset_dir / 'images' / task

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # ---- cache corpus image embeddings (vision model) ----
    cache = SmartCache(cache_dir, f'{model_id}_nomic_it2i', task, long_context, batch_size, sub_dir='img_emb')
    cache.load()
    
    to_encode, to_encode_ids = cache.get_uncached_docs(doc_ids, documents)

    if len(to_encode) > 0:
        new_embs = []
        for i in trange(0, len(to_encode), batch_size, desc="Encoding corpus images (Nomic vision)"):
            batch_paths = to_encode[i:i+batch_size]
            imgs = []
            for path in batch_paths:
                try:
                    imgs.append(Image.open(path).convert("RGB"))
                except:
                    imgs.append(Image.new("RGB", (224, 224), color="white"))

            inputs = vision_processor(images=imgs, return_tensors="pt").to('cuda')
            out = vision_model(**inputs)
            emb = out.last_hidden_state[:, 0]
            emb = F.normalize(emb, p=2, dim=1)
            new_embs.append(emb.cpu().numpy())

        cache.update(new_embs, to_encode_ids)
        cache.save()

    corpus_emb = _l2norm_np(cache.get_embeddings_array(doc_ids))

    # ---- encode fused query (text + query image) ----
    query_images_map = kwargs.get('query_images_map', {})

    q_embs = []
    for i in trange(0, len(queries), batch_size, desc="Encoding queries (Nomic IT)"):
        bq = queries[i:i+batch_size]
        bqid = query_ids[i:i+batch_size]
        qimg_paths = _get_query_image_paths(bqid, query_images_map, image_dir)

        # text: add prefix like your IT→T code
        bq_pref = [f"search_query: {t}" for t in bq]
        enc = text_tokenizer(bq_pref, padding=True, truncation=True, return_tensors='pt', max_length=8192).to('cuda')
        out = text_model(**enc)
        t = mean_pooling(out, enc['attention_mask'])
        t = F.layer_norm(t, normalized_shape=(t.shape[1],))
        t = F.normalize(t, p=2, dim=1)

        # image emb
        imgs = []
        for qp in qimg_paths:
            if qp is not None:
                try:
                    imgs.append(Image.open(qp).convert("RGB"))
                except:
                    imgs.append(Image.new("RGB", (224, 224), color="white"))
            else:
                imgs.append(Image.new("RGB", (224, 224), color="white"))

        vin = vision_processor(images=imgs, return_tensors="pt").to('cuda')
        vout = vision_model(**vin)
        v = vout.last_hidden_state[:, 0]
        v = F.normalize(v, p=2, dim=1)

        fused = (t + v) / 2.0
        fused = F.normalize(fused, p=2, dim=1)
        q_embs.append(fused.cpu().numpy())

    query_emb = _l2norm_np(np.concatenate(q_embs, axis=0))

    return _scores_topk_from_embeddings(query_ids, query_emb, doc_ids, corpus_emb,
                                        excluded_ids=excluded_ids, topk=topk, chunk_size=chunk_size, scale=100.0)


# ==================== IT→I: BGE-VL Large ====================
# ==================== IT→I: BGE-VL Large (FIXED: no encode_multimodal) ====================
@torch.no_grad()
def retrieval_bge_vl_it2i(queries, query_ids, documents, doc_ids, task, model_id, instructions,
                          cache_dir, excluded_ids, long_context, **kwargs):
    import os
    import json
    import numpy as np
    import torch
    import torch.nn.functional as F
    from pathlib import Path
    from tqdm import trange
    from PIL import Image, UnidentifiedImageError
    from transformers import AutoModel

    model_name = kwargs.get("model_name", "BAAI/BGE-VL-large")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
    model.set_processor(model_name)
    model.eval()

    device = next(model.parameters()).device
    batch_size = int(kwargs.get("batch_size", 64))
    topk = int(kwargs.get("topk", 1000))
    chunk_size = int(kwargs.get("chunk_size", 50000))

    dataset_dir = Path(kwargs.get("dataset_dir", "."))
    image_dir = dataset_dir / "images" / task

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    def ultra_safe_truncate(text, max_chars=150):
        text = (text or "").strip()
        if len(text) > max_chars:
            text = text[:max_chars]
            last_space = text.rfind(" ")
            if last_space > max_chars * 0.7:
                text = text[:last_space]
        return text.strip()

    def ensure_blank(path_dir: Path, size=(224, 224)) -> str:
        path_dir.mkdir(parents=True, exist_ok=True)
        p = path_dir / "blank.png"
        if not p.exists():
            img = Image.new("RGB", size, color="white")
            img.save(p)
        return str(p)

    def safe_image_path(p: str, blank_path: str) -> str:
        if p is None:
            return blank_path
        p = str(p)
        if p.lower().endswith(".svg"):
            return blank_path
        try:
            with Image.open(p) as im:
                im.load()
            return p
        except (UnidentifiedImageError, OSError, ValueError):
            return blank_path

    def encode_text_one(t: str):
        # some text inputs can still break; retry with shorter
        t = ultra_safe_truncate(t, 150)
        if not t:
            t = "query"
        try:
            e = model.encode(text=t)
        except Exception:
            t2 = ultra_safe_truncate(t, 60) or "query"
            e = model.encode(text=t2)
        e = _to_numpy(e)
        if e.ndim == 1:
            e = e[None, :]
        return _l2norm_np(e)

    def encode_image_one(p: str):
        try:
            e = model.encode(images=p)
        except Exception:
            # last resort: try blank
            e = model.encode(images=blank_img)
        e = _to_numpy(e)
        if e.ndim == 1:
            e = e[None, :]
        return _l2norm_np(e)

    def encode_images_batch(paths):
        # fast path: batch; fallback: per-image if something is corrupt
        try:
            e = model.encode(images=paths)
            e = _to_numpy(e)
            if e.ndim == 1:
                e = e[None, :]
            return _l2norm_np(e)
        except Exception:
            embs = []
            for p in paths:
                embs.append(encode_image_one(p)[0])
            return _l2norm_np(np.stack(embs, axis=0))

    # ------------------------------------------------------------
    # 1) cache corpus IMAGE embeddings
    # ------------------------------------------------------------
    cache = SmartCache(cache_dir, f"{model_id}_bgevl_it2i", task, long_context, batch_size, sub_dir='img_emb')
    cache.load()
    
    to_encode, to_encode_ids = cache.get_uncached_docs(doc_ids, documents)

    if len(to_encode) > 0:
        new_embs = []
        for i in trange(0, len(to_encode), batch_size, desc="Encoding corpus images (BGE-VL)"):
            batch_paths = to_encode[i:i + batch_size]
            paths = [safe_image_path(p, blank_img) for p in batch_paths]
            
            emb = encode_images_batch(paths)
            new_embs.append(emb)

        cache.update(new_embs, to_encode_ids)
        cache.save()

    corpus_emb = _l2norm_np(cache.get_embeddings_array(doc_ids))

    # ------------------------------------------------------------
    # 3) encode QUERY = average(text_only, image_only)  (NO encode_multimodal)
    # ------------------------------------------------------------
    query_images_map = kwargs.get('query_images_map', {})

    q_embs = []
    for i in trange(len(queries), desc="Encoding queries (BGE-VL IT fused)"):
        qtext = queries[i]
        qid = str(query_ids[i])
        
        ipaths = query_images_map.get(str(qid), []) or []
        qimg = None
        if ipaths:
            img_name = Path(ipaths[0]).name
            p = image_dir / img_name
            if p.exists():
                qimg = safe_image_path(str(p), blank_img)

        t_emb = encode_text_one(qtext)  # (1, D)
        if qimg is not None:
            i_emb = encode_image_one(qimg)  # (1, D)
            fused = _l2norm_np((t_emb + i_emb) / 2.0)
            q_embs.append(fused)
        else:
            q_embs.append(t_emb)

    query_emb = np.concatenate(q_embs, axis=0)

    # ------------------------------------------------------------
    # 4) score topk
    # ------------------------------------------------------------
    return _scores_topk_from_embeddings(
        query_ids=query_ids,
        query_emb_np=query_emb,
        corpus_ids=doc_ids,
        corpus_emb_np=corpus_emb,
        excluded_ids=excluded_ids,
        topk=topk,
        chunk_size=chunk_size,
        scale=100.0
    )


# ==================== IT→I: GME Qwen2-VL (2B / 7B) ====================
@torch.no_grad()
def retrieval_gme_it2i(queries, query_ids, documents, doc_ids, task, model_id, instructions,
                       cache_dir, excluded_ids, long_context, **kwargs):
    from sentence_transformers import SentenceTransformer

    # model_id is 'gme-qwen2-vl-2b' or 'gme-qwen2-vl-7b'
    model_name = 'Alibaba-NLP/gme-Qwen2-VL-2B-Instruct' if model_id == 'gme-qwen2-vl-2b' else 'Alibaba-NLP/gme-Qwen2-VL-7B-Instruct'
    gme = SentenceTransformer(model_name)

    batch_size = kwargs.get('batch_size', 8)
    topk = kwargs.get('topk', 1000)
    chunk_size = kwargs.get('chunk_size', 50000)

    dataset_dir = Path(kwargs.get('dataset_dir', '.'))
    image_dir = dataset_dir / 'images' / task

    # ---- cache corpus image embeddings ----
    cache = SmartCache(cache_dir, f'{model_id}_gme_it2i', task, long_context, batch_size, sub_dir='img_emb')
    cache.load()
    
    to_encode, to_encode_ids = cache.get_uncached_docs(doc_ids, documents)

    if len(to_encode) > 0:
        new_embs = []
        for i in trange(0, len(to_encode), batch_size, desc="Encoding corpus images (GME)"):
            batch_paths = to_encode[i:i+batch_size]
            inputs = [{"image": p} for p in batch_paths]
            emb = gme.encode(inputs, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
            new_embs.append(emb.cpu().float().numpy())

        cache.update(new_embs, to_encode_ids)
        cache.save()

    corpus_emb = _l2norm_np(cache.get_embeddings_array(doc_ids))

    # ---- encode fused query (text+image dict) ----
    query_images_map = kwargs.get('query_images_map', {})

    q_embs = []
    for i in trange(len(queries), desc="Encoding queries (GME IT)"):
        qtext = queries[i][:4000]
        qid = query_ids[i]

        qimgs = query_images_map.get(qid, []) or []
        qimg_path = None
        if qimgs:
            img_name = Path(qimgs[0]).name
            p = image_dir / img_name
            if p.exists() and p.suffix.lower() != '.svg':
                qimg_path = str(p)

        if qimg_path is not None:
            inp = {"text": qtext, "image": qimg_path}
        else:
            inp = qtext

        emb = gme.encode([inp], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        emb = emb[0].cpu().float().numpy()[None, :]
        q_embs.append(emb)

    query_emb = _l2norm_np(np.concatenate(q_embs, axis=0))

    return _scores_topk_from_embeddings(query_ids, query_emb, doc_ids, corpus_emb,
                                        excluded_ids=excluded_ids, topk=topk, chunk_size=chunk_size, scale=100.0)


# ==================== IT→I RETRIEVAL FUNCTIONS DICTIONARY ====================
MULTIMODAL_IMAGE_RETRIEVAL_FUNCS = {
    'clip': retrieval_clip_it2i,
    'siglip': retrieval_siglip_it2i,
    'jina-clip': retrieval_jina_clip_it2i,
    'nomic-vision': retrieval_nomic_it2i,
    'bge-vl-large': retrieval_bge_vl_it2i,
    'gme-qwen2-vl-2b': retrieval_gme_it2i,
    'gme-qwen2-vl-7b': retrieval_gme_it2i,
}
