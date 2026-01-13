"""
Multimodal Pair Retrievers (IT→IT)
Models supported (ONLY what you requested):
- bge-vl-large
- clip
- siglip
- jina-clip
- gme-qwen2-vl-2b
- gme-qwen2-vl-7b
- nomic-vision

Interface matches your pipeline:
retrieval_fn(queries, query_ids, documents, doc_ids, task, model_id, instructions, cache_dir, excluded_ids, long_context, **kwargs)

For IT→IT:
- documents = pair_texts (strings)
- kwargs["doc_images"] = list of abs image paths or None (parallel to documents/doc_ids)
"""

import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path

from PIL import Image, ImageOps, ImageFile, UnidentifiedImageError
from tqdm import tqdm, trange
from sklearn.metrics.pairwise import cosine_similarity

from src.caching import SmartCache
from src.utils import cut_text, add_instruct_concatenate, last_token_pool, safe_image_path, ensure_blank
from src.retrievers.task1_text import calculate_retrieval_metrics

ImageFile.LOAD_TRUNCATED_IMAGES = True
import requests
from io import BytesIO
import clip  # pip install git+https://github.com/openai/CLIP.git

def safe_pil_rgb(img_path: str, fallback_size=(224, 224)) -> Image.Image:
    """Always return a PIL.Image in RGB."""
    if not img_path:
        return Image.new("RGB", fallback_size, color="white")
    try:
        with Image.open(img_path) as im:
            im = ImageOps.exif_transpose(im)
            im.load()
            return im.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, RuntimeError):
        return Image.new("RGB", fallback_size, color="white")


# --------------------------
# Utilities
# --------------------------

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True

def _safe_open_image(path, fallback_size=(224, 224)):
    """
    Always return a PIL RGB image.
    Accepts str/Path/PIL.Image/None.
    """
    if path is None:
        return Image.new("RGB", fallback_size, color="white")
    
    # Already a PIL Image - just convert to RGB
    if hasattr(path, 'convert'):
        try:
            im = ImageOps.exif_transpose(path)
            return im.convert("RGB")
        except Exception:
            return Image.new("RGB", fallback_size, color="white")

    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            return im
    except (UnidentifiedImageError, OSError, ValueError, RuntimeError, TypeError):
        return Image.new("RGB", fallback_size, color="white")


import torch

def _safe_vproc_pixel_values(vproc, pil_images, device, fallback_size=(224, 224)):
    """
    Process images ONE BY ONE to avoid batch crash from a single grayscale/corrupt image.
    Always returns dict(pixel_values=...) on device.
    """
    px_list = []

    for im in pil_images:
        # force PIL RGB no matter what
        if not isinstance(im, Image.Image):
            im = _safe_open_image(im, fallback_size=fallback_size)
        else:
            im = ImageOps.exif_transpose(im).convert("RGB")

        try:
            out = vproc(images=im, return_tensors="pt")
        except Exception:
            out = vproc(images=Image.new("RGB", fallback_size, "white"), return_tensors="pt")

        px_list.append(out["pixel_values"][0])  # (C,H,W)

    pixel_values = torch.stack(px_list, dim=0).to(device)
    return {"pixel_values": pixel_values}


def _ensure_blank_image_file(cache_dir_path: Path, size=(224, 224)) -> str:
    _ensure_dir(cache_dir_path)
    blank_path = cache_dir_path / "__blank_fallback.png"
    if not blank_path.exists():
        Image.new("RGB", size, color="white").save(blank_path)
    return str(blank_path)


def _normalize_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def _load_cache(emb_path: Path, map_path: Path):
    if emb_path.exists() and map_path.exists():
        emb = np.load(emb_path)
        with open(map_path, "r") as f:
            mp = json.load(f)
        return emb, mp
    return None, {}


def _save_cache(emb: np.ndarray, mp: dict, emb_path: Path, map_path: Path):
    np.save(emb_path, emb.astype(np.float32))
    with open(map_path, "w") as f:
        json.dump(mp, f, indent=2)


def _compute_topk_scores_chunked(
    query_emb: torch.Tensor,                 # (Q, D) on CPU
    doc_emb_cpu: np.ndarray,                 # (N, D) float32 normalized
    doc_ids: list,
    query_ids: list,
    excluded_ids: dict,
    chunk_size: int = 50000,
    topk: int = 1000,
    device: str = "cuda"
):
    """
    Chunked scoring to avoid building full QxN matrix.
    Returns {qid: {doc_id: score}}
    """
    Q = query_emb.shape[0]
    N = len(doc_ids)
    topk = min(topk, N)

    # map doc_id -> position in doc_ids list
    doc_pos = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    out = {}
    for qi in range(Q):
        qid = str(query_ids[qi])
        qv = query_emb[qi].to(device)  # (D,)
        best_vals = torch.empty(0, device="cpu")
        best_idx = torch.empty(0, dtype=torch.long, device="cpu")

        excl = excluded_ids.get(qid, [])
        excl_pos = []
        if excl:
            for d in excl:
                p = doc_pos.get(d)
                if p is not None:
                    excl_pos.append(p)
        excl_pos = set(excl_pos)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = torch.from_numpy(doc_emb_cpu[start:end]).to(device)  # (M,D)
            scores = torch.matmul(chunk, qv) * 100.0                    # (M,)

            # mask excluded inside this chunk
            if excl_pos:
                local = [p - start for p in excl_pos if start <= p < end]
                if local:
                    scores[torch.tensor(local, device=device)] = -1e9

            k = min(topk, end - start)
            vals, idx = torch.topk(scores, k=k, largest=True)
            idx = idx + start  # global positions

            vals = vals.detach().cpu()
            idx = idx.detach().cpu()

            if best_vals.numel() == 0:
                best_vals, best_idx = vals, idx
            else:
                merged_vals = torch.cat([best_vals, vals], dim=0)
                merged_idx = torch.cat([best_idx, idx], dim=0)
                kk = min(topk, merged_vals.numel())
                vals2, sel = torch.topk(merged_vals, k=kk, largest=True)
                idx2 = merged_idx[sel]
                best_vals, best_idx = vals2, idx2

        # build dict
        pairs = sorted(zip(best_idx.tolist(), best_vals.tolist()), key=lambda x: x[1], reverse=True)
        out[qid] = {str(doc_ids[i]): float(s) for i, s in pairs[:topk]}

    return out



def _clip_safe_images(processor, images, device, size=(224,224)):
    """
    Process CLIP images ONE BY ONE.
    If any image fails, replace with blank RGB.
    """
    pixel_values = []

    for im in images:
        try:
            out = processor(images=im, return_tensors="pt")
            pv = out["pixel_values"][0]
            if pv.shape[0] != 3:   # channel sanity check
                raise ValueError("non-RGB tensor")
        except Exception:
            blank = Image.new("RGB", size, "white")
            out = processor(images=blank, return_tensors="pt")
            pv = out["pixel_values"][0]

        pixel_values.append(pv)

    return {"pixel_values": torch.stack(pixel_values, dim=0).to(device)}

# --------------------------
# CLIP IT→IT
# --------------------------
@torch.no_grad()
def retrieval_clip_it2it(queries, query_ids, documents, doc_ids, task, model_id,
                         instructions, cache_dir, excluded_ids, long_context, **kwargs):
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = kwargs.get("model_name", "openai/clip-vit-large-patch14")
    batch_size = int(kwargs.get("batch_size", 8))
    chunk_size = int(kwargs.get("chunk_size", 50000))
    topk = int(kwargs.get("topk", 1000))
    doc_images = kwargs["doc_images"]

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    cache_dir_path = Path(cache_dir) / "pair_emb" / "clip" / task / f"bs_{batch_size}"
    _ensure_dir(cache_dir_path)
    emb_path = cache_dir_path / "embeddings.npy"
    map_path = cache_dir_path / "doc_id_mapping.json"

    emb_arr, mp = _load_cache(emb_path, map_path)
    if emb_arr is None:
        emb_arr = np.zeros((0, model.config.projection_dim), dtype=np.float32)

    # encode missing docs
    missing = [i for i, did in enumerate(doc_ids) if did not in mp]
    if missing:
        new_embs = []
        for i in trange(0, len(missing), batch_size, desc="Encoding pair docs (CLIP)"):
            idxs = missing[i:i+batch_size]
            batch_texts = [documents[j] for j in idxs]
            batch_imgs = [_safe_open_image(doc_images[j], fallback_size=(224,224)) for j in idxs]

            t_in = processor(text=batch_texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=77).to(device)
            # i_in = processor(images=batch_imgs, return_tensors="pt").to(device)
            i_in = _clip_safe_images(processor, batch_imgs, device)
            t_feat = model.get_text_features(**t_in)
            i_feat = model.get_image_features(**i_in)
            comb = (t_feat + i_feat) / 2.0
            comb = F.normalize(comb, p=2, dim=1).detach().cpu().float().numpy()
            new_embs.append(comb)

        new_embs = np.concatenate(new_embs, axis=0) if new_embs else np.zeros((0, emb_arr.shape[1]), dtype=np.float32)
        start = emb_arr.shape[0]
        emb_arr = np.concatenate([emb_arr, new_embs], axis=0)
        for k, j in enumerate(missing):
            mp[doc_ids[j]] = start + k
        _save_cache(emb_arr, mp, emb_path, map_path)

    # build doc matrix in current order (CPU)
    idxs = [mp[did] for did in doc_ids]
    doc_emb_cpu = emb_arr[idxs].astype(np.float32)
    doc_emb_cpu = _normalize_np(doc_emb_cpu)

    # encode queries (text+image from query_images_map passed via kwargs)
    query_images_map = kwargs.get('query_images_map', {})

    q_embs = []
    for i in trange(0, len(queries), batch_size, desc="Encoding queries (CLIP)"):
        batch_q = queries[i:i+batch_size]
        batch_ids = query_ids[i:i+batch_size]

        imgs = []
        for qid in batch_ids:
            qimgs = query_images_map.get(str(qid), []) or []
            if qimgs:
                first_img = qimgs[0]
                # Check if it's a PIL Image or path string
                if hasattr(first_img, 'convert'):
                    imgs.append(_safe_open_image(first_img, fallback_size=(224,224)))
                else:
                    # It's a file path - try to load it
                    imgs.append(_safe_open_image(str(first_img), fallback_size=(224,224)))
            else:
                imgs.append(_safe_open_image(None, fallback_size=(224,224)))

        t_in = processor(text=batch_q, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        i_in = processor(images=imgs, return_tensors="pt").to(device)

        t_feat = model.get_text_features(**t_in)
        i_feat = model.get_image_features(**i_in)
        comb = (t_feat + i_feat) / 2.0
        comb = F.normalize(comb, p=2, dim=1).detach().cpu()

        q_embs.append(comb)

    query_emb = torch.cat(q_embs, dim=0).cpu()

    return _compute_topk_scores_chunked(
        query_emb=query_emb,
        doc_emb_cpu=doc_emb_cpu,
        doc_ids=doc_ids,
        query_ids=query_ids,
        excluded_ids=excluded_ids,
        chunk_size=chunk_size,
        topk=topk,
        device=device
    )

def _siglip_safe_images(processor, images, device, size=(384, 384)):
    """
    Process SigLIP images one-by-one.
    Replace any broken image with blank RGB.
    """
    pixel_values = []

    for im in images:
        try:
            # Ensure PIL.Image
            if not isinstance(im, Image.Image):
                raise ValueError("not a PIL image")

            im = ImageOps.exif_transpose(im).convert("RGB")

            out = processor(images=im, return_tensors="pt")
            pv = out["pixel_values"][0]

            # Sanity: must be 3-channel
            if pv.shape[0] != 3:
                raise ValueError("non-RGB tensor")

        except Exception:
            blank = Image.new("RGB", size, "white")
            out = processor(images=blank, return_tensors="pt")
            pv = out["pixel_values"][0]

        pixel_values.append(pv)

    return {"pixel_values": torch.stack(pixel_values, dim=0).to(device)}

# --------------------------
# SigLIP IT→IT
# --------------------------
@torch.no_grad()
def retrieval_siglip_it2it(queries, query_ids, documents, doc_ids, task, model_id,
                           instructions, cache_dir, excluded_ids, long_context, **kwargs):
    from transformers import AutoModel, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = kwargs.get("model_name", "google/siglip-so400m-patch14-384")
    batch_size = int(kwargs.get("batch_size", 8))
    chunk_size = int(kwargs.get("chunk_size", 50000))
    topk = int(kwargs.get("topk", 1000))
    doc_images = kwargs["doc_images"]

    model = AutoModel.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()

    cache_dir_path = Path(cache_dir) / "pair_emb" / "siglip" / task / f"bs_{batch_size}"
    _ensure_dir(cache_dir_path)
    emb_path = cache_dir_path / "embeddings.npy"
    map_path = cache_dir_path / "doc_id_mapping.json"

    emb_arr, mp = _load_cache(emb_path, map_path)
    if emb_arr is None:
        # projection dim unknown until first encode; use temporary list then stack
        emb_arr = np.zeros((0, 1152), dtype=np.float32)  # safe placeholder

    missing = [i for i, did in enumerate(doc_ids) if did not in mp]
    if missing:
        new_embs = []
        for i in trange(0, len(missing), batch_size, desc="Encoding pair docs (SigLIP)"):
            idxs = missing[i:i+batch_size]
            batch_texts = [documents[j] for j in idxs]
            batch_imgs = [_safe_open_image(doc_images[j], fallback_size=(384,384)) for j in idxs]

            # inputs = processor(text=batch_texts, images=batch_imgs, return_tensors="pt",
            #                    padding="max_length", truncation=True).to(device)
            text_inputs = processor(
                text=batch_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True
            ).to(device)
            image_inputs = _siglip_safe_images(processor, batch_imgs, device)
            
            inputs = {
                "input_ids": text_inputs["input_ids"],
                "attention_mask": text_inputs.get("attention_mask"),
                "pixel_values": image_inputs["pixel_values"],
            }

            t_feat = model.get_text_features(input_ids=inputs["input_ids"],
                                             attention_mask=inputs.get("attention_mask", None))
            i_feat = model.get_image_features(pixel_values=inputs["pixel_values"])
            comb = (t_feat + i_feat) / 2.0
            comb = F.normalize(comb, p=2, dim=1).detach().cpu().float().numpy()
            new_embs.append(comb)

        new_embs = np.concatenate(new_embs, axis=0) if new_embs else np.zeros((0, emb_arr.shape[1]), dtype=np.float32)

        # if placeholder dim mismatched, fix on first run
        if emb_arr.shape[0] == 0 and emb_arr.shape[1] != new_embs.shape[1]:
            emb_arr = np.zeros((0, new_embs.shape[1]), dtype=np.float32)

        start = emb_arr.shape[0]
        emb_arr = np.concatenate([emb_arr, new_embs], axis=0)
        for k, j in enumerate(missing):
            mp[doc_ids[j]] = start + k
        _save_cache(emb_arr, mp, emb_path, map_path)

    idxs = [mp[did] for did in doc_ids]
    doc_emb_cpu = emb_arr[idxs].astype(np.float32)
    doc_emb_cpu = _normalize_np(doc_emb_cpu)

    # query images (from query_images_map passed via kwargs)
    query_images_map = kwargs.get('query_images_map', {})

    q_embs = []
    for i in trange(0, len(queries), batch_size, desc="Encoding queries (SigLIP)"):
        batch_q = queries[i:i+batch_size]
        batch_ids = query_ids[i:i+batch_size]

        imgs = []
        for qid in batch_ids:
            qimgs = query_images_map.get(str(qid), []) or []
            if qimgs:
                first_img = qimgs[0]
                if hasattr(first_img, 'convert'):
                    imgs.append(_safe_open_image(first_img, fallback_size=(384,384)))
                else:
                    imgs.append(_safe_open_image(str(first_img), fallback_size=(384,384)))
            else:
                imgs.append(_safe_open_image(None, fallback_size=(384,384)))

        inputs = processor(text=batch_q, images=imgs, return_tensors="pt",
                           padding="max_length", truncation=True).to(device)
        t_feat = model.get_text_features(input_ids=inputs["input_ids"],
                                         attention_mask=inputs.get("attention_mask", None))
        i_feat = model.get_image_features(pixel_values=inputs["pixel_values"])
        comb = (t_feat + i_feat) / 2.0
        comb = F.normalize(comb, p=2, dim=1).detach().cpu()
        q_embs.append(comb)

    query_emb = torch.cat(q_embs, dim=0).cpu()

    return _compute_topk_scores_chunked(
        query_emb=query_emb,
        doc_emb_cpu=doc_emb_cpu,
        doc_ids=doc_ids,
        query_ids=query_ids,
        excluded_ids=excluded_ids,
        chunk_size=chunk_size,
        topk=topk,
        device=device
    )


# --------------------------
# Jina-CLIP IT→IT
# --------------------------
@torch.no_grad()
def retrieval_jina_clip_it2it(queries, query_ids, documents, doc_ids, task, model_id,
                              instructions, cache_dir, excluded_ids, long_context, **kwargs):
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = kwargs.get("model_name", "jinaai/jina-clip-v1")
    batch_size = int(kwargs.get("batch_size", 8))
    chunk_size = int(kwargs.get("chunk_size", 50000))
    topk = int(kwargs.get("topk", 1000))
    doc_images = kwargs["doc_images"]

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    cache_dir_path = Path(cache_dir) / "pair_emb" / "jina_clip" / task / f"bs_{batch_size}"
    _ensure_dir(cache_dir_path)
    emb_path = cache_dir_path / "embeddings.npy"
    map_path = cache_dir_path / "doc_id_mapping.json"

    emb_arr, mp = _load_cache(emb_path, map_path)
    if emb_arr is None:
        emb_arr = np.zeros((0, 768), dtype=np.float32)  # placeholder

    missing = [i for i, did in enumerate(doc_ids) if did not in mp]
    if missing:
        new_embs = []
        for i in trange(0, len(missing), batch_size, desc="Encoding pair docs (Jina-CLIP)"):
            idxs = missing[i:i+batch_size]
            batch_texts = [documents[j] for j in idxs]
            batch_imgs = [_safe_open_image(doc_images[j], fallback_size=(224,224)) for j in idxs]

            t = model.encode_text(batch_texts)
            iemb = model.encode_image(batch_imgs)

            if isinstance(t, np.ndarray):
                t = torch.from_numpy(t)
            if isinstance(iemb, np.ndarray):
                iemb = torch.from_numpy(iemb)

            t = t.detach().cpu().float()
            iemb = iemb.detach().cpu().float()
            comb = (t + iemb) / 2.0
            comb = F.normalize(comb, p=2, dim=1).cpu().numpy()
            new_embs.append(comb)

        new_embs = np.concatenate(new_embs, axis=0) if new_embs else np.zeros((0, emb_arr.shape[1]), dtype=np.float32)

        if emb_arr.shape[0] == 0 and emb_arr.shape[1] != new_embs.shape[1]:
            emb_arr = np.zeros((0, new_embs.shape[1]), dtype=np.float32)

        start = emb_arr.shape[0]
        emb_arr = np.concatenate([emb_arr, new_embs], axis=0)
        for k, j in enumerate(missing):
            mp[doc_ids[j]] = start + k
        _save_cache(emb_arr, mp, emb_path, map_path)

    idxs = [mp[did] for did in doc_ids]
    doc_emb_cpu = emb_arr[idxs].astype(np.float32)
    doc_emb_cpu = _normalize_np(doc_emb_cpu)

    # query images from jsonl
    # query images (from query_images_map passed via kwargs)
    query_images_map = kwargs.get('query_images_map', {})

    q_embs = []
    for i in trange(0, len(queries), batch_size, desc="Encoding queries (Jina-CLIP)"):
        batch_q = queries[i:i+batch_size]
        batch_ids = query_ids[i:i+batch_size]

        imgs = []
        for qid in batch_ids:
            qimgs = query_images_map.get(str(qid), []) or []
            if qimgs:
                first_img = qimgs[0]
                if hasattr(first_img, 'convert'):
                    imgs.append(_safe_open_image(first_img, fallback_size=(224,224)))
                else:
                    imgs.append(_safe_open_image(str(first_img), fallback_size=(224,224)))
            else:
                imgs.append(_safe_open_image(None, fallback_size=(224,224)))

        t = model.encode_text(batch_q)
        iemb = model.encode_image(imgs)

        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t)
        if isinstance(iemb, np.ndarray):
            iemb = torch.from_numpy(iemb)

        comb = (t.detach().cpu().float() + iemb.detach().cpu().float()) / 2.0
        comb = F.normalize(comb, p=2, dim=1).cpu()
        q_embs.append(comb)

    query_emb = torch.cat(q_embs, dim=0).cpu()

    return _compute_topk_scores_chunked(
        query_emb=query_emb,
        doc_emb_cpu=doc_emb_cpu,
        doc_ids=doc_ids,
        query_ids=query_ids,
        excluded_ids=excluded_ids,
        chunk_size=chunk_size,
        topk=topk,
        device=device
    )


# --------------------------
# BGE-VL-large IT→IT (robust truncation like your code)
# --------------------------
@torch.no_grad()
def retrieval_bge_vl_large_it2it(
    queries, query_ids, documents, doc_ids, task, model_id,
    instructions, cache_dir, excluded_ids, long_context, **kwargs
):
    """
    BGE-VL-large IT→IT (robust):
    - Pass image as PATH (string). Do NOT pass PIL image (your error).
    - Sanitize images -> RGB PNG on disk (so grayscale/corrupt won't break CLIP normalize).
    - Aggressive text truncation + retries to avoid long-context runtime issues.
    - Cache doc embeddings.
    """
    import json, hashlib
    import numpy as np
    import torch
    import torch.nn.functional as F
    from pathlib import Path
    from tqdm import tqdm
    from PIL import Image, ImageOps, ImageFile, UnidentifiedImageError
    from io import BytesIO
    from transformers import AutoModel

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = kwargs.get("model_name", "BAAI/BGE-VL-large")
    batch_size = int(kwargs.get("batch_size", 16))
    chunk_size = int(kwargs.get("chunk_size", 50000))
    topk = int(kwargs.get("topk", 1000))
    doc_images = kwargs["doc_images"]

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.set_processor(model_name)
    model.eval().to(device)

    def ultra_safe_truncate(text, max_chars=150):
        text = (text or "").strip()
        if len(text) > max_chars:
            t = text[:max_chars]
            last_space = t.rfind(" ")
            if last_space > max_chars * 0.7:
                t = t[:last_space]
            return t.strip()
        return text

    cache_dir_path = Path(cache_dir) / "pair_emb" / "bge_vl_large" / task / f"bs_{batch_size}"
    _ensure_dir(cache_dir_path)
    emb_path = cache_dir_path / "embeddings.npy"
    map_path = cache_dir_path / "doc_id_mapping.json"

    blank_path = _ensure_blank_image_file(cache_dir_path, size=(224, 224))

    rgb_fixed_dir = cache_dir_path / "__rgb_fixed"
    _ensure_dir(rgb_fixed_dir)

    def safe_rgb_path(img_path: str) -> str:
        """Return path to a valid RGB PNG on disk; else blank_path."""
        if not img_path:
            return blank_path

        p = Path(str(img_path))
        if (not p.exists()) or p.suffix.lower() == ".svg":
            return blank_path

        try:
            sig = f"{p.resolve()}|{p.stat().st_size}|{p.stat().st_mtime}"
        except Exception:
            sig = str(p)

        key = hashlib.md5(sig.encode("utf-8")).hexdigest()
        fixed = rgb_fixed_dir / f"{key}.png"
        if fixed.exists():
            return str(fixed)

        try:
            with Image.open(str(p)) as im:
                im = ImageOps.exif_transpose(im)
                im.load()

                # 🔥 FORCE remove palette & transparency
                if im.mode in ("P", "RGBA", "LA"):
                    bg = Image.new("RGB", im.size, (255, 255, 255))
                    if im.mode in ("RGBA", "LA"):
                        bg.paste(im, mask=im.split()[-1])
                    else:
                        bg.paste(im)
                    im = bg
                else:
                    im = im.convert("RGB")

                # 🔥 KILL leftover PNG metadata
                im.info.pop("transparency", None)

                im.save(str(fixed), format="PNG")
            return str(fixed)

        except (UnidentifiedImageError, OSError, ValueError, RuntimeError):
            return blank_path


    def _encode_one(text: str, img_path_or_none: str):
        """
        Encode one (text, image?) item robustly.
        IMPORTANT: pass image as PATH string (or file-like BytesIO if needed).
        """
        if img_path_or_none is None:
            emb = model.encode(text=[text])
        else:
            img_path = safe_rgb_path(img_path_or_none)

            # 1) primary: PATH (works with your current BGE wrapper that does Image.open(path))
            try:
                emb = model.encode(images=[img_path], text=[text])
            except Exception as e:
                msg = str(e)

                # 2) fallback: file-like (works if wrapper does Image.open(fp))
                try:
                    with Image.open(img_path) as im:
                        im = ImageOps.exif_transpose(im).convert("RGB")
                        buf = BytesIO()
                        im.save(buf, format="PNG")
                        buf.seek(0)
                    emb = model.encode(images=[buf], text=[text])
                except Exception:
                    # last resort: no-image
                    emb = model.encode(text=[text])

        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        if emb.ndim == 1:
            emb = emb.view(1, -1)
        emb = F.normalize(emb.float(), p=2, dim=1)
        return emb[0].detach().cpu().numpy()

    emb_arr, mp = _load_cache(emb_path, map_path)
    if emb_arr is None:
        emb_arr = np.zeros((0, 1024), dtype=np.float32)

    # --------------------------
    # Encode missing doc pairs
    # --------------------------
    missing = [i for i, did in enumerate(doc_ids) if did not in mp]
    if missing:
        new_embs = []
        for j in tqdm(missing, desc="Encoding pair docs (BGE-VL-large)"):
            raw_text = documents[j]
            imgp = doc_images[j]  # abs path or None

            emb_vec = None
            for char_limit in [150, 120, 100, 80, 60, 40, 20]:
                try:
                    tt = ultra_safe_truncate(raw_text, max_chars=char_limit)
                    if len(tt) < 3:
                        tt = (raw_text or "")[:20].strip() or "document"
                    emb_vec = _encode_one(tt, imgp)
                    break
                except RuntimeError as e:
                    if "size of tensor" in str(e):
                        continue
                    raise

            if emb_vec is None:
                tt = (raw_text or "")[:15].strip() or "document"
                emb_vec = _encode_one(tt, imgp)

            new_embs.append(emb_vec[None, :])

        new_embs = np.concatenate(new_embs, axis=0).astype(np.float32)
        if emb_arr.shape[0] == 0 and emb_arr.shape[1] != new_embs.shape[1]:
            emb_arr = np.zeros((0, new_embs.shape[1]), dtype=np.float32)

        start = emb_arr.shape[0]
        emb_arr = np.concatenate([emb_arr, new_embs], axis=0)
        for k, j in enumerate(missing):
            mp[doc_ids[j]] = start + k
        _save_cache(emb_arr, mp, emb_path, map_path)

    # doc matrix in current order
    idxs = [mp[did] for did in doc_ids]
    doc_emb_cpu = emb_arr[idxs].astype(np.float32)
    doc_emb_cpu = _normalize_np(doc_emb_cpu)

    # --------------------------
    # Encode queries
    # --------------------------
    query_images_map = kwargs.get('query_images_map', {})

    q_embs = []
    for i in tqdm(range(len(queries)), desc="Encoding queries (BGE-VL-large)"):
        raw_q = queries[i]
        qid = str(query_ids[i])

        qimgs = query_images_map.get(qid, []) or []

        img_use = None
        if qimgs:
            first_img = qimgs[0]
            if hasattr(first_img, 'convert'):
                # PIL Image - save to temp file for encoding
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    first_img.convert('RGB').save(tmp, format='PNG')
                    img_use = tmp.name
            elif isinstance(first_img, str):
                img_use = first_img

        emb_vec = None
        for char_limit in [150, 120, 100, 80, 60, 40, 20]:
            try:
                tt = ultra_safe_truncate(raw_q, max_chars=char_limit)
                if len(tt) < 3:
                    tt = (raw_q or "")[:20].strip() or "query"
                emb_vec = _encode_one(tt, img_use)
                break
            except RuntimeError as e:
                if "size of tensor" in str(e):
                    continue
                raise

        if emb_vec is None:
            tt = (raw_q or "")[:15].strip() or "query"
            emb_vec = _encode_one(tt, img_use)

        q_embs.append(torch.from_numpy(emb_vec))

    query_emb = torch.stack(q_embs, dim=0).float().cpu()

    return _compute_topk_scores_chunked(
        query_emb=query_emb,
        doc_emb_cpu=doc_emb_cpu,
        doc_ids=doc_ids,
        query_ids=query_ids,
        excluded_ids=excluded_ids,
        chunk_size=chunk_size,
        topk=topk,
        device=device
    )





# --------------------------
# GME-Qwen2-VL IT→IT (FIXED: no mixed batches)
# --------------------------
@torch.no_grad()
def retrieval_gme_qwen2_vl_it2it(
    queries, query_ids, documents, doc_ids, task, model_id,
    instructions, cache_dir, excluded_ids, long_context, **kwargs
):
    from sentence_transformers import SentenceTransformer
    from pathlib import Path
    import numpy as np
    import json
    import torch
    from PIL import Image, UnidentifiedImageError
    from tqdm import trange

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = int(kwargs.get("batch_size", 8))
    chunk_size = int(kwargs.get("chunk_size", 50000))
    topk = int(kwargs.get("topk", 1000))
    doc_images = kwargs["doc_images"]  # list aligned with documents/doc_ids

    if model_id == "gme-qwen2_vl_7b" or model_id == "gme-qwen2-vl-7b":
        model_name = kwargs.get("model_name", "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct")
        tag = "gme_qwen2_vl_7b"
    else:
        model_name = kwargs.get("model_name", "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct")
        tag = "gme_qwen2_vl_2b"

    gme = SentenceTransformer(model_name, trust_remote_code=True)

    cache_dir_path = Path(cache_dir) / "pair_emb" / tag / task / f"bs_{batch_size}"
    _ensure_dir(cache_dir_path)
    emb_path = cache_dir_path / "embeddings.npy"
    map_path = cache_dir_path / "doc_id_mapping.json"

    blank_path = _ensure_blank_image_file(cache_dir_path, size=(224, 224))

    emb_arr, mp = _load_cache(emb_path, map_path)
    if emb_arr is None:
        emb_arr = np.zeros((0, 1), dtype=np.float32)  # resized after first encode
    if mp is None:
        mp = {}

    def safe_img_path(p: str) -> str:
        """Return a readable local image path, else blank_path."""
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

    def gme_encode_text_only(text_list):
        # IMPORTANT: do NOT pass extra kwargs like all_images
        return gme.encode(
            text_list,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def gme_encode_multimodal(mm_list):
        # mm_list: list of {"text":..., "image":...}
        return gme.encode(
            mm_list,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    # --------------------------
    # 1) Encode missing DOC embeddings (NO mixed batches)
    # --------------------------
    missing = [i for i, did in enumerate(doc_ids) if did not in mp]
    if missing:
        # split indices by whether they have an image
        idx_noimg = [j for j in missing if doc_images[j] is None]
        idx_img = [j for j in missing if doc_images[j] is not None]

        # will store embeddings by doc index
        emb_by_j = {}

        # --- text-only docs ---
        for i in trange(0, len(idx_noimg), batch_size, desc="Encoding pair docs (GME text-only)"):
            batch_js = idx_noimg[i:i + batch_size]
            texts = [(documents[j] or "")[:4000] for j in batch_js]
            emb = gme_encode_text_only(texts).detach().cpu().float().numpy()
            for k, j in enumerate(batch_js):
                emb_by_j[j] = emb[k]

        # --- multimodal docs ---
        for i in trange(0, len(idx_img), batch_size, desc="Encoding pair docs (GME multimodal)"):
            batch_js = idx_img[i:i + batch_size]
            inputs = []
            for j in batch_js:
                txt = (documents[j] or "")[:4000]
                imgp = safe_img_path(doc_images[j])
                inputs.append({"text": txt, "image": imgp})
            emb = gme_encode_multimodal(inputs).detach().cpu().float().numpy()
            for k, j in enumerate(batch_js):
                emb_by_j[j] = emb[k]

        # build new_embs aligned with `missing` order (important for caching)
        new_embs = np.stack([emb_by_j[j] for j in missing], axis=0).astype(np.float32)

        # resize empty emb_arr to correct dim
        if emb_arr.shape[1] == 1 and new_embs.shape[1] != 1:
            emb_arr = np.zeros((0, new_embs.shape[1]), dtype=np.float32)

        start = emb_arr.shape[0]
        emb_arr = np.concatenate([emb_arr, new_embs], axis=0)

        for k, j in enumerate(missing):
            mp[doc_ids[j]] = start + k

        _save_cache(emb_arr, mp, emb_path, map_path)

    # gather doc embeddings in the same order as doc_ids
    idxs = [mp[did] for did in doc_ids]
    doc_emb_cpu = emb_arr[idxs].astype(np.float32)
    doc_emb_cpu = _normalize_np(doc_emb_cpu)

    # --------------------------
    # 2) Load query images (from query_images_map via kwargs)
    # --------------------------
    query_images_map = kwargs.get('query_images_map', {})

    # --------------------------
    # 3) Encode QUERY embeddings (NO mixed batches)
    # --------------------------
    q_emb_out = [None] * len(queries)

    # split query indices by whether they have a readable image
    q_idx_noimg = []
    q_idx_img = []

    for i in range(len(queries)):
        qid = str(query_ids[i])
        qimgs = query_images_map.get(qid, []) or []
        img_ok = False
        img_path = None

        if qimgs:
            first_img = qimgs[0]
            if hasattr(first_img, 'convert'):
                # PIL Image - save to temp file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    first_img.convert('RGB').save(tmp, format='PNG')
                    img_path = tmp.name
                    img_ok = True
            elif isinstance(first_img, str):
                img_path = safe_img_path(first_img)
                img_ok = True

        if img_ok:
            q_idx_img.append((i, img_path))
        else:
            q_idx_noimg.append(i)

    # --- text-only queries ---
    for i in trange(0, len(q_idx_noimg), batch_size, desc="Encoding queries (GME text-only)"):
        batch_is = q_idx_noimg[i:i + batch_size]
        texts = [(queries[j] or "")[:4000] for j in batch_is]
        emb = gme_encode_text_only(texts).detach().cpu().float().numpy()
        for k, j in enumerate(batch_is):
            q_emb_out[j] = emb[k]

    # --- multimodal queries ---
    for i in trange(0, len(q_idx_img), batch_size, desc="Encoding queries (GME multimodal)"):
        batch = q_idx_img[i:i + batch_size]  # list of (query_index, img_path)
        inputs = []
        idxs_local = []
        for (qi, imgp) in batch:
            idxs_local.append(qi)
            txt = (queries[qi] or "")[:4000]
            inputs.append({"text": txt, "image": imgp})
        emb = gme_encode_multimodal(inputs).detach().cpu().float().numpy()
        for k, qi in enumerate(idxs_local):
            q_emb_out[qi] = emb[k]

    query_emb = torch.tensor(np.stack(q_emb_out, axis=0), dtype=torch.float32).cpu()

    # --------------------------
    # 4) Compute scores
    # --------------------------
    return _compute_topk_scores_chunked(
        query_emb=query_emb,
        doc_emb_cpu=doc_emb_cpu,
        doc_ids=doc_ids,
        query_ids=query_ids,
        excluded_ids=excluded_ids,
        chunk_size=chunk_size,
        topk=topk,
        device=device,
    )


# --------------------------
# Nomic-vision IT→IT
# --------------------------
@torch.no_grad()
def retrieval_nomic_it2it(queries, query_ids, documents, doc_ids, task, model_id,
                         instructions, cache_dir, excluded_ids, long_context, **kwargs):
    from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = int(kwargs.get("batch_size", 8))
    chunk_size = int(kwargs.get("chunk_size", 50000))
    topk = int(kwargs.get("topk", 1000))
    doc_images = kwargs["doc_images"]

    text_model_name = "nomic-ai/nomic-embed-text-v1.5"
    vision_model_name = "nomic-ai/nomic-embed-vision-v1.5"

    tok = AutoTokenizer.from_pretrained(text_model_name)
    text_model = AutoModel.from_pretrained(text_model_name, trust_remote_code=True).to(device).eval()

    vproc = AutoImageProcessor.from_pretrained(vision_model_name)
    vision_model = AutoModel.from_pretrained(vision_model_name, trust_remote_code=True).to(device).eval()

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    cache_dir_path = Path(cache_dir) / "pair_emb" / "nomic_it2it" / task / f"bs_{batch_size}"
    _ensure_dir(cache_dir_path)
    emb_path = cache_dir_path / "embeddings.npy"
    map_path = cache_dir_path / "doc_id_mapping.json"

    emb_arr, mp = _load_cache(emb_path, map_path)
    if emb_arr is None:
        emb_arr = np.zeros((0, 768), dtype=np.float32)  # placeholder

    missing = [i for i, did in enumerate(doc_ids) if did not in mp]
    if missing:
        new_embs = []
        for i in trange(0, len(missing), batch_size, desc="Encoding pair docs (Nomic)"):
            idxs = missing[i:i+batch_size]
            batch_texts = [(documents[j] or "")[:8000] for j in idxs]

            enc = tok(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=8192).to(device)
            out = text_model(**enc)
            t_emb = mean_pooling(out, enc["attention_mask"])
            t_emb = F.layer_norm(t_emb, normalized_shape=(t_emb.shape[1],))
            t_emb = F.normalize(t_emb, p=2, dim=1)

            # image emb: compute only for those with image
            img_emb = torch.zeros_like(t_emb)
            real_idx = []
            real_imgs = []
            for k, j in enumerate(idxs):
                if doc_images[j] is not None:
                    real_idx.append(k)
                    real_imgs.append(_safe_open_image(doc_images[j], fallback_size=(224,224)))
            if real_imgs:
                vin  = _safe_vproc_pixel_values(vproc, real_imgs, device, fallback_size=(224, 224))
                vout = vision_model(**vin)
                v_emb = vout.last_hidden_state[:, 0]
                v_emb = F.normalize(v_emb, p=2, dim=1)
                img_emb[torch.tensor(real_idx, device=device)] = v_emb

            comb = (t_emb + img_emb) / 2.0
            comb = F.normalize(comb, p=2, dim=1).detach().cpu().float().numpy()
            new_embs.append(comb)

        new_embs = np.concatenate(new_embs, axis=0) if new_embs else np.zeros((0, emb_arr.shape[1]), dtype=np.float32)
        if emb_arr.shape[0] == 0 and emb_arr.shape[1] != new_embs.shape[1]:
            emb_arr = np.zeros((0, new_embs.shape[1]), dtype=np.float32)

        start = emb_arr.shape[0]
        emb_arr = np.concatenate([emb_arr, new_embs], axis=0)
        for k, j in enumerate(missing):
            mp[doc_ids[j]] = start + k
        _save_cache(emb_arr, mp, emb_path, map_path)

    idxs = [mp[did] for did in doc_ids]
    doc_emb_cpu = emb_arr[idxs].astype(np.float32)
    doc_emb_cpu = _normalize_np(doc_emb_cpu)

    # query images (from query_images_map passed via kwargs)
    query_images_map = kwargs.get('query_images_map', {})

    q_embs = []
    for i in trange(0, len(queries), batch_size, desc="Encoding queries (Nomic)"):
        batch_q = queries[i:i+batch_size]
        batch_ids = query_ids[i:i+batch_size]

        # text query prefix
        qtexts = [f"search_query: {(t or '')[:8000]}" for t in batch_q]
        enc = tok(qtexts, padding=True, truncation=True, return_tensors="pt", max_length=8192).to(device)
        out = text_model(**enc)
        t_emb = mean_pooling(out, enc["attention_mask"])
        t_emb = F.layer_norm(t_emb, normalized_shape=(t_emb.shape[1],))
        t_emb = F.normalize(t_emb, p=2, dim=1)

        img_emb = torch.zeros_like(t_emb)
        real_idx = []
        real_imgs = []
        for k, qid in enumerate(batch_ids):
            qimgs = query_images_map.get(str(qid), []) or []
            if qimgs:
                first_img = qimgs[0]
                if hasattr(first_img, 'convert'):
                    real_idx.append(k)
                    real_imgs.append(_safe_open_image(first_img, fallback_size=(224,224)))
                elif isinstance(first_img, str):
                    real_idx.append(k)
                    real_imgs.append(_safe_open_image(first_img, fallback_size=(224,224)))

        if real_imgs:
            vin = _safe_vproc_pixel_values(vproc, real_imgs, device, fallback_size=(224, 224))
            vout = vision_model(**vin)
            v_emb = vout.last_hidden_state[:, 0]
            v_emb = F.normalize(v_emb, p=2, dim=1)
            img_emb[torch.tensor(real_idx, device=device)] = v_emb

        comb = (t_emb + img_emb) / 2.0
        comb = F.normalize(comb, p=2, dim=1).detach().cpu()
        q_embs.append(comb)

    query_emb = torch.cat(q_embs, dim=0).cpu()

    return _compute_topk_scores_chunked(
        query_emb=query_emb,
        doc_emb_cpu=doc_emb_cpu,
        doc_ids=doc_ids,
        query_ids=query_ids,
        excluded_ids=excluded_ids,
        chunk_size=chunk_size,
        topk=topk,
        device=device
    )


# --------------------------
# Registry (ONLY requested models)
# --------------------------
MULTIMODAL_PAIR_RETRIEVAL_FUNCS = {
    "clip": retrieval_clip_it2it,
    "siglip": retrieval_siglip_it2it,
    "jina-clip": retrieval_jina_clip_it2it,
    "bge-vl-large": retrieval_bge_vl_large_it2it,
    "gme-qwen2-vl-2b": retrieval_gme_qwen2_vl_it2it,
    "gme-qwen2-vl-7b": retrieval_gme_qwen2_vl_it2it,
    "nomic-vision": retrieval_nomic_it2it,
}
