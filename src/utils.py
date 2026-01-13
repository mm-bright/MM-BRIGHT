
import re
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import tiktoken

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def norm_id(x: str) -> str:
    """Canonicalize IDs to always use forward slashes and no leading './'"""
    return Path(str(x)).as_posix().lstrip("./")

def is_readable_image(p: Path) -> bool:
    """Check if an image file can be opened and loaded by PIL."""
    try:
        with Image.open(p) as im:
            im.load()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False

def base_key_from_passage_id(passage_id: str) -> str:
    """
    Extract base key from passage ID.
    Example: "academia/a7beca61_6123.txt" -> "a7beca61"
    """
    m = re.search(r"/([0-9a-f]+)_\d+\.txt$", passage_id)
    return m.group(1) if m else passage_id

def base_key_from_image_rel(image_rel: str, domain: str) -> str:
    """
    Extract base key from image relative path.
    Example: "academia/academia_a7beca61_0002/image_1.webp" -> "a7beca61"
    """
    # Includes domain prefix matching to be strict
    m = re.search(rf"{re.escape(domain)}_([0-9a-f]+)_\d+", image_rel)
    return m.group(1) if m else ""

def cut_text(text, tokenizer, threshold):
    """Truncate text to threshold tokens."""
    text_ids = tokenizer(text)['input_ids']
    if len(text_ids) > threshold:
        text = tokenizer.decode(text_ids[:threshold])
    return text

def cut_text_openai(text, tokenizer, threshold=6000):
    """Truncate text for OpenAI models."""
    token_ids = tokenizer.encode(text, disallowed_special=())
    if len(token_ids) > threshold:
        text = tokenizer.decode(token_ids[:threshold])
    return text

def add_instruct_concatenate(texts, task, instruction):
    """Concatenate instructions to texts."""
    return [instruction.format(task=task, query=t) for t in texts]

def last_token_pool(last_hidden_states, attention_mask):
    """Pooling to get the embedding of the last token (EOS)."""
    import torch
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def ensure_blank(path_dir, size=(224, 224)):
    """Ensure a blank image exists and return its path."""
    import os
    path_dir = Path(path_dir)
    path_dir.mkdir(parents=True, exist_ok=True)
    p = path_dir / "blank.png"
    if not p.exists():
        img = Image.new("RGB", size, color="white")
        img.save(p)
    return str(p)

def safe_image_path(p, blank_path):
    """Return p if valid image, else blank_path."""
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
