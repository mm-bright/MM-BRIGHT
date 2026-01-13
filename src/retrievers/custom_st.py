import math
import logging
from io import BytesIO
from typing import Any, Dict, Optional, List
import torch
from PIL import Image
from sentence_transformers.models import Transformer as BaseTransformer
from transformers import AutoModelForVision2Seq, AutoProcessor


class MultiModalTransformer(BaseTransformer):
    def __init__(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        tokenizer_args: Optional[Dict[str, Any]] = None,
        min_image_tokens: int = 256,
        max_image_tokens: int = 1280,
        max_length: int = 1800,
        **kwargs,
    ):
        super().__init__(model_name_or_path, **kwargs)
        if tokenizer_args is None:
            tokenizer_args = {}
        tokenizer_args.pop("trust_remote_code", None)

        # Initialize processor
        min_pixels = min_image_tokens * 28 * 28
        max_pixels = max_image_tokens * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path, min_pixels=min_pixels, max_pixels=max_pixels, **kwargs
        )
        self.processor.tokenizer.padding_side = 'right'
        self.sep = ' '
        self.max_length = max_length
        self.normalize = True

    def _load_model(
            self,
            model_name_or_path: str,
            config,
            cache_dir: str,
            backend: str,
            is_peft_model: bool,
            **model_args,
    ) -> None:
        model_args.pop("trust_remote_code", None)
        self.auto_model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path, torch_dtype=torch.float16, **model_args
        )

    def forward(
        self, features: Dict[str, torch.Tensor], **kwargs
    ) -> Dict[str, torch.Tensor]:       
        if features.get("inputs_embeds", None) is None:
            features["inputs_embeds"] = self.auto_model.base_model.get_input_embeddings()(features["input_ids"])
            if features.get("pixel_values", None) is not None:
                features["pixel_values"] = features["pixel_values"].type(self.auto_model.visual.get_dtype())
                image_embeds = self.auto_model.visual(
                    features["pixel_values"], grid_thw=features["image_grid_thw"]
                )
                image_mask = features["input_ids"] == self.auto_model.config.image_token_id
                features["inputs_embeds"][image_mask] = image_embeds
                # features.pop("pixel_values")
                # features.pop("image_grid_thw")
        # features.pop("input_ids")
        inputs = {k: v for k, v in features.items() if k in 'position_ids,attention_mask,inputs_embeds'}
        outputs = self.auto_model.model(
            **inputs,
            return_dict=True,
            output_hidden_states=True,
            # **kwargs
        )
        # pooling_mask = features["attention_mask"] if features.get("pooling_mask", None) is None else features["pooling_mask"]
        # left_padding = (pooling_mask[:, -1].sum() == pooling_mask.shape[0])  # TODO
        # if left_padding:
        #     embeddings = outputs.last_hidden_state
        # else:
        #     sequence_lengths = pooling_mask.sum(dim=1) - 1
        #     embeddings = outputs.last_hidden_state[torch.arange(
        #         outputs.last_hidden_state.shape[0], device=outputs.last_hidden_state.device
        #     ), sequence_lengths]
        features.update({"token_embeddings": outputs.last_hidden_state})
        return features 

    def tokenize(self, texts: List[List[Dict[str, Any]]] | List[str]) -> Dict[str, torch.Tensor]:
        default_instruction = 'You are a helpful assistant.'

        all_texts, all_images = list(), list()
        for item in texts:
            if isinstance(item, str):
                txt, img, inst = item, None, default_instruction
            elif isinstance(item, dict):
                txt = item.get('text', None)
                img = item.get('image', None)
                inst = item.get('prompt', default_instruction)
            else:
                raise RuntimeError(f'Input format not supported! {item=}')

            input_str = ''
            if img is None:
                all_images = None  # All examples in the same batch are consistent
                # or will have ValueError: Could not make a flat list of images from xxxx
            else:
                input_str += '<|vision_start|><|image_pad|><|vision_end|>'
                img = fetch_image(img)
                all_images.append(img)
            if txt is not None:
                input_str += txt
            msg = f'<|im_start|>system\n{inst}<|im_end|>\n<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>'
            all_texts.append(msg)

        inputs = self.processor(
            text=all_texts,
            images=all_images,
            padding="longest",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        return inputs


### Copied from qwen_vl_utils.vision_process.py
import base64
from io import BytesIO
import requests

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    if max(h_bar, w_bar) / min(h_bar, w_bar) > MAX_RATIO:
        logging.warning(
            f"Absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(h_bar, w_bar) / min(h_bar, w_bar)}"
        )
        if h_bar > w_bar:
            h_bar = w_bar * MAX_RATIO
        else:
            w_bar = h_bar * MAX_RATIO
    return h_bar, w_bar


def fetch_image(image: str | Image.Image, size_factor: int = IMAGE_FACTOR) -> Image.Image:
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")
    ## resize
    # if "resized_height" in ele and "resized_width" in ele:
    #     resized_height, resized_width = smart_resize(
    #         ele["resized_height"],
    #         ele["resized_width"],
    #         factor=size_factor,
    #     )
    # else:
    width, height = image.size
    # min_pixels = ele.get("min_pixels", MIN_PIXELS)
    # max_pixels = ele.get("max_pixels", MAX_PIXELS)
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    image = image.resize((resized_width, resized_height))

    return image
###
