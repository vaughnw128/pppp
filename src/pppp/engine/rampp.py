from __future__ import annotations

import time

# built-in
from dataclasses import dataclass
from threading import Lock

# external
import transformers.modeling_utils as _mu
import transformers.pytorch_utils as _pu

# patch for transformers
for name in (
    "apply_chunking_to_forward",
    "find_pruneable_heads_and_indices",
    "prune_linear_layer",
):
    if not hasattr(_mu, name) and hasattr(_pu, name):
        setattr(_mu, name, getattr(_pu, name))

import torch
from ram import get_transform
from ram import inference_ram as inference
from ram.models import ram_plus

# project
from pppp.settings import settings
from pppp.utils.images import iter_image_frames

_lock = Lock()
_model = None
_transform = None


@dataclass(frozen=True)
class TagsResult:
    tags: list[str]
    engine: str
    elapsed_ms: int


def _get_model_and_transform():
    """Singleton for fucked up ram library because it has version conflicts."""

    global _model, _transform

    if _model is not None and _transform is not None:
        return _model, _transform

    with _lock:
        if _model is not None and _transform is not None:
            return _model, _transform

        device = None
        if settings.rampp_use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        _transform = get_transform(image_size=settings.rampp_image_size)

        model = ram_plus(
            pretrained=settings.rampp_checkpoint,
            image_size=settings.rampp_image_size,
            vit=settings.rampp_vit,
        )
        model.eval()
        model = model.to(device)

        _model = model
        return _model, _transform


def tag_bytes(
    image_bytes: bytes,
    *,
    content_type: str | None,
    top_k: int = 50,
) -> TagsResult:
    """Generate tags using RAM++."""

    start = time.perf_counter()

    model, transform = _get_model_and_transform()
    device = next(model.parameters()).device

    tags_set: dict[str, None] = {}

    for _frame_index, rgb in iter_image_frames(image_bytes, content_type=content_type):
        # normalize on each
        image = transform(rgb).unsqueeze(0).to(device)
        tags_str, _tags_zh = inference(image, model)
        for t in (tags_str or "").split("|"):
            t = t.strip()
            if t:
                tags_set[t] = None

    tags = list(tags_set.keys())
    if top_k > 0:
        tags = tags[:top_k]

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return TagsResult(tags=tags, engine="ram++", elapsed_ms=elapsed_ms)
