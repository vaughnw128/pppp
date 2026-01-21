from __future__ import annotations

# built-in
from collections.abc import Iterable
from io import BytesIO

# external
from PIL import Image, ImageSequence


def is_gif(image_bytes: bytes, *, content_type: str | None) -> bool:
    """Is it, a gif?"""

    if content_type:
        ct = content_type.split(";", 1)[0].strip().lower()
        if ct == "image/gif":
            return True

    return image_bytes[:6] in {b"GIF87a", b"GIF89a"}


def iter_image_frames(image_bytes: bytes, *, content_type: str | None) -> Iterable[tuple[int, Image.Image]]:
    """Yield (frame_index, frame_image) for a still or animated image."""

    im = Image.open(BytesIO(image_bytes))

    if is_gif(image_bytes, content_type=content_type):
        for frame_index, frame in enumerate(ImageSequence.Iterator(im)):
            yield frame_index, frame.copy().convert("RGB")
        return

    yield 0, im.convert("RGB")
