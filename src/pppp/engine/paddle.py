from __future__ import annotations

# built-in
import difflib
import os
import tempfile
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np

# external
from paddleocr import PaddleOCR

# project
from pppp.settings import settings
from pppp.utils.images import is_gif, iter_image_frames

_lock = Lock()


_ocr: PaddleOCR | None = None


def get_ocr():
    """Get the OCR engine singleton, downloading on first init."""

    global _ocr
    if _ocr is not None:
        return _ocr

    with _lock:
        if _ocr is not None:
            return _ocr

        _ocr = PaddleOCR(
            lang=settings.paddle_lang,
            use_gpu=settings.paddle_use_gpu,
            use_angle_cls=settings.paddle_use_angle_cls,
            enable_mkldnn=settings.paddle_enable_mkldnn,
        )
        return _ocr


def _suffix_for_content_type(content_type: str | None) -> str:
    """Get a suitable file suffix for the given content type."""

    if not content_type:
        return ".img"

    ct = content_type.split(";", 1)[0].strip().lower()
    if ct == "image/png":
        return ".png"
    if ct in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    if ct == "image/gif":
        return ".gif"
    if ct == "image/webp":
        return ".webp"
    if ct == "image/tiff":
        return ".tiff"
    if ct == "image/bmp":
        return ".bmp"
    return ".img"


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float | None
    lines: list[dict[str, Any]]
    elapsed_ms: int


def _normalize_text_for_compare(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _token_jaccard(a: str, b: str) -> float:
    a_tokens = {t for t in a.split(" ") if t}
    b_tokens = {t for t in b.split(" ") if t}
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return inter / union if union else 0.0


def _text_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return max(difflib.SequenceMatcher(None, a, b).ratio(), _token_jaccard(a, b))


def _parse_paddleocr_raw(
    raw: Any,
    *,
    min_line_confidence: float | None,
) -> tuple[str, float | None, list[dict[str, Any]]]:
    """Parse the raw output from PaddleOCR into structured data."""

    items = raw[0] if raw and isinstance(raw[0], list) else raw

    lines: list[dict[str, Any]] = []
    texts: list[str] = []
    scores: list[float] = []

    for item in items or []:
        if not item or len(item) < 2:
            continue
        box, payload = item[0], item[1]
        if not payload or len(payload) < 2:
            continue
        line_text = str(payload[0])
        try:
            line_score = float(payload[1])
        except Exception:
            line_score = None

        if min_line_confidence is not None and line_score is not None:
            if line_score < min_line_confidence:
                continue

        if min_line_confidence is not None and line_score is None:
            pass

        texts.append(line_text)
        if line_score is not None:
            scores.append(line_score)
        lines.append({"text": line_text, "confidence": line_score, "box": box})

    text = "\n".join(t for t in texts if t)
    confidence = (sum(scores) / len(scores)) if scores else None
    return text, confidence, lines


def ocr_bytes(image_bytes: bytes, *, content_type: str | None) -> OcrResult:
    """Run OCR on the given image bytes."""

    ocr = get_ocr()
    start = time.perf_counter()

    # On gifs we do frame by frame processing to get all text
    if is_gif(image_bytes, content_type=content_type):
        texts: list[str] = []
        confidences: list[float] = []
        lines_all: list[dict[str, Any]] = []

        last_added_norm = ""
        saw_any_frame = False

        for frame_index, rgb in iter_image_frames(image_bytes, content_type=content_type):
            saw_any_frame = True

            np_bgr = np.asarray(rgb)[:, :, ::-1].copy()

            raw = ocr.ocr(np_bgr, cls=settings.paddle_use_angle_cls)
            frame_text, frame_conf, frame_lines = _parse_paddleocr_raw(
                raw,
                min_line_confidence=settings.paddle_min_line_confidence,
            )

            frame_norm = _normalize_text_for_compare(frame_text)
            if not frame_norm:
                continue

            # fuzzy dedup
            if last_added_norm:
                sim = _text_similarity(frame_norm, last_added_norm)
                thresh = 0.97 if min(len(frame_norm), len(last_added_norm)) < 15 else 0.90
                if sim >= thresh:
                    continue

            if frame_norm:
                texts.append(frame_text)
                last_added_norm = frame_norm
                if frame_conf is not None:
                    confidences.append(frame_conf)
                for line in frame_lines:
                    lines_all.append({**line, "frame": frame_index})

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        if not saw_any_frame:
            return OcrResult(text="", confidence=None, lines=[], elapsed_ms=elapsed_ms)

        text = "\n".join(t for t in texts if t)
        confidence = (sum(confidences) / len(confidences)) if confidences else None
        return OcrResult(text=text, confidence=confidence, lines=lines_all, elapsed_ms=elapsed_ms)

    suffix = _suffix_for_content_type(content_type)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            tmp_path = f.name
            f.write(image_bytes)

        raw = ocr.ocr(tmp_path, cls=settings.paddle_use_angle_cls)

        text, confidence, lines = _parse_paddleocr_raw(
            raw,
            min_line_confidence=settings.paddle_min_line_confidence,
        )
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        return OcrResult(text=text, confidence=confidence, lines=lines, elapsed_ms=elapsed_ms)

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
