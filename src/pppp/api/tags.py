from __future__ import annotations

# built-in
import time
from typing import Any

# external
from fastapi import APIRouter, Body, HTTPException

# project
from pppp.api.image_io import decode_image_b64, detect_mime_type, fetch_image
from pppp.api.models import OcrB64Request, OcrUrlRequest, TagsResponse
from pppp.engine.rampp import tag_bytes
from pppp.settings import settings

router = APIRouter(tags=["tags"])


@router.post("/tags/bytes", response_model=TagsResponse)
async def tags_bytes_endpoint(
    image: bytes = Body(..., description="Raw image bytes"),
    top_k: int = 50,
) -> dict[str, Any]:
    if not image:
        raise HTTPException(status_code=400, detail="empty body")
    if len(image) > settings.max_image_bytes:
        raise HTTPException(status_code=413, detail="image too large")

    ct = detect_mime_type(image)

    start = time.perf_counter()
    result = tag_bytes(image, content_type=ct, top_k=top_k)
    elapsed_ms_total = int((time.perf_counter() - start) * 1000)

    return {
        "tags": result.tags,
        "engine": result.engine,
        "timings_ms": {"tagging": result.elapsed_ms, "total": elapsed_ms_total},
    }


@router.post("/tags/url", response_model=TagsResponse)
async def tags_url_endpoint(payload: OcrUrlRequest, top_k: int = 50) -> dict[str, Any]:
    image_bytes = await fetch_image(payload.image_url)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image_url")

    ct = detect_mime_type(image_bytes)

    start = time.perf_counter()
    result = tag_bytes(image_bytes, content_type=ct, top_k=top_k)
    elapsed_ms_total = int((time.perf_counter() - start) * 1000)

    return {
        "tags": result.tags,
        "engine": result.engine,
        "timings_ms": {"tagging": result.elapsed_ms, "total": elapsed_ms_total},
    }


@router.post("/tags/b64", response_model=TagsResponse)
async def tags_b64_endpoint(payload: OcrB64Request, top_k: int = 50) -> dict[str, Any]:
    image_bytes = decode_image_b64(payload.image_b64)

    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image_b64")
    if len(image_bytes) > settings.max_image_bytes:
        raise HTTPException(status_code=413, detail="image too large")

    ct = detect_mime_type(image_bytes)

    start = time.perf_counter()
    result = tag_bytes(image_bytes, content_type=ct, top_k=top_k)
    elapsed_ms_total = int((time.perf_counter() - start) * 1000)

    return {
        "tags": result.tags,
        "engine": result.engine,
        "timings_ms": {"tagging": result.elapsed_ms, "total": elapsed_ms_total},
    }
