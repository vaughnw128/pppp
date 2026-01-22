from __future__ import annotations

# built-in
import time

# external
from fastapi import APIRouter, Body, HTTPException

# project
from pppp.api.image_io import decode_image_b64, detect_mime_type, fetch_image
from pppp.api.models import OcrB64Request, OcrResponse, OcrUrlRequest
from pppp.engine.paddle import ocr_bytes
from pppp.settings import settings

router = APIRouter(tags=["ocr"])


@router.post("/ocr/bytes", response_model=OcrResponse)
async def ocr_bytes_endpoint(
    image: bytes = Body(..., description="Raw image bytes"),
    verbose: bool = False,
) -> OcrResponse:
    if not image:
        raise HTTPException(status_code=400, detail="empty body")
    if len(image) > settings.max_image_bytes:
        raise HTTPException(status_code=413, detail="image too large")

    ct = detect_mime_type(image)

    start = time.perf_counter()
    result = ocr_bytes(image, content_type=ct)
    elapsed_ms_total = int((time.perf_counter() - start) * 1000)

    return OcrResponse(
        text=result.text,
        engine="paddleocr",
        confidence=result.confidence,
        timings_ms={"ocr": result.elapsed_ms, "total": elapsed_ms_total},
        lines=(result.lines if verbose else None),
    )


@router.post("/ocr/url", response_model=OcrResponse)
async def ocr_url_endpoint(payload: OcrUrlRequest, verbose: bool = False) -> OcrResponse:
    image_bytes = await fetch_image(payload.image_url)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image_url")

    ct = detect_mime_type(image_bytes)

    start = time.perf_counter()
    result = ocr_bytes(image_bytes, content_type=ct)
    elapsed_ms_total = int((time.perf_counter() - start) * 1000)

    return OcrResponse(
        text=result.text,
        engine="paddleocr",
        confidence=result.confidence,
        timings_ms={"ocr": result.elapsed_ms, "total": elapsed_ms_total},
        lines=(result.lines if verbose else None),
    )


@router.post("/ocr/b64", response_model=OcrResponse)
async def ocr_b64_endpoint(payload: OcrB64Request, verbose: bool = False) -> OcrResponse:
    image_bytes = decode_image_b64(payload.image_b64)

    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image_b64")
    if len(image_bytes) > settings.max_image_bytes:
        raise HTTPException(status_code=413, detail="image too large")

    ct = detect_mime_type(image_bytes)

    start = time.perf_counter()
    result = ocr_bytes(image_bytes, content_type=ct)
    elapsed_ms_total = int((time.perf_counter() - start) * 1000)

    return OcrResponse(
        text=result.text,
        engine="paddleocr",
        confidence=result.confidence,
        timings_ms={"ocr": result.elapsed_ms, "total": elapsed_ms_total},
        lines=(result.lines if verbose else None),
    )
