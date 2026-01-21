from __future__ import annotations

from pydantic import BaseModel, Field


class OcrUrlRequest(BaseModel):
    image_url: str = Field(..., description="Remote image URL (https; host must match allowlist)")


class OcrB64Request(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded image bytes (no data: URL prefix)")


class OcrResponse(BaseModel):
    text: str
    engine: str
    confidence: float | None = None
    timings_ms: dict[str, int] | None = None
    lines: list[dict] | None = None


class TagsResponse(BaseModel):
    tags: list[str]
    engine: str
    timings_ms: dict[str, int] | None = None
