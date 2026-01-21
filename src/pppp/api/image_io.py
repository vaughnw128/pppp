from __future__ import annotations

# built-in
import base64
import re
from urllib.parse import urlparse

# external
import httpx
from fastapi import HTTPException
from magika import Magika

# project
from pppp.settings import settings

_magika = Magika()


def detect_mime_type(image_bytes: bytes) -> str:
    try:
        res = _magika.identify_bytes(image_bytes)
        ct = (getattr(res.output, "mime_type", None) or "").strip().lower()
    except Exception:
        raise HTTPException(status_code=415, detail="unable to detect image mime type")

    if not ct or not isinstance(ct, str):
        raise HTTPException(status_code=415, detail="unable to detect image mime type")

    ct = ct.split(";", 1)[0].strip().lower()

    aliases = {
        "image/jpg": "image/jpeg",
        "image/x-ms-bmp": "image/bmp",
        "image/x-bmp": "image/bmp",
        "image/tif": "image/tiff",
    }
    ct = aliases.get(ct, ct)

    if ct not in settings.allowed_mime_types:
        raise HTTPException(status_code=415, detail=f"unsupported image type: {ct}")

    return ct


def decode_image_b64(image_b64: str) -> bytes:
    """Decode base64-encoded image data."""

    try:
        return base64.b64decode(image_b64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image_b64")


async def fetch_image(url: str) -> bytes:
    """Fetch image bytes from a remote URL."""

    def _validate_parsed(p, *, context: str) -> None:
        scheme = (p.scheme or "").lower()
        if scheme not in {"http", "https"}:
            raise HTTPException(status_code=400, detail=f"{context} must be http(s)")
        if not p.hostname:
            raise HTTPException(status_code=400, detail=f"{context} is invalid")

        hostname = p.hostname.lower().strip(".")
        is_local = hostname in {"localhost", "127.0.0.1", "::1"}

        if scheme == "http" and not is_local:
            raise HTTPException(
                status_code=400,
                detail=f"{context} must use https (http allowed for localhost)",
            )

        try:
            host_re = re.compile(settings.image_url_host_regex, re.IGNORECASE)
        except re.error:
            raise HTTPException(
                status_code=500,
                detail="server misconfigured: invalid image_url_host_regex",
            )

        if not host_re.fullmatch(hostname):
            raise HTTPException(status_code=400, detail=f"{context} host is not allowed")

    parsed = urlparse(url)
    _validate_parsed(parsed, context="image_url")

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=settings.fetch_timeout_s,
        ) as client, client.stream("GET", url) as resp:
            _validate_parsed(urlparse(str(resp.url)), context="image_url (final)")

            if resp.status_code < 200 or resp.status_code >= 300:
                raise HTTPException(status_code=400, detail=f"image_url returned {resp.status_code}")

            size = 0
            chunks: list[bytes] = []
            try:
                async for chunk in resp.aiter_bytes():
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > settings.max_image_bytes:
                        raise HTTPException(status_code=413, detail="image_url image too large")
                    chunks.append(chunk)
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(status_code=400, detail="failed to read image_url response")

            data = b"".join(chunks)
            if not data:
                raise HTTPException(status_code=400, detail="image_url returned empty body")

            return data
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="failed to fetch image_url")
