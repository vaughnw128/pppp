from __future__ import annotations

# built-in
from contextlib import asynccontextmanager

# external
import torch
from fastapi import FastAPI

# project
from pppp.api.ocr import router as ocr_router
from pppp.api.tags import router as tags_router
from pppp.engine.paddle import get_ocr
from pppp.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.warmup_on_start:
        get_ocr()

    yield


app = FastAPI(lifespan=lifespan, title="pppp", version="0.1.0")


@app.get("/")
async def root():
    return {"message": "Nothing here, teehehheheheheheh!"}


@app.get("/health")
async def health():
    return {"status": "ok"}


app.include_router(ocr_router)
app.include_router(tags_router)
