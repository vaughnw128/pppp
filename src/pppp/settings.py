from __future__ import annotations

# external
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PPPP_", case_sensitive=False)
    paddle_lang: str = "en"
    paddle_use_gpu: bool = False
    paddle_use_angle_cls: bool = True
    paddle_enable_mkldnn: bool = True
    paddle_min_line_confidence: float = 0.7
    warmup_on_start: bool = True
    max_image_bytes: int = 25 * 1024 * 1024
    fetch_timeout_s: int = 20

    allowed_mime_types: list[str] = [
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
        "image/tiff",
        "image/bmp",
    ]

    # allowed urls
    image_url_host_regex: str = (
        r"^(localhost|127\.0\.0\.1|::1|.+\.amazonaws\.com|.+\.amazonaws\.com|.+\.cloudfront\.net)$"
    )

    # recognize-anything settings
    rampp_checkpoint: str = (
        "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth"
    )
    rampp_image_size: int = 384
    rampp_vit: str = "swin_l"
    rampp_use_gpu: bool = False


settings = Settings()
