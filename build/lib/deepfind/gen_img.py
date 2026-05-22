from __future__ import annotations

import base64
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image-preview"
DEFAULT_IMAGE_DIR = "tmp"
DEFAULT_IMAGE_ASPECT_RATIO = "1:1"
DEFAULT_IMAGE_SIZE = "2K"
SUPPORTED_ASPECT_RATIOS = {
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
}
SUPPORTED_IMAGE_SIZES = {"1K", "2K", "4K"}
REPO_ROOT = Path(__file__).resolve().parent.parent
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class ImageGenerationError(RuntimeError):
    """Base error for Nano Banana image generation failures."""


class MissingImageApiKeyError(ImageGenerationError):
    """Raised when no Nano Banana API key is configured."""


def resolve_image_root(image_dir: str | None) -> Path:
    raw = (image_dir or DEFAULT_IMAGE_DIR).strip() or DEFAULT_IMAGE_DIR
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _normalize_prompt(prompt: str) -> str:
    text = prompt.strip()
    if not text:
        raise ImageGenerationError("prompt cannot be empty.")
    return text


def _normalize_aspect_ratio(value: str | None) -> str:
    aspect_ratio = (value or DEFAULT_IMAGE_ASPECT_RATIO).strip() or DEFAULT_IMAGE_ASPECT_RATIO
    if aspect_ratio not in SUPPORTED_ASPECT_RATIOS:
        raise ImageGenerationError(
            f"Unsupported aspect_ratio '{aspect_ratio}'. Use one of: {', '.join(sorted(SUPPORTED_ASPECT_RATIOS))}."
        )
    return aspect_ratio


def _normalize_image_size(value: str | None) -> str:
    image_size = (value or DEFAULT_IMAGE_SIZE).strip().upper() or DEFAULT_IMAGE_SIZE
    if image_size not in SUPPORTED_IMAGE_SIZES:
        raise ImageGenerationError(
            f"Unsupported image_size '{image_size}'. Use one of: {', '.join(sorted(SUPPORTED_IMAGE_SIZES))}."
        )
    return image_size


def _mime_extension(mime_type: str) -> str:
    mapping = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    return mapping.get(mime_type.lower(), ".bin")


def _file_name(prompt: str, mime_type: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", prompt).strip("-").lower()[:24] or "image"
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{slug}-{digest}{_mime_extension(mime_type)}"


def _error_message(payload: Any, fallback: str) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
    return fallback


def _candidate_parts(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if isinstance(parts, list):
            return [part for part in parts if isinstance(part, dict)]
    return []


def _part_text(part: dict[str, Any]) -> str:
    text = part.get("text")
    return str(text).strip() if text else ""


def _part_inline_data(part: dict[str, Any]) -> dict[str, Any] | None:
    inline_data = part.get("inlineData")
    if isinstance(inline_data, dict):
        return inline_data
    inline_data = part.get("inline_data")
    if isinstance(inline_data, dict):
        return inline_data
    return None


def generate_image(
    prompt: str,
    *,
    api_key: str | None,
    model: str = DEFAULT_IMAGE_MODEL,
    image_dir: str | None = None,
    aspect_ratio: str = DEFAULT_IMAGE_ASPECT_RATIO,
    image_size: str = DEFAULT_IMAGE_SIZE,
    timeout: int = 90,
) -> dict[str, str]:
    prompt = _normalize_prompt(prompt)
    resolved_key = (api_key or "").strip()
    if not resolved_key:
        raise MissingImageApiKeyError(
            "Set GOOGLE_NANO_BANANA_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY to use gen_img."
        )
    aspect_ratio = _normalize_aspect_ratio(aspect_ratio)
    image_size = _normalize_image_size(image_size)

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "imageConfig": {
                "aspectRatio": aspect_ratio,
                "imageSize": image_size,
            }
        },
    }
    model_name = model.strip() or DEFAULT_IMAGE_MODEL
    url = f"{API_BASE_URL}/{urllib.parse.quote(model_name, safe='')}:generateContent"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": resolved_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=max(timeout, 1)) as response:
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = None
        raise ImageGenerationError(
            _error_message(payload, f"Nano Banana request failed with status {exc.code}.")
        ) from exc
    except urllib.error.URLError as exc:
        raise ImageGenerationError(f"Nano Banana request failed: {exc.reason}") from exc

    try:
        response_payload = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise ImageGenerationError("Nano Banana API returned invalid JSON.") from exc

    text_parts: list[str] = []
    image_bytes: bytes | None = None
    mime_type = "image/png"
    for part in _candidate_parts(response_payload):
        text = _part_text(part)
        if text:
            text_parts.append(text)

        inline_data = _part_inline_data(part)
        if not inline_data or image_bytes is not None:
            continue
        encoded = inline_data.get("data")
        part_mime = inline_data.get("mimeType") or inline_data.get("mime_type")
        if not isinstance(encoded, str) or not encoded.strip():
            continue
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise ImageGenerationError("Nano Banana API returned invalid image data.") from exc
        if isinstance(part_mime, str) and part_mime.strip():
            mime_type = part_mime.strip()

    if image_bytes is None:
        raise ImageGenerationError(
            _error_message(response_payload, "Nano Banana API returned no image data.")
        )

    image_root = resolve_image_root(image_dir)
    image_root.mkdir(parents=True, exist_ok=True)
    image_path = image_root / _file_name(prompt, mime_type)
    image_path.write_bytes(image_bytes)

    return {
        "prompt": prompt,
        "model": model_name,
        "aspect_ratio": aspect_ratio,
        "image_size": image_size,
        "mime_type": mime_type,
        "image_path": str(image_path),
        "response_text": "\n".join(text_parts).strip(),
    }


__all__ = [
    "DEFAULT_IMAGE_ASPECT_RATIO",
    "DEFAULT_IMAGE_DIR",
    "DEFAULT_IMAGE_MODEL",
    "DEFAULT_IMAGE_SIZE",
    "ImageGenerationError",
    "MissingImageApiKeyError",
    "generate_image",
    "resolve_image_root",
]
