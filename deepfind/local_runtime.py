from __future__ import annotations

from dataclasses import dataclass
import subprocess
from urllib.parse import urlsplit, urlunsplit

import httpx

from .config import Settings


@dataclass(frozen=True)
class GpuStatus:
    available: bool
    name: str = ""
    memory_total_mb: int | None = None


@dataclass(frozen=True)
class LocalModelStatus:
    available: bool
    model: str
    base_url: str
    backend: str = "ollama"
    gpu: GpuStatus = GpuStatus(False)
    reason: str = ""


def detect_gpu() -> GpuStatus:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return GpuStatus(available=False)

    first_line = next((line.strip() for line in completed.stdout.splitlines() if line.strip()), "")
    if not first_line:
        return GpuStatus(available=False)

    parts = [item.strip() for item in first_line.split(",", 1)]
    if len(parts) != 2:
        return GpuStatus(available=False)

    name, raw_memory = parts
    try:
        memory_total_mb = int(raw_memory)
    except ValueError:
        memory_total_mb = None

    return GpuStatus(
        available=True,
        name=name,
        memory_total_mb=memory_total_mb,
    )


def ollama_tags_url(base_url: str) -> str:
    parsed = urlsplit(base_url.strip())
    scheme = parsed.scheme or "http"
    netloc = parsed.netloc or parsed.path
    path = parsed.path if parsed.netloc else ""
    clean_path = path.rstrip("/")
    if clean_path.endswith("/v1"):
        clean_path = clean_path[:-3]
    if not clean_path:
        clean_path = ""
    return urlunsplit((scheme, netloc, f"{clean_path}/api/tags", "", ""))


def list_ollama_models(base_url: str) -> list[str]:
    response = httpx.get(ollama_tags_url(base_url), timeout=2.5)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("models")
    if not isinstance(models, list):
        return []

    names: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("model") or "").strip()
        if name:
            names.append(name)
    return names


def detect_local_model(settings: Settings) -> LocalModelStatus:
    gpu = detect_gpu()
    if not gpu.available:
        return LocalModelStatus(
            available=False,
            model=settings.local_model,
            base_url=settings.local_base_url,
            gpu=gpu,
            reason="No NVIDIA GPU was detected.",
        )

    if not settings.local_model.strip():
        return LocalModelStatus(
            available=False,
            model=settings.local_model,
            base_url=settings.local_base_url,
            gpu=gpu,
            reason="Set DEEPFIND_LOCAL_MODEL to an Ollama model name before enabling GPU mode.",
        )

    try:
        model_names = list_ollama_models(settings.local_base_url)
    except httpx.HTTPError:
        return LocalModelStatus(
            available=False,
            model=settings.local_model,
            base_url=settings.local_base_url,
            gpu=gpu,
            reason=f"Ollama is not reachable at {settings.local_base_url}.",
        )

    if settings.local_model not in model_names:
        return LocalModelStatus(
            available=False,
            model=settings.local_model,
            base_url=settings.local_base_url,
            gpu=gpu,
            reason=f"Model {settings.local_model} is not loaded in Ollama.",
        )

    return LocalModelStatus(
        available=True,
        model=settings.local_model,
        base_url=settings.local_base_url,
        gpu=gpu,
    )
