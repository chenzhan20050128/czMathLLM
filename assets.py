from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from huggingface_hub.errors import (
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)

DEFAULT_MODELS_ROOT = Path(os.environ.get("MATH_LLM_MODELS", "models"))

DEFAULT_PRIMARY_ENDPOINT = "https://hf-mirror.com"
DEFAULT_SECONDARY_ENDPOINT = "https://aliendao.cn"

os.environ.setdefault("MATH_LLM_PRIMARY_ENDPOINT", DEFAULT_PRIMARY_ENDPOINT)
os.environ.setdefault(
    "MATH_LLM_SECONDARY_ENDPOINT",
    DEFAULT_SECONDARY_ENDPOINT,
)
os.environ.setdefault("HF_ENDPOINT", DEFAULT_PRIMARY_ENDPOINT)
os.environ.setdefault("HF_API_URL", DEFAULT_PRIMARY_ENDPOINT)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _strip_endpoint(endpoint: Optional[str]) -> Optional[str]:
    if not endpoint:
        return None
    endpoint = endpoint.strip()
    if not endpoint:
        return None
    return endpoint.rstrip("/")


def _is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _candidate_endpoints() -> list[Optional[str]]:
    candidates: list[Optional[str]] = []

    env_primary = _strip_endpoint(
        os.environ.get("MATH_LLM_PRIMARY_ENDPOINT")
        or os.environ.get("HF_PRIMARY_ENDPOINT")
        or DEFAULT_PRIMARY_ENDPOINT
    )
    env_secondary = _strip_endpoint(
        os.environ.get("MATH_LLM_SECONDARY_ENDPOINT")
        or os.environ.get("HF_SECONDARY_ENDPOINT")
        or DEFAULT_SECONDARY_ENDPOINT
    )
    env_global = _strip_endpoint(os.environ.get("HF_ENDPOINT"))

    disable_fallback = _is_truthy(os.environ.get("MATH_LLM_DISABLE_HF_FALLBACK"))
    fallback = None if disable_fallback else _strip_endpoint("https://huggingface.co")

    for endpoint in (env_primary, env_secondary, env_global, fallback):
        if endpoint not in candidates:
            candidates.append(endpoint)

    return candidates


def _safe_dir_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def ensure_model(
    model_id: str,
    *,
    local_path: Optional[str] = None,
    force: bool = False,
) -> Path:
    if local_path:
        return Path(local_path)

    target_dir = DEFAULT_MODELS_ROOT / _safe_dir_name(model_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    if force or not any(target_dir.iterdir()):
        download_kwargs = {
            "repo_id": model_id,
            "local_dir": str(target_dir),
            "local_dir_use_symlinks": False,
            "resume_download": True,
            "token": os.environ.get("HF_TOKEN"),
        }

        last_error: Exception | None = None
        original_hf_endpoint = os.environ.get("HF_ENDPOINT")
        downloaded = False
        try:
            for endpoint in _candidate_endpoints():
                try:
                    if endpoint:
                        os.environ["HF_ENDPOINT"] = endpoint
                        download_kwargs["endpoint"] = endpoint
                    else:
                        os.environ.pop("HF_ENDPOINT", None)
                        download_kwargs.pop("endpoint", None)
                    mirror_label = endpoint or "https://huggingface.co"
                    print("[czMathLLM] Attempting download from", mirror_label)
                    snapshot_download(**download_kwargs)
                    downloaded = True
                    break
                except (
                    HfHubHTTPError,
                    LocalEntryNotFoundError,
                    RepositoryNotFoundError,
                ) as exc:  # pragma: no cover - network dependent
                    print(
                        "[czMathLLM] Download from"
                        f" {mirror_label} failed: {exc.__class__.__name__}:"
                        f" {exc}"
                    )
                    last_error = exc
                    continue
        finally:
            if original_hf_endpoint is not None:
                os.environ["HF_ENDPOINT"] = original_hf_endpoint
            else:
                os.environ.pop("HF_ENDPOINT", None)

        if not downloaded and last_error:
            raise RuntimeError(
                "Failed to download model from all configured endpoints. "
                "Please verify network connectivity, mirror availability, "
                "and HF_TOKEN permissions."
            ) from last_error
    return target_dir
