"""模型资源管理工具。

该模块负责下载与缓存大模型权重，封装了镜像站点优先级、环境变量回退
等细节。利用 Hugging Face Hub 的 `snapshot_download` 可以在保留文件
结构的情况下增量更新模型，适合大文件的断点续传。"""

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

# 默认存储目录支持通过环境变量覆盖；否则回退到项目根目录下的 `models/`。
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
    """安全地剥离镜像地址两端的空白字符，并统一去掉末尾的 `/`。"""
    if not endpoint:
        return None
    endpoint = endpoint.strip()
    if not endpoint:
        return None
    return endpoint.rstrip("/")


def _is_truthy(value: Optional[str]) -> bool:
    """将字符串解析为布尔值，用于解析环境变量型开关。"""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _candidate_endpoints() -> list[Optional[str]]:
    """根据环境变量给出候选的下载镜像列表，保证尝试顺序稳定。"""
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
    """将类似 ``Qwen/Qwen3`` 的模型标识映射为文件系统友好的目录名。"""
    return model_id.replace("/", "__")


def ensure_model(
    model_id: str,
    *,
    local_path: Optional[str] = None,
    force: bool = False,
) -> Path:
    """确保本地存在指定模型并返回路径。

    参数说明：
    - ``model_id``：Hugging Face 仓库名或组织/仓库组合；
    - ``local_path``：若提供则直接使用自定义路径；
    - ``force``：为 ``True`` 时强制重新下载。

    下载逻辑充分考虑了网络环境，按候选镜像顺序逐一尝试，
    并在全部失败后抛出详细异常提示。"""
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
