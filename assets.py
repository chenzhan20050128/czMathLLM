# -*- coding: utf-8 -*-
"""模型资源管理工具。

该模块负责下载与缓存大模型权重，封装了镜像站点优先级、环境变量回退
等细节。利用 Hugging Face Hub 的 `snapshot_download` 可以在保留文件
结构的情况下增量更新模型，适合大文件的断点续传。
"""

# from __future__ import annotations: 同样是为了支持延迟解析类型注解。
# 在 Python 3.10+ 中，类型注解可以在定义前使用，此导入是为了向前兼容。
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# 从 huggingface_hub 库导入核心下载函数和特定的异常类型。
# huggingface_hub 是与 Hugging Face Hub 交互的官方 Python 库。
from huggingface_hub import snapshot_download
from huggingface_hub.errors import (
    HfHubHTTPError,             # 当发生 HTTP 错误时（如 404 Not Found, 500 Server Error）抛出。
    LocalEntryNotFoundError,    # 当尝试访问本地缓存中不存在的文件时抛出。
    RepositoryNotFoundError,    # 当在 Hub 上找不到指定的模型仓库时抛出。
)

# --- 常量与环境变量配置 ---

# 默认的模型存储根目录。
# 优先从环境变量 `MATH_LLM_MODELS` 读取。如果环境变量未设置，则默认为当前工作目录下的 "models" 文件夹。
# 这种模式（`os.environ.get(KEY, DEFAULT_VALUE)`）是配置应用程序的常见做法，提供了灵活性。
DEFAULT_MODELS_ROOT = Path(os.environ.get("MATH_LLM_MODELS", "models"))

# 定义默认的 Hugging Face 镜像站点。
# 使用国内镜像是为了加速模型的下载。
DEFAULT_PRIMARY_ENDPOINT = "https://hf-mirror.com"
DEFAULT_SECONDARY_ENDPOINT = "https://aliendao.cn"

# `os.environ.setdefault(KEY, VALUE)`:
#   - 如果环境变量 `KEY` 已经存在，则不做任何操作。
#   - 如果 `KEY` 不存在，则设置其值为 `VALUE`。
#   - 这是一种无侵入式地为环境提供默认值的方式，避免覆盖用户已有的配置。

# 设置主要的下载镜像地址。
os.environ.setdefault("MATH_LLM_PRIMARY_ENDPOINT", DEFAULT_PRIMARY_ENDPOINT)
# 设置备用的下载镜像地址。
os.environ.setdefault(
    "MATH_LLM_SECONDARY_ENDPOINT",
    DEFAULT_SECONDARY_ENDPOINT,
)
# `huggingface_hub` 库会读取 `HF_ENDPOINT` 环境变量作为其全局 API 端点。
# 这里将其默认设置为我们的主镜像，以引导所有 `huggingface_hub` 的流量，避免每次调用都需指定。
os.environ.setdefault("HF_ENDPOINT", DEFAULT_PRIMARY_ENDPOINT)
# 兼容旧版 huggingface_hub，有些版本可能使用 HF_API_URL。
os.environ.setdefault("HF_API_URL", DEFAULT_PRIMARY_ENDPOINT)

# 启用 `hf_transfer` 库进行高速下载。
# 这是一个由 Hugging Face 官方维护的库，使用 Rust 编写，通过多线程和优化的网络协议
# 可以显著提升大文件（尤其是模型权重）的下载速度。
# 将环境变量 `HF_HUB_ENABLE_HF_TRANSFER` 设置为 "1" 即可启用。
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


# --- 辅助函数 ---

def _strip_endpoint(endpoint: Optional[str]) -> Optional[str]:
    """安全地剥离镜像地址两端的空白字符，并统一去掉末尾的 `/`。

    这确保了 URL 格式的规范性，避免因多余的字符（如用户误输入的空格或斜杠）导致连接失败。
    例如 " https://hf-mirror.com/ " -> "https://hf-mirror.com"。
    """
    if not endpoint:
        return None
    endpoint = endpoint.strip() # 去除首尾空白
    if not endpoint:
        return None
    return endpoint.rstrip("/") # 去除末尾的斜杠


def _is_truthy(value: Optional[str]) -> bool:
    """将字符串解析为布尔值，用于解析环境变量中的开关。

    环境变量的值都是字符串，所以需要一个函数来判断 "1", "true", "yes", "on" 等
    字符串是否代表 "真"。这在处理布尔类型的配置时非常有用。
    所有其他值（包括 None 或空字符串）都被视为 "假"。
    """
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _candidate_endpoints() -> list[Optional[str]]:
    """根据环境变量给出一组候选的下载镜像列表，保证尝试顺序稳定。

    这个函数是实现“多镜像重试”和“回退到官方源”逻辑的核心。
    它按以下优先级顺序构建一个端点列表：
    1.  `MATH_LLM_PRIMARY_ENDPOINT` 或 `HF_PRIMARY_ENDPOINT` (主镜像)
    2.  `MATH_LLM_SECONDARY_ENDPOINT` 或 `HF_SECONDARY_ENDPOINT` (备用镜像)
    3.  全局的 `HF_ENDPOINT` (huggingface_hub 的通用配置)
    4.  官方 Hugging Face 源 `https://huggingface.co` (作为最终回退，除非被禁用)

    `None` 在列表中代表使用 `huggingface_hub` 的默认行为（即官方源）。
    通过 `if endpoint not in candidates:` 确保列表中的端点不重复，避免重复尝试。
    """
    candidates: list[Optional[str]] = []

    # 读取主镜像配置，支持项目特定和通用的环境变量
    env_primary = _strip_endpoint(
        os.environ.get("MATH_LLM_PRIMARY_ENDPOINT")
        or os.environ.get("HF_PRIMARY_ENDPOINT")
        or DEFAULT_PRIMARY_ENDPOINT
    )
    # 读取备用镜像配置
    env_secondary = _strip_endpoint(
        os.environ.get("MATH_LLM_SECONDARY_ENDPOINT")
        or os.environ.get("HF_SECONDARY_ENDPOINT")
        or DEFAULT_SECONDARY_ENDPOINT
    )
    # 读取全局 HF_ENDPOINT 配置
    env_global = _strip_endpoint(os.environ.get("HF_ENDPOINT"))

    # 检查是否禁用了到官方源的回退
    disable_fallback = _is_truthy(os.environ.get("MATH_LLM_DISABLE_HF_FALLBACK"))
    # 如果没有禁用，则将官方源作为最后的选项
    fallback = None if disable_fallback else _strip_endpoint("https://huggingface.co")

    # 按优先级将不重复的端点添加到候选列表
    for endpoint in (env_primary, env_secondary, env_global, fallback):
        if endpoint not in candidates:
            candidates.append(endpoint)

    return candidates


def _safe_dir_name(model_id: str) -> str:
    """将模型ID（如 "Qwen/Qwen3-4B"）转换为文件系统安全（filesystem-friendly）的目录名。

    文件系统通常不允许路径中包含斜杠 `/`。这个函数将斜杠替换为双下划线 `__`，
    这是一种常见且可读性较好的约定，用于在本地文件名中表示命名空间。
    例如 "Qwen/Qwen3-4B" -> "Qwen__Qwen3-4B"。
    """
    return model_id.replace("/", "__")


# --- 主函数 ---

def ensure_model(
    model_id: str,
    *, # Python 语法：星号 `*` 强制后面的参数必须以关键字形式传递，例如 `ensure_model(..., force=True)`
       # 这可以提高代码的可读性，避免因参数位置错误导致的 bug。
    local_path: Optional[str] = None,
    force: bool = False,
) -> Path:
    """确保本地存在指定模型，如果不存在则下载，并返回其本地路径。

    这是本模块的核心功能函数，封装了所有下载、缓存和重试逻辑。

    参数说明：
    - `model_id`: Hugging Face Hub 上的模型仓库ID，例如 "Qwen/Qwen3-4B-Thinking-2507"。
    - `local_path`: 如果提供了这个路径，函数将直接返回该路径，跳过所有下载逻辑。
                    这为用户提供了覆盖默认缓存位置的灵活性，例如使用一个已经下载好的模型副本。
    - `force`: 如果为 `True`，将强制重新下载模型，即使本地已经存在。这对于更新模型或修复损坏的缓存很有用。

    下载逻辑：
    1.  如果 `local_path` 已指定，直接将其转换为 Path 对象并返回。
    2.  根据 `model_id` 确定模型在本地的存储目录。
    3.  如果目录不为空且 `force=False`，则认为模型已存在，直接返回路径。
    4.  如果需要下载，则遍历 `_candidate_endpoints()` 生成的镜像列表。
    5.  对每个镜像，尝试使用 `snapshot_download` 下载。
        - `resume_download=True`: 支持断点续传，对于大模型下载至关重要。
        - `local_dir_use_symlinks=False`: 将文件直接下载到目标目录，而不是使用符号链接指向全局缓存。
                                         这使得模型目录自包含，易于移动和管理。
    6.  如果下载成功，立即中断循环。
    7.  如果下载失败（捕获网络或仓库相关的特定异常），则打印错误信息并尝试下一个镜像。
    8.  使用 `try...finally` 结构确保无论下载成功与否，`HF_ENDPOINT` 环境变量最终都会被恢复到原始状态，
        避免影响程序其他部分的网络请求。
    9.  如果遍历完所有镜像后仍然下载失败，则抛出一个 `RuntimeError`，并附上最后一次的错误信息，
        以便用户排查问题。
    """
    if local_path:
        return Path(local_path)

    # 构造模型在本地的存储路径，例如 "models/Qwen__Qwen3-4B-Thinking-2507"
    target_dir = DEFAULT_MODELS_ROOT / _safe_dir_name(model_id)
    # 确保目录存在，如果不存在则创建
    target_dir.mkdir(parents=True, exist_ok=True)

    # `any(target_dir.iterdir())` 检查目录是否包含任何文件或子目录。
    # 如果目录不为空且不强制重新下载，则认为模型已缓存，直接返回路径。
    if force or not any(target_dir.iterdir()):
        print(f"[czMathLLM] 模型 '{model_id}' 不存在或被强制更新，开始下载...")
        # 准备 `snapshot_download` 函数的参数
        download_kwargs = {
            "repo_id": model_id,
            "local_dir": str(target_dir),
            "local_dir_use_symlinks": False, # 推荐设置为 False，避免复杂的缓存链接管理
            "resume_download": True,         # 启用断点续传
            "token": os.environ.get("HF_TOKEN"), # 从环境变量读取 Hugging Face API token，用于访问私有仓库
        }

        last_error: Exception | None = None
        original_hf_endpoint = os.environ.get("HF_ENDPOINT") # 保存原始的 HF_ENDPOINT
        downloaded = False
        try:
            # 遍历所有候选镜像进行尝试
            for endpoint in _candidate_endpoints():
                try:
                    # 动态设置当前尝试的镜像地址
                    if endpoint:
                        os.environ["HF_ENDPOINT"] = endpoint
                        download_kwargs["endpoint"] = endpoint
                    else: # 如果 endpoint 是 None，表示使用官方源
                        os.environ.pop("HF_ENDPOINT", None)
                        download_kwargs.pop("endpoint", None)

                    mirror_label = endpoint or "https://huggingface.co (官方)"
                    print(f"[czMathLLM] 正在尝试从 {mirror_label} 下载...")

                    # 调用 huggingface_hub 的核心下载函数
                    snapshot_download(**download_kwargs)

                    downloaded = True
                    print(f"[czMathLLM] 从 {mirror_label} 下载成功。模型已存放在: {target_dir}")
                    break # 下载成功，跳出循环
                except (
                    HfHubHTTPError,
                    LocalEntryNotFoundError,
                    RepositoryNotFoundError,
                ) as exc:  # pragma: no cover - network dependent, 忽略测试覆盖率检查
                    # 捕获预期的下载异常
                    print(
                        f"[czMathLLM] 从 {mirror_label} 下载失败: {exc.__class__.__name__}: {exc}"
                    )
                    last_error = exc # 记录最后一次的错误
                    continue # 继续尝试下一个镜像
        finally:
            # 无论成功、失败或异常，都确保恢复原始的 HF_ENDPOINT 环境变量
            if original_hf_endpoint is not None:
                os.environ["HF_ENDPOINT"] = original_hf_endpoint
            else:
                os.environ.pop("HF_ENDPOINT", None)

        # 如果所有镜像都尝试失败
        if not downloaded and last_error:
            # 抛出一个运行时错误，并把底层的异常作为原因（`from last_error`）
            # 这样可以保留完整的错误堆栈信息，便于调试。
            raise RuntimeError(
                f"从所有配置的镜像下载模型 '{model_id}' 失败。请检查您的网络连接、镜像可用性以及 HF_TOKEN 权限。"
            ) from last_error

    return target_dir
