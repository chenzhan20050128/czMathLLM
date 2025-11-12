"""FastAPI 对外服务接口。

该应用围绕 czMathLLM 提供的训练与推理能力，封装以下能力：

1. 模型加载：可自由指定 LoRA、GRPO 或合并后模型检查点，并组合多级适配器；
2. 模型发现：扫描输出目录，告知当前可用的检查点与模型类型；
3. 数学题推理：输入 Prompt，返回包含思考过程的答案，支持自定义采样参数；
4. 健康检查：展示服务器硬件/软件状态；
5. 管理员命令：校验密码后可执行任意 shell 指令，便于远程运维。

为了便于测试与扩展，我们将模型加载逻辑抽象为 `ModelManager`，并允许在
单元测试中注入自定义加载器或模拟模型，避免真正拉起大模型导致的资源消耗。
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, List, Literal, Optional

import torch
from fastapi import FastAPI, HTTPException, status  # type: ignore[import]
from pydantic import (  # type: ignore[import]
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from .config import TrainingConfig
from .modeling import (
    generate_answers,
    load_base_model,
    prepare_for_inference,
)

from peft import PeftModel
from torch import version as torch_version

APP_DESCRIPTION = """czMathLLM 推理服务，支持模型加载、推理、健康检查及管理命令。"""

app = FastAPI(
    title="czMathLLM API",
    description=APP_DESCRIPTION,
    version="0.1.0",
)

ADMIN_PASSWORD = "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"
DEFAULT_OUTPUT_ROOT = Path("outputs")
DEFAULT_MODEL_ROOT = Path("models")
SERVER_START_TS = time.time()


class LoadModelRequest(BaseModel):
    """模型加载请求体。"""

    model_config = ConfigDict(extra="forbid")

    checkpoint_name: Optional[str] = Field(
        default=None, description="可读名称，仅用于元数据展示"
    )
    base_model: Optional[str] = Field(
        default=None,
        description="基座模型路径或 Hugging Face 仓库 ID。"
        "若为空则沿用 TrainingConfig 中的默认值。",
    )
    merged_path: Optional[str] = Field(
        default=None,
        description="已合并完成的完整模型目录；若填写则忽略 LoRA/GRPO。",
    )
    lora_path: Optional[str] = Field(
        default=None, description="监督微调（LoRA）权重目录"
    )
    grpo_path: Optional[str] = Field(default=None, description="GRPO 强化学习权重目录")
    load_in_4bit: bool = Field(
        default=True, description="是否以 4bit 量化方式加载基座模型"
    )
    load_in_8bit: bool = Field(
        default=False, description="是否以 8bit 量化方式加载基座模型"
    )
    dtype: Optional[str] = Field(
        default=None, description="显式指定权重精度，例如 fp16、bf16"
    )
    max_seq_length: Optional[int] = Field(
        default=None, description="推理时的最大序列长度，默认沿用模型配置"
    )

    @model_validator(mode="after")
    def _validate_paths(self) -> "LoadModelRequest":
        if not (
            self.merged_path or self.base_model or self.lora_path or self.grpo_path
        ):
            raise ValueError(
                "需至少提供 merged_path，或提供 base_model/适配器路径之一。"
            )
        return self

    def adapter_paths(self) -> list[str]:
        paths: list[str] = []
        if self.lora_path:
            paths.append(self.lora_path)
        if self.grpo_path:
            paths.append(self.grpo_path)
        return paths


class ModelInfo(BaseModel):
    """模型或检查点的元信息。"""

    name: str
    path: str
    type: Literal["adapter", "merged", "checkpoint", "base"]
    updated_at: float
    size_bytes: Optional[int] = None


class ModelListResponse(BaseModel):
    items: List[ModelInfo]


class GenerateRequest(BaseModel):
    """推理请求体。"""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., description="用户输入的数学题或问题")
    temperature: float = Field(
        default=0.7, ge=0.0, description="采样温度，0 表示贪心解码"
    )
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="核采样概率")
    top_k: Optional[int] = Field(
        default=None, ge=1, description="Top-K 采样阈值，若为空则不启用"
    )
    repetition_penalty: float = Field(default=1.0, ge=0.0, description="重复惩罚系数")
    max_new_tokens: int = Field(
        default=512, ge=1, description="生成阶段允许的最大新增 tokens"
    )
    system_prompt: Optional[str] = Field(
        default=None, description="可选系统提示词，覆盖默认模板"
    )


class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    latency_ms: float
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    uptime_seconds: float
    cuda_available: bool
    gpu_devices: List[Dict[str, Any]]
    torch_version: str
    cuda_version: Optional[str]
    cudnn_version: Optional[int]


class AdminCommandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    password: str
    command: str


class AdminCommandResponse(BaseModel):
    return_code: int
    stdout: str
    stderr: str
    duration_seconds: float


class ModelManager:
    """负责管理模型加载、卸载与推理的线程安全封装。"""

    def __init__(self) -> None:
        self._lock = RLock()
        self._model = None
        self._tokenizer = None
        self._metadata: Dict[str, Any] | None = None
        self._loader: Optional[
            Callable[[LoadModelRequest], tuple[Any, Any, Dict[str, Any]]]
        ] = None

    def set_loader(
        self,
        loader: Callable[[LoadModelRequest], tuple[Any, Any, Dict[str, Any]]],
    ) -> None:
        """允许测试注入轻量级 loader。"""

        with self._lock:
            self._loader = loader

    def load_model(self, request: LoadModelRequest) -> Dict[str, Any]:
        with self._lock:
            self._unload_locked()
            loader = self._loader or self._default_loader
            model, tokenizer, metadata = loader(request)
            self._model = model
            self._tokenizer = tokenizer
            self._metadata = metadata
            return metadata

    def _default_loader(
        self, request: LoadModelRequest
    ) -> tuple[Any, Any, Dict[str, Any]]:
        config = TrainingConfig()
        if request.base_model:
            config.base_model_path = request.base_model
            config.base_model_id = request.base_model
        if request.dtype:
            config.dtype = request.dtype
        if request.max_seq_length:
            config.max_seq_length = request.max_seq_length
        config.load_in_4bit = request.load_in_4bit
        config.load_in_8bit = request.load_in_8bit

        if request.merged_path:
            model, tokenizer = load_base_model(
                config,
                model_path=request.merged_path,
            )
            adapter_type = "merged"
            applied_adapters: list[str] = []
        else:
            model, tokenizer = load_base_model(config)
            adapter_type = "base"
            applied_adapters = []
            for adapter_path in request.adapter_paths():
                model = PeftModel.from_pretrained(
                    model,
                    adapter_path,
                    is_trainable=False,
                )
                applied_adapters.append(adapter_path)
            if applied_adapters:
                adapter_type = "+".join(
                    ["adapter"] + [Path(p).name for p in applied_adapters]
                )

        prepare_for_inference(model)
        metadata = {
            "loaded_at": time.time(),
            "base_model": request.base_model or config.base_model_local_path(),
            "merged_path": request.merged_path,
            "adapter_type": adapter_type,
            "adapters": applied_adapters,
            "max_seq_length": config.max_seq_length,
            "checkpoint_name": request.checkpoint_name
            or request.merged_path
            or ",".join(request.adapter_paths())
            or request.base_model,
        }
        return model, tokenizer, metadata

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        with self._lock:
            if self._model is None or self._tokenizer is None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="尚未加载任何模型，请先调用 /models/load",
                )
            start = time.time()
            prompts = [request.prompt]
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "repetition_penalty": request.repetition_penalty,
                "do_sample": request.temperature > 0,
            }
            if request.top_k is not None:
                generate_kwargs["top_k"] = request.top_k
            completions = generate_answers(
                self._model,
                self._tokenizer,
                prompts,
                **generate_kwargs,
            )
            duration = (time.time() - start) * 1000
            return GenerateResponse(
                prompt=request.prompt,
                completion=completions[0],
                latency_ms=duration,
                metadata=self._metadata or {},
            )

    def unload(self) -> None:
        with self._lock:
            self._unload_locked()

    def _unload_locked(self) -> None:
        if self._model is not None:
            try:
                if hasattr(self._model, "cpu"):
                    self._model.cpu()
            finally:
                self._model = None
        self._tokenizer = None
        self._metadata = None
        if torch.cuda.is_available():  # 尝试释放显存
            torch.cuda.empty_cache()

    def info(self) -> Dict[str, Any]:
        with self._lock:
            if self._metadata is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="当前无模型加载",
                )
            return self._metadata


model_manager = ModelManager()


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for file in path.rglob("*"):
        if file.is_file():
            try:
                total += file.stat().st_size
            except OSError:  # pragma: no cover - 忽略短暂文件异常
                continue
    return total


def _append_record(
    records: list[ModelInfo],
    path: Path,
    type_: Literal["adapter", "merged", "checkpoint", "base"],
) -> None:
    if not path.exists():
        return
    try:
        stat = path.stat()
        updated_at = stat.st_mtime
    except OSError:
        updated_at = time.time()
    size_bytes = _dir_size_bytes(path) if path.is_dir() else path.stat().st_size
    records.append(
        ModelInfo(
            name=path.name,
            path=str(path.resolve()),
            type=type_,
            updated_at=updated_at,
            size_bytes=size_bytes,
        )
    )


def discover_models(
    outputs_root: Path = DEFAULT_OUTPUT_ROOT,
    model_root: Path = DEFAULT_MODEL_ROOT,
) -> list[ModelInfo]:
    """扫描指定目录下的模型/检查点。"""

    records: list[ModelInfo] = []

    if model_root.exists():
        for child in model_root.iterdir():
            if child.is_dir():
                _append_record(records, child, "base")

    if outputs_root.exists():
        for project_dir in outputs_root.iterdir():
            if not project_dir.is_dir():
                continue
            if (project_dir / "adapter_config.json").exists():
                _append_record(records, project_dir, "adapter")
            if (project_dir / "config.json").exists() and any(
                project_dir.glob("*.safetensors")
            ):
                _append_record(records, project_dir, "merged")
            checkpoints_dir = project_dir / "checkpoints"
            if checkpoints_dir.exists():
                for ckpt in checkpoints_dir.iterdir():
                    if ckpt.is_dir() and ckpt.name.startswith("checkpoint-"):
                        marker = ckpt / "pytorch_model.bin"
                        if marker.exists() or any(ckpt.glob("*.safetensors")):
                            _append_record(records, ckpt, "checkpoint")

    return sorted(records, key=lambda item: item.updated_at, reverse=True)


@app.get("/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """返回可用的模型/检查点列表。"""

    records = discover_models()
    return ModelListResponse(items=records)


@app.post("/models/load")
async def load_model(request: LoadModelRequest) -> Dict[str, Any]:
    """加载指定模型或适配器。"""

    metadata = model_manager.load_model(request)
    return metadata


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """执行一次推理任务。"""

    return model_manager.generate(request)


@app.get("/models/current")
async def current_model() -> Dict[str, Any]:
    """查看当前已加载模型的元数据。"""

    return model_manager.info()


@app.post("/models/unload", status_code=status.HTTP_204_NO_CONTENT)
async def unload_model() -> None:
    """卸载当前模型并释放显存。"""

    model_manager.unload()


def _collect_gpu_info() -> list[Dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    devices = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        devices.append(
            {
                "index": idx,
                "name": props.name,
                "total_memory_mb": props.total_memory // (1024 * 1024),
                "multi_processor_count": props.multi_processor_count,
                "capability": f"{props.major}.{props.minor}",
            }
        )
    return devices


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """汇报服务器健康状态与硬件信息。"""

    cuda_available = torch.cuda.is_available()
    status_flag: Literal["ok", "degraded", "error"] = "ok"
    if not cuda_available:
        status_flag = "degraded"
    gpu_devices = _collect_gpu_info()
    return HealthResponse(
        status=status_flag,
        uptime_seconds=time.time() - SERVER_START_TS,
        cuda_available=cuda_available,
        gpu_devices=gpu_devices,
        torch_version=torch.__version__,
        cuda_version=getattr(torch_version, "cuda", None),
        cudnn_version=(
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
    )


async def _run_admin_command(command: str) -> subprocess.CompletedProcess[str]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
            check=False,
        ),
    )


@app.post("/admin/exec", response_model=AdminCommandResponse)
async def admin_exec(request: AdminCommandRequest) -> AdminCommandResponse:
    """执行管理员命令，需要提供正确密码。"""

    if request.password != ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="密码错误",
        )
    start = time.time()
    result = await _run_admin_command(request.command)
    duration = time.time() - start
    return AdminCommandResponse(
        return_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        duration_seconds=duration,
    )


@app.get("/")
async def root() -> Dict[str, Any]:
    """提供简单的欢迎信息。"""

    return {
        "message": "czMathLLM inference service is running",
        "uptime_seconds": time.time() - SERVER_START_TS,
        "available_endpoints": [
            "/models",
            "/models/load",
            "/models/current",
            "/models/unload",
            "/generate",
            "/health",
            "/admin/exec",
        ],
    }


__all__ = [
    "app",
    "discover_models",
    "model_manager",
    "LoadModelRequest",
    "GenerateRequest",
    "GenerateResponse",
]
