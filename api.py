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

# from __future__ import annotations
# 这是一个未来的语言特性，允许在类型提示中使用尚未完整定义的类型。
# 例如，在一个类的方法中，可以将该类本身作为类型提示，而无需使用字符串形式。
# 在 Python 3.10+ 中，这已成为默认行为，所以这行导入主要是为了兼容旧版本。
from __future__ import annotations

import asyncio
import subprocess  # 用于执行子进程，这里主要用于管理员命令功能。
import time
from pathlib import Path  # 面向对象的路径操作库，比传统的 os.path 更现代和方便。
from threading import RLock  # 可重入锁，用于保证 ModelManager 的线程安全。
from typing import Any, Callable, Dict, List, Literal, Optional

import torch
# FastAPI 是一个现代、高性能的 Python Web 框架，用于构建 API。
from fastapi import FastAPI, HTTPException, status  # type: ignore[import]
# Pydantic 用于数据验证和设置管理，FastAPI 深度集成它来处理请求和响应体。
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

from peft import PeftModel  # PEFT (Parameter-Efficient Fine-Tuning) 库，用于加载 LoRA 等适配器。
from torch import version as torch_version

# API 的描述信息，会显示在 Swagger UI/OpenAPI 文档中。
APP_DESCRIPTION = """czMathLLM 推理服务，支持模型加载、推理、健康检查及管理命令。"""

# 创建 FastAPI 应用实例。
app = FastAPI(
    title="czMathLLM API",
    description=APP_DESCRIPTION,
    version="0.1.0",
)

ADMIN_PASSWORD = "--"  # 管理员命令的密码，为了安全应从环境变量或配置文件读取。
DEFAULT_OUTPUT_ROOT = Path("outputs")  # 存放训练输出（如适配器、合并模型）的默认根目录。
DEFAULT_MODEL_ROOT = Path("models")  # 存放基座模型的默认根目录。
SERVER_START_TS = time.time()  # 服务器启动的时间戳，用于计算运行时长。


# Pydantic 模型，定义了加载模型的请求体结构。
class LoadModelRequest(BaseModel):
    """模型加载请求体。"""

    # model_config 用于配置 Pydantic 模型的行为。
    # extra="forbid" 表示不允许请求体中包含未在此模型中定义的额外字段。
    model_config = ConfigDict(extra="forbid")

    # Field(...) 用于为字段提供额外的元数据和验证规则。
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

    # @model_validator 是 Pydantic 的一个装饰器，用于定义模型级别的复杂验证逻辑。
    # mode="after" 表示在单个字段的验证完成后执行此验证函数。
    @model_validator(mode="after")
    def _validate_paths(self) -> "LoadModelRequest":
        # 确保至少提供了一种有效的模型路径组合。
        if not (
            self.merged_path or self.base_model or self.lora_path or self.grpo_path
        ):
            raise ValueError(
                "需至少提供 merged_path，或提供 base_model/适配器路径之一。"
            )
        return self

    def adapter_paths(self) -> list[str]:
        """辅助方法，返回一个包含所有有效适配器路径的列表。"""
        paths: list[str] = []
        if self.lora_path:
            paths.append(self.lora_path)
        if self.grpo_path:
            paths.append(self.grpo_path)
        return paths


class ModelInfo(BaseModel):
    """模型或检查点的元信息。"""

    name: str  # 模型或检查点的名称
    path: str  # 绝对路径
    # Literal[...] 是一个类型提示，表示变量的值只能是给定的几个字符串之一。
    type: Literal["adapter", "merged", "checkpoint", "base"]
    updated_at: float  # 文件或目录的最后修改时间戳
    size_bytes: Optional[int] = None  # 总大小（字节）


class ModelListResponse(BaseModel):
    """模型列表的响应体。"""
    items: List[ModelInfo]


class GenerateRequest(BaseModel):
    """推理请求体。"""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., description="用户输入的数学题或问题")
    # ge=0.0 表示 greater than or equal to 0.0，是 Pydantic 的验证规则。
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
    """推理结果的响应体。"""
    prompt: str
    completion: str
    latency_ms: float
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    """健康检查的响应体。"""
    status: Literal["ok", "degraded", "error"]
    uptime_seconds: float
    cuda_available: bool
    gpu_devices: List[Dict[str, Any]]
    torch_version: str
    cuda_version: Optional[str]
    cudnn_version: Optional[int]


class AdminCommandRequest(BaseModel):
    """管理员命令的请求体。"""
    model_config = ConfigDict(extra="forbid")

    password: str
    command: str


class AdminCommandResponse(BaseModel):
    """管理员命令的响应体。"""
    return_code: int
    stdout: str
    stderr: str
    duration_seconds: float


class ModelManager:
    """负责管理模型加载、卸载与推理的线程安全封装。

    这个类是整个 API 的核心状态管理者。它确保在任何时候只有一个模型被加载，
    并通过锁机制（RLock）来处理并发请求，避免在模型加载或推理时发生冲突。
    """

    def __init__(self) -> None:
        # RLock (Re-entrant Lock) 是一种可重入锁。
        # 同一个线程可以多次获取这个锁而不会被自己阻塞，这在复杂的调用链中非常有用。
        # 例如，一个加锁的方法调用了同一个对象的另一个也需要加锁的方法。
        self._lock = RLock()
        self._model = None  # 当前加载的模型对象
        self._tokenizer = None  # 当前加载的分词器对象
        self._metadata: Dict[str, Any] | None = None  # 当前模型的元数据
        # _loader 属性使用了“依赖注入”的设计模式。
        # 默认情况下它是 None，但在测试时，可以调用 set_loader() 注入一个模拟的加载器，
        # 这样就可以在不实际加载大模型的情况下测试 API 逻辑。
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
        """加载一个新模型。

        此方法是线程安全的。它首先卸载当前模型（如果有），然后调用加载器加载新模型。
        """
        with self._lock:
            self._unload_locked()  # 先释放旧模型资源
            loader = self._loader or self._default_loader  # 优先使用注入的 loader
            model, tokenizer, metadata = loader(request)
            self._model = model
            self._tokenizer = tokenizer
            self._metadata = metadata
            return metadata

    def _default_loader(
        self, request: LoadModelRequest
    ) -> tuple[Any, Any, Dict[str, Any]]:
        """默认的模型加载实现。"""
        config = TrainingConfig()  # 加载默认的训练配置
        # 根据请求参数覆盖配置
        if request.base_model:
            config.base_model_path = request.base_model
            config.base_model_id = request.base_model
        if request.dtype:
            config.dtype = request.dtype
        if request.max_seq_length:
            config.max_seq_length = request.max_seq_length
        config.load_in_4bit = request.load_in_4bit
        config.load_in_8bit = request.load_in_8bit

        # 核心加载逻辑：区分是加载合并后的模型还是基座+适配器
        if request.merged_path:
            # 如果提供了合并模型的路径，直接加载完整模型
            model, tokenizer = load_base_model(
                config,
                model_path=request.merged_path,
            )
            adapter_type = "merged"
            applied_adapters: list[str] = []
        else:
            # 否则，先加载基座模型
            model, tokenizer = load_base_model(config)
            adapter_type = "base"
            applied_adapters = []
            # 依次应用所有指定的适配器（如 LoRA, GRPO）
            for adapter_path in request.adapter_paths():
                # PeftModel.from_pretrained 会将适配器层合并到基座模型中
                model = PeftModel.from_pretrained(
                    model,
                    adapter_path,
                    is_trainable=False,  # 推理模式下设为 False
                )
                applied_adapters.append(adapter_path)
            if applied_adapters:
                # 如果应用了适配器，更新类型名称
                adapter_type = "+".join(
                    ["adapter"] + [Path(p).name for p in applied_adapters]
                )

        prepare_for_inference(model)  # 对模型进行一些推理前的准备工作，如设置为评估模式
        # 构建并返回元数据
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
        """使用当前加载的模型进行推理。"""
        with self._lock:
            if self._model is None or self._tokenizer is None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="尚未加载任何模型，请先调用 /models/load",
                )
            start = time.time()
            prompts = [request.prompt]
            # 准备传递给 `generate` 方法的参数
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "repetition_penalty": request.repetition_penalty,
                "do_sample": request.temperature > 0,  # 温度为0时，使用贪心解码，无需采样
            }
            if request.top_k is not None:
                generate_kwargs["top_k"] = request.top_k

            # 调用核心生成函数
            completions = generate_answers(
                self._model,
                self._tokenizer,
                prompts,
                **generate_kwargs,
            )
            duration = (time.time() - start) * 1000  # 计算延迟，单位毫秒
            return GenerateResponse(
                prompt=request.prompt,
                completion=completions[0],
                latency_ms=duration,
                metadata=self._metadata or {},
            )

    def unload(self) -> None:
        """卸载当前模型并释放资源。"""
        with self._lock:
            self._unload_locked()

    def _unload_locked(self) -> None:
        """在持有锁的情况下执行卸载操作。"""
        if self._model is not None:
            try:
                # 如果模型在 GPU 上，尝试将其移回 CPU，这有助于更快地释放显存
                if hasattr(self._model, "cpu"):
                    self._model.cpu()
            finally:
                self._model = None
        self._tokenizer = None
        self._metadata = None
        if torch.cuda.is_available():  # 尝试手动触发 CUDA 缓存清理
            torch.cuda.empty_cache()

    def info(self) -> Dict[str, Any]:
        """返回当前加载模型的元数据。"""
        with self._lock:
            if self._metadata is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="当前无模型加载",
                )
            return self._metadata


# 创建 ModelManager 的全局单例
model_manager = ModelManager()


def _dir_size_bytes(path: Path) -> int:
    """递归计算目录的总大小。"""
    total = 0
    # path.rglob('*') 会递归地遍历目录下的所有文件和子目录
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
    """辅助函数，用于创建 ModelInfo 记录并添加到列表中。"""
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
            path=str(path.resolve()),  # 保存绝对路径
            type=type_,
            updated_at=updated_at,
            size_bytes=size_bytes,
        )
    )


def discover_models(
    outputs_root: Path = DEFAULT_OUTPUT_ROOT,
    model_root: Path = DEFAULT_MODEL_ROOT,
) -> list[ModelInfo]:
    """扫描指定目录下的模型/检查点。

    这个函数通过检查特定文件名（如 adapter_config.json）或文件结构
    来判断目录中包含的模型类型。
    """
    records: list[ModelInfo] = []

    # 扫描基座模型目录
    if model_root.exists():
        for child in model_root.iterdir():
            if child.is_dir():
                _append_record(records, child, "base")

    # 扫描输出目录
    if outputs_root.exists():
        for project_dir in outputs_root.iterdir():
            if not project_dir.is_dir():
                continue
            # 如果目录包含 adapter_config.json，则认为是适配器
            if (project_dir / "adapter_config.json").exists():
                _append_record(records, project_dir, "adapter")
            # 如果目录包含 config.json 和 .safetensors 文件，则认为是合并后的模型
            if (project_dir / "config.json").exists() and any(
                project_dir.glob("*.safetensors")
            ):
                _append_record(records, project_dir, "merged")

            # 扫描子目录中的 checkpoints
            checkpoints_dir = project_dir / "checkpoints"
            if checkpoints_dir.exists():
                for ckpt in checkpoints_dir.iterdir():
                    if ckpt.is_dir() and ckpt.name.startswith("checkpoint-"):
                        # 检查点目录的标志性文件
                        marker = ckpt / "pytorch_model.bin"
                        if marker.exists() or any(ckpt.glob("*.safetensors")):
                            _append_record(records, ckpt, "checkpoint")

    # 按更新时间降序排序
    return sorted(records, key=lambda item: item.updated_at, reverse=True)


# FastAPI 路径操作函数（路由）
# @app.get(...) 将下面的函数注册为处理 GET /models 请求的处理器。
# response_model 指定了响应体应该遵循的 Pydantic 模型，FastAPI 会自动进行数据转换和验证。
@app.get("/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """返回可用的模型/检查点列表。"""
    records = discover_models()
    return ModelListResponse(items=records)


@app.post("/models/load")
async def load_model(request: LoadModelRequest) -> Dict[str, Any]:
    """加载指定模型或适配器。"""
    # FastAPI 会自动将请求的 JSON body 解析并验证为 LoadModelRequest 对象。
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


# status_code=status.HTTP_204_NO_CONTENT 表示成功时返回 204 状态码，且没有响应体。
@app.post("/models/unload", status_code=status.HTTP_204_NO_CONTENT)
async def unload_model() -> None:
    """卸载当前模型并释放显存。"""
    model_manager.unload()


def _collect_gpu_info() -> list[Dict[str, Any]]:
    """收集所有可用的 NVIDIA GPU 信息。"""
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
        status_flag = "degraded"  # 如果 CUDA 不可用，状态降级
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
    """在独立的线程中异步执行 shell 命令。"""
    loop = asyncio.get_running_loop()
    # subprocess.run 是一个阻塞操作，直接在 async 函数中调用会阻塞整个事件循环。
    # loop.run_in_executor(None, ...) 会将这个阻塞操作放到一个独立的线程池中执行，
    # 从而避免阻塞主线程，这是在 asyncio 中处理阻塞 IO 的标准做法。
    return await loop.run_in_executor(
        None,
        lambda: subprocess.run(
            # 注意：["bash", "-lc", command] 在 Windows 上可能无法直接工作。
            # 在 Windows 上通常使用 cmd.exe 或 powershell.exe。
            # 例如：["cmd", "/c", command]
            ["bash", "-lc", command],
            capture_output=True,  # 捕获 stdout 和 stderr
            text=True,  # 以文本模式返回输出
            check=False,  # 如果命令失败（返回非零），不抛出异常
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
    """提供简单的欢迎信息和 API 端点列表。"""
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


# __all__ 定义了当其他模块使用 `from czMathLLM.api import *` 时，
# 哪些对象会被导入。这是一种良好的编程实践，可以明确模块的公共 API。
__all__ = [
    "app",
    "discover_models",
    "model_manager",
    "LoadModelRequest",
    "GenerateRequest",
    "GenerateResponse",
]
