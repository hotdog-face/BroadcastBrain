"""Runtime capability detection.

BroadcastBrain is designed to run *now*, with whatever is installed, and to
improve as backends are added. Analyzers ask this module whether their backend
exists; if it does not, the result is recorded as ``skipped/backend-missing``
rather than a failure, and ``bb reanalyze --missing`` will pick those clips up
once the dependency lands. That is why a first overnight run on a bare machine
still produces a usable library.
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import os
import platform
import shutil
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Capability:
    """One optional feature: is it present, and if not, how do I get it?"""

    name: str
    available: bool
    detail: str = ""
    install_hint: str = ""
    version: Optional[str] = None

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "available": self.available,
            "detail": self.detail,
            "install_hint": self.install_hint,
            "version": self.version,
        }


@dataclass
class GpuInfo:
    available: bool = False
    vendor: str = "none"
    name: str = ""
    memory_total_mb: int = 0
    memory_used_mb: int = 0
    utilization_pct: float = 0.0
    devices: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "vendor": self.vendor,
            "name": self.name,
            "memory_total_mb": self.memory_total_mb,
            "memory_used_mb": self.memory_used_mb,
            "utilization_pct": round(self.utilization_pct, 1),
            "devices": self.devices,
        }


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installs
        return False


def _module_version(module: str) -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(module)
    except (ImportError, PackageNotFoundError, ValueError):  # pragma: no cover
        return None


@functools.lru_cache(maxsize=1)
def detect_all(ffmpeg_binary: str = "ffmpeg", ffprobe_binary: str = "ffprobe") -> dict[str, Capability]:
    """Detect every optional backend once per process.

    Cached because probing imports is slow and the answer cannot change while
    the process runs. Call :func:`reset` after installing something.
    """
    caps: dict[str, Capability] = {}

    def add(
        name: str,
        available: bool,
        *,
        detail: str = "",
        hint: str = "",
        distribution: Optional[str] = None,
    ) -> None:
        caps[name] = Capability(
            name=name,
            available=available,
            detail=detail,
            install_hint=hint,
            version=_module_version(distribution) if distribution and available else None,
        )

    # --- external binaries ---------------------------------------------------
    ffmpeg = shutil.which(ffmpeg_binary)
    ffprobe = shutil.which(ffprobe_binary)
    add("ffmpeg", ffmpeg is not None, detail=ffmpeg or "", hint="brew install ffmpeg / apt install ffmpeg")
    add("ffprobe", ffprobe is not None, detail=ffprobe or "", hint="ships with ffmpeg")
    fpcalc = shutil.which("fpcalc")
    add("chromaprint", fpcalc is not None, detail=fpcalc or "", hint="brew install chromaprint")
    for tool, hint in (
        ("tesseract", "brew install tesseract / apt install tesseract-ocr"),
        ("7z", "brew install p7zip -- used to read .iso without mounting"),
        ("mediainfo", "optional: richer container metadata"),
    ):
        path = shutil.which(tool)
        add(tool, path is not None, detail=path or "", hint=hint)

    # --- python backends -----------------------------------------------------
    add("numpy", _module_available("numpy"), hint="pip install numpy", distribution="numpy")
    add(
        "opencv",
        _module_available("cv2"),
        hint="pip install 'broadcastbrain[vision]'",
        distribution="opencv-python-headless",
    )
    add("pillow", _module_available("PIL"), hint="pip install Pillow", distribution="pillow")
    add(
        "pytesseract",
        _module_available("pytesseract"),
        hint="pip install 'broadcastbrain[ocr]'",
        distribution="pytesseract",
    )
    add(
        "whisper",
        _module_available("faster_whisper"),
        hint="pip install 'broadcastbrain[speech]'",
        distribution="faster-whisper",
    )
    add(
        "insightface",
        _module_available("insightface"),
        hint="pip install 'broadcastbrain[faces]'",
        distribution="insightface",
    )
    add(
        "ultralytics",
        _module_available("ultralytics"),
        hint="pip install 'broadcastbrain[objects]'",
        distribution="ultralytics",
    )
    add(
        "sentence_transformers",
        _module_available("sentence_transformers"),
        hint="pip install 'broadcastbrain[embeddings]'",
        distribution="sentence-transformers",
    )
    add(
        "acoustid",
        _module_available("acoustid"),
        hint="pip install 'broadcastbrain[audioid]'",
        distribution="pyacoustid",
    )
    add(
        "internetarchive",
        _module_available("internetarchive"),
        hint="pip install 'broadcastbrain[archive]'",
        distribution="internetarchive",
    )
    add(
        "fastapi",
        _module_available("fastapi") and _module_available("uvicorn"),
        hint="pip install 'broadcastbrain[dashboard]'",
        distribution="fastapi",
    )
    add("psutil", _module_available("psutil"), hint="pip install psutil", distribution="psutil")
    add("torch", _module_available("torch"), hint="see pytorch.org for your CUDA version", distribution="torch")

    gpu = detect_gpu()
    caps["gpu"] = Capability(
        name="gpu",
        available=gpu.available,
        detail=f"{gpu.vendor}: {gpu.name}" if gpu.available else "no GPU detected",
        install_hint="install vendor drivers; ffmpeg needs matching hwaccel support",
    )
    return caps


def reset() -> None:
    """Forget cached detection (after installing a backend mid-session)."""
    detect_all.cache_clear()
    detect_gpu.cache_clear()


def has(name: str) -> bool:
    """True when the named backend is usable right now."""
    return detect_all().get(name, Capability(name, False)).available


def require(name: str, analyzer: str) -> None:
    """Raise :class:`BackendUnavailableError` if a backend is missing.

    Analyzers call this at the top of ``run``; the runner turns the exception
    into a ``skipped`` record instead of a failure.
    """
    from .errors import BackendUnavailableError

    capability = detect_all().get(name)
    if capability is None or not capability.available:
        hint = capability.install_hint if capability else ""
        raise BackendUnavailableError(analyzer, f"backend '{name}' unavailable. {hint}".strip())


@functools.lru_cache(maxsize=1)
def detect_gpu() -> GpuInfo:
    """Best-effort GPU discovery across NVIDIA, Apple Silicon and ROCm."""
    info = GpuInfo()

    # NVIDIA via NVML (accurate, gives live utilization for the dashboard).
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            info.available = True
            info.vendor = "nvidia"
            info.name = name.decode() if isinstance(name, bytes) else str(name)
            info.memory_total_mb = int(memory.total // (1024 * 1024))
            info.memory_used_mb = int(memory.used // (1024 * 1024))
            info.devices = [f"cuda:{i}" for i in range(count)]
            return info
    except Exception:
        pass

    if shutil.which("nvidia-smi"):
        info.available = True
        info.vendor = "nvidia"
        info.name = "nvidia-smi detected"
        info.devices = ["cuda:0"]
        return info

    if platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}:
        info.available = True
        info.vendor = "apple"
        info.name = "Apple Silicon (Metal / VideoToolbox)"
        info.devices = ["mps"]
        return info

    if shutil.which("rocm-smi"):
        info.available = True
        info.vendor = "amd"
        info.name = "ROCm device"
        info.devices = ["rocm:0"]
    return info


def gpu_snapshot() -> GpuInfo:
    """Live GPU utilization for the dashboard (NVML only; cheap elsewhere)."""
    info = detect_gpu()
    if info.vendor != "nvidia":
        return info
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
        snapshot = GpuInfo(
            available=True,
            vendor=info.vendor,
            name=info.name,
            memory_total_mb=int(memory.total // (1024 * 1024)),
            memory_used_mb=int(memory.used // (1024 * 1024)),
            utilization_pct=float(rates.gpu),
            devices=list(info.devices),
        )
    except Exception:
        return info
    return snapshot


def preferred_hwaccel(configured: str = "auto") -> Optional[str]:
    """Resolve the ffmpeg ``-hwaccel`` value to use, or None for software.

    ``auto`` picks by platform: VideoToolbox on macOS, CUDA where NVIDIA is
    present. An explicit value is always honoured -- if you know your box, say
    so in the config and skip the guessing.
    """
    configured = (configured or "auto").lower()
    if configured == "none":
        return None
    if configured != "auto":
        return configured
    gpu = detect_gpu()
    if gpu.vendor == "apple":
        return "videotoolbox"
    if gpu.vendor == "nvidia":
        return "cuda"
    if gpu.vendor == "amd" and os.name == "posix":
        return "vaapi"
    return None


def torch_device(preference: str = "auto") -> str:
    """Device string for torch-based analyzers (``cuda``, ``mps``, ``cpu``)."""
    preference = (preference or "auto").lower()
    if preference != "auto":
        return preference
    gpu = detect_gpu()
    if gpu.vendor == "nvidia":
        return "cuda"
    if gpu.vendor == "apple":
        return "mps"
    return "cpu"


def summary() -> list[dict[str, object]]:
    """Capability matrix for ``bb doctor`` and the dashboard."""
    return [cap.as_dict() for cap in detect_all().values()]
