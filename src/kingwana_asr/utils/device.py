"""Explicit, testable PyTorch device and dtype selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SUPPORTED_DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


class DeviceSelectionError(RuntimeError):
    """Raised when an explicitly requested accelerator is unavailable."""


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    requested: str
    device: str
    dtype: str
    accelerator_name: str | None = None
    total_memory_bytes: int | None = None


def select_device(requested: str = "auto", *, torch_module: Any | None = None) -> DeviceSpec:
    """Resolve auto/cuda/mps/cpu and choose a conservative inference dtype."""
    requested = requested.lower()
    if requested not in SUPPORTED_DEVICE_CHOICES:
        choices = ", ".join(SUPPORTED_DEVICE_CHOICES)
        raise DeviceSelectionError(f"Unknown device {requested!r}; choose one of: {choices}")
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError as exc:
            raise DeviceSelectionError(
                "PyTorch is not installed; install the project's 'asr' extra"
            ) from exc

    cuda_available = bool(torch_module.cuda.is_available())
    mps_backend = getattr(torch_module.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())

    resolved = requested
    if requested == "auto":
        resolved = "cuda" if cuda_available else "mps" if mps_available else "cpu"
    if resolved == "cuda" and not cuda_available:
        raise DeviceSelectionError("CUDA was requested but is not available")
    if resolved == "mps" and not mps_available:
        raise DeviceSelectionError("MPS was requested but is not available")

    if resolved == "cuda":
        properties = torch_module.cuda.get_device_properties(0)
        return DeviceSpec(
            requested=requested,
            device="cuda:0",
            dtype="float16",
            accelerator_name=torch_module.cuda.get_device_name(0),
            total_memory_bytes=int(properties.total_memory),
        )
    if resolved == "mps":
        return DeviceSpec(
            requested=requested, device="mps", dtype="float32", accelerator_name="MPS"
        )
    return DeviceSpec(requested=requested, device="cpu", dtype="float32")
