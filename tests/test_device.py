from types import SimpleNamespace

import pytest

from kingwana_asr.utils.device import DeviceSelectionError, select_device


def fake_torch(*, cuda: bool = False, mps: bool = False) -> SimpleNamespace:
    cuda_api = SimpleNamespace(
        is_available=lambda: cuda,
        get_device_name=lambda index: "Synthetic GPU",
        get_device_properties=lambda index: SimpleNamespace(total_memory=8_000_000_000),
    )
    return SimpleNamespace(
        cuda=cuda_api,
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


def test_auto_prefers_cuda_and_records_hardware() -> None:
    result = select_device("auto", torch_module=fake_torch(cuda=True, mps=True))
    assert result.device == "cuda:0"
    assert result.dtype == "float16"
    assert result.accelerator_name == "Synthetic GPU"
    assert result.total_memory_bytes == 8_000_000_000


def test_auto_falls_back_to_cpu() -> None:
    result = select_device("auto", torch_module=fake_torch())
    assert result.device == "cpu"
    assert result.dtype == "float32"


def test_explicit_unavailable_cuda_fails() -> None:
    with pytest.raises(DeviceSelectionError, match="not available"):
        select_device("cuda", torch_module=fake_torch())
