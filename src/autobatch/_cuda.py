from collections.abc import Sequence
from dataclasses import dataclass

import torch

from autobatch._errors import InvalidConfigurationError


@dataclass(slots=True)
class DeviceRecord:
    device: str
    peak_allocated_bytes: int
    peak_reserved_bytes: int


def configure_devices(devices: list[int]) -> list[int]:
    return select_devices(devices)


def select_devices(devices: list[int]) -> list[int]:
    if not torch.cuda.is_available():
        msg = "CUDA is required"
        raise InvalidConfigurationError(msg)

    device_count = torch.cuda.device_count()

    for device in devices:
        if device >= device_count:
            msg = "device id is outside the CUDA device range"
            raise InvalidConfigurationError(msg)

    torch.cuda.set_device(devices[0])

    return devices


def reset_peak_stats(devices: Sequence[int]) -> None:
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)


def synchronize_devices(devices: Sequence[int]) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


def read_memory_stats(devices: Sequence[int]) -> tuple[DeviceRecord, ...]:
    return tuple(
        (
            DeviceRecord(
                device=f"cuda:{device}",
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
        )
        for device in devices
    )


def validate_step_counts(*, warmup_steps: int, measure_steps: int) -> None:
    if type(warmup_steps) is not int:
        msg = "warmup_steps must be an integer"
        raise InvalidConfigurationError(msg)

    if type(measure_steps) is not int:
        msg = "measure_steps must be an integer"
        raise InvalidConfigurationError(msg)

    if warmup_steps < 0:
        msg = "warmup_steps must be non-negative"
        raise InvalidConfigurationError(msg)

    if measure_steps <= 0:
        msg = "measure_steps must be positive"
        raise InvalidConfigurationError(msg)
