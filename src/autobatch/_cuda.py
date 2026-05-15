from collections.abc import Sequence

import torch

from autobatch._errors import InvalidConfigurationError
from autobatch._protocol import DeviceRecord


def validate_memory_policy(*, reserve_fraction: float, reserve_bytes: int) -> None:
    if type(reserve_fraction) not in {int, float}:
        msg = "reserve_fraction must be a number"
        raise InvalidConfigurationError(msg)

    if reserve_fraction < 0 or reserve_fraction >= 1:
        msg = "reserve_fraction must satisfy 0 <= reserve_fraction < 1"
        raise InvalidConfigurationError(msg)

    if type(reserve_bytes) is not int:
        msg = "reserve_bytes must be an integer"
        raise InvalidConfigurationError(msg)

    if reserve_bytes < 0:
        msg = "reserve_bytes must be non-negative"
        raise InvalidConfigurationError(msg)


def memory_budget_bytes(
    total_bytes: int,
    reserve_fraction: float,
    reserve_bytes: int,
) -> int:
    validate_memory_policy(
        reserve_fraction=reserve_fraction,
        reserve_bytes=reserve_bytes,
    )

    if type(total_bytes) is not int:
        msg = "total_bytes must be an integer"
        raise InvalidConfigurationError(msg)

    if total_bytes <= 0:
        msg = "total_bytes must be positive"
        raise InvalidConfigurationError(msg)

    fraction_budget = int(total_bytes * (1.0 - reserve_fraction))
    byte_budget = total_bytes - reserve_bytes
    budget = min(fraction_budget, byte_budget)

    if budget <= 0:
        msg = "memory reserve leaves no usable device memory"
        raise InvalidConfigurationError(msg)

    return budget


def select_devices(devices: list[int]) -> list[int]:
    if not torch.cuda.is_available():
        msg = "CUDA is required"
        raise InvalidConfigurationError(msg)

    torch.cuda.set_device(devices[0])

    return devices


def reset_peak_stats(devices: Sequence[int]) -> None:
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)


def synchronize_devices(devices: Sequence[int]) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


def read_memory_stats(
    devices: Sequence[int],
    *,
    reserve_fraction: float,
    reserve_bytes: int,
) -> tuple[DeviceRecord, ...]:
    records = []

    for device in devices:
        properties = torch.cuda.get_device_properties(device)
        total_bytes = properties.total_memory
        budget = memory_budget_bytes(total_bytes, reserve_fraction, reserve_bytes)
        records.append(
            DeviceRecord(
                device=f"cuda:{device}",
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                budget_bytes=budget,
            )
        )

    return tuple(records)


def memory_stats_fit(record: DeviceRecord) -> bool:
    return record.peak_reserved_bytes <= record.budget_bytes


def records_fit_budget(records: Sequence[DeviceRecord]) -> bool:
    return all(memory_stats_fit(record) for record in records)


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
