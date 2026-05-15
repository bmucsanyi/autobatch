import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from autobatch._errors import WorkloadError


@dataclass
class DeviceRecord:
    device: str
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    budget_bytes: int

    def to_json(self) -> dict[str, object]:
        return {
            "device": self.device,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "budget_bytes": self.budget_bytes,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "DeviceRecord":
        return cls(
            device=_field_as(data, "device", str),
            peak_allocated_bytes=_field_as(data, "peak_allocated_bytes", int),
            peak_reserved_bytes=_field_as(data, "peak_reserved_bytes", int),
            budget_bytes=_field_as(data, "budget_bytes", int),
        )


@dataclass
class ProbeRequest:
    workload: str
    kwargs: Mapping[str, object]
    value: int
    warmup_steps: int
    measure_steps: int
    reserve_fraction: float
    reserve_bytes: int
    devices: list[int]
    need_timing: bool
    rank: int | None
    world_size: int | None
    local_rank: int | None
    master_addr: str | None
    master_port: int | None
    candidate_id: int | None

    def to_json(self) -> dict[str, object]:
        return {
            "workload": self.workload,
            "kwargs": dict(self.kwargs),
            "value": self.value,
            "warmup_steps": self.warmup_steps,
            "measure_steps": self.measure_steps,
            "reserve_fraction": self.reserve_fraction,
            "reserve_bytes": self.reserve_bytes,
            "devices": self.devices,
            "need_timing": self.need_timing,
            "rank": self.rank,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "candidate_id": self.candidate_id,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "ProbeRequest":
        return cls(
            workload=_field_as(data, "workload", str),
            kwargs=_field_as_mapping(data, "kwargs"),
            value=_field_as(data, "value", int),
            warmup_steps=_field_as(data, "warmup_steps", int),
            measure_steps=_field_as(data, "measure_steps", int),
            reserve_fraction=_field_as_float(data, "reserve_fraction"),
            reserve_bytes=_field_as(data, "reserve_bytes", int),
            devices=_field_as_devices(data, "devices"),
            need_timing=_field_as(data, "need_timing", bool),
            rank=_field_as_optional(data, "rank", int),
            world_size=_field_as_optional(data, "world_size", int),
            local_rank=_field_as_optional(data, "local_rank", int),
            master_addr=_field_as_optional(data, "master_addr", str),
            master_port=_field_as_optional(data, "master_port", int),
            candidate_id=_field_as_optional(data, "candidate_id", int),
        )


@dataclass
class ProbeOutcome:
    status: str
    value: int
    devices: tuple[DeviceRecord, ...] = ()
    timing_seconds: tuple[float, ...] = ()
    steps_completed: int = 0
    reason: str | None = None
    exception_type: str | None = None
    exception_message: str | None = None
    failure_class: str | None = None
    stderr_path: str | None = None
    rank: int | None = None

    def __post_init__(self) -> None:
        self.devices = tuple(self.devices)
        self.timing_seconds = tuple(self.timing_seconds)

        if self.status not in {"safe", "unsafe", "failed", "timeout"}:
            msg = "worker status must be safe, unsafe, failed, or timeout"
            raise WorkloadError(msg)

        if self.status == "safe" and self.reason is not None:
            msg = "safe outcome cannot have a reason"
            raise WorkloadError(msg)

        if self.status != "safe" and self.reason is None:
            msg = "non-safe outcome must have a reason"
            raise WorkloadError(msg)

    def to_json(self) -> dict[str, object]:
        return {
            "status": self.status,
            "value": self.value,
            "devices": [device.to_json() for device in self.devices],
            "timing_seconds": list(self.timing_seconds),
            "steps_completed": self.steps_completed,
            "reason": self.reason,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "failure_class": self.failure_class,
            "stderr_path": self.stderr_path,
            "rank": self.rank,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "ProbeOutcome":
        devices_raw = _field_as_list(data, "devices")
        devices = tuple(
            DeviceRecord.from_json(_require_mapping(item, "device"))
            for item in devices_raw
        )

        return cls(
            status=_field_as(data, "status", str),
            value=_field_as(data, "value", int),
            devices=devices,
            timing_seconds=_field_as_float_tuple(data, "timing_seconds"),
            steps_completed=_field_as(data, "steps_completed", int),
            reason=_field_as_optional(data, "reason", str),
            exception_type=_field_as_optional(data, "exception_type", str),
            exception_message=_field_as_optional(data, "exception_message", str),
            failure_class=_field_as_optional(data, "failure_class", str),
            stderr_path=_field_as_optional(data, "stderr_path", str),
            rank=_field_as_optional(data, "rank", int),
        )


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        msg = "json document must be an object"
        raise TypeError(msg)

    return data


def write_json_atomic(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )

    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump(data, file, sort_keys=True, separators=(",", ":"))
            file.flush()
            os.fsync(file.fileno())

        Path(temporary_name).replace(path)
        _fsync_directory(path.parent)
    except Exception:
        temporary_path = Path(temporary_name)

        if temporary_path.exists():
            temporary_path.unlink()

        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)

    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _field_as[T](data: Mapping[str, object], name: str, expected: type[T]) -> T:
    if name not in data:
        msg = f"{name} is required"
        raise WorkloadError(msg)

    value = data[name]

    if type(value) is not expected:
        msg = f"{name} must be a {expected.__name__}"
        raise WorkloadError(msg)

    return cast("T", value)


def _field_as_optional[T](
    data: Mapping[str, object], name: str, expected: type[T]
) -> T | None:
    if name not in data:
        msg = f"{name} is required"
        raise WorkloadError(msg)

    value = data[name]

    if value is None:
        return None

    if type(value) is not expected:
        msg = f"{name} must be a {expected.__name__}"
        raise WorkloadError(msg)

    return cast("T", value)


def _field_as_float(data: Mapping[str, object], name: str) -> float:
    if name not in data:
        msg = f"{name} is required"
        raise WorkloadError(msg)

    value = data[name]

    if type(value) is int:
        return float(value)

    if type(value) is not float:
        msg = f"{name} must be a number"
        raise WorkloadError(msg)

    return value


def _field_as_mapping(data: Mapping[str, object], name: str) -> dict[str, object]:
    if name not in data:
        msg = f"{name} is required"
        raise WorkloadError(msg)

    return _require_mapping(data[name], name)


def _field_as_list(data: Mapping[str, object], name: str) -> list[object]:
    if name not in data:
        msg = f"{name} is required"
        raise WorkloadError(msg)

    value = data[name]

    if not isinstance(value, list):
        msg = f"{name} must be a list"
        raise WorkloadError(msg)

    return list(value)


def _field_as_float_tuple(data: Mapping[str, object], name: str) -> tuple[float, ...]:
    samples = []

    for item in _field_as_list(data, name):
        if type(item) is int:
            samples.append(float(item))
        elif type(item) is float:
            samples.append(item)
        else:
            msg = f"{name} must be a number"
            raise WorkloadError(msg)

    return tuple(samples)


def _field_as_devices(data: Mapping[str, object], name: str) -> list[int]:
    if name not in data:
        msg = f"{name} is required"
        raise WorkloadError(msg)

    value = data[name]

    if isinstance(value, list) and all(type(item) is int for item in value):
        return cast("list[int]", list(value))

    msg = f"{name} must be an integer list"
    raise WorkloadError(msg)


def _require_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        msg = f"{name} must be an object"
        raise WorkloadError(msg)

    result = {}

    for key, item in value.items():
        if type(key) is not str:
            msg = f"{name} keys must be strings"
            raise WorkloadError(msg)

        result[key] = item

    return result
