import json
from collections.abc import Mapping, Sequence
from typing import Protocol

import torch
import torch.distributed as dist

from autobatch._config import FindConfig, cache_key_identity
from autobatch._cuda import DeviceRecord
from autobatch._errors import DistributedError
from autobatch._probe import ProbeOutcome
from autobatch._types import StagedProbe

_STATUS_SAFE = 0
_STATUS_UNSAFE = 1
_STATUS_FAILED = 2


class _Collectives(Protocol):
    def max_int(self, value: int) -> int: ...

    def max_float_tuple(self, values: tuple[float, ...]) -> tuple[float, ...]: ...

    def gather_int(self, value: int) -> tuple[int, ...]: ...

    def gather_optional_int(self, value: int | None) -> tuple[int | None, ...]: ...

    def gather_mapping(
        self,
        value: Mapping[str, object],
    ) -> tuple[dict[str, object], ...]: ...


class _ValueCache(Protocol):
    def read_value(self) -> int | None: ...

    def write_value(self, value: int) -> None: ...


class TorchCollectives:
    def __init__(self) -> None:
        self.group = _control_group()

    def close(self) -> None:
        if self.group is None:
            return

        dist.destroy_process_group(self.group)
        self.group = None

    def max_int(self, value: int) -> int:
        tensor = torch.tensor([value], dtype=torch.int64, device="cpu")
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=self.group)

        return _require_int(tensor.item())

    def max_float_tuple(self, values: tuple[float, ...]) -> tuple[float, ...]:
        if len(values) == 0:
            return ()

        tensor = torch.tensor(values, dtype=torch.float64, device="cpu")
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=self.group)

        return tuple(tensor.tolist())

    def gather_int(self, value: int) -> tuple[int, ...]:
        tensor = torch.tensor([value], dtype=torch.int64, device="cpu")
        gathered = [
            torch.zeros(1, dtype=torch.int64, device="cpu")
            for _ in range(dist.get_world_size(self.group))
        ]
        dist.all_gather(gathered, tensor, group=self.group)

        return tuple(_require_int(item.item()) for item in gathered)

    def gather_optional_int(self, value: int | None) -> tuple[int | None, ...]:
        values = _gather_json(value, group=self.group)

        return tuple(_require_optional_int(item) for item in values)

    def gather_mapping(
        self,
        value: Mapping[str, object],
    ) -> tuple[dict[str, object], ...]:
        values = _gather_json(value, group=self.group)

        return tuple(_require_mapping(value) for value in values)


class DistributedCache:
    def __init__(self, cache: _ValueCache, collectives: _Collectives) -> None:
        self.cache = cache
        self.collectives = collectives

    def read_value(self) -> int | None:
        value = self.cache.read_value()
        values = self.collectives.gather_optional_int(value)
        first = values[0]

        if any(value != first for value in values):
            msg = "distributed cache values differ across ranks"
            raise DistributedError(msg)

        return first

    def write_value(self, value: int) -> None:
        self.cache.write_value(value)


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def reduce_probe_outcome(
    *,
    value: int,
    local: ProbeOutcome,
    collectives: _Collectives,
) -> ProbeOutcome:
    status_code = collectives.max_int(_status_code(local))

    if status_code == _STATUS_SAFE:
        timing_seconds = collectives.max_float_tuple(local.timing_seconds)

        return ProbeOutcome(
            status="safe",
            value=value,
            devices=local.devices,
            timing_seconds=timing_seconds,
            steps_completed=local.steps_completed,
        )

    outcomes = tuple(
        _outcome_from_data(item)
        for item in collectives.gather_mapping(_outcome_to_data(local))
    )

    if status_code == _STATUS_UNSAFE:
        return _first_status(
            value=value,
            outcomes=outcomes,
            status="unsafe",
        )

    if status_code == _STATUS_FAILED:
        return _first_status(
            value=value,
            outcomes=outcomes,
            status="failed",
        )

    msg = "distributed status code is invalid"
    raise DistributedError(msg)


def validate_distributed_config(
    config: FindConfig,
    collectives: _Collectives,
) -> None:
    identity = _config_identity(config)
    identities = collectives.gather_mapping(identity)

    if any(item != identity for item in identities):
        msg = "distributed ranks must use identical autobatch configuration"
        raise DistributedError(msg)


def validate_distributed_selection(value: int, collectives: _Collectives) -> int:
    values = collectives.gather_int(value)

    if any(item != value for item in values):
        msg = "distributed ranks selected different values"
        raise DistributedError(msg)

    return value


def _status_code(outcome: ProbeOutcome) -> int:
    if outcome.status == "safe":
        return _STATUS_SAFE

    if outcome.status == "unsafe":
        return _STATUS_UNSAFE

    if outcome.status == "failed":
        return _STATUS_FAILED

    msg = "probe status is invalid"
    raise DistributedError(msg)


def _first_status(
    *,
    value: int,
    outcomes: Sequence[ProbeOutcome],
    status: str,
) -> ProbeOutcome:
    for outcome in outcomes:
        if outcome.status == status:
            return ProbeOutcome(
                status=status,
                value=value,
                devices=outcome.devices,
                timing_seconds=outcome.timing_seconds,
                steps_completed=outcome.steps_completed,
                reason=outcome.reason,
                exception_type=outcome.exception_type,
                exception_message=outcome.exception_message,
            )

    msg = "distributed reduced status is missing from gathered outcomes"
    raise DistributedError(msg)


def _outcome_to_data(outcome: ProbeOutcome) -> dict[str, object]:
    return {
        "devices": [
            {
                "device": device.device,
                "peak_allocated_bytes": device.peak_allocated_bytes,
                "peak_reserved_bytes": device.peak_reserved_bytes,
            }
            for device in outcome.devices
        ],
        "exception_message": outcome.exception_message,
        "exception_type": outcome.exception_type,
        "reason": outcome.reason,
        "status": outcome.status,
        "steps_completed": outcome.steps_completed,
        "timing_seconds": list(outcome.timing_seconds),
        "value": outcome.value,
    }


def _outcome_from_data(data: Mapping[str, object]) -> ProbeOutcome:
    devices_raw = _require_sequence(data["devices"])
    devices = []

    for item in devices_raw:
        device = _require_mapping(item)
        devices.append(
            DeviceRecord(
                device=_require_str(device["device"]),
                peak_allocated_bytes=_require_int(device["peak_allocated_bytes"]),
                peak_reserved_bytes=_require_int(device["peak_reserved_bytes"]),
            )
        )

    return ProbeOutcome(
        status=_require_str(data["status"]),
        value=_require_int(data["value"]),
        devices=tuple(devices),
        timing_seconds=tuple(
            _require_float(item) for item in _require_sequence(data["timing_seconds"])
        ),
        steps_completed=_require_int(data["steps_completed"]),
        reason=_require_optional_str(data["reason"]),
        exception_type=_require_optional_str(data["exception_type"]),
        exception_message=_require_optional_str(data["exception_message"]),
    )


def _require_int(value: object) -> int:
    if type(value) is not int:
        msg = "distributed integer reduction must return an integer"
        raise DistributedError(msg)

    return value


def _require_float(value: object) -> float:
    if type(value) is not float:
        msg = "distributed value must be a float"
        raise DistributedError(msg)

    return value


def _require_str(value: object) -> str:
    if type(value) is not str:
        msg = "distributed value must be a string"
        raise DistributedError(msg)

    return value


def _require_optional_str(value: object) -> str | None:
    if value is None:
        return None

    return _require_str(value)


def _require_optional_int(value: object) -> int | None:
    if value is None:
        return None

    if type(value) is not int:
        msg = "distributed cached value must be an integer or None"
        raise DistributedError(msg)

    return value


def _require_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        msg = "distributed value must be a mapping"
        raise DistributedError(msg)

    data = {}

    for key, item in value.items():
        data[_require_str(key)] = item

    return data


def _require_sequence(value: object) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        msg = "distributed value must be a sequence"
        raise DistributedError(msg)

    return tuple(value)


def _config_identity(config: FindConfig) -> dict[str, object]:
    return {
        "cache_key": cache_key_identity(config.cache_key),
        "device_count": len(config.devices),
        "domain": list(config.domain.values),
        "goal": config.goal.to_json(),
        "measure_steps": config.measure_steps,
        "stage_count": _stage_count(config),
        "warmup_steps": config.warmup_steps,
    }


def _stage_count(config: FindConfig) -> int:
    if isinstance(config.probe, StagedProbe):
        return len(config.probe.stages)

    return 0


def _gather_json(
    value: object,
    *,
    group: dist.ProcessGroup | None,
) -> tuple[object, ...]:
    data = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    local_length = torch.tensor([len(data)], dtype=torch.int64, device="cpu")
    gathered_lengths = [
        torch.zeros(1, dtype=torch.int64, device="cpu")
        for _ in range(dist.get_world_size(group))
    ]
    dist.all_gather(gathered_lengths, local_length, group=group)
    lengths = [_require_int(tensor.item()) for tensor in gathered_lengths]
    max_length = max(lengths)
    local_buffer = torch.zeros(max_length, dtype=torch.uint8, device="cpu")
    encoded = torch.tensor(tuple(data), dtype=torch.uint8, device="cpu")
    local_buffer[: len(data)] = encoded
    gathered_buffers = [
        torch.zeros(max_length, dtype=torch.uint8, device="cpu")
        for _ in range(dist.get_world_size(group))
    ]
    dist.all_gather(gathered_buffers, local_buffer, group=group)
    values = []

    for buffer, length in zip(gathered_buffers, lengths, strict=True):
        values.append(json.loads(_bytes_from_tensor(buffer, length).decode("utf-8")))

    return tuple(values)


def _bytes_from_tensor(tensor: torch.Tensor, length: int) -> bytes:
    return bytes(tensor.tolist()[:length])


def _control_group() -> dist.ProcessGroup | None:
    if dist.get_backend() == "gloo":
        return None

    if not dist.is_gloo_available():
        msg = "distributed autobatch requires a gloo control group"
        raise DistributedError(msg)

    return dist.new_group(backend="gloo")
