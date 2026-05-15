from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from autobatch._cuda import validate_memory_policy, validate_step_counts
from autobatch._domain import Domain
from autobatch._errors import InvalidConfigurationError
from autobatch._goals import Goal
from autobatch._workload import split_import_path


@dataclass(kw_only=True)
class FindConfig:
    workload: str
    domain: Domain
    goal: Goal
    kwargs: Mapping[str, object]
    reserve_fraction: float
    reserve_bytes: int
    warmup_steps: int
    measure_steps: int
    timeout_s: float
    devices: list[int]
    cache_dir: Path

    def __post_init__(self) -> None:
        self.kwargs = dict(self.kwargs)


def validate_config(
    *,
    workload: str,
    values: Sequence[int],
    goal: Goal,
    kwargs: Mapping[str, object],
    reserve_fraction: float,
    reserve_bytes: int,
    warmup_steps: int,
    measure_steps: int,
    timeout_s: float,
    devices: object,
    cache_dir: str | Path,
) -> FindConfig:
    _validate_workload(workload)
    selected_devices = _validate_devices(devices)
    _validate_goal(goal)
    validate_memory_policy(
        reserve_fraction=reserve_fraction,
        reserve_bytes=reserve_bytes,
    )
    validate_step_counts(warmup_steps=warmup_steps, measure_steps=measure_steps)
    _validate_timeout(timeout_s)
    _validate_kwargs(kwargs)
    selected_domain = Domain(values)
    selected_cache_dir = Path(cache_dir)

    return FindConfig(
        workload=workload,
        domain=selected_domain,
        goal=goal,
        kwargs=kwargs,
        reserve_fraction=reserve_fraction,
        reserve_bytes=reserve_bytes,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        timeout_s=float(timeout_s),
        devices=selected_devices,
        cache_dir=selected_cache_dir,
    )


def _validate_workload(workload: object) -> None:
    if type(workload) is not str:
        msg = "workload must be a string"
        raise InvalidConfigurationError(msg)

    split_import_path(workload)


def _validate_goal(goal: object) -> None:
    if not isinstance(goal, Goal):
        msg = "goal must be a Goal"
        raise InvalidConfigurationError(msg)


def _validate_devices(devices: object) -> list[int]:
    if type(devices) is not list:
        msg = "devices must be an integer list"
        raise InvalidConfigurationError(msg)

    selected_devices = []

    for device in devices:
        if type(device) is not int:
            msg = "devices must be an integer list"
            raise InvalidConfigurationError(msg)

        selected_devices.append(device)

    if len(selected_devices) == 0:
        msg = "devices must not be empty"
        raise InvalidConfigurationError(msg)

    return selected_devices


def _validate_timeout(timeout_s: object) -> None:
    if type(timeout_s) is not float and type(timeout_s) is not int:
        msg = "timeout_s must be a number"
        raise InvalidConfigurationError(msg)

    if timeout_s <= 0:
        msg = "timeout_s must be positive"
        raise InvalidConfigurationError(msg)


def _validate_kwargs(kwargs: Mapping[str, object]) -> None:
    if not isinstance(kwargs, Mapping):
        msg = "kwargs must be a mapping"
        raise InvalidConfigurationError(msg)
