import json
import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import TypeGuard

from autobatch._cuda import validate_step_counts
from autobatch._domain import Domain
from autobatch._errors import InvalidConfigurationError
from autobatch._goals import Goal
from autobatch._types import StagedProbe


@dataclass(kw_only=True)
class FindConfig:
    probe: Callable[[int], None] | StagedProbe
    domain: Domain
    goal: Goal
    cache_key: Hashable
    warmup_steps: int
    measure_steps: int
    devices: list[int]


def validate_config(
    *,
    probe: object,
    values: Sequence[int],
    goal: Goal,
    cache_key: Hashable,
    warmup_steps: int,
    measure_steps: int,
    devices: object,
) -> FindConfig:
    selected_probe = _validate_probe(probe)
    selected_devices = _validate_devices(devices)
    _validate_goal(goal)
    _validate_cache_key(cache_key)
    validate_step_counts(warmup_steps=warmup_steps, measure_steps=measure_steps)
    selected_domain = Domain(values)

    return FindConfig(
        probe=selected_probe,
        domain=selected_domain,
        goal=goal,
        cache_key=cache_key,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        devices=selected_devices,
    )


def _validate_probe(probe: object) -> Callable[[int], None] | StagedProbe:
    if isinstance(probe, StagedProbe):
        return probe

    if not _is_probe(probe):
        msg = "probe must be callable"
        raise InvalidConfigurationError(msg)

    return probe


def _is_probe(probe: object) -> TypeGuard[Callable[[int], None]]:
    return callable(probe)


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


def _validate_cache_key(cache_key: object) -> None:
    if not isinstance(cache_key, Hashable):
        msg = "cache_key must be hashable"
        raise InvalidConfigurationError(msg)

    cache_key_identity(cache_key)


def cache_key_identity(cache_key: object) -> str:
    return json.dumps(
        cache_key_to_json(cache_key),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def cache_key_to_json(cache_key: object) -> object:
    if type(cache_key) is float:
        if not math.isfinite(cache_key):
            msg = "cache_key float values must be finite"
            raise InvalidConfigurationError(msg)

        return cache_key

    if cache_key is None or type(cache_key) in {bool, int, str}:
        return cache_key

    if type(cache_key) is tuple:
        return [cache_key_to_json(item) for item in cache_key]

    msg = "cache_key must contain only JSON scalar values and tuples"
    raise InvalidConfigurationError(msg)
