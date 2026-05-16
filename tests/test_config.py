from typing import Any

import pytest

from autobatch._config import validate_config
from autobatch._errors import InvalidConfigurationError
from autobatch._goals import Goal
from autobatch._types import ProbeStage, StagedProbe


def noop_probe(value: int) -> None:
    _ = value


def bad_stage() -> Any:
    return noop_probe


def missing_sync() -> Any:
    return None


def test_config_requires_declared_values() -> None:
    with pytest.raises(InvalidConfigurationError):
        validate_config(
            probe=noop_probe,
            values=[],
            goal=Goal.fastest_step(),
            cache_key=("config-values", "a"),
            warmup_steps=1,
            measure_steps=1,
            devices=[0],
        )


def test_config_rejects_bad_devices() -> None:
    with pytest.raises(InvalidConfigurationError):
        validate_config(
            probe=noop_probe,
            values=[1, 2],
            goal=Goal.largest_safe(),
            cache_key=("config-devices", "a"),
            warmup_steps=1,
            measure_steps=1,
            devices=[],
        )


def test_config_rejects_negative_devices() -> None:
    with pytest.raises(InvalidConfigurationError, match="non-negative"):
        validate_config(
            probe=noop_probe,
            values=[1, 2],
            goal=Goal.largest_safe(),
            cache_key=("config-devices", "negative"),
            warmup_steps=1,
            measure_steps=1,
            devices=[-1],
        )


def test_config_rejects_duplicate_devices() -> None:
    with pytest.raises(InvalidConfigurationError, match="duplicates"):
        validate_config(
            probe=noop_probe,
            values=[1, 2],
            goal=Goal.largest_safe(),
            cache_key=("config-devices", "duplicate"),
            warmup_steps=1,
            measure_steps=1,
            devices=[0, 0],
        )


def test_config_rejects_non_callable_probe() -> None:
    with pytest.raises(InvalidConfigurationError):
        validate_config(
            probe="bad",
            values=[1, 2],
            goal=Goal.largest_safe(),
            cache_key=("config-probe", "a"),
            warmup_steps=1,
            measure_steps=1,
            devices=[0],
        )


def test_config_rejects_unhashable_cache_key() -> None:
    key = []

    with pytest.raises(InvalidConfigurationError):
        validate_config(
            probe=noop_probe,
            values=[1, 2],
            goal=Goal.largest_safe(),
            cache_key=key,
            warmup_steps=1,
            measure_steps=1,
            devices=[0],
        )


def test_config_rejects_opaque_cache_key_item() -> None:
    with pytest.raises(InvalidConfigurationError):
        validate_config(
            probe=noop_probe,
            values=[1, 2],
            goal=Goal.largest_safe(),
            cache_key=("config-cache-key", object()),
            warmup_steps=1,
            measure_steps=1,
            devices=[0],
        )


def test_config_rejects_nan_cache_key_float() -> None:
    with pytest.raises(InvalidConfigurationError):
        validate_config(
            probe=noop_probe,
            values=[1, 2],
            goal=Goal.largest_safe(),
            cache_key=("config-cache-key", float("nan")),
            warmup_steps=1,
            measure_steps=1,
            devices=[0],
        )


def test_staged_probe_requires_probe_stage_items() -> None:
    with pytest.raises(TypeError, match="ProbeStage"):
        StagedProbe([bad_stage()])


def test_probe_stage_requires_callable_sync() -> None:
    with pytest.raises(TypeError, match="sync"):
        ProbeStage(noop_probe, missing_sync())
