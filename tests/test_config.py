import pytest

from autobatch._config import validate_config
from autobatch._errors import InvalidConfigurationError
from autobatch._goals import Goal


def test_config_requires_declared_values() -> None:
    with pytest.raises(InvalidConfigurationError):
        validate_config(
            workload="tests.fixtures.probes:step",
            values=[],
            goal=Goal.fastest_step(tie="smaller"),
            kwargs={},
            reserve_fraction=0.05,
            reserve_bytes=0,
            warmup_steps=1,
            measure_steps=1,
            timeout_s=1.0,
            devices=[0],
            cache_dir="cache",
        )


def test_config_rejects_bad_devices() -> None:
    with pytest.raises(InvalidConfigurationError):
        validate_config(
            workload="tests.fixtures.probes:step",
            values=[1, 2],
            goal=Goal.largest_safe(),
            kwargs={},
            reserve_fraction=0.05,
            reserve_bytes=0,
            warmup_steps=1,
            measure_steps=1,
            timeout_s=1.0,
            devices=[],
            cache_dir="cache",
        )
