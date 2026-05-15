import pytest

from autobatch._errors import InvalidConfigurationError
from autobatch._goals import Goal


def test_goal_must_be_explicit() -> None:
    arguments = {}

    with pytest.raises(TypeError):
        Goal.fastest_step(**arguments)


def test_timing_goals_require_timing() -> None:
    assert Goal.fastest_step(tie="smaller").requires_timing()
    assert Goal.best_value_rate(tie="larger").requires_timing()
    assert not Goal.largest_safe().requires_timing()


def test_target_goal_requires_positive_rate() -> None:
    with pytest.raises(InvalidConfigurationError):
        Goal.smallest_value_meeting_rate(0.0, direction="increasing")
