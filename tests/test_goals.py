import pytest

from autobatch._errors import InvalidConfigurationError
from autobatch._goals import Goal


def test_timing_goals_require_timing() -> None:
    assert Goal.fastest_step().requires_timing()
    assert Goal.best_value_rate().requires_timing()
    assert not Goal.largest_safe().requires_timing()


def test_target_goal_requires_positive_rate() -> None:
    with pytest.raises(InvalidConfigurationError, match="positive"):
        Goal.smallest_value_meeting_rate(0.0)


def test_target_goal_rejects_nan_rate() -> None:
    with pytest.raises(InvalidConfigurationError, match="positive"):
        Goal.smallest_value_meeting_rate(float("nan"))


def test_goal_json_has_no_policy_fields() -> None:
    assert Goal.best_value_rate().to_json() == {"kind": "best_value_rate"}
    assert Goal.smallest_value_meeting_rate(3.0).to_json() == {
        "kind": "smallest_value_meeting_rate",
        "target_per_second": 3.0,
    }


def test_goal_rejects_unused_target_fields() -> None:
    with pytest.raises(InvalidConfigurationError, match="target"):
        Goal("largest_safe", target_per_second=1.0, max_seconds=None)

    with pytest.raises(InvalidConfigurationError, match="max_seconds"):
        Goal(
            "smallest_value_meeting_rate",
            target_per_second=1.0,
            max_seconds=1.0,
        )

    with pytest.raises(InvalidConfigurationError, match="target_per_second"):
        Goal(
            "largest_value_under_latency",
            target_per_second=1.0,
            max_seconds=1.0,
        )
