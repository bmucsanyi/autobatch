import math

from autobatch._errors import InvalidConfigurationError


class Goal:
    __hash__ = None

    def __init__(
        self,
        kind: str,
        *,
        target_per_second: float | None,
        max_seconds: float | None,
    ) -> None:
        self.kind = kind
        self.target_per_second = target_per_second
        self.max_seconds = max_seconds
        self._validate()

    @classmethod
    def largest_safe(cls) -> "Goal":
        return cls(
            "largest_safe",
            target_per_second=None,
            max_seconds=None,
        )

    @classmethod
    def smallest_safe(cls) -> "Goal":
        return cls(
            "smallest_safe",
            target_per_second=None,
            max_seconds=None,
        )

    @classmethod
    def fastest_step(cls) -> "Goal":
        return cls(
            "fastest_step",
            target_per_second=None,
            max_seconds=None,
        )

    @classmethod
    def best_value_rate(cls) -> "Goal":
        return cls(
            "best_value_rate",
            target_per_second=None,
            max_seconds=None,
        )

    @classmethod
    def smallest_value_meeting_rate(
        cls,
        target_per_second: float,
    ) -> "Goal":
        return cls(
            "smallest_value_meeting_rate",
            target_per_second=target_per_second,
            max_seconds=None,
        )

    @classmethod
    def largest_value_under_latency(
        cls,
        max_seconds: float,
    ) -> "Goal":
        return cls(
            "largest_value_under_latency",
            target_per_second=None,
            max_seconds=max_seconds,
        )

    def requires_timing(self) -> bool:
        return self.kind in {
            "fastest_step",
            "best_value_rate",
            "smallest_value_meeting_rate",
            "largest_value_under_latency",
        }

    def to_json(self) -> dict[str, object]:
        data = dict[str, object](kind=self.kind)

        if self.target_per_second is not None:
            data["target_per_second"] = self.target_per_second

        if self.max_seconds is not None:
            data["max_seconds"] = self.max_seconds

        return data

    def _validate(self) -> None:
        valid_kinds = {
            "largest_safe",
            "smallest_safe",
            "fastest_step",
            "best_value_rate",
            "smallest_value_meeting_rate",
            "largest_value_under_latency",
        }

        if self.kind not in valid_kinds:
            msg = f"unknown goal: {self.kind}"
            raise InvalidConfigurationError(msg)

        if self.kind in {
            "largest_safe",
            "smallest_safe",
            "fastest_step",
            "best_value_rate",
        }:
            self._validate_no_target()
        elif self.kind == "smallest_value_meeting_rate":
            self._validate_rate_target()
        else:
            self._validate_latency_target()

    def _validate_no_target(self) -> None:
        if self.target_per_second is not None or self.max_seconds is not None:
            msg = f"{self.kind} does not accept a target"
            raise InvalidConfigurationError(msg)

    def _validate_rate_target(self) -> None:
        if self.max_seconds is not None:
            msg = "smallest_value_meeting_rate does not accept max_seconds"
            raise InvalidConfigurationError(msg)

        _require_positive_number(self.target_per_second, "target_per_second")

    def _validate_latency_target(self) -> None:
        if self.target_per_second is not None:
            msg = "largest_value_under_latency does not accept target_per_second"
            raise InvalidConfigurationError(msg)

        _require_positive_number(self.max_seconds, "max_seconds")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Goal):
            return False

        return self.to_json() == other.to_json()

    def __repr__(self) -> str:
        return f"Goal({self.to_json()!r})"


def _require_positive_number(value: object, field: str) -> None:
    if (
        (type(value) is int or type(value) is float)
        and math.isfinite(value)
        and value > 0
    ):
        return

    msg = f"{field} must be positive"
    raise InvalidConfigurationError(msg)
