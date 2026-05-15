from collections.abc import Sequence

from autobatch._errors import InvalidConfigurationError


class Domain:
    def __init__(self, values: Sequence[int]) -> None:
        self.values = tuple(values)
        self._validate()

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> int:
        return self.values[index]

    def _validate(self) -> None:
        if len(self.values) == 0:
            msg = "domain must contain at least one value"
            raise InvalidConfigurationError(msg)

        seen = set()
        previous = None

        for value in self.values:
            _require_positive_int(value, "value")

            if value in seen:
                msg = "domain values must not contain duplicates"
                raise InvalidConfigurationError(msg)

            if previous is not None and value <= previous:
                msg = "domain values must be strictly increasing"
                raise InvalidConfigurationError(msg)

            seen.add(value)
            previous = value


def _require_positive_int(value: object, field: str) -> None:
    if type(value) is not int:
        msg = f"{field} must be an integer"
        raise InvalidConfigurationError(msg)

    if value <= 0:
        msg = f"{field} must be positive"
        raise InvalidConfigurationError(msg)
