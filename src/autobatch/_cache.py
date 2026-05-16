from collections.abc import Hashable

from autobatch._config import cache_key_identity
from autobatch._errors import InvalidConfigurationError

_VALUES = {}


class Cache:
    def __init__(self, key: Hashable) -> None:
        self.key = _validate_key(key)

    def read_value(self) -> int | None:
        return _VALUES.get(self.key)

    def write_value(self, value: int) -> None:
        if type(value) is not int:
            msg = "cached value must be an integer"
            raise ValueError(msg)

        _VALUES[self.key] = value


def _validate_key(key: Hashable) -> str:
    if not isinstance(key, Hashable):
        msg = "cache key must be hashable"
        raise InvalidConfigurationError(msg)

    return cache_key_identity(key)
