import pytest

from autobatch._cache import Cache
from autobatch._errors import InvalidConfigurationError


def test_cache_round_trip() -> None:
    cache = Cache(("cache-round-trip", "a"))

    assert cache.read_value() is None

    cache.write_value(2)

    assert cache.read_value() == 2


def test_cache_entries_are_separate() -> None:
    first = Cache(("cache-separate-first", "a"))
    second = Cache(("cache-separate-second", "b"))

    first.write_value(2)
    second.write_value(3)

    assert first.read_value() == 2
    assert second.read_value() == 3


def test_cache_distinguishes_bool_and_int_keys() -> None:
    bool_key = Cache(("cache-bool-int", True))
    int_key = Cache(("cache-bool-int", 1))

    bool_key.write_value(2)
    int_key.write_value(3)

    assert bool_key.read_value() == 2
    assert int_key.read_value() == 3


def test_cache_rejects_unhashable_key() -> None:
    key = []

    with pytest.raises(InvalidConfigurationError):
        Cache(key)


def test_cache_rejects_opaque_key_item() -> None:
    with pytest.raises(InvalidConfigurationError):
        Cache(("cache-opaque", object()))
