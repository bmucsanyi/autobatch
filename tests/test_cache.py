import json
from pathlib import Path

import pytest

from autobatch import _api
from autobatch._cache import Cache
from autobatch._config import FindConfig, validate_config
from autobatch._goals import Goal


def make_config(cache_dir: Path) -> FindConfig:
    return validate_config(
        workload="tests.fixtures.probes:step",
        values=[1, 2, 3],
        goal=Goal.largest_safe(),
        kwargs={"shape": 4},
        reserve_fraction=0.05,
        reserve_bytes=0,
        warmup_steps=1,
        measure_steps=1,
        timeout_s=1.0,
        devices=[0],
        cache_dir=cache_dir,
    )


def test_cache_round_trip(tmp_path: Path) -> None:
    cache = Cache(tmp_path, "key")

    assert cache.read_value() is None

    cache.write_value(2, identity={"workload": "tests.fixtures.probes:step"})

    assert cache.read_value() == 2


def test_cache_key_changes_with_domain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_api, "runtime_fingerprint", lambda: {"torch": "test"})
    monkeypatch.setattr(
        _api, "workload_source_fingerprint", lambda path: {"path": path}
    )
    first = _api.cache_key_for_config(make_config(tmp_path))
    changed = validate_config(
        workload="tests.fixtures.probes:step",
        values=[1, 2, 3, 4],
        goal=Goal.largest_safe(),
        kwargs={"shape": 4},
        reserve_fraction=0.05,
        reserve_bytes=0,
        warmup_steps=1,
        measure_steps=1,
        timeout_s=1.0,
        devices=[0],
        cache_dir=tmp_path,
    )

    second = _api.cache_key_for_config(changed)

    assert first != second


def test_cache_read_rejects_corrupt_json(tmp_path: Path) -> None:
    cache = Cache(tmp_path, "key")
    path = tmp_path / "entries" / "key.json"
    path.parent.mkdir(parents=True)
    path.write_text("bad", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        cache.read_value()


def test_cache_read_rejects_record_without_selected_value(tmp_path: Path) -> None:
    cache = Cache(tmp_path, "key")
    path = tmp_path / "entries" / "key.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"key": "key", "identity": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="selected_value"):
        cache.read_value()
