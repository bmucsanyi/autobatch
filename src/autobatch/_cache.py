import fcntl
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from autobatch._protocol import read_json, write_json_atomic


class Cache:
    def __init__(self, directory: str | Path, key: str) -> None:
        _validate_key(key)
        self.key = key
        self.entries = Path(directory) / "entries"

    def read_value(self) -> int | None:
        path = self.entries / f"{self.key}.json"

        if not path.exists():
            return None

        with _file_lock(self.entries / f"{self.key}.lock"):
            record = read_json(path)

        if "key" not in record:
            msg = "cache record key is required"
            raise ValueError(msg)

        if record["key"] != self.key:
            msg = "cache record key does not match filename"
            raise ValueError(msg)

        if "selected_value" not in record:
            msg = "cache record selected_value is required"
            raise ValueError(msg)

        value = record["selected_value"]

        if type(value) is not int:
            msg = "cache record selected_value must be an integer"
            raise ValueError(msg)

        return value

    def write_value(self, value: int, *, identity: Mapping[str, object]) -> None:
        record = {
            "key": self.key,
            "selected_value": value,
            "identity": dict(identity),
        }

        with _file_lock(self.entries / f"{self.key}.lock"):
            write_json_atomic(self.entries / f"{self.key}.json", record)


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a+", encoding="utf-8") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)

        try:
            yield
        finally:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def _validate_key(key: str) -> None:
    if not key or "/" in key or "\\" in key:
        msg = "cache key is not a safe filename"
        raise ValueError(msg)
