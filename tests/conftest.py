import pytest

from tests.fixtures import probes


@pytest.fixture
def probe_calls() -> list[int]:
    probes.CALLS.clear()

    return probes.CALLS
