import pytest

from autobatch._cuda import (
    configure_devices,
    select_devices,
)
from autobatch._errors import InvalidConfigurationError


def test_select_devices_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("autobatch._cuda.torch.cuda.is_available", lambda: False)

    with pytest.raises(InvalidConfigurationError):
        select_devices([0])


def test_select_devices_sets_declared_device(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = []

    def set_device(device: int) -> None:
        selected.append(device)

    monkeypatch.setattr("autobatch._cuda.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("autobatch._cuda.torch.cuda.set_device", set_device)

    assert select_devices([2, 3]) == [2, 3]
    assert selected == [2]


def test_configure_devices_selects_declared_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = []

    def set_device(device: int) -> None:
        selected.append(device)

    monkeypatch.setattr("autobatch._cuda.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("autobatch._cuda.torch.cuda.set_device", set_device)

    assert configure_devices([2, 3]) == [2, 3]
    assert selected == [2]
