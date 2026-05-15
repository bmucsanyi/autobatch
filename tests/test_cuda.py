import pytest

from autobatch._cuda import memory_budget_bytes, records_fit_budget, select_devices
from autobatch._errors import InvalidConfigurationError
from autobatch._protocol import DeviceRecord


def test_memory_budget_uses_stricter_reserve() -> None:
    budget = memory_budget_bytes(
        total_bytes=1000, reserve_fraction=0.10, reserve_bytes=250
    )

    assert budget == 750


def test_memory_budget_rejects_fraction_at_one() -> None:
    with pytest.raises(InvalidConfigurationError):
        memory_budget_bytes(total_bytes=1000, reserve_fraction=1.0, reserve_bytes=0)


def test_memory_budget_rejects_empty_budget() -> None:
    with pytest.raises(InvalidConfigurationError):
        memory_budget_bytes(total_bytes=1000, reserve_fraction=0.0, reserve_bytes=1000)


def test_peak_reserved_controls_memory_fit() -> None:
    records = (DeviceRecord("cuda:0", 100, 120, 110),)

    assert not records_fit_budget(records)


def test_records_fit_budget_requires_all_devices() -> None:
    records = (DeviceRecord("cuda:0", 10, 20, 30), DeviceRecord("cuda:1", 10, 40, 30))

    assert not records_fit_budget(records)


def test_records_fit_budget_accepts_all_devices_inside_budget() -> None:
    records = (DeviceRecord("cuda:0", 10, 20, 30), DeviceRecord("cuda:1", 10, 25, 30))

    assert records_fit_budget(records)


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
