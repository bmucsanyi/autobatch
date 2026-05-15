import pytest

from autobatch._domain import Domain
from autobatch._errors import InvalidConfigurationError


def test_domain_keeps_declared_values() -> None:
    domain = Domain([2, 4, 6, 8])

    assert domain.values == (2, 4, 6, 8)


def test_domain_rejects_empty_values() -> None:
    with pytest.raises(InvalidConfigurationError):
        Domain([])


def test_explicit_domain_rejects_duplicates() -> None:
    with pytest.raises(InvalidConfigurationError):
        Domain([16, 16])


def test_explicit_domain_requires_sorted_values() -> None:
    with pytest.raises(InvalidConfigurationError):
        Domain([32, 16])
