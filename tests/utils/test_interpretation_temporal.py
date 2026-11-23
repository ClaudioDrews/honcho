from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from src.utils.interpretation import (
    TEMPORAL_CLASSES,
    TemporalValidity,
    fingerprint_prompt,
    reported_model,
)


@pytest.mark.parametrize("kind", TEMPORAL_CLASSES)
def test_classes(kind):
    temporal = TemporalValidity(temporal_class=kind)
    assert temporal.temporal_class == kind
    assert temporal.valid_from is None and temporal.valid_to is None


def test_unknown_bounds_not_invented():
    temporal = TemporalValidity.model_validate(
        {"temporal_class": "historical", "valid_from": "1980"}
    )
    assert temporal.valid_from is None
    assert temporal.valid_to is None


def test_half_open_interval():
    temporal = TemporalValidity.model_validate(
        {
            "temporal_class": "temporary",
            "valid_from": "2026-01-01T00:00:00Z",
            "valid_to": "2026-01-08T00:00:00Z",
        }
    )
    assert temporal.includes(datetime(2026, 1, 1, tzinfo=UTC))
    assert not temporal.includes(datetime(2026, 1, 8, tzinfo=UTC))
    assert not temporal.includes(datetime(2025, 12, 31, tzinfo=UTC))


def test_invalid_window_clears_without_invention():
    temporal = TemporalValidity.model_validate(
        {"valid_from": "2026-06-01T00:00:00Z", "valid_to": "2026-01-01T00:00:00Z"}
    )
    assert temporal.valid_from is None and temporal.valid_to is None


def test_provider_report_not_selected_model():
    assert (
        reported_model(SimpleNamespace(model="provider/actual-version"))
        == "provider/actual-version"
    )
    assert reported_model({"model_version": "version"}) == "version"
    assert reported_model(None) is None


def test_prompt_hash_varies_with_payload_and_tool():
    a = fingerprint_prompt([{"content": "a", "role": "user"}], None, None)
    assert a == fingerprint_prompt([{"role": "user", "content": "a"}], None, None)
    assert a != fingerprint_prompt([{"content": "b", "role": "user"}], None, None)
    assert a != fingerprint_prompt(
        [{"content": "a", "role": "user"}], [{"name": "write"}], None
    )
    assert len(a) == 64
