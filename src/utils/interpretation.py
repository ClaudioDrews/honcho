"""Epistemic, temporal and computational provenance of interpretations.

No application/DB imports: these types are shared by the LLM executor and
persistence layer. Validity and provenance do not establish the truth of a claim.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

TemporalClass = Literal[
    "historical",
    "durable",
    "mutable",
    "temporary",
    "situational",
    "recurring",
    "prospective",
    "unknown",
]
TEMPORAL_CLASSES = (
    "historical",
    "durable",
    "mutable",
    "temporary",
    "situational",
    "recurring",
    "prospective",
    "unknown",
)
TEMPORAL_DESCRIPTION = (
    "Classify the claim, not the age of the message: historical (past event), "
    "durable (stable attribute), mutable (state that can change), temporary "
    "(bounded period), situational (immediate context), recurring (pattern), "
    "prospective (plan or intention), unknown (uncertain). Bounds must be "
    "supported by evidence; do not substitute the message timestamp for "
    "an unknown start. Use ISO 8601 UTC timestamps or null. The end is exclusive. "
    "Do not guess an expiry or convert a date-only fact to a precise instant."
)


class TemporalValidity(BaseModel):
    """World-time bounds, distinct from the time an interpretation is recorded."""

    temporal_class: TemporalClass = Field(
        default="unknown", description=TEMPORAL_DESCRIPTION
    )
    valid_from: datetime | None = Field(
        default=None, description="Inclusive start, if known"
    )
    valid_to: datetime | None = Field(
        default=None, description="Exclusive end, if known"
    )

    @field_validator("temporal_class", mode="before")
    @classmethod
    def normalize_class(cls, value: Any) -> Any:
        return value if value in TEMPORAL_CLASSES else "unknown"

    @field_validator("valid_from", "valid_to", mode="before")
    @classmethod
    def parse_supported_time(cls, value: Any) -> datetime | None:
        # Fail soft on uncertain or malformed model output, without fabricating
        # timezone/date precision. Numbers are not accepted as Unix timestamps.
        if isinstance(value, str):
            if "T" not in value:
                return None
            try:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        if not isinstance(value, datetime) or value.utcoffset() is None:
            return None
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def reject_reversed_interval(self) -> TemporalValidity:
        if (
            self.valid_from is not None
            and self.valid_to is not None
            and self.valid_from >= self.valid_to
        ):
            self.valid_from = None
            self.valid_to = None
        return self

    def includes(self, instant: datetime) -> bool:
        if instant.utcoffset() is None:
            raise ValueError("instant must include a timezone")
        return (self.valid_from is None or self.valid_from <= instant) and (
            self.valid_to is None or instant < self.valid_to
        )

    def key(self) -> tuple[str, datetime | None, datetime | None]:
        return self.temporal_class, self.valid_from, self.valid_to


def fingerprint_prompt(
    messages: Any, tools: Any, response_schema: Any, tool_choice: Any = None
) -> str:
    """Digest effective wire messages/tool signatures, never store prompt text.

    This fingerprints the final input assembled by Honcho, not any provider's
    hidden instructions or subsequent structured-output repair requests.
    """
    payload = json.dumps(
        [messages, tools, response_schema, tool_choice],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def reported_model(raw_response: Any) -> str | None:
    for key in ("model_version", "model"):
        value = (
            raw_response.get(key)
            if isinstance(raw_response, Mapping)
            else getattr(raw_response, key, None)
        )
        if isinstance(value, str) and value and "\x00" not in value:
            return value
    return None
