"""Derivation records and assertion basis for the conclusion ledger.

A derivation record says who produced a conclusion and how: the producer
(`deriver`, `dreamer.<specialist>`, `api`), the configured model and its
fallback, a fingerprint of the prompt scaffold, and the run id. Producers bind
it with `derivation_scope` around the code that writes conclusions; the
document CRUD layer reads it with `current_derivation`. A contextvar keeps the
existing call signatures unchanged, as `tenant_context` does for tenancy.

The assertion basis distinguishes exact utterances from model readings. Only an
entire authored message, unchanged except surrounding whitespace, qualifies as
verbatim. Similarity is evidence discovery only and never establishes meaning.
A verbatim utterance is not necessarily a factual assertion (e.g. a quotation).
The legacy near-verbatim label is readable but never minted by this version.

Validity is the conclusion's standing for the present, not its truth:

- `applicable`: in force (the default);
- `historical`: was true, superseded by a later conclusion;
- `incompatible`: contradicted and resolved against;
- `unknown`: contested and not yet resolved.
"""

from __future__ import annotations

import contextvars
import difflib
import hashlib
import inspect
import re
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from functools import cache
from typing import Any, Literal

AssertionBasis = Literal[
    "declared_verbatim",
    "declared_near_verbatim",
    "llm_reading",
    "llm_inference",
    "api_supplied",
]
Validity = Literal["applicable", "historical", "incompatible", "unknown"]

ASSERTION_BASES: tuple[str, ...] = (
    "declared_verbatim",
    "declared_near_verbatim",
    "llm_reading",
    "llm_inference",
    "api_supplied",
)
VALIDITIES: tuple[str, ...] = ("applicable", "historical", "incompatible", "unknown")

_INFERENCE_LEVELS = frozenset({"deductive", "inductive", "contradiction"})
_WS_RE = re.compile(r"\s+")
_EDGE_PUNCT = ".,;:!?\"'()[]{}«»“”‘’"


@dataclass(frozen=True)
class DerivationRecord:
    """Who produced a conclusion and how.

    `model` and `fallback_model` are the configured pair; `model_used` is the
    provider-reported identifier when available, otherwise the selected model.
    `selected_model` and `reported_model` retain that distinction.
    """

    producer: str
    model: str | None = None
    fallback_model: str | None = None
    model_used: str | None = None
    selected_model: str | None = None
    reported_model: str | None = None
    prompt_sha256: str | None = None
    transport: str | None = None
    attempt: int | None = None
    iteration: int | None = None
    span_id: str | None = None
    tool_call_id: str | None = None
    fallback: bool | None = None
    prompt_version: str | None = None
    run_id: str | None = None

    def as_metadata(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


_current: contextvars.ContextVar[DerivationRecord | None] = contextvars.ContextVar(
    "honcho_derivation_record", default=None
)


@contextmanager
def derivation_scope(record: DerivationRecord) -> Generator[DerivationRecord]:
    """Bind `record` for conclusions written inside the block."""
    token = _current.set(record)
    try:
        yield record
    finally:
        _current.reset(token)


def current_derivation() -> DerivationRecord | None:
    return _current.get()


class model_used_scope:  # noqa: N801 - used like a function, as contextmanagers are
    """Record `model` as the answering model on the bound record, if any.

    A no-op without a bound record or a model, so the LLM layer can call it
    unconditionally around tool execution. A class rather than a
    @contextmanager so `__exit__` is typed `None`: type checkers then know it
    never swallows exceptions, which keeps flow analysis in the tool loop
    intact.
    """

    def __init__(self, model: str | None, **call: Any) -> None:
        self._model: str | None = model
        self._call = call
        self._token: contextvars.Token[DerivationRecord | None] | None = None

    def __enter__(self) -> None:
        record = _current.get()
        if record is not None and (self._model or self._call):
            self._token = _current.set(
                replace(
                    record, model_used=self._model or record.model_used, **self._call
                )
            )

    def __exit__(self, *_exc: object) -> None:
        if self._token is not None:
            _current.reset(self._token)
            self._token = None


def prompt_fingerprint(*parts: str | None) -> str:
    """Stable short fingerprint of a prompt scaffold.

    Pass the parts that define the prompt *version* (the template rendered
    with placeholders, custom instructions), not per-call content such as the
    messages being read, so the same version always hashes the same.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(b"\x1f")
        digest.update((part or "").encode("utf-8"))
    return digest.hexdigest()[:16]


@cache
def _builder_scaffold(builder: Callable[..., Any]) -> str:
    """The identity of a prompt builder's version.

    The builder's own source, not a rendering of it: a rendering would embed
    the peer and the messages, and rendering it a second time just to
    fingerprint it would run the builder twice per call.
    """
    try:
        return inspect.getsource(builder)
    except (OSError, TypeError):  # pragma: no cover - source not always available
        return getattr(builder, "__qualname__", repr(builder))


def prompt_scaffold_fingerprint(builder: Callable[..., Any], *parts: str | None) -> str:
    """Fingerprint of a prompt builder's version, without rendering it.

    `builder` is the function that assembles the prompt; `parts` are the
    inputs that change its text (custom instructions, enabled features, the
    tool set). The result depends on the prompt's source and on those parts,
    never on the peer or the conversation being processed.
    """
    return prompt_fingerprint(_builder_scaffold(builder), *parts)


def normalize_for_match(text: str) -> str:
    return _WS_RE.sub(" ", text).strip().strip(_EDGE_PUNCT).strip().casefold()


@dataclass(frozen=True)
class DeclaredSpan:
    """The part of a cited message a conclusion rests on."""

    message_id: str
    start: int
    end: int
    text: str
    match: Literal["verbatim", "near_verbatim", "closest"]
    score: float

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("text")
        data["sha256"] = hashlib.sha256(self.text.encode()).hexdigest()
        data["score"] = round(self.score, 4)
        return data


def find_declared_spans(
    content: str,
    cited_messages: Iterable[tuple[str, str]],
    *,
    verbatim_min_chars: int = 12,
    near_verbatim_min_ratio: float = 0.9,
) -> tuple[Literal["verbatim", "near_verbatim", "none"], list[DeclaredSpan]]:
    """Return whole-message evidence, never promote fuzzy or partial matches.

    Callers must pass only live messages authored by the observed peer. Exact
    whole-message match preserves context (negations, qualifications, quotes).
    Matching does not prove truth or endorsement. The ratio remains a legacy
    configuration argument; it does not determine epistemic status.
    """
    del near_verbatim_min_ratio
    needle = content.strip()
    messages = [(mid, text) for mid, text in cited_messages if text.strip()]
    if not needle or not messages:
        return "none", []
    # Equality pass first: cheap, and any whole-message exact match makes the
    # fuzzy scores below unnecessary.
    exact: list[DeclaredSpan] = []
    if len(needle) >= verbatim_min_chars:
        for mid, text in messages:
            stripped = text.strip()
            if stripped == needle:
                start = len(text) - len(text.lstrip())
                exact.append(
                    DeclaredSpan(
                        mid, start, start + len(stripped), stripped, "verbatim", 1.0
                    )
                )
    if exact:
        return "verbatim", exact
    # Fuzzy pass only when no exact whole-message match exists; the best span
    # is reference-only evidence and never changes the epistemic status.
    best: DeclaredSpan | None = None
    for mid, text in messages:
        stripped = text.strip()
        start = len(text) - len(text.lstrip())
        end = start + len(stripped)
        score = difflib.SequenceMatcher(
            None,
            normalize_for_match(needle),
            normalize_for_match(stripped),
            autojunk=True,
        ).ratio()
        if best is None or score > best.score:
            best = DeclaredSpan(mid, start, end, stripped, "closest", score)
    return "none", [best] if best is not None else []


def classify_assertion_basis(
    *,
    level: str,
    content: str,
    cited_message_contents: Iterable[str] = (),
    producer: str | None = None,
    verbatim_min_chars: int = 12,
    near_verbatim_min_ratio: float = 0.9,
) -> AssertionBasis:
    """Deterministic assertion basis for a new conclusion."""
    basis, _ = classify_with_spans(
        level=level,
        content=content,
        cited_messages=[
            (str(i), text) for i, text in enumerate(cited_message_contents)
        ],
        producer=producer,
        verbatim_min_chars=verbatim_min_chars,
        near_verbatim_min_ratio=near_verbatim_min_ratio,
    )
    return basis


def classify_with_spans(
    *,
    level: str,
    content: str,
    cited_messages: Iterable[tuple[str, str]] = (),
    producer: str | None = None,
    verbatim_min_chars: int = 12,
    near_verbatim_min_ratio: float = 0.9,
) -> tuple[AssertionBasis, list[DeclaredSpan]]:
    """Assertion basis plus the declared spans it rests on.

    Spans are computed for explicit conclusions only: derived levels rest on
    other conclusions (their `source_ids`), not on message text.
    """
    if level in _INFERENCE_LEVELS:
        return ("api_supplied" if producer == "api" else "llm_inference"), []
    kind, spans = find_declared_spans(
        content,
        cited_messages,
        verbatim_min_chars=verbatim_min_chars,
        near_verbatim_min_ratio=near_verbatim_min_ratio,
    )
    if producer == "api":
        return "api_supplied", spans
    if kind == "verbatim":
        return "declared_verbatim", spans
    # Unreachable on purpose: find_declared_spans never returns
    # "near_verbatim" in this version. The label is legacy read-only, and
    # this branch only keeps stored legacy values classifiable.
    if kind == "near_verbatim":
        return "declared_near_verbatim", spans
    return "llm_reading", spans
