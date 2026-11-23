"""Regressions: enabling history must not silently change memory semantics."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src import models, schemas
from src.config import settings
from src.crud import conclusion_ledger, document
from src.dreamer.specialists import DeductionSpecialist
from src.utils.agent_tools import with_ledger_temporal_tools


@pytest.fixture
def passive_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = settings.CONCLUSION_LEDGER
    monkeypatch.setattr(ledger, "ENABLED", True)
    for name in (
        "TEMPORAL_EXTRACTION_ENABLED",
        "TEMPORAL_DEDUP_ENABLED",
        "RETRIEVAL_FILTER_ENABLED",
        "CONTRADICTION_TOOL_ENABLED",
        "AUTO_RESOLVE_CONTRADICTIONS",
        "MARK_SUPERSEDED_HISTORICAL",
    ):
        monkeypatch.setattr(ledger, name, False)


def test_passive_mode_does_not_filter_or_change_dedup(passive_ledger: None) -> None:
    assert conclusion_ledger.retrieval_clause() is None
    assert document._dedup_current_filters() == []
    incoming = schemas.DocumentCreate(
        content="A claim",
        session_name="session-a",
        level="explicit",
        metadata=schemas.DocumentMetadata(
            message_ids=[], message_created_at="2026-11-22T00:00:00Z"
        ),
        embedding=[],
    )
    assert document._ledger_dedup_suffix(incoming) == ()


def test_passive_mode_keeps_model_tool_schemas(passive_ledger: None) -> None:
    tools = [
        {
            "name": "create_observations_deductive",
            "input_schema": {
                "properties": {
                    "observations": {
                        "items": {"properties": {"content": {"type": "string"}}}
                    }
                }
            },
        }
    ]
    assert with_ledger_temporal_tools(tools) is tools
    assert not any(
        tool["name"] == "create_observations_contradiction"
        for tool in DeductionSpecialist().get_tools()
    )


def test_temporal_policy_requires_explicit_opt_in(
    passive_ledger: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = models.Document(
        content="A claim",
        session_name="session-a",
        level="explicit",
        temporal_class="mutable",
    )
    assert document._ledger_dedup_suffix(doc) == ()
    monkeypatch.setattr(settings.CONCLUSION_LEDGER, "TEMPORAL_DEDUP_ENABLED", True)
    assert document._ledger_dedup_suffix(doc) == ("mutable", None, None, ())
    assert document._dedup_current_filters()
    # Retrieval remains independent of the dedup setting.
    assert conclusion_ledger.retrieval_clause() is None


def test_contested_is_not_silently_excluded(
    passive_ledger: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings.CONCLUSION_LEDGER, "RETRIEVAL_FILTER_ENABLED", True)
    monkeypatch.setattr(
        settings.CONCLUSION_LEDGER, "RETRIEVAL_EXCLUDE", ["historical", "incompatible"]
    )
    clause = conclusion_ledger.retrieval_clause()
    assert clause is not None
    sql = str(clause.compile(compile_kwargs={"literal_binds": True}))
    assert "historical" in sql and "incompatible" in sql
    assert "unknown" not in sql


@pytest.mark.asyncio
async def test_single_pgvector_duplicate_search_applies_opt_in_filter_before_limit(
    passive_ledger: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The single-document and batch pgvector paths must agree on eligibility.

    A historical/incompatible nearest match must be excluded in SQL, BEFORE
    LIMIT 1. Post-filtering would incorrectly suppress an eligible runner-up.
    """
    monkeypatch.setattr(document, "_uses_pgvector", lambda: True)
    monkeypatch.setattr(settings.CONCLUSION_LEDGER, "TEMPORAL_DEDUP_ENABLED", True)
    # Retrieval policy is deliberately independent and remains disabled.
    assert settings.CONCLUSION_LEDGER.RETRIEVAL_FILTER_ENABLED is False

    db = MagicMock()
    db.execute = AsyncMock()
    # AsyncMock's own return_value is another AsyncMock, so `.scalars()`
    # would hand back a coroutine. Pin a plain MagicMock as the result.
    db.execute.return_value = MagicMock()
    db.execute.return_value.scalars.return_value.all.return_value = []
    incoming = schemas.DocumentCreate(
        content="Candidate",
        session_name="s",
        level="deductive",
        metadata=schemas.DocumentMetadata(
            message_ids=[], message_created_at="2026-11-22T00:00:00Z"
        ),
        embedding=[0.1] * 1536,
    )
    result, existing = await document._semantic_dup_decision(
        db,
        incoming,
        "workspace",
        observer="alice",
        observed="alice",
    )
    assert existing is None
    assert result is document.SemanticRejectionResult.NOT_DUPLICATE
    query_args = db.execute.await_args
    assert query_args is not None
    sql = str(query_args.args[0])
    # The eligibility clause belongs to the vector query, before top_k. Match
    # the *predicate*, not the column name: the selected columns always list
    # `documents.validity`, which says nothing about the filter.
    predicate = "documents.validity NOT IN"
    assert predicate in sql
    assert "documents.valid_from" in sql
    assert "documents.valid_until" in sql
    assert " LIMIT " in sql.upper()
    assert sql.index(predicate) < sql.upper().index(" LIMIT ")


@pytest.mark.asyncio
async def test_single_pgvector_duplicate_search_preserves_default_policy(
    passive_ledger: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enabling the audit ledger alone must not filter dedup candidates."""
    monkeypatch.setattr(document, "_uses_pgvector", lambda: True)
    db = MagicMock()
    db.execute = AsyncMock()
    # AsyncMock's own return_value is another AsyncMock, so `.scalars()`
    # would hand back a coroutine. Pin a plain MagicMock as the result.
    db.execute.return_value = MagicMock()
    db.execute.return_value.scalars.return_value.all.return_value = []
    incoming = schemas.DocumentCreate(
        content="Candidate",
        session_name="s",
        level="deductive",
        metadata=schemas.DocumentMetadata(
            message_ids=[], message_created_at="2026-11-22T00:00:00Z"
        ),
        embedding=[0.1] * 1536,
    )
    await document._semantic_dup_decision(
        db, incoming, "workspace", observer="alice", observed="alice"
    )
    query_args = db.execute.await_args
    assert query_args is not None
    sql = str(query_args.args[0])
    # No eligibility predicate: the selected columns still mention
    # `documents.validity`, but the query must not filter on it.
    assert "documents.validity NOT IN" not in sql
    assert "documents.valid_until IS NULL" not in sql
