"""Conclusion ledger: assertion basis, validity, derivation record, permanent history."""

import datetime
from collections.abc import Iterator

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.crud import conclusion_ledger
from src.exceptions import ValidationException
from src.models import Peer, Workspace
from src.utils.derivation import DerivationRecord, derivation_scope
from src.utils.interpretation import TemporalValidity


@pytest.fixture
def ledger_on() -> Iterator[None]:
    original = settings.CONCLUSION_LEDGER.ENABLED
    settings.CONCLUSION_LEDGER.ENABLED = True
    try:
        yield
    finally:
        settings.CONCLUSION_LEDGER.ENABLED = original


async def _seed(
    db_session: AsyncSession, workspace: Workspace, peer: Peer, contents: list[str]
) -> tuple[str, list[str]]:
    """Seed a session with messages; returns (session_name, public ids)."""
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add(session)
    await db_session.flush()
    messages = [
        models.Message(
            session_name=session.name,
            workspace_name=workspace.name,
            peer_name=peer.name,
            content=content,
            seq_in_session=seq,
        )
        for seq, content in enumerate(contents, start=1)
    ]
    db_session.add_all(messages)
    db_session.add(
        models.Collection(
            workspace_name=workspace.name, observer=peer.name, observed=peer.name
        )
    )
    await db_session.flush()
    public_ids = [m.public_id for m in messages]
    await db_session.commit()
    return session.name, public_ids


def _create(
    content: str,
    *,
    session_name: str,
    level: str = "explicit",
    source_message_ids: list[str] | None = None,
    source_ids: list[str] | None = None,
) -> schemas.DocumentCreate:
    return schemas.DocumentCreate(
        content=content,
        session_name=session_name,
        level=level,  # pyright: ignore[reportArgumentType]
        metadata=schemas.DocumentMetadata(
            message_ids=[1], message_created_at="2026-01-01T00:00:00Z"
        ),
        embedding=[0.1] * 1536,
        source_message_ids=source_message_ids,
        source_ids=source_ids,
    )


async def _doc(db_session: AsyncSession, workspace_name: str, content: str):
    """Fetch a conclusion by its content.

    `populate_existing` matters after `expire_all()`: without it the ORM
    refreshes the expired instance with a separate query, which lazy-loads it
    outside the async greenlet and raises MissingGreenlet.
    """
    return (
        await db_session.execute(
            select(models.Document)
            .where(
                models.Document.workspace_name == workspace_name,
                models.Document.content == content,
            )
            .execution_options(populate_existing=True)
        )
    ).scalar_one()


async def _revisions(db_session: AsyncSession, workspace_name: str):
    # populate_existing: after a rollback the session's instances are expired,
    # and reading them would lazy-load outside the async greenlet.
    return list(
        (
            await db_session.execute(
                select(models.ConclusionRevision)
                .where(models.ConclusionRevision.workspace_name == workspace_name)
                .order_by(models.ConclusionRevision.id)
                .execution_options(populate_existing=True)
            )
        ).scalars()
    )


async def test_ledger_off_writes_nothing(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    session_name, ids = await _seed(
        db_session, workspace, peer, ["I moved to the infra team"]
    )
    await crud.create_documents(
        db_session,
        [
            _create(
                "I moved to the infra team",
                session_name=session_name,
                source_message_ids=ids,
            )
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()

    doc = await _doc(db_session, workspace.name, "I moved to the infra team")
    assert doc.assertion_basis is None
    assert doc.validity == "applicable"
    assert "derivation" not in (doc.internal_metadata or {})
    assert await _revisions(db_session, workspace.name) == []


@pytest.mark.usefixtures("ledger_on")
async def test_created_records_basis_validity_and_derivation(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    session_name, ids = await _seed(
        db_session, workspace, peer, ["I moved to the infra team"]
    )
    record = DerivationRecord(
        producer="deriver", model="m-1", prompt_version="abc", run_id="run-1"
    )
    with derivation_scope(record):
        await crud.create_documents(
            db_session,
            [
                _create(
                    "I moved to the infra team",
                    session_name=session_name,
                    source_message_ids=ids,
                ),
                _create(
                    "User works in infrastructure",
                    session_name=session_name,
                    source_message_ids=ids,
                ),
            ],
            workspace.name,
            observer=peer.name,
            observed=peer.name,
        )
    await db_session.commit()

    verbatim = await _doc(db_session, workspace.name, "I moved to the infra team")
    reading = await _doc(db_session, workspace.name, "User works in infrastructure")
    assert verbatim.assertion_basis == "declared_verbatim"
    assert reading.assertion_basis == "llm_reading"
    assert verbatim.internal_metadata["derivation"] == record.as_metadata()

    revisions = await _revisions(db_session, workspace.name)
    assert [r.kind for r in revisions] == ["created", "created"]
    assert {r.conclusion_id for r in revisions} == {verbatim.id, reading.id}
    assert all(r.derivation == record.as_metadata() for r in revisions)
    assert all(r.validity == "applicable" for r in revisions)


@pytest.mark.usefixtures("ledger_on")
async def test_contradiction_marks_sources_unknown(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings.CONCLUSION_LEDGER, "AUTO_RESOLVE_CONTRADICTIONS", True)
    workspace, peer = sample_data
    # Capture the name before expire_all(): after it, reading an expired
    # attribute lazy-loads it outside the async greenlet (MissingGreenlet).
    workspace_name = workspace.name
    session_name, ids = await _seed(db_session, workspace, peer, ["a", "b"])
    await crud.create_documents(
        db_session,
        [
            _create(
                "User is vegetarian",
                session_name=session_name,
                source_message_ids=ids[:1],
            ),
            _create(
                "User eats steak weekly",
                session_name=session_name,
                source_message_ids=ids[1:],
            ),
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    a = await _doc(db_session, workspace.name, "User is vegetarian")
    b = await _doc(db_session, workspace.name, "User eats steak weekly")

    with derivation_scope(DerivationRecord(producer="dreamer.deduction")):
        await crud.create_documents(
            db_session,
            [
                _create(
                    "User's statements about meat conflict",
                    session_name=session_name,
                    level="contradiction",
                    source_ids=[a.id, b.id],
                )
            ],
            workspace.name,
            observer=peer.name,
            observed=peer.name,
        )
    await db_session.commit()
    db_session.expire_all()

    a = await _doc(db_session, workspace_name, "User is vegetarian")
    b = await _doc(db_session, workspace_name, "User eats steak weekly")
    assert a.validity == b.validity == "unknown"
    contradiction = await _doc(
        db_session, workspace_name, "User's statements about meat conflict"
    )
    assert contradiction.assertion_basis == "llm_inference"

    revalidated = [
        r
        for r in await _revisions(db_session, workspace_name)
        if r.kind == "revalidated"
    ]
    assert {r.conclusion_id for r in revalidated} == {a.id, b.id}
    assert all(r.previous_validity == "applicable" for r in revalidated)
    assert all(r.reason == "contradiction" for r in revalidated)


@pytest.mark.usefixtures("ledger_on")
async def test_set_validity_and_history(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    session_name, ids = await _seed(db_session, workspace, peer, ["x"])
    await crud.create_documents(
        db_session,
        [
            _create(
                "User lives in Pelotas",
                session_name=session_name,
                source_message_ids=ids,
            )
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    doc = await _doc(db_session, workspace.name, "User lives in Pelotas")

    await conclusion_ledger.set_conclusion_validity(
        db_session, workspace.name, doc.id, "historical", reason="moved"
    )
    await db_session.commit()

    history = await conclusion_ledger.get_conclusion_history(
        db_session, workspace.name, doc.id
    )
    assert [r.kind for r in history] == ["created", "revalidated"]
    assert history[-1].validity == "historical"
    assert history[-1].previous_validity == "applicable"
    assert history[-1].reason == "moved"

    with pytest.raises(ValidationException):
        await conclusion_ledger.set_conclusion_validity(
            db_session, workspace.name, doc.id, "maybe"
        )


@pytest.mark.usefixtures("ledger_on")
async def test_dreamer_removal_is_kept_user_deletion_erases(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    session_name, ids = await _seed(db_session, workspace, peer, ["x", "y"])
    await crud.create_documents(
        db_session,
        [
            _create(
                "Fact one about the user",
                session_name=session_name,
                source_message_ids=ids[:1],
            ),
            _create(
                "Fact two about the user",
                session_name=session_name,
                source_message_ids=ids[1:],
            ),
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    one = await _doc(db_session, workspace.name, "Fact one about the user")
    two = await _doc(db_session, workspace.name, "Fact two about the user")

    # A writer's removal is recorded and kept.
    await crud.delete_documents(
        db_session,
        workspace.name,
        [one.id],
        observer=peer.name,
        observed=peer.name,
    )
    history = await conclusion_ledger.get_conclusion_history(
        db_session, workspace.name, one.id
    )
    assert [r.kind for r in history] == ["created", "removed"]
    assert history[-1].content == "Fact one about the user"

    # A user-requested deletion erases the conclusion's revisions.
    await crud.delete_document_by_id(db_session, workspace.name, two.id)
    assert (
        await conclusion_ledger.get_conclusion_history(
            db_session, workspace.name, two.id
        )
        == []
    )


@pytest.mark.usefixtures("ledger_on")
async def test_session_deletion_erases_its_revisions(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    session_name, ids = await _seed(db_session, workspace, peer, ["x"])
    await crud.create_documents(
        db_session,
        [
            _create(
                "Fact from a session", session_name=session_name, source_message_ids=ids
            )
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    assert len(await _revisions(db_session, workspace.name)) == 1

    await conclusion_ledger.delete_session_revisions(
        db_session, workspace.name, session_name
    )
    await db_session.commit()
    assert await _revisions(db_session, workspace.name) == []


@pytest.fixture
def most_recent_resolution() -> Iterator[None]:
    original = settings.CONCLUSION_LEDGER.CONTRADICTION_RESOLUTION
    settings.CONCLUSION_LEDGER.CONTRADICTION_RESOLUTION = "most_recent"
    try:
        yield
    finally:
        settings.CONCLUSION_LEDGER.CONTRADICTION_RESOLUTION = original


def _create_at(
    content: str, *, session_name: str, at: str, source_message_ids: list[str]
) -> schemas.DocumentCreate:
    doc = _create(
        content, session_name=session_name, source_message_ids=source_message_ids
    )
    doc.metadata.message_created_at = at
    # Explicit world-time evidence supplied by this test, not inferred from message date.
    doc.metadata.temporal = TemporalValidity.model_validate(
        {"temporal_class": "mutable", "valid_from": at}
    )
    return doc


@pytest.mark.usefixtures("ledger_on")
async def test_spans_and_window_on_created(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    session_name, ids = await _seed(
        db_session, workspace, peer, ["Hi. I have moved to the infra team."]
    )
    await crud.create_documents(
        db_session,
        [
            _create_at(
                "I moved to the infra team",
                session_name=session_name,
                at="2026-03-01T12:00:00Z",
                source_message_ids=ids,
            )
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    doc = await _doc(db_session, workspace.name, "I moved to the infra team")
    assert doc.assertion_basis == "llm_reading"
    assert doc.declared_spans is not None
    assert doc.declared_spans[0]["message_id"] == ids[0]
    assert "text" not in doc.declared_spans[0]
    assert len(doc.declared_spans[0]["sha256"]) == 64
    assert doc.valid_from is not None and doc.valid_from.isoformat().startswith(
        "2026-03-01T12:00:00"
    )
    assert doc.valid_until is None


@pytest.mark.usefixtures("ledger_on")
async def test_exact_duplicate_is_recorded_as_reinforced(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    session_name, ids = await _seed(db_session, workspace, peer, ["x", "y"])
    for mid in ids:
        await crud.create_documents(
            db_session,
            [
                _create(
                    "User likes green tea",
                    session_name=session_name,
                    source_message_ids=[mid],
                )
            ],
            workspace.name,
            observer=peer.name,
            observed=peer.name,
        )
        await db_session.commit()
    doc = await _doc(db_session, workspace.name, "User likes green tea")
    history = await conclusion_ledger.get_conclusion_history(
        db_session, workspace.name, doc.id
    )
    assert [r.kind for r in history] == ["created", "reinforced"]
    assert history[-1].times_derived == 2


@pytest.mark.usefixtures("ledger_on", "most_recent_resolution")
async def test_contradiction_most_recent_resolves(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings.CONCLUSION_LEDGER, "AUTO_RESOLVE_CONTRADICTIONS", True)
    workspace, peer = sample_data
    # Capture the name before expire_all(): after it, reading an expired
    # attribute lazy-loads it outside the async greenlet (MissingGreenlet).
    workspace_name = workspace.name
    session_name, ids = await _seed(db_session, workspace, peer, ["a", "b"])
    await crud.create_documents(
        db_session,
        [
            _create_at(
                "User is vegetarian",
                session_name=session_name,
                at="2026-01-01T00:00:00Z",
                source_message_ids=ids[:1],
            ),
            _create_at(
                "User eats steak weekly",
                session_name=session_name,
                at="2026-06-01T00:00:00Z",
                source_message_ids=ids[1:],
            ),
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    old = await _doc(db_session, workspace.name, "User is vegetarian")
    new = await _doc(db_session, workspace.name, "User eats steak weekly")
    await crud.create_documents(
        db_session,
        [
            _create(
                "Statements about meat conflict",
                session_name=session_name,
                level="contradiction",
                source_ids=[old.id, new.id],
            )
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    db_session.expire_all()
    old = await _doc(db_session, workspace_name, "User is vegetarian")
    new = await _doc(db_session, workspace_name, "User eats steak weekly")
    assert old.validity == "incompatible"
    assert old.valid_until is None
    assert new.validity == "applicable"


@pytest.mark.usefixtures("ledger_on")
async def test_retrieval_excludes_non_current_unless_asked(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings.CONCLUSION_LEDGER, "RETRIEVAL_FILTER_ENABLED", True)
    workspace, peer = sample_data
    session_name, ids = await _seed(db_session, workspace, peer, ["x", "y"])
    await crud.create_documents(
        db_session,
        [
            _create(
                "User lives in Pelotas",
                session_name=session_name,
                source_message_ids=ids[:1],
            ),
            _create(
                "User lives in Recife",
                session_name=session_name,
                source_message_ids=ids[1:],
            ),
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    pelotas = await _doc(db_session, workspace.name, "User lives in Pelotas")
    await conclusion_ledger.set_conclusion_validity(
        db_session, workspace.name, pelotas.id, "historical", reason="moved"
    )
    await db_session.commit()

    current = await crud.query_documents_recent(
        db_session, workspace.name, observer=peer.name, observed=peer.name
    )
    assert {d.content for d in current} == {"User lives in Recife"}
    everything = await crud.query_documents_recent(
        db_session,
        workspace.name,
        observer=peer.name,
        observed=peer.name,
        include_non_current=True,
    )
    assert {d.content for d in everything} == {
        "User lives in Pelotas",
        "User lives in Recife",
    }


@pytest.mark.usefixtures("ledger_on")
async def test_set_validity_window(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    import datetime

    workspace, peer = sample_data
    session_name, ids = await _seed(db_session, workspace, peer, ["x"])
    await crud.create_documents(
        db_session,
        [
            _create(
                "User works at a bakery",
                session_name=session_name,
                source_message_ids=ids,
            )
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    doc = await _doc(db_session, workspace.name, "User works at a bakery")

    until = datetime.datetime(2026, 5, 1, tzinfo=datetime.UTC)
    await conclusion_ledger.set_conclusion_validity(
        db_session, workspace.name, doc.id, "historical", valid_until=until
    )
    await db_session.commit()
    assert doc.valid_until == until

    # An explicit clear is required; status changes do not fabricate world time.
    await conclusion_ledger.set_conclusion_validity(
        db_session, workspace.name, doc.id, "applicable", valid_until=None
    )
    await db_session.commit()
    assert doc.valid_until is None

    with pytest.raises(ValidationException):
        await conclusion_ledger.set_conclusion_validity(
            db_session,
            workspace.name,
            doc.id,
            "historical",
            valid_from=datetime.datetime(2026, 6, 1, tzinfo=datetime.UTC),
            valid_until=datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC),
        )


@pytest.mark.usefixtures("ledger_on")
async def test_superseded_chain_is_followed(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """Only explicit opt-in promotes a replaced row to historical validity."""
    monkeypatch.setattr(settings.CONCLUSION_LEDGER, "MARK_SUPERSEDED_HISTORICAL", True)
    workspace, peer = sample_data
    session_name, ids = await _seed(db_session, workspace, peer, ["x", "y"])
    await crud.create_documents(
        db_session,
        [
            _create_at(
                "User drinks coffee",
                session_name=session_name,
                at="2026-01-01T00:00:00Z",
                source_message_ids=ids[:1],
            )
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    old = await _doc(db_session, workspace.name, "User drinks coffee")
    await crud.create_documents(
        db_session,
        [
            _create_at(
                "User drinks coffee daily",
                session_name=session_name,
                at="2026-02-01T00:00:00Z",
                source_message_ids=ids[1:],
            )
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    new = await _doc(db_session, workspace.name, "User drinks coffee daily")

    # Supersede directly, as semantic dedup does inside create_documents.
    await conclusion_ledger.record_created(
        db_session, [], replaced=[old], replaced_by={old.id: new}
    )
    await db_session.commit()
    assert old.validity == "historical"
    assert old.valid_until is None

    history = await conclusion_ledger.get_conclusion_history(
        db_session, workspace.name, old.id
    )
    assert [(r.conclusion_id, r.kind) for r in history] == [
        (old.id, "created"),
        (old.id, "superseded"),
        (new.id, "created"),
    ]


@pytest.mark.usefixtures("ledger_on")
async def test_missing_valid_from_and_foreign_author(db_session, sample_data):
    workspace, peer = sample_data
    claim = "I live in Pelotas."
    session_name, ids = await _seed(db_session, workspace, peer, [claim])
    other = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add(other)
    await db_session.flush()
    db_session.add(
        models.Collection(
            workspace_name=workspace.name, observer=peer.name, observed=other.name
        )
    )
    await db_session.commit()
    await crud.create_documents(
        db_session,
        [_create(claim, session_name=session_name, source_message_ids=ids)],
        workspace.name,
        observer=peer.name,
        observed=other.name,
    )
    await db_session.commit()
    doc = await _doc(db_session, workspace.name, claim)
    assert doc.assertion_basis == "llm_reading"
    assert doc.declared_spans is None
    assert doc.valid_from is None
    assert doc.temporal_class == "unknown"


@pytest.mark.usefixtures("ledger_on")
async def test_every_candidate_keeps_original_content_and_citations(
    db_session, sample_data
):
    workspace, peer = sample_data
    session_name, ids = await _seed(db_session, workspace, peer, ["a", "b", "c"])
    claim = "User likes green tea"
    await crud.create_documents(
        db_session,
        [
            _create(claim, session_name=session_name, source_message_ids=[mid])
            for mid in ids
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    rows = await _revisions(db_session, workspace.name)
    assert len(rows) == 3
    assert [r.kind for r in rows] == ["created", "reinforced", "reinforced"]
    candidates = [r.candidate for r in rows]
    assert all(candidate is not None for candidate in candidates)
    assert [
        candidate["source_message_ids"]
        for candidate in candidates
        if candidate is not None
    ] == [[i] for i in ids]
    assert all(
        candidate["content"] == claim
        for candidate in candidates
        if candidate is not None
    )


@pytest.mark.usefixtures("ledger_on")
async def test_expired_and_future_windows_are_excluded(
    db_session, sample_data, monkeypatch
):
    monkeypatch.setattr(settings.CONCLUSION_LEDGER, "RETRIEVAL_FILTER_ENABLED", True)
    workspace, peer = sample_data
    session_name, ids = await _seed(db_session, workspace, peer, ["a", "b"])
    expired = _create(
        "Expired trip", session_name=session_name, source_message_ids=ids[:1]
    )
    future = _create(
        "Future trip", session_name=session_name, source_message_ids=ids[1:]
    )
    expired.metadata.temporal = TemporalValidity.model_validate(
        {"temporal_class": "temporary", "valid_to": "2000-01-01T00:00:00Z"}
    )
    future.metadata.temporal = TemporalValidity.model_validate(
        {"temporal_class": "prospective", "valid_from": "2999-01-01T00:00:00Z"}
    )
    await crud.create_documents(
        db_session,
        [expired, future],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    assert (
        list(
            await crud.query_documents_recent(
                db_session, workspace.name, observer=peer.name, observed=peer.name
            )
        )
        == []
    )
    assert (
        len(
            await crud.query_documents_recent(
                db_session,
                workspace.name,
                observer=peer.name,
                observed=peer.name,
                include_non_current=True,
            )
        )
        == 2
    )


@pytest.mark.usefixtures("ledger_on")
async def test_manual_api_creation_records_revision(
    db_session, sample_data, monkeypatch
):
    from unittest.mock import AsyncMock

    from src.crud import document

    workspace, peer = sample_data
    await _seed(db_session, workspace, peer, [])
    monkeypatch.setattr(
        document.embedding_client,
        "simple_batch_embed",
        AsyncMock(return_value=[[0.1] * 1536]),
    )
    rows = await crud.create_observations(
        db_session,
        [
            schemas.ConclusionCreate(
                content="Manual API assertion",
                observer_id=peer.name,
                observed_id=peer.name,
            )
        ],
        workspace.name,
    )
    assert rows[0].assertion_basis == "api_supplied"
    revisions = await conclusion_ledger.get_conclusion_history(
        db_session, workspace.name, rows[0].id
    )
    assert len(revisions) == 1
    derivation = revisions[0].derivation
    assert derivation is not None
    assert derivation["producer"] == "api"


@pytest.mark.usefixtures("ledger_on")
async def test_tombstoned_session_hides_history(db_session, sample_data):
    workspace, peer = sample_data
    session_name, ids = await _seed(db_session, workspace, peer, ["a"])
    await crud.create_documents(
        db_session,
        [
            _create(
                "Historical assertion",
                session_name=session_name,
                source_message_ids=ids,
            )
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    doc = await _doc(db_session, workspace.name, "Historical assertion")
    session = (
        await db_session.scalars(
            select(models.Session).where(
                models.Session.workspace_name == workspace.name,
                models.Session.name == session_name,
            )
        )
    ).one()
    session.deleted_at = datetime.datetime.now(datetime.UTC)
    await db_session.commit()
    assert (
        await conclusion_ledger.get_conclusion_history(
            db_session, workspace.name, doc.id
        )
        == []
    )


@pytest.mark.usefixtures("ledger_on")
async def test_semantic_rejection_saves_original_candidate(
    db_session, sample_data, monkeypatch
):
    from src.crud import document

    workspace, peer = sample_data
    session_name, ids = await _seed(db_session, workspace, peer, ["a", "b"])
    old_text = "User drinks green tea every morning"
    new_text = "User drinks tea"
    await crud.create_documents(
        db_session,
        [_create(old_text, session_name=session_name, source_message_ids=ids[:1])],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    old = await _doc(db_session, workspace.name, old_text)
    monkeypatch.setattr(
        document,
        "_resolve_semantic_dup",
        lambda *args: (document.SemanticRejectionResult.REJECTED, old),
    )
    await crud.create_documents(
        db_session,
        [_create(new_text, session_name=session_name, source_message_ids=ids[1:])],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
        deduplicate=True,
    )
    await db_session.commit()
    revisions = await conclusion_ledger.get_conclusion_history(
        db_session, workspace.name, old.id
    )
    assert revisions[-1].content == old_text
    candidate = revisions[-1].candidate
    assert candidate is not None
    assert candidate["content"] == new_text
    assert candidate["source_message_ids"] == ids[1:]
    assert revisions[-1].reason == "semantic_duplicate_rejected"


@pytest.mark.usefixtures("ledger_on")
async def test_session_erasure_keeps_independent_reinforcement(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
) -> None:
    """Same canonical conclusion reinforced in another session survives erasure."""
    workspace, peer = sample_data
    session_a, _ = await _seed(db_session, workspace, peer, ["A"])
    session_b = str(generate_nanoid())
    db_session.add(models.Session(name=session_b, workspace_name=workspace.name))
    await db_session.flush()

    # Derived levels can dedup across sessions in the unchanged Honcho policy.
    await crud.create_documents(
        db_session,
        [
            _create(
                "Independent repeated insight",
                session_name=session_a,
                level="deductive",
            )
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    await crud.create_documents(
        db_session,
        [
            _create(
                "Independent repeated insight",
                session_name=session_b,
                level="deductive",
            )
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    doc = await _doc(db_session, workspace.name, "Independent repeated insight")
    revisions = await conclusion_ledger.get_conclusion_history(
        db_session, workspace.name, doc.id
    )
    assert [revision.kind for revision in revisions] == ["created", "reinforced"]

    await conclusion_ledger.delete_session_revisions(
        db_session, workspace.name, session_a
    )
    await db_session.commit()
    # The canonical row's session is A. The retained event's candidate session
    # is B and must therefore remain readable after A is tombstoned.
    session = (
        await db_session.scalars(
            select(models.Session).where(models.Session.name == session_a)
        )
    ).one()
    session.deleted_at = datetime.datetime.now(datetime.UTC)
    await db_session.commit()

    surviving = await conclusion_ledger.get_conclusion_history(
        db_session, workspace.name, doc.id
    )
    assert [revision.kind for revision in surviving] == ["reinforced"]
    assert surviving[0].candidate is not None
    assert surviving[0].candidate["session_name"] == session_b


@pytest.mark.usefixtures("ledger_on")
async def test_candidate_exposes_authored_utterance_references(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
) -> None:
    workspace, peer = sample_data
    session, ids = await _seed(
        db_session, workspace, peer, ["I moved to the infra team"]
    )
    await crud.create_documents(
        db_session,
        [
            _create(
                "alice moved to the infra team",
                session_name=session,
                source_message_ids=ids,
            )
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    doc = await _doc(db_session, workspace.name, "alice moved to the infra team")
    assert doc.assertion_basis == "llm_reading"
    history = await conclusion_ledger.get_conclusion_history(
        db_session, workspace.name, doc.id
    )
    assert len(history) == 1 and history[0].candidate is not None
    pointers = history[0].candidate["source_utterances"]
    assert len(pointers) == 1
    assert pointers[0]["message_id"] == ids[0]
    assert pointers[0]["start"] == 0
    assert pointers[0]["end"] == len("I moved to the infra team")
    assert len(pointers[0]["sha256"]) == 64
    assert "text" not in pointers[0]


# ---------------------------------------------------------------------------
# Session erasure: which events belong to a session, and which conclusions
# were originated there. A conclusion merely reinforced in a session was not
# originated there, and erasing that session must not take another session's
# independent derivations with it.
# ---------------------------------------------------------------------------


async def _second_session(db_session: AsyncSession, workspace_name: str) -> str:
    name = str(generate_nanoid())
    db_session.add(models.Session(name=name, workspace_name=workspace_name))
    await db_session.flush()
    return name


@pytest.mark.usefixtures("ledger_on")
async def test_erasure_keeps_other_sessions_derivation(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
) -> None:
    """T01: origin A, reinforcement B, derivation A.

    Erasing B removes only the event B produced. The conclusion C originated
    in A, and the conclusion D that A derived from C, both survive: B touched
    C, it did not originate it.
    """
    workspace, peer = sample_data
    session_a, _ = await _seed(db_session, workspace, peer, ["A"])
    session_b = await _second_session(db_session, workspace.name)

    # C: originated in A.
    await crud.create_documents(
        db_session,
        [_create("Shared insight", session_name=session_a, level="deductive")],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    shared = await _doc(db_session, workspace.name, "Shared insight")

    # D: derived from C, also in A.
    await crud.create_documents(
        db_session,
        [
            _create(
                "Insight derived from the shared one",
                session_name=session_a,
                level="deductive",
                source_ids=[shared.id],
            )
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    derived = await _doc(
        db_session, workspace.name, "Insight derived from the shared one"
    )

    # B repeats C: the same content dedups across sessions.
    await crud.create_documents(
        db_session,
        [_create("Shared insight", session_name=session_b, level="deductive")],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()

    history = await conclusion_ledger.get_conclusion_history(
        db_session, workspace.name, shared.id
    )
    assert [revision.kind for revision in history] == ["created", "reinforced"]

    await conclusion_ledger.delete_session_revisions(
        db_session, workspace.name, session_b
    )
    await db_session.commit()

    surviving = await conclusion_ledger.get_conclusion_history(
        db_session, workspace.name, shared.id
    )
    assert [revision.kind for revision in surviving] == ["created"]
    # D was derived in A from a conclusion A originated: B never touched it.
    derived_history = await conclusion_ledger.get_conclusion_history(
        db_session, workspace.name, derived.id
    )
    assert [revision.kind for revision in derived_history] == ["created"]


@pytest.mark.usefixtures("ledger_on")
async def test_erasure_removes_derivations_of_erased_origin(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
) -> None:
    """T02: origin B, derivation A.

    A conclusion originated in the erased session goes away with it, and so
    does the event that directly depended on it -- that is the documented
    erasure policy for derived data.
    """
    workspace, peer = sample_data
    session_a, _ = await _seed(db_session, workspace, peer, ["A"])
    session_b = await _second_session(db_session, workspace.name)

    await crud.create_documents(
        db_session,
        [_create("Erasable origin", session_name=session_b, level="deductive")],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    origin = await _doc(db_session, workspace.name, "Erasable origin")

    await crud.create_documents(
        db_session,
        [
            _create(
                "Derived from the erasable origin",
                session_name=session_a,
                level="deductive",
                source_ids=[origin.id],
            )
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    derived = await _doc(db_session, workspace.name, "Derived from the erasable origin")

    assert len(await _revisions(db_session, workspace.name)) == 2

    await conclusion_ledger.delete_session_revisions(
        db_session, workspace.name, session_b
    )
    await db_session.commit()

    assert await _revisions(db_session, workspace.name) == []
    assert (
        await conclusion_ledger.get_conclusion_history(
            db_session, workspace.name, derived.id
        )
        == []
    )


@pytest.mark.usefixtures("ledger_on")
async def test_tombstoned_session_hides_only_its_own_candidate(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
) -> None:
    """T03: a candidate from a tombstoned session is not exposed."""
    workspace, peer = sample_data
    session_a, _ = await _seed(db_session, workspace, peer, ["A"])
    session_b = await _second_session(db_session, workspace.name)

    await crud.create_documents(
        db_session,
        [_create("Tombstone candidate", session_name=session_a, level="deductive")],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    shared = await _doc(db_session, workspace.name, "Tombstone candidate")
    await crud.create_documents(
        db_session,
        [_create("Tombstone candidate", session_name=session_b, level="deductive")],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()

    session = (
        await db_session.scalars(
            select(models.Session).where(models.Session.name == session_b)
        )
    ).one()
    session.deleted_at = datetime.datetime.now(datetime.UTC)
    await db_session.commit()

    history = await conclusion_ledger.get_conclusion_history(
        db_session, workspace.name, shared.id
    )
    assert [revision.kind for revision in history] == ["created"]
    assert all(
        (revision.candidate or {}).get("session_name") != session_b
        for revision in history
    )


@pytest.mark.usefixtures("ledger_on")
async def test_erasure_without_source_ids_removes_only_its_own(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
) -> None:
    """T04: an event with no source reference is erased with its session.

    Nothing else is pulled in: a conclusion with no declared source cannot
    depend on the erased session, and the limit is explicit -- unlinked
    derived claims are not discoverable from the ledger.
    """
    workspace, peer = sample_data
    session_a, _ = await _seed(db_session, workspace, peer, ["A"])
    session_b = await _second_session(db_session, workspace.name)

    await crud.create_documents(
        db_session,
        [
            _create(
                "Erased source-less claim", session_name=session_b, level="deductive"
            )
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    await crud.create_documents(
        db_session,
        [
            _create(
                "Unrelated surviving claim", session_name=session_a, level="deductive"
            )
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()

    await conclusion_ledger.delete_session_revisions(
        db_session, workspace.name, session_b
    )
    await db_session.commit()

    remaining = await _revisions(db_session, workspace.name)
    assert len(remaining) == 1
    assert remaining[0].content == "Unrelated surviving claim"


@pytest.mark.usefixtures("ledger_on")
async def test_erasure_is_scoped_to_the_current_tenant(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T05: another tenant's rows keep their own session, with the same name."""
    from src.db import tenant_context

    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    tenant_a = str(generate_nanoid())
    tenant_b = str(generate_nanoid())
    db_session.add_all(
        [
            models.Tenant(tenant_id=tenant_a, vector_correlation_id="tenant-a-app"),
            models.Tenant(tenant_id=tenant_b, vector_correlation_id="tenant-b-app"),
        ]
    )
    await db_session.commit()

    workspace_name = str(generate_nanoid())
    shared_session_name = str(generate_nanoid())
    for tenant in (tenant_a, tenant_b):
        token = tenant_context.set(tenant)
        try:
            db_session.add(models.Workspace(name=workspace_name, tenant_id=tenant))
            await db_session.flush()
            peer_name = str(generate_nanoid())
            db_session.add(
                models.Peer(
                    name=peer_name,
                    workspace_name=workspace_name,
                    tenant_id=tenant,
                )
            )
            await db_session.commit()
            db_session.add(
                models.Collection(
                    workspace_name=workspace_name,
                    observer=peer_name,
                    observed=peer_name,
                    tenant_id=tenant,
                )
            )
            db_session.add(
                models.Session(
                    name=shared_session_name,
                    workspace_name=workspace_name,
                    tenant_id=tenant,
                )
            )
            await db_session.commit()
            await crud.create_documents(
                db_session,
                [
                    _create(
                        f"Tenant {tenant} claim",
                        session_name=shared_session_name,
                        level="deductive",
                    )
                ],
                workspace_name,
                observer=peer_name,
                observed=peer_name,
            )
            await db_session.commit()
        finally:
            tenant_context.reset(token)

    async def _rows_of(tenant: str) -> list[models.ConclusionRevision]:
        return list(
            (
                await db_session.execute(
                    select(models.ConclusionRevision)
                    .where(models.ConclusionRevision.tenant_id == tenant)
                    .order_by(models.ConclusionRevision.id)
                    .execution_options(populate_existing=True)
                )
            ).scalars()
        )

    assert len(await _rows_of(tenant_a)) == 1
    assert len(await _rows_of(tenant_b)) == 1

    token = tenant_context.set(tenant_a)
    try:
        await conclusion_ledger.delete_session_revisions(
            db_session, workspace_name, shared_session_name
        )
        await db_session.commit()
    finally:
        tenant_context.reset(token)

    # Tenant A's event is gone; tenant B's identically named session is not.
    assert await _rows_of(tenant_a) == []
    remaining = await _rows_of(tenant_b)
    assert len(remaining) == 1
    assert remaining[0].content == f"Tenant {tenant_b} claim"


@pytest.mark.usefixtures("ledger_on")
async def test_erasure_rolls_back_with_its_transaction(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
) -> None:
    """T06: an exception before the commit restores every event."""
    workspace, peer = sample_data
    workspace_name = workspace.name
    session_a, _ = await _seed(db_session, workspace, peer, ["A"])
    session_b = await _second_session(db_session, workspace.name)

    await crud.create_documents(
        db_session,
        [_create("Rolled back claim", session_name=session_b, level="deductive")],
        workspace_name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    assert len(await _revisions(db_session, workspace_name)) == 1

    await conclusion_ledger.delete_session_revisions(
        db_session, workspace_name, session_b
    )
    await db_session.rollback()

    # The rollback expired the ORM state: read the rows with a fresh query.
    surviving = await _revisions(db_session, workspace_name)
    assert len(surviving) == 1
    assert surviving[0].content == "Rolled back claim"
    assert surviving[0].session_name == session_b
