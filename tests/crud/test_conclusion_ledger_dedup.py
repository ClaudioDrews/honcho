"""Temporal dedup: the candidate search must see the temporally matching rows.

The nearest neighbour overall is not necessarily the same claim. When the
candidate search limited itself to that neighbour before comparing the
temporal key, the nearest neighbour that *was* the same claim was never
looked at, and a duplicate was recorded as a new conclusion -- the history
fragmented exactly where the temporal classes were meant to hold it together.

The key comparison is null-safe on every part: an unset bound is an open
bound on both sides, and a plain SQL equality would drop those rows.
"""

from collections.abc import Iterator

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.crud.document import SemanticRejectionResult, _semantic_dup_decision
from src.models import Peer, Workspace
from src.utils.interpretation import TemporalValidity

DIM = settings.EMBEDDING.VECTOR_DIMENSIONS

# Two directions a few thousandths apart: within the 0.05 dedup radius of one
# another, with EMBED_NEAR strictly nearer to the query than EMBED_FAR.
EMBED_NEAR = [1.0] + [0.0] * (DIM - 1)
EMBED_FAR = [0.995, 0.0999] + [0.0] * (DIM - 2)

HISTORICAL = {"temporal_class": "historical"}
MUTABLE = {"temporal_class": "mutable"}
PROSPECTIVE = {"temporal_class": "prospective"}

HISTORICAL_CONTENT = "The team ships on Fridays"
MUTABLE_CONTENT = "Fridays are shipping days for this team"
# The mutable claim plus one word: strictly richer in tokens, so the incoming
# document wins the comparison whatever the tokenizer does with the words.
INCOMING = "Fridays are shipping days for this team always"


@pytest.fixture
def ledger_on() -> Iterator[None]:
    original = settings.CONCLUSION_LEDGER.ENABLED
    settings.CONCLUSION_LEDGER.ENABLED = True
    try:
        yield
    finally:
        settings.CONCLUSION_LEDGER.ENABLED = original


@pytest.fixture
def temporal_dedup_on() -> Iterator[None]:
    original = settings.CONCLUSION_LEDGER.TEMPORAL_DEDUP_ENABLED
    settings.CONCLUSION_LEDGER.TEMPORAL_DEDUP_ENABLED = True
    try:
        yield
    finally:
        settings.CONCLUSION_LEDGER.TEMPORAL_DEDUP_ENABLED = original


async def _seed(
    db_session: AsyncSession, workspace: Workspace, peer: Peer
) -> tuple[str, list[str]]:
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add(session)
    await db_session.flush()
    message = models.Message(
        session_name=session.name,
        workspace_name=workspace.name,
        peer_name=peer.name,
        content=HISTORICAL_CONTENT,
        seq_in_session=1,
    )
    db_session.add(message)
    db_session.add(
        models.Collection(
            workspace_name=workspace.name, observer=peer.name, observed=peer.name
        )
    )
    await db_session.flush()
    await db_session.commit()
    return session.name, [message.public_id]


def _doc(
    content: str,
    *,
    session_name: str,
    temporal: dict[str, str],
    embedding: list[float],
) -> schemas.DocumentCreate:
    doc = schemas.DocumentCreate(
        content=content,
        session_name=session_name,
        level="explicit",
        metadata=schemas.DocumentMetadata(
            message_ids=[1], message_created_at="2026-01-01T00:00:00Z"
        ),
        embedding=embedding,
    )
    doc.metadata.temporal = TemporalValidity.model_validate(temporal)
    return doc


async def _live_documents(
    db_session: AsyncSession, workspace_name: str
) -> list[models.Document]:
    return list(
        (
            await db_session.execute(
                select(models.Document)
                .where(
                    models.Document.workspace_name == workspace_name,
                    models.Document.deleted_at.is_(None),
                )
                .order_by(models.Document.id)
                .execution_options(populate_existing=True)
            )
        ).scalars()
    )


async def _plant_pair(
    db_session: AsyncSession,
    workspace: Workspace,
    peer: Peer,
    session_name: str,
) -> tuple[models.Document, models.Document]:
    """A historical claim and a mutable one, both about the same thing.

    They differ in content, so the exact-match dedup keeps them apart, and in
    embedding, so the vector search can put one of them strictly nearer.
    """
    await crud.create_documents(
        db_session,
        [
            _doc(
                HISTORICAL_CONTENT,
                session_name=session_name,
                temporal=HISTORICAL,
                embedding=EMBED_NEAR,
            ),
            _doc(
                MUTABLE_CONTENT,
                session_name=session_name,
                temporal=MUTABLE,
                embedding=EMBED_FAR,
            ),
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    await db_session.commit()
    documents = await _live_documents(db_session, workspace.name)
    assert len(documents) == 2
    by_content = {document.content: document for document in documents}
    return (
        by_content[HISTORICAL_CONTENT],
        by_content[MUTABLE_CONTENT],
    )


async def _create_incoming(
    db_session: AsyncSession,
    workspace: Workspace,
    peer: Peer,
    session_name: str,
    *,
    temporal: dict[str, str],
) -> None:
    await crud.create_documents(
        db_session,
        [
            _doc(
                INCOMING,
                session_name=session_name,
                temporal=temporal,
                embedding=EMBED_NEAR,
            )
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
        deduplicate=True,
    )
    await db_session.commit()


@pytest.mark.usefixtures("ledger_on", "temporal_dedup_on")
async def test_dedup_finds_the_temporally_matching_neighbour(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
) -> None:
    """T07: the nearest neighbour is not the same claim.

    The incoming claim is mutable, like the second planted document and
    unlike the first, which is nearer to it in embedding space. It must be
    reconciled with the mutable one, not left as a third conclusion.
    """
    workspace, peer = sample_data
    session_name, _ = await _seed(db_session, workspace, peer)
    historical, _ = await _plant_pair(db_session, workspace, peer, session_name)

    await _create_incoming(db_session, workspace, peer, session_name, temporal=MUTABLE)

    documents = await _live_documents(db_session, workspace.name)
    contents = {document.content for document in documents}
    assert len(documents) == 2
    # The richer text replaced the mutable row; the historical one is untouched.
    assert INCOMING in contents
    assert HISTORICAL_CONTENT in contents
    assert MUTABLE_CONTENT not in contents
    by_content = {document.content: document for document in documents}
    assert by_content[HISTORICAL_CONTENT].id == historical.id
    assert by_content[HISTORICAL_CONTENT].times_derived == 1


@pytest.mark.usefixtures("ledger_on", "temporal_dedup_on")
async def test_dedup_matches_open_bounds_null_safely(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
) -> None:
    """T08: an unset bound is an open bound, on both sides.

    The temporal keys of the planted pair share open bounds. A plain SQL
    equality would compare NULL to NULL as unknown and drop every candidate.
    """
    workspace, peer = sample_data
    session_name, _ = await _seed(db_session, workspace, peer)
    await _plant_pair(db_session, workspace, peer, session_name)

    await _create_incoming(db_session, workspace, peer, session_name, temporal=MUTABLE)

    documents = await _live_documents(db_session, workspace.name)
    assert len(documents) == 2
    contents = {document.content for document in documents}
    assert INCOMING in contents
    assert MUTABLE_CONTENT not in contents


@pytest.mark.usefixtures("ledger_on", "temporal_dedup_on")
async def test_dedup_keeps_claims_with_different_temporal_keys(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
) -> None:
    """T09: the temporal key still separates claims.

    A prospective claim is not the mutable claim, and not the historical one.
    Narrowing the candidates must not merge claims that differ in time.
    """
    workspace, peer = sample_data
    session_name, _ = await _seed(db_session, workspace, peer)
    await _plant_pair(db_session, workspace, peer, session_name)

    await _create_incoming(
        db_session, workspace, peer, session_name, temporal=PROSPECTIVE
    )

    documents = await _live_documents(db_session, workspace.name)
    assert len(documents) == 3
    assert INCOMING in {document.content for document in documents}


@pytest.mark.usefixtures("ledger_on")
async def test_dedup_without_the_ledger_ignores_the_temporal_key(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
) -> None:
    """T10: with the ledger off, nothing changes.

    The temporal key is not part of what makes two documents the same claim
    when the ledger is off, so the nearest neighbour wins -- the historical
    one, which is nearer -- and the mutable one is left alone.
    """
    workspace, peer = sample_data
    session_name, _ = await _seed(db_session, workspace, peer)
    await _plant_pair(db_session, workspace, peer, session_name)

    await _create_incoming(db_session, workspace, peer, session_name, temporal=MUTABLE)

    documents = await _live_documents(db_session, workspace.name)
    contents = {document.content for document in documents}
    assert len(documents) == 2
    assert INCOMING in contents
    # The nearest neighbour was the historical one: it is the one that moved.
    assert MUTABLE_CONTENT in contents
    assert HISTORICAL_CONTENT not in contents
    by_content = {document.content: document for document in documents}
    assert by_content[MUTABLE_CONTENT].times_derived == 1


@pytest.mark.usefixtures("ledger_on", "temporal_dedup_on")
async def test_single_document_dedup_picks_the_matching_claim(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
) -> None:
    """T11: the one-document path resolves the same candidate.

    The same narrowing applies when a single document is classified on its
    own, not only when a batch is resolved together.
    """
    workspace, peer = sample_data
    session_name, _ = await _seed(db_session, workspace, peer)
    _historical, mutable = await _plant_pair(db_session, workspace, peer, session_name)

    result, existing = await _semantic_dup_decision(
        db_session,
        _doc(
            INCOMING,
            session_name=session_name,
            temporal=MUTABLE,
            embedding=EMBED_NEAR,
        ),
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )

    assert existing is not None
    assert existing.id == mutable.id
    assert result is not SemanticRejectionResult.NOT_DUPLICATE
