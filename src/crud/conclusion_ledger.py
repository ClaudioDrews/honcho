"""Conclusion ledger: permanent history of how conclusions were interpreted.

Active only with `CONCLUSION_LEDGER_ENABLED`. Every function that writes is a
no-op with the flag off, so call sites need no guard of their own, and the
caller commits: revisions land in the transaction that makes the change.

What it adds over Representation History (`crud.dream_history`):

- every writer is recorded, not only dreams: deriver creation, reinforcement
  and semantic dedup, the dreamer's tools, the dialectic's tools and the
  conclusions API;
- each new conclusion carries its assertion basis, the declared spans of the
  cited messages it rests on, its validity and validity window, and its
  derivation record (producer, configured and answering model, prompt
  fingerprint, run id);
- baseline deduplication, retrieval and LLM schemas are unchanged when only
  CONCLUSION_LEDGER_ENABLED=True. Behavior-changing features are opt-in;
- optionally mark superseded conclusions historical, resolve contradictions,
  or filter by validity status and world-time bounds;
- event-scoped session deletion preserves unrelated later interpretations.

Deletion follows Representation History: deleting a conclusion through the API,
its session or its workspace deletes its revisions.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import (
    ColumnElement,
    CursorResult,
    and_,
    case,
    delete,
    func,
    or_,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.util import AliasedClass

from src import models
from src.config import settings
from src.crud.lifecycle import live_document_row
from src.exceptions import ResourceNotFoundException, ValidationException
from src.utils.derivation import (
    VALIDITIES,
    DerivationRecord,
    classify_with_spans,
    current_derivation,
)
from src.utils.interpretation import TEMPORAL_CLASSES, TemporalValidity
from src.utils.scopes import SCOPE_PEER_PREFIX

logger = logging.getLogger(__name__)


def ledger_enabled() -> bool:
    return settings.CONCLUSION_LEDGER.ENABLED


def temporal_from_metadata(metadata: dict[str, Any]) -> TemporalValidity:
    value = metadata.get("temporal")
    return (
        TemporalValidity.model_validate(value)
        if isinstance(value, dict)
        else TemporalValidity()
    )


def temporal_key(doc: Any) -> tuple[Any, ...]:
    if isinstance(doc, models.Document):
        return (doc.temporal_class or "unknown", doc.valid_from, doc.valid_until)
    return (doc.metadata.temporal or TemporalValidity()).key()


def retrieval_clause(
    document: type[models.Document] | AliasedClass[models.Document] = models.Document,
    *,
    include_non_current: bool = False,
    for_dedup: bool = False,
) -> ColumnElement[bool] | None:
    policy_enabled = (
        settings.CONCLUSION_LEDGER.TEMPORAL_DEDUP_ENABLED
        if for_dedup
        else settings.CONCLUSION_LEDGER.RETRIEVAL_FILTER_ENABLED
    )
    if include_non_current or not ledger_enabled() or not policy_enabled:
        return None
    excluded = settings.CONCLUSION_LEDGER.RETRIEVAL_EXCLUDE
    return and_(
        document.validity.notin_(excluded),
        or_(document.valid_from.is_(None), document.valid_from <= func.now()),
        or_(document.valid_until.is_(None), func.now() < document.valid_until),
    )


def _live_parents(row: Any) -> ColumnElement[bool]:
    """Fail closed for tombstoned or missing workspace, peer and source session.

    A reinforcement inherits the canonical conclusion's session stamp while
    its candidate may be authored in a different session. History liveness
    must follow the actual event's source, not the surviving projection.
    """
    source_session = case(
        # JSONB stores Python None as JSON null on some SQLAlchemy paths;
        # IS NOT NULL alone would incorrectly treat that as a candidate.
        (
            func.jsonb_typeof(row.candidate) == "object",
            row.candidate["session_name"].astext,
        ),
        else_=row.session_name,
    )
    return and_(
        select(models.Workspace.name)
        .where(
            models.Workspace.tenant_id == row.tenant_id,
            models.Workspace.name == row.workspace_name,
            models.Workspace.deleted_at.is_(None),
        )
        .exists(),
        *(
            select(models.Peer.name)
            .where(
                models.Peer.tenant_id == row.tenant_id,
                models.Peer.workspace_name == row.workspace_name,
                models.Peer.name == peer,
                models.Peer.deleted_at.is_(None),
            )
            .exists()
            for peer in (row.observer, row.observed)
        ),
        or_(
            ~row.observer.startswith(SCOPE_PEER_PREFIX),
            and_(
                source_session.is_not(None),
                select(models.SessionPeer.peer_name)
                .where(
                    models.SessionPeer.tenant_id == row.tenant_id,
                    models.SessionPeer.workspace_name == row.workspace_name,
                    models.SessionPeer.peer_name == row.observer,
                    models.SessionPeer.session_name == source_session,
                    models.SessionPeer.left_at.is_(None),
                )
                .exists(),
            ),
        ),
        or_(
            source_session.is_(None),
            select(models.Session.name)
            .where(
                models.Session.tenant_id == row.tenant_id,
                models.Session.workspace_name == row.workspace_name,
                models.Session.name == source_session,
                models.Session.deleted_at.is_(None),
            )
            .exists(),
        ),
    )


def _revision(
    doc: models.Document,
    kind: str,
    *,
    derivation: DerivationRecord | None,
    superseded_by: str | None = None,
    previous_validity: str | None = None,
    reason: str | None = None,
    candidate: dict[str, Any] | None = None,
) -> models.ConclusionRevision:
    return models.ConclusionRevision(
        workspace_name=doc.workspace_name,
        observer=doc.observer,
        observed=doc.observed,
        session_name=doc.session_name,
        conclusion_id=doc.id,
        kind=kind,
        level=doc.level,
        content=doc.content,
        assertion_basis=doc.assertion_basis,
        temporal_class=doc.temporal_class,
        candidate=candidate,
        validity=doc.validity,
        previous_validity=previous_validity,
        superseded_by=superseded_by,
        source_ids=doc.source_ids,
        source_message_ids=doc.source_message_ids,
        declared_spans=doc.declared_spans,
        valid_from=doc.valid_from,
        valid_until=doc.valid_until,
        times_derived=doc.times_derived,
        derivation=derivation.as_metadata() if derivation else None,
        reason=reason,
    )


async def _cited_messages(
    db: AsyncSession, workspace_name: str, public_ids: Sequence[str]
) -> dict[str, tuple[str, str, str]]:
    ids = sorted({mid for mid in public_ids if "\x00" not in mid})
    if not ids:
        return {}
    m = models.Message
    rows = await db.execute(
        select(m.public_id, m.content, m.peer_name, m.session_name).where(
            m.tenant_id == models.scoped_tenant_id(),
            m.workspace_name == workspace_name,
            m.public_id.in_(ids),
            m.deleted_at.is_(None),
            select(models.Peer.name)
            .where(
                models.Peer.tenant_id == m.tenant_id,
                models.Peer.workspace_name == m.workspace_name,
                models.Peer.name == m.peer_name,
                models.Peer.deleted_at.is_(None),
            )
            .exists(),
            select(models.Session.name)
            .where(
                models.Session.tenant_id == m.tenant_id,
                models.Session.workspace_name == m.workspace_name,
                models.Session.name == m.session_name,
                models.Session.deleted_at.is_(None),
            )
            .exists(),
        )
    )
    return {
        mid: (content, author, session)
        for mid, content, author, session in rows.tuples()
    }


def _basis(
    content: str,
    level: str,
    ids: Sequence[str],
    observed: str,
    session: str | None,
    cited: dict[str, tuple[str, str, str]],
    producer: str | None,
) -> tuple[str, list[dict[str, Any]]]:
    basis, spans = classify_with_spans(
        level=level,
        content=content,
        cited_messages=[
            (mid, cited[mid][0])
            for mid in ids
            if mid in cited and cited[mid][1] == observed and cited[mid][2] == session
        ],
        producer=producer,
        verbatim_min_chars=settings.CONCLUSION_LEDGER.VERBATIM_MIN_CHARS,
    )
    return basis, [span.as_dict() for span in spans]


async def snapshot_candidates(
    db: AsyncSession, workspace_name: str, observed: str, documents: Sequence[Any]
) -> dict[int, dict[str, Any]]:
    """Freeze each input before any deduplication or citation merging."""
    if not ledger_enabled():
        return {}
    cited = await _cited_messages(
        db,
        workspace_name,
        [mid for doc in documents for mid in (doc.source_message_ids or [])],
    )
    derivation = current_derivation()
    producer = derivation.producer if derivation else None
    snapshots: dict[int, dict[str, Any]] = {}
    for doc in documents:
        basis, spans = _basis(
            doc.content,
            doc.level,
            doc.source_message_ids or [],
            observed,
            doc.session_name,
            cited,
            producer,
        )
        authored_sources = [
            (mid, cited[mid][0])
            for mid in (doc.source_message_ids or [])
            if mid in cited
            and cited[mid][1] == observed
            and cited[mid][2] == doc.session_name
        ]
        snapshots[id(doc)] = {
            "source_utterances": [
                {
                    "message_id": mid,
                    "start": 0,
                    "end": len(text),
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
                for mid, text in authored_sources
            ],
            "content": doc.content,
            "level": doc.level,
            "session_name": doc.session_name,
            "source_ids": [sid for sid in (doc.source_ids or []) if "\x00" not in sid],
            "source_message_ids": [
                mid for mid in (doc.source_message_ids or []) if "\x00" not in mid
            ],
            "message_ids": list(doc.metadata.message_ids),
            "temporal": (doc.metadata.temporal or TemporalValidity()).model_dump(
                mode="json"
            ),
            "assertion_basis": basis,
            "declared_spans": spans,
        }
    return snapshots


def _annotate(
    doc: models.Document,
    basis: str,
    spans: list[dict[str, Any]],
    temporal: TemporalValidity,
    derivation: DerivationRecord | None,
) -> None:
    doc.assertion_basis = basis
    doc.declared_spans = spans or None
    doc.validity = "applicable"
    doc.temporal_class = temporal.temporal_class
    doc.valid_from = temporal.valid_from
    doc.valid_until = temporal.valid_to
    doc.internal_metadata = {
        **(doc.internal_metadata or {}),
        "temporal": temporal.model_dump(mode="json"),
        **({"derivation": derivation.as_metadata()} if derivation else {}),
    }


async def annotate_new_documents(
    db: AsyncSession,
    workspace_name: str,
    documents: Sequence[models.Document],
    *,
    candidates: dict[int, dict[str, Any]] | None = None,
) -> None:
    if not ledger_enabled() or not documents:
        return
    derivation = current_derivation()
    producer = derivation.producer if derivation else None
    cited = await _cited_messages(
        db,
        workspace_name,
        [mid for doc in documents for mid in (doc.source_message_ids or [])],
    )
    for doc in documents:
        snapshot = (candidates or {}).get(id(doc))
        if snapshot is not None:
            basis, spans = snapshot["assertion_basis"], snapshot["declared_spans"]
            temporal = TemporalValidity.model_validate(snapshot["temporal"])
        else:
            basis, spans = _basis(
                doc.content,
                doc.level,
                doc.source_message_ids or [],
                doc.observed,
                doc.session_name,
                cited,
                producer,
            )
            temporal = temporal_from_metadata(doc.internal_metadata or {})
        _annotate(doc, basis, spans, temporal, derivation)


async def _resolve_contradictions(
    db: AsyncSession,
    created: Sequence[models.Document],
    derivation: DerivationRecord | None,
) -> list[models.ConclusionRevision]:
    revisions: list[models.ConclusionRevision] = []
    if not settings.CONCLUSION_LEDGER.AUTO_RESOLVE_CONTRADICTIONS:
        return revisions
    for contradiction in created:
        if contradiction.level != "contradiction" or not contradiction.source_ids:
            continue
        # Locked in id order, the order document writers take rows in.
        contested = list(
            (
                await db.execute(
                    select(models.Document)
                    .where(
                        models.Document.tenant_id == models.scoped_tenant_id(),
                        models.Document.workspace_name == contradiction.workspace_name,
                        models.Document.observer == contradiction.observer,
                        models.Document.observed == contradiction.observed,
                        models.Document.id.in_(sorted(set(contradiction.source_ids))),
                        live_document_row(contradiction.workspace_name),
                    )
                    .order_by(models.Document.id)
                    .with_for_update()
                )
            ).scalars()
        )
        if len(contested) < 2:
            continue

        winner: models.Document | None = None
        if settings.CONCLUSION_LEDGER.CONTRADICTION_RESOLUTION == "most_recent":
            dated = [d for d in contested if d.valid_from is not None]
            if len(dated) == len(contested):
                latest = max(d.valid_from for d in dated if d.valid_from)
                newest = [d for d in dated if d.valid_from == latest]
                if len(newest) == 1:
                    winner = newest[0]

        for doc in contested:
            previous = doc.validity
            if winner is None:
                target, until, reason = "unknown", doc.valid_until, "contradiction"
            elif doc is winner:
                target, until, reason = (
                    "applicable",
                    doc.valid_until,
                    "contradiction:most_recent",
                )
            else:
                target = "incompatible"
                until = doc.valid_until
                reason = "contradiction:most_recent"
            if doc.validity == target and doc.valid_until == until:
                continue
            doc.validity = target
            doc.valid_until = until
            revisions.append(
                _revision(
                    doc,
                    "revalidated",
                    derivation=derivation,
                    superseded_by=contradiction.id,
                    previous_validity=previous,
                    reason=reason,
                )
            )
    return revisions


async def record_created(
    db: AsyncSession,
    created: Sequence[models.Document],
    *,
    replaced: Sequence[models.Document] = (),
    replaced_by: dict[str, models.Document | None] | None = None,
    reinforced: Sequence[models.Document] = (),
    candidates: dict[int, dict[str, Any]] | None = None,
    discarded: Sequence[tuple[models.Document, dict[str, Any], str]] = (),
) -> None:
    """Record new, reinforced and superseded conclusions, and contradictions.

    Run after the new rows are flushed (their ids are set) and in the same
    transaction.
    """
    if not ledger_enabled() or not (created or replaced or reinforced or discarded):
        return
    derivation = current_derivation()
    revisions: list[models.ConclusionRevision] = [
        _revision(
            doc,
            "created",
            derivation=derivation,
            candidate=(candidates or {}).get(id(doc)),
        )
        for doc in created
    ]

    # One revision per discarded input; do not collapse by canonical ID.
    revisions.extend(
        _revision(
            doc, "reinforced", derivation=derivation, candidate=snapshot, reason=reason
        )
        for doc, snapshot, reason in discarded
    )
    # Compatibility for direct callers that have only the surviving row.
    for doc in reinforced:
        revisions.append(
            _revision(
                doc,
                "reinforced",
                derivation=derivation,
                reason="duplicate_without_candidate",
            )
        )

    for old in replaced:
        new = (replaced_by or {}).get(old.id)
        previous = old.validity
        # Superseding a record is not evidence the fact stopped being true.
        if settings.CONCLUSION_LEDGER.MARK_SUPERSEDED_HISTORICAL:
            old.validity = "historical"
        revisions.append(
            _revision(
                old,
                "superseded",
                derivation=derivation,
                superseded_by=new.id if new is not None else None,
                previous_validity=previous,
                reason="semantic_duplicate",
            )
        )

    revisions.extend(await _resolve_contradictions(db, created, derivation))
    db.add_all(revisions)


@dataclass(frozen=True)
class RemovedRow:
    id: str
    level: str
    content: str
    session_name: str | None
    assertion_basis: str | None
    validity: str
    valid_from: datetime.datetime | None
    valid_until: datetime.datetime | None
    times_derived: int | None
    temporal_class: str | None = None


async def record_removed_rows(
    db: AsyncSession,
    workspace_name: str,
    *,
    observer: str,
    observed: str,
    rows: Sequence[RemovedRow],
    reason: str,
) -> None:
    """Record conclusions a writer removed (dreamer, agent tools).

    Not for user-requested erasure, which deletes revisions instead.
    """
    if not ledger_enabled() or not rows:
        return
    derivation = current_derivation()
    db.add_all(
        [
            models.ConclusionRevision(
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
                session_name=row.session_name,
                conclusion_id=row.id,
                kind="removed",
                level=row.level,
                content=row.content,
                assertion_basis=row.assertion_basis,
                temporal_class=row.temporal_class,
                validity=row.validity,
                valid_from=row.valid_from,
                valid_until=row.valid_until,
                times_derived=row.times_derived,
                derivation=derivation.as_metadata() if derivation else None,
                reason=reason,
            )
            for row in rows
        ]
    )


_UNSET: Any = object()


async def set_conclusion_validity(
    db: AsyncSession,
    workspace_name: str,
    conclusion_id: str,
    validity: str,
    *,
    reason: str | None = None,
    valid_from: datetime.datetime | None = _UNSET,
    valid_until: datetime.datetime | None = _UNSET,
    temporal_class: str | None = _UNSET,
) -> models.Document:
    """Change a live conclusion's validity or window and record it.

    Omitted bounds remain unchanged. A state transition is not evidence of a
    world-time transition. The caller commits.
    """
    if not ledger_enabled():
        raise ValidationException("The conclusion ledger is disabled")
    if validity not in VALIDITIES:
        raise ValidationException(f"Unknown validity {validity!r}")
    doc = (
        await db.execute(
            select(models.Document)
            .where(
                models.Document.tenant_id == models.scoped_tenant_id(),
                models.Document.workspace_name == workspace_name,
                models.Document.id == conclusion_id,
                live_document_row(workspace_name),
            )
            .with_for_update()
        )
    ).scalar_one_or_none()
    if doc is None:
        raise ResourceNotFoundException(
            f"Conclusion {conclusion_id} not found in workspace {workspace_name}"
        )

    new_from = doc.valid_from if valid_from is _UNSET else valid_from
    new_until = doc.valid_until if valid_until is _UNSET else valid_until
    new_class = doc.temporal_class if temporal_class is _UNSET else temporal_class
    if new_class is not None and new_class not in TEMPORAL_CLASSES:
        raise ValidationException("Unknown temporal class")
    for bound in (new_from, new_until):
        if bound is not None and bound.utcoffset() is None:
            raise ValidationException("Validity bounds must include a timezone")
    if new_from is not None and new_until is not None and new_until <= new_from:
        raise ValidationException("valid_until must be later than valid_from")
    if (doc.validity, doc.valid_from, doc.valid_until, doc.temporal_class) == (
        validity,
        new_from,
        new_until,
        new_class,
    ):
        return doc
    previous = doc.validity
    doc.validity = validity
    doc.valid_from = new_from
    doc.valid_until = new_until
    doc.temporal_class = new_class
    # Merge, don't replace: a legacy or future temporal dict may carry keys
    # TemporalValidity does not model; keep them.
    temporal_meta = dict((doc.internal_metadata or {}).get("temporal") or {})
    temporal_meta.update(
        TemporalValidity(
            temporal_class=new_class or "unknown",
            valid_from=new_from,
            valid_to=new_until,
        ).model_dump(mode="json")
    )
    doc.internal_metadata = {**(doc.internal_metadata or {}), "temporal": temporal_meta}
    db.add(
        _revision(
            doc,
            "revalidated",
            derivation=current_derivation(),
            previous_validity=previous,
            reason=reason,
        )
    )
    await db.flush()
    return doc


async def get_conclusion_history(
    db: AsyncSession, workspace_name: str, conclusion_id: str
) -> list[models.ConclusionRevision]:
    """A conclusion's revisions in order, followed along supersessions.

    The result runs from the first revision of `conclusion_id` through the
    revisions of each conclusion that superseded it (`kind == "superseded"`).
    """
    tenant_id = models.scoped_tenant_id()
    history: list[models.ConclusionRevision] = []
    seen: set[str] = set()
    current: str | None = conclusion_id
    while current is not None and current not in seen:
        seen.add(current)
        rows = list(
            (
                await db.execute(
                    select(models.ConclusionRevision)
                    .where(
                        models.ConclusionRevision.tenant_id == tenant_id,
                        models.ConclusionRevision.workspace_name == workspace_name,
                        models.ConclusionRevision.conclusion_id == current,
                        _live_parents(models.ConclusionRevision),
                        or_(
                            models.ConclusionRevision.candidate[
                                "session_name"
                            ].astext.is_(None),
                            select(models.Session.name)
                            .where(
                                models.Session.tenant_id == tenant_id,
                                models.Session.workspace_name == workspace_name,
                                models.Session.name
                                == models.ConclusionRevision.candidate[
                                    "session_name"
                                ].astext,
                                models.Session.deleted_at.is_(None),
                            )
                            .exists(),
                        ),
                    )
                    .order_by(models.ConclusionRevision.id)
                )
            ).scalars()
        )
        history.extend(rows)
        current = next(
            (
                row.superseded_by
                for row in reversed(rows)
                if row.kind == "superseded" and row.superseded_by
            ),
            None,
        )
    return history


async def erase_conclusion_revisions(
    db: AsyncSession, workspace_name: str, conclusion_id: str
) -> int:
    """Delete a conclusion's revisions, for a user-requested deletion."""
    row = models.ConclusionRevision
    scope = (
        row.tenant_id == models.scoped_tenant_id(),
        row.workspace_name == workspace_name,
    )
    frontier = {conclusion_id}
    visited: set[str] = set()
    count = 0
    while frontier - visited:
        ids = sorted(frontier - visited)
        visited.update(ids)
        related = or_(
            row.conclusion_id.in_(ids),
            row.superseded_by.in_(ids),
            *(row.source_ids.contains([cid]) for cid in ids),
            *(row.candidate["source_ids"].contains([cid]) for cid in ids),
        )
        rows = (
            await db.execute(
                select(row.id, row.conclusion_id)
                .where(*scope, related)
                .order_by(row.id)
                .with_for_update()
            )
        ).all()
        if not rows:
            break
        result = cast(
            CursorResult[Any],
            await db.execute(
                delete(row).where(*scope, row.id.in_([r.id for r in rows]))
            ),
        )
        count += result.rowcount
        frontier = {r.conclusion_id for r in rows} - visited
    return count


async def delete_session_revisions(
    db: AsyncSession, workspace_name: str, session_name: str
) -> None:
    """Erase this session's events, and only what directly depends on them.

    Two things are kept apart, because they are not the same:

    - the **events** this session produced. An event that carries a candidate
      belongs to the candidate's session, not to the canonical row's session,
      so a conclusion reinforced here belongs here even when it was created
      elsewhere.
    - the **conclusions this session originated**. Only a ``created`` event
      counts. A conclusion merely *reinforced* here was not originated here,
      and promoting its id to an origin would erase another session's
      independent derivations that happen to cite it.

    Erased with the session are its own events, plus the events that directly
    depend on a conclusion it originated -- by the event's own reference, the
    candidate's ``source_ids`` when it carries an interpretation, the canonical
    projection's otherwise.

    This is event-scoped erasure, not a proof that every unlinked derived claim
    has been scrubbed. Full data-subject erasure still requires separate review.
    """
    row = models.ConclusionRevision
    scope = (
        row.tenant_id == models.scoped_tenant_id(),
        row.workspace_name == workspace_name,
    )
    # The session an event effectively belongs to: the candidate's own session
    # when the event carries an interpretation, the canonical row's otherwise.
    effective_session = case(
        (
            func.jsonb_typeof(row.candidate) == "object",
            row.candidate["session_name"].astext,
        ),
        else_=row.session_name,
    )
    session_events = effective_session == session_name

    # Conclusions this session originated: a `created` event and nothing else.
    # A `reinforced` or `revalidated` event says the conclusion was *touched*
    # here, not that it came from here.
    origin_ids = sorted(
        set(
            await db.scalars(
                select(row.conclusion_id)
                .where(*scope, row.kind == "created", session_events)
                .distinct()
            )
        )
    )

    if not origin_ids:
        await db.execute(delete(row).where(*scope, session_events))
        return

    # Dependencies are resolved per event: the candidate's own source ids when
    # the event carries an interpretation, the canonical projection's
    # otherwise. The canonical row's references can belong to another
    # interpretation and must not decide what is erased here.
    def _depends_on(chunk: Sequence[str]) -> ColumnElement[bool]:
        return or_(
            *(
                or_(
                    row.candidate["source_ids"].contains([cid]),
                    and_(
                        func.coalesce(func.jsonb_typeof(row.candidate), "null")
                        != "object",
                        row.source_ids.contains([cid]),
                    ),
                )
                for cid in chunk
            )
        )

    # Chunked: the reference list grows with the session's conclusion count,
    # and one DELETE with thousands of JSONB contains clauses is a burden on
    # the planner and the wire.
    for offset in range(0, len(origin_ids), 500):
        chunk = origin_ids[offset : offset + 500]
        await db.execute(
            delete(row).where(*scope, or_(session_events, _depends_on(chunk)))
        )


async def delete_workspace_revisions(db: AsyncSession, workspace_name: str) -> None:
    await db.execute(
        delete(models.ConclusionRevision).where(
            models.ConclusionRevision.tenant_id == models.scoped_tenant_id(),
            models.ConclusionRevision.workspace_name == workspace_name,
        )
    )


__all__ = [
    "RemovedRow",
    "annotate_new_documents",
    "delete_session_revisions",
    "delete_workspace_revisions",
    "erase_conclusion_revisions",
    "get_conclusion_history",
    "ledger_enabled",
    "record_created",
    "record_removed_rows",
    "retrieval_clause",
    "set_conclusion_validity",
]
