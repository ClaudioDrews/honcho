"""Check the conclusion ledger table and document columns in the revision pipeline."""

from src import models
from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_before_upgrade("c3d9a41e7b52")
def prepare_conclusion_ledger(verifier: MigrationVerifier) -> None:
    """The table and the columns are absent."""
    verifier.assert_table_exists("conclusion_revisions", exists=False)
    for column in (
        "assertion_basis",
        "validity",
        "declared_spans",
        "valid_from",
        "valid_until",
    ):
        verifier.assert_column_exists("documents", column, exists=False)


@register_after_upgrade("c3d9a41e7b52")
def verify_conclusion_ledger(verifier: MigrationVerifier) -> None:
    """Columns and table match the models; indexes are valid."""
    for column in ("assertion_basis", "declared_spans", "valid_from", "valid_until"):
        verifier.assert_column_exists("documents", column, nullable=True)
    verifier.assert_column_exists("documents", "validity", nullable=False)

    inspector = verifier.get_inspector()
    verifier.assert_table_exists("conclusion_revisions")
    assert {
        c["name"]
        for c in inspector.get_columns("conclusion_revisions", schema=verifier.schema)
    } == {column.name for column in models.ConclusionRevision.__table__.columns} - {
        "temporal_class",
        "candidate",
    }
    assert inspector.get_pk_constraint("conclusion_revisions", schema=verifier.schema)[
        "constrained_columns"
    ] == ["tenant_id", "id"]
    verifier.assert_indexes_valid(
        [
            ("conclusion_revisions", "ix_conclusion_revisions_conclusion"),
            ("conclusion_revisions", "ix_conclusion_revisions_collection_created"),
            ("conclusion_revisions", "ix_conclusion_revisions_session"),
        ]
    )
    # No foreign key to documents: revisions outlive the hard delete.
    referred = {
        fk["referred_table"]
        for fk in inspector.get_foreign_keys(
            "conclusion_revisions", schema=verifier.schema
        )
    }
    assert referred == {"tenants", "workspaces"}
