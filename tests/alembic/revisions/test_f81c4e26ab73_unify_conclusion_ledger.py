from src import models
from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_before_upgrade("f81c4e26ab73")
def prepare_unified_ledger(verifier: MigrationVerifier) -> None:
    verifier.assert_column_exists("documents", "temporal_class", exists=False)
    verifier.assert_column_exists("conclusion_revisions", "candidate", exists=False)


@register_after_upgrade("f81c4e26ab73")
def verify_unified_ledger(verifier: MigrationVerifier) -> None:
    verifier.assert_column_exists("documents", "temporal_class", nullable=True)
    verifier.assert_column_exists(
        "conclusion_revisions", "temporal_class", nullable=True
    )
    verifier.assert_column_exists("conclusion_revisions", "candidate", nullable=True)
    assert {
        c["name"]
        for c in verifier.get_inspector().get_columns(
            "conclusion_revisions", schema=verifier.schema
        )
    } == {c.name for c in models.ConclusionRevision.__table__.columns}
