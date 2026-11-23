"""Add temporal class and lossless candidate snapshots to the conclusion ledger.

Revision ID: f81c4e26ab73
Revises: c3d9a41e7b52
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

from migrations.utils import column_exists, get_schema

revision = "f81c4e26ab73"
down_revision = "c3d9a41e7b52"
branch_labels = None
depends_on = None
schema = get_schema()


def upgrade() -> None:
    for table in ("documents", "conclusion_revisions"):
        if not column_exists(table, "temporal_class"):
            op.add_column(
                table,
                sa.Column("temporal_class", sa.Text, nullable=True),
                schema=schema,
            )
    if not column_exists("conclusion_revisions", "candidate"):
        op.add_column(
            "conclusion_revisions",
            sa.Column("candidate", JSONB, nullable=True),
            schema=schema,
        )


def downgrade() -> None:
    if column_exists("conclusion_revisions", "candidate"):
        op.drop_column("conclusion_revisions", "candidate", schema=schema)
    for table in ("conclusion_revisions", "documents"):
        if column_exists(table, "temporal_class"):
            op.drop_column(table, "temporal_class", schema=schema)
