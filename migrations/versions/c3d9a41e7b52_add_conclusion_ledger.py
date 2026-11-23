"""add conclusion ledger

`conclusion_revisions` is the permanent, append-only history of how conclusions
were interpreted: one row per conclusion created, superseded, removed or
revalidated, by any writer (deriver, dreamer, API), each a copy of the
conclusion as it read then, with the derivation record that wrote it. It is
tenant-scoped like the other data tables (tenant_id leads the primary key and
every foreign key), minus physical partitioning, and has no foreign key to
documents so it outlives their hard delete; deletion removes its rows
explicitly.

New document columns: `assertion_basis`, `declared_spans`, `valid_from` and
`valid_until` (nullable) and `validity` (default `applicable`). Adding a
nullable column and a column with a constant default is metadata-only on
Postgres 11+, so none of them rewrites the table. Rows written before this revision keep a NULL assertion basis: nothing
is backfilled, because the basis of an old conclusion cannot be recovered.
Their values are checked in the application, not by a CHECK constraint, which
would scan documents under an ACCESS EXCLUSIVE lock.

Everything here is written only with CONCLUSION_LEDGER_ENABLED.

Revision ID: c3d9a41e7b52
Revises: 7dbf160eb7c9
Create Date: 2026-11-22

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

from migrations.utils import column_exists, get_schema, table_exists

# revision identifiers, used by Alembic.
revision: str = "c3d9a41e7b52"
down_revision: str | None = "7dbf160eb7c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if not column_exists("documents", "assertion_basis", inspector):
        op.add_column(
            "documents",
            sa.Column("assertion_basis", sa.TEXT, nullable=True),
            schema=schema,
        )
    if not column_exists("documents", "validity", inspector):
        op.add_column(
            "documents",
            sa.Column(
                "validity",
                sa.TEXT,
                nullable=False,
                server_default=sa.text("'applicable'"),
            ),
            schema=schema,
        )
    if not column_exists("documents", "declared_spans", inspector):
        op.add_column(
            "documents",
            sa.Column("declared_spans", JSONB, nullable=True),
            schema=schema,
        )
    for column in ("valid_from", "valid_until"):
        if not column_exists("documents", column, inspector):
            op.add_column(
                "documents",
                sa.Column(column, sa.DateTime(timezone=True), nullable=True),
                schema=schema,
            )

    if not table_exists("conclusion_revisions", inspector):
        op.create_table(
            "conclusion_revisions",
            sa.Column("tenant_id", sa.TEXT, nullable=False),
            sa.Column("id", sa.BigInteger, sa.Identity(), nullable=False),
            sa.Column("workspace_name", sa.TEXT, nullable=False),
            sa.Column("observer", sa.TEXT, nullable=False),
            sa.Column("observed", sa.TEXT, nullable=False),
            sa.Column("session_name", sa.TEXT, nullable=True),
            sa.Column("conclusion_id", sa.TEXT, nullable=False),
            sa.Column("kind", sa.TEXT, nullable=False),
            sa.Column("level", sa.TEXT, nullable=True),
            sa.Column("content", sa.TEXT, nullable=True),
            sa.Column("assertion_basis", sa.TEXT, nullable=True),
            sa.Column("validity", sa.TEXT, nullable=True),
            sa.Column("previous_validity", sa.TEXT, nullable=True),
            sa.Column("superseded_by", sa.TEXT, nullable=True),
            sa.Column("source_ids", JSONB, nullable=True),
            sa.Column("source_message_ids", JSONB, nullable=True),
            sa.Column("declared_spans", JSONB, nullable=True),
            sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
            sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
            sa.Column("times_derived", sa.Integer, nullable=True),
            sa.Column("derivation", JSONB, nullable=True),
            sa.Column("reason", sa.TEXT, nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.PrimaryKeyConstraint("tenant_id", "id"),
            sa.ForeignKeyConstraint(["tenant_id"], [f"{schema}.tenants.tenant_id"]),
            sa.ForeignKeyConstraint(
                ["workspace_name", "tenant_id"],
                [f"{schema}.workspaces.name", f"{schema}.workspaces.tenant_id"],
                name="fk_conclusion_revisions_ws_tenant_workspaces",
            ),
            sa.CheckConstraint(
                "kind IN ('created', 'reinforced', 'superseded', 'removed', "
                + "'revalidated')",
                name="kind_valid",
            ),
            schema=schema,
        )
        op.create_index(
            "ix_conclusion_revisions_conclusion",
            "conclusion_revisions",
            ["tenant_id", "workspace_name", "conclusion_id", "id"],
            schema=schema,
        )
        op.create_index(
            "ix_conclusion_revisions_collection_created",
            "conclusion_revisions",
            ["tenant_id", "workspace_name", "observer", "observed", "created_at"],
            schema=schema,
        )
        op.create_index(
            "ix_conclusion_revisions_session",
            "conclusion_revisions",
            ["tenant_id", "workspace_name", "session_name"],
            postgresql_where=sa.text("session_name IS NOT NULL"),
            schema=schema,
        )


def downgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if table_exists("conclusion_revisions", inspector):
        op.drop_table("conclusion_revisions", schema=schema)
    for column in ("valid_until", "valid_from", "declared_spans"):
        if column_exists("documents", column, inspector):
            op.drop_column("documents", column, schema=schema)
    if column_exists("documents", "validity", inspector):
        op.drop_column("documents", "validity", schema=schema)
    if column_exists("documents", "assertion_basis", inspector):
        op.drop_column("documents", "assertion_basis", schema=schema)
