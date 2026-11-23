"""Row-level security on the conclusion ledger's own table.

``conclusion_revisions`` holds what the system believed about a conclusion,
including the conclusions a tenant read from another tenant's workspace. The
row-level-security policies are applied by ``apply_tenant_isolation`` after the
migrations run under ``MULTI_TENANT``, so a plain single-tenant database has
none: this test applies them the same way to a dedicated throwaway database
and proves the ledger table honours them.

Postgres bypasses row-level security for superusers and table owners, so the
assertions run under ``SET ROLE`` to an ordinary, non-owner role. Connected as
the bootstrap superuser the policies would not apply and the checks would pass
without proving anything.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from nanoid import generate as generate_nanoid
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import DBAPIError
from sqlalchemy_utils import (
    create_database,  # pyright: ignore[reportUnknownVariableType]
)

from src.config import settings
from src.startup.tenant_isolation_validator import apply_tenant_isolation
from tests.conftest import (
    CONNECTION_URI,
    _get_test_db_url,  # pyright: ignore[reportPrivateUsage]
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_INI = _REPO_ROOT / "alembic.ini"
_MIGRATIONS_DIR = _REPO_ROOT / "migrations"

# Per xdist worker: the database and the role are cluster-global names, so two
# workers running this module would otherwise create and drop each other's.
_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "master")
_LEDGER_TEST_DB_URL: URL = _get_test_db_url(_WORKER, tag="rlsledger")
_RLS_ROLE = f"honcho_rls_ledger_probe_{_WORKER}"

_TENANT_A = "ledger-tenant-a"
_TENANT_B = "ledger-tenant-b"
_WORKSPACE = "ws-ledger"


def _new_id() -> str:
    return generate_nanoid()


def _run_on_maintenance_db(*statements: str) -> None:
    """Run AUTOCOMMIT statements against the postgres maintenance database."""
    maint = create_engine(
        CONNECTION_URI.set(database="postgres").render_as_string(hide_password=False),
        isolation_level="AUTOCOMMIT",
    )
    try:
        with maint.connect() as conn:
            for statement in statements:
                conn.exec_driver_sql(statement)
    finally:
        maint.dispose()


def _force_drop_test_db() -> None:
    _run_on_maintenance_db(
        f'DROP DATABASE IF EXISTS "{_LEDGER_TEST_DB_URL.database}" WITH (FORCE)'
    )


def _drop_probe_role() -> None:
    _run_on_maintenance_db(f"DROP ROLE IF EXISTS {_RLS_ROLE}")


@pytest.fixture(scope="session")
def ledger_rls_engine() -> Generator[Engine]:
    """A dedicated database with the real schema and the cloud RLS policies applied."""
    _force_drop_test_db()
    create_database(_LEDGER_TEST_DB_URL)

    db_url_str = _LEDGER_TEST_DB_URL.render_as_string(hide_password=False)
    engine = create_engine(db_url_str)
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # The real migrations, so the schema is exactly a self-hoster's schema.
        previous_uri = settings.DB.CONNECTION_URI
        previous_owner_uri = settings.DB.OWNER_CONNECTION_URI
        settings.DB.CONNECTION_URI = db_url_str
        settings.DB.OWNER_CONNECTION_URI = None
        try:
            cfg = Config(str(_ALEMBIC_INI))
            cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
            cfg.set_main_option("sqlalchemy.url", db_url_str)
            command.upgrade(cfg, "head")
        finally:
            settings.DB.CONNECTION_URI = previous_uri
            settings.DB.OWNER_CONNECTION_URI = previous_owner_uri

        with engine.begin() as conn:
            conn.execute(text(f"DROP ROLE IF EXISTS {_RLS_ROLE}"))
            conn.execute(text(f"CREATE ROLE {_RLS_ROLE} NOLOGIN"))
            conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {_RLS_ROLE}"))
            apply_tenant_isolation(conn, schema="public", app_role=_RLS_ROLE)
            conn.execute(
                text(
                    "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES "
                    + f"IN SCHEMA public TO {_RLS_ROLE}"
                )
            )

        _seed_ledger_rows(engine)
        yield engine
    finally:
        engine.dispose()
        _force_drop_test_db()
        _drop_probe_role()


def _seed_ledger_rows(engine: Engine) -> None:
    """One ledger row per tenant, in a workspace both tenants can name."""
    with engine.begin() as conn:
        for tenant in (_TENANT_A, _TENANT_B):
            conn.execute(
                text("INSERT INTO tenants (tenant_id) VALUES (:t)"), {"t": tenant}
            )
            conn.execute(
                text(
                    "INSERT INTO workspaces (tenant_id, id, name) VALUES (:t, :id, :n)"
                ),
                {"t": tenant, "id": _new_id(), "n": _WORKSPACE},
            )
            conn.execute(
                text(
                    "INSERT INTO conclusion_revisions "
                    + "(tenant_id, workspace_name, observer, observed, "
                    + "conclusion_id, kind, content) "
                    + "VALUES (:t, :w, 'peer-1', 'peer-2', :c, 'created', :content)"
                ),
                {
                    "t": tenant,
                    "w": _WORKSPACE,
                    "c": f"conclusion-{tenant}",
                    "content": f"a conclusion only {tenant} may read",
                },
            )


def _bind_tenant(conn: Connection, tenant: str) -> None:
    # Session-scoped (is_local=false), mirroring the app's checkout hook.
    conn.execute(text("SELECT set_config('app.tenant', :t, false)"), {"t": tenant})


@contextmanager
def _as_probe_role(engine: Engine) -> Generator[Connection]:
    raw = engine.connect()
    conn = raw.execution_options(isolation_level="AUTOCOMMIT")
    try:
        conn.execute(text(f"SET ROLE {_RLS_ROLE}"))
        yield conn
        conn.execute(text("RESET ROLE"))
    finally:
        raw.close()


def test_ledger_rows_are_tenant_scoped(ledger_rls_engine: Engine) -> None:
    """Each tenant reads only its own ledger rows, in a same-named workspace."""
    with _as_probe_role(ledger_rls_engine) as conn:
        _bind_tenant(conn, _TENANT_A)
        rows = conn.execute(
            text(
                "SELECT conclusion_id, content FROM conclusion_revisions "
                "WHERE workspace_name = :w"
            ),
            {"w": _WORKSPACE},
        ).all()
        assert [row[0] for row in rows] == [f"conclusion-{_TENANT_A}"]
        assert rows[0][1] == f"a conclusion only {_TENANT_A} may read"

        # An explicit cross-tenant predicate sees nothing.
        assert (
            conn.execute(
                text(
                    "SELECT count(*) FROM conclusion_revisions WHERE tenant_id = :other"
                ),
                {"other": _TENANT_B},
            ).scalar()
            == 0
        )

        # Switching the bound tenant flips the visible set.
        _bind_tenant(conn, _TENANT_B)
        rows = conn.execute(
            text(
                "SELECT conclusion_id FROM conclusion_revisions "
                "WHERE workspace_name = :w"
            ),
            {"w": _WORKSPACE},
        ).all()
        assert [row[0] for row in rows] == [f"conclusion-{_TENANT_B}"]


def test_ledger_read_fails_closed_without_a_tenant(
    ledger_rls_engine: Engine,
) -> None:
    """An unset tenant matches no ledger row."""
    with _as_probe_role(ledger_rls_engine) as conn:
        _bind_tenant(conn, "")
        assert (
            conn.execute(text("SELECT count(*) FROM conclusion_revisions")).scalar()
            == 0
        )


def test_ledger_write_for_another_tenant_is_refused(
    ledger_rls_engine: Engine,
) -> None:
    """The policy's WITH CHECK refuses a cross-tenant ledger write."""
    with _as_probe_role(ledger_rls_engine) as conn:
        _bind_tenant(conn, _TENANT_A)
        with pytest.raises(DBAPIError) as exc:
            conn.execute(
                text(
                    "INSERT INTO conclusion_revisions "
                    + "(tenant_id, workspace_name, observer, observed, "
                    + "conclusion_id, kind, content) "
                    + "VALUES (:other, :w, 'peer-1', 'peer-2', 'evil', 'created', 'x')"
                ),
                {"other": _TENANT_B, "w": _WORKSPACE},
            )
        assert "row-level security" in str(exc.value).lower()

        # A same-tenant write passes WITH CHECK and is visible to that tenant.
        conn.execute(
            text(
                "INSERT INTO conclusion_revisions "
                + "(tenant_id, workspace_name, observer, observed, "
                + "conclusion_id, kind, content) "
                + "VALUES (:t, :w, 'peer-1', 'peer-2', 'own', 'created', 'y')"
            ),
            {"t": _TENANT_A, "w": _WORKSPACE},
        )
        rows = conn.execute(
            text(
                "SELECT conclusion_id FROM conclusion_revisions "
                "WHERE workspace_name = :w"
            ),
            {"w": _WORKSPACE},
        ).all()
        assert sorted(row[0] for row in rows) == [
            f"conclusion-{_TENANT_A}",
            "own",
        ]
