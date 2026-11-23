import asyncio
from unittest.mock import AsyncMock

import pytest

from src.config import settings
from src.dreamer.specialists import DeductionSpecialist
from src.utils import agent_tools


@pytest.mark.asyncio
async def test_contradiction_advertised_and_dispatched(monkeypatch):
    monkeypatch.setattr(settings.CONCLUSION_LEDGER, "ENABLED", True)
    monkeypatch.setattr(settings.CONCLUSION_LEDGER, "CONTRADICTION_TOOL_ENABLED", True)
    tool = next(
        t
        for t in DeductionSpecialist().get_tools()
        if t["name"] == "create_observations_contradiction"
    )
    assert (
        tool["input_schema"]["properties"]["observations"]["items"]["properties"][
            "source_ids"
        ]["minItems"]
        == 2
    )
    handler = agent_tools.get_tool_handler(tool["name"])
    assert handler is not None
    impl = AsyncMock(return_value="OK")
    monkeypatch.setattr(agent_tools, "_handle_create_observations_impl", impl)
    ctx = agent_tools.ToolContext(
        workspace_name="workspace",
        observer="alice",
        observed="alice",
        session_name=None,
        current_messages=None,
        include_observation_ids=False,
        db_lock=asyncio.Lock(),
        agent_type="deduction",
    )
    body = {
        "observations": [{"content": "Conflicting claims", "source_ids": ["a", "b"]}]
    }
    assert await handler(ctx, body) == "OK"
    impl.assert_awaited_once_with(ctx, body, forced_level="contradiction")
    monkeypatch.setattr(settings.CONCLUSION_LEDGER, "ENABLED", False)
    assert all(t["name"] != tool["name"] for t in DeductionSpecialist().get_tools())
    assert (await handler(ctx, body)).startswith("ERROR:")
