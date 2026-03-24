"""
Tests for RFC flow phase progression and API key security.
"""
import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.graphs.main_graph import build_graph


class TestRFCFlow:
    @pytest.mark.asyncio
    async def test_rfc_open_questions_invoked_for_rfc_intent(self):
        """When triage returns rfc intent, rfc_open_questions is called."""
        config = {
            "configurable": {
                "thread_id": "test-user:rfc-thread-1",
                "openai_api_key": "sk-test",
            }
        }

        mock_triage = AsyncMock(return_value={"intent": "rfc"})
        mock_rfc_open = AsyncMock(
            return_value={
                "messages": [AIMessage(content="What is the change title?")],
                "rfc_step": 1,
            }
        )

        with (
            patch("src.graphs.main_graph.triage_node", new=mock_triage),
            patch("src.graphs.main_graph.rfc_open_questions_node", new=mock_rfc_open),
        ):
            graph = build_graph(MemorySaver())
            chunks = []
            async for chunk in graph.astream(
                {
                    "messages": [HumanMessage(content="I need to open an RFC")],
                    "thread_id": "test-user:rfc-thread-1",
                },
                config=config,
                stream_mode="updates",
            ):
                chunks.append(chunk)

        node_names = [list(c.keys())[0] for c in chunks if isinstance(c, dict)]
        assert "rfc_open_questions" in node_names

    @pytest.mark.asyncio
    async def test_triage_skipped_when_rfc_already_in_progress(self):
        """triage is a no-op (returns {}) when intent=rfc and rfc_step > 0."""
        config = {
            "configurable": {
                "thread_id": "test-user:rfc-thread-2",
                "openai_api_key": "sk-test",
            }
        }

        # When already in RFC flow, triage returns {} (its early-exit path)
        mock_triage = AsyncMock(return_value={})
        mock_rfc_open = AsyncMock(
            return_value={
                "messages": [AIMessage(content="Step 2 question.")],
                "rfc_step": 2,
            }
        )

        with (
            patch("src.graphs.main_graph.triage_node", new=mock_triage),
            patch("src.graphs.main_graph.rfc_open_questions_node", new=mock_rfc_open),
        ):
            graph = build_graph(MemorySaver())
            chunks = []
            async for chunk in graph.astream(
                {
                    "messages": [HumanMessage(content="A deployment change")],
                    "thread_id": "test-user:rfc-thread-2",
                    "intent": "rfc",
                    "rfc_step": 1,
                },
                config=config,
                stream_mode="updates",
            ):
                chunks.append(chunk)

        node_names = [list(c.keys())[0] for c in chunks if isinstance(c, dict)]
        assert "rfc_open_questions" in node_names

    @pytest.mark.asyncio
    async def test_rfc_auto_advances_through_all_phases(self):
        """When each phase completes, the graph auto-advances through all RFC phases."""
        config = {
            "configurable": {
                "thread_id": "test-user:rfc-thread-3",
                "openai_api_key": "sk-test",
            }
        }

        mock_triage = AsyncMock(return_value={"intent": "rfc"})
        mock_rfc_open = AsyncMock(
            return_value={
                "messages": [AIMessage(content="Open questions done.")],
                "rfc_step": 5,
                "rfc_open_complete": True,
                "rfc_data": {"title": "Test RFC"},
            }
        )
        mock_rfc_closed = AsyncMock(
            return_value={
                "messages": [AIMessage(content="Closed questions done.")],
                "rfc_closed_complete": True,
            }
        )
        mock_rfc_summary = AsyncMock(
            return_value={
                "messages": [AIMessage(content="RFC confirmed.")],
                "rfc_confirmed": True,
            }
        )
        mock_rfc_execute = AsyncMock(
            return_value={"messages": [], "rfc_execute_confirmed": True}
        )

        with (
            patch("src.graphs.main_graph.triage_node", new=mock_triage),
            patch("src.graphs.main_graph.rfc_open_questions_node", new=mock_rfc_open),
            patch("src.graphs.main_graph.rfc_closed_questions_node", new=mock_rfc_closed),
            patch("src.graphs.main_graph.rfc_summary_confirm_node", new=mock_rfc_summary),
            patch("src.graphs.main_graph.rfc_execute_node", new=mock_rfc_execute),
        ):
            graph = build_graph(MemorySaver())
            chunks = []
            async for chunk in graph.astream(
                {
                    "messages": [HumanMessage(content="Create an RFC for a network change")],
                    "thread_id": "test-user:rfc-thread-3",
                },
                config=config,
                stream_mode="updates",
            ):
                chunks.append(chunk)

        node_names = [list(c.keys())[0] for c in chunks if isinstance(c, dict)]
        assert "rfc_open_questions" in node_names
        assert "rfc_closed_questions" in node_names
        assert "rfc_summary_confirm" in node_names
        assert "rfc_execute" in node_names

    @pytest.mark.asyncio
    async def test_api_key_not_in_state_updates(self):
        """API key must never appear in any state update chunk."""
        config = {
            "configurable": {
                "thread_id": "test-user:rfc-thread-4",
                "openai_api_key": "sk-must-not-leak",
            }
        }

        mock_triage = AsyncMock(return_value={"intent": "unknown"})
        mock_fallback = AsyncMock(return_value={"messages": [AIMessage(content="Hello")]})

        with (
            patch("src.graphs.main_graph.triage_node", new=mock_triage),
            patch("src.graphs.main_graph.fallback_response_node", new=mock_fallback),
        ):
            graph = build_graph(MemorySaver())
            async for chunk in graph.astream(
                {
                    "messages": [HumanMessage(content="Run workflow")],
                    "thread_id": "test-user:rfc-thread-4",
                },
                config=config,
                stream_mode="updates",
            ):
                chunk_str = json.dumps(chunk, default=str)
                assert "sk-must-not-leak" not in chunk_str, (
                    f"API key leaked into state chunk: {chunk_str}"
                )
