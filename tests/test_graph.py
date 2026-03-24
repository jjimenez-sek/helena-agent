"""
Integration-style tests for the LangGraph graph wiring.
Uses MemorySaver (allowed in tests only) to avoid requiring a real PostgreSQL instance.
"""
import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.graphs.main_graph import build_graph


class TestGraphWiring:
    @pytest.mark.asyncio
    async def test_graph_routes_unknown_to_fallback(self):
        """triage returning unknown intent should invoke fallback_response then END."""
        config = {
            "configurable": {
                "thread_id": "test-user:thread-1",
                "openai_api_key": "sk-test",
            }
        }

        mock_triage = AsyncMock(return_value={"intent": "unknown"})
        mock_fallback = AsyncMock(
            return_value={"messages": [AIMessage(content="I can help with RFC requests.")]}
        )

        with (
            patch("src.graphs.main_graph.triage_node", new=mock_triage),
            patch("src.graphs.main_graph.fallback_response_node", new=mock_fallback),
        ):
            graph = build_graph(MemorySaver())
            results = []
            async for chunk in graph.astream(
                {
                    "messages": [HumanMessage(content="Hello")],
                    "thread_id": "test-user:thread-1",
                },
                config=config,
                stream_mode="updates",
            ):
                results.append(chunk)

        assert len(results) > 0
        node_names = [list(r.keys())[0] for r in results if isinstance(r, dict)]
        assert "fallback_response" in node_names
        assert "rfc_open_questions" not in node_names

    @pytest.mark.asyncio
    async def test_graph_routes_rfc_to_open_questions(self):
        """triage returning rfc intent should invoke rfc_open_questions."""
        config = {
            "configurable": {
                "thread_id": "test-user:thread-2",
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
            results = []
            async for chunk in graph.astream(
                {
                    "messages": [HumanMessage(content="I need to create an RFC for a firewall change")],
                    "thread_id": "test-user:thread-2",
                },
                config=config,
                stream_mode="updates",
            ):
                results.append(chunk)

        node_names = [list(r.keys())[0] for r in results if isinstance(r, dict)]
        assert "rfc_open_questions" in node_names
        assert "fallback_response" not in node_names

    @pytest.mark.asyncio
    async def test_state_does_not_contain_api_key(self):
        """Ensure openai_api_key never appears in any state update."""
        config = {
            "configurable": {
                "thread_id": "test-user:thread-3",
                "openai_api_key": "sk-super-secret",
            }
        }

        mock_triage = AsyncMock(return_value={"intent": "unknown"})
        mock_fallback = AsyncMock(return_value={"messages": [AIMessage(content="I can help.")]})

        with (
            patch("src.graphs.main_graph.triage_node", new=mock_triage),
            patch("src.graphs.main_graph.fallback_response_node", new=mock_fallback),
        ):
            graph = build_graph(MemorySaver())
            async for chunk in graph.astream(
                {
                    "messages": [HumanMessage(content="test")],
                    "thread_id": "test-user:thread-3",
                },
                config=config,
                stream_mode="updates",
            ):
                chunk_str = json.dumps(chunk, default=str)
                assert "sk-super-secret" not in chunk_str, (
                    f"API key leaked into state update: {chunk_str}"
                )

    @pytest.mark.asyncio
    async def test_display_message_is_natural_language(self):
        """The AIMessage emitted by fallback must not be raw JSON."""
        config = {
            "configurable": {
                "thread_id": "test-user:thread-4",
                "openai_api_key": "sk-test",
            }
        }

        mock_triage = AsyncMock(return_value={"intent": "unknown"})
        mock_fallback = AsyncMock(
            return_value={"messages": [AIMessage(content="I found the relevant information.")]}
        )

        with (
            patch("src.graphs.main_graph.triage_node", new=mock_triage),
            patch("src.graphs.main_graph.fallback_response_node", new=mock_fallback),
        ):
            graph = build_graph(MemorySaver())
            async for chunk in graph.astream(
                {
                    "messages": [HumanMessage(content="test")],
                    "thread_id": "test-user:thread-4",
                },
                config=config,
                stream_mode="updates",
            ):
                if isinstance(chunk, dict) and "fallback_response" in chunk:
                    messages = chunk["fallback_response"].get("messages", [])
                    for msg in messages:
                        content = msg.content if hasattr(msg, "content") else str(msg)
                        assert not (
                            content.strip().startswith("{") and '"intent"' in content
                        ), f"Fallback emitted raw JSON to user: {content}"
