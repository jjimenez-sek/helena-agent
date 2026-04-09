import json
from typing import Literal

import structlog
from langchain_core.runnables import RunnableConfig

from ..graphs.state import AgentState
from ..llm import get_openai_client, resolve_api_key
from ..observability import get_langfuse, record_node_invocation
from ..prompts import fetch_active_prompt
from .utils import format_user_context

logger = structlog.get_logger(__name__)

_TRIAGE_SYSTEM_PROMPT = """You are the triage agent for Helena, a security operations AI assistant.

Your job is to classify the user's intent and return ONLY a JSON object.

Available intents:
- **rfc**: The user wants to create, submit, or work on a Request for Change (RFC / change request).
  Use this for any request related to: infrastructure changes, configuration changes, deployments,
  network changes, security policy changes, or any planned modification to production systems.
- **incident**: The user is reporting or responding to a security incident, alert, or active threat.
  (placeholder — not yet implemented)
- **knowledge**: The user is asking a question about documentation, runbooks, policies, or past incidents.
  (placeholder — not yet implemented)
- **escalation**: The issue requires immediate human intervention.
  (placeholder — not yet implemented)
- **unknown**: The request does not match any available intent or is a greeting.

Rules:
- Respond with ONLY a JSON object on one line — no markdown, no extra text.
- JSON schema: {"intent": "<rfc|incident|knowledge|escalation|unknown>"}
- Respond in the same language the user writes in.
"""


async def triage_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("triage")

    # If already mid-RFC flow and not yet complete, skip re-classification
    if (
        state.get("intent") == "rfc"
        and state.get("rfc_step", 0) > 0
        and not state.get("rfc_execute_confirmed", False)
    ):
        return {}

    api_key: str = resolve_api_key(config)
    client = get_openai_client(api_key)
    thread_id: str = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="triage",
        metadata={"thread_id": thread_id, "node": "triage"},
    )

    system_prompt = await fetch_active_prompt("TRIAGE", _TRIAGE_SYSTEM_PROMPT)
    messages_payload = [{"role": "system", "content": system_prompt + format_user_context(state)}]
    for msg in state["messages"]:
        if hasattr(msg, "type"):
            role = "assistant" if msg.type == "ai" else "user"
        else:
            role = "user"
        content = msg.content if hasattr(msg, "content") else str(msg)
        messages_payload.append({"role": role, "content": content})

    generation = trace.generation(
        name="triage_llm",
        model="gpt-5",
        input={"messages": messages_payload},
    )

    stream = await client.chat.completions.create(
        model="gpt-5",
        messages=messages_payload,
        stream=True,
        stream_options={"include_usage": True},
    )

    full_response = ""
    prompt_tokens = 0
    completion_tokens = 0

    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else ""
        if delta:
            full_response += delta
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

    generation.end(
        output=full_response,
        usage={"input": prompt_tokens, "output": completion_tokens},
    )

    try:
        parsed = json.loads(full_response.strip())
        intent: str = parsed.get("intent", "unknown")
    except (json.JSONDecodeError, AttributeError):
        logger.warning("triage_json_parse_failed", raw=full_response[:200])
        intent = "unknown"

    valid_intents = {"rfc", "incident", "knowledge", "escalation", "unknown"}
    if intent not in valid_intents:
        intent = "unknown"

    logger.info("triage_classified", thread_id=thread_id, intent=intent)

    # Triage is silent — downstream nodes handle all user-facing messages
    return {"intent": intent}


def route_from_triage(
    state: AgentState,
) -> Literal["rfc_open_questions", "rfc_closed_questions", "rfc_summary_confirm", "rfc_execute", "fallback_response"]:
    intent = state.get("intent", "unknown")

    if intent == "rfc":
        # Once execution is confirmed and complete, treat as unknown so the user can start fresh
        if state.get("rfc_execute_confirmed", False):
            return "fallback_response"

        rfc_confirmed = state.get("rfc_confirmed", False)
        rfc_closed_complete = state.get("rfc_closed_complete", False)
        rfc_open_complete = state.get("rfc_open_complete", False)

        if rfc_confirmed:
            return "rfc_execute"
        if rfc_closed_complete:
            return "rfc_summary_confirm"
        if rfc_open_complete:
            return "rfc_closed_questions"
        return "rfc_open_questions"

    return "fallback_response"
