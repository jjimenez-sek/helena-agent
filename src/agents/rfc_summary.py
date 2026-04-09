import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..llm import get_openai_client, resolve_api_key
from ..observability import get_langfuse, record_node_invocation
from ..prompts import fetch_active_prompt

logger = structlog.get_logger(__name__)

_SUMMARY_SYSTEM_PROMPT = """You are Helena, a security operations AI assistant.

You have collected all information needed for an RFC (Request for Change).
Generate a clear, professional summary of the RFC based on the data provided.

Format the summary as a readable document with sections:
- **Change Overview** (title, description, type, category)
- **Business Justification & Impact** (justification, affected systems/users, impact level)
- **Implementation Plan** (steps, rollback plan, dependencies, resources)
- **Scheduling** (start/end dates, environment, change window)
- **Risk & Testing** (risk level, mitigation, testing plan, approvers)
- **Structured Parameters** (priority, approval type, downtime, communication, compliance)

After the summary, ask the user to confirm:
"Does this RFC look correct? Reply **confirm** to submit it, or tell me what needs to be corrected."

Respond in the same language the user writes in.
"""

_CORRECTION_SYSTEM_PROMPT = """You are Helena, a security operations AI assistant.

The user wants to correct something in their RFC. Their correction request is:
"{correction}"

Current RFC data:
{rfc_data}

Acknowledge the correction, confirm what you've updated, and present a brief updated summary.
Then ask again for confirmation to submit.

Respond in the same language the user writes in.
"""

_CONFIRM_KEYWORDS = {"confirm", "yes", "approve", "submit", "ok", "okay", "sí", "si", "confirmar", "enviar"}


async def rfc_summary_confirm_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("rfc_summary_confirm")

    api_key: str = resolve_api_key(config)
    client = get_openai_client(api_key)
    thread_id: str = state.get("thread_id", "unknown")
    rfc_data: dict = dict(state.get("rfc_data") or {})

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="rfc_summary_confirm",
        metadata={"thread_id": thread_id, "node": "rfc_summary_confirm"},
    )

    write = get_stream_writer()
    write({"type": "rfc_step_progress", "step": 7, "total_open_steps": 7, "topic": "Summary & Confirmation"})

    has_new_message = bool(state["messages"]) and getattr(state["messages"][-1], "type", "") == "human"
    rfc_confirmed = False
    rfc_data_updated = rfc_data

    if has_new_message:
        user_text = (state["messages"][-1].content or "").strip().lower()
        # Check if user confirmed
        if any(kw in user_text for kw in _CONFIRM_KEYWORDS):
            rfc_confirmed = True
            logger.info("rfc_confirmed_by_user", thread_id=thread_id)

    rfc_data_str = "\n".join(f"- **{k}**: {v}" for k, v in rfc_data.items())

    if rfc_confirmed:
        # Brief confirmation acknowledgement — next step will present workflows and ask for final execution confirm
        intro_messages = [
            {
                "role": "system",
                "content": (
                    "The user has confirmed the RFC summary. "
                    "Acknowledge briefly that the RFC information is confirmed. "
                    "Tell them you are now going to show them the workflows that will be triggered and ask for final confirmation before submitting. "
                    "Be concise. Do NOT say you are submitting yet."
                ),
            }
        ]
    elif has_new_message and not any(kw in (state["messages"][-1].content or "").strip().lower() for kw in _CONFIRM_KEYWORDS):
        # User sent a correction
        correction_text = state["messages"][-1].content or ""

        # Apply correction to rfc_data via LLM
        correction_template = await fetch_active_prompt("RFC_SUMMARY", _CORRECTION_SYSTEM_PROMPT)
        correction_messages = [
            {
                "role": "system",
                "content": correction_template.format(
                    correction=correction_text,
                    rfc_data=rfc_data_str,
                ),
            }
        ]
        intro_messages = correction_messages
    else:
        # First visit — generate full summary
        summary_prompt = await fetch_active_prompt("RFC_SUMMARY", _SUMMARY_SYSTEM_PROMPT)
        intro_messages = [
            {
                "role": "system",
                "content": summary_prompt,
            },
            {
                "role": "user",
                "content": f"RFC data collected:\n\n{rfc_data_str}",
            },
        ]

    gen = trace.generation(
        name="rfc_summary_llm",
        model="gpt-5",
        input={"messages": intro_messages},
    )

    stream = await client.chat.completions.create(
        model="gpt-5",
        messages=intro_messages,
        stream=True,
        stream_options={"include_usage": True},
    )

    full_response = ""
    prompt_tokens = 0
    completion_tokens = 0

    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else ""
        if delta:
            write({"type": "token", "content": delta})
            full_response += delta
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

    gen.end(output=full_response, usage={"input": prompt_tokens, "output": completion_tokens})

    return {
        "messages": [AIMessage(content=full_response)],
        "rfc_data": rfc_data_updated,
        "rfc_confirmed": rfc_confirmed,
        "rfc_step": 7,
    }
