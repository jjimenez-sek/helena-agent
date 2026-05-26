import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..llm import get_openai_client, resolve_api_key
from ..observability import get_langfuse, record_node_invocation

logger = structlog.get_logger(__name__)

_CONFIRM_KEYWORDS = {"confirm", "yes", "approve", "submit", "ok", "okay", "sí", "si", "confirmar", "enviar"}


async def rfc_reuse_confirm_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("rfc_reuse_confirm")

    api_key: str = resolve_api_key(config)
    client = get_openai_client(api_key)
    thread_id: str = state.get("thread_id", "unknown")
    rfc_data: dict = dict(state.get("rfc_template_data") or {})
    conversation_id: str = state.get("conversation_id", "")
    project_workflows: list = list(state.get("project_workflows") or [])

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="rfc_reuse_confirm",
        metadata={"thread_id": thread_id, "node": "rfc_reuse_confirm"},
    )

    write = get_stream_writer()
    write({"type": "rfc_step_progress", "step": 2, "total_open_steps": 2, "topic": "Confirmation"})

    has_new_message = (
        bool(state["messages"])
        and getattr(state["messages"][-1], "type", "") == "human"
    )

    rfc_reuse_confirmed = False
    rfc_reuse_validated = state.get("rfc_reuse_validated", True)
    is_correction = False

    if has_new_message:
        user_text = (state["messages"][-1].content or "").strip().lower()
        if any(kw in user_text for kw in _CONFIRM_KEYWORDS):
            rfc_reuse_confirmed = True
            logger.info("rfc_reuse_confirmed_by_user", thread_id=thread_id)
        else:
            # User sent a correction — invalidate so we go back to validate
            is_correction = True
            rfc_reuse_validated = False
            logger.info("rfc_reuse_correction_received", thread_id=thread_id)

    if rfc_reuse_confirmed:
        # Emit execute_workflow and acknowledge
        workflows_to_run = [
            {"workflow_id": wf["workflow_id"], "name": wf.get("name", "workflow")}
            for wf in project_workflows
            if wf.get("workflow_id")
        ]

        write({
            "type": "execute_workflow",
            "conversation_id": conversation_id,
            "workflows": workflows_to_run,
            "rfc_data": rfc_data,
        })

        logger.info(
            "rfc_reuse_execute_triggered",
            thread_id=thread_id,
            conversation_id=conversation_id,
            workflow_count=len(workflows_to_run),
        )

        # Stream brief confirmation
        confirm_messages = [
            {
                "role": "system",
                "content": (
                    "The user has confirmed the RFC from a previous template. "
                    "Acknowledge briefly that the RFC has been confirmed and workflows are being triggered. "
                    "Be concise. Respond in the same language the user writes in (likely Spanish)."
                ),
            },
        ]

        gen = trace.generation(
            name="rfc_reuse_confirm_llm",
            model="gpt-5",
            input={"messages": confirm_messages},
        )

        stream = await client.chat.completions.create(
            model="gpt-5",
            messages=confirm_messages,
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
            "rfc_reuse_confirmed": True,
            "rfc_execute_confirmed": True,
        }

    if is_correction:
        # Template submissions skip re-validation — ask user to confirm as-is or edit the template
        correction_messages = [
            {
                "role": "system",
                "content": (
                    "The user sent a message that is not a confirmation keyword. "
                    "The RFC was submitted from a pre-filled template and does not go through re-validation. "
                    "Let the user know that to make changes they should edit the template directly and resubmit. "
                    "To proceed with the current RFC, they should confirm with a word like 'confirmar' or 'enviar'. "
                    "Be concise. Respond in the same language the user writes in (likely Spanish)."
                ),
            },
        ]

        gen = trace.generation(
            name="rfc_reuse_correction_ack_llm",
            model="gpt-5",
            input={"messages": correction_messages},
        )

        stream = await client.chat.completions.create(
            model="gpt-5",
            messages=correction_messages,
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
        }

    # No user message yet — should not normally happen, but handle gracefully
    return {"messages": []}
