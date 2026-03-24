import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..llm import get_openai_client, resolve_api_key
from ..observability import get_langfuse, record_node_invocation

logger = structlog.get_logger(__name__)

_FALLBACK_SYSTEM_PROMPT = """You are Helena, a security operations AI assistant.

The user's request does not match any of the currently available capabilities.

Explain what Helena can currently help with:
- **RFC / Change Requests**: Creating and submitting Requests for Change through a guided multi-step process.
  This covers infrastructure changes, deployments, configuration changes, network changes, and security policy updates.

The following capabilities are coming soon (not yet available):
- **Incident Response**: Handling security incidents and alerts.
- **Knowledge Base**: Answering questions from internal documentation and runbooks.
- **Escalation**: Escalating issues to human operators.

Be helpful and ask the user to clarify what they need, guiding them toward available capabilities.
Respond in the same language the user writes in.
"""


async def fallback_response_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("fallback")

    api_key: str = resolve_api_key(config)
    client = get_openai_client(api_key)
    thread_id: str = state.get("thread_id", "unknown")

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="fallback_response",
        metadata={"thread_id": thread_id, "node": "fallback"},
    )

    messages_payload = [{"role": "system", "content": _FALLBACK_SYSTEM_PROMPT}]
    for msg in state["messages"]:
        if hasattr(msg, "type"):
            role = "assistant" if msg.type == "ai" else "user"
        else:
            role = "user"
        content = msg.content if hasattr(msg, "content") else str(msg)
        messages_payload.append({"role": role, "content": content})

    gen = trace.generation(
        name="fallback_llm",
        model="gpt-5",
        input={"messages": messages_payload},
    )

    write = get_stream_writer()

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
            write({"type": "token", "content": delta})
            full_response += delta
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

    gen.end(output=full_response, usage={"input": prompt_tokens, "output": completion_tokens})
    logger.info("fallback_response_sent", thread_id=thread_id, intent=state.get("intent", "unknown"))

    return {
        "messages": [AIMessage(content=full_response)],
    }
