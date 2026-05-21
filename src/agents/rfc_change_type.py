import json

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..llm import get_openai_client, resolve_api_key
from ..observability import get_langfuse, record_node_invocation

logger = structlog.get_logger(__name__)

CHANGE_TYPE_QUESTION: dict = {
    "id": "rfc_change_type",
    "section": "Tipo de Cambio",
    "field": "Tipo de cambio",
    "question": "¿Cuál es el tipo de cambio?",
    "type": "single_select",
    "options": [
        "Recurrente",
        "Normal",
        "Estándar",
        "Emergencia",
        "Sugiéreme uno",
    ],
    "required": True,
}

_EXTRACTION_SYSTEM_PROMPT = """You are an RFC data extraction assistant.

The user was asked to select a change type from these options:
- Recurrente
- Normal
- Estándar
- Emergencia
- Sugiéreme uno

Map the user's answer to the EXACT option string from the list above.
If the answer is ambiguous or doesn't match any option, return an empty JSON object.
The user may answer with partial text, a number, or synonyms — map to the closest valid option.

Respond with only a JSON object like {"rfc_change_type": "<option>"} — no markdown, no explanation.
"""


async def rfc_change_type_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("rfc_change_type")

    api_key: str = resolve_api_key(config)
    client = get_openai_client(api_key)
    thread_id: str = state.get("thread_id", "unknown")
    rfc_data: dict = dict(state.get("rfc_data") or {})

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="rfc_change_type",
        metadata={"thread_id": thread_id, "node": "rfc_change_type"},
    )

    write = get_stream_writer()

    has_new_message = bool(state["messages"]) and getattr(state["messages"][-1], "type", "") == "human"

    rfc_change_type_complete = False

    if has_new_message and not rfc_data.get("rfc_change_type"):
        extraction_messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": state["messages"][-1].content},
        ]

        extraction_gen = trace.generation(
            name="change_type_extraction_llm",
            model="gpt-5",
            input={"messages": extraction_messages},
        )

        extraction_stream = await client.chat.completions.create(
            model="gpt-5",
            messages=extraction_messages,
            stream=True,
        )

        extraction_raw = ""
        async for chunk in extraction_stream:
            delta = chunk.choices[0].delta.content if chunk.choices else ""
            if delta:
                extraction_raw += delta

        extraction_gen.end(output=extraction_raw)

        try:
            extracted = json.loads(extraction_raw.strip())
            if isinstance(extracted, dict) and extracted.get("rfc_change_type"):
                rfc_data["rfc_change_type"] = extracted["rfc_change_type"]
        except Exception:
            logger.warning("rfc_change_type_extraction_failed", raw=extraction_raw[:200])

    rfc_change_type_complete = bool(rfc_data.get("rfc_change_type"))

    if rfc_change_type_complete:
        selected = rfc_data["rfc_change_type"]
        if selected == "Sugiéreme uno":
            full_response = (
                "Entendido, te sugeriré el tipo de cambio más adecuado "
                "una vez que hayamos recopilado toda la información del RFC. "
                "Continuemos con las preguntas."
            )
        else:
            full_response = (
                f"Perfecto, el tipo de cambio seleccionado es **{selected}**. "
                "Continuemos con las preguntas del RFC."
            )
        for char in full_response:
            write({"type": "token", "content": char})
    else:
        full_response = (
            "¡Hola! Vamos a crear un nuevo RFC. "
            "Primero, selecciona el tipo de cambio que necesitas."
        )
        for char in full_response:
            write({"type": "token", "content": char})

        write({
            "type": "closed_questions",
            "questions": [CHANGE_TYPE_QUESTION],
        })

    logger.info(
        "rfc_change_type_emitted",
        thread_id=thread_id,
        complete=rfc_change_type_complete,
    )

    return {
        "messages": [AIMessage(content=full_response)],
        "rfc_data": rfc_data,
        "rfc_change_type_complete": rfc_change_type_complete,
    }
