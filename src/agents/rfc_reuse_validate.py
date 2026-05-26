import json

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..llm import get_openai_client, resolve_api_key
from ..observability import get_langfuse, record_node_invocation

logger = structlog.get_logger(__name__)

# All fields that must be present for a complete RFC
REQUIRED_OPEN_FIELDS = [
    "client_name", "client_authorizer",
    "technical_responsible_name", "change_supervisor",
    "work_team", "change_objective", "technical_description",
    "infrastructure_involved", "impact_during_change",
    "impact_after_change", "systems_involved",
    "implementation_steps", "estimated_duration",
    "rollback_conditions", "rollback_procedure",
    "technical_validations", "acceptance_criteria",
]

REQUIRED_CLOSED_FIELDS = [
    "change_type", "service_impact", "monitoring_loss",
    "remote_access_loss", "backup_status", "rollback_available",
    "risk_level", "execution_location",
]

_VALIDATION_SYSTEM_PROMPT = """You are Helena, a security operations AI assistant.

You are reviewing a pre-filled RFC (Request for Change) that a user wants to resubmit based on a previous RFC.

Your job:
1. Check if the data is COMPLETE — all required fields must have meaningful, non-placeholder values.
2. Check for LOGICAL CONSISTENCY:
   - If risk_level is "Alto" or "Crítico", there MUST be a detailed rollback_procedure and rollback_conditions.
   - If service_impact is "Impacto total", implementation_steps should mention a maintenance window.
   - If backup_status is "No", rollback_procedure must be especially detailed.
   - If change_type is "Cambio de emergencia", estimated_duration should be reasonable for urgency.
3. Check for STALE or PLACEHOLDER text — generic phrases like "TBD", "TODO", "same as before", or copy-paste artifacts.

Return ONLY a JSON object (no markdown, no extra text):
{
  "valid": true/false,
  "missing_fields": ["field_name", ...],
  "inconsistencies": ["description of issue", ...],
  "suggestions": ["suggestion for improvement", ...]
}

If valid is true, missing_fields and inconsistencies must be empty arrays.
"""

_PRESENT_VALID_PROMPT = """You are Helena, a security operations AI assistant.

The user has submitted a pre-filled RFC based on a previous one. The data has been validated and is COMPLETE and CONSISTENT.

Present a clean, professional summary of the RFC data below, organized by sections.
After the summary, ask: "Does this RFC look correct? Reply **confirm** to submit it, or tell me what needs to be corrected."

Respond in the same language the user writes in (likely Spanish).
"""

_PRESENT_INVALID_PROMPT = """You are Helena, a security operations AI assistant.

The user submitted a pre-filled RFC but it has issues that need to be resolved before submission.

Validation results:
{validation_json}

Current RFC data:
{rfc_data_str}

Present the issues clearly:
1. List any missing fields and ask the user to provide them.
2. List any logical inconsistencies and explain what needs to change.
3. List any suggestions for improvement.

Be helpful and specific. Tell the user exactly what information you need.
Respond in the same language the user writes in (likely Spanish).
"""


async def rfc_reuse_validate_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("rfc_reuse_validate")

    api_key: str = resolve_api_key(config)
    client = get_openai_client(api_key)
    thread_id: str = state.get("thread_id", "unknown")
    rfc_data: dict = dict(state.get("rfc_template_data") or {})

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="rfc_reuse_validate",
        metadata={"thread_id": thread_id, "node": "rfc_reuse_validate"},
    )

    write = get_stream_writer()
    write({"type": "rfc_step_progress", "step": 1, "total_open_steps": 2, "topic": "Validation"})

    # If user sent a new message (correction after failed validation), merge it
    has_new_message = (
        bool(state["messages"])
        and getattr(state["messages"][-1], "type", "") == "human"
    )

    if has_new_message and not state.get("rfc_reuse_validated", False):
        # User is providing missing info — use LLM to extract and merge
        user_text = state["messages"][-1].content or ""
        extraction_messages = [
            {
                "role": "system",
                "content": (
                    "The user is providing missing or corrected information for an RFC. "
                    "Extract any field values from their message and return a JSON object "
                    "mapping field names (snake_case) to their values. "
                    "Only include fields the user explicitly provided. "
                    "Return ONLY JSON, no markdown."
                ),
            },
            {
                "role": "user",
                "content": f"Current RFC data:\n{json.dumps(rfc_data, indent=2, ensure_ascii=False)}\n\nUser correction:\n{user_text}",
            },
        ]

        extraction_resp = await client.chat.completions.create(
            model="gpt-5",
            messages=extraction_messages,
        )
        try:
            extracted = json.loads(extraction_resp.choices[0].message.content.strip())
            if isinstance(extracted, dict):
                rfc_data.update(extracted)
                logger.info("rfc_reuse_correction_merged", thread_id=thread_id, fields=list(extracted.keys()))
        except (json.JSONDecodeError, AttributeError):
            logger.warning("rfc_reuse_correction_parse_failed", thread_id=thread_id)

    # Step 1: Field presence check
    missing_fields = []
    for field in REQUIRED_OPEN_FIELDS + REQUIRED_CLOSED_FIELDS:
        value = rfc_data.get(field)
        if not value or (isinstance(value, str) and not value.strip()):
            missing_fields.append(field)

    # Step 2: LLM consistency check
    rfc_data_str = json.dumps(rfc_data, indent=2, ensure_ascii=False)

    validation_messages = [
        {"role": "system", "content": _VALIDATION_SYSTEM_PROMPT},
        {"role": "user", "content": f"RFC data to validate:\n\n{rfc_data_str}"},
    ]

    gen = trace.generation(
        name="rfc_reuse_validate_llm",
        model="gpt-5",
        input={"messages": validation_messages},
    )

    validation_resp = await client.chat.completions.create(
        model="gpt-5",
        messages=validation_messages,
    )

    validation_text = validation_resp.choices[0].message.content or "{}"
    gen.end(output=validation_text, usage={
        "input": validation_resp.usage.prompt_tokens if validation_resp.usage else 0,
        "output": validation_resp.usage.completion_tokens if validation_resp.usage else 0,
    })

    try:
        validation_result = json.loads(validation_text.strip())
    except (json.JSONDecodeError, AttributeError):
        logger.warning("rfc_reuse_validation_parse_failed", thread_id=thread_id, raw=validation_text[:200])
        validation_result = {
            "valid": False,
            "missing_fields": missing_fields,
            "inconsistencies": ["Could not parse validation response"],
            "suggestions": [],
        }

    # Merge programmatic missing fields with LLM-detected ones
    llm_missing = validation_result.get("missing_fields", [])
    all_missing = list(set(missing_fields + llm_missing))
    is_valid = validation_result.get("valid", False) and len(all_missing) == 0

    logger.info(
        "rfc_reuse_validation_result",
        thread_id=thread_id,
        valid=is_valid,
        missing_count=len(all_missing),
        inconsistency_count=len(validation_result.get("inconsistencies", [])),
    )

    # Step 3: Stream the response
    if is_valid:
        rfc_data_str_formatted = "\n".join(f"- **{k}**: {v}" for k, v in rfc_data.items())
        response_messages = [
            {"role": "system", "content": _PRESENT_VALID_PROMPT},
            {"role": "user", "content": f"RFC data:\n\n{rfc_data_str_formatted}"},
        ]
    else:
        validation_result["missing_fields"] = all_missing
        response_messages = [
            {
                "role": "system",
                "content": _PRESENT_INVALID_PROMPT.format(
                    validation_json=json.dumps(validation_result, indent=2, ensure_ascii=False),
                    rfc_data_str="\n".join(f"- **{k}**: {v}" for k, v in rfc_data.items()),
                ),
            },
        ]

    gen2 = trace.generation(
        name="rfc_reuse_validate_response_llm",
        model="gpt-5",
        input={"messages": response_messages},
    )

    stream = await client.chat.completions.create(
        model="gpt-5",
        messages=response_messages,
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

    gen2.end(output=full_response, usage={"input": prompt_tokens, "output": completion_tokens})

    return {
        "messages": [AIMessage(content=full_response)],
        "rfc_template_data": rfc_data,
        "rfc_reuse_validated": is_valid,
    }
