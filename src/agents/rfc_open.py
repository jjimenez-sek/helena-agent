import json

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..llm import get_openai_client, resolve_api_key
from ..observability import get_langfuse, record_node_invocation
from .utils import format_user_context

logger = structlog.get_logger(__name__)

# Each step covers a group of RFC fields.
# The LLM collects answers conversationally and extracts them as JSON.
_STEP_CONFIGS = [
    {
        "step": 1,
        "topic": "Identificación del RFC y Cliente",
        "fields": ["client_name"],
        "questions": (
            "Empecemos con la identificación del RFC:\n"
            "1. ¿Cuál es el nombre del cliente afectado?"
        ),
    },
    {
        "step": 2,
        "topic": "Responsables del Cambio",
        "fields": [
            "client_authorizer",
            "technical_responsible_name",
            "technical_responsible_phone",
            "change_supervisor",
            "consulted_parties",
            "informed_parties",
        ],
        "questions": (
            "Ahora necesito los responsables del cambio:\n"
            "1. ¿Cuál es el nombre del autorizador del cliente? (nombre y cargo)\n"
            "2. ¿Cuál es el nombre del responsable técnico que ejecutará el cambio? (nombre y cargo)\n"
            "3. ¿Cuál es el número de teléfono del responsable técnico?\n"
            "4. ¿Cuál es el nombre del supervisor del cambio? (nombre y cargo)\n"
            "5. ¿Quién será consultado durante el cambio?\n"
            "6. ¿Quién será informado del cambio?"
        ),
    },
    {
        "step": 3,
        "topic": "Equipo de Trabajo y Detalle del Cambio",
        "fields": ["work_team", "change_objective", "technical_description"],
        "questions": (
            "Cuéntame sobre el equipo y el detalle técnico del cambio:\n"
            "1. ¿Cuál es el área o equipo que ejecutará el cambio?\n"
            "2. ¿Cuál es el objetivo del cambio?\n"
            "3. Descripción técnica detallada: ¿en qué consiste el cambio y qué actividades incluye?"
        ),
    },
    {
        "step": 4,
        "topic": "Impacto al Negocio y Equipos Involucrados",
        "fields": ["impact_during_change", "impact_after_change", "systems_involved", "systems_hostnames_ips"],
        "questions": (
            "Ahora el impacto esperado y los sistemas involucrados:\n"
            "1. ¿Cuál es el impacto esperado durante el cambio? (disponibilidad de servicios, duración aproximada)\n"
            "2. ¿Cuál es el impacto esperado después del cambio? (mejoras, beneficios)\n"
            "3. ¿Qué sistemas e infraestructura están involucrados en el cambio? (ej. cluster, nodos, bases de datos, load balancers, plataforma de monitoreo, servicios en la nube, etc.)\n"
            "4. Para los sistemas que serán directamente afectados por el cambio, indica el nombre de host (hostname) y la dirección IP."
        ),
    },
    {
        "step": 5,
        "topic": "Plan de Trabajo, Rollback y Pruebas",
        "fields": [
            "implementation_steps",
            "estimated_duration",
            "rollback_conditions",
            "rollback_procedure",
            "estimated_rollback_time",
            "technical_validations",
            "acceptance_criteria",
        ],
        "questions": (
            "Por último, los planes de trabajo, rollback y pruebas:\n"
            "1. ¿Cuáles son los pasos del plan de trabajo?\n"
            "2. ¿Cuál es la duración estimada del cambio?\n"
            "3. ¿Cuáles son las condiciones que activarían un rollback?\n"
            "4. ¿Cuál es el procedimiento de rollback paso a paso?\n"
            "5. ¿Cuál es el tiempo estimado de rollback?\n"
            "6. ¿Cuáles son las validaciones técnicas del plan de pruebas?\n"
            "7. ¿Cuáles son los criterios de aceptación para considerar el cambio exitoso?"
        ),
    },
]

_EXTRACTION_SYSTEM_PROMPT = """You are an RFC data extraction assistant.

You will be given:
1. The questions that were asked to the user for this step (so you know what each field means)
2. Fields already collected (so you know what is still missing)
3. The user's latest message(s)

Your task: extract the user's answers and map them to the correct field names.
Return a JSON object with the field names as keys and the user's answers as values.
If a field was not answered in the user's message, omit it from the JSON (do not include null values).
Keep answers concise but complete — preserve the user's exact wording where possible.

Rules:
- Use the question text to understand what each field represents
- An answer does not need to use the exact field name — infer from context
- Do not invent or assume values not stated by the user
- If a user message contains a name, role, or contact detail in response to an approvers question, extract it as the "approvers" value

Respond with only the JSON object — no markdown, no explanation.
"""

_CONVERSATIONAL_SYSTEM_PROMPT = """You are Helena, a security operations AI assistant helping a user fill out an RFC (Request for Change).

You are currently on step {step} of 5 (open questions phase), covering: {topic}.

Your task:
1. Ask the questions for this step in a friendly, conversational way.
2. If the user has already provided some answers, acknowledge them and ask only for the missing ones.
3. If all answers for this step are complete, say so and let them know you'll move to the next step.

Missing fields that still need answers: {missing_fields}
Already collected fields: {collected_fields}

Rules:
- Be concise and professional.
- Respond in the same language the user writes in.
- Do NOT ask about fields from other steps.
- Do NOT output JSON — this is a natural language response shown directly to the user.
"""


async def rfc_open_questions_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("rfc_open_questions")

    api_key: str = resolve_api_key(config)
    client = get_openai_client(api_key)
    thread_id: str = state.get("thread_id", "unknown")
    rfc_data: dict = dict(state.get("rfc_data") or {})
    rfc_step: int = state.get("rfc_step", 0)

    # Determine current step config (1-5)
    current_step_idx = max(0, min(rfc_step, 4))  # clamp to 0-4 for indexing
    step_config = _STEP_CONFIGS[current_step_idx]
    step_num = step_config["step"]
    fields = step_config["fields"]
    topic = step_config["topic"]

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="rfc_open_questions",
        metadata={"thread_id": thread_id, "node": "rfc_open_questions", "step": step_num},
    )

    write = get_stream_writer()

    # --- Step 1: Emit progress event ---
    write({"type": "rfc_step_progress", "step": step_num, "total_open_steps": 5, "topic": topic})

    # --- Step 2: Extract answers from latest user message ---
    # Only extract if the user has sent a message (not first visit to this step)
    has_new_message = bool(state["messages"]) and getattr(state["messages"][-1], "type", "") == "human"
    extracted_fields: dict = {}

    if has_new_message:
        already_collected = {k: rfc_data[k] for k in fields if rfc_data.get(k)}
        still_missing = [f for f in fields if not rfc_data.get(f)]
        extraction_messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Step {step_num} — {topic}\n\n"
                    f"Questions asked to the user:\n{step_config['questions']}\n\n"
                    f"Fields already collected (skip these): {list(already_collected.keys())}\n"
                    f"Fields still needed: {still_missing}\n\n"
                    f"User's latest message:\n{state['messages'][-1].content}"
                ),
            },
        ]

        extraction_gen = trace.generation(
            name="rfc_extraction_llm",
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
            extracted_fields = json.loads(extraction_raw.strip())
            if not isinstance(extracted_fields, dict):
                extracted_fields = {}
        except (json.JSONDecodeError, ValueError):
            logger.warning("rfc_extraction_parse_failed", step=step_num, raw=extraction_raw[:200])
            extracted_fields = {}

        # Merge into rfc_data
        rfc_data.update({k: v for k, v in extracted_fields.items() if v})
        logger.info("rfc_fields_extracted", thread_id=thread_id, step=step_num, fields=list(extracted_fields.keys()))

    # --- Step 3: Determine missing fields for this step ---
    missing_fields = [f for f in fields if not rfc_data.get(f)]
    collected_fields = {f: rfc_data[f] for f in fields if rfc_data.get(f)}

    # --- Step 4: If all fields for this step are collected, advance ---
    step_complete = len(missing_fields) == 0
    new_rfc_step = rfc_step
    rfc_open_complete = state.get("rfc_open_complete", False)

    if step_complete and has_new_message:
        new_rfc_step = min(rfc_step + 1, 5)
        if new_rfc_step >= 5:
            rfc_open_complete = True
            logger.info("rfc_open_questions_complete", thread_id=thread_id)

    # --- Step 5: Generate conversational response ---
    collected_summary = ", ".join(f"{k}={repr(v)}" for k, v in collected_fields.items()) or "none yet"
    missing_summary = ", ".join(missing_fields) if missing_fields else "all answered"

    user_ctx_str = format_user_context(state)

    if step_complete and rfc_open_complete:
        # Transition message to closed questions
        conv_prompt = (
            "You have finished collecting all open-ended RFC information across 5 steps. "
            "Tell the user they've completed the open questions phase and that you'll now ask "
            "some quick structured questions (select from options) to finalize the RFC. "
            "Keep it brief and positive."
        ) + user_ctx_str
        conv_messages = [
            {"role": "system", "content": conv_prompt},
        ]
    elif step_complete and has_new_message:
        # Step complete, moving to next step
        next_step_config = _STEP_CONFIGS[new_rfc_step] if new_rfc_step < 5 else None
        next_topic = next_step_config["topic"] if next_step_config else ""
        conv_prompt = (
            f"You have collected all answers for step {step_num} ({topic}). "
            f"Briefly acknowledge the completion of this step and transition to step {new_rfc_step + 1}: {next_topic}. "
            f"Then ask the questions for the next step."
        ) + user_ctx_str
        next_questions = _STEP_CONFIGS[new_rfc_step]["questions"] if new_rfc_step < 5 else ""
        conv_messages = [
            {"role": "system", "content": conv_prompt},
            {"role": "assistant", "content": f"Next step questions to present:\n{next_questions}"},
        ]
    else:
        # Ask/re-ask questions for current step
        conv_messages = [
            {
                "role": "system",
                "content": _CONVERSATIONAL_SYSTEM_PROMPT.format(
                    step=step_num,
                    topic=topic,
                    missing_fields=missing_summary,
                    collected_fields=collected_summary,
                ) + user_ctx_str,
            },
            {
                "role": "user",
                "content": (
                    state["messages"][-1].content
                    if has_new_message
                    else f"Start step {step_num}: {_STEP_CONFIGS[current_step_idx]['questions']}"
                ),
            },
        ]

    conv_gen = trace.generation(
        name="rfc_conversational_llm",
        model="gpt-5",
        input={"messages": conv_messages},
    )

    conv_stream = await client.chat.completions.create(
        model="gpt-5",
        messages=conv_messages,
        stream=True,
        stream_options={"include_usage": True},
    )

    full_response = ""
    prompt_tokens = 0
    completion_tokens = 0

    async for chunk in conv_stream:
        delta = chunk.choices[0].delta.content if chunk.choices else ""
        if delta:
            write({"type": "token", "content": delta})
            full_response += delta
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

    conv_gen.end(
        output=full_response,
        usage={"input": prompt_tokens, "output": completion_tokens},
    )

    return {
        "messages": [AIMessage(content=full_response)],
        "rfc_data": rfc_data,
        "rfc_step": new_rfc_step,
        "rfc_open_complete": rfc_open_complete,
    }
