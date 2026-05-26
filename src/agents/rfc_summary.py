import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..llm import get_openai_client, resolve_api_key
from ..observability import get_langfuse, record_node_invocation
from ..prompts import fetch_active_prompt

logger = structlog.get_logger(__name__)

_SUMMARY_SYSTEM_PROMPT = """Eres Helena, una asistente de IA para operaciones de seguridad.

Has recopilado toda la información necesaria para un RFC (Solicitud de Cambio).
Genera un resumen claro y profesional del RFC con base en los datos proporcionados.

Estructura el resumen como un documento legible con las siguientes secciones:
- **Descripción del Cambio** (título, descripción, tipo, categoría)
- **Justificación e Impacto al Negocio** (justificación, sistemas/usuarios afectados, nivel de impacto)
- **Plan de Implementación** (pasos, plan de rollback, dependencias, recursos)
- **Ventana de Cambio** (fecha, hora de inicio, hora de término, entorno)
- **Riesgo y Pruebas** (nivel de riesgo, mitigación, plan de pruebas, aprobadores)
- **Parámetros Clasificatorios** (tipo de cambio, impacto en servicio, monitoreo, acceso remoto, backup, rollback, ubicación de ejecución)

{change_type_instruction}

Al finalizar el resumen, pide confirmación al usuario:
"¿Este RFC es correcto? Responde **confirmar** para enviarlo, o indícame qué necesita corregirse."

Responde en español por defecto. Si el usuario escribe en otro idioma, responde en ese idioma.
"""

_SUGGEST_CHANGE_TYPE_INSTRUCTION = """IMPORTANT: The user requested that you suggest the most appropriate change type.
Based on ALL the RFC data collected (impact, risk level, urgency, scope, rollback plans, etc.),
determine the best change type from: Recurrente, Normal, Estándar, Emergencia.
Include your recommendation and a brief justification in the **Change Overview** section under "Tipo de cambio (sugerido)"."""

_EXPLICIT_CHANGE_TYPE_INSTRUCTION = ""

_CORRECTION_SYSTEM_PROMPT = """Eres Helena, una asistente de IA para operaciones de seguridad.

El usuario desea corregir algo en su RFC. La corrección solicitada es:
"{correction}"

Datos actuales del RFC:
{rfc_data}

Confirma la corrección, indica qué fue actualizado y presenta un resumen breve actualizado.
Luego solicita nuevamente la confirmación para enviar.

Responde en español por defecto. Si el usuario escribe en otro idioma, responde en ese idioma.
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
    write({"type": "rfc_step_progress", "step": 7, "total_open_steps": 7, "topic": "Resumen y Confirmación"})

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
                    "El usuario ha confirmado el resumen del RFC. "
                    "Confirma brevemente que la información del RFC ha sido aprobada. "
                    "Indícale que ahora se mostrarán los flujos de trabajo que serán ejecutados y que se solicitará confirmación final antes de enviarlo. "
                    "Sé conciso. NO indiques que ya se está enviando. "
                    "Responde en español por defecto. Si el usuario escribe en otro idioma, responde en ese idioma."
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
        change_type = rfc_data.get("rfc_change_type", "")
        if change_type == "Sugiéreme uno":
            change_type_instruction = _SUGGEST_CHANGE_TYPE_INSTRUCTION
        else:
            change_type_instruction = _EXPLICIT_CHANGE_TYPE_INSTRUCTION
        summary_prompt = await fetch_active_prompt("RFC_SUMMARY", _SUMMARY_SYSTEM_PROMPT)
        summary_prompt = summary_prompt.format(change_type_instruction=change_type_instruction)
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
