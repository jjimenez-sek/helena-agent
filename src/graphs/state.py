from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    thread_id: str
    # Conversation context (stable for the lifetime of the conversation)
    project_id: str          # project linked to this conversation
    conversation_id: str     # NestJS conversation UUID (used by execute_workflow SSE event)
    project_workflows: list  # [{workflow_id, name, description}] from project_steps
    # Triage
    intent: str  # "rfc" | "incident" | "knowledge" | "escalation" | "unknown"
    # RFC chain state
    rfc_step: int           # 0 = not started, 1-5 = open question steps, 6 = closed, 7 = summary/confirm
    rfc_data: dict          # accumulated RFC fields across all steps
    rfc_open_complete: bool
    rfc_closed_complete: bool
    rfc_confirmed: bool
    rfc_execute_confirmed: bool  # True once user confirms the execute step (prevents re-execution)
