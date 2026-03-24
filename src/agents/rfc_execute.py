import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..graphs.state import AgentState
from ..observability import record_node_invocation

logger = structlog.get_logger(__name__)


async def rfc_execute_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    record_node_invocation("rfc_execute")

    thread_id: str = state.get("thread_id", "unknown")
    rfc_data: dict = dict(state.get("rfc_data") or {})
    conversation_id: str = state.get("conversation_id", "")
    project_workflows: list = list(state.get("project_workflows") or [])

    write = get_stream_writer()

    workflows_to_run = [
        {"workflow_id": wf["workflow_id"], "name": wf.get("name", "workflow")}
        for wf in project_workflows
        if wf.get("workflow_id")
    ]

    # User already confirmed the RFC in rfc_summary_confirm — execute immediately.
    # NestJS resolves webhook URLs, calls n8n, formats the result with OpenAI,
    # and emits the result as a new chat bubble.
    write({
        "type": "execute_workflow",
        "conversation_id": conversation_id,
        "workflows": workflows_to_run,
        "rfc_data": rfc_data,
    })

    logger.info(
        "rfc_execute_triggered",
        thread_id=thread_id,
        conversation_id=conversation_id,
        workflow_count=len(workflows_to_run),
        workflow_ids=[w["workflow_id"] for w in workflows_to_run],
    )

    return {
        "messages": [],
        "rfc_execute_confirmed": True,
    }
