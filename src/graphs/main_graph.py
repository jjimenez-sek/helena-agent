from typing import Literal

from langgraph.graph import END, START, StateGraph

from ..agents.fallback import fallback_response_node
from ..agents.rfc_closed import rfc_closed_questions_node
from ..agents.rfc_execute import rfc_execute_node
from ..agents.rfc_open import rfc_open_questions_node
from ..agents.rfc_summary import rfc_summary_confirm_node
from ..agents.triage import route_from_triage, triage_node
from .state import AgentState


def _route_from_rfc_open(state: AgentState) -> Literal["rfc_closed_questions", "__end__"]:
    """After open questions: if all done, proceed to closed questions immediately."""
    if state.get("rfc_open_complete"):
        return "rfc_closed_questions"
    return END


def _route_from_rfc_closed(state: AgentState) -> Literal["rfc_summary_confirm", "__end__"]:
    """After closed questions: if all done, proceed to summary immediately."""
    if state.get("rfc_closed_complete"):
        return "rfc_summary_confirm"
    return END


def _route_from_rfc_summary(state: AgentState) -> Literal["rfc_execute", "__end__"]:
    """After summary confirmation: proceed to execute phase (which will present workflows and ask for final confirm)."""
    if state.get("rfc_confirmed"):
        return "rfc_execute"
    return END


def build_graph(checkpointer):
    graph = StateGraph(AgentState)

    graph.add_node("triage", triage_node)
    graph.add_node("rfc_open_questions", rfc_open_questions_node)
    graph.add_node("rfc_closed_questions", rfc_closed_questions_node)
    graph.add_node("rfc_summary_confirm", rfc_summary_confirm_node)
    graph.add_node("rfc_execute", rfc_execute_node)
    graph.add_node("fallback_response", fallback_response_node)

    graph.add_edge(START, "triage")

    graph.add_conditional_edges(
        "triage",
        route_from_triage,
        {
            "rfc_open_questions": "rfc_open_questions",
            "rfc_closed_questions": "rfc_closed_questions",
            "rfc_summary_confirm": "rfc_summary_confirm",
            "rfc_execute": "rfc_execute",
            "fallback_response": "fallback_response",
        },
    )

    # Auto-advance when a phase completes within the same turn
    graph.add_conditional_edges(
        "rfc_open_questions",
        _route_from_rfc_open,
        {"rfc_closed_questions": "rfc_closed_questions", END: END},
    )
    graph.add_conditional_edges(
        "rfc_closed_questions",
        _route_from_rfc_closed,
        {"rfc_summary_confirm": "rfc_summary_confirm", END: END},
    )

    graph.add_conditional_edges(
        "rfc_summary_confirm",
        _route_from_rfc_summary,
        {"rfc_execute": "rfc_execute", END: END},
    )
    graph.add_edge("rfc_execute", END)
    graph.add_edge("fallback_response", END)

    return graph.compile(checkpointer=checkpointer)
