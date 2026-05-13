from typing import Literal

from langgraph.graph import END, START, StateGraph

from ..agents.fallback import fallback_response_node
from ..agents.rfc_closed import rfc_closed_questions_node
from ..agents.rfc_execute import rfc_execute_node
from ..agents.rfc_open import rfc_open_questions_node
from ..agents.rfc_reuse_confirm import rfc_reuse_confirm_node
from ..agents.rfc_reuse_validate import rfc_reuse_validate_node
from ..agents.rfc_summary import rfc_summary_confirm_node
from ..agents.triage import route_from_triage, triage_node
from .state import AgentState


# ── Original RFC flow routing ────────────────────────────────────────────────

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
    """After summary confirmation: proceed to execute phase."""
    if state.get("rfc_confirmed"):
        return "rfc_execute"
    return END


# ── RFC reuse flow routing ───────────────────────────────────────────────────

def _route_from_rfc_reuse_validate(
    state: AgentState,
) -> Literal["rfc_reuse_confirm", "__end__"]:
    """After validation: if valid, proceed to confirmation. Otherwise wait for user input."""
    if state.get("rfc_reuse_validated"):
        return "rfc_reuse_confirm"
    return END


def _route_from_rfc_reuse_confirm(
    state: AgentState,
) -> Literal["rfc_reuse_validate", "__end__"]:
    """After confirmation: if user corrected, re-validate. Otherwise done (execute already emitted or waiting)."""
    if state.get("rfc_reuse_confirmed"):
        return END  # execute_workflow already emitted
    if not state.get("rfc_reuse_validated"):
        return "rfc_reuse_validate"  # Re-validate after correction
    return END  # Wait for user confirmation


# ── Graph construction ───────────────────────────────────────────────────────

def build_graph(checkpointer):
    graph = StateGraph(AgentState)

    graph.add_node("triage", triage_node)
    graph.add_node("rfc_open_questions", rfc_open_questions_node)
    graph.add_node("rfc_closed_questions", rfc_closed_questions_node)
    graph.add_node("rfc_summary_confirm", rfc_summary_confirm_node)
    graph.add_node("rfc_execute", rfc_execute_node)
    graph.add_node("fallback_response", fallback_response_node)

    # RFC reuse nodes
    graph.add_node("rfc_reuse_validate", rfc_reuse_validate_node)
    graph.add_node("rfc_reuse_confirm", rfc_reuse_confirm_node)

    graph.add_edge(START, "triage")

    graph.add_conditional_edges(
        "triage",
        route_from_triage,
        {
            "rfc_open_questions": "rfc_open_questions",
            "rfc_closed_questions": "rfc_closed_questions",
            "rfc_summary_confirm": "rfc_summary_confirm",
            "rfc_execute": "rfc_execute",
            "rfc_reuse_validate": "rfc_reuse_validate",
            "rfc_reuse_confirm": "rfc_reuse_confirm",
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

    # RFC reuse edges
    graph.add_conditional_edges(
        "rfc_reuse_validate",
        _route_from_rfc_reuse_validate,
        {"rfc_reuse_confirm": "rfc_reuse_confirm", END: END},
    )
    graph.add_conditional_edges(
        "rfc_reuse_confirm",
        _route_from_rfc_reuse_confirm,
        {"rfc_reuse_validate": "rfc_reuse_validate", END: END},
    )

    return graph.compile(checkpointer=checkpointer)
