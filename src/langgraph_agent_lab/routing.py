"""Routing functions for conditional edges.

Each function takes AgentState and returns a string — the name of the next node.
These strings MUST match node names registered in graph.py.
"""

from __future__ import annotations

from .state import AgentState


def route_after_classify(state: AgentState) -> str:
    """Map classified route to the next graph node.

    Mapping:
    - "simple"       -> "answer"
    - "tool"         -> "tool"
    - "missing_info" -> "clarify"
    - "risky"        -> "risky_action"
    - "error"        -> "retry"
    - unknown/default -> "answer"
    """
    route = state.get("route", "")
    mapping = {
        "simple": "answer",
        "tool": "tool",
        "missing_info": "clarify",
        "risky": "risky_action",
        "error": "retry",
    }
    return mapping.get(route, "answer")


def route_after_evaluate(state: AgentState) -> str:
    """Decide if tool result is satisfactory or needs retry.

    - If evaluation_result == "needs_retry" -> "retry"
    - Otherwise -> "answer"
    """
    verdict = state.get("evaluation_result")
    if verdict == "needs_retry":
        return "retry"
    return "answer"


def route_after_retry(state: AgentState) -> str:
    """Decide whether to retry the tool or give up.

    MUST be bounded.
    - If attempt < max_attempts -> "tool" (try again)
    - If attempt >= max_attempts -> "dead_letter" (give up, escalate)
    """
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    if attempt < max_attempts:
        return "tool"
    return "dead_letter"


def route_after_approval(state: AgentState) -> str:
    """Route based on human approval decision.

    - If approved -> "tool" (proceed with risky action)
    - If rejected -> "clarify" (ask user for alternative)
    """
    approval = state.get("approval")
    if isinstance(approval, dict):
        is_approved = bool(approval.get("approved", False))
    elif hasattr(approval, "approved"):
        is_approved = bool(getattr(approval, "approved", False))
    else:
        is_approved = False

    if is_approved:
        return "tool"
    return "clarify"


def route_after_guardrail(state: AgentState) -> str:
    """Route based on prompt security evaluation.

    - If safe (is_safe == True) -> "query_rewrite" (proceed to query decomposition/refinement)
    - If unsafe (is_safe == False) -> "clarify" (fail-fast refusal)
    """
    is_safe = state.get("is_safe", True)
    if is_safe:
        return "query_rewrite"
    return "clarify"

