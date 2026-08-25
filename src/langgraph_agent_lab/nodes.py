"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, ApprovalDecision, make_event


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict[str, Any]:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── Classification Structured Output Schema ────────────────────────
class ClassificationOutput(BaseModel):
    """Structured output schema for intent classification."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description=(
            "Route based on priority: risky > tool > missing_info > error > simple.\n"
            "- risky: sensitive actions with side effects (refunds, deletion, emails, cancel).\n"


            "- tool: information lookup (order status, tracking, database search).\n"
            "- missing_info: vague, incomplete, non-actionable queries missing details.\n"
            "- error: system failure, timeout, crash, technical outage.\n"
            "- simple: general informational questions, FAQs answerable directly."
        )
    )
    risk_level: Literal["low", "high"] = Field(
        description="Set to 'high' for risky actions, 'low' for all other routes."
    )
    reasoning: str = Field(description="Brief explanation for the classification decision.")


def classify_node(state: AgentState) -> dict[str, Any]:
    """Classify the query into a route using an LLM with structured output."""
    query = state.get("query", "").strip()

    prompt = (
        "You are an expert customer support triage agent. "
        "Classify the following customer ticket query into exactly one of five routes: "
        "'simple', 'tool', 'missing_info', 'risky', or 'error'.\n\n"
        "Strict Priority Rule: risky > tool > missing_info > error > simple\n\n"
        "Routing Definitions:\n"
        "- risky: Actions involving modifications, side effects, money/refunds, deletions, "
        "or external communications.\n"
        "- tool: Specific information lookups or queries checking order, database, or status.\n"
        "- missing_info: Vague or underspecified queries lacking actionable context "
        "(e.g. 'Can you fix it?').\n"
        "- error: System failures, timeouts, crashes, or unrecoverable technical errors.\n"
        "- simple: General FAQ or how-to questions answered directly with standard knowledge.\n\n"
        f"Customer Query: {query}"
    )


    route: str = "simple"
    risk_level: str = "low"
    reasoning: str = ""
    errors: list[str] = []
    event_type = "completed"
    event_message = ""
    error_type: str | None = None

    try:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(ClassificationOutput)
        decision = structured_llm.invoke(prompt)

        if isinstance(decision, ClassificationOutput):
            parsed = decision
        else:
            candidate = decision if isinstance(decision, dict) else {
                "route": getattr(decision, "route", "simple"),
                "risk_level": getattr(decision, "risk_level", "low"),
                "reasoning": getattr(decision, "reasoning", ""),
            }
            parsed = ClassificationOutput.model_validate(candidate)

        route = str(parsed.route)
        risk_level = "high" if route == "risky" else "low"
        reasoning = str(parsed.reasoning)
        event_message = f"Classified route={route} risk={risk_level}"
    except Exception as exc:
        error_type = type(exc).__name__
        route = "simple"
        risk_level = "low"
        reasoning = f"LLM classification fallback: {error_type}"
        errors = [f"classify LLM fallback: {error_type}"]
        event_type = "fallback"
        event_message = "LLM classification fallback"

    return {
        "route": route,
        "risk_level": risk_level,
        "errors": errors,
        "events": [
            make_event(
                "classify",
                event_type,
                event_message,
                route=route,
                risk_level=risk_level,
                reasoning=reasoning,
                error_type=error_type,
            )
        ],
    }


def tool_node(state: AgentState) -> dict[str, Any]:
    """Execute a mock tool call with error simulation."""
    route = state.get("route", "")
    attempt = state.get("attempt", 0)
    query = state.get("query", "")

    if route == "risky":
        approval = state.get("approval")
        approved = (
            bool(approval.get("approved", False))
            if isinstance(approval, dict)
            else bool(getattr(approval, "approved", False))
        )
        if not approved:
            return {
                "errors": ["Risky tool blocked: approval required"],
                "events": [
                    make_event(
                        "tool",
                        "blocked",
                        "Risky tool blocked before approval",
                        attempt=attempt,
                    )
                ],
            }

    # If route is error and attempt < 2, simulate transient failure
    if route == "error" and attempt < 2:
        result = f"ERROR: Transient failure during tool execution on attempt {attempt}"
    else:
        result = f"SUCCESS: Tool executed successfully for query '{query}'"

    return {
        "tool_results": [result],
        "events": [
            make_event(
                "tool",
                "completed",
                f"Tool executed on attempt {attempt}",
                attempt=attempt,
                result=result,
            )
        ],
    }


def evaluate_node(state: AgentState) -> dict[str, Any]:
    """Evaluate tool results to drive retry-loop gating."""
    tool_results = state.get("tool_results", [])
    latest_result = tool_results[-1] if tool_results else ""

    if "ERROR" in latest_result:
        evaluation_result = "needs_retry"
    else:
        evaluation_result = "success"

    return {
        "evaluation_result": evaluation_result,
        "events": [
            make_event(
                "evaluate",
                "verdict",
                f"Evaluation verdict: {evaluation_result}",
                verdict=evaluation_result,
                latest_result=latest_result,
            )
        ],
    }


def answer_node(state: AgentState) -> dict[str, Any]:
    """Generate a grounded final response using an LLM."""
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")
    route = state.get("route", "")

    context_parts = [f"User Query: {query}"]
    if tool_results:
        context_parts.append("Tool Results:\n" + "\n".join(f"- {r}" for r in tool_results))
    if approval:
        context_parts.append(f"Approval Decision: {approval}")
    if route:
        context_parts.append(f"Route: {route}")

    context = "\n\n".join(context_parts)
    prompt = (
        "You are a helpful and professional customer support assistant. "
        "Answer the user's inquiry based strictly on the provided context below. "
        "Be concise, clear, and polite.\n\n"
        f"Context:\n{context}\n\n"
        "Response:"
    )

    try:
        llm = get_llm(temperature=0.0)
        response = llm.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            final_answer = "\n".join(str(c) for c in content)
        else:
            final_answer = str(content)
    except Exception as exc:
        error_type = type(exc).__name__
        return {
            "final_answer": (
                "I could not generate a complete answer because the language model "
                "is unavailable. Please retry or contact support."
            ),
            "errors": [f"answer LLM fallback: {error_type}"],
            "events": [
                make_event(
                    "answer",
                    "fallback",
                    "LLM answer fallback",
                    error_type=error_type,
                )
            ],
        }

    return {
        "final_answer": final_answer,
        "events": [make_event("answer", "completed", "Generated final answer")],
    }


def ask_clarification_node(state: AgentState) -> dict[str, Any]:
    """Ask for missing information or handle rejected approval."""
    query = state.get("query", "")
    approval = state.get("approval")

    if approval and not approval.get("approved", True):
        comment = approval.get("comment", "Action rejected by supervisor.")
        question = (
            f"Your request '{query}' could not be approved ({comment}). "
            "Please provide additional verification or details to proceed."
        )
    else:
        question = (
            f"We need more information to assist you with: '{query}'. "
            "Could you please specify the order number, account email, or specific issue details?"
        )

    return {
        "pending_question": question,
        "final_answer": question,
        "events": [
            make_event("clarify", "requested", "Clarification requested", question=question)
        ],
    }


def risky_action_node(state: AgentState) -> dict[str, Any]:
    """Prepare a risky action for human approval."""
    query = state.get("query", "")
    proposed_action = (
        f"Execute sensitive action for query: '{query}'. Requires human supervisor authorization."
    )

    return {
        "proposed_action": proposed_action,
        "events": [
            make_event(
                "risky_action",
                "proposed",
                "Prepared proposed risky action",
                action=proposed_action,
            )
        ],
    }


def approval_node(state: AgentState) -> dict[str, Any]:
    """Human-in-the-loop approval step with mock default and interrupt extension."""
    if os.getenv("LANGGRAPH_INTERRUPT", "false").lower() == "true":
        try:
            from langgraph.types import interrupt

            decision = interrupt(
                {
                    "message": "Approval required for risky action",
                    "proposed_action": state.get("proposed_action"),
                    "query": state.get("query"),
                }
            )
        except Exception as exc:
            decision = {
                "approved": False,
                "reviewer": "system",
                "comment": f"Approval unavailable: {type(exc).__name__}",
            }
    else:
        decision = {
            "approved": True,
            "reviewer": "mock-reviewer",
            "comment": "Approved by policy reviewer",
        }

    if isinstance(decision, ApprovalDecision):
        decision = decision.model_dump()
    else:
        decision = ApprovalDecision.model_validate(decision).model_dump()

    return {
        "approval": decision,
        "events": [
            make_event(
                "approval",
                "decision",
                f"Approval decision: approved={decision.get('approved')}",
                approved=decision.get("approved"),
                reviewer=decision.get("reviewer"),
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict[str, Any]:
    """Record a retry attempt and increment attempt counter."""
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    new_attempt = attempt + 1
    error_msg = f"Retry attempt {new_attempt}/{max_attempts} due to failure"

    return {
        "attempt": new_attempt,
        "errors": [error_msg],
        "events": [
            make_event(
                "retry",
                "attempted",
                f"Incremented attempt to {new_attempt}",
                attempt=new_attempt,
                max_attempts=max_attempts,
            )
        ],
    }


def dead_letter_node(state: AgentState) -> dict[str, Any]:
    """Handle unresolvable failures after max retries exceeded."""
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    msg = (
        f"Unable to complete request after {attempt}/{max_attempts} attempts. "
        "The request has been routed to human operations (dead letter)."
    )

    return {
        "final_answer": msg,
        "events": [
            make_event(
                "dead_letter",
                "exhausted",
                "Max retries exhausted; routed to dead letter",
                attempt=attempt,
                max_attempts=max_attempts,
            )
        ],
    }


def finalize_node(state: AgentState) -> dict[str, Any]:
    """Emit a final audit event. All routes must pass through here before END."""
    return {
        "events": [make_event("finalize", "completed", "workflow finished")],
    }
