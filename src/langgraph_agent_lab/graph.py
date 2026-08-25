from __future__ import annotations

from typing import TYPE_CHECKING

from .state import AgentState

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    interrupt_before: list[str] | None = None,
) -> CompiledStateGraph:
    """Build and compile the LangGraph workflow with parallel fan-out and full orchestration."""
    from langgraph.graph import END, START, StateGraph

    from .nodes import (
        aggregate_answers_node,
        answer_node,
        approval_node,
        ask_clarification_node,
        classify_node,
        dead_letter_node,
        evaluate_node,
        finalize_node,
        intake_node,
        parallel_worker_node,
        prompt_guardrail_node,
        query_rewrite_node,
        retry_or_fallback_node,
        risky_action_node,
        tool_node,
    )
    from .routing import (
        route_after_approval,
        route_after_classify,
        route_after_evaluate,
        route_after_guardrail,
        route_after_retry,
        route_after_rewrite,
    )

    builder = StateGraph(AgentState)

    # 1. Register all 15 nodes
    builder.add_node("intake", intake_node)
    builder.add_node("prompt_guardrail", prompt_guardrail_node)
    builder.add_node("query_rewrite", query_rewrite_node)
    builder.add_node("parallel_worker", parallel_worker_node)
    builder.add_node("aggregate_answers", aggregate_answers_node)
    builder.add_node("classify", classify_node)
    builder.add_node("answer", answer_node)
    builder.add_node("tool", tool_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("clarify", ask_clarification_node)
    builder.add_node("risky_action", risky_action_node)
    builder.add_node("approval", approval_node)
    builder.add_node("retry", retry_or_fallback_node)
    builder.add_node("dead_letter", dead_letter_node)
    builder.add_node("finalize", finalize_node)

    # 2. Fixed edges
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "prompt_guardrail")
    builder.add_edge("parallel_worker", "aggregate_answers")
    builder.add_edge("aggregate_answers", "finalize")
    builder.add_edge("tool", "evaluate")
    builder.add_edge("risky_action", "approval")
    builder.add_edge("answer", "finalize")
    builder.add_edge("clarify", "finalize")
    builder.add_edge("dead_letter", "finalize")
    builder.add_edge("finalize", END)

    # 3. Conditional edges
    builder.add_conditional_edges(
        "prompt_guardrail",
        route_after_guardrail,
        {
            "query_rewrite": "query_rewrite",
            "clarify": "clarify",
        },
    )

    builder.add_conditional_edges(
        "query_rewrite",
        route_after_rewrite,
        ["parallel_worker", "classify"],
    )

    builder.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "answer": "answer",
            "tool": "tool",
            "clarify": "clarify",
            "risky_action": "risky_action",
            "retry": "retry",
        },
    )

    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "retry": "retry",
            "answer": "answer",
        },
    )

    builder.add_conditional_edges(
        "retry",
        route_after_retry,
        {
            "tool": "tool",
            "dead_letter": "dead_letter",
        },
    )

    builder.add_conditional_edges(
        "approval",
        route_after_approval,
        {
            "tool": "tool",
            "clarify": "clarify",
        },
    )

    return builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


