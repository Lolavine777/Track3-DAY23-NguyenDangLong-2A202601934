"""Tests for Parallel Workers Fan-Out using LangGraph Send()."""

from types import SimpleNamespace

import langgraph_agent_lab.nodes as nodes
from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state


class _MockStructured:
    def __init__(self, query: str):
        self.query = query

    def invoke(self, prompt: str):
        if "pre-processor" in prompt:
            # Query rewrite mock: if query has 'AND', decompose into sub_queries
            if "and" in self.query.lower():
                parts = [p.strip() for p in self.query.split("and")]
                return nodes.RewriteOutput(
                    rewritten_query=self.query,
                    sub_queries=parts,
                    is_multi_intent=True,
                    reasoning="Decomposed multi-intent query",
                )
            return nodes.RewriteOutput(
                rewritten_query=self.query,
                sub_queries=[self.query],
                is_multi_intent=False,
                reasoning="Single intent",
            )

        if "guardrail" in prompt.lower():
            return nodes.GuardrailOutput(
                is_safe=True,
                risk_category="safe",
                reasoning="safe query",
            )

        return nodes.ClassificationOutput(
            route="simple",
            risk_level="low",
            reasoning="test",
        )


class _MockLLM:
    def __init__(self, query: str = ""):
        self.query = query

    def with_structured_output(self, _schema):
        return _MockStructured(self.query)

    def invoke(self, prompt: str):
        return SimpleNamespace(content=f"Processed response for prompt: {prompt[:30]}...")


def test_parallel_worker_fanout_execution(monkeypatch):
    """Verify that multi-intent query fans out to parallel workers and aggregates answers."""
    query = "Check status of order 12345 and explain return policy"
    monkeypatch.setattr(nodes, "get_llm", lambda **_kwargs: _MockLLM(query))

    saver = build_checkpointer("memory")
    graph = build_graph(checkpointer=saver)

    scenario = Scenario(
        id="test_parallel",
        query=query,
        expected_route=Route.TOOL,
    )
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": "thread-test-parallel-fanout"}}

    result = graph.invoke(state, config=config)

    assert result["is_multi_intent"] is True
    assert len(result["sub_queries"]) == 2
    assert len(result["sub_answers"]) == 2
    assert result["route"] == "parallel_multi_intent"
    assert result["final_answer"] is not None

    events = result.get("events", [])
    worker_events = [e for e in events if e.get("node") == "parallel_worker"]
    aggregate_events = [e for e in events if e.get("node") == "aggregate_answers"]
    finalize_events = [e for e in events if e.get("node") == "finalize"]

    assert len(worker_events) == 2, "Expected 2 parallel worker execution events"
    assert len(aggregate_events) == 1, "Expected 1 aggregation event"
    assert len(finalize_events) == 1, "Workflow must conclude at finalize"


def test_single_intent_bypasses_parallel_workers(monkeypatch):
    """Verify that single-intent queries bypass parallel workers and use standard classify."""
    query = "How do I reset my password?"
    monkeypatch.setattr(nodes, "get_llm", lambda **_kwargs: _MockLLM(query))

    saver = build_checkpointer("memory")
    graph = build_graph(checkpointer=saver)

    scenario = Scenario(
        id="test_single",
        query=query,
        expected_route=Route.SIMPLE,
    )
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": "thread-test-single-bypass"}}

    result = graph.invoke(state, config=config)

    assert result["is_multi_intent"] is False
    assert len(result["sub_answers"]) == 0
    assert result["route"] == "simple"

    events = result.get("events", [])
    worker_events = [e for e in events if e.get("node") == "parallel_worker"]
    classify_events = [e for e in events if e.get("node") == "classify"]

    assert len(worker_events) == 0, "Single-intent should not invoke parallel workers"
    assert len(classify_events) == 1, "Single-intent should invoke classify node"
