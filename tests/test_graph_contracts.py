from types import SimpleNamespace

from langgraph.checkpoint.memory import MemorySaver

import langgraph_agent_lab.nodes as nodes
from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.state import Route, Scenario, initial_state, make_event


class _FakeStructured:
    def invoke(self, prompt):
        query = prompt.split("Customer Query:", 1)[-1].lower()
        route = "risky" if "delete" in query else "error"
        risk = "high" if route == "risky" else "low"
        return nodes.ClassificationOutput(route=route, risk_level=risk, reasoning="test")


class _FakeLLM:
    def with_structured_output(self, _schema):
        return _FakeStructured()

    def invoke(self, _prompt):
        return SimpleNamespace(content="test answer")


def test_rejected_risky_path_never_runs_tool(monkeypatch):
    monkeypatch.setattr(nodes, "get_llm", lambda **_kwargs: _FakeLLM())

    def reject_approval(_state):
        return {
            "approval": {"approved": False, "reviewer": "test", "comment": "rejected"},
            "events": [make_event("approval", "decision", "rejected", approved=False)],
        }

    monkeypatch.setattr(nodes, "approval_node", reject_approval)
    graph = build_graph(checkpointer=MemorySaver(), interrupt_before=[])
    scenario = Scenario(
        id="rejected-risky",
        query="Delete the customer account",
        expected_route=Route.RISKY,
    )
    state = initial_state(scenario)
    result = graph.invoke(
        state,
        config={"configurable": {"thread_id": state["thread_id"]}},
    )


    event_nodes = [event["node"] for event in result["events"]]
    assert event_nodes == ["intake", "prompt_guardrail", "query_rewrite", "classify", "risky_action", "approval", "clarify", "finalize"]
    assert "tool" not in event_nodes


def test_dead_letter_path_stops_at_retry_bound(monkeypatch):
    monkeypatch.setattr(nodes, "get_llm", lambda **_kwargs: _FakeLLM())
    graph = build_graph(checkpointer=MemorySaver())
    scenario = Scenario(
        id="dead-letter",
        query="System failure cannot recover",
        expected_route=Route.ERROR,
        max_attempts=1,
    )
    state = initial_state(scenario)
    result = graph.invoke(
        state,
        config={"configurable": {"thread_id": state["thread_id"]}},
    )

    event_nodes = [event["node"] for event in result["events"]]
    assert event_nodes == ["intake", "prompt_guardrail", "query_rewrite", "classify", "retry", "dead_letter", "finalize"]
    assert result["attempt"] == 1

