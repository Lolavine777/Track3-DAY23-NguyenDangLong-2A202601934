import langgraph.types as langgraph_types

import langgraph_agent_lab.nodes as nodes
from langgraph_agent_lab.metrics import metric_from_state
from langgraph_agent_lab.state import make_event


def test_classify_llm_failure_is_audited(monkeypatch):
    def fail(**_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(nodes, "get_llm", fail)

    update = nodes.classify_node({"query": "How do I reset my password?"})

    assert update["route"] == "simple"
    assert update["errors"] == ["classify LLM fallback: RuntimeError"]
    assert update["events"][0]["event_type"] == "fallback"
    assert update["events"][0]["metadata"]["error_type"] == "RuntimeError"


def test_classify_normalizes_risk_level_from_route(monkeypatch):
    class FakeStructured:
        def invoke(self, _prompt):
            return nodes.ClassificationOutput(
                route="risky", risk_level="low", reasoning="inconsistent test output"
            )

    class FakeLLM:
        def with_structured_output(self, _schema):
            return FakeStructured()

    monkeypatch.setattr(nodes, "get_llm", lambda **_kwargs: FakeLLM())

    update = nodes.classify_node({"query": "Delete the account"})

    assert update["risk_level"] == "high"


def test_answer_llm_failure_is_audited_and_not_presented_as_success(monkeypatch):
    def fail(**_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(nodes, "get_llm", fail)

    update = nodes.answer_node({"query": "How do I reset my password?"})

    assert "could not generate" in update["final_answer"].lower()
    assert update["errors"] == ["answer LLM fallback: RuntimeError"]
    assert update["events"][0]["event_type"] == "fallback"


def test_risky_tool_is_blocked_without_approval():
    update = nodes.tool_node(
        {
            "route": "risky",
            "query": "Delete the account",
            "attempt": 0,
            "approval": {"approved": False},
        }
    )

    assert update.get("tool_results", []) == []
    assert update["errors"] == ["Risky tool blocked: approval required"]
    assert update["events"][0]["event_type"] == "blocked"


def test_interrupt_failure_fails_closed(monkeypatch):
    monkeypatch.setenv("LANGGRAPH_INTERRUPT", "true")

    def fail(_payload):
        raise RuntimeError("interrupt unavailable")

    monkeypatch.setattr(langgraph_types, "interrupt", fail)

    update = nodes.approval_node({"proposed_action": "Delete the account"})

    assert update["approval"]["approved"] is False
    assert "RuntimeError" in update["approval"]["comment"]


def test_metrics_fail_when_graph_contains_llm_fallback():
    state = {
        "scenario_id": "fallback",
        "route": "simple",
        "final_answer": "The model was unavailable.",
        "events": [
            make_event("intake", "completed", "ok"),
            make_event("classify", "fallback", "fallback", error_type="RuntimeError"),
            make_event("answer", "fallback", "fallback", error_type="RuntimeError"),
            make_event("finalize", "completed", "workflow finished"),
        ],
        "errors": ["classify LLM fallback: RuntimeError", "answer LLM fallback: RuntimeError"],
        "approval": None,
    }

    metric = metric_from_state(state, expected_route="simple", approval_required=False)

    assert metric.success is False
