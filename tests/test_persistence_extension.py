"""Tests for persistence and recovery extensions."""

from types import SimpleNamespace

import langgraph_agent_lab.nodes as nodes
from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state


class _FakeStructured:
    def invoke(self, prompt):
        query = prompt.split("Customer Query:", 1)[-1].lower()
        route = "tool" if "lookup" in query or "order status" in query else "simple"
        return nodes.ClassificationOutput(route=route, risk_level="low", reasoning="test")


class _FakeLLM:
    def with_structured_output(self, _schema):
        return _FakeStructured()

    def invoke(self, _prompt):
        return SimpleNamespace(content="test answer")


def _use_fake_llm(monkeypatch):
    monkeypatch.setattr(nodes, "get_llm", lambda **_kwargs: _FakeLLM())


def test_memory_checkpointer_state_history(monkeypatch):
    _use_fake_llm(monkeypatch)
    saver = build_checkpointer("memory")
    graph = build_graph(checkpointer=saver)

    scenario = Scenario(id="test_mem", query="How do I reset my password?", expected_route=Route.SIMPLE)
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": "thread-test-mem-history"}}

    result = graph.invoke(state, config=config)
    assert result["route"] == "simple"

    # Verify state can be retrieved by thread_id
    latest_state = graph.get_state(config)
    assert latest_state.values["route"] == "simple"

    # Verify checkpoint history contains multiple execution snapshots
    history = list(graph.get_state_history(config))
    assert len(history) >= 3


def test_sqlite_checkpointer_durability(monkeypatch, tmp_path):
    _use_fake_llm(monkeypatch)
    db_file = str(tmp_path / "checkpoints.db")

    # First session: run graph with SQLite checkpointer.
    saver1 = build_checkpointer("sqlite", database_url=db_file)
    graph1 = build_graph(checkpointer=saver1)

    scenario = Scenario(
        id="test_sqlite",
        query="Please lookup order status for order 12345",
        expected_route=Route.TOOL,
    )
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": "thread-sqlite-durability"}}

    result = graph1.invoke(state, config=config)
    assert result["route"] == "tool"

    # Simulate process restart with a fresh checkpointer and graph.
    saver2 = build_checkpointer("sqlite", database_url=db_file)
    graph2 = build_graph(checkpointer=saver2)

    recovered_state = graph2.get_state(config)
    assert recovered_state.values["route"] == "tool"
    assert recovered_state.values["scenario_id"] == "test_sqlite"
    assert len(recovered_state.values["events"]) > 0

    history = list(graph2.get_state_history(config))
    assert len(history) >= 4
