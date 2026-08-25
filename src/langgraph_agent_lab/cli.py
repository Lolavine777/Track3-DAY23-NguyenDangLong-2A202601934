"""CLI for the lab."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer
import yaml
from langchain_core.runnables import RunnableConfig

from .graph import build_graph
from .judge import evaluate_run_with_llm_judge
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state

app = typer.Typer(no_args_is_help=True)


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)
    metrics = []
    for scenario in scenarios:
        state = initial_state(scenario)
        run_config: RunnableConfig = {"configurable": {"thread_id": state["thread_id"]}}
        start_time = time.perf_counter()
        final_state = graph.invoke(state, config=run_config)
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        metrics.append(
            metric_from_state(
                final_state,
                scenario.expected_route.value,
                scenario.requires_approval,
                latency_ms=latency_ms,
            )
        )
    report = summarize_metrics(metrics)
    write_metrics(report, output)
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])
    typer.echo(f"Wrote metrics to {output}")


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


@app.command("judge")
def judge_scenarios(
    config: Annotated[Path, typer.Option("--config")] = Path("configs/lab.yaml"),
) -> None:
    """Run LLM-as-a-Judge evaluation over all scenarios to verify correct agent behavior."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)

    typer.echo("=" * 70)
    typer.echo("RUNNING LLM-AS-A-JUDGE BEHAVIOR AUDIT ON ALL SCENARIOS")
    typer.echo("=" * 70)

    total_score = 0
    passed_count = 0

    for sc in scenarios:
        st = initial_state(sc)
        run_cfg: RunnableConfig = {"configurable": {"thread_id": f"judge_{sc.id}"}}
        out = graph.invoke(st, config=run_cfg)

        exec_result = {
            "route": out.get("route"),
            "risk_level": out.get("risk_level"),
            "path": [e.get("node") for e in out.get("events", []) if e.get("node")],
            "final_answer": out.get("final_answer"),
            "pending_question": out.get("pending_question"),
            "proposed_action": out.get("proposed_action"),
            "attempt": out.get("attempt", 0),
            "approval": out.get("approval"),
            "events": out.get("events", []),
        }

        sc_info = {
            "id": sc.id,
            "query": sc.query,
            "expected_route": sc.expected_route.value,
            "requires_approval": sc.requires_approval,
            "max_attempts": sc.max_attempts,
        }

        verdict = evaluate_run_with_llm_judge(sc_info, exec_result)
        total_score += verdict.overall_score
        if verdict.is_correct_behavior:
            passed_count += 1

        color = typer.colors.GREEN if verdict.is_correct_behavior else typer.colors.RED
        typer.secho(
            f"[{sc.id}] -> {verdict.verdict} | Overall Score: {verdict.overall_score}/100",
            fg=color,
            bold=True,
        )
        scores_line = (
            f"  Scores: Route={verdict.route_accuracy_score}/10 "
            f"| Safety={verdict.safety_compliance_score}/10 "
            f"| Ground={verdict.groundedness_score}/10 "
            f"| Robust={verdict.robustness_score}/10"
        )
        typer.echo(scores_line)
        typer.echo(f"  Critique: {verdict.critique}")

        typer.echo("")


    avg_score = total_score / len(scenarios) if scenarios else 0
    typer.echo("=" * 70)
    summary_text = (
        f"LLM JUDGE SUMMARY: {passed_count}/{len(scenarios)} Passed | "
        f"Avg Score: {avg_score:.1f}/100"
    )
    typer.secho(
        summary_text,
        fg=typer.colors.GREEN if passed_count == len(scenarios) else typer.colors.YELLOW,
        bold=True,
    )
    typer.echo("=" * 70)


if __name__ == "__main__":
    app()
