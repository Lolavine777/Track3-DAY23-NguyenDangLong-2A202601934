"""FastAPI server with live streaming execution, multi-turn chat memory, HITL popup, and LLM Judge."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.judge import evaluate_run_with_llm_judge
from langgraph_agent_lab.metrics import metric_from_state, summarize_metrics
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.scenarios import load_scenarios
from langgraph_agent_lab.state import Route, Scenario, initial_state

load_dotenv()

app = FastAPI(title="LangGraph Live Agentic Studio with Chat Memory", version="2.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SCENARIOS_PATH = Path("data/sample/scenarios.jsonl")
STATIC_HTML = Path("web/index.html")

PENDING_HITL_RUNS: dict[str, Any] = {}
THREAD_STATES: dict[str, Any] = {}


class RunRequest(BaseModel):
    query: str
    checkpointer: str = "memory"
    max_attempts: int = 3
    thread_id: str | None = None


class ChatStreamRequest(BaseModel):
    message: str
    thread_id: str
    checkpointer: str = "memory"
    max_attempts: int = 3


class ResumeApprovalRequest(BaseModel):
    thread_id: str
    approved: bool
    reviewer: str = "supervisor-on-duty"
    comment: str = ""
    checkpointer: str = "memory"


class JudgeRunRequest(BaseModel):
    scenario_info: dict[str, Any]
    execution_result: dict[str, Any]


class BatchRunRequest(BaseModel):
    checkpointer: str = "memory"


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> FileResponse:
    if not STATIC_HTML.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    return FileResponse(STATIC_HTML)


@app.get("/api/scenarios")
async def get_scenarios() -> list[dict[str, Any]]:
    if not SCENARIOS_PATH.exists():
        return []
    scenarios = load_scenarios(SCENARIOS_PATH)
    return [s.model_dump() for s in scenarios]


@app.get("/api/graph-structure")
async def get_graph_structure() -> dict[str, Any]:
    nodes = [
        {"id": "intake", "label": "1. Intake", "category": "ingress", "desc": "Chuẩn hóa query & audit", "x": 500, "y": 45},
        {"id": "prompt_guardrail", "label": "2. Guardrail", "category": "security", "desc": "Prompt Injection Shield", "x": 500, "y": 135},
        {"id": "query_rewrite", "label": "3. Rewrite", "category": "preprocess", "desc": "Decompose & Context", "x": 500, "y": 225},
        
        {"id": "parallel_worker", "label": "Parallel Worker (Send)", "category": "execution", "desc": "Sub-query Classify & Run", "x": 740, "y": 360},
        {"id": "aggregate_answers", "label": "Answer Aggregator", "category": "llm", "desc": "Fan-in Multi-intent Synthesis", "x": 740, "y": 515},

        {"id": "classify", "label": "4. Classify", "category": "llm", "desc": "LLM Intent Classifier", "x": 380, "y": 315},
        
        {"id": "tool", "label": "Tool Node", "category": "execution", "desc": "Execute Tool & Mock", "x": 260, "y": 415},
        {"id": "risky_action", "label": "Risky Action", "category": "prep", "desc": "Prepare Proposal", "x": 500, "y": 415},
        
        {"id": "evaluate", "label": "Evaluate Gate", "category": "gate", "desc": "Check Tool Result", "x": 260, "y": 515},
        {"id": "approval", "label": "Approval Gate", "category": "hitl", "desc": "HITL Supervisor Gate", "x": 500, "y": 515},
        
        {"id": "answer", "label": "Answer Node", "category": "llm", "desc": "Grounded LLM Answer", "x": 120, "y": 625},
        {"id": "retry", "label": "Retry Loop", "category": "recovery", "desc": "Increment Attempt", "x": 370, "y": 625},
        {"id": "clarify", "label": "Clarify Node", "category": "interaction", "desc": "Ask or Reject Info", "x": 880, "y": 625},
        {"id": "dead_letter", "label": "Dead Letter", "category": "escalation", "desc": "Escalate to Tier-2", "x": 370, "y": 705},
        
        {"id": "finalize", "label": "Finalize Node", "category": "egress", "desc": "Final Audit Trail & End", "x": 500, "y": 795},
    ]
    
    edges = [
        {"from": "intake", "to": "prompt_guardrail", "type": "fixed", "label": ""},
        {"from": "prompt_guardrail", "to": "query_rewrite", "type": "conditional", "label": "safe"},
        {"from": "prompt_guardrail", "to": "clarify", "type": "conditional", "label": "blocked"},
        
        {"from": "query_rewrite", "to": "parallel_worker", "type": "conditional", "label": "multi (Send)"},
        {"from": "query_rewrite", "to": "classify", "type": "conditional", "label": "single"},
        
        {"from": "parallel_worker", "to": "aggregate_answers", "type": "fixed", "label": "fan-in"},
        {"from": "aggregate_answers", "to": "approval", "type": "conditional", "label": "risky"},
        {"from": "aggregate_answers", "to": "finalize", "type": "conditional", "label": "safe"},

        {"from": "classify", "to": "answer", "type": "conditional", "label": "simple"},
        {"from": "classify", "to": "tool", "type": "conditional", "label": "tool"},
        {"from": "classify", "to": "clarify", "type": "conditional", "label": "missing"},
        {"from": "classify", "to": "risky_action", "type": "conditional", "label": "risky"},
        {"from": "classify", "to": "retry", "type": "conditional", "label": "error"},
        
        {"from": "risky_action", "to": "approval", "type": "fixed", "label": ""},
        {"from": "approval", "to": "tool", "type": "conditional", "label": "approved"},
        {"from": "approval", "to": "clarify", "type": "conditional", "label": "rejected"},
        {"from": "approval", "to": "finalize", "type": "conditional", "label": "multi_done"},
        
        {"from": "tool", "to": "evaluate", "type": "fixed", "label": ""},
        {"from": "evaluate", "to": "answer", "type": "conditional", "label": "success"},
        {"from": "evaluate", "to": "retry", "type": "conditional", "label": "needs_retry"},
        
        {"from": "retry", "to": "tool", "type": "conditional", "label": "attempt < max"},
        {"from": "retry", "to": "dead_letter", "type": "conditional", "label": "attempt >= max"},
        
        {"from": "answer", "to": "finalize", "type": "fixed", "label": ""},
        {"from": "clarify", "to": "finalize", "type": "fixed", "label": ""},
        {"from": "dead_letter", "to": "finalize", "type": "fixed", "label": ""},
    ]
    return {"nodes": nodes, "edges": edges}




@app.post("/api/run-stream")
async def run_stream(req: RunRequest) -> StreamingResponse:
    async def event_generator() -> AsyncGenerator[str, None]:
        db_url = "checkpoints.db" if req.checkpointer == "sqlite" else None
        checkpointer = build_checkpointer(req.checkpointer, db_url)
        graph = build_graph(checkpointer=checkpointer)

        thread_id = req.thread_id or f"web_{int(time.time() * 1000)}"
        scenario = Scenario(
            id=thread_id,
            query=req.query.strip(),
            expected_route=Route.SIMPLE,
            max_attempts=req.max_attempts,
        )
        state = initial_state(scenario)
        start_time = time.perf_counter()
        run_config: Any = {"configurable": {"thread_id": state["thread_id"]}}
        
        yield f"event: start\ndata: {json.dumps({'thread_id': thread_id, 'query': req.query})}\n\n"
        await asyncio.sleep(0.05)

        accumulated_state = dict(state)
        path = []
        stopped_for_hitl = False

        for chunk in graph.stream(state, config=run_config, stream_mode="updates"):
            for node_name, update in chunk.items():
                path.append(node_name)
                for k, v in update.items():
                    if k in ["messages", "tool_results", "errors", "events"]:
                        accumulated_state[k] = accumulated_state.get(k, []) + v
                    else:
                        accumulated_state[k] = v
                
                node_event = {
                    "node": node_name,
                    "update": update,
                    "accumulated_path": path,
                    "latest_event": update.get("events", [{}])[-1] if update.get("events") else None,
                    "accumulated_state": dict(accumulated_state),
                    "timestamp": time.time(),
                }
                yield f"event: node_step\ndata: {json.dumps(node_event)}\n\n"
                await asyncio.sleep(0.25)

                if node_name == "risky_action" or (
                    node_name == "aggregate_answers" and accumulated_state.get("risk_level") == "high"
                ):
                    stopped_for_hitl = True
                    break
            if stopped_for_hitl:
                break

        if stopped_for_hitl:
            PENDING_HITL_RUNS[thread_id] = {
                "state": accumulated_state,
                "path": path,
                "start_time": start_time,
                "checkpointer": req.checkpointer,
                "max_attempts": req.max_attempts,
            }
            hitl_payload = {
                "thread_id": thread_id,
                "node": "approval",
                "proposed_action": accumulated_state.get("proposed_action") or accumulated_state.get("query"),
                "query": accumulated_state.get("query"),
                "risk_level": accumulated_state.get("risk_level", "high"),
            }
            yield f"event: hitl_interrupt\ndata: {json.dumps(hitl_payload)}\n\n"
            return

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        
        llm_reasoning = ""
        for ev in accumulated_state.get("events", []):
            if ev.get("node") == "classify" and ev.get("metadata", {}).get("reasoning"):
                llm_reasoning = ev["metadata"]["reasoning"]
                break

        finish_payload = {
            "scenario_id": thread_id,
            "latency_ms": latency_ms,
            "path": path,
            "route": accumulated_state.get("route"),
            "risk_level": accumulated_state.get("risk_level"),
            "final_answer": accumulated_state.get("final_answer"),
            "pending_question": accumulated_state.get("pending_question"),
            "proposed_action": accumulated_state.get("proposed_action"),
            "attempt": accumulated_state.get("attempt", 0),
            "max_attempts": accumulated_state.get("max_attempts", 3),
            "llm_reasoning": llm_reasoning,
            "events": accumulated_state.get("events", []),
            "messages": accumulated_state.get("messages", []),
            "tool_results": accumulated_state.get("tool_results", []),
            "errors": accumulated_state.get("errors", []),
            "approval": accumulated_state.get("approval"),
            "raw_state": accumulated_state,
        }
        yield f"event: finish\ndata: {json.dumps(finish_payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/chat-stream")
async def chat_stream(req: ChatStreamRequest) -> StreamingResponse:
    """Multi-turn conversational stream with live word-by-word streaming & latency tracking."""
    async def chat_generator() -> AsyncGenerator[str, None]:
        db_url = "checkpoints.db" if req.checkpointer == "sqlite" else None
        checkpointer = build_checkpointer(req.checkpointer, db_url)
        graph = build_graph(checkpointer=checkpointer)

        thread_id = req.thread_id
        run_config: Any = {"configurable": {"thread_id": thread_id}}
        
        # Initialize or retrieve conversation state
        if thread_id not in THREAD_STATES:
            scenario = Scenario(
                id=thread_id,
                query=req.message.strip(),
                expected_route=Route.SIMPLE,
                max_attempts=req.max_attempts,
            )
            state = initial_state(scenario)
            state["messages"] = [f"user:{req.message.strip()}"]
            THREAD_STATES[thread_id] = state
        else:
            state = THREAD_STATES[thread_id]
            state["query"] = req.message.strip()
            state["messages"] = state.get("messages", []) + [f"user:{req.message.strip()}"]
            state["route"] = None
            state["final_answer"] = None
            state["pending_question"] = None
            state["errors"] = []

        start_time = time.perf_counter()
        yield f"event: start\ndata: {json.dumps({'thread_id': thread_id, 'message': req.message})}\n\n"
        await asyncio.sleep(0.05)

        accumulated_state = dict(state)
        path = []
        stopped_for_hitl = False

        for chunk in graph.stream(state, config=run_config, stream_mode="updates"):
            for node_name, update in chunk.items():
                path.append(node_name)
                for k, v in update.items():
                    if k in ["messages", "tool_results", "errors", "events"]:
                        accumulated_state[k] = accumulated_state.get(k, []) + v
                    else:
                        accumulated_state[k] = v
                
                cur_latency = int((time.perf_counter() - start_time) * 1000)
                node_event = {
                    "node": node_name,
                    "update": update,
                    "accumulated_path": path,
                    "latest_event": update.get("events", [{}])[-1] if update.get("events") else None,
                    "elapsed_ms": cur_latency,
                    "timestamp": time.time(),
                }
                yield f"event: node_step\ndata: {json.dumps(node_event)}\n\n"
                await asyncio.sleep(0.18)

                # If answer or clarify is ready, stream text tokens progressively
                if node_name in ["answer", "clarify"]:
                    full_text = update.get("final_answer") or update.get("pending_question") or ""
                    if full_text:
                        words = full_text.split(" ")
                        for idx in range(0, len(words), 3):
                            chunk_text = " ".join(words[idx:idx+3]) + (" " if idx+3 < len(words) else "")
                            yield f"event: text_chunk\ndata: {json.dumps({'chunk': chunk_text})}\n\n"
                            await asyncio.sleep(0.04)

                if node_name == "risky_action" or (
                    node_name == "aggregate_answers" and accumulated_state.get("risk_level") == "high"
                ):
                    stopped_for_hitl = True
                    break
            if stopped_for_hitl:
                break

        if stopped_for_hitl:
            PENDING_HITL_RUNS[thread_id] = {
                "state": accumulated_state,
                "path": path,
                "start_time": start_time,
                "checkpointer": req.checkpointer,
                "max_attempts": req.max_attempts,
            }
            hitl_payload = {
                "thread_id": thread_id,
                "node": "approval",
                "proposed_action": accumulated_state.get("proposed_action") or accumulated_state.get("query"),
                "query": req.message,
                "risk_level": accumulated_state.get("risk_level", "high"),
            }
            yield f"event: hitl_interrupt\ndata: {json.dumps(hitl_payload)}\n\n"
            return

        THREAD_STATES[thread_id] = accumulated_state
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        finish_payload = {
            "thread_id": thread_id,
            "latency_ms": latency_ms,
            "path": path,
            "route": accumulated_state.get("route"),
            "risk_level": accumulated_state.get("risk_level"),
            "final_answer": accumulated_state.get("final_answer"),
            "pending_question": accumulated_state.get("pending_question"),
            "proposed_action": accumulated_state.get("proposed_action"),
            "attempt": accumulated_state.get("attempt", 0),
            "max_attempts": accumulated_state.get("max_attempts", 3),
            "messages": accumulated_state.get("messages", []),
            "events": accumulated_state.get("events", []),
        }
        yield f"event: finish\ndata: {json.dumps(finish_payload)}\n\n"

    return StreamingResponse(chat_generator(), media_type="text/event-stream")


@app.post("/api/resume-approval")
async def resume_approval(req: ResumeApprovalRequest) -> StreamingResponse:
    async def resume_generator() -> AsyncGenerator[str, None]:
        pending = PENDING_HITL_RUNS.get(req.thread_id)
        if not pending:
            raise HTTPException(status_code=404, detail="No pending HITL run found for thread_id")

        accumulated_state = pending["state"]
        path = list(pending["path"])
        start_time = pending["start_time"]
        
        approval_dict = {
            "approved": req.approved,
            "reviewer": req.reviewer,
            "comment": req.comment or ("Đã phê duyệt bởi người giám sát." if req.approved else "Bị từ chối bởi người giám sát."),
        }
        accumulated_state["approval"] = approval_dict

        path.append("approval")
        approval_event = {
            "node": "approval",
            "event_type": "decision",
            "message": f"Approval decision: approved={req.approved} reviewer={req.reviewer}",
            "latency_ms": 0,
            "metadata": {"approved": req.approved, "reviewer": req.reviewer, "comment": req.comment}
        }
        accumulated_state["events"].append(approval_event)
        
        yield f"event: node_step\ndata: {json.dumps({'node': 'approval', 'update': {'approval': approval_dict}, 'accumulated_path': path, 'latest_event': approval_event, 'timestamp': time.time()})}\n\n"
        await asyncio.sleep(0.2)

        from langgraph_agent_lab.nodes import answer_node, ask_clarification_node, evaluate_node, finalize_node, tool_node

        if accumulated_state.get("route") == "parallel_multi_intent":
            if not req.approved:
                rejection_msg = (
                    f"Phần tác vụ nhạy cảm trong yêu cầu của bạn đã bị từ chối phê duyệt "
                    f"({approval_dict['comment']}). Vui lòng liên hệ bộ phận hỗ trợ nếu cần thêm chi tiết."
                )
                accumulated_state["final_answer"] = (
                    f"{accumulated_state.get('final_answer', '')}\n\n[CẬP NHẬT TỪ SUPERVISOR]: {rejection_msg}"
                )
                accumulated_state["pending_question"] = rejection_msg
            
            fin_up = finalize_node(accumulated_state)
            path.append("finalize")
            accumulated_state["events"].extend(fin_up.get("events", []))
            yield f"event: node_step\ndata: {json.dumps({'node': 'finalize', 'update': fin_up, 'accumulated_path': path, 'latest_event': fin_up['events'][-1], 'timestamp': time.time()})}\n\n"
            await asyncio.sleep(0.1)
        elif req.approved:
            tool_up = tool_node(accumulated_state)
            path.append("tool")
            accumulated_state["tool_results"].extend(tool_up.get("tool_results", []))
            accumulated_state["events"].extend(tool_up.get("events", []))
            yield f"event: node_step\ndata: {json.dumps({'node': 'tool', 'update': tool_up, 'accumulated_path': path, 'latest_event': tool_up['events'][-1], 'timestamp': time.time()})}\n\n"
            await asyncio.sleep(0.2)

            eval_up = evaluate_node(accumulated_state)
            path.append("evaluate")
            accumulated_state["evaluation_result"] = eval_up.get("evaluation_result")
            accumulated_state["events"].extend(eval_up.get("events", []))
            yield f"event: node_step\ndata: {json.dumps({'node': 'evaluate', 'update': eval_up, 'accumulated_path': path, 'latest_event': eval_up['events'][-1], 'timestamp': time.time()})}\n\n"
            await asyncio.sleep(0.2)

            ans_up = answer_node(accumulated_state)
            path.append("answer")
            accumulated_state["final_answer"] = ans_up.get("final_answer")
            accumulated_state["events"].extend(ans_up.get("events", []))
            yield f"event: node_step\ndata: {json.dumps({'node': 'answer', 'update': ans_up, 'accumulated_path': path, 'latest_event': ans_up['events'][-1], 'timestamp': time.time()})}\n\n"
            
            full_text = ans_up.get("final_answer", "")
            if full_text:
                words = full_text.split(" ")
                for idx in range(0, len(words), 3):
                    chunk_text = " ".join(words[idx:idx+3]) + (" " if idx+3 < len(words) else "")
                    yield f"event: text_chunk\ndata: {json.dumps({'chunk': chunk_text})}\n\n"
                    await asyncio.sleep(0.03)

            fin_up = finalize_node(accumulated_state)
            path.append("finalize")
            accumulated_state["events"].extend(fin_up.get("events", []))
            yield f"event: node_step\ndata: {json.dumps({'node': 'finalize', 'update': fin_up, 'accumulated_path': path, 'latest_event': fin_up['events'][-1], 'timestamp': time.time()})}\n\n"
            await asyncio.sleep(0.1)
        else:
            clarify_up = ask_clarification_node(accumulated_state)
            path.append("clarify")
            accumulated_state["pending_question"] = clarify_up.get("pending_question")
            accumulated_state["events"].extend(clarify_up.get("events", []))
            yield f"event: node_step\ndata: {json.dumps({'node': 'clarify', 'update': clarify_up, 'accumulated_path': path, 'latest_event': clarify_up['events'][-1], 'timestamp': time.time()})}\n\n"
            
            full_text = clarify_up.get("pending_question", "")
            if full_text:
                words = full_text.split(" ")
                for idx in range(0, len(words), 3):
                    chunk_text = " ".join(words[idx:idx+3]) + (" " if idx+3 < len(words) else "")
                    yield f"event: text_chunk\ndata: {json.dumps({'chunk': chunk_text})}\n\n"
                    await asyncio.sleep(0.03)

            fin_up = finalize_node(accumulated_state)
            path.append("finalize")
            accumulated_state["events"].extend(fin_up.get("events", []))
            yield f"event: node_step\ndata: {json.dumps({'node': 'finalize', 'update': fin_up, 'accumulated_path': path, 'latest_event': fin_up['events'][-1], 'timestamp': time.time()})}\n\n"
            await asyncio.sleep(0.1)

        THREAD_STATES[req.thread_id] = accumulated_state
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        finish_payload = {
            "scenario_id": req.thread_id,
            "thread_id": req.thread_id,
            "latency_ms": latency_ms,
            "path": path,
            "route": accumulated_state.get("route"),
            "risk_level": accumulated_state.get("risk_level"),
            "final_answer": accumulated_state.get("final_answer"),
            "pending_question": accumulated_state.get("pending_question"),
            "proposed_action": accumulated_state.get("proposed_action"),
            "attempt": accumulated_state.get("attempt", 0),
            "max_attempts": accumulated_state.get("max_attempts", 3),
            "events": accumulated_state.get("events", []),
            "messages": accumulated_state.get("messages", []),
            "tool_results": accumulated_state.get("tool_results", []),
            "errors": accumulated_state.get("errors", []),
            "approval": accumulated_state.get("approval"),
        }
        yield f"event: finish\ndata: {json.dumps(finish_payload)}\n\n"
        PENDING_HITL_RUNS.pop(req.thread_id, None)

    return StreamingResponse(resume_generator(), media_type="text/event-stream")


@app.post("/api/judge-run")
async def judge_run(req: JudgeRunRequest) -> dict[str, Any]:
    try:
        verdict = evaluate_run_with_llm_judge(req.scenario_info, req.execution_result)
        return verdict.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/judge-batch")
async def judge_batch() -> dict[str, Any]:
    scenarios = load_scenarios(SCENARIOS_PATH)
    graph = build_graph()
    verdicts = []
    total_score = 0
    passed_count = 0

    for sc in scenarios:
        st = initial_state(sc)
        out = graph.invoke(st, config={"configurable": {"thread_id": f"judge_web_{sc.id}"}})
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
        v = evaluate_run_with_llm_judge(sc_info, exec_result)
        total_score += v.overall_score
        if v.is_correct_behavior:
            passed_count += 1
        
        v_dict = v.model_dump()
        v_dict["query"] = sc.query
        v_dict["expected_route"] = sc.expected_route.value
        v_dict["actual_route"] = out.get("route")
        verdicts.append(v_dict)

    avg_score = total_score / len(scenarios) if scenarios else 0
    return {
        "summary": {
            "total_scenarios": len(scenarios),
            "passed_count": passed_count,
            "average_score": round(avg_score, 1),
            "pass_rate": round(passed_count / len(scenarios) * 100, 1),
        },
        "verdicts": verdicts,
    }


@app.post("/api/run-batch")
async def run_batch(req: BatchRunRequest) -> dict[str, Any]:
    scenarios = load_scenarios(SCENARIOS_PATH)
    db_url = "checkpoints.db" if req.checkpointer == "sqlite" else None
    checkpointer = build_checkpointer(req.checkpointer, db_url)
    graph = build_graph(checkpointer=checkpointer)

    results = []
    metric_objects = []
    for sc in scenarios:
        st_init = initial_state(sc)
        run_cfg: Any = {"configurable": {"thread_id": st_init["thread_id"]}}
        t0 = time.perf_counter()
        out = graph.invoke(st_init, config=run_cfg)
        lat = int((time.perf_counter() - t0) * 1000)
        m = metric_from_state(out, sc.expected_route.value, sc.requires_approval, latency_ms=lat)
        metric_objects.append(m)
        results.append({
            "id": sc.id,
            "query": sc.query,
            "expected_route": sc.expected_route.value,
            "actual_route": out.get("route"),
            "success": m.success,
            "nodes_visited": m.nodes_visited,
            "retry_count": m.retry_count,
            "interrupt_count": m.interrupt_count,
            "latency_ms": lat,
            "path": [e.get("node") for e in out.get("events", []) if e.get("node")],
        })

    summary = summarize_metrics(metric_objects)
    return {
        "summary": summary.model_dump(),
        "scenarios": results,
    }


@app.get("/api/checkpoints")
async def get_checkpoints() -> dict[str, Any]:
    db_file = Path("checkpoints.db")
    if not db_file.exists():
        return {"exists": False, "rows": []}
    try:
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cursor.fetchall()]
        
        rows = []
        if "checkpoints" in tables:
            cursor.execute("SELECT thread_id, checkpoint_id, parent_checkpoint_id, type FROM checkpoints ORDER BY rowid DESC LIMIT 20;")
            rows = [{"thread_id": r[0], "checkpoint_id": r[1], "parent_id": r[2], "type": r[3]} for r in cursor.fetchall()]
        conn.close()
        return {"exists": True, "tables": tables, "rows": rows}
    except Exception as exc:
        return {"exists": True, "error": str(exc), "rows": []}
