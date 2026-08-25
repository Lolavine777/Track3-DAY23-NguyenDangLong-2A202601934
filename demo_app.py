"""Interactive Streamlit Demo App for LangGraph Support-Ticket Agent."""

from __future__ import annotations

import os
import time
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.scenarios import load_scenarios
from langgraph_agent_lab.state import Route, Scenario, initial_state

load_dotenv()

st.set_page_config(
    page_title="LangGraph Agentic Orchestration Demo",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 LangGraph Support-Ticket Agent - Interactive Demo")
st.caption("Day 08 / Day 23 Lab - Typed State, Conditional Routing, Bounded Retry, HITL & Persistence")

# Load sample scenarios for presets
SAMPLE_FILE = "data/sample/scenarios.jsonl"
scenarios = load_scenarios(SAMPLE_FILE) if os.path.exists(SAMPLE_FILE) else []

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Cấu Hình Hệ Thống")
    checkpointer_type = st.selectbox(
        "Checkpointer Backend",
        ["memory", "sqlite"],
        index=0,
        help="MemorySaver cho phiên tạm thời, SqliteSaver cho lưu trữ bền vững (checkpoints.db)",
    )
    db_url = "checkpoints.db" if checkpointer_type == "sqlite" else None

    st.divider()
    st.subheader("🛡️ Cổng Phê Duyệt (Approval Gate)")
    approval_mode = st.radio(
        "Quyết định khi gặp tác vụ Rủi ro:",
        ["Phê duyệt (Approved)", "Từ chối (Rejected)"],
        index=0,
    )
    approved_bool = approval_mode.startswith("Phê duyệt")

    st.divider()
    st.subheader("🔄 Giới Hạn Thử Lại (Max Attempts)")
    max_attempts = st.slider("Max retry attempts:", min_value=1, max_value=5, value=3)

    st.divider()
    st.subheader("📋 Kịch Bản Mẫu (Presets)")
    preset_choice = st.selectbox(
        "Chọn kịch bản nhanh:",
        ["-- Tự nhập yêu cầu --"] + [f"{s.id}: {s.query[:45]}..." for s in scenarios],
    )

# Selected query logic
default_query = "Tôi muốn kiểm tra tình trạng đơn hàng #12345 của mình"
if preset_choice != "-- Tự nhập yêu cầu --":
    idx = int(preset_choice.split(":")[0].replace("S0", "").replace("S", "")) - 1
    if 0 <= idx < len(scenarios):
        selected_scenario = scenarios[idx]
        default_query = selected_scenario.query
        if selected_scenario.id == "S07_dead_letter":
            max_attempts = 1

tab1, tab2, tab3 = st.tabs([
    "🚀 Thử Nghiệm Trực Tiếp (Live Run)",
    "📊 Chạy Batch 7 Scenarios",
    "🗺️ Kiến Trúc Đồ Thị (Mermaid)",
])

with tab1:
    col_in, col_btn = st.columns([4, 1])
    with col_in:
        user_query = st.text_area(
            "Nội dung yêu cầu hỗ trợ (Support Ticket Query):",
            value=default_query,
            height=100,
        )
    with col_btn:
        st.write("")
        st.write("")
        run_button = st.button("🚀 Chạy Graph", type="primary", use_container_width=True)

    if run_button and user_query.strip():
        with st.spinner("StateGraph đang thực thi các node và gọi LLM..."):
            start_time = time.perf_counter()

            # Setup custom scenario & state
            custom_scenario = Scenario(
                id=f"demo_{int(time.time())}",
                query=user_query.strip(),
                expected_route=Route.SIMPLE,
                max_attempts=max_attempts,
            )
            state = initial_state(custom_scenario)

            if not approved_bool:
                state["approval"] = {
                    "approved": False,
                    "reviewer": "supervisor-ui",
                    "comment": "Bị từ chối bởi người giám sát.",
                }

            checkpointer = build_checkpointer(checkpointer_type, db_url)
            graph = build_graph(checkpointer=checkpointer)

            config: Any = {"configurable": {"thread_id": state["thread_id"]}}
            final_state = graph.invoke(state, config=config)
            latency_ms = int((time.perf_counter() - start_time) * 1000)

        st.success(f"✅ Hoàn tất thực thi trong **{latency_ms} ms**!")

        # Results Dashboard
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LLM Route", str(final_state.get("route", "N/A")).upper())
        c2.metric("Risk Level", str(final_state.get("risk_level", "low")).upper())
        c3.metric("Số lần Retry", final_state.get("attempt", 0))
        c4.metric("Số Node đã đi qua", len(final_state.get("events", [])))

        st.subheader("💬 Phản Hồi Cuối Cùng (Final Answer / Question)")
        if final_state.get("final_answer"):
            st.info(final_state["final_answer"])
        elif final_state.get("pending_question"):
            st.warning(f"**Cần làm rõ / Từ chối:** {final_state[pending_question]}")

        # Trace timeline
        st.subheader("🛤️ Dấu Vết Thực Thi Các Node (Execution Path)")
        events = final_state.get("events", [])
        path_list = [f"**`{e.get('node', '').upper()}`**" for e in events]
        st.markdown(" ➔ ".join(path_list))

        with st.expander("🔍 Xem Chi Tiết Toàn Bộ State và Audit Events (JSON)"):
            st.json(final_state)

with tab2:
    st.subheader("Chạy Kiểm Thử Toàn Bộ 7 Scenarios Mẫu")
    if st.button("▶️ Chạy Toàn Bộ Kịch Bản", type="secondary"):
        with st.spinner("Đang chạy 7 kịch bản qua LLM và StateGraph..."):
            checkpointer = build_checkpointer("memory")
            graph = build_graph(checkpointer=checkpointer)
            results = []
            for s in scenarios:
                t0 = time.perf_counter()
                st_init = initial_state(s)
                cfg: Any = {"configurable": {"thread_id": st_init["thread_id"]}}
                out = graph.invoke(st_init, config=cfg)
                lat = int((time.perf_counter() - t0) * 1000)
                is_match = out.get("route") == s.expected_route.value
                results.append({
                    "Kịch bản": s.id,
                    "Truy vấn": s.query,
                    "Expected": s.expected_route.value,
                    "Actual Route": out.get("route"),
                    "Kết quả": "PASS ✅" if is_match else "FAIL ❌",
                    "Retries": out.get("attempt", 0),
                    "Latency (ms)": lat,
                })
            st.dataframe(results, use_container_width=True)

with tab3:
    st.subheader("Sơ Đồ Kiến Trúc StateGraph (11 Nodes, 12 Edges)")
    mermaid_code = (
        "graph TD;\n"
        "\t__start__([__start__])\n"
        "\tintake(intake: Chuẩn hóa query)\n"
        "\tclassify(classify: LLM Structured Output)\n"
        "\tanswer(answer: LLM Grounded Answer)\n"
        "\ttool(tool: Mock Execution)\n"
        "\tevaluate(evaluate: Check Retry Gate)\n"
        "\tclarify(clarify: Ask Clarification)\n"
        "\trisky_action(risky_action: Prepare Proposal)\n"
        "\tapproval(approval: HITL Approval Gate)\n"
        "\tretry(retry: Increment Attempt)\n"
        "\tdead_letter(dead_letter: Escalate Tier-2)\n"
        "\tfinalize(finalize: Audit Trail)\n"
        "\t__end__([__end__])\n"
        "\t__start__ --> intake;\n"
        "\tanswer --> finalize;\n"
        "\tapproval -.-> clarify;\n"
        "\tapproval -.-> tool;\n"
        "\tclarify --> finalize;\n"
        "\tclassify -.-> answer;\n"
        "\tclassify -.-> clarify;\n"
        "\tclassify -.-> retry;\n"
        "\tclassify -.-> risky_action;\n"
        "\tclassify -.-> tool;\n"
        "\tdead_letter --> finalize;\n"
        "\tevaluate -.-> answer;\n"
        "\tevaluate -.-> retry;\n"
        "\tintake --> classify;\n"
        "\tretry -.-> dead_letter;\n"
        "\tretry -.-> tool;\n"
        "\trisky_action --> approval;\n"
        "\ttool --> evaluate;\n"
        "\tfinalize --> __end__;\n"
    )
    st.markdown(f"```mermaid\n{mermaid_code}\n```")
