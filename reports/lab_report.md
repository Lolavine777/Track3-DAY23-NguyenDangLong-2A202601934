# Báo Cáo Lab Day 08 / Day 23 - LangGraph Agentic Orchestration

## 1. Thông tin sinh viên / nhóm

- Họ và tên: Nguyễn Đăng Long
- Mã sinh viên: 2A202601934
- Repo: Lolavine777/phase2-k3-4-track3-day8-langgraph-agent-2A202601934-NguyenDangLong
- Commit Hash: `ac3c09d9db37c1ce07b4123533a524e280cafac4`
- Ngày hoàn thành: 2026-08-25
- LLM Provider: OpenAI (gpt-4o-mini) với structured outputs

## 2. Kiến trúc StateGraph (Architecture)

Quy trình xử lý support-ticket được xây dựng dạng đồ thị StateGraph gồm 11 node:
- `intake`: Chuẩn hóa dữ liệu đầu vào và khởi tạo audit log.
- `classify`: Phân loại intent bằng LLM với structured output (pydantic), áp dụng độ ưu tiên `risky > tool > missing_info > error > simple`.
- `answer`: Tạo câu trả lời grounded từ kết quả tool, thông tin duyệt và query ban đầu.
- `tool`: Thực thi mock tool và mô phỏng lỗi tạm thời cho kịch bản retry.
- `evaluate`: Đánh giá kết quả tool (thành công hoặc `needs_retry`).
- `clarify`: Yêu cầu người dùng cung cấp thêm thông tin hoặc xử lý khi bị từ chối.
- `risky_action`: Chuẩn bị mô tả hành động nhạy cảm để đưa qua approval gate.
- `approval`: Kiểm soát phê duyệt hành động rủi ro (hỗ trợ cả mock lẫn interrupt).
- `retry`: Ghi nhận số lần thử lại (attempt counter), kiểm soát vòng lặp hữu hạn.
- `dead_letter`: Xử lý ngoại lệ khi vượt quá số lần retry tối đa (escalate).
- `finalize`: Node kết thúc ghi nhận audit event cuối cùng trước khi tới END.

Hệ thống có 8 cạnh cố định và 4 hàm định tuyến có điều kiện:
- `route_after_classify`: Phân nhánh sang answer, tool, clarify, risky_action hoặc retry.
- `route_after_evaluate`: Quyết định retry nếu lỗi hoặc answer nếu thành công.
- `route_after_retry`: Chuyển sang tool nếu attempt < max_attempts, hoặc dead_letter nếu attempt >= max_attempts.
- `route_after_approval`: Chuyển sang tool nếu được duyệt (approved=True), hoặc clarify nếu bị từ chối.
Mọi nhánh trong graph đều đảm bảo kết thúc tại `finalize -> END`.

## 3. Thiết kế State Schema & Reducers

Bảng phân loại các trường trong `AgentState`:

| Tên Field | Kiểu dữ liệu | Cơ chế cập nhật | Mục đích sử dụng |
|---|---|---|---|
| `thread_id` | `str` | Overwrite | Định danh phiên làm việc cho Checkpointer |
| `scenario_id` | `str` | Overwrite | Định danh kịch bản phục vụ báo cáo và metrics |
| `query` | `str` | Overwrite | Nội dung yêu cầu hỗ trợ sau khi chuẩn hóa |
| `route` | `str` | Overwrite | Nhãn phân loại ban đầu từ classify_node |
| `risk_level` | `str` | Overwrite | Mức độ rủi ro ('low' hoặc 'high') |
| `attempt` | `int` | Overwrite | Bộ đếm số lần đã retry trong vòng lặp |
| `max_attempts` | `int` | Overwrite | Giới hạn retry tối đa |
| `final_answer` | `str | None` | Overwrite | Câu trả lời cuối cùng |
| `evaluation_result` | `str | None` | Overwrite | Kết quả đánh giá tool |
| `pending_question` | `str | None` | Overwrite | Câu hỏi làm rõ thông tin |
| `proposed_action` | `str | None` | Overwrite | Hành động đề xuất chờ duyệt |
| `approval` | `dict | None` | Overwrite | Quyết định duyệt |
| `messages` | `list[str]` | Append-only | Lịch sử trao đổi qua các node |
| `tool_results` | `list[str]` | Append-only | Lịch sử kết quả tool |
| `errors` | `list[str]` | Append-only | Lịch sử lỗi theo thời gian |
| `events` | `list[dict]` | Append-only | Audit trail chuẩn hóa |

## 4. Kết quả thực thi Scenarios (Metrics)

### Tổng quan metrics:
- **Tổng số kịch bản**: 7
- **Tỷ lệ thành công**: 100.00%
- **Số node trung bình đi qua**: 8.43
- **Tổng số lần retry**: 3
- **Tổng số approval-node visits**: 2
- **Resume evidence**: chưa chứng minh
- `total_interrupts` đếm approval-node visits; không tự chứng minh interrupt thật hoặc resume thành công.

### Bảng chi tiết từng kịch bản:

| Kịch bản | Expected | Actual | Kết quả | Retries | Approval | Latency |
|---|---|---|:---:|---:|---:|---:|
| `S01_simple` | `simple` | `simple` | **PASS** | 0 | 0 | 49483 ms |
| `S02_tool` | `tool` | `tool` | **PASS** | 0 | 0 | 8677 ms |
| `S03_missing` | `missing_info` | `missing_info` | **PASS** | 0 | 0 | 8641 ms |
| `S04_risky` | `risky` | `risky` | **PASS** | 0 | 1 | 9594 ms |
| `S05_error` | `error` | `error` | **PASS** | 2 | 0 | 15874 ms |
| `S06_delete` | `risky` | `risky` | **PASS** | 0 | 1 | 11371 ms |
| `S07_dead_letter` | `error` | `error` | **PASS** | 1 | 0 | 27564 ms |

## 5. Phân tích các chế độ lỗi (Failure Analysis)

### Failure Mode 1: Lỗi công cụ tạm thời và giới hạn vòng lặp Retry
- **Cơ chế phát hiện**: Khi node `tool` trả về chuỗi chứa `ERROR`, node `evaluate` phát hiện và gắn `evaluation_result = 'needs_retry'`.
- **Luồng xử lý**: Conditional edge `route_after_evaluate` điều hướng về node `retry`.
Tại đây, `attempt` được tăng thêm 1 và ghi nhận vào `errors`.
`route_after_retry` kiểm tra `attempt < max_attempts`.
Nếu còn lượt, graph gọi lại `tool`; nếu đã chạm ngưỡng `max_attempts` 
(như kịch bản S07 với max_attempts=1), chuyển thẳng sang `dead_letter` và `finalize`.
- **Đảm bảo an toàn**: Bộ đếm chỉ được tăng tại duy nhất node `retry`, ngăn chặn triệt để nguy cơ vòng lặp vô hạn mà không cần dựa vào recursion limit.

### Failure Mode 2: Hành động rủi ro bị từ chối phê duyệt (Approval Gate)
- **Cơ chế phát hiện**: Khi người dùng yêu cầu hoàn tiền hoặc xóa tài khoản, LLM phân loại vào route `risky`.
Graph đi qua `risky_action` để tạo đề xuất rồi tới `approval`.
- **Luồng xử lý**: Contract test offline xác nhận nếu quyết định là `approved: False`, conditional edge `route_after_approval` lập tức chuyển hướng sang node `clarify`.
Tại `clarify`, agent tạo câu hỏi phản hồi giải thích lý do từ chối và hướng dẫn người dùng cung cấp thêm xác minh, sau đó đi thẳng tới `finalize`.
- **Đảm bảo an toàn**: Tuyệt đối không có đường dẫn nào cho phép tool thực thi tác vụ rủi ro nếu chưa qua gate phê duyệt thành công.

### Failure Mode 3: LLM hoặc interrupt provider unavailable
- Classify và answer dùng fallback có đánh dấu `fallback` trong event trail và ghi error type trong `errors`, không ghi exception raw vào output.
- Nếu real interrupt lỗi, approval fail closed với `approved: false`; workflow không tự động thực hiện risky action.

## 6. Bằng chứng Persistence và Phục hồi (Persistence Evidence)

- Sample run dùng `MemorySaver` theo `configs/lab.yaml`; metrics hiện ghi `resume_success = false`.
- Mỗi phiên chạy nhận một `thread_id` duy nhất (ví dụ `thread-S01_simple`), đảm bảo cô lập state hoàn toàn giữa các kịch bản.
- SQLite adapter có WAL mode và test deterministic tạo graph/checkpointer mới để đọc lại state/history; đây là evidence riêng, không được nâng `resume_success` nếu chưa replay qua process restart thực.
- Lệnh kiểm chứng local: `python -m pytest tests/test_persistence_extension.py -q` với kết quả 2 passed.

## 7. Các phần mở rộng (Bonus Extensions)

1. **SQLite Checkpointer (`SqliteSaver`)**: Đã implement adapter, WAL mode và test đọc lại state/history bằng graph/checkpointer mới.
2. **Mermaid graph export**: Đã sinh `outputs/graph_mermaid.md` từ graph thực tế.
3. **Real HITL interrupt/resume**: Chưa claim đã chạy; code path `LANGGRAPH_INTERRUPT=true` là extension cần evidence resume riêng.

## 8. Kế hoạch hoàn thiện và đưa vào sản xuất (Improvement Plan)

Nếu có thêm thời gian phát triển, các hạng mục ưu tiên gồm:
1. **Parallel Fan-out với `Send()`**: Tối ưu hóa việc gọi đồng thời nhiều tool độc lập khi khách hàng có nhiều yêu cầu phức tạp trong một ticket.
2. **LLM-as-Judge tự động**: Tích hợp module đánh giá chất lượng câu trả lời và mức độ an toàn trước khi gửi cho khách hàng.
3. **Streaming & Dashboard UI**: Xây dựng giao diện trực quan bằng Streamlit hiển thị thời gian thực luồng chuyển đổi trạng thái và nút bấm phê duyệt cho supervisor.
