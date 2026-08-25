# StateGraph Mermaid Architecture

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	intake(intake)
	prompt_guardrail(prompt_guardrail)
	query_rewrite(query_rewrite)
	parallel_worker(parallel_worker)
	aggregate_answers(aggregate_answers)
	classify(classify)
	answer(answer)
	tool(tool)
	evaluate(evaluate)
	clarify(clarify)
	risky_action(risky_action)
	approval(approval)
	retry(retry)
	dead_letter(dead_letter)
	finalize(finalize)
	__end__([<p>__end__</p>]):::last
	__start__ --> intake;
	aggregate_answers -.-> approval;
	aggregate_answers -.-> finalize;
	answer --> finalize;
	approval -.-> clarify;
	approval -.-> finalize;
	approval -.-> tool;
	clarify --> finalize;
	classify -.-> answer;
	classify -.-> clarify;
	classify -.-> retry;
	classify -.-> risky_action;
	classify -.-> tool;
	dead_letter --> finalize;
	evaluate -.-> answer;
	evaluate -.-> retry;
	intake --> prompt_guardrail;
	parallel_worker --> aggregate_answers;
	prompt_guardrail -.-> clarify;
	prompt_guardrail -.-> query_rewrite;
	query_rewrite -.-> classify;
	query_rewrite -.-> parallel_worker;
	retry -.-> dead_letter;
	retry -.-> tool;
	risky_action --> approval;
	tool --> evaluate;
	finalize --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```
