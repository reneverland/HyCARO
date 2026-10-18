# External context-management / agentic-search baselines

The package does **not** silently reimplement LongCodeZip, RepoDistill, LocAgent, or CodeScout. Their published systems have task-specific interfaces, trained weights, and/or search procedures; approximating them locally would not be a fair baseline.

Run each method with its official implementation, then export a JSONL file with at least:

```json
{"instance_id":"...", "method":"LocAgent", "selected_paths":["a.py","b.py"], "selector_time_seconds":1.23}
```

For ARB, normalize/evaluate it with:

```bash
python scripts/evaluate_external_arb.py --samples ... --pred external_locagent.jsonl --out results/locagent_arb.jsonl
```

For ContextBench, convert the method output to ContextBench's unified `pred_files` / `pred_spans` trajectory format and use the official evaluator. For SWE-bench, serialize the selected context into the same downstream prompt used by `scripts/run_swebench.py` so that only context selection differs across methods.
