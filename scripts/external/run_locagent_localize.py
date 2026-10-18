#!/usr/bin/env python
"""Official LocAgent auto-search localize, then emit our ranking JSONL.

Must run with PYTHONPATH=external/LocAgent and GRAPH_INDEX_DIR / BM25_INDEX_DIR
set. Search LLM is the same OpenAI-compatible endpoint as downstream generation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external" / "LocAgent"))

# LocAgent's search loop only needs torch.multiprocessing. A full CUDA torch
# install is unnecessary in the localize venv; fall back to stdlib if missing.
try:
    import torch.multiprocessing  # noqa: F401
except ModuleNotFoundError:
    import multiprocessing as _mp
    import types

    _torch = types.ModuleType("torch")
    _torch.multiprocessing = _mp
    _torch.Tensor = object
    sys.modules.setdefault("torch", _torch)
    sys.modules["torch.multiprocessing"] = _mp


def _locs_to_ranking(row: dict) -> list[dict]:
    ranking: list[dict] = []
    seen: set[str] = set()
    rank = 0

    def add(path: str, name: str = "", line: int = 0) -> None:
        nonlocal rank
        path = str(path or "").replace("\\", "/").lstrip("./")
        if not path:
            return
        key = f"{path}|{name}|{line}"
        if key in seen:
            return
        seen.add(key)
        ranking.append({"path": path, "line": int(line or 0), "rank": rank, "name": name})
        rank += 1

    entities = row.get("found_entities") or []
    files = row.get("found_files") or []
    modules = row.get("found_modules") or []
    # A merged row is a flat list; a raw row is list-of-samples.
    if entities and isinstance(entities[0], str):
        entity_iter = [entities]
        file_iter = [files] if files and isinstance(files[0], str) else files
        module_iter = [modules] if modules and isinstance(modules[0], str) else modules
    else:
        entity_iter = entities
        file_iter = files
        module_iter = modules
    for sample in entity_iter:
        for ent in sample or []:
            text = str(ent)
            path, _, rest = text.partition(":")
            add(path, rest or text)
    for sample in module_iter:
        for mod in sample or []:
            text = str(mod)
            path, _, rest = text.partition(":")
            add(path, rest or text)
    for sample in file_iter:
        for path in sample or []:
            add(str(path))
    return ranking


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def localize_one(bug: dict, args) -> dict:
    from auto_search_main import auto_search_process, get_task_instruction
    from plugins import LocationToolsRequirement
    from plugins.location_tools.repo_ops.repo_ops import reset_current_issue, set_current_issue
    from util.prompts.pipelines import auto_search_prompt as auto_search
    from util.prompts.prompt import PromptManager
    from util.process_output import get_loc_results_from_raw_outputs
    from util.runtime import function_calling
    import torch.multiprocessing as mp

    instance_id = bug["instance_id"]
    set_current_issue(instance_data=bug, rank=0)
    prompt_manager = PromptManager(
        prompt_dir=str(ROOT / "external" / "LocAgent" / "util" / "prompts"),
        agent_skills_docs=LocationToolsRequirement.documentation,
    )
    raw_output_loc = []
    for _ in range(int(args.num_samples)):
        if args.use_function_calling:
            system_prompt = function_calling.SYSTEM_PROMPT
        else:
            system_prompt = prompt_manager.system_message
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({
            "role": "user",
            "content": get_task_instruction(bug, include_pr=True, include_hint=True),
        })
        ctx = mp.get_context("fork")
        result_queue = ctx.Manager().Queue()
        tools = None
        if args.use_function_calling:
            tools = function_calling.get_tools(
                codeact_enable_search_keyword=True,
                codeact_enable_search_entity=True,
                codeact_enable_tree_structure_traverser=True,
                simple_desc=args.simple_desc,
            )
        process = ctx.Process(
            target=auto_search_process,
            kwargs={
                "result_queue": result_queue,
                "model_name": args.model,
                "messages": messages,
                "fake_user_msg": auto_search.FAKE_USER_MSG_FOR_LOC,
                "temp": 1,
                "tools": tools,
                "use_function_calling": args.use_function_calling,
            },
        )
        process.start()
        process.join(timeout=args.timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            continue
        try:
            result = result_queue.get_nowait()
        except Exception:
            continue
        if isinstance(result, dict) and "error" in result:
            continue
        loc_result, _messages, _traj = result
        if loc_result:
            raw_output_loc.append(loc_result)
    reset_current_issue()
    if not raw_output_loc:
        return {
            "instance_id": instance_id,
            "found_files": [[]],
            "found_modules": [[]],
            "found_entities": [[]],
        }
    files, modules, entities = get_loc_results_from_raw_outputs(instance_id, raw_output_loc)
    return {
        "instance_id": instance_id,
        "found_files": files,
        "found_modules": modules,
        "found_entities": entities,
        "raw_output_loc": raw_output_loc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--locs-out", default="cache/locagent_locs_lite300.jsonl")
    ap.add_argument("--out", default="cache/locagent_rankings_lite300.jsonl")
    ap.add_argument("--model", default=os.environ.get("LOCAGENT_MODEL", "openai/gpt-5.6-luna"))
    ap.add_argument("--num-samples", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--use-function-calling", action="store_true", default=True)
    ap.add_argument("--no-function-calling", action="store_true")
    ap.add_argument("--simple-desc", action="store_true", default=True)
    args = ap.parse_args()
    if args.no_function_calling:
        args.use_function_calling = False

    if not os.environ.get("GRAPH_INDEX_DIR"):
        raise SystemExit("GRAPH_INDEX_DIR is required")
    if not os.environ.get("BM25_INDEX_DIR"):
        os.environ["BM25_INDEX_DIR"] = str(ROOT / "cache" / "locagent_bm25")
        Path(os.environ["BM25_INDEX_DIR"]).mkdir(parents=True, exist_ok=True)

    rows = [
        json.loads(l)
        for l in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    locs_out = Path(args.locs_out)
    rank_out = Path(args.out)
    done = set()
    if locs_out.exists():
        for line in locs_out.open(encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line).get("instance_id") or "")
    print(
        f"LocAgent localize {len(rows)} rows, {len(done)} already written, "
        f"model={args.model} -> {rank_out}",
        flush=True,
    )
    for i, row in enumerate(rows, 1):
        iid = row["instance_id"]
        if iid in done:
            continue
        bug = {
            "instance_id": iid,
            "repo": row.get("repo") or "",
            "base_commit": row.get("base_commit") or "",
            "problem_statement": row.get("query") or row.get("issue") or "",
            "patch": "",
        }
        print(f"[{i}/{len(rows)}] {iid}", flush=True)
        t0 = time.time()
        try:
            loc = localize_one(bug, args)
        except Exception as exc:
            print(f"  localize failed {iid}: {exc}", flush=True)
            loc = {
                "instance_id": iid,
                "found_files": [[]],
                "found_modules": [[]],
                "found_entities": [[]],
                "error": str(exc),
            }
        loc["time"] = time.time() - t0
        _append(locs_out, loc)
        _append(rank_out, {
            "instance_id": iid,
            "root": str(Path(row["root"]).resolve()),
            "query": row.get("query") or row.get("issue") or "",
            "ranking": _locs_to_ranking(loc),
        })
        done.add(iid)
    print(f"done {len(done)}", flush=True)


if __name__ == "__main__":
    main()
