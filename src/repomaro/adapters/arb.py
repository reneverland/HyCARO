from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..checkout import ensure_checkout
from ..schema import SelectionInstance


def load_arb_samples(
    sample_paths: Iterable[str | Path],
    checkout_cache: str | Path,
    limit: int | None = None,
    offset: int = 0,
) -> list[SelectionInstance]:
    """Load official ARB releases through the installed `agent_retrieval_bench` package."""
    try:
        from agent_retrieval_bench.baseline import iter_samples, query_text_for_eval, target_gold_files  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Agent Retrieval Bench is not installed. Run scripts/setup_benchmarks.sh first."
        ) from e
    paths = [Path(p) for p in sample_paths]
    out: list[SelectionInstance] = []
    eligible = 0
    for sample in iter_samples(paths):
        gold = target_gold_files(sample)
        if not gold:
            continue
        repo = sample.get("repo")
        commit = sample.get("base_commit")
        if not repo or not commit:
            continue
        if eligible < int(offset or 0):
            eligible += 1
            continue
        eligible += 1
        if not (len(out) % 25):
            print(f"  checkout {offset + len(out) + 1} {repo}@{str(commit)[:10]}", flush=True)
        try:
            root = ensure_checkout(str(repo), str(commit), checkout_cache)
        except Exception as exc:
            print(f"  skip {sample.get('id')}: {exc}", flush=True)
            continue
        out.append(
            SelectionInstance(
                instance_id=str(sample.get("id")),
                query=query_text_for_eval(sample),
                repo_root=root,
                gold_paths=list(gold),
                metadata={"repo": repo, "base_commit": commit, "task_type": sample.get("task_type")},
            )
        )
        if limit and len(out) >= limit:
            break
    return out
