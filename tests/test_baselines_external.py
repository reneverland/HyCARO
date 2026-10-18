import json

from repomaro.baselines_external import locagent_baseline, longcodezip_baseline, repomap_baseline
from repomaro.fitness import SurrogateFitness


def _write_ranking(tmp_path, pool, ranking):
    path = tmp_path / "ranking.jsonl"
    path.write_text(json.dumps({
        "instance_id": "x",
        "root": str(pool.repo_root.resolve()),
        "query": pool.query,
        "ranking": ranking,
    }) + "\n")
    return str(path)


def test_file_level_longcodezip_packs_named_path(toy_pool, tmp_path):
    ranking = _write_ranking(tmp_path, toy_pool, [{"path": "f2.py", "line": 0, "rank": 0}])
    fitness = SurrogateFitness(toy_pool, 0.45, 0.35, 0.2)
    result = longcodezip_baseline(toy_pool, 40, fitness, ranking)
    assert result.name == "LongCodeZip (coarse)"
    assert result.mask[2]
    assert int(result.mask.sum()) == 1


def test_locagent_file_hits_all_units_on_path(toy_pool, tmp_path):
    ranking = _write_ranking(tmp_path, toy_pool, [{"path": "f0.py", "line": 0, "rank": 0}])
    fitness = SurrogateFitness(toy_pool, 0.45, 0.35, 0.2)
    result = locagent_baseline(toy_pool, 40, fitness, ranking)
    assert result.mask[0]
    assert result.metadata["units_matched"] == 1


def test_repomap_line_hit_still_works(toy_pool, tmp_path):
    ranking = _write_ranking(tmp_path, toy_pool, [{"path": "f4.py", "line": 1, "rank": 0}])
    fitness = SurrogateFitness(toy_pool, 0.45, 0.35, 0.2)
    result = repomap_baseline(toy_pool, 40, fitness, ranking)
    assert result.mask[4]
