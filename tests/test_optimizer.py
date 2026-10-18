from repomaro.optimizer import RepoMARO


def test_repomaro_returns_feasible(toy_pool):
    fcfg = {"alpha":0.45,"beta":0.35,"gamma":0.2,"sigma":2.0}
    ocfg = {
        "population_size": 8,
        "max_iterations": 12,
        "threshold": 0.5,
        "lambda_min":0.15,
        "lambda_max":0.8,
        "crossover_rate":0.7,
        "elite_fraction":0.5,
        "local_proposals":6,
        "local_search_patience":2,
        "stopping_patience":5,
        "epsilon":1e-6,
        "seed_high":0.9,
        "seed_low":0.1,
        "seed_noise":0.02,
        "enable_cache":True,
        "enable_incremental_local":True,
        "enable_local_search":True,
        "enable_early_stopping":True,
        "enable_retrieval_seeding":True,
    }
    opt = RepoMARO(toy_pool, budget=120, fitness_cfg=fcfg, optimizer_cfg=ocfg, seed=3)
    result = opt.optimize()
    assert opt.decoder.feasible(result.mask)
    assert result.mask.any()
    assert result.iterations <= 12
    assert len(result.history) == result.iterations
    assert result.cache_misses > 0


def test_node_repomaro_uses_unrestricted_pool(toy_pool):
    fcfg = {"alpha": 0.45, "beta": 0.35, "gamma": 0.2, "sigma": 2.0}
    ocfg = {
        "population_size": 8,
        "max_iterations": 6,
        "threshold": 0.5,
        "lambda_min": 0.15,
        "lambda_max": 0.8,
        "crossover_rate": 0.7,
        "elite_fraction": 0.5,
        "local_proposals": 6,
        "local_search_patience": 2,
        "stopping_patience": 5,
        "epsilon": 1e-6,
        "unrestricted_pool": True,
        "ball_anchors": 1,
        "ball_radius": 0,
    }
    opt = RepoMARO(toy_pool, budget=120, fitness_cfg=fcfg, optimizer_cfg=ocfg, seed=3)
    assert int(opt.member.sum()) == len(toy_pool.units)
    result = opt.optimize()
    assert opt.decoder.feasible(result.mask)
