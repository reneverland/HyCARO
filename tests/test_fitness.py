import numpy as np
from repomaro.fitness import FitnessCache, SurrogateFitness


def test_batch_matches_scalar(toy_pool):
    f = SurrogateFitness(toy_pool, 0.45, 0.35, 0.2, sigma=2.0)
    masks = np.array([
        [1,0,1,0,0,0],
        [1,1,0,0,0,0],
        [0,0,1,0,1,0],
    ], dtype=bool)
    batch = f.evaluate_batch(masks)
    scalar = np.array([f.evaluate(m) for m in masks])
    np.testing.assert_allclose(batch, scalar, atol=1e-12)


def test_incremental_add_drop_swap_exact(toy_pool):
    f = SurrogateFitness(toy_pool, 0.45, 0.35, 0.2, sigma=2.0)
    base = np.array([1,0,1,0,0,0], dtype=bool)
    s = f.build_local_state(base)

    add = base.copy(); add[4] = True
    assert abs(s.score_add(4) - f.evaluate(add)) < 1e-12

    drop = base.copy(); drop[2] = False
    assert abs(s.score_drop(2) - f.evaluate(drop)) < 1e-12

    swap = base.copy(); swap[2] = False; swap[4] = True
    assert abs(s.score_swap(2, 4) - f.evaluate(swap)) < 1e-12


def test_cache_deduplicates(toy_pool):
    f = SurrogateFitness(toy_pool, 0.45, 0.35, 0.2)
    c = FitnessCache(f)
    m = np.array([1,0,1,0,0,0], dtype=bool)
    vals = c.evaluate_batch(np.stack([m,m,m]))
    assert np.allclose(vals, vals[0])
    assert c.misses == 1
    assert c.hits >= 2
