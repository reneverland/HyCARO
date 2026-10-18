import numpy as np
from repomaro.decoder import BudgetAwareDecoder


def test_decoder_respects_budget():
    d = BudgetAwareDecoder(np.array([40, 50, 35, 60, 30]), budget=100, threshold=0.5)
    x = np.array([0.9, 0.8, 0.7, 0.2, 0.6])
    mask = d.decode(x)
    assert d.feasible(mask)
    assert int(d.costs[mask].sum()) <= 100
    assert mask.any()


def test_batch_decoder_shape():
    d = BudgetAwareDecoder(np.array([20, 30, 40]), budget=50, threshold=0.5)
    X = np.array([[0.9, 0.8, 0.1], [0.1, 0.9, 0.8]])
    M = d.decode_batch(X)
    assert M.shape == X.shape
    assert all(d.feasible(m) for m in M)
