from repomaro.metrics import file_prf, gold_coverage


def test_file_metrics():
    prf = file_prf(["a.py","b.py"], ["b.py","c.py"])
    assert prf.precision == 0.5
    assert prf.recall == 0.5
    assert prf.f1 == 0.5
    assert gold_coverage(["a.py","b.py"], ["b.py","c.py"]) == 0.5
