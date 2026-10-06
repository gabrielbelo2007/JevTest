import pytest
from bench.metrics import (
    compute_accuracy,
    compute_macro_f1,
    compute_brier_score,
    compute_ece,
    compute_score_metrics,
    compute_latency_stats,
    compute_paired_bootstrap_accuracy_diff,
)
from bench.clients.base import Prediction
from bench.tasks import Case

def test_compute_accuracy():
    assert compute_accuracy(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert compute_accuracy(["a", "b", "c"], ["a", "x", "c"]) == pytest.approx(2 / 3)
    assert compute_accuracy([], []) == 0.0

def test_compute_macro_f1():
    golds = ["a", "b", "a", "b"]
    preds = ["a", "b", "a", "b"]
    assert compute_macro_f1(golds, preds) == 1.0

def test_compute_brier_score_binary():
    cases = [
        Case("1", "sst2", "noul", {}, {}, True),
        Case("2", "sst2", "noul", {}, {}, False),
    ]
    preds = [
        Prediction("1", "sst2", "m", True, None, True, None, 0.9, None, 10.0),
        Prediction("2", "sst2", "m", True, None, False, None, 0.1, None, 10.0),
    ]
    # (0.9 - 1)^2 = 0.01; (0.1 - 0)^2 = 0.01; mean = 0.01
    brier = compute_brier_score(cases, preds)
    assert brier == pytest.approx(0.01)

def test_compute_ece():
    cases = [
        Case("1", "sst2", "noul", {}, {}, True),
        Case("2", "sst2", "noul", {}, {}, False),
    ]
    preds = [
        Prediction("1", "sst2", "m", True, None, True, None, 1.0, None, 10.0),
        Prediction("2", "sst2", "m", True, None, False, None, 0.0, None, 10.0),
    ]
    # Perfect calibration at 1.0 confidence
    ece = compute_ece(cases, preds)
    assert ece == pytest.approx(0.0, abs=1e-5)

def test_compute_score_metrics():
    cases = [
        Case("1", "sst5", "score", {}, {}, 4),
        Case("2", "sst5", "score", {}, {}, 2),
    ]
    preds = [
        Prediction("1", "sst5", "m", True, None, 4, None, None, 3.8, 10.0),
        Prediction("2", "sst5", "m", True, None, 2, None, None, 2.0, 10.0),
    ]
    res = compute_score_metrics(cases, preds)
    assert res["exact_acc"] == 1.0
    assert res["mae"] == pytest.approx(0.1)
    assert res["spearman"] == pytest.approx(1.0)

def test_compute_latency_stats():
    lats = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = compute_latency_stats(lats)
    assert stats["p50"] == 30.0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["mean"] == 30.0

def test_compute_paired_bootstrap_accuracy_diff():
    g = ["a", "b", "c", "d"]
    pa = ["a", "b", "c", "d"]
    pb = ["x", "x", "x", "x"]
    delta, low, high = compute_paired_bootstrap_accuracy_diff(g, pa, pb, n_boot=200)
    assert delta == 1.0
    assert low == 1.0
    assert high == 1.0
