from __future__ import annotations

from typing import Any
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score
from bench.clients.base import Prediction
from bench.tasks import Case

def compute_accuracy(golds: list[Any], preds: list[Any]) -> float:
    if not golds:
        return 0.0
    correct = sum(1 for g, p in zip(golds, preds) if g == p)
    return correct / len(golds)

def compute_macro_f1(golds: list[Any], preds: list[Any]) -> float:
    if not golds:
        return 0.0
    # Convert bools or ints to string representation for unified comparison
    g_str = [str(g) for g in golds]
    p_str = [str(p) if p is not None else "__NONE__" for p in preds]
    labels = sorted(list(set(g_str)))
    return float(f1_score(g_str, p_str, labels=labels, average="macro", zero_division=0))

def compute_brier_score(
    cases: list[Case],
    preds: list[Prediction],
) -> float | None:
    """Compute Brier score across choice or noul predictions with probability distributions."""
    errors = []
    for c, p in zip(cases, preds):
        if not p.ok:
            continue
        if c.qtype == "noul":
            if p.p_true is not None:
                y = 1.0 if c.gold is True else 0.0
                errors.append((float(p.p_true) - y) ** 2)
        elif c.qtype == "choice":
            if p.probs:
                # multi-class Brier score
                all_labels = set(p.probs.keys())
                all_labels.add(str(c.gold))
                sq_err = 0.0
                for lbl in all_labels:
                    prob = float(p.probs.get(lbl, 0.0))
                    target = 1.0 if str(c.gold) == lbl else 0.0
                    sq_err += (prob - target) ** 2
                errors.append(sq_err)
    if not errors:
        return None
    return float(np.mean(errors))

def compute_ece(
    cases: list[Case],
    preds: list[Prediction],
    n_bins: int = 10,
) -> float | None:
    """Compute Expected Calibration Error (ECE) for top-predicted class."""
    confidences = []
    accuracies = []
    for c, p in zip(cases, preds):
        if not p.ok:
            continue
        if c.qtype == "choice" and p.probs:
            best_label = max(p.probs, key=p.probs.get)
            conf = float(p.probs[best_label])
            acc = 1.0 if best_label == str(c.gold) else 0.0
            confidences.append(conf)
            accuracies.append(acc)
        elif c.qtype == "noul" and p.p_true is not None:
            conf = float(p.p_true) if float(p.p_true) >= 0.5 else 1.0 - float(p.p_true)
            pred_bool = float(p.p_true) >= 0.5
            acc = 1.0 if pred_bool == bool(c.gold) else 0.0
            confidences.append(conf)
            accuracies.append(acc)

    if not confidences:
        return None

    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[bin_mask])
            bin_conf = np.mean(confidences[bin_mask])
            ece += (bin_size / len(confidences)) * abs(bin_acc - bin_conf)

    return float(ece)

def compute_score_metrics(cases: list[Case], preds: list[Prediction]) -> dict[str, float | None]:
    """Compute MAE, Exact Accuracy, and Spearman correlation for score tasks."""
    golds = []
    estimates = []
    discrete_preds = []
    for c, p in zip(cases, preds):
        if not p.ok:
            continue
        golds.append(float(c.gold))
        est = p.score_value if p.score_value is not None else (float(p.pred) if p.pred is not None else 0.0)
        estimates.append(float(est))
        discrete_preds.append(round(float(est)))

    if not golds:
        return {"mae": None, "spearman": None, "exact_acc": None}

    mae = float(np.mean(np.abs(np.array(estimates) - np.array(golds))))
    exact_acc = float(np.mean(np.array(discrete_preds) == np.array(golds)))
    
    if len(set(estimates)) > 1 and len(set(golds)) > 1:
        corr, _ = stats.spearmanr(estimates, golds)
        spearman = float(corr)
    else:
        spearman = 0.0

    return {"mae": mae, "spearman": spearman, "exact_acc": exact_acc}

def compute_latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {
            "mean": 0.0,
            "std": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    arr = np.array(latencies_ms)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

def compute_paired_bootstrap_accuracy_diff(
    golds: list[Any],
    preds_a: list[Any],
    preds_b: list[Any],
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute mean delta and 95% bootstrap confidence interval (A - B)."""
    n = len(golds)
    if n == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    acc_a_point = sum(1 for g, p in zip(golds, preds_a) if g == p) / n
    acc_b_point = sum(1 for g, p in zip(golds, preds_b) if g == p) / n
    delta_point = acc_a_point - acc_b_point

    correct_a = np.array([1 if g == p else 0 for g, p in zip(golds, preds_a)])
    correct_b = np.array([1 if g == p else 0 for g, p in zip(golds, preds_b)])

    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d = np.mean(correct_a[idx]) - np.mean(correct_b[idx])
        deltas.append(d)

    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))
    return float(delta_point), ci_low, ci_high
