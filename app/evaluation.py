"""Evaluation and the approval gate. The gate lives here only (single source of truth)."""
from __future__ import annotations

import re
import time
from collections import Counter

from .util import percentile

_PUNCT = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return 1.0 if p == g else 0.0
    overlap = sum((Counter(p) & Counter(g)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def evaluate(predictor, rows: list[dict], task_type: str, max_new_tokens: int = 128) -> dict:
    if not rows:
        raise ValueError("validation set is empty")
    ems, f1s, lat, empty, errors = [], [], [], 0, 0
    for row in rows:
        start = time.perf_counter()
        try:
            out = predictor.generate(row["prompt"], max_new_tokens, 0.0).text
        except Exception:
            errors += 1
            out = ""
        lat.append((time.perf_counter() - start) * 1000.0)
        if not out.strip():
            empty += 1
        ems.append(exact_match(out, row["response"]))
        f1s.append(token_f1(out, row["response"]))
    n = len(rows)
    em, f1 = sum(ems) / n, sum(f1s) / n
    return {
        "n": n, "exact_match": round(em, 4), "token_f1": round(f1, 4),
        "primary": round(em if task_type == "classification" else f1, 4),
        "primary_metric": "exact_match" if task_type == "classification" else "token_f1",
        "empty_rate": round(empty / n, 4), "errors": errors,
        "p50_ms": round(percentile(lat, 50), 2), "p95_ms": round(percentile(lat, 95), 2),
    }


def decide(candidate: dict, baseline: dict, gate: dict, min_val_for_auto_approve: int) -> tuple[str, list[str]]:
    """Return (approved | rejected | needs_review, reasons)."""
    reasons: list[str] = []
    if candidate["errors"] > 0:
        reasons.append(f"{candidate['errors']} generation errors during evaluation.")
    if candidate["primary"] < gate["min_score"]:
        reasons.append(f"Score {candidate['primary']:.3f} is below the minimum {gate['min_score']:.3f}.")
    gain = candidate["primary"] - baseline["primary"]
    if gain < gate["min_improvement"]:
        reasons.append(
            f"Improvement over baseline is {gain:+.3f}; at least {gate['min_improvement']:+.3f} is required."
        )
    if candidate["empty_rate"] > 0.1:
        reasons.append(f"{candidate['empty_rate']:.0%} of outputs were empty.")
    if candidate["p95_ms"] > gate["max_p95_latency_ms"]:
        reasons.append(f"p95 latency {candidate['p95_ms']:.0f} ms is above the limit {gate['max_p95_latency_ms']:.0f} ms.")
    if reasons:
        return "rejected", reasons
    if candidate["n"] < min_val_for_auto_approve:
        return "needs_review", [
            f"Validation set has {candidate['n']} examples (< {min_val_for_auto_approve}). "
            "Scores are too noisy for automatic approval."
        ]
    return "approved", [f"Score {candidate['primary']:.3f}, {gain:+.3f} better than baseline."]
