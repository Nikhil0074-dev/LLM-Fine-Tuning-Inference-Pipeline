"""A tiny, dependency-free 'model': TF-IDF nearest-neighbour over (prompt -> response) pairs.

Why it exists: it lets you run and test the *whole* pipeline (validate, train, evaluate, gate,
deploy, serve, cache, roll back) on a laptop with no GPU and no cost. It is a strong baseline for
FAQ and classification data. It is NOT a language model and cannot write new text.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Callable

from .base import Backend, GenResult, Predictor, TrainingCancelled

_TOKEN = re.compile(r"[a-z0-9]+")
FALLBACK = "I do not have enough information to answer that."
MIN_SIMILARITY = 0.15


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class NearestPredictor:
    def __init__(self, examples: list[dict], idf: dict[str, float]):
        self.idf = idf
        self.examples = []
        for ex in examples:
            vec = self._vector(ex["prompt"])
            self.examples.append((vec, self._norm(vec), ex["response"]))

    def _vector(self, text: str) -> dict[str, float]:
        counts = Counter(tokenize(text))
        return {t: c * self.idf.get(t, 0.0) for t, c in counts.items() if t in self.idf}

    @staticmethod
    def _norm(vec: dict[str, float]) -> float:
        return math.sqrt(sum(v * v for v in vec.values()))

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> GenResult:
        q = self._vector(prompt)
        qn = self._norm(q)
        best_score, best = 0.0, FALLBACK
        if qn > 0:
            for vec, norm, response in self.examples:
                if norm == 0:
                    continue
                dot = sum(w * vec.get(t, 0.0) for t, w in q.items())
                score = dot / (qn * norm)
                if score > best_score:
                    best_score, best = score, response
        text = best if best_score >= MIN_SIMILARITY else FALLBACK
        words = text.split()
        if len(words) > max_new_tokens:
            text = " ".join(words[:max_new_tokens])
        return GenResult(text=text, input_tokens=len(tokenize(prompt)), output_tokens=len(text.split()))


class MajorityPredictor:
    """Baseline: always answers with the most common training response."""

    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> GenResult:
        text = " ".join(self.response.split()[:max_new_tokens])
        return GenResult(text=text, input_tokens=len(tokenize(prompt)), output_tokens=len(text.split()))


class SimpleBackend(Backend):
    name = "simple"
    uses_gpu = False

    def train(self, train_rows: list[dict], config: dict, should_cancel: Callable[[], bool]) -> bytes:
        if not train_rows:
            raise ValueError("no training rows")
        df: Counter = Counter()
        for i, row in enumerate(train_rows):
            if i % 50 == 0 and should_cancel():
                raise TrainingCancelled()
            df.update(set(tokenize(row["prompt"])))
        n = len(train_rows)
        idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
        return json.dumps({"type": "simple-nn", "examples": train_rows, "idf": idf}).encode("utf-8")

    def load(self, artifact: bytes) -> Predictor:
        data = json.loads(artifact.decode("utf-8"))
        if data.get("type") != "simple-nn":
            raise ValueError("not a simple-nn artifact")
        return NearestPredictor(data["examples"], data["idf"])

    def baseline(self, train_rows: list[dict], config: dict) -> Predictor:
        common = Counter(r["response"] for r in train_rows).most_common(1)
        return MajorityPredictor(common[0][0] if common else FALLBACK)
