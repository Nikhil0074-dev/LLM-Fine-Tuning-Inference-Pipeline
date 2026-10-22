from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Protocol


class TrainingCancelled(Exception):
    pass


@dataclass
class GenResult:
    text: str
    input_tokens: int
    output_tokens: int


class Predictor(Protocol):
    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> GenResult: ...


class Backend(ABC):
    name: str = ""
    uses_gpu: bool = False

    @abstractmethod
    def train(self, train_rows: list[dict], config: dict, should_cancel: Callable[[], bool]) -> bytes:
        """Train and return the model artifact as bytes."""

    @abstractmethod
    def load(self, artifact: bytes) -> Predictor:
        """Load a model from the bytes returned by train()."""

    @abstractmethod
    def baseline(self, train_rows: list[dict], config: dict) -> Predictor:
        """The model *before* fine-tuning. The candidate must beat it to be approved."""


def get_backend(name: str) -> Backend:
    if name == "simple":
        from .simple import SimpleBackend

        return SimpleBackend()
    if name == "lora":
        from .lora import LoraBackend

        return LoraBackend()
    raise ValueError(f"unknown backend {name!r}")
