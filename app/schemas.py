from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

NAME_PATTERN = r"^[a-z0-9][a-z0-9\-]{2,40}$"


class Hyperparameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    epochs: int = Field(3, ge=1, le=10)
    learning_rate: float = Field(2e-4, ge=1e-6, le=1e-2)
    batch_size: int = Field(4, ge=1, le=32)
    max_seq_len: int = Field(256, ge=64, le=1024)
    lora_r: int = Field(8, ge=1, le=64)
    lora_alpha: int = Field(16, ge=1, le=128)
    seed: int = Field(42, ge=0, le=2**31 - 1)


class Gate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_score: float = Field(0.5, ge=0.0, le=1.0)
    min_improvement: float = Field(0.05, ge=0.0, le=1.0)
    max_p95_latency_ms: float = Field(5000.0, gt=0, le=600000)



def _default_hyperparameters() -> Hyperparameters:
    return Hyperparameters(epochs=3, learning_rate=2e-4, batch_size=4, max_seq_len=256,
                            lora_r=8, lora_alpha=16, seed=42)


def _default_gate() -> Gate:
    return Gate(min_score=0.5, min_improvement=0.05, max_p95_latency_ms=5000.0)


class TrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str = Field(min_length=1, max_length=64)
    model_name: str = Field(pattern=NAME_PATTERN)
    backend: str = "simple"
    base_model: str | None = Field(default=None, max_length=200)
    task_type: Literal["generation", "classification"] = "generation"
    hyperparameters: Hyperparameters = Field(default_factory=_default_hyperparameters)
    gate: Gate = Field(default_factory=_default_gate)


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str = Field(pattern=NAME_PATTERN)
    version: int | None = Field(default=None, ge=1)
    prompt: str = Field(min_length=1, max_length=12000)
    max_new_tokens: int = Field(150, ge=1, le=512)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    use_cache: bool = True


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["approve", "reject"]
    note: str = Field("", max_length=500)


class StageUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["VALIDATING", "TRAINING", "EVALUATING", "REGISTERING"]


class FailUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    error: str = Field(max_length=500)


class Metrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int = Field(ge=1)
    exact_match: float = Field(ge=0, le=1)
    token_f1: float = Field(ge=0, le=1)
    primary: float = Field(ge=0, le=1)
    primary_metric: str
    empty_rate: float = Field(ge=0, le=1)
    errors: int = Field(ge=0)
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)


class CompleteUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_key: str = Field(max_length=400)
    candidate: Metrics
    baseline: Metrics
