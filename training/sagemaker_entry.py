"""Container entry point for SageMaker. It trains, then evaluates the candidate AND the base model
on the same validation rows, and uploads metrics.json. The API makes the approve/reject decision.

SageMaker layout:  /opt/ml/input/data/{train,validation}/*.jsonl
                   /opt/ml/input/config/hyperparameters.json   (all values are strings)
                   /opt/ml/model/   -> packed by SageMaker into model.tar.gz
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.evaluation import evaluate  # noqa: E402
from app.ml.lora import DEFAULT_HP  # noqa: E402

BASE = os.environ.get("SM_BASE", "/opt/ml")


def parse_hyperparameters(raw: dict) -> tuple[dict, dict]:
    """Return (numeric training hyperparameters, run settings)."""
    def clean(v):
        v = str(v)
        return json.loads(v) if len(v) >= 2 and v[0] == v[-1] == '"' else v

    raw = {k: clean(v) for k, v in raw.items()}
    hp = {k: type(default)(raw[k]) for k, default in DEFAULT_HP.items() if k in raw}
    run = {k: raw[k] for k in ("base_model", "task_type", "metrics_s3_uri") if k in raw}
    missing = [k for k in ("base_model", "task_type", "metrics_s3_uri") if k not in run]
    if missing:
        raise ValueError(f"missing hyperparameters: {missing}")
    return hp, run


def read_rows(channel_dir: str) -> list[dict]:
    files = sorted(f for f in glob.glob(os.path.join(channel_dir, "*")) if os.path.isfile(f))
    if not files:
        raise FileNotFoundError(f"no files in {channel_dir}")
    with open(files[0], encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def upload_json(s3_uri: str, payload: dict) -> None:
    import boto3

    bucket, _, key = s3_uri.removeprefix("s3://").partition("/")
    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=json.dumps(payload).encode(),
                                  ServerSideEncryption="AES256")


def run(base: str = BASE, train_fn=None, predictor_fn=None, upload_fn=upload_json) -> dict:
    if train_fn is None or predictor_fn is None:
        from app.ml.lora import HFPredictor, train_lora

        train_fn = train_fn or train_lora
        predictor_fn = predictor_fn or HFPredictor
    with open(os.path.join(base, "input/config/hyperparameters.json")) as fh:
        hp, cfg = parse_hyperparameters(json.load(fh))
    train_rows = read_rows(os.path.join(base, "input/data/train"))
    val_rows = read_rows(os.path.join(base, "input/data/validation"))
    model_dir = os.path.join(base, "model")
    train_fn(train_rows, cfg["base_model"], hp, model_dir)
    metrics = {
        "candidate": evaluate(predictor_fn(cfg["base_model"], model_dir), val_rows, cfg["task_type"]),
        "baseline": evaluate(predictor_fn(cfg["base_model"], None), val_rows, cfg["task_type"]),
    }
    upload_fn(cfg["metrics_s3_uri"], metrics)
    return metrics


if __name__ == "__main__":
    run()
