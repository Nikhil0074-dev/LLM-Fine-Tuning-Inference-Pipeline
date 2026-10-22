"""Settings. Every value can be overridden with an environment variable named LLMP_<FIELD>."""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import cast


@dataclass(frozen=True)
class Settings:
    # --- storage / runtime ---
    data_dir: str = "./data"
    storage_backend: str = "local"        # local | s3
    s3_bucket: str = ""
    s3_prefix: str = "llm-pipeline"
    aws_region: str = "us-east-1"
    redis_url: str = ""                   # empty -> in-memory cache
    orchestrator: str = "local"           # local | stepfunctions
    state_machine_arn: str = ""
    sagemaker_instance_type: str = "ml.g5.xlarge"
    training_image: str = ""
    callback_url: str = ""                # not used by API itself; kept for deployment docs

    # --- auth ---
    # format: "key:role:tenant:name;key2:role:tenant:name"   role = admin | member
    api_keys: str = ""
    internal_api_key: str = ""            # used only by Lambda callbacks

    # --- allow-lists and limits ---
    allowed_backends: tuple = ("simple", "lora")
    allowed_base_models: tuple = ("HuggingFaceTB/SmolLM2-135M-Instruct",)
    max_upload_bytes: int = 5 * 1024 * 1024
    max_field_chars: int = 8000
    min_dataset_rows: int = 20
    val_ratio: float = 0.2
    block_on_pii: bool = True
    max_malformed_ratio: float = 0.2

    # --- quality gate ---
    min_val_for_auto_approve: int = 20

    # --- cost control ---
    max_concurrent_jobs: int = 1
    max_active_jobs_per_tenant: int = 2
    monthly_budget_usd: float = 50.0
    gpu_hourly_usd: float = 1.41          # CHECK current AWS price for your instance and region
    assumed_tokens_per_sec: float = 1500.0
    training_startup_seconds: float = 180.0

    # --- inference ---
    rate_limit_per_minute: int = 60
    cache_ttl_seconds: int = 3600
    cache_max_temperature: float = 0.3
    inference_timeout_seconds: float = 30.0
    inference_workers: int = 4
    max_prompt_chars: int = 12000
    max_loaded_models: int = 3
    system_prompt_version: str = "v1"

    # --- misc ---
    code_version: str = "dev"
    docs_enabled: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        values: dict[str, object] = {}
        for f in fields(cls):
            raw = os.getenv("LLMP_" + f.name.upper())
            if raw is None:
                continue
            default = f.default
            if isinstance(default, bool):
                values[f.name] = raw.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(default, int):
                values[f.name] = int(raw)
            elif isinstance(default, float):
                values[f.name] = float(raw)
            elif isinstance(default, tuple):
                values[f.name] = tuple(x.strip() for x in raw.split(",") if x.strip())
            else:
                values[f.name] = raw
        return cls(**cast(dict, values))
