import json
import os
import sys
import time

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.cache import MemoryCache  # noqa: E402
from app.config import Settings  # noqa: E402
from app.container import build_context  # noqa: E402
from app.main import create_app  # noqa: E402

ADMIN = {"X-API-Key": "admin-key-aaaaaaaaaaaa"}
MEMBER = {"X-API-Key": "member-key-bbbbbbbbbbb"}
OTHER_ADMIN = {"X-API-Key": "other-admin-key-cccccc"}
KEYS = ("admin-key-aaaaaaaaaaaa:admin:acme:alice;member-key-bbbbbbbbbbb:member:acme:bob;"
        "other-admin-key-cccccc:admin:globex:carol")


def make_settings(tmp_path, **over):
    base = dict(data_dir=str(tmp_path), api_keys=KEYS, internal_api_key="internal-secret-key-1",
                min_dataset_rows=20, rate_limit_per_minute=1000, max_concurrent_jobs=2,
                max_active_jobs_per_tenant=5, inference_timeout_seconds=5.0)
    base.update(over)
    return Settings(**base)


@pytest.fixture
def client(tmp_path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as c:
        c.app_ctx = app.state.ctx
        yield c


def jsonl_bytes(n=100, labels=3):
    rows = [{"prompt": f"what is the answer to question number {i} about topic{i % labels}",
             "response": f"label{i % labels}"} for i in range(n)]
    return "\n".join(json.dumps(r) for r in rows).encode()


def wait_job(client, job_id, headers=ADMIN, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/v1/training-jobs/{job_id}", headers=headers).json()
        if job["status"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def upload(client, data=None, name="support-data", fmt="jsonl", headers=ADMIN):
    return client.post(f"/api/v1/datasets?name={name}&format={fmt}", content=data or jsonl_bytes(), headers=headers)


def train(client, dataset_id, key="idem-key-0001", headers=ADMIN, **over):
    body = {"dataset_id": dataset_id, "model_name": "support-bot", "backend": "simple",
            "task_type": "classification"}
    body.update(over)
    return client.post("/api/v1/training-jobs", json=body, headers={**headers, "Idempotency-Key": key})


def full_pipeline(client, name="support-bot", key="idem-key-0001", data=None):
    ds = upload(client, data).json()
    job = train(client, ds["id"], key=key, model_name=name).json()
    job = wait_job(client, job["id"])
    assert job["status"] == "SUCCEEDED", job
    return ds, job
