import time

from fastapi.testclient import TestClient

from app.main import create_app
from conftest import ADMIN, MEMBER, OTHER_ADMIN, full_pipeline, make_settings, upload, train, wait_job

BODY = {"model_name": "support-bot", "prompt": "answer to question about topic2"}


def deployed(client, **kw):
    ds, job = full_pipeline(client, **kw)
    assert client.post(f"/api/v1/models/{job['model_version_id']}/deploy", headers=ADMIN).status_code == 200
    return job


def test_validation_and_auth(client):
    deployed(client)
    assert client.post("/api/v1/inference", json=BODY).status_code == 401
    assert client.post("/api/v1/inference", json={**BODY, "max_new_tokens": 99999}, headers=MEMBER).status_code == 422
    assert client.post("/api/v1/inference", json={**BODY, "temperature": 9}, headers=MEMBER).status_code == 422
    assert client.post("/api/v1/inference", json={**BODY, "prompt": ""}, headers=MEMBER).status_code == 422
    assert client.post("/api/v1/inference", json={**BODY, "extra": 1}, headers=MEMBER).status_code == 422
    assert client.post("/api/v1/inference", json={**BODY, "prompt": "x" * 13000}, headers=MEMBER).status_code == 422
    # another tenant cannot reach this deployment
    assert client.post("/api/v1/inference", json=BODY, headers=OTHER_ADMIN).status_code == 404


def test_cache_rules(client):
    deployed(client)
    first = client.post("/api/v1/inference", json=BODY, headers=MEMBER).json()
    assert first["cached"] is False
    assert client.post("/api/v1/inference", json={**BODY, "prompt": "  answer to   question about topic2 "},
                       headers=MEMBER).json()["cached"] is True           # whitespace is normalised
    assert client.post("/api/v1/inference", json={**BODY, "max_new_tokens": 5}, headers=MEMBER).json()["cached"] is False
    assert client.post("/api/v1/inference", json={**BODY, "use_cache": False}, headers=MEMBER).json()["cached"] is False
    hot = {**BODY, "temperature": 1.0}
    client.post("/api/v1/inference", json=hot, headers=MEMBER)
    assert client.post("/api/v1/inference", json=hot, headers=MEMBER).json()["cached"] is False  # too random to cache


def test_cache_is_per_tenant_and_per_model_version(client):
    job = deployed(client)
    client.post("/api/v1/inference", json=BODY, headers=MEMBER)
    # a second tenant with an identical model must not see acme's cached answer
    ds = upload(client, headers=OTHER_ADMIN).json()
    j = wait_job(client, train(client, ds["id"], key="globex-key-1", headers=OTHER_ADMIN).json()["id"], OTHER_ADMIN)
    client.post(f"/api/v1/models/{j['model_version_id']}/deploy", headers=OTHER_ADMIN)
    assert client.post("/api/v1/inference", json=BODY, headers=OTHER_ADMIN).json()["cached"] is False
    # new model version -> old cache entries are unreachable (no invalidation needed)
    ds2 = upload(client).json()
    j2 = wait_job(client, train(client, ds2["id"], key="acme-key-v2xx").json()["id"])
    client.post(f"/api/v1/models/{j2['model_version_id']}/deploy", headers=ADMIN)
    assert client.post("/api/v1/inference", json=BODY, headers=MEMBER).json()["cached"] is False


def test_cold_start_after_eviction(client):
    job = deployed(client)
    client.app_ctx.pool.evict(job["model_version_id"])
    r = client.post("/api/v1/inference", json={**BODY, "use_cache": False}, headers=MEMBER).json()
    assert r["cold_start"] is True
    assert client.post("/api/v1/inference", json={**BODY, "use_cache": False}, headers=MEMBER).json()["cold_start"] is False


def test_rate_limit(tmp_path):
    with TestClient(create_app(make_settings(tmp_path, rate_limit_per_minute=3))) as c:
        deployed(c)
        codes = [c.post("/api/v1/inference", json=BODY, headers=MEMBER).status_code for _ in range(5)]
        assert codes == [200, 200, 200, 429, 429]
        r = c.post("/api/v1/inference", json=BODY, headers=MEMBER)
        assert "retry-after" in r.headers


def test_timeout_returns_504(tmp_path):
    with TestClient(create_app(make_settings(tmp_path, inference_timeout_seconds=0.2))) as c:
        job = deployed(c)
        pred, _ = c.app.state.ctx.pool.get({"id": job["model_version_id"], "backend": "simple", "artifact_key": ""})
        original = pred.generate
        pred.generate = lambda *a, **k: (time.sleep(1.0), original(*a, **k))[1]
        r = c.post("/api/v1/inference", json={**BODY, "use_cache": False}, headers=MEMBER)
        assert r.status_code == 504 and r.json()["error"]["code"] == "timeout"


def test_logs_never_contain_prompts(client):
    deployed(client)
    client.post("/api/v1/inference", json={**BODY, "prompt": "secret-marker-12345 topic1"}, headers=MEMBER)
    import sqlite3
    with sqlite3.connect(client.app_ctx.db.path) as conn:
        dump = "\n".join(conn.iterdump())
    assert "secret-marker-12345" not in dump


def test_health(client):
    r = client.get("/api/v1/health").json()
    assert r["status"] == "healthy" and r["checks"]["storage"] is True
