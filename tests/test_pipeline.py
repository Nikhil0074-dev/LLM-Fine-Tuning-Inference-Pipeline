import json
import time

from conftest import ADMIN, MEMBER, OTHER_ADMIN, full_pipeline, jsonl_bytes, train, upload, wait_job, make_settings


def test_full_flow_train_evaluate_deploy_infer(client):
    ds, job = full_pipeline(client)
    assert job["progress"] == 1.0 and job["model_version_id"]
    model = client.get(f"/api/v1/models/{job['model_version_id']}", headers=ADMIN).json()
    assert model["approval_status"] == "approved", model["decision_reasons"]
    assert model["metrics"]["candidate"]["primary"] > model["metrics"]["baseline"]["primary"]
    assert model["dataset_sha256"] == ds["sha256"] and model["version"] == 1

    # not served before deploy
    body = {"model_name": "support-bot", "prompt": "what is the answer to question number 3 about topic0"}
    assert client.post("/api/v1/inference", json=body, headers=MEMBER).status_code == 404

    dep = client.post(f"/api/v1/models/{model['id']}/deploy", headers=ADMIN)
    assert dep.status_code == 200 and dep.json()["status"] == "active"
    r1 = client.post("/api/v1/inference", json=body, headers=MEMBER).json()
    assert r1["output"] == "label0" and r1["cached"] is False and r1["cold_start"] is False
    assert r1["usage"]["input_tokens"] > 0
    r2 = client.post("/api/v1/inference", json=body, headers=MEMBER).json()
    assert r2["cached"] is True and r2["output"] == "label0"

    m = client.get("/api/v1/metrics", headers=ADMIN).json()
    assert m["inference"]["requests"] >= 2 and m["inference"]["cache_hit_rate"] > 0
    assert client.get("/api/v1/metrics", headers=MEMBER).status_code == 403


def test_idempotency(client):
    ds = upload(client).json()
    a = train(client, ds["id"], key="same-key-12345")
    b = train(client, ds["id"], key="same-key-12345")
    assert a.status_code == 201 and b.status_code == 200 and a.json()["id"] == b.json()["id"]
    c = train(client, ds["id"], key="same-key-12345", model_name="another-name")
    assert c.status_code == 422 and c.json()["error"]["code"] == "idempotency_key_reuse"
    assert client.post("/api/v1/training-jobs", json={"dataset_id": ds["id"], "model_name": "abc-model"},
                       headers=ADMIN).status_code == 400
    assert len(client.get("/api/v1/training-jobs", headers=ADMIN).json()["items"]) == 1


def test_job_input_validation(client):
    ds = upload(client).json()
    assert train(client, ds["id"], key="k-aaaaaaaa", backend="gpt-evil").status_code == 422
    assert train(client, ds["id"], key="k-bbbbbbbb", hyperparameters={"epochs": 999}).status_code == 422
    assert train(client, ds["id"], key="k-cccccccc", hyperparameters={"run": "rm -rf /"}).status_code == 422
    assert train(client, ds["id"], key="k-dddddddd", backend="lora", base_model="evil/model").status_code == 422
    assert train(client, "ds_missing", key="k-eeeeeeee").status_code == 404
    assert train(client, ds["id"], key="k-ffffffff", headers=MEMBER).status_code == 403
    assert train(client, ds["id"], key="k-gggggggg", headers=OTHER_ADMIN).status_code == 404
    bad = upload(client, jsonl_bytes(3), name="tiny-data").json()
    assert train(client, bad["id"], key="k-hhhhhhhh").json()["error"]["code"] == "dataset_invalid"


def test_gate_rejects_weak_model_and_blocks_deploy(client):
    ds = upload(client).json()
    job = train(client, ds["id"], key="gate-key-001", gate={"min_score": 0.99, "min_improvement": 0.9}).json()
    job = wait_job(client, job["id"])
    assert job["status"] == "SUCCEEDED"  # training finished...
    mv = client.get(f"/api/v1/models/{job['model_version_id']}", headers=ADMIN).json()
    assert mv["approval_status"] == "rejected" and mv["decision_reasons"]  # ...but not approved
    assert client.post(f"/api/v1/models/{mv['id']}/deploy", headers=ADMIN).status_code == 409
    assert client.post(f"/api/v1/models/{mv['id']}/review", json={"decision": "approve"},
                       headers=ADMIN).status_code == 409  # gate rejections cannot be overridden


def test_small_validation_set_needs_review_then_manual_approval(client):
    data = jsonl_bytes(40)  # 8 validation rows < 20
    ds, job = full_pipeline(client, data=data)
    mv = client.get(f"/api/v1/models/{job['model_version_id']}", headers=ADMIN).json()
    assert mv["approval_status"] == "needs_review"
    assert client.post(f"/api/v1/models/{mv['id']}/deploy", headers=ADMIN).status_code == 409
    r = client.post(f"/api/v1/models/{mv['id']}/review", json={"decision": "approve", "note": "ok"}, headers=ADMIN)
    assert r.status_code == 200 and r.json()["approval_status"] == "approved" and r.json()["approved_by"] == "alice"
    assert client.post(f"/api/v1/models/{mv['id']}/deploy", headers=ADMIN).status_code == 200


def test_versions_deploy_and_rollback(client):
    ds = upload(client).json()
    ids = []
    for i in range(2):
        job = wait_job(client, train(client, ds["id"], key=f"ver-key-000{i}").json()["id"])
        ids.append(job["model_version_id"])
    versions = [client.get(f"/api/v1/models/{i}", headers=ADMIN).json()["version"] for i in ids]
    assert versions == [1, 2]
    assert client.post(f"/api/v1/deployments/support-bot/rollback", headers=ADMIN).status_code == 404
    client.post(f"/api/v1/models/{ids[0]}/deploy", headers=ADMIN)
    client.post(f"/api/v1/models/{ids[1]}/deploy", headers=ADMIN)
    body = {"model_name": "support-bot", "prompt": "answer to question about topic1"}
    assert client.post("/api/v1/inference", json=body, headers=MEMBER).json()["model_version"] == 2
    rb = client.post("/api/v1/deployments/support-bot/rollback", headers=ADMIN)
    assert rb.status_code == 200 and rb.json()["version"] == 1
    assert client.post("/api/v1/inference", json=body, headers=MEMBER).json()["model_version"] == 1
    pinned = client.post("/api/v1/inference", json={**body, "version": 2}, headers=MEMBER)
    assert pinned.status_code == 200 and pinned.json()["model_version"] == 2
    history = client.get("/api/v1/deployments?model_name=support-bot", headers=ADMIN).json()["items"]
    assert [h["status"] for h in history] == ["rolled_back", "active"]  # newest deployment first


def test_smoke_test_blocks_broken_deploy(client):
    ds, job = full_pipeline(client)
    mid = job["model_version_id"]
    client.post(f"/api/v1/models/{mid}/deploy", headers=ADMIN)
    # corrupt the stored artifact of a second approved version
    job2 = wait_job(client, train(client, ds["id"], key="broken-key-1").json()["id"])
    mv2 = client.app_ctx.db.one("SELECT * FROM model_versions WHERE id=?", (job2["model_version_id"],))
    client.app_ctx.storage.put_bytes(mv2["artifact_key"], b"not a model")
    r = client.post(f"/api/v1/models/{mv2['id']}/deploy", headers=ADMIN)
    assert r.status_code == 422 and r.json()["error"]["code"] == "smoke_test_failed"
    body = {"model_name": "support-bot", "prompt": "answer to question about topic1"}
    assert client.post("/api/v1/inference", json=body, headers=MEMBER).json()["model_version"] == 1  # unchanged


def test_cancel_running_and_finished(tmp_path):
    from fastapi.testclient import TestClient
    from app.main import create_app
    from app.ml import base, simple

    class Slow(simple.SimpleBackend):
        def train(self, rows, config, should_cancel):
            for _ in range(200):
                if should_cancel():
                    raise base.TrainingCancelled()
                time.sleep(0.02)
            return super().train(rows, config, should_cancel)

    orig = base.get_backend
    import app.jobs as jobs_mod
    jobs_mod.get_backend = lambda name: Slow() if name == "simple" else orig(name)
    try:
        with TestClient(create_app(make_settings(tmp_path))) as c:
            ds = upload(c).json()
            job = train(c, ds["id"], key="cancel-key-1").json()
            for _ in range(100):
                if c.get(f"/api/v1/training-jobs/{job['id']}", headers=ADMIN).json()["status"] == "TRAINING":
                    break
                time.sleep(0.02)
            assert c.post(f"/api/v1/training-jobs/{job['id']}/cancel", headers=ADMIN).status_code == 200
            assert wait_job(c, job["id"])["status"] == "CANCELLED"
            assert c.post(f"/api/v1/training-jobs/{job['id']}/cancel", headers=ADMIN).status_code == 409
    finally:
        jobs_mod.get_backend = orig


def test_budget_guard_and_active_job_cap(tmp_path):
    from fastapi.testclient import TestClient
    from app.main import create_app
    s = make_settings(tmp_path, allowed_backends=("simple", "lora"), monthly_budget_usd=0.001, max_active_jobs_per_tenant=1)
    with TestClient(create_app(s)) as c:
        ds = upload(c).json()
        r = train(c, ds["id"], key="budget-key-1", backend="lora")
        assert r.status_code == 409 and r.json()["error"]["code"] == "budget_exceeded"
        assert c.get("/api/v1/training-jobs", headers=ADMIN).json()["items"] == []


def test_audit_trail(client):
    ds, job = full_pipeline(client)
    client.post(f"/api/v1/models/{job['model_version_id']}/deploy", headers=ADMIN)
    actions = [r["action"] for r in client.app_ctx.db.query("SELECT action FROM audit_log ORDER BY id")]
    assert actions == ["dataset.create", "job.create", "model.register", "deployment.activate"]
