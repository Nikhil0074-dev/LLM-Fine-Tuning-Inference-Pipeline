import io
import json
import os
import sys
import tarfile

import boto3
import fakeredis
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

from app.cache import RedisCache
from app.container import build_context
from app.evaluation import evaluate
from app.main import create_app
from app.ml.base import get_backend
from app.orchestrator import StepFunctionsOrchestrator
from app.storage import LocalStorage, S3Storage, check_key
from conftest import ADMIN, MEMBER, make_settings, upload, train, jsonl_bytes

INTERNAL = {"X-Internal-Key": "internal-secret-key-1"}
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lambda_functions"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------- storage ----------
def test_unsafe_keys_rejected(tmp_path):
    for bad in ("../x", "/etc/passwd", "a/../b", "a//b", "a b", ""):
        with pytest.raises(ValueError):
            check_key(bad)
    with pytest.raises(ValueError):
        LocalStorage(str(tmp_path)).put_bytes("../escape", b"x")


def test_local_storage_uri_is_windows_safe(tmp_path):
    uri = LocalStorage(str(tmp_path)).uri("acme/jobs/job_1/metrics.json")
    assert uri.startswith("file:///")
    assert "\\\\" not in uri
    assert uri.endswith("acme/jobs/job_1/metrics.json")


@mock_aws
def test_s3_storage_roundtrip():
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="bucket-one")
    s = S3Storage("bucket-one", "pre", client=client)
    assert s.ping() and not s.exists("a/b.txt")
    s.put_bytes("a/b.txt", b"hello")
    assert s.exists("a/b.txt") and s.get_bytes("a/b.txt") == b"hello"
    assert s.uri("a/b.txt") == "s3://bucket-one/pre/a/b.txt"
    assert client.head_object(Bucket="bucket-one", Key="pre/a/b.txt")["ServerSideEncryption"] == "AES256"
    assert not S3Storage("missing-bucket", "", client=client).ping()


@mock_aws
def test_full_pipeline_on_s3(tmp_path):
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="bucket-two")
    ctx = build_context(make_settings(tmp_path), storage=S3Storage("bucket-two", "p", client=client))
    from conftest import wait_job
    with TestClient(create_app(make_settings(tmp_path), ctx=ctx)) as c:
        ds = upload(c).json()
        job = wait_job(c, train(c, ds["id"]).json()["id"])
        assert job["status"] == "SUCCEEDED"
        keys = [o["Key"] for o in client.list_objects_v2(Bucket="bucket-two")["Contents"]]
        assert any(k.endswith("artifact.bin") for k in keys) and any(k.endswith("train.jsonl") for k in keys)


# ---------- redis ----------
def test_redis_cache_and_counter():
    cache = RedisCache(client=fakeredis.FakeRedis(decode_responses=True))
    assert cache.get("k") is None
    cache.set("k", "v", 60)
    assert cache.get("k") == "v"
    assert [cache.incr("c", 60) for _ in range(3)] == [1, 2, 3]
    assert cache.ping()


def test_redis_down_does_not_break_inference(tmp_path):
    class Broken:
        def get(self, *a, **k): raise ConnectionError("down")
        def set(self, *a, **k): raise ConnectionError("down")
        def pipeline(self): raise ConnectionError("down")
        def ping(self): raise ConnectionError("down")

    ctx = build_context(make_settings(tmp_path), cache=RedisCache(client=Broken()))
    from conftest import full_pipeline
    with TestClient(create_app(make_settings(tmp_path), ctx=ctx)) as c:
        ds, job = full_pipeline(c)
        c.post(f"/api/v1/models/{job['model_version_id']}/deploy", headers=ADMIN)
        r = c.post("/api/v1/inference", json={"model_name": "support-bot", "prompt": "question about topic1"},
                   headers=MEMBER)
        assert r.status_code == 200 and r.json()["cached"] is False
        assert c.get("/api/v1/health").json()["checks"]["cache"] is False


# ---------- step functions orchestration ----------
class FakeSFN:
    def __init__(self): self.started, self.stopped = [], []
    def start_execution(self, **kw): self.started.append(kw); return {"executionArn": "arn:exec:" + kw["name"]}
    def stop_execution(self, **kw): self.stopped.append(kw)


@pytest.fixture
def sfn_client(tmp_path):
    s = make_settings(tmp_path, orchestrator="stepfunctions", state_machine_arn="arn:sm", allowed_backends=("simple", "lora"),
                      s3_bucket="bucket-k")
    fake = FakeSFN()
    ctx = build_context(s, orchestrator=StepFunctionsOrchestrator(s, client=fake))
    with TestClient(create_app(s, ctx=ctx)) as c:
        c.fake, c.ctx = fake, ctx
        yield c


def test_start_payload_and_cancel(sfn_client):
    c = sfn_client
    ds = upload(c).json()
    job = train(c, ds["id"], backend="lora", key="sfn-key-0001").json()
    assert job["status"] == "QUEUED"
    p = json.loads(c.fake.started[0]["input"])
    assert p["job_id"] == job["id"] and p["hyperparameters"]["epochs"] == "3"
    assert all(isinstance(v, str) for v in p["hyperparameters"].values())
    assert p["train_prefix_uri"].endswith("train.jsonl") and p["output_uri"].endswith("/")
    assert p["metrics_s3_uri"].endswith(f"{job['id']}/metrics.json")
    r = c.post(f"/api/v1/training-jobs/{job['id']}/cancel", headers=ADMIN)
    assert r.json()["status"] == "CANCELLED"  # QUEUED jobs cancel immediately


def test_cancel_running_stops_execution(sfn_client):
    c = sfn_client
    ds = upload(c).json()
    job = train(c, ds["id"], backend="lora", key="sfn-key-0002").json()
    c.post(f"/internal/v1/jobs/{job['id']}/stage", json={"status": "TRAINING"}, headers=INTERNAL)
    r = c.post(f"/api/v1/training-jobs/{job['id']}/cancel", headers=ADMIN).json()
    assert r["status"] == "CANCELLED" and len(c.fake.stopped) == 1


def test_orchestrator_failure_marks_job_failed(tmp_path):
    class Boom(FakeSFN):
        def start_execution(self, **kw): raise RuntimeError("aws down")

    s = make_settings(tmp_path, orchestrator="stepfunctions", state_machine_arn="arn:sm", s3_bucket="bucket-k")
    ctx = build_context(s, orchestrator=StepFunctionsOrchestrator(s, client=Boom()))
    with TestClient(create_app(s, ctx=ctx)) as c:
        ds = upload(c).json()
        r = train(c, ds["id"], key="boom-key-0001")
        assert r.status_code == 502
        jobs = c.get("/api/v1/training-jobs", headers=ADMIN).json()["items"]
        assert jobs[0]["status"] == "FAILED"


def _metrics_for(ctx, rows):
    b = get_backend("simple")
    art = b.train(rows, {}, lambda: False)
    return art, evaluate(b.load(art), rows[:30], "classification"), evaluate(b.baseline(rows, {}), rows[:30], "classification")


def test_internal_callbacks_register_model(sfn_client):
    c = sfn_client
    ds = upload(c).json()
    job = train(c, ds["id"], key="sfn-key-0003").json()
    assert c.post(f"/internal/v1/jobs/{job['id']}/stage", json={"status": "TRAINING"}).status_code == 403
    assert c.post(f"/internal/v1/jobs/{job['id']}/stage", json={"status": "TRAINING"}, headers=ADMIN).status_code == 403
    assert c.post(f"/internal/v1/jobs/{job['id']}/stage", json={"status": "TRAINING"}, headers=INTERNAL).status_code == 200
    rows = [json.loads(l) for l in jsonl_bytes(100).decode().splitlines()]
    art, cand, base = _metrics_for(c.ctx, rows)
    key = f"acme/jobs/{job['id']}/artifact.bin"
    c.ctx.storage.put_bytes(key, art)
    bad = c.post(f"/internal/v1/jobs/{job['id']}/complete", headers=INTERNAL,
                 json={"artifact_key": "acme/jobs/other/artifact.bin", "candidate": cand, "baseline": base})
    assert bad.status_code == 422
    ok = c.post(f"/internal/v1/jobs/{job['id']}/complete", headers=INTERNAL,
                json={"artifact_key": key, "candidate": cand, "baseline": base})
    assert ok.status_code == 200 and ok.json()["approval_status"] == "approved"
    assert c.get(f"/api/v1/training-jobs/{job['id']}", headers=ADMIN).json()["status"] == "SUCCEEDED"
    again = c.post(f"/internal/v1/jobs/{job['id']}/complete", headers=INTERNAL,
                   json={"artifact_key": key, "candidate": cand, "baseline": base})
    assert again.status_code == 409  # replay cannot create a second model
    junk = c.post(f"/internal/v1/jobs/{job['id']}/complete", headers=INTERNAL,
                  json={"artifact_key": key, "candidate": {"n": 1}, "baseline": base})
    assert junk.status_code == 422


def test_internal_endpoints_disabled_without_key(tmp_path):
    with TestClient(create_app(make_settings(tmp_path, internal_api_key=""))) as c:
        r = c.post("/internal/v1/jobs/x/stage", json={"status": "TRAINING"}, headers={"X-Internal-Key": ""})
        assert r.status_code == 403


# ---------- lambda ----------
@mock_aws
def test_lambda_complete_translates_s3_uris(monkeypatch):
    import callback
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="bucket-k")
    metrics = {"candidate": {"n": 1}, "baseline": {"n": 1}}
    s3.put_object(Bucket="bucket-k", Key="pre/acme/jobs/job_1/metrics.json", Body=json.dumps(metrics).encode())
    monkeypatch.setenv("S3_BUCKET", "bucket-k")
    monkeypatch.setenv("S3_PREFIX", "pre")
    monkeypatch.setenv("API_URL", "http://api")
    monkeypatch.setenv("INTERNAL_API_KEY", "k")
    sent = {}
    monkeypatch.setattr(callback, "_post", lambda path, body: sent.update(path=path, body=body) or {})
    callback.handler({"job_id": "job_1", "action": "complete", "metrics_s3_uri": "s3://bucket-k/pre/acme/jobs/job_1/metrics.json",
                      "model_artifact_s3_uri": "s3://bucket-k/pre/acme/jobs/job_1/output/llmp-job-1/output/model.tar.gz"})
    assert sent["path"] == "/internal/v1/jobs/job_1/complete"
    assert sent["body"]["artifact_key"] == "acme/jobs/job_1/output/llmp-job-1/output/model.tar.gz"
    with pytest.raises(ValueError):
        callback.handler({"job_id": "j", "action": "complete", "metrics_s3_uri": "s3://bucket-k/pre/m.json",
                          "model_artifact_s3_uri": "s3://evil-bucket/x"})
    callback.handler({"job_id": "job_1", "action": "fail", "error": "x" * 900})
    assert len(sent["body"]["error"]) == 500


# ---------- sagemaker entry ----------
@mock_aws
def test_sagemaker_entry_with_stand_ins(tmp_path):
    from training import sagemaker_entry as entry
    base = tmp_path / "ml"
    for d in ("input/config", "input/data/train", "input/data/validation", "model"):
        (base / d).mkdir(parents=True)
    rows = [json.loads(l) for l in jsonl_bytes(60).decode().splitlines()]
    (base / "input/data/train/train.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:40]))
    (base / "input/data/validation/val.jsonl").write_text("\n".join(json.dumps(r) for r in rows[40:]))
    (base / "input/config/hyperparameters.json").write_text(json.dumps({
        "epochs": "2", "learning_rate": "0.0002", "base_model": '"x/y"', "task_type": "classification",
        "metrics_s3_uri": "s3://bucket-k/m.json"}))
    simple = get_backend("simple")
    state = {}

    def fake_train(train_rows, base_model, hp, out_dir):
        state["hp"] = hp
        state["art"] = simple.train(train_rows, {}, lambda: False)

    uploads = {}
    m = entry.run(str(base), train_fn=fake_train,
                  predictor_fn=lambda bm, d: simple.load(state["art"]) if d else simple.baseline(rows[:40], {}),
                  upload_fn=lambda uri, payload: uploads.update({uri: payload}))
    assert state["hp"]["epochs"] == 2 and state["hp"]["learning_rate"] == 0.0002 and isinstance(state["hp"]["epochs"], int)
    assert uploads["s3://bucket-k/m.json"] == m and m["candidate"]["n"] == 20
    from app.schemas import Metrics
    Metrics(**m["candidate"]); Metrics(**m["baseline"])  # the API will accept what the container writes


def test_tar_artifact_extraction_is_safe(tmp_path):
    from app.ml.lora import extract_artifact
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = b"x"
        info = tarfile.TarInfo("../evil.txt"); info.size = 1
        tf.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError):
        extract_artifact(buf.getvalue(), str(tmp_path / "out"))
    buf2 = io.BytesIO()
    with tarfile.open(fileobj=buf2, mode="w:gz") as tf:
        info = tarfile.TarInfo("adapter.bin"); info.size = 3
        tf.addfile(info, io.BytesIO(b"abc"))
    extract_artifact(buf2.getvalue(), str(tmp_path / "ok"))
    assert (tmp_path / "ok" / "adapter.bin").read_bytes() == b"abc"


# ---------- state machine definition ----------
def test_asl_is_consistent():
    path = os.path.join(os.path.dirname(__file__), "..", "workflows", "training_state_machine.asl.json")
    sm = json.load(open(path))
    states = sm["States"]
    assert sm["StartAt"] in states
    for name, st in states.items():
        targets = [st.get("Next")] + [c["Next"] for c in st.get("Catch", [])]
        for t in filter(None, targets):
            assert t in states, f"{name} -> {t}"
        assert st.get("End") or st.get("Next") or st["Type"] in ("Fail", "Succeed")
    assert states["Failed"]["Type"] == "Fail"
