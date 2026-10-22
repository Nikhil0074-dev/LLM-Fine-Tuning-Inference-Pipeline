from __future__ import annotations

import hashlib
import json
import logging
import sqlite3

from . import audit
from .cost import estimate_training_cost, month_spend
from .errors import ApiError
from .evaluation import decide, evaluate
from .ml.base import TrainingCancelled, get_backend
from .util import new_id, now_iso

log = logging.getLogger("llmp.jobs")
TERMINAL = ("SUCCEEDED", "FAILED", "CANCELLED")
PROGRESS = {"QUEUED": 0.0, "VALIDATING": 0.05, "TRAINING": 0.2, "EVALUATING": 0.7, "REGISTERING": 0.9,
            "SUCCEEDED": 1.0, "FAILED": 1.0, "CANCELLED": 1.0}


def present_job(row: dict) -> dict:
    return {
        "id": row["id"], "dataset_id": row["dataset_id"], "model_name": row["model_name"],
        "backend": row["backend"], "config": json.loads(row["config_json"]), "status": row["status"],
        "progress": row["progress"], "cancel_requested": bool(row["cancel_requested"]), "error": row["error"],
        "estimated_cost_usd": row["estimated_cost_usd"], "model_version_id": row["model_version_id"],
        "created_by": row["created_by"], "created_at": row["created_at"],
        "started_at": row["started_at"], "completed_at": row["completed_at"],
    }


def get_job_row(db, tenant: str, job_id: str) -> dict:
    row = db.one("SELECT * FROM training_jobs WHERE id=? AND tenant=?", (job_id, tenant))
    if not row:
        raise ApiError(404, "not_found", "Training job not found.")
    return row


def set_status(db, job_id: str, status: str, error: str | None = None) -> bool:
    """Move a job forward. Terminal states are final: returns False if the job was already finished."""
    ts = now_iso()
    n = db.execute(
        "UPDATE training_jobs SET status=?, progress=?, error=COALESCE(?, error),"
        " started_at=COALESCE(started_at, ?), completed_at=CASE WHEN ? THEN ? ELSE completed_at END"
        " WHERE id=? AND status NOT IN ('SUCCEEDED','FAILED','CANCELLED')",
        (status, PROGRESS[status], error, ts, status in TERMINAL, ts, job_id),
    )
    return n == 1


def create_job(ctx, principal, idem_key: str, req) -> tuple[dict, bool]:
    """Returns (job, created). Replaying the same Idempotency-Key + body returns the original job."""
    s = ctx.settings
    request_hash = hashlib.sha256(json.dumps(req.model_dump(), sort_keys=True).encode()).hexdigest()
    existing = ctx.db.one("SELECT * FROM training_jobs WHERE tenant=? AND idem_key=?", (principal.tenant, idem_key))
    if existing:
        if existing["request_hash"] != request_hash:
            raise ApiError(422, "idempotency_key_reuse", "This Idempotency-Key was used with a different request.")
        return present_job(existing), False

    ds = ctx.db.one("SELECT * FROM datasets WHERE id=? AND tenant=?", (req.dataset_id, principal.tenant))
    if not ds:
        raise ApiError(404, "not_found", "Dataset not found.")
    if ds["status"] != "valid":
        raise ApiError(409, "dataset_invalid", "Dataset failed validation. Fix it and upload again.")
    if req.backend not in s.allowed_backends:
        raise ApiError(422, "backend_not_allowed", f"Backend must be one of {list(s.allowed_backends)}.")
    base_model = req.base_model
    if req.backend == "lora":
        base_model = base_model or s.allowed_base_models[0]
        if base_model not in s.allowed_base_models:
            raise ApiError(422, "base_model_not_allowed", f"Base model must be one of {list(s.allowed_base_models)}.")
    else:
        base_model = None

    backend = get_backend(req.backend)
    report = json.loads(ds["report_json"])
    train_bytes = ctx.storage.get_bytes(ds["train_key"])
    avg_chars = len(train_bytes) / max(report["train_count"], 1)
    est = estimate_training_cost(s, backend.uses_gpu, report["train_count"], req.hyperparameters.epochs, avg_chars)
    spent = month_spend(ctx.db, principal.tenant)
    if spent + est["usd"] > s.monthly_budget_usd:
        raise ApiError(409, "budget_exceeded",
                       f"Estimated cost ${est['usd']:.2f} plus ${spent:.2f} already planned this month "
                       f"exceeds the budget of ${s.monthly_budget_usd:.2f}.")
    active = ctx.db.one("SELECT COUNT(*) AS c FROM training_jobs WHERE tenant=? AND status NOT IN "
                        "('SUCCEEDED','FAILED','CANCELLED')", (principal.tenant,))["c"]
    if active >= s.max_active_jobs_per_tenant:
        raise ApiError(429, "too_many_active_jobs", "Too many training jobs are already running for this tenant.")

    config = {"task_type": req.task_type, "base_model": base_model, "dataset_id": ds["id"],
              "hyperparameters": req.hyperparameters.model_dump(), "gate": req.gate.model_dump(),
              "cost_estimate": est}
    job_id = new_id("job")
    try:
        ctx.db.execute(
            "INSERT INTO training_jobs (id, tenant, idem_key, request_hash, dataset_id, model_name, backend,"
            " config_json, status, progress, estimated_cost_usd, created_by, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,'QUEUED',0,?,?,?)",
            (job_id, principal.tenant, idem_key, request_hash, ds["id"], req.model_name, req.backend,
             json.dumps(config), est["usd"], principal.name, now_iso()))
    except sqlite3.IntegrityError:  # a parallel request with the same key won the race
        row = ctx.db.one("SELECT * FROM training_jobs WHERE tenant=? AND idem_key=?", (principal.tenant, idem_key))
        if row and row["request_hash"] == request_hash:
            return present_job(row), False
        raise ApiError(409, "conflict", "Could not create the job. Try again.")

    row = ctx.db.one("SELECT * FROM training_jobs WHERE id=?", (job_id,))
    try:
        ref = ctx.orchestrator.start(row, ds)
        ctx.db.execute("UPDATE training_jobs SET execution_ref=? WHERE id=?", (ref, job_id))
    except Exception as exc:
        log.exception("could not start job %s", job_id)
        set_status(ctx.db, job_id, "FAILED", "Could not start the workflow.")
        raise ApiError(502, "orchestrator_error", "Could not start the training workflow.") from exc
    audit.record(ctx.db, principal.tenant, principal.name, "job.create", job_id, {"backend": req.backend})
    return present_job(ctx.db.one("SELECT * FROM training_jobs WHERE id=?", (job_id,))), True


def cancel_job(ctx, principal, job_id: str) -> dict:
    row = get_job_row(ctx.db, principal.tenant, job_id)
    if row["status"] in TERMINAL:
        raise ApiError(409, "already_finished", f"Job is already {row['status']}.")
    ctx.db.execute("UPDATE training_jobs SET cancel_requested=1 WHERE id=?", (job_id,))
    if row["status"] == "QUEUED":
        set_status(ctx.db, job_id, "CANCELLED")
    else:
        try:
            if ctx.orchestrator.cancel(row):
                set_status(ctx.db, job_id, "CANCELLED")
        except Exception:
            log.exception("could not stop execution for %s", job_id)
    audit.record(ctx.db, principal.tenant, principal.name, "job.cancel", job_id)
    return present_job(get_job_row(ctx.db, principal.tenant, job_id))


def _read_jsonl(raw: bytes) -> list[dict]:
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def run_local_job(ctx, job_id: str) -> None:
    """Whole pipeline on this machine: validate -> train -> evaluate -> register."""
    db = ctx.db
    try:
        row = db.one("SELECT * FROM training_jobs WHERE id=?", (job_id,))
        if not row or row["status"] != "QUEUED":
            return  # cancelled before it started
        config = json.loads(row["config_json"])

        def cancelled() -> bool:
            r = db.one("SELECT cancel_requested FROM training_jobs WHERE id=?", (job_id,))
            return bool(r and r["cancel_requested"])

        if not set_status(db, job_id, "VALIDATING"):
            return
        ds = db.one("SELECT * FROM datasets WHERE id=?", (row["dataset_id"],))
        if not ds or ds["status"] != "valid":
            set_status(db, job_id, "FAILED", "Dataset is missing or invalid.")
            return
        train_rows = _read_jsonl(ctx.storage.get_bytes(ds["train_key"]))
        val_rows = _read_jsonl(ctx.storage.get_bytes(ds["val_key"]))
        if cancelled():
            set_status(db, job_id, "CANCELLED")
            return

        backend = get_backend(row["backend"])
        if not set_status(db, job_id, "TRAINING"):
            return
        try:
            artifact = backend.train(train_rows, config, cancelled)
        except TrainingCancelled:
            set_status(db, job_id, "CANCELLED")
            return
        if cancelled():
            set_status(db, job_id, "CANCELLED")
            return

        set_status(db, job_id, "EVALUATING")
        task = config["task_type"]
        cand = evaluate(backend.load(artifact), val_rows, task)
        base = evaluate(backend.baseline(train_rows, config), val_rows, task)

        set_status(db, job_id, "REGISTERING")
        key = f"{row['tenant']}/jobs/{job_id}/artifact.bin"
        ctx.storage.put_bytes(key, artifact)
        finalize_job(ctx, job_id, key, cand, base)
    except Exception:
        log.exception("job %s failed", job_id)
        set_status(db, job_id, "FAILED", "Training failed. See server logs for details.")


def finalize_job(ctx, job_id: str, artifact_key: str, cand: dict, base: dict) -> dict:
    """Apply the gate and register the model version. Used by local runs and the cloud callback."""
    db = ctx.db
    row = db.one("SELECT * FROM training_jobs WHERE id=?", (job_id,))
    if not row:
        raise ApiError(404, "not_found", "Training job not found.")
    if row["status"] in TERMINAL:
        raise ApiError(409, "already_finished", f"Job is already {row['status']}.")
    prefix = f"{row['tenant']}/jobs/{job_id}/"
    if not artifact_key.startswith(prefix):
        raise ApiError(422, "bad_artifact_key", "Artifact key is outside this job's folder.")
    config = json.loads(row["config_json"])
    ds = db.one("SELECT * FROM datasets WHERE id=?", (row["dataset_id"],))
    status, reasons = decide(cand, base, config["gate"], ctx.settings.min_val_for_auto_approve)
    for _ in range(5):
        mv_id = new_id("mv")
        try:
            with db.tx() as conn:
                v = conn.execute("SELECT COALESCE(MAX(version),0) AS v FROM model_versions WHERE tenant=? AND name=?",
                                 (row["tenant"], row["model_name"])).fetchone()["v"] + 1
                conn.execute(
                    "INSERT INTO model_versions (id, tenant, name, version, training_job_id, dataset_id, dataset_sha256,"
                    " backend, base_model, artifact_key, metrics_json, approval_status, decision_reasons_json,"
                    " code_version, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (mv_id, row["tenant"], row["model_name"], v, job_id, ds["id"], ds["sha256"], row["backend"],
                     config.get("base_model"), artifact_key,
                     json.dumps({"candidate": cand, "baseline": base}), status, json.dumps(reasons),
                     ctx.settings.code_version, now_iso()))
                conn.execute("UPDATE training_jobs SET model_version_id=? WHERE id=?", (mv_id, job_id))
            break
        except sqlite3.IntegrityError:
            continue
    else:
        raise ApiError(409, "conflict", "Could not allocate a model version.")
    set_status(db, job_id, "SUCCEEDED")  # "training finished" is separate from "model approved"
    audit.record(db, row["tenant"], "system", "model.register", mv_id, {"approval_status": status})
    return {"model_version_id": mv_id, "approval_status": status, "reasons": reasons}
