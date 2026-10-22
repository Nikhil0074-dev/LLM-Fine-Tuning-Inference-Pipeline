from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from . import audit
from .errors import ApiError
from .util import new_id, now_iso

_smoke_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="smoke")


def present_model(row: dict) -> dict:
    return {
        "id": row["id"], "name": row["name"], "version": row["version"], "backend": row["backend"],
        "base_model": row["base_model"], "training_job_id": row["training_job_id"],
        "dataset_id": row["dataset_id"], "dataset_sha256": row["dataset_sha256"],
        "metrics": json.loads(row["metrics_json"]), "approval_status": row["approval_status"],
        "decision_reasons": json.loads(row["decision_reasons_json"]), "approved_by": row["approved_by"],
        "code_version": row["code_version"], "created_at": row["created_at"],
    }


def get_model_row(db, tenant: str, model_id: str) -> dict:
    row = db.one("SELECT * FROM model_versions WHERE id=? AND tenant=?", (model_id, tenant))
    if not row:
        raise ApiError(404, "not_found", "Model version not found.")
    return row


def list_models(db, tenant: str, name: str | None) -> list[dict]:
    if name:
        rows = db.query("SELECT * FROM model_versions WHERE tenant=? AND name=? ORDER BY version DESC", (tenant, name))
    else:
        rows = db.query("SELECT * FROM model_versions WHERE tenant=? ORDER BY created_at DESC LIMIT 200", (tenant,))
    return [present_model(r) for r in rows]


def set_approval(ctx, principal, model_id: str, decision: str, note: str) -> dict:
    row = get_model_row(ctx.db, principal.tenant, model_id)
    if row["approval_status"] != "needs_review":
        raise ApiError(409, "not_reviewable",
                       f"Only models in needs_review can be decided by hand (this one is {row['approval_status']}). "
                       "A model rejected by the gate must be retrained.")
    new_status = "approved" if decision == "approve" else "rejected"
    ctx.db.execute("UPDATE model_versions SET approval_status=?, approved_by=? WHERE id=?",
                   (new_status, principal.name, model_id))
    audit.record(ctx.db, principal.tenant, principal.name, f"model.{decision}", model_id, {"note": note})
    return present_model(get_model_row(ctx.db, principal.tenant, model_id))


def smoke_test(ctx, mv_row: dict) -> str | None:
    """Load the model and generate once. Returns None if healthy, else a short reason."""
    try:
        predictor, _ = ctx.pool.get(mv_row)
        future = _smoke_pool.submit(predictor.generate, "health check", 8, 0.0)
        out = future.result(timeout=ctx.settings.inference_timeout_seconds)
        if not out.text.strip():
            return "Model returned an empty answer."
        return None
    except FutureTimeout:
        return "Model did not answer in time."
    except Exception as exc:
        return f"Model failed to load or run ({type(exc).__name__})."


def deploy(ctx, principal, model_id: str) -> dict:
    mv = get_model_row(ctx.db, principal.tenant, model_id)
    if mv["approval_status"] != "approved":
        raise ApiError(409, "not_approved", f"Model is {mv['approval_status']}; only approved models can be deployed.")
    problem = smoke_test(ctx, mv)
    dep_id = new_id("dep")
    if problem:
        ctx.pool.evict(mv["id"])
        ctx.db.execute("INSERT INTO deployments (id, tenant, model_name, model_version_id, status, note, created_by,"
                       " created_at) VALUES (?,?,?,?,'failed',?,?,?)",
                       (dep_id, mv["tenant"], mv["name"], mv["id"], problem, principal.name, now_iso()))
        audit.record(ctx.db, principal.tenant, principal.name, "deployment.failed", mv["id"], {"reason": problem})
        raise ApiError(422, "smoke_test_failed", f"Deployment blocked. {problem} The active model is unchanged.")
    with ctx.db.tx() as conn:  # swap atomically: there is never a moment with no active model
        conn.execute("UPDATE deployments SET status='superseded' WHERE tenant=? AND model_name=? AND status='active'",
                     (mv["tenant"], mv["name"]))
        conn.execute("INSERT INTO deployments (id, tenant, model_name, model_version_id, status, created_by,"
                     " created_at) VALUES (?,?,?,?,'active',?,?)",
                     (dep_id, mv["tenant"], mv["name"], mv["id"], principal.name, now_iso()))
    audit.record(ctx.db, principal.tenant, principal.name, "deployment.activate", mv["id"])
    return get_deployment(ctx.db, dep_id)


def rollback(ctx, principal, model_name: str) -> dict:
    cur = ctx.db.one("SELECT * FROM deployments WHERE tenant=? AND model_name=? AND status='active'",
                     (principal.tenant, model_name))
    if not cur:
        raise ApiError(404, "not_found", "No active deployment for this model name.")
    prev = ctx.db.one("SELECT * FROM deployments WHERE tenant=? AND model_name=? AND status='superseded'"
                      " ORDER BY rowid DESC LIMIT 1", (principal.tenant, model_name))
    if not prev:
        raise ApiError(409, "nothing_to_roll_back_to", "There is no earlier deployment to restore.")
    mv = get_model_row(ctx.db, principal.tenant, prev["model_version_id"])
    problem = smoke_test(ctx, mv)
    if problem:
        raise ApiError(422, "smoke_test_failed", f"Rollback blocked. {problem} The active model is unchanged.")
    with ctx.db.tx() as conn:
        conn.execute("UPDATE deployments SET status='rolled_back' WHERE id=?", (cur["id"],))
        conn.execute("UPDATE deployments SET status='active' WHERE id=?", (prev["id"],))
    audit.record(ctx.db, principal.tenant, principal.name, "deployment.rollback", model_name,
                 {"from": cur["model_version_id"], "to": prev["model_version_id"]})
    return get_deployment(ctx.db, prev["id"])


def get_deployment(db, dep_id: str) -> dict:
    row = db.one("SELECT d.*, m.version AS version FROM deployments d JOIN model_versions m"
                 " ON m.id=d.model_version_id WHERE d.id=?", (dep_id,))
    return {k: row[k] for k in ("id", "model_name", "model_version_id", "version", "status", "note",
                                "created_by", "created_at")}


def list_deployments(db, tenant: str, model_name: str | None) -> list[dict]:
    sql = ("SELECT d.id FROM deployments d WHERE d.tenant=?" + (" AND d.model_name=?" if model_name else "")
           + " ORDER BY d.rowid DESC LIMIT 200")
    ids = db.query(sql, (tenant, model_name) if model_name else (tenant,))
    return [get_deployment(db, r["id"]) for r in ids]
