from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Query, Request, Response
from fastapi.concurrency import run_in_threadpool

from . import datasets, jobs, metrics, registry
from .errors import ApiError
from .inference import run_inference
from .schemas import (ApprovalRequest, CompleteUpdate, FailUpdate, InferenceRequest, StageUpdate, TrainingRequest)
from .security import Principal, current_principal, get_ctx, require_admin, require_internal

router = APIRouter(prefix="/api/v1")
internal = APIRouter(prefix="/internal/v1", dependencies=[Depends(require_internal)])


@router.get("/health")
def health(request: Request):
    ctx = request.app.state.ctx
    checks = {"database": True, "storage": ctx.storage.ping(), "cache": ctx.cache.ping()}
    try:
        ctx.db.one("SELECT 1 AS x")
    except Exception:
        checks["database"] = False
    ok = checks["database"] and checks["storage"]  # a down cache degrades speed, not correctness
    return {"status": "healthy" if ok else "unhealthy", "checks": checks}


# ---------------- datasets ----------------
@router.post("/datasets", status_code=201)
async def upload_dataset(request: Request, name: str = Query(pattern=r"^[a-z0-9][a-z0-9\-]{2,40}$"),
                         fmt: str = Query("jsonl", alias="format", pattern="^(jsonl|csv)$"),
                         principal: Principal = Depends(require_admin)):
    ctx = request.app.state.ctx
    limit = ctx.settings.max_upload_bytes
    declared = request.headers.get("content-length")
    try:
        if declared and int(declared) > limit:
            raise ApiError(413, "too_large", f"Upload is larger than {limit} bytes.")
    except ValueError:
        raise ApiError(400, "bad_request", "Invalid Content-Length header.")
    buf = bytearray()
    async for chunk in request.stream():
        buf.extend(chunk)
        if len(buf) > limit:
            raise ApiError(413, "too_large", f"Upload is larger than {limit} bytes.")
    if not buf:
        raise ApiError(400, "empty_body", "Send the dataset file as the request body.")
    return await run_in_threadpool(datasets.create_dataset, ctx, principal, name, fmt, bytes(buf))


@router.get("/datasets")
def list_datasets(ctx=Depends(get_ctx), principal: Principal = Depends(current_principal)):
    return {"items": datasets.list_datasets(ctx, principal.tenant)}


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, ctx=Depends(get_ctx), principal: Principal = Depends(current_principal)):
    return datasets.get_dataset(ctx, principal.tenant, dataset_id)


# ---------------- training jobs ----------------
@router.post("/training-jobs", status_code=201)
def create_training_job(req: TrainingRequest, response: Response, ctx=Depends(get_ctx),
                        principal: Principal = Depends(require_admin),
                        idempotency_key: str | None = Header(default=None)):
    if not idempotency_key or not (8 <= len(idempotency_key) <= 100):
        raise ApiError(400, "idempotency_key_required",
                       "Send an Idempotency-Key header (8-100 characters) so retries never start a second GPU job.")
    job, created = jobs.create_job(ctx, principal, idempotency_key, req)
    if not created:
        response.status_code = 200
    return job


@router.get("/training-jobs")
def list_training_jobs(ctx=Depends(get_ctx), principal: Principal = Depends(current_principal)):
    rows = ctx.db.query("SELECT * FROM training_jobs WHERE tenant=? ORDER BY created_at DESC LIMIT 200",
                        (principal.tenant,))
    return {"items": [jobs.present_job(r) for r in rows]}


@router.get("/training-jobs/{job_id}")
def get_training_job(job_id: str, ctx=Depends(get_ctx), principal: Principal = Depends(current_principal)):
    return jobs.present_job(jobs.get_job_row(ctx.db, principal.tenant, job_id))


@router.post("/training-jobs/{job_id}/cancel")
def cancel_training_job(job_id: str, ctx=Depends(get_ctx), principal: Principal = Depends(require_admin)):
    return jobs.cancel_job(ctx, principal, job_id)


# ---------------- models and deployments ----------------
@router.get("/models")
def list_models(name: str | None = Query(None), ctx=Depends(get_ctx),
                principal: Principal = Depends(current_principal)):
    return {"items": registry.list_models(ctx.db, principal.tenant, name)}


@router.get("/models/{model_id}")
def get_model(model_id: str, ctx=Depends(get_ctx), principal: Principal = Depends(current_principal)):
    return registry.present_model(registry.get_model_row(ctx.db, principal.tenant, model_id))


@router.post("/models/{model_id}/review")
def review_model(model_id: str, body: ApprovalRequest, ctx=Depends(get_ctx),
                 principal: Principal = Depends(require_admin)):
    return registry.set_approval(ctx, principal, model_id, body.decision, body.note)


@router.post("/models/{model_id}/deploy")
def deploy_model(model_id: str, ctx=Depends(get_ctx), principal: Principal = Depends(require_admin)):
    return registry.deploy(ctx, principal, model_id)


@router.get("/deployments")
def list_deployments(model_name: str | None = Query(None), ctx=Depends(get_ctx),
                     principal: Principal = Depends(current_principal)):
    return {"items": registry.list_deployments(ctx.db, principal.tenant, model_name)}


@router.post("/deployments/{model_name}/rollback")
def rollback(model_name: str, ctx=Depends(get_ctx), principal: Principal = Depends(require_admin)):
    return registry.rollback(ctx, principal, model_name)


# ---------------- inference + metrics ----------------
@router.post("/inference")
def inference(req: InferenceRequest, ctx=Depends(get_ctx), principal: Principal = Depends(current_principal)):
    return run_inference(ctx, principal, req)


@router.get("/metrics")
def get_metrics(ctx=Depends(get_ctx), principal: Principal = Depends(require_admin)):
    return metrics.collect(ctx, principal.tenant)


# ---------------- internal callbacks (Step Functions Lambdas only) ----------------
def _internal_job(ctx, job_id: str) -> dict:
    row = ctx.db.one("SELECT * FROM training_jobs WHERE id=?", (job_id,))
    if not row:
        raise ApiError(404, "not_found", "Training job not found.")
    return row


@internal.post("/jobs/{job_id}/stage")
def internal_stage(job_id: str, body: StageUpdate, ctx=Depends(get_ctx)):
    _internal_job(ctx, job_id)
    return {"updated": jobs.set_status(ctx.db, job_id, body.status)}


@internal.post("/jobs/{job_id}/fail")
def internal_fail(job_id: str, body: FailUpdate, ctx=Depends(get_ctx)):
    _internal_job(ctx, job_id)
    return {"updated": jobs.set_status(ctx.db, job_id, "FAILED", body.error)}


@internal.post("/jobs/{job_id}/complete")
def internal_complete(job_id: str, body: CompleteUpdate, ctx=Depends(get_ctx)):
    _internal_job(ctx, job_id)
    jobs.set_status(ctx.db, job_id, "REGISTERING")
    return jobs.finalize_job(ctx, job_id, body.artifact_key, body.candidate.model_dump(), body.baseline.model_dump())
