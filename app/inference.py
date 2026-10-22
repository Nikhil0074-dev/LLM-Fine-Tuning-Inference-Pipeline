from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import TimeoutError as FutureTimeout

from .errors import ApiError
from .ml.base import get_backend
from .util import collapse_ws, new_id, now_iso

log = logging.getLogger("llmp.inference")


class PredictorPool:
    """Keeps recently used models in memory (LRU). The first request after a load is a 'cold start'."""

    def __init__(self, storage, max_models: int):
        self.storage = storage
        self.max_models = max(1, max_models)
        self._models: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._key_locks: dict = defaultdict(threading.Lock)

    def get(self, mv: dict):
        key = mv["id"]
        with self._lock:
            if key in self._models:
                self._models.move_to_end(key)
                return self._models[key], False
            klock = self._key_locks[key]
        with klock:  # only one thread loads a given model; the others wait and reuse it
            with self._lock:
                if key in self._models:
                    self._models.move_to_end(key)
                    return self._models[key], False
            predictor = get_backend(mv["backend"]).load(self.storage.get_bytes(mv["artifact_key"]))
            with self._lock:
                self._models[key] = predictor
                while len(self._models) > self.max_models:
                    self._models.popitem(last=False)
            return predictor, True

    def evict(self, key: str) -> None:
        with self._lock:
            self._models.pop(key, None)


def cache_key(tenant: str, model_version_id: str, req, system_prompt_version: str) -> str:
    """Everything that can change the answer is in the key. The model version id is immutable,
    so a new model never sees old answers, and a rollback safely reuses its own old ones."""
    payload = {"tenant": tenant, "mv": model_version_id, "prompt": collapse_ws(req.prompt),
               "max_new_tokens": req.max_new_tokens, "temperature": req.temperature, "sys": system_prompt_version}
    return "inf:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _resolve(ctx, tenant: str, req) -> dict:
    if req.version is not None:
        mv = ctx.db.one("SELECT * FROM model_versions WHERE tenant=? AND name=? AND version=?",
                        (tenant, req.model_name, req.version))
        if not mv:
            raise ApiError(404, "not_found", "Model version not found.")
        if mv["approval_status"] != "approved":
            raise ApiError(409, "not_approved", "Only approved model versions can serve requests.")
        return mv
    mv = ctx.db.one("SELECT m.* FROM deployments d JOIN model_versions m ON m.id=d.model_version_id"
                    " WHERE d.tenant=? AND d.model_name=? AND d.status='active'", (tenant, req.model_name))
    if not mv:
        raise ApiError(404, "no_active_deployment", "This model name has no active deployment.")
    return mv


def _check_rate(ctx, principal) -> None:
    limit = ctx.settings.rate_limit_per_minute
    if limit <= 0:
        return
    window = int(time.time() // 60)
    try:
        count = ctx.cache.incr(f"rl:{principal.tenant}:{principal.name}:{window}", 70)
    except Exception:  # counter store is down: fail open so the cache cannot take inference offline
        log.warning("rate limiter unavailable; allowing request")
        return
    if count > limit:
        raise ApiError(429, "rate_limited", "Too many requests. Slow down.",
                       headers={"Retry-After": str(60 - int(time.time()) % 60)})


def run_inference(ctx, principal, req) -> dict:
    s = ctx.settings
    started = time.perf_counter()
    request_id = new_id("req")
    mv_id, status, cached, cold, tin, tout = None, 200, False, False, 0, 0
    try:
        _check_rate(ctx, principal)
        if len(req.prompt) > s.max_prompt_chars:
            raise ApiError(422, "prompt_too_long", f"Prompt is longer than {s.max_prompt_chars} characters.")
        mv = _resolve(ctx, principal.tenant, req)
        mv_id = mv["id"]
        eligible = req.use_cache and req.temperature <= s.cache_max_temperature
        key = cache_key(principal.tenant, mv_id, req, s.system_prompt_version) if eligible else None
        hit = ctx.cache.get(key) if key else None
        if hit:
            data = json.loads(hit)
            output, tin, tout, cached = data["output"], data["input_tokens"], data["output_tokens"], True
        else:
            try:
                predictor, cold = ctx.pool.get(mv)
            except Exception as exc:
                raise ApiError(503, "model_unavailable", "The model could not be loaded.") from exc
            future = ctx.gen_pool.submit(
                predictor.generate, req.prompt, req.max_new_tokens, req.temperature)
            try:
                result = future.result(timeout=s.inference_timeout_seconds)
            except FutureTimeout as exc:
                future.cancel()
                raise ApiError(504, "timeout", "The model took too long to answer.") from exc
            except Exception as exc:
                raise ApiError(500, "generation_failed", "The model failed to generate an answer.") from exc
            output, tin, tout = result.text, result.input_tokens, result.output_tokens
            if key:
                ctx.cache.set(key, json.dumps({"output": output, "input_tokens": tin, "output_tokens": tout}),
                              s.cache_ttl_seconds)
        latency = (time.perf_counter() - started) * 1000.0
        return {"request_id": request_id, "model_name": mv["name"], "model_version": mv["version"],
                "model_version_id": mv_id, "output": output, "latency_ms": round(latency, 2),
                "cached": cached, "cold_start": cold, "usage": {"input_tokens": tin, "output_tokens": tout}}
    except ApiError as exc:
        status = exc.status
        raise
    except Exception:
        status = 500
        raise
    finally:
        # Only metadata is stored. Prompts and answers are never written to the database.
        try:
            ctx.db.execute(
                "INSERT INTO inference_logs (request_id, tenant, model_version_id, latency_ms, input_tokens,"
                " output_tokens, status_code, cached, cold_start, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (request_id, principal.tenant, mv_id, (time.perf_counter() - started) * 1000.0, tin, tout, status,
                 int(cached), int(cold), now_iso()))
        except Exception:
            pass
