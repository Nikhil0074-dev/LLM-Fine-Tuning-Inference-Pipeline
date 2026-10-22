from __future__ import annotations

from .cost import month_spend
from .util import percentile


def collect(ctx, tenant: str) -> dict:
    rows = ctx.db.query("SELECT * FROM inference_logs WHERE tenant=? ORDER BY id DESC LIMIT 10000", (tenant,))
    ok = [r for r in rows if r["status_code"] == 200]
    cached = [r["latency_ms"] for r in ok if r["cached"]]
    fresh = [r["latency_ms"] for r in ok if not r["cached"]]
    per_model: dict = {}
    for r in rows:
        m = per_model.setdefault(r["model_version_id"] or "unresolved", {"requests": 0, "errors": 0})
        m["requests"] += 1
        m["errors"] += int(r["status_code"] >= 400)
    jobs = {r["status"]: r["c"] for r in ctx.db.query(
        "SELECT status, COUNT(*) AS c FROM training_jobs WHERE tenant=? GROUP BY status", (tenant,))}
    spent = month_spend(ctx.db, tenant)
    return {
        "inference": {
            "requests": len(rows), "errors": len(rows) - len(ok),
            "cache_hit_rate": round(len(cached) / len(ok), 4) if ok else 0.0,
            "cold_starts": sum(r["cold_start"] for r in rows),
            "latency_ms": {"p50_uncached": percentile(fresh, 50), "p95_uncached": percentile(fresh, 95),
                           "p50_cached": percentile(cached, 50), "p95_cached": percentile(cached, 95)},
            "input_tokens": sum(r["input_tokens"] for r in ok), "output_tokens": sum(r["output_tokens"] for r in ok),
            "per_model_version": per_model,
        },
        "training": {"jobs_by_status": jobs, "estimated_spend_this_month_usd": round(spent, 4),
                     "budget_usd": ctx.settings.monthly_budget_usd,
                     "budget_remaining_usd": round(ctx.settings.monthly_budget_usd - spent, 4)},
    }
