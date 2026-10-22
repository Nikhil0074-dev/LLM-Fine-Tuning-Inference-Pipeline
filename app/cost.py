from __future__ import annotations

from .util import month_start_iso


def estimate_training_cost(settings, uses_gpu: bool, rows: int, epochs: int, avg_chars: float) -> dict:
    """Rough estimate. It rests on assumed throughput, so treat it as a guard rail, not a bill."""
    if not uses_gpu:
        return {"hours": 0.0, "usd": 0.0, "assumptions": "CPU-only backend, no cloud cost"}
    tokens = rows * epochs * (avg_chars / 4.0 + 20)
    seconds = tokens / max(settings.assumed_tokens_per_sec, 1.0) + settings.training_startup_seconds
    hours = seconds / 3600.0
    return {"hours": round(hours, 4), "usd": round(hours * settings.gpu_hourly_usd, 4),
            "assumptions": f"{settings.assumed_tokens_per_sec:.0f} tokens/s, ${settings.gpu_hourly_usd}/h, "
                           f"{settings.training_startup_seconds:.0f}s startup"}


def month_spend(db, tenant: str) -> float:
    row = db.one("SELECT COALESCE(SUM(estimated_cost_usd), 0) AS s FROM training_jobs WHERE tenant=? AND created_at>=?",
                 (tenant, month_start_iso()))
    return float(row["s"])
