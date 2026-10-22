from __future__ import annotations

import json

from .util import now_iso


def record(db, tenant: str, actor: str, action: str, target: str = "", detail: dict | None = None) -> None:
    db.execute(
        "INSERT INTO audit_log (tenant, actor, action, target, detail_json, created_at) VALUES (?,?,?,?,?,?)",
        (tenant, actor, action, target, json.dumps(detail or {}, default=str), now_iso()),
    )
