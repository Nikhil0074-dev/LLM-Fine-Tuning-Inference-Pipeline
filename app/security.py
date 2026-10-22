from __future__ import annotations

import hmac
from dataclasses import dataclass

from fastapi import Depends, Header, Request

from .errors import ApiError


@dataclass(frozen=True)
class Principal:
    name: str
    role: str      # admin | member
    tenant: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def parse_api_keys(spec: str) -> list[tuple[str, Principal]]:
    result = []
    for chunk in filter(None, (c.strip() for c in spec.split(";"))):
        parts = chunk.split(":")
        if len(parts) != 4 or not all(parts):
            raise ValueError("api key entry must look like key:role:tenant:name")
        key, role, tenant, name = parts
        if role not in ("admin", "member"):
            raise ValueError(f"unknown role {role!r}")
        if len(key) < 16:
            raise ValueError("api keys must be at least 16 characters")
        result.append((key, Principal(name=name, role=role, tenant=tenant)))
    return result


def lookup(keys: list[tuple[str, Principal]], presented: str | None) -> Principal | None:
    if not presented:
        return None
    found = None
    for key, principal in keys:  # compare against every key: no early exit timing signal
        if hmac.compare_digest(key.encode(), presented.encode()):
            found = principal
    return found


def get_ctx(request: Request):
    return request.app.state.ctx


def current_principal(request: Request, x_api_key: str | None = Header(default=None)) -> Principal:
    principal = lookup(request.app.state.ctx.api_keys, x_api_key)
    if principal is None:
        raise ApiError(401, "unauthorized", "Missing or invalid X-API-Key header.")
    return principal


def require_admin(principal: Principal = Depends(current_principal)) -> Principal:
    if not principal.is_admin:
        raise ApiError(403, "forbidden", "This action needs the admin role.")
    return principal


def require_internal(request: Request, x_internal_key: str | None = Header(default=None)) -> None:
    expected = request.app.state.ctx.settings.internal_api_key
    if not expected or not x_internal_key or not hmac.compare_digest(expected.encode(), x_internal_key.encode()):
        raise ApiError(403, "forbidden", "Invalid internal key.")
