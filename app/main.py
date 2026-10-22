from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .api import internal, router
from .config import Settings
from .container import AppContext, build_context
from .errors import ApiError

log = logging.getLogger("llmp")


def create_app(settings: Settings | None = None, ctx: AppContext | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    ctx = ctx or build_context(settings)
    app = FastAPI(title="LLM Fine-Tuning & Inference Pipeline", version="1.0.0",
                  docs_url="/docs" if settings.docs_enabled else None,
                  redoc_url=None, openapi_url="/openapi.json" if settings.docs_enabled else None)
    app.state.ctx = ctx
    app.include_router(router)
    app.include_router(internal)

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return JSONResponse(status_code=exc.status, headers=exc.headers,
                            content={"error": {"code": exc.code, "message": exc.message}})

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        details = [{"field": ".".join(str(p) for p in e["loc"]), "problem": e["msg"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"error": {"code": "validation_error",
                                                                "message": "Invalid request.", "details": details}})

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse(status_code=500, content={"error": {"code": "internal_error",
                                                                "message": "Something went wrong."}})

    return app


def app_factory() -> FastAPI:  # for: uvicorn app.main:app_factory --factory
    return create_app()
