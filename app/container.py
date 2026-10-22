from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .cache import Cache, MemoryCache, RedisCache
from .config import Settings
from .db import Database
from .inference import PredictorPool
from .orchestrator import LocalOrchestrator, StepFunctionsOrchestrator
from .security import Principal, parse_api_keys
from .storage import LocalStorage, S3Storage, Storage


@dataclass
class AppContext:
    settings: Settings
    db: Database
    storage: Storage
    cache: Cache
    pool: PredictorPool
    orchestrator: object
    gen_pool: ThreadPoolExecutor
    api_keys: list = field(default_factory=list)


def build_context(settings: Settings, *, storage: Storage | None = None, cache: Cache | None = None,
                  orchestrator=None) -> AppContext:
    os.makedirs(settings.data_dir, exist_ok=True)
    db = Database(os.path.join(settings.data_dir, "pipeline.db"))
    if storage is None:
        if settings.storage_backend == "s3":
            storage = S3Storage(settings.s3_bucket, settings.s3_prefix, settings.aws_region)
        else:
            storage = LocalStorage(os.path.join(settings.data_dir, "objects"))
    if cache is None:
        cache = RedisCache(settings.redis_url) if settings.redis_url else MemoryCache()
    if orchestrator is None:
        orchestrator = (StepFunctionsOrchestrator(settings) if settings.orchestrator == "stepfunctions"
                        else LocalOrchestrator(settings.max_concurrent_jobs))
    ctx = AppContext(settings=settings, db=db, storage=storage, cache=cache,
                     pool=PredictorPool(storage, settings.max_loaded_models), orchestrator=orchestrator,
                     gen_pool=ThreadPoolExecutor(max_workers=settings.inference_workers, thread_name_prefix="gen"),
                     api_keys=parse_api_keys(settings.api_keys))
    orchestrator.ctx = ctx
    return ctx
