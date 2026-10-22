from __future__ import annotations

import sqlite3
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY, tenant TEXT NOT NULL, name TEXT NOT NULL, version INTEGER NOT NULL,
  fmt TEXT NOT NULL, status TEXT NOT NULL, sha256 TEXT NOT NULL,
  train_key TEXT, val_key TEXT, report_json TEXT NOT NULL,
  created_by TEXT NOT NULL, created_at TEXT NOT NULL,
  UNIQUE (tenant, name, version)
);
CREATE TABLE IF NOT EXISTS training_jobs (
  id TEXT PRIMARY KEY, tenant TEXT NOT NULL, idem_key TEXT NOT NULL, request_hash TEXT NOT NULL,
  dataset_id TEXT NOT NULL REFERENCES datasets(id),
  model_name TEXT NOT NULL, backend TEXT NOT NULL, config_json TEXT NOT NULL,
  status TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0,
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  error TEXT, estimated_cost_usd REAL NOT NULL DEFAULT 0,
  execution_ref TEXT, model_version_id TEXT,
  created_by TEXT NOT NULL, created_at TEXT NOT NULL, started_at TEXT, completed_at TEXT,
  UNIQUE (tenant, idem_key)
);
CREATE TABLE IF NOT EXISTS model_versions (
  id TEXT PRIMARY KEY, tenant TEXT NOT NULL, name TEXT NOT NULL, version INTEGER NOT NULL,
  training_job_id TEXT NOT NULL, dataset_id TEXT NOT NULL, dataset_sha256 TEXT NOT NULL,
  backend TEXT NOT NULL, base_model TEXT, artifact_key TEXT NOT NULL,
  metrics_json TEXT NOT NULL, approval_status TEXT NOT NULL,
  decision_reasons_json TEXT NOT NULL, approved_by TEXT, code_version TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (tenant, name, version)
);
CREATE TABLE IF NOT EXISTS deployments (
  id TEXT PRIMARY KEY, tenant TEXT NOT NULL, model_name TEXT NOT NULL,
  model_version_id TEXT NOT NULL REFERENCES model_versions(id),
  status TEXT NOT NULL, note TEXT, created_by TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_deployments_lookup ON deployments (tenant, model_name, status);
CREATE TABLE IF NOT EXISTS inference_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL, tenant TEXT NOT NULL,
  model_version_id TEXT, latency_ms REAL NOT NULL, input_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0, status_code INTEGER NOT NULL,
  cached INTEGER NOT NULL DEFAULT 0, cold_start INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_inference_tenant ON inference_logs (tenant, id);
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, tenant TEXT NOT NULL, actor TEXT NOT NULL,
  action TEXT NOT NULL, target TEXT, detail_json TEXT, created_at TEXT NOT NULL
);
"""


class Database:
    """Thin SQLite wrapper. One short-lived connection per operation (thread safe)."""

    def __init__(self, path: str):
        self.path = path
        with self.tx() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def tx(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self.tx() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def one(self, sql: str, params: tuple = ()) -> dict | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params: tuple = ()) -> int:
        with self.tx() as conn:
            return conn.execute(sql, params).rowcount
