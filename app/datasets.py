"""Dataset validation and registration.

Input records need two text fields: "prompt" and "response".
For a classification task, "response" is the label.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import cast

from . import audit
from .errors import ApiError
from .util import collapse_ws, new_id, now_iso

REQUIRED = ("prompt", "response")

PII_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "api_token": re.compile(r"\b(?:sk|pk|ghp|xox[bp])[-_][A-Za-z0-9_\-]{16,}\b"),
}
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn_ok(digits: str) -> bool:
    total, flip = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if flip:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        flip = not flip
    return total % 10 == 0


def find_pii(text: str) -> list[str]:
    found = [name for name, rx in PII_PATTERNS.items() if rx.search(text)]
    for match in _CARD.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            found.append("credit_card")
            break
    return found


@dataclass
class ValidationResult:
    ok: bool
    report: dict
    train_rows: list = field(default_factory=list)
    val_rows: list = field(default_factory=list)


def _parse(raw: bytes, fmt: str) -> tuple[list[tuple[int, dict[str, object]]], int, list[str]]:
    """Return (records as (source_row, object), malformed_count, blocking_errors)."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return [], 0, ["File is not valid UTF-8."]
    records: list[tuple[int, dict[str, object]]] = []
    malformed = 0
    if fmt == "jsonl":
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(obj, dict):
                malformed += 1
                continue
            records.append((line_no, cast(dict[str, object], obj)))
        return records, malformed, []
    if fmt == "csv":
        try:
            reader = csv.DictReader(io.StringIO(text, newline=""))
            header = reader.fieldnames or []
            missing = [c for c in REQUIRED if c not in header]
            if missing:
                return [], 0, [f"CSV is missing required columns: {', '.join(missing)}."]
            for row_no, row in enumerate(reader, start=2):
                if None in row:  # more cells than headers
                    malformed += 1
                    continue
                records.append((row_no, row))
        except csv.Error:
            return [], 0, ["CSV could not be parsed."]
        return records, malformed, []
    return [], 0, [f"Unsupported format {fmt!r}. Use jsonl or csv."]


def validate_dataset(raw: bytes, fmt: str, *, min_rows: int, val_ratio: float, max_field_chars: int,
                     block_on_pii: bool, max_malformed_ratio: float) -> ValidationResult:
    min_rows = max(4, min_rows)
    records, malformed, errors = _parse(raw, fmt)
    report: dict = {
        "format": fmt, "total_records": len(records) + malformed, "malformed_records": malformed,
        "empty_records": 0, "too_long_records": 0, "duplicate_records": 0, "valid_records": 0,
        "pii_findings": {}, "pii_source_rows": [], "train_count": 0, "val_count": 0,
        "distinct_responses": 0, "label_distribution": {}, "errors": errors, "warnings": [],
    }
    if errors:
        return ValidationResult(False, report)

    seen: set[str] = set()
    clean: list[dict] = []
    pii_counter: Counter = Counter()
    pii_rows: list[int] = []
    for source_row, obj in records:
        prompt, response = obj.get("prompt"), obj.get("response")
        if not isinstance(prompt, str) or not isinstance(response, str):
            if prompt is None or response is None:
                report["empty_records"] += 1  # missing field
            else:
                report["malformed_records"] += 1
            continue
        prompt, response = collapse_ws(prompt), collapse_ws(response)
        if not prompt or not response:
            report["empty_records"] += 1
            continue
        if len(prompt) > max_field_chars or len(response) > max_field_chars:
            report["too_long_records"] += 1
            continue
        hits = find_pii(prompt) + find_pii(response)
        if hits:
            pii_counter.update(set(hits))
            if len(pii_rows) < 50:
                pii_rows.append(source_row)
        key = prompt.lower()
        if key in seen:
            report["duplicate_records"] += 1
            continue
        seen.add(key)
        clean.append({"prompt": prompt, "response": response})

    report["valid_records"] = len(clean)
    report["pii_findings"] = dict(pii_counter)
    report["pii_source_rows"] = pii_rows

    if report["total_records"] == 0:
        errors.append("The file has no records.")
    elif report["malformed_records"] / report["total_records"] > max_malformed_ratio:
        errors.append(
            f"Too many malformed records ({report['malformed_records']} of {report['total_records']})."
        )
    elif report["malformed_records"]:
        report["warnings"].append(f"{report['malformed_records']} malformed records were skipped.")
    if pii_counter:
        msg = "Possible personal data or secrets found: " + ", ".join(sorted(pii_counter)) + "."
        (errors if block_on_pii else report["warnings"]).append(msg)
    if report["duplicate_records"]:
        report["warnings"].append(f"{report['duplicate_records']} duplicate prompts were removed.")
    if len(clean) < min_rows:
        errors.append(f"Only {len(clean)} usable unique records; at least {min_rows} are required.")

    if errors:
        return ValidationResult(False, report)

    # Deterministic split. Prompts are already unique, so train and val can never share a prompt.
    clean.sort(key=lambda r: hashlib.sha256(r["prompt"].lower().encode()).hexdigest())
    val_n = min(len(clean) - 1, max(1, round(len(clean) * val_ratio)))
    val_rows, train_rows = clean[:val_n], clean[val_n:]
    assert not ({r["prompt"].lower() for r in val_rows} & {r["prompt"].lower() for r in train_rows})

    dist = Counter(r["response"] for r in clean)
    report["train_count"], report["val_count"] = len(train_rows), len(val_rows)
    report["distinct_responses"] = len(dist)
    if len(dist) <= 50:
        report["label_distribution"] = dict(dist.most_common())
    if len(val_rows) < 20:
        report["warnings"].append(
            f"Validation set has only {len(val_rows)} examples. Models trained on it will need manual approval."
        )
    return ValidationResult(True, report, train_rows, val_rows)


def _jsonl(rows: list[dict]) -> bytes:
    return ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")


def create_dataset(ctx, principal, name: str, fmt: str, raw: bytes) -> dict:
    s = ctx.settings
    result = validate_dataset(
        raw, fmt, min_rows=s.min_dataset_rows, val_ratio=s.val_ratio, max_field_chars=s.max_field_chars,
        block_on_pii=s.block_on_pii, max_malformed_ratio=s.max_malformed_ratio,
    )
    dataset_id = new_id("ds")
    sha = hashlib.sha256(raw).hexdigest()
    train_key = val_key = None
    status = "invalid"
    if result.ok:
        # Only validated, PII-free data is stored. Invalid uploads are discarded (the report is kept).
        train_key = f"{principal.tenant}/datasets/{dataset_id}/train.jsonl"
        val_key = f"{principal.tenant}/datasets/{dataset_id}/val.jsonl"
        ctx.storage.put_bytes(train_key, _jsonl(result.train_rows))
        ctx.storage.put_bytes(val_key, _jsonl(result.val_rows))
        status = "valid"
    for _ in range(5):  # retry if two uploads race for the same version number
        try:
            with ctx.db.tx() as conn:
                row = conn.execute(
                    "SELECT COALESCE(MAX(version), 0) AS v FROM datasets WHERE tenant=? AND name=?",
                    (principal.tenant, name),
                ).fetchone()
                version = row["v"] + 1
                conn.execute(
                    "INSERT INTO datasets (id, tenant, name, version, fmt, status, sha256, train_key, val_key,"
                    " report_json, created_by, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (dataset_id, principal.tenant, name, version, fmt, status, sha, train_key, val_key,
                     json.dumps(result.report), principal.name, now_iso()),
                )
            break
        except sqlite3.IntegrityError:
            continue
    else:
        raise ApiError(409, "conflict", "Could not allocate a dataset version. Try again.")
    audit.record(ctx.db, principal.tenant, principal.name, "dataset.create", dataset_id, {"status": status})
    return get_dataset(ctx, principal.tenant, dataset_id)


def _present(row: dict) -> dict:
    return {
        "id": row["id"], "name": row["name"], "version": row["version"], "format": row["fmt"],
        "status": row["status"], "sha256": row["sha256"], "report": json.loads(row["report_json"]),
        "created_by": row["created_by"], "created_at": row["created_at"],
    }


def get_dataset(ctx, tenant: str, dataset_id: str) -> dict:
    row = ctx.db.one("SELECT * FROM datasets WHERE id=? AND tenant=?", (dataset_id, tenant))
    if not row:
        raise ApiError(404, "not_found", "Dataset not found.")
    return _present(row)


def list_datasets(ctx, tenant: str) -> list[dict]:
    rows = ctx.db.query("SELECT * FROM datasets WHERE tenant=? ORDER BY created_at DESC LIMIT 200", (tenant,))
    return [_present(r) for r in rows]
