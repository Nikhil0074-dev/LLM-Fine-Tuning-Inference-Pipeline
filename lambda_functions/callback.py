"""Step Functions -> API bridge. The API stays the only owner of the database and of the approval gate.

event = {"job_id": ..., "action": "stage" | "fail" | "complete", ...}
Environment: API_URL, INTERNAL_API_KEY, S3_BUCKET, S3_PREFIX
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request


def _post(path: str, body: dict) -> dict:
    url = os.environ["API_URL"].rstrip("/") + path
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "X-Internal-Key": os.environ["INTERNAL_API_KEY"]})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        if exc.code == 409:  # job already finished: a retried callback must not fail the workflow
            return {"already_finished": True}
        raise


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("expected an s3://bucket/key URI")
    return parsed.netloc, parsed.path.lstrip("/")


def _relative_key(s3_uri: str, job_id: str, tenant: str = "acme", *, expected_suffix: str | None = None) -> str:
    bucket, key = _parse_s3_uri(s3_uri)
    if bucket != os.environ["S3_BUCKET"]:
        raise ValueError("artifact is in an unexpected bucket")
    prefix = os.environ.get("S3_PREFIX", "").strip("/")
    full_prefix = f"{prefix}/" if prefix else ""
    if not key.startswith(full_prefix):
        raise ValueError("artifact is outside the configured prefix")
    key = key[len(full_prefix):]
    job_prefix = f"{tenant}/jobs/{job_id}/"
    if not key.startswith(job_prefix):
        raise ValueError("artifact is outside the current job folder")
    if expected_suffix and not key.endswith(expected_suffix):
        raise ValueError("unexpected artifact filename")
    return key


def _read_json(s3_uri: str, job_id: str, tenant: str = "acme") -> dict:
    import boto3

    bucket, key = _parse_s3_uri(s3_uri)
    if bucket != os.environ["S3_BUCKET"]:
        raise ValueError("metrics are in an unexpected bucket")
    prefix = os.environ.get("S3_PREFIX", "").strip("/")
    full_prefix = f"{prefix}/" if prefix else ""
    relative = key[len(full_prefix):] if key.startswith(full_prefix) else ""
    if relative != f"{tenant}/jobs/{job_id}/metrics.json":
        raise ValueError("metrics are outside the current job folder")
    body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body)


def handler(event, context=None):
    job_id, action = event["job_id"], event["action"]
    if action == "stage":
        return _post(f"/internal/v1/jobs/{job_id}/stage", {"status": event["status"]})
    if action == "fail":
        return _post(f"/internal/v1/jobs/{job_id}/fail", {"error": str(event.get("error", "unknown"))[:500]})
    if action == "complete":
        tenant = event.get("tenant", "acme")
        artifact_key = _relative_key(event["model_artifact_s3_uri"], job_id, tenant, expected_suffix="/output/model.tar.gz")
        metrics = _read_json(event["metrics_s3_uri"], job_id, tenant)
        return _post(f"/internal/v1/jobs/{job_id}/complete", {
            "artifact_key": artifact_key, "candidate": metrics["candidate"], "baseline": metrics["baseline"]})
    raise ValueError(f"unknown action {action!r}")
