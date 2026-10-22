"""End-to-end demo against a running API (no third-party packages needed).

  PowerShell:
    $env:LLMP_API_KEYS = "admin-key-0123456789ab:admin:demo:alice"
    python -m uvicorn app.main:app_factory --factory
  In a second PowerShell window:
    python scripts/demo.py --key admin-key-0123456789ab
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid


def call(base, key, method, path, body=None, raw=None, headers=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    h = {"X-API-Key": key, **(headers or {})}
    if body is not None:
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {path} -> {e.code}: {e.read().decode()}")
    except urllib.error.URLError as e:
        sys.exit(f"Cannot connect to {base}: {e.reason}. Start the API first.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--key", required=True)
    ap.add_argument("--data", default="sample_data/support.jsonl")
    a = ap.parse_args()
    api = "/api/v1"

    with open(a.data, "rb") as fh:
        raw = fh.read()
    ds = call(a.base, a.key, "POST", f"{api}/datasets?name=support-data&format=jsonl", raw=raw)
    print("dataset:", ds["status"], "train/val =", ds["report"]["train_count"], "/", ds["report"]["val_count"])
    if ds["status"] != "valid":
        sys.exit(json.dumps(ds["report"]["errors"]))

    job = call(a.base, a.key, "POST", f"{api}/training-jobs",
               {"dataset_id": ds["id"], "model_name": "support-bot", "backend": "simple", "task_type": "generation"},
               headers={"Idempotency-Key": uuid.uuid4().hex})
    while job["status"] not in ("SUCCEEDED", "FAILED", "CANCELLED"):
        time.sleep(0.5)
        job = call(a.base, a.key, "GET", f"{api}/training-jobs/{job['id']}")
    print("job:", job["status"], job["error"] or "")
    if job["status"] != "SUCCEEDED":
        sys.exit(1)

    model = call(a.base, a.key, "GET", f"{api}/models/{job['model_version_id']}")
    c, b = model["metrics"]["candidate"], model["metrics"]["baseline"]
    print(f"model v{model['version']}: {model['approval_status']} | score {c['primary']} vs baseline {b['primary']}")
    print("reasons:", model["decision_reasons"])
    if model["approval_status"] != "approved":
        sys.exit("Not approved: it will not be deployed.")

    print("deployed:", call(a.base, a.key, "POST", f"{api}/models/{model['id']}/deploy")["status"])
    q = {"model_name": "support-bot", "prompt": "how can I reset my password?", "temperature": 0.0}
    for i in (1, 2):
        r = call(a.base, a.key, "POST", f"{api}/inference", q)
        print(f"ask #{i}: cached={r['cached']} latency={r['latency_ms']}ms -> {r['output']}")
    print(json.dumps(call(a.base, a.key, "GET", f"{api}/metrics")["inference"]["latency_ms"], indent=2))


if __name__ == "__main__":
    main()
