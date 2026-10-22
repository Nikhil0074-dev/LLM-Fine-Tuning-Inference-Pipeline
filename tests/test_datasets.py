import json

from app.datasets import find_pii, validate_dataset
from conftest import ADMIN, MEMBER, OTHER_ADMIN, jsonl_bytes, upload

KW = dict(min_rows=20, val_ratio=0.2, max_field_chars=8000, block_on_pii=True, max_malformed_ratio=0.2)


def test_valid_jsonl_splits_without_leakage():
    r = validate_dataset(jsonl_bytes(100), "jsonl", **KW)
    assert r.ok and len(r.train_rows) == 80 and len(r.val_rows) == 20
    assert not ({x["prompt"] for x in r.train_rows} & {x["prompt"] for x in r.val_rows})


def test_split_is_deterministic():
    a = validate_dataset(jsonl_bytes(60), "jsonl", **KW)
    b = validate_dataset(jsonl_bytes(60), "jsonl", **KW)
    assert a.val_rows == b.val_rows


def test_duplicates_empty_and_malformed_are_counted():
    rows = [json.dumps({"prompt": f"q{i}", "response": "a"}) for i in range(30)]
    rows += ["{bad json", json.dumps({"prompt": "q1", "response": "b"}), json.dumps({"prompt": " ", "response": "x"}),
             json.dumps([1, 2]), json.dumps({"prompt": "only prompt"})]
    r = validate_dataset("\n".join(rows).encode(), "jsonl", **KW)
    assert r.ok
    assert r.report["duplicate_records"] == 1 and r.report["empty_records"] == 2
    assert r.report["malformed_records"] == 2 and r.report["valid_records"] == 30


def test_too_many_malformed_rejects():
    rows = ["nope"] * 10 + [json.dumps({"prompt": f"q{i}", "response": "a"}) for i in range(25)]
    r = validate_dataset("\n".join(rows).encode(), "jsonl", **KW)
    assert not r.ok and any("malformed" in e for e in r.report["errors"])


def test_too_few_rows_rejects():
    assert not validate_dataset(jsonl_bytes(5), "jsonl", **KW).ok


def test_pii_blocks_and_is_never_echoed():
    rows = [{"prompt": f"q{i}", "response": "a"} for i in range(30)]
    rows[3]["response"] = "mail me at jane.doe@example.com"
    r = validate_dataset("\n".join(json.dumps(x) for x in rows).encode(), "jsonl", **KW)
    assert not r.ok and r.report["pii_findings"] == {"email": 1}
    assert "jane.doe" not in json.dumps(r.report)
    ok = validate_dataset("\n".join(json.dumps(x) for x in rows).encode(), "jsonl", **{**KW, "block_on_pii": False})
    assert ok.ok and ok.report["warnings"]


def test_card_numbers_use_luhn():
    assert "credit_card" in find_pii("4111 1111 1111 1111")
    assert "credit_card" not in find_pii("order number 1234 5678 9012 3456")


def test_csv_and_missing_columns_and_bad_utf8():
    csv_text = "prompt,response\n" + "\n".join(f"question {i},answer {i % 2}" for i in range(30))
    assert validate_dataset(csv_text.encode(), "csv", **KW).ok
    assert not validate_dataset(b"a,b\n1,2\n", "csv", **KW).ok
    assert not validate_dataset(b"\xff\xfe\x00bad", "jsonl", **KW).ok


def test_upload_endpoint_valid_invalid_and_limits(client):
    r = upload(client)
    assert r.status_code == 201 and r.json()["status"] == "valid" and r.json()["version"] == 1
    assert upload(client).json()["version"] == 2
    bad = upload(client, jsonl_bytes(3), name="tiny-data")
    assert bad.status_code == 201 and bad.json()["status"] == "invalid"
    assert client.post("/api/v1/datasets?name=x", content=b"abc", headers=ADMIN).status_code == 422  # bad name
    assert client.post("/api/v1/datasets?name=empty-one", content=b"", headers=ADMIN).status_code == 400


def test_upload_size_limit(tmp_path):
    from fastapi.testclient import TestClient
    from app.main import create_app
    from conftest import make_settings
    with TestClient(create_app(make_settings(tmp_path, max_upload_bytes=1000))) as c:
        assert upload(c, b"x" * 5000).status_code == 413


def test_roles_and_tenant_isolation(client):
    assert upload(client, headers=MEMBER).status_code == 403
    assert client.get("/api/v1/datasets").status_code == 401
    ds = upload(client).json()
    assert client.get(f"/api/v1/datasets/{ds['id']}", headers=MEMBER).status_code == 200
    assert client.get(f"/api/v1/datasets/{ds['id']}", headers=OTHER_ADMIN).status_code == 404
    assert client.get("/api/v1/datasets", headers=OTHER_ADMIN).json()["items"] == []
