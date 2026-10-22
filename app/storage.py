from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
import re

_SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\-]*$")


def check_key(key: str) -> str:
    if not _SAFE_KEY.match(key) or ".." in key.split("/") or "//" in key:
        raise ValueError(f"unsafe storage key: {key!r}")
    return key


class Storage(ABC):
    @abstractmethod
    def put_bytes(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def get_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def uri(self, key: str) -> str: ...

    def ping(self) -> bool:
        return True


class LocalStorage(Storage):
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def _path(self, key: str) -> str:
        path = os.path.abspath(os.path.join(self.root, check_key(key)))
        if not path.startswith(self.root + os.sep):
            raise ValueError("path escapes storage root")
        return path

    def put_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)  # atomic: readers never see half-written files
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def get_bytes(self, key: str) -> bytes:
        with open(self._path(key), "rb") as fh:
            return fh.read()

    def exists(self, key: str) -> bool:
        return os.path.isfile(self._path(key))

    def uri(self, key: str) -> str:
        return Path(self._path(key)).as_uri()

    def ping(self) -> bool:
        return os.path.isdir(self.root) and os.access(self.root, os.W_OK)


class S3Storage(Storage):
    def __init__(self, bucket: str, prefix: str = "", region: str = "us-east-1", client=None):
        if not bucket:
            raise ValueError("LLMP_S3_BUCKET is required when storage_backend=s3")
        import boto3  # imported lazily so local mode needs no AWS packages

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = client or boto3.client("s3", region_name=region)

    def _full(self, key: str) -> str:
        key = check_key(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_bytes(self, key: str, data: bytes) -> None:
        self.client.put_object(
            Bucket=self.bucket, Key=self._full(key), Body=data, ServerSideEncryption="AES256"
        )

    def get_bytes(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=self._full(key))["Body"].read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._full(key))
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{self._full(key)}"

    def ping(self) -> bool:
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return True
        except Exception:
            return False
