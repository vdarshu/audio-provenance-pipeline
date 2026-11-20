"""
Access to the landing zone.

Vendors (and our own reference downloads) drop batches as folders:

    <landing>/<role>/<batch_id>/_batch.json     batch-level metadata
    <landing>/<role>/<batch_id>/...audio files...

The landing zone can be a local folder (default, easiest on a laptop) or a
MinIO / S3 bucket with the same key layout. Either way the ingest step only
ever reads from here; nothing in the landing zone is modified.
"""
import json
import shutil
from pathlib import Path

from pipeline import config


class LocalLanding:
    def __init__(self, root=config.LANDING_DIR):
        self.root = Path(root)

    def list_batches(self, role):
        base = self.root / role
        if not base.exists():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_dir())

    def batch_meta(self, role, batch_id):
        meta_path = self.root / role / batch_id / "_batch.json"
        return json.loads(meta_path.read_text()) if meta_path.exists() else {}

    def list_files(self, role, batch_id):
        base = self.root / role / batch_id
        for p in sorted(base.rglob("*")):
            if p.is_file() and not p.name.startswith("_"):
                yield p.relative_to(base).as_posix()

    def fetch(self, role, batch_id, rel_path, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.root / role / batch_id / rel_path, dest)


class MinioLanding:
    """Same interface, backed by an S3 bucket (MinIO in docker-compose)."""

    def __init__(self):
        import boto3  # only needed when this backend is used
        self.s3 = boto3.client(
            "s3",
            endpoint_url=config.MINIO_ENDPOINT,
            aws_access_key_id=config.MINIO_ACCESS_KEY,
            aws_secret_access_key=config.MINIO_SECRET_KEY,
            region_name="us-east-1",   # MinIO ignores it but boto3 wants one
        )
        self.bucket = config.MINIO_LANDING_BUCKET

    def _keys(self, prefix):
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]

    def list_batches(self, role):
        resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=f"{role}/", Delimiter="/")
        return sorted(p["Prefix"].split("/")[1] for p in resp.get("CommonPrefixes", []))

    def batch_meta(self, role, batch_id):
        try:
            body = self.s3.get_object(Bucket=self.bucket, Key=f"{role}/{batch_id}/_batch.json")["Body"]
            return json.loads(body.read())
        except self.s3.exceptions.NoSuchKey:
            return {}

    def list_files(self, role, batch_id):
        prefix = f"{role}/{batch_id}/"
        for key in sorted(self._keys(prefix)):
            rel = key[len(prefix):]
            if rel and not rel.split("/")[-1].startswith("_"):
                yield rel

    def fetch(self, role, batch_id, rel_path, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket, f"{role}/{batch_id}/{rel_path}", str(dest))


def get_landing():
    return MinioLanding() if config.LANDING_BACKEND == "minio" else LocalLanding()
