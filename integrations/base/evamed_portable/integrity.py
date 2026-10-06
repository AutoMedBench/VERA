"""Standalone canonical identities, source validation, and fresh execution signatures."""
from __future__ import annotations
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from uuid import NAMESPACE_URL, uuid4, uuid5
import blake3
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(value):
    return blake3.blake3(canonical(value)).hexdigest()


def byte_digest(value):
    return blake3.blake3(value).hexdigest()


def file_digest(path):
    hasher = blake3.blake3()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError("nonfinite_json")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError("invalid_relative_path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(p in {".", ".."} for p in path.parts):
        raise ValueError("invalid_relative_path")
    return path


def contained_file(root, name):
    root = Path(root).resolve(strict=True)
    path = root.joinpath(*relative_path(name).parts)
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("source_symlink_forbidden")
    if not path.is_file():
        raise ValueError("source_file_missing")
    return path


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("xb") as stream:
            stream.write(canonical(value))
    else:
        temporary = path.with_name(path.name + ".tmp-" + str(uuid4()))
        with temporary.open("xb") as stream:
            stream.write(canonical(value))
        temporary.replace(path)


class Signer:
    """A distinct fresh execution key, never a historical authority replacement."""
    def __init__(self, key_path):
        self.path = Path(key_path)
        if not self.path.is_absolute() or self.path.is_symlink():
            raise ValueError("fresh_key_requires_absolute_regular_path")
        if self.path.stat().st_mode & 0o077:
            raise ValueError("fresh_private_key_permissions")
        key = serialization.load_pem_private_key(self.path.read_bytes(), password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError("fresh_key_type")
        self.key = key
        self.public = base64.b64encode(key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()
        self.key_id = str(uuid5(NAMESPACE_URL, "eva-medresearch-fresh:" + self.public))

    @classmethod
    def create(cls, key_path):
        path = Path(key_path)
        if not path.is_absolute():
            raise ValueError("fresh_key_requires_absolute_path")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.exists():
            return cls(path)
        key = Ed25519PrivateKey.generate()
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(key.private_bytes(serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        return cls(path)

    def sign(self, payload):
        return {"schema": "eva.medresearch-fresh-execution-receipt.v1", "key_id": self.key_id,
                "public_key_base64": self.public, "payload": payload,
                "payload_blake3": digest(payload),
                "signature_base64": base64.b64encode(self.key.sign(canonical(payload))).decode(),
                "historical_authority_claimed": False}


def verify_receipt(receipt, public_key):
    if receipt.get("schema") != "eva.medresearch-fresh-execution-receipt.v1" or receipt.get("historical_authority_claimed") is not False:
        raise ValueError("fresh_receipt_schema")
    if receipt["public_key_base64"] != public_key or receipt["payload_blake3"] != digest(receipt["payload"]):
        raise ValueError("fresh_receipt_binding")
    Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key, validate=True)).verify(
        base64.b64decode(receipt["signature_base64"], validate=True), canonical(receipt["payload"]))
    return receipt["payload"]


def verify_upstream(path, *, size, sha256):
    raw = Path(path).read_bytes()
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("upstream_source_commitment_mismatch")
    return raw


def timestamp():
    return datetime.now(timezone.utc).isoformat()
