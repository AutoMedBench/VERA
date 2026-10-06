"""Atomic, replayable teacher API exchanges; credentials remain in authentication."""
import base64
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time

from .feedback import ROOT, committed, read, require
from evamed_portable.integrity import byte_digest, digest, timestamp
from .trial_runner import journal


def blob(path, raw):
    path = Path(path).resolve();require(path.is_relative_to(ROOT), 'teacher_artifact_outside_workspace')
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    if path.exists():
        require(path.read_bytes() == raw, 'teacher_immutable_blob_changed')
        return
    fd, temporary = tempfile.mkstemp(dir=path.parent,prefix='.'+path.name)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        try:os.link(temporary,path)
        except FileExistsError:require(path.read_bytes() == raw, 'teacher_immutable_blob_changed')
    finally:os.unlink(temporary)


def api_client():
    spec = importlib.util.spec_from_file_location('harness_teacher_api',ROOT/'evamed-codex/scripts/api-conformance.py')
    api = importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    return api


def exchange(output, payload):
    output = Path(output).resolve();require(output.is_relative_to(ROOT), 'teacher_exchange_outside_workspace')
    output.mkdir(parents=True,exist_ok=True,mode=0o700)
    api = api_client();keys = api._secret_keys()
    require(all(key not in json.dumps(payload) for key in keys), 'credential_in_teacher_payload')
    journal(output/'request.json',payload);(output/'request.json').chmod(0o600)
    path = output/'exchange.json'
    if path.exists():
        result = committed(path)
        require(result['request_blake3'] == digest(payload), 'teacher_exchange_request_changed')
    else:
        require(not (output/'request-intent.json').exists(), 'teacher_exchange_interrupted_without_response')
        started_at = timestamp()
        journal(output/'request-intent.json',{'timestamp_utc':started_at,'request_blake3':digest(payload)})
        started = time.monotonic()
        base = read(ROOT/'evamed-codex/config/model-matrix.json')['base_url'].rstrip('/')
        try:status, raw = api._request(base+'/chat/completions',keys[0],payload,240)
        except Exception as exc:
            journal(output/'failure.json',{'stage':'provider_request','error_type':type(exc).__name__,
                    'timestamp_utc':timestamp(),'promoted':False})
            raise
        require(all(key.encode() not in raw for key in keys), 'credential_in_teacher_response')
        result = {'schema':'eva.harness-teacher-exchange.v1','timestamp_utc':started_at,
            'http_status':status,'elapsed_seconds':time.monotonic()-started,'requested_model':payload['model'],
            'request_blake3':digest(payload),'raw_response_blake3':byte_digest(raw),
            'response_base64':base64.b64encode(raw).decode()}
        result['document_blake3'] = digest(result)
        journal(path,result);path.chmod(0o600)
    raw = base64.b64decode(result['response_base64'],validate=True)
    require(byte_digest(raw) == result['raw_response_blake3'] and all(key.encode() not in raw for key in keys),
            'teacher_response_integrity_or_credential_failure')
    blob(output/'response.json',raw)
    return result, json.loads(raw)
