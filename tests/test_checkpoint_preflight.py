"""Reproducible CPU-only checkpoint metadata checks; no model data or Torch.

Tiny synthetic objects implement the DCP metadata fields consumed by the audit.
The shard is deliberately not a tensor payload: these tests assert metadata and
storage-boundary validation, never tensor-value integrity.
"""

from copy import deepcopy
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def _load_script(name):
    path = Path(__file__).resolve().parents[1] / "training" / "slime" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"eva_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = _load_script("checkpoint_preflight")
LM_KEY = "language_model.embedding.word_embeddings.weight"


@dataclass(frozen=True)
class StorageIndex:
    fqn: str
    offset: tuple[int, ...]


@pytest.fixture
def synthetic_checkpoint(tmp_path):
    (tmp_path / "sample.distcp").write_bytes(b"\x00" * 32)
    chunks = [SimpleNamespace(offsets=(row, 0), sizes=(1, 2)) for row in (0, 1)]
    tensor = SimpleNamespace(size=(2, 2), properties=SimpleNamespace(dtype="torch.bfloat16"), chunks=chunks)
    storage = {
        StorageIndex(LM_KEY, (row, 0)): SimpleNamespace(relative_path="sample.distcp", offset=8 * row, length=8)
        for row in (0, 1)
    }
    metadata = SimpleNamespace(state_dict_metadata={LM_KEY: tensor}, storage_data=storage)
    return metadata, {LM_KEY: (2, 2)}, tmp_path


def test_complete_synthetic_metadata_is_accepted(synthetic_checkpoint):
    report = preflight._validate_metadata(*synthetic_checkpoint)
    assert report["language_tensor_chunks"] == 2
    assert report["storage_extent_entries"] == 2
    assert report["shards"]["sample.distcp"]["file_bytes"] == 32
    assert all(not item["completeness_checked"] for item in report["frozen_auxiliary_exceptions"].values())
    json.dumps(report)  # The caller embeds this report in its run receipt.


@pytest.mark.parametrize("mutation, error", [
    ("missing_key", "language key mismatch"),
    ("wrong_shape", "tensor shape differs"),
    ("empty_chunks", "tensor has no chunks"),
    ("overlapping_chunks", "overlapping tensor chunks"),
    ("missing_storage", "chunk/storage index mismatch"),
    ("unexpected_key", "language key mismatch"),
])
def test_six_original_metadata_tamper_cases_fail_closed(synthetic_checkpoint, mutation, error):
    metadata, expected, directory = synthetic_checkpoint
    tensor = metadata.state_dict_metadata[LM_KEY]
    if mutation == "missing_key":
        metadata.state_dict_metadata.pop(LM_KEY)
    elif mutation == "wrong_shape":
        tensor.size = (3, 2)
    elif mutation == "empty_chunks":
        tensor.chunks = []
    elif mutation == "overlapping_chunks":
        tensor.chunks.append(deepcopy(tensor.chunks[0]))
    elif mutation == "missing_storage":
        metadata.storage_data.pop(StorageIndex(LM_KEY, (0, 0)))
    elif mutation == "unexpected_key":
        metadata.state_dict_metadata["language_model.unexpected.weight"] = deepcopy(tensor)
    with pytest.raises(ValueError, match=error):
        preflight._validate_metadata(metadata, expected, directory)


@pytest.mark.parametrize("mutation, error", [
    ("wrong_dtype", "language dtype differs"),
    ("truncated_shard", "extent missing/truncated"),
    ("incomplete_coverage", "incomplete tensor chunk coverage"),
])
def test_dtype_and_storage_boundaries_fail_closed(synthetic_checkpoint, mutation, error):
    metadata, expected, directory = synthetic_checkpoint
    if mutation == "wrong_dtype":
        metadata.state_dict_metadata[LM_KEY].properties.dtype = "torch.float32"
    elif mutation == "truncated_shard":
        metadata.storage_data[StorageIndex(LM_KEY, (1, 0))].length = 64
    elif mutation == "incomplete_coverage":
        metadata.state_dict_metadata[LM_KEY].chunks.pop()
    with pytest.raises(ValueError, match=error):
        preflight._validate_metadata(metadata, expected, directory)


def test_warmstart_preflight_failure_precedes_gpu_or_output_creation(tmp_path, monkeypatch):
    launcher = _load_script("run_full_parameter")
    data = tmp_path / "tasks.jsonl"
    data.write_text(json.dumps({"metadata": {"judge_backend": "native_astra"}}) + "\n")
    args = SimpleNamespace(stage="grpo", steps=2, batch_size=4, samples_per_prompt=4,
        data=data, output=tmp_path / "must-not-exist", model=tmp_path / "model",
        load=tmp_path / "checkpoint", max_tokens=24576, max_response_tokens=4096,
        generation_function="eva_agent.training.slime_rollout.generate_grpo_rollout",
        judge_backend="native_astra", dry_run=False)
    calls = []

    def reject(root, model_path):
        calls.append((root, model_path))
        raise ValueError("synthetic incomplete language checkpoint")

    def forbid_gpu(*args, **kwargs):
        pytest.fail("GPU/subprocess access occurred before strict checkpoint rejection")

    monkeypatch.setattr(launcher, "arguments", lambda: args)
    monkeypatch.setitem(sys.modules, "checkpoint_preflight", SimpleNamespace(verify_checkpoint=reject))
    monkeypatch.setattr(launcher.subprocess, "check_output", forbid_gpu)
    with pytest.raises(ValueError, match="synthetic incomplete language checkpoint"):
        launcher.main()
    assert calls == [(args.load, args.model)]
    assert not args.output.exists()
