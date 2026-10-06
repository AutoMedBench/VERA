"""Strict, CPU-only audit of our trusted Qwen3.5-9B Megatron checkpoints.

``verify_checkpoint`` accepts ONLY locally produced, trusted DCP metadata:
PyTorch's .metadata is pickle, so calling it explicitly trusts that file. Never
point it at a downloaded/untrusted checkpoint. No model or GPU is instantiated.

The expected map is independent of the checkpoint: Qwen config -> HF tensor
shapes -> reviewed Qwen35Bridge fusion and GatedDeltaNet DCP split rules. Local
safetensors headers must exactly corroborate the complete HF language map.
This proves declared tensor completeness and storage presence, not tensor-value
integrity, optimizer completeness, or that Megatron actually loaded the values.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pickle
import re
import struct
from typing import Any

from blake3 import blake3


EXPECTED_LANGUAGE_PARAMETERS = 8_953_803_264
SCHEMA = "eva.qwen35-dcp-language-preflight.v1"


def _digest(value: Any) -> str:
    return blake3(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _local_file(root: Path, relative: str) -> Path:
    path = root / relative
    _require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
             "checkpoint/header contains an unsafe relative path")
    _require(path.is_file() and not path.is_symlink(), f"missing or symlinked file: {path}")
    _require(path.resolve().is_relative_to(root.resolve()), "file escapes its root")
    return path


def expected_language_map(model_path: Path) -> tuple[dict[str, tuple[int, ...]], dict]:
    """Derive all language tensors from architecture, not saved DCP inventory."""
    model_path = Path(model_path).resolve()
    config_bytes = _local_file(model_path, "config.json").read_bytes()
    config = json.loads(config_bytes)
    c = config["text_config"]
    _require(config.get("model_type") == "qwen3_5" and
             config.get("tie_word_embeddings") is False and
             c.get("attention_bias") is False and c.get("attn_output_gate") is True,
             "unsupported Qwen architecture/embedding/attention configuration")
    h, f, n, vocab = (int(c[k]) for k in
                      ("hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size"))
    heads, kv, d = (int(c[k]) for k in
                    ("num_attention_heads", "num_key_value_heads", "head_dim"))
    kh, vh, kd, vd, kernel = (int(c[k]) for k in (
        "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
        "linear_value_head_dim", "linear_conv_kernel_dim"))
    kinds = c["layer_types"]
    _require(len(kinds) == n and n == 32 and all(x in {"linear_attention", "full_attention"} for x in kinds),
             "unsupported Qwen layer inventory")
    _require(not c.get("num_experts") and c.get("hidden_act") == "silu", "only dense SwiGLU Qwen is supported")
    hf: dict[str, tuple[int, ...]] = {}
    dcp: dict[str, tuple[int, ...]] = {}

    def pair(dst: str, src: str, shape: tuple[int, ...]) -> None:
        _require(dst not in dcp and src not in hf, "duplicate architecture mapping")
        dcp[dst], hf[src] = shape, shape

    pair("language_model.embedding.word_embeddings.weight", "model.language_model.embed_tokens.weight", (vocab, h))
    pair("language_model.output_layer.weight", "lm_head.weight", (vocab, h))
    pair("language_model.decoder.final_layernorm.weight", "model.language_model.norm.weight", (h,))
    for i, kind in enumerate(kinds):
        p, q = f"language_model.decoder.layers.{i}.", f"model.language_model.layers.{i}."
        pair(p + "mlp.linear_fc1.layer_norm_weight", q + "post_attention_layernorm.weight", (h,))
        pair(p + "mlp.linear_fc2.weight", q + "mlp.down_proj.weight", (h, f))
        hf[q + "mlp.gate_proj.weight"] = hf[q + "mlp.up_proj.weight"] = (f, h)
        dcp[p + "mlp.linear_fc1.weight"] = (2 * f, h)
        a = p + "self_attention."
        if kind == "full_attention":
            pair(a + "linear_qkv.layer_norm_weight", q + "input_layernorm.weight", (h,))
            pair(a + "q_layernorm.weight", q + "self_attn.q_norm.weight", (d,))
            pair(a + "k_layernorm.weight", q + "self_attn.k_norm.weight", (d,))
            pair(a + "linear_proj.weight", q + "self_attn.o_proj.weight", (h, heads * d))
            hf[q + "self_attn.q_proj.weight"] = (2 * heads * d, h)  # output gate
            hf[q + "self_attn.k_proj.weight"] = hf[q + "self_attn.v_proj.weight"] = (kv * d, h)
            dcp[a + "linear_qkv.weight"] = (2 * (heads + kv) * d, h)
        else:
            pair(a + "in_proj.layer_norm_weight", q + "input_layernorm.weight", (h,))
            pair(a + "A_log", q + "linear_attn.A_log", (vh,))
            pair(a + "dt_bias", q + "linear_attn.dt_bias", (vh,))
            pair(a + "out_norm.weight", q + "linear_attn.norm.weight", (vd,))
            pair(a + "out_proj.weight", q + "linear_attn.out_proj.weight", (h, vh * vd))
            hf[q + "linear_attn.in_proj_qkv.weight"] = (2 * kh * kd + vh * vd, h)
            hf[q + "linear_attn.in_proj_z.weight"] = (vh * vd, h)
            hf[q + "linear_attn.in_proj_b.weight"] = hf[q + "linear_attn.in_proj_a.weight"] = (vh, h)
            hf[q + "linear_attn.conv1d.weight"] = (2 * kh * kd + vh * vd, 1, kernel)
            for name, width in (("query", kh * kd), ("key", kh * kd), ("value", vh * vd),
                                ("z", vh * vd), ("beta", vh), ("alpha", vh)):
                dcp[a + "in_proj.weight." + name] = (width, h)
            for name, width in (("query", kh * kd), ("key", kh * kd), ("value", vh * vd)):
                dcp[a + "conv1d.weight." + name] = (width, 1, kernel)

    count = sum(math.prod(s) for s in dcp.values())
    _require(count == EXPECTED_LANGUAGE_PARAMETERS and count == sum(math.prod(s) for s in hf.values()),
             f"architecture-derived language element count differs: {count}")
    index_bytes = _local_file(model_path, "model.safetensors.index.json").read_bytes()
    index = json.loads(index_bytes)["weight_map"]
    indexed_lm = {k for k in index if k.startswith("model.language_model.") or k == "lm_head.weight"}
    _require(indexed_lm == set(hf), "local HF index does not contain exactly the architecture-derived language tensors")
    headers, header_receipts = {}, {}
    for name in sorted(set(index[k] for k in hf)):
        path = _local_file(model_path, name)
        with path.open("rb") as stream:
            length_bytes = stream.read(8)
            _require(len(length_bytes) == 8, "truncated safetensors header length")
            length = struct.unpack("<Q", length_bytes)[0]
            _require(0 < length <= 16 * 1024**2, "invalid/oversized safetensors header")
            raw = stream.read(length)
        _require(len(raw) == length, "truncated safetensors header")
        headers[name] = json.loads(raw)
        header_receipts[name] = {"header_blake3": blake3(length_bytes + raw).hexdigest(),
                                 "header_bytes": 8 + length, "file_bytes": path.stat().st_size}
    for key, shape in hf.items():
        item = headers[index[key]].get(key)
        # The published HF GDN A_log/norm tensors are FP32; our explicit BF16
        # Megatron provider converts those parameters to BF16 during import.
        source_dtype = "F32" if key.endswith(("linear_attn.A_log", "linear_attn.norm.weight")) else "BF16"
        _require(item is not None and tuple(item["shape"]) == shape and item["dtype"] == source_dtype,
                 f"HF architecture/header mismatch: {key}")
        offsets = item["data_offsets"]
        file_info = header_receipts[index[key]]
        _require(0 <= offsets[0] < offsets[1] <= file_info["file_bytes"] - file_info["header_bytes"] and
                 offsets[1] - offsets[0] == math.prod(shape) * (4 if source_dtype == "F32" else 2),
                 f"HF tensor byte size differs: {key}")
    return dcp, {"config_blake3": blake3(config_bytes).hexdigest(),
                 "hf_index_blake3": blake3(index_bytes).hexdigest(), "hf_headers": header_receipts,
                 "hf_language_tensors": len(hf), "expected_dcp_language_tensors": len(dcp),
                 "language_parameters": count, "expected_map_blake3": _digest(dcp),
                 "mapping_basis": "Qwen35Bridge dense LM mappings; Megatron GatedDeltaNet.sharded_state_dict split rules"}


def _checkpoint_directory(root: Path) -> tuple[Path, dict]:
    root = Path(root).resolve()
    if (root / ".metadata").is_file():
        return root, {"selection": "explicit_iteration_directory"}
    pointer = _local_file(root, "latest_checkpointed_iteration.txt")
    raw = pointer.read_bytes()
    value = raw.decode().strip()
    _require(re.fullmatch(r"[0-9]+", value) is not None, "checkpoint pointer is not a completed numeric iteration")
    chosen = root / f"iter_{int(value):07d}"
    _require(chosen.is_dir() and not chosen.is_symlink(), "checkpoint pointer target is missing/symlinked")
    return chosen, {"selection": "latest_checkpointed_iteration", "iteration": int(value),
                    "pointer_blake3": blake3(raw).hexdigest()}


def _validate_metadata(metadata: Any, expected: dict[str, tuple[int, ...]], directory: Path) -> dict:
    """Validate exact LM entries, chunk coverage and physical storage extents."""
    entries = metadata.state_dict_metadata
    storage = metadata.storage_data
    _require(isinstance(entries, dict) and isinstance(storage, dict), "invalid DCP metadata structure")
    language = {k: v for k, v in entries.items()
                if k.startswith("language_model.") and not k.startswith("language_model.mtp.")}
    tensors = {k: v for k, v in language.items() if hasattr(v, "size")}
    missing, extra = sorted(set(expected) - set(tensors)), sorted(set(tensors) - set(expected))
    _require(not missing and not extra, f"DCP language key mismatch: missing={missing[:8]}, unexpected={extra[:8]}")
    # TE extra-state objects are non-parameter data, not a blanket tensor exception.
    unknown = [k for k, v in language.items() if not hasattr(v, "size") and
               not re.fullmatch(r"language_model\..+\._extra_state/shard_[0-9_]+", k)]
    _require(not unknown, f"unknown non-tensor language entries: {unknown[:8]}")
    by_tensor: dict[str, dict[tuple[int, ...], Any]] = {}
    shard_info: dict[str, dict] = {}
    for key, info in storage.items():
        path = _local_file(directory, info.relative_path)
        stat = path.stat()
        _require(isinstance(info.offset, int) and isinstance(info.length, int) and
                 info.offset >= 0 and info.length > 0 and info.offset + info.length <= stat.st_size,
                 f"DCP shard extent missing/truncated: {key.fqn}")
        shard_info[info.relative_path] = {"file_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if key.fqn in tensors:
            _require(key.offset is not None, f"tensor storage has no offset: {key.fqn}")
            offset = tuple(key.offset)
            group = by_tensor.setdefault(key.fqn, {})
            _require(offset not in group, f"duplicate tensor storage chunk: {key.fqn}")
            group[offset] = info
    chunk_count = 0
    for key, shape in expected.items():
        item = tensors[key]
        _require(tuple(item.size) == shape, f"DCP language tensor shape differs: {key}: {tuple(item.size)} != {shape}")
        _require(str(item.properties.dtype) == "torch.bfloat16", f"DCP language dtype differs: {key}")
        chunks = list(item.chunks)
        _require(bool(chunks), f"DCP language tensor has no chunks: {key}")
        boxes = []
        for chunk in chunks:
            offsets, sizes = tuple(chunk.offsets), tuple(chunk.sizes)
            _require(len(offsets) == len(shape) == len(sizes) and
                     all(o >= 0 and s > 0 and o + s <= limit for o, s, limit in zip(offsets, sizes, shape)),
                     f"DCP chunk lies outside tensor: {key}")
            for old_offsets, old_sizes in boxes:
                _require(not all(o < oo + ss and oo < o + s for o, s, oo, ss in zip(offsets, sizes, old_offsets, old_sizes)),
                         f"DCP overlapping tensor chunks: {key}")
            boxes.append((offsets, sizes))
        _require(sum(math.prod(sizes) for _, sizes in boxes) == math.prod(shape), f"DCP incomplete tensor chunk coverage: {key}")
        _require(set(by_tensor.get(key, {})) == {offsets for offsets, _ in boxes}, f"DCP chunk/storage index mismatch: {key}")
        chunk_count += len(boxes)
    aux = {}
    for label, prefix in (("frozen_vision", "vision_model."), ("frozen_mtp", "language_model.mtp.")):
        values = [v for k, v in entries.items() if k.startswith(prefix) and hasattr(v, "size")]
        aux[label] = {"prefix": prefix, "stored_tensors": len(values),
                      "stored_elements": sum(math.prod(v.size) for v in values),
                      "completeness_checked": False, "exception": "frozen auxiliary; outside full-LM training scope"}
    return {"language_tensor_chunks": chunk_count, "shards": shard_info, "frozen_auxiliary_exceptions": aux,
            "language_extra_state_entries": len(language) - len(tensors), "storage_extent_entries": len(storage)}


def verify_checkpoint(root: Path, model_path: Path) -> dict:
    """Audit a *trusted locally produced* checkpoint before allowing GRPO load.

    Loading PyTorch .metadata executes pickle deserialization. The caller must
    control/trust this local checkpoint; this is not an untrusted-file verifier.
    Raises ValueError/FileNotFoundError on any failure; never permits partial LM.
    """
    expected, architecture = expected_language_map(Path(model_path))
    directory, selection = _checkpoint_directory(Path(root))
    metadata_path = _local_file(directory, ".metadata")
    before = metadata_path.stat()
    raw = metadata_path.read_bytes()
    _require(len(raw) <= 64 * 1024**2, "unexpectedly large DCP metadata")
    metadata = pickle.loads(raw)  # Trust boundary explicitly documented above.
    checked = _validate_metadata(metadata, expected, directory)
    after = metadata_path.stat()
    _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "DCP metadata changed during audit")
    for name, info in checked["shards"].items():
        stat = _local_file(directory, name).stat()
        _require((stat.st_size, stat.st_mtime_ns) == (info["file_bytes"], info["mtime_ns"]), "DCP shard changed during audit")
    if selection["selection"] == "latest_checkpointed_iteration":
        pointer = _local_file(Path(root).resolve(), "latest_checkpointed_iteration.txt")
        _require(blake3(pointer.read_bytes()).hexdigest() == selection["pointer_blake3"],
                 "checkpoint pointer changed during audit")
    _local_file(directory, "common.pt")
    return {"schema": SCHEMA, "valid": True, "checkpoint_directory": str(directory),
            "trusted_local_pickle_metadata": True, "metadata_blake3": blake3(raw).hexdigest(),
            "metadata_bytes": len(raw), "selection": selection, "architecture": architecture, **checked,
            "complete_language_key_shape_dtype_chunk_storage_audit": True,
            "tensor_payload_values_verified": False, "optimizer_completeness_verified": False,
            "gpu_allocations": 0, "full_model_instantiated": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--trusted-local-checkpoint", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(verify_checkpoint(args.checkpoint, args.model_path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
