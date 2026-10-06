"""Offline prescribed CheXagent-2-3b report generation; no scorer/proxy model.

Host verifies ALL report/report_vision manifest bytes and admits a fresh isolated
Transformers4.40 worker before calling run_report(). BF16 weights alone occupy
6,281,493,504 bytes (3,140,746,752 parameters, including the vision tower); a
2048-token full KV cache adds ~640MiB. The proposed 12GiB allocator/24GiB free
headroom profile is NOT measured peak memory or a successful GPU canary.

Official model code/tokenizer/template are loaded unchanged from digest-checked
local source. Only the nested vision repository locator is rebound in memory to
the exact verified local snapshot. No CPU/quantized/model-family fallback.
Generated suffix IDs and raw decode are retained, and only an actual terminal
EOS is removed for report.txt. Short/invalid reports are never padded or retried.

Licenses: CheXagent MIT; XraySigLIP CC-BY-NC-4.0. Original CheXpert Plus/mirror
notices remain applicable; dataset mirror license unspecified. This is private
local research evaluation only, not permission to redistribute raw images or
reports. Nothing in this helper publishes assets or claims clinical scores.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import string
import time
from types import SimpleNamespace

from blake3 import blake3

if __package__:
    from ._track_io import artifact, case_output, public_input
else:
    from _track_io import artifact, case_output, public_input

MODEL_REPO = "StanfordAIMI/CheXagent-2-3b"
MODEL_REVISION = "8f19b53a2eceda4c33b0acec6c81fbc293ad80d0"
VISION_REPO = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"
VISION_REVISION = "f0edbf5d90dba44edb7f4f96d8663537cb0749bf"
SOURCE_PINS = {
    "configuration_chexagent.py": "ef8d1bd73e11011d6c3361b9ee68ef8d6bb3f26ea8ec0de6fa4debe85b9d732d",
    "modeling_chexagent.py": "d20a28da8023e24c9004f3dc42d0babc9ee2535a6ec18058b9dc257f19923eba",
    "modeling_visual.py": "f08026b568864cf6131ad2bcd77a4421f43c00570c721b5d155f429dd9a34dc8",
    "tokenization_chexagent.py": "2491a1e5542d52f031aff70acc51eaa1ac65be2f391735075b96c2a963e85918",
}
PROMPT_VERSION = "eva.automed.report.official-chexagent-template.v1"
DATA_NOTICE = "Private local evaluation only; mirror license unspecified; retain upstream CheXpert Plus notices; no raw redistribution"


def validate_controls(prompt, max_new_tokens):
    if not isinstance(prompt, str) or not 10 <= len(prompt) <= 1024 or not prompt.strip():
        raise ValueError("Report requires an explicit 10..1024 character prompt")
    if "<|" in prompt or "|>" in prompt or "\x00" in prompt:
        raise ValueError("Report prompt cannot inject tokenizer/image control tags")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 512:
        raise ValueError("Report requires explicit max_new_tokens in 1..512")


def public_frontal_image(binding, case):
    manifest_path = public_input(binding, case, "manifest.json")
    if manifest_path.stat().st_size > 16384:
        raise ValueError("Public report manifest is oversized")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("case_id") != case or manifest.get("task_id") != "chexpert-plus-cxr-task"
            or manifest.get("image_count") != 1 or manifest.get("image_files") != ["01.jpg"]):
        raise ValueError("Only the pinned single-frontal-JPEG report contract is supported")
    return public_input(binding, case, "images/01.jpg")


def validate_assets(binding, assets, cache_root):
    """Read only source/config metadata here; full weight hashes are host-owned."""
    workspace = Path(binding["workspace"]).resolve(strict=True)
    cache_root = Path(cache_root).resolve(strict=True)
    if cache_root.is_relative_to(workspace) or binding["model"] != assets["report"]:
        raise ValueError("Report requires a host-verified external public model cache")
    paths = {}
    for key, repo, revision in (("report", MODEL_REPO, MODEL_REVISION),
                                ("report_vision", VISION_REPO, VISION_REVISION)):
        entry = assets[key]
        if (entry.get("repository"), entry.get("revision")) != (repo, revision):
            raise ValueError("Wrong prescribed report model/vision revision")
        root = Path(entry["path"]).resolve(strict=True)
        if not root.is_relative_to(cache_root) or root.is_relative_to(workspace):
            raise ValueError("Report asset escapes the bound public cache")
        rows = {row["path"]: row for row in entry["files"]}
        if not rows or len(rows) != len(entry["files"]):
            raise ValueError("Report asset inventory must be nonempty and unique")
        blob_root = cache_root / "hf-cache" / ("models--" + repo.replace("/", "--")) / "blobs"
        for name, row in rows.items():
            relative = Path(name)
            target = (root / relative).resolve(strict=True)
            if relative.is_absolute() or ".." in relative.parts or not (target.is_relative_to(root) or target.is_relative_to(blob_root)):
                raise ValueError("Report asset path escapes its verified repository")
            if target.stat().st_size != row["bytes"]:
                raise ValueError("Report asset size changed after host preflight")
            if relative.suffix in (".py", ".json") and blake3(target.read_bytes()).hexdigest() != row["blake3"]:
                raise ValueError("Report source/config commitment changed")
        paths[key] = root
    for name, digest in SOURCE_PINS.items():
        if blake3((paths["report"] / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Custom CheXagent source is not the fixed official source")
    config = json.loads((paths["report"] / "config.json").read_text())
    if config.get("architectures") != ["CheXagentForCausalLM"] or config.get("max_position_embeddings") != 2048 or config.get("visual", {}).get("vision_model_name_or_path") != VISION_REPO:
        raise ValueError("Pinned CheXagent architecture/vision/context contract changed")
    return paths


def load_components(paths, cache_root, torch):
    """The real pinned loader, invoked only after host GPU/overlay admission."""
    expected = Path(cache_root) / "runtimes/chexagent-transformers440/transformers/__init__.py"
    spec = importlib.util.find_spec("transformers")
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected.resolve():
        raise RuntimeError("Report must use its isolated Transformers 4.40.0 overlay")
    if any(os.environ.get(key) != "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise RuntimeError("Report model loading requires explicit offline mode")
    import transformers
    if transformers.__version__ != "4.40.0":
        raise RuntimeError("CheXagent requires Transformers 4.40.0")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(paths["report"]), trust_remote_code=True,
                                                          local_files_only=True, use_fast=False)
    config = transformers.AutoConfig.from_pretrained(str(paths["report"]), trust_remote_code=True, local_files_only=True)
    config.name_or_path = str(paths["report"])
    # No on-disk config/source mutation or unpinned 'main' lookup by nested loaders.
    config.visual = {**config.visual, "vision_model_name_or_path": str(paths["report_vision"])}
    model, info = transformers.AutoModelForCausalLM.from_pretrained(str(paths["report"]), config=config,
        trust_remote_code=True, local_files_only=True, use_safetensors=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": "cuda:0"}, attn_implementation="eager", output_loading_info=True)
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError("CheXagent checkpoint did not load completely and exactly")
    model = model.to(dtype=torch.bfloat16).eval()
    if str(model.device) not in ("cuda", "cuda:0"):
        raise RuntimeError("Report model is not on its admitted CUDA device")
    return SimpleNamespace(tokenizer=tokenizer, model=model, context_length=2048)


def make_request(components, image_path, prompt, max_new_tokens):
    tokenizer = components.tokenizer
    query = tokenizer.from_list_format([{"image": str(image_path)}, {"text": prompt}])
    conversation = [{"from": "system", "value": "You are a helpful assistant."}, {"from": "human", "value": query}]
    ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt")
    tokens = ids[0].tolist()
    if tokens.count(tokenizer.img_start_id) != 1 or tokens.count(tokenizer.img_end_id) != 1:
        raise ValueError("Report input must contain exactly one bound image span")
    start, end = tokens.index(tokenizer.img_start_id), tokens.index(tokenizer.img_end_id)
    span = tokens[start + 1:end]
    if tokenizer.img_pad_id not in span or tokenizer.decode(span[:span.index(tokenizer.img_pad_id)]) != str(image_path):
        raise ValueError("Tokenizer image locator differs from the bound public image")
    if len(tokens) + max_new_tokens > components.context_length:
        raise ValueError("Report request exceeds actual 2048-token model context; no silent truncation")
    return ids, {"prompt_version": PROMPT_VERSION, "prompt": prompt, "conversation": conversation,
                 "input_token_ids": tokens, "input_token_count": len(tokens), "max_new_tokens": max_new_tokens,
                 "context_length": components.context_length, "do_sample": False, "num_beams": 1,
                 "temperature": 1.0, "top_p": 1.0, "use_cache": True}


def decode_generated(tokenizer, generated, prompt_tokens):
    if generated[:len(prompt_tokens)] != prompt_tokens:
        raise ValueError("CheXagent generation changed or omitted the input prefix")
    suffix = generated[len(prompt_tokens):]
    if not suffix:
        raise ValueError("CheXagent returned no generated tokens")
    eos = tokenizer.eos_token_id
    removed_eos = suffix[-1] == eos
    candidate_ids = suffix[:-1] if removed_eos else suffix
    raw = tokenizer.decode(suffix, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    candidate = tokenizer.decode(candidate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return raw, candidate, {"generated_token_ids": suffix, "terminal_eos_removed": removed_eos,
                             "generated_token_count": len(suffix)}


def format_observation(text):
    """Observable format facts only; native scorer remains authoritative."""
    alpha = sum(character.isalpha() for character in text)
    valid = 40 <= len(text) <= 8000 and alpha >= 20 and bool(text.strip()) and all(c in string.printable for c in text)
    return {"char_count": len(text), "alpha_char_count": alpha, "basic_format_valid": valid,
            "native_score_computed": False}


def run_report(binding, torch, *, output_root, assets, cache_root, prompt, max_new_tokens, progress=None):
    validate_controls(prompt, max_new_tokens)
    task = binding["task"]
    if task.get("track") != "report" or task.get("no_private_reference_access") is not True or task.get("public_config", {}).get("task_id") != "chexpert-plus-cxr-task":
        raise ValueError("Expected the public-only CheXpert Plus report task")
    if not 1 <= len(binding["selected"]) <= 32:
        raise ValueError("Report requires 1..32 selected public cases")
    paths = validate_assets(binding, assets, cache_root)
    inputs = {case: public_frontal_image(binding, case) for case in binding["selected"]}
    components = load_components(paths, cache_root, torch)
    results = []
    for case, image_path in inputs.items():
        started = time.monotonic()
        output = case_output(binding, output_root, case)
        ids, request = make_request(components, image_path, prompt, max_new_tokens)
        request_path = output / "generation-request.json"
        request_path.write_text(json.dumps(request, indent=2) + "\n")
        with torch.inference_mode():
            generated = components.model.generate(ids.to("cuda:0"), do_sample=False, num_beams=1,
                temperature=1.0, top_p=1.0, use_cache=True, max_new_tokens=max_new_tokens)[0].tolist()
        raw, candidate, evidence = decode_generated(components.tokenizer, generated, request["input_token_ids"])
        if evidence["generated_token_count"] > max_new_tokens:
            raise ValueError("CheXagent generated beyond the admitted output budget")
        raw_path, candidate_path = output / "raw-generation.json", output / "report.txt"
        raw_path.write_text(json.dumps({**evidence, "raw_decoded_text": raw}, ensure_ascii=False, indent=2) + "\n")
        candidate_path.write_text(candidate, encoding="utf-8")  # Exact text, no appended medical prose/newline.
        smoke_path = (binding["_publication"].auxiliary_path("smoke_forward.json")
                      if binding.get("_publication") else output.parent / "smoke_forward.json")
        smoke_created = not results
        if smoke_created:
            with smoke_path.open("x", encoding="utf-8") as stream:
                json.dump({"model_name": MODEL_REPO, "actual_generate_returned": True,
                           "device": "cuda:0", "wall_s": time.monotonic() - started,
                           "raw_output_sample": candidate, "success": format_observation(candidate)["basic_format_valid"],
                           "clinical_score": None}, stream, indent=2)
        result = {"case_id": case, "raw_output": artifact(binding, raw_path),
                  "candidate_output": artifact(binding, candidate_path), "request": artifact(binding, request_path),
                  "smoke_output": artifact(binding, smoke_path) if smoke_created else None,
                  "input_image": artifact(binding, image_path), "runtime_s": time.monotonic() - started,
                  "analysis_model": {"repository": MODEL_REPO, "revision": MODEL_REVISION,
                                     "vision_repository": VISION_REPO, "vision_revision": VISION_REVISION},
                  "source_blake3": blake3(Path(__file__).read_bytes()).hexdigest(),
                  "dtype": "bfloat16", "generated_token_count": evidence["generated_token_count"],
                  "format_observation": format_observation(candidate), "clinical_score": None,
                  "postprocessing": "only_actual_terminal_EOS_removed; provided_tool_not_actor_authored",
                  "private_reference_access": False, "final_submission_written": False, "data_notice": DATA_NOTICE}
        results.append(result)
        if progress is not None:
            progress(result)
    return results
