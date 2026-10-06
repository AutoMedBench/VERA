"""Prescribed LLaVA-Med analysis tool, with exact raw decodes and public inputs.

Call only in the host-admitted, isolated Transformers 4.36.2 worker after the
host verifies ALL vqa/vqa_vision/vqa_source manifest bytes. No model is loaded at
module import, and no actor Python or scoring/reference path is accepted here.

The pinned official loader/source and Mistral conversation template are reused.
An explicit montage includes every public image; this is the task-permitted
concatenation adaptation, not a claim of preserving full image resolution.
Length checks prevent the source's silent 2048-token multimodal truncation.
Qwen remains the coding orchestrator; this helper is a provided_analysis_tool.
The actor must still author its pipeline/postprocessor, calibration and final
submission. Candidate answers and smoke evidence stay in the job directory.

Original LLaVA-Med source: microsoft/LLaVA-Med at 30697ca50b5c29a8e955c99330b259776aef27b9.
Its original licenses/research-use notices remain in the pinned public cache.
"""
from __future__ import annotations

from collections.abc import Mapping
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import time
from types import SimpleNamespace

from blake3 import blake3

if __package__:
    from ._track_io import artifact, case_output, public_input
else:
    from _track_io import artifact, case_output, public_input

MODEL_REPO = "microsoft/llava-med-v1.5-mistral-7b"
MODEL_REVISION = "91bb16c122001ddc9cf1fd36ce1dae09448943a2"
VISION_REPO = "openai/clip-vit-large-patch14-336"
VISION_REVISION = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
SOURCE_REPO = "microsoft/LLaVA-Med"
SOURCE_REVISION = "30697ca50b5c29a8e955c99330b259776aef27b9"
PROMPT_VERSION = "eva.automed.vqa.answer-prefix-montage.v1"
CONVERSATION_TEMPLATE = "mistral_instruct"
MAX_IMAGE_PIXELS = 32 * 1024**2
MONTAGE_CELL = 336
PADDING_SEED = 0


def _read_json(path):
    if path.stat().st_size > 128 * 1024:
        raise ValueError("Public VQA JSON exceeds the bounded input contract")
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Duplicate public JSON key")
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)


def public_question(binding, case):
    path = public_input(binding, case, "question.json")
    question = _read_json(path)
    if not isinstance(question, dict) or question.get("question_id") != case:
        raise ValueError("Public VQA question identity mismatch")
    text, options, images = question.get("question"), question.get("options"), question.get("images")
    if not isinstance(text, str) or not text.strip() or len(text) > 16000:
        raise ValueError("Invalid public VQA question text")
    if not isinstance(options, dict) or not options or not set(options).issubset(set("ABCDE")):
        raise ValueError("Public options must use the benchmark A-E labels")
    if any(not isinstance(value, str) or not value.strip() or len(value) > 8000 for value in options.values()):
        raise ValueError("Invalid public VQA option text")
    if not isinstance(images, list) or not 1 <= len(images) <= 6 or any(not isinstance(p, str) for p in images):
        raise ValueError("VQA montage requires all 1..6 declared public images")
    paths = [public_input(binding, case, relative) for relative in images]
    return question, paths


def validate_assets(binding, assets, cache_root):
    """Validate identities/layout and small source bytes; host verifies weights."""
    workspace = Path(binding["workspace"]).resolve(strict=True)
    cache_root = Path(cache_root).resolve(strict=True)
    if cache_root.is_relative_to(workspace):
        raise ValueError("Public model cache must be outside actor-writable workspace")
    if binding["model"] != assets["vqa"]:
        raise ValueError("VQA binding differs from the host-verified asset manifest")
    paths = {}
    for key, repository, revision in (("vqa", MODEL_REPO, MODEL_REVISION),
                                      ("vqa_vision", VISION_REPO, VISION_REVISION),
                                      ("vqa_source", SOURCE_REPO, SOURCE_REVISION)):
        entry = assets[key]
        if (entry.get("repository"), entry.get("revision")) != (repository, revision):
            raise ValueError("Wrong prescribed VQA source/model/vision revision")
        path = Path(entry["path"]).resolve(strict=True)
        if not path.is_relative_to(cache_root) or path.is_relative_to(workspace):
            raise ValueError("VQA asset escapes the host-bound public cache")
        paths[key] = path
        for row in entry["files"]:
            relative = Path(row["path"])
            target = (path / relative).resolve(strict=True)
            # snapshot_download uses links into this same public repository's
            # blob directory. Accept those bound cache links, not arbitrary
            # outside targets or another repository's cache.
            blob_root = cache_root / "hf-cache" / ("models--" + repository.replace("/", "--")) / "blobs"
            allowed_target = target.is_relative_to(path) or (key != "vqa_source" and target.is_relative_to(blob_root))
            if relative.is_absolute() or ".." in relative.parts or not allowed_target:
                raise ValueError("Unconfined VQA asset file")
            if target.stat().st_size != row["bytes"]:
                raise ValueError("VQA asset size changed after host preflight")
            if relative.suffix in (".py", ".json"):
                if blake3(target.read_bytes()).hexdigest() != row["blake3"]:
                    raise ValueError("VQA source/config changed after host preflight")
    config = _read_json(paths["vqa"] / "config.json")
    if config.get("model_type") != "llava_mistral" or config.get("mm_vision_tower") != VISION_REPO:
        raise ValueError("Pinned LLaVA/CLIP configuration mismatch")
    if config.get("tokenizer_model_max_length") != 2048:
        raise ValueError("Pinned multimodal input truncation limit changed")
    for variable, expected in (("HF_HUB_CACHE", cache_root / "hf-cache"),
                               ("HF_HOME", cache_root / "runtime-cache")):
        if not os.environ.get(variable) or Path(os.environ[variable]).resolve() != expected:
            raise RuntimeError("VQA requires the isolated task-specific offline HF cache")
    if any(os.environ.get(key) != "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise RuntimeError("VQA requires explicit offline-only model loading")
    repo_cache = cache_root / "hf-cache" / ("models--" + VISION_REPO.replace("/", "--"))
    if (repo_cache / "refs/main").read_text().strip() != VISION_REVISION:
        raise ValueError("Offline CLIP main alias is not the pinned revision")
    if paths["vqa_vision"] != (repo_cache / "snapshots" / VISION_REVISION).resolve():
        raise ValueError("Offline CLIP resolver does not bind the verified vision snapshot")
    return paths


def load_components(paths, cache_root):
    """Actual loader; only invoked after the host's larger VQA GPU admission."""
    source = paths["vqa_source"]
    expected_origins = {
        "llava": source / "llava/__init__.py",
        "transformers": Path(cache_root).resolve() / "runtimes/llava-transformers436/transformers/__init__.py",
    }
    for name, expected in expected_origins.items():
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected.resolve():
            raise RuntimeError("Launch VQA with its isolated 4.36 overlay and pinned source PYTHONPATH")
    import transformers
    if transformers.__version__ != "4.36.2":
        raise RuntimeError("VQA requires the pinned Transformers 4.36.2 runtime")
    from llava.model.builder import load_pretrained_model
    from llava.conversation import conv_templates
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.mm_utils import tokenizer_image_token, process_images

    # Snapshot directory names are hashes, not model names. The original loader
    # branches on these substrings, so use the TRUE canonical model name.
    # Its device='cuda' path also avoids an incompatible low_cpu_mem_usage=False
    # + device_map combination. Its fp16 default is retained, not quantized.
    tokenizer, model, processor, context_length = load_pretrained_model(
        model_path=str(paths["vqa"]), model_base=None,
        model_name=MODEL_REPO.rsplit("/", 1)[1], load_8bit=False, load_4bit=False, device="cuda")
    model.eval()
    if str(model.device) not in ("cuda", "cuda:0"):
        raise RuntimeError("VQA analysis model did not load on the admitted CUDA device")
    return SimpleNamespace(tokenizer=tokenizer, model=model, processor=processor,
                           context_length=context_length, conv_templates=conv_templates,
                           image_token=DEFAULT_IMAGE_TOKEN, image_token_index=IMAGE_TOKEN_INDEX,
                           tokenizer_image_token=tokenizer_image_token, process_images=process_images)


def montage_all_images(paths):
    """Row-major fit/letterbox of every image, never a first-image-only shortcut."""
    from PIL import Image, ImageOps
    if not 1 <= len(paths) <= 6:
        raise ValueError("Expected all 1..6 public images")
    columns = math.ceil(math.sqrt(len(paths)))
    rows = math.ceil(len(paths) / columns)
    canvas = Image.new("RGB", (columns * MONTAGE_CELL, rows * MONTAGE_CELL), (127, 127, 127))
    layout = []
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Public VQA image exceeds the bounded pixel budget")
            original = list(image.size)
            tile = ImageOps.contain(image.convert("RGB"), (MONTAGE_CELL, MONTAGE_CELL), Image.Resampling.LANCZOS)
        x = (index % columns) * MONTAGE_CELL + (MONTAGE_CELL - tile.width) // 2
        y = (index // columns) * MONTAGE_CELL + (MONTAGE_CELL - tile.height) // 2
        canvas.paste(tile, (x, y))
        layout.append({"index": index, "original_size": original, "placed_box": [x, y, x + tile.width, y + tile.height]})
    return canvas, {"mode": "montage", "grid_columns": columns, "grid_rows": rows,
                    "canvas_size": list(canvas.size), "images": layout,
                    "full_resolution_preserved": False, "all_images_included": True}


def build_prompt(question, components):
    options = question["options"]
    text = "Question: " + question["question"] + "\nOptions:\n"
    text += "\n".join(f"{key}) {options[key]}" for key in sorted(options))
    text += "\nAnswer in the format Answer: <one option letter>. Do not add an explanation."
    # The fixed config uses one image token and no im_start/im_end wrapper.
    if getattr(components.model.config, "mm_use_im_start_end", False):
        raise ValueError("Unexpected image-token wrapper in fixed LLaVA configuration")
    conversation = components.conv_templates[CONVERSATION_TEMPLATE].copy()
    conversation.append_message(conversation.roles[0], components.image_token + "\n" + text)
    conversation.append_message(conversation.roles[1], None)
    return conversation.get_prompt()


def enforce_context_budget(input_ids, components, max_new_tokens):
    markers = int((input_ids == components.image_token_index).sum().item())
    if markers != 1:
        raise ValueError("Montage input must contain exactly one real image token")
    patches = int(components.model.get_vision_tower().num_patches)
    effective = int(input_ids.shape[-1]) - markers + patches
    limit = min(int(components.context_length),
                int(components.model.config.tokenizer_model_max_length),
                int(components.model.config.max_position_embeddings))
    if effective + max_new_tokens > limit:
        raise ValueError("VQA context would exceed the unchanged model limit; no text/image truncation allowed")
    return {"text_and_marker_tokens": int(input_ids.shape[-1]), "vision_tokens": patches,
            "effective_input_tokens": effective, "max_new_tokens": max_new_tokens,
            "conservative_context_limit": limit, "truncated": False}


def candidate_answer(question, raw, runtime_s):
    """A labelled provided postprocessor; unparsed actual output stays unparsed."""
    label = ""
    for pattern in (r"(?:FINAL\s+ANSWER|ANSWER|OPTION|CHOICE)\s*[:\-]?\s*\(?([A-E])\)?\b",
                    r"^\(?([A-E])\)?(?:[\).\s]|$)", r"\b([A-E])\b"):
        match = re.search(pattern, raw.upper(), flags=re.MULTILINE)
        if match and match.group(1) in question["options"]:
            label = match.group(1)
            break
    return {"question_id": question["question_id"], "predicted_label": label,
            "predicted_answer": question["options"].get(label, ""), "raw_model_output": raw,
            "model_name": MODEL_REPO, "runtime_s": runtime_s}


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def run_vqa(binding, torch, *, output_root, assets, cache_root, multi_image_mode, max_new_tokens, progress=None):
    """One actual fixed-model decode per selected case; no fallback predictions."""
    if binding["task"].get("track") != "vqa" or binding["task"].get("no_private_reference_access") is not True:
        raise ValueError("Expected a public-only VQA task")
    if multi_image_mode != "montage" or type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 256:
        raise ValueError("Explicit montage and a bounded 1..256 generation budget are required")
    selected = binding["selected"]
    if not 1 <= len(selected) <= 2005 or not set(selected).issubset(binding["task"]["case_ids"]):
        raise ValueError("VQA case selection exceeds the public task binding")
    output_root = Path(output_root).resolve(strict=True)
    workspace = Path(binding["workspace"]).resolve(strict=True)
    if not output_root.is_relative_to(workspace / "outputs"):
        raise ValueError("VQA job output must stay within the public workspace outputs")
    if any((output_root / case).exists() for case in selected) or (output_root / "smoke_forward.json").exists():
        raise FileExistsError("VQA output directory already contains inference artifacts")
    prepared = {case: public_question(binding, case) for case in selected}
    paths = validate_assets(binding, assets, cache_root)
    components = load_components(paths, cache_root)
    results = []
    with torch.inference_mode():
        for case, (question, image_paths) in prepared.items():
            started = time.monotonic()
            directory = case_output(binding, output_root, case)
            montage, layout = montage_all_images(image_paths)
            montage_path = directory / "model-input-montage.png"
            montage.save(montage_path)
            prompt = build_prompt(question, components)
            input_ids = components.tokenizer_image_token(prompt, components.tokenizer,
                         components.image_token_index, return_tensors="pt").unsqueeze(0)
            budget = enforce_context_budget(input_ids, components, max_new_tokens)
            # The unchanged official padding helper has a 0/1-pixel jitter.
            # Pin and disclose its seed, rather than modify upstream transforms.
            random.seed(PADDING_SEED)
            image_tensor = components.process_images([montage], components.processor, components.model.config)
            if isinstance(image_tensor, list):
                raise ValueError("One montage must produce a single fixed-size tensor batch")
            device = components.model.device
            request = {"prompt_version": PROMPT_VERSION, "prompt": prompt,
                       "conversation_template": CONVERSATION_TEMPLATE, "budget": budget,
                       "multi_image_mode": multi_image_mode, "montage_layout": layout,
                       "padding_random_seed": PADDING_SEED, "do_sample": False,
                       "num_beams": 1, "max_new_tokens": max_new_tokens,
                       "authored_by": "provided_analysis_tool"}
            _write_json(directory / "model-request.json", request)
            output_ids = components.model.generate(
                input_ids.to(device=device), images=image_tensor.to(device=device, dtype=torch.float16),
                do_sample=False, num_beams=1, max_new_tokens=max_new_tokens, use_cache=True)
            ids = output_ids.detach().cpu().tolist()
            if len(ids) != 1:
                raise ValueError("Expected one actual VQA generation sequence")
            # Official LlavaMistral.generate forwards inputs_embeds. Decode all
            # returned IDs as in the pinned script; never slice off prompt length.
            decoded = components.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            if len(decoded) != 1 or not isinstance(decoded[0], str):
                raise ValueError("Actual VQA decoder returned an invalid value")
            raw = decoded[0]
            elapsed = time.monotonic() - started
            answer = candidate_answer(question, raw, elapsed)
            response = {"generation_token_ids": ids[0], "raw_model_output": raw,
                        "runtime_s": elapsed, "model_name": MODEL_REPO,
                        "skip_special_tokens": True, "actual_generate_returned": True}
            _write_json(directory / "model-response.json", response)
            _write_json(directory / "answer.json", answer)
            # No padding, prefix fabrication, extra model call, or inferred
            # success for an empty/short decode. Native verification is separate.
            smoke_path = (binding["_publication"].auxiliary_path("smoke_forward.json")
                          if binding.get("_publication") else output_root / "smoke_forward.json")
            smoke_created = False
            if len(raw.strip()) >= 5 and not smoke_path.exists() and not (output_root / "smoke_forward.json").exists():
                _write_json(smoke_path, {"model_name": MODEL_REPO, "device": str(device),
                                        "wall_s": elapsed, "raw_output_sample": raw, "success": True})
                smoke_created = True
            record = {"case_id": case, "authored_by": "provided_analysis_tool",
                      "analysis_model": {"repository": MODEL_REPO, "revision": MODEL_REVISION,
                                         "vision_repository": VISION_REPO, "vision_revision": VISION_REVISION,
                                         "source_revision": SOURCE_REVISION, "dtype": "float16"},
                      "request": artifact(binding, directory / "model-request.json"),
                      "raw_output": artifact(binding, directory / "model-response.json"),
                      "candidate_output": artifact(binding, directory / "answer.json"),
                      "model_input": artifact(binding, montage_path),
                      "smoke_output": artifact(binding, smoke_path) if smoke_created else None,
                      "decoded_characters": len(raw), "parsed_label": answer["predicted_label"],
                      "runtime_s": elapsed, "clinical_score": None,
                      "private_reference_access": False, "model_loaded_once_for_batch": True,
                      "postprocessing": "provided_label_extraction_and_public_option_copy; not_actor_authored",
                      "final_submission_written": False,
                      "smoke_artifact_present": smoke_path.exists() or (output_root / "smoke_forward.json").exists()}
            results.append(record)
            if progress is not None:
                progress(record)
    return results
