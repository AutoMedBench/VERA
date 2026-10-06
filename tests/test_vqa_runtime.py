"""Mocked CPU VQA integration: never load a real model, CUDA, provider or gold."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from blake3 import blake3
import numpy as np
from PIL import Image
import pytest

from training.benchmark_models import vqa_runtime as runtime


class Tensor:
    def __init__(self, value): self.value = np.asarray(value)
    @property
    def shape(self): return self.value.shape
    def __eq__(self, value): return Tensor(self.value == value)
    def sum(self): return Tensor(self.value.sum())
    def item(self): return self.value.item()
    def unsqueeze(self, axis): return Tensor(np.expand_dims(self.value, axis))
    def to(self, **kwargs): return self
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return self.value.tolist()


class Conversation:
    roles = ("USER", "ASSISTANT")
    def __init__(self): self.messages = []
    def copy(self): return Conversation()
    def append_message(self, role, message): self.messages.append((role, message))
    def get_prompt(self): return "[INST] " + self.messages[0][1] + " [/INST]"


def fixture(tmp_path, *, image_count=2):
    workspace = tmp_path / "actor"
    case = "MM-0"
    case_dir = workspace / "inputs" / case
    case_dir.mkdir(parents=True)
    image_names = []
    for index in range(image_count):
        name = f"image{index}.png"
        Image.new("RGB", (30 + index, 20 + index), (index * 30, 120, 200)).save(case_dir / name)
        image_names.append(name)
    question = {"question_id": case, "question": "Which labelled option matches the public fixture?",
                "options": {"A": "First option", "B": "Second option", "C": "Third option"}, "images": image_names}
    (case_dir / "question.json").write_text(json.dumps(question))
    selected = {case: [{"path": str(path.relative_to(workspace)), "bytes": path.stat().st_size,
                       "blake3": blake3(path.read_bytes()).hexdigest()} for path in case_dir.iterdir()]}
    binding = {"workspace": workspace, "task": {"track": "vqa", "no_private_reference_access": True,
               "case_ids": [case]}, "selected": selected, "model": {"unused_mock": True}}
    output = workspace / "outputs/job"
    output.mkdir(parents=True)
    return binding, output, question


def components(raw="  Answer: B\n", token_count=200):
    calls = {"generate": [], "images": [], "decoded": []}
    def generate(*args, **kwargs):
        calls["generate"].append((args, kwargs))
        return Tensor([[1, 52, 2]])
    def decode(ids, **kwargs):
        calls["decoded"].append(ids.tolist())
        return [raw]
    def process(images, *args):
        calls["images"].append(images)
        return Tensor(np.zeros((1, 3, 336, 336), dtype=np.float16))
    cfg = SimpleNamespace(tokenizer_model_max_length=2048, max_position_embeddings=32768,
                          mm_use_im_start_end=False)
    value = SimpleNamespace(
        model=SimpleNamespace(config=cfg, device="cuda:0", generate=generate,
                              get_vision_tower=lambda: SimpleNamespace(num_patches=576)),
        tokenizer=SimpleNamespace(batch_decode=decode), processor=object(), context_length=2048,
        image_token="<image>", image_token_index=-200,
        tokenizer_image_token=lambda *args, **kwargs: Tensor([-200] + [5] * (token_count - 1)),
        process_images=process, conv_templates={runtime.CONVERSATION_TEMPLATE: Conversation()})
    return value, calls


def test_montage_includes_every_image_in_order(tmp_path):
    binding, _, _ = fixture(tmp_path, image_count=6)
    _, paths = runtime.public_question(binding, "MM-0")
    montage, info = runtime.montage_all_images(paths)
    assert montage.size == (1008, 672)
    assert len(info["images"]) == 6
    assert info["all_images_included"] is True and info["full_resolution_preserved"] is False
    for index, image in enumerate(info["images"]):
        x0, y0, x1, y1 = image["placed_box"]
        assert montage.getpixel(((x0 + x1) // 2, (y0 + y1) // 2)) == (index * 30, 120, 200)


def test_public_images_cannot_escape_or_use_undeclared_input(tmp_path):
    binding, _, _ = fixture(tmp_path)
    path = binding["workspace"] / "inputs/MM-0/question.json"
    value = json.loads(path.read_text())
    value["images"] = ["../../not-declared.png"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="committed public"):
        runtime.public_question(binding, "MM-0")


def test_prompt_contains_all_options_and_exact_mistral_image_token(tmp_path):
    _, _, question = fixture(tmp_path)
    api, _ = components()
    prompt = runtime.build_prompt(question, api)
    assert prompt.startswith("[INST] <image>\nQuestion:")
    assert prompt.count("<image>") == 1
    assert all(f"{key}) {value}" in prompt for key, value in question["options"].items())
    assert "Answer: <one option letter>" in prompt


def test_budget_counts_vision_tokens_and_rejects_silent_truncation():
    api, _ = components()
    result = runtime.enforce_context_budget(Tensor([[-200] + [5] * 199]), api, 64)
    assert result["effective_input_tokens"] == 775
    assert result["truncated"] is False
    with pytest.raises(ValueError, match="no text/image truncation"):
        runtime.enforce_context_budget(Tensor([[-200] + [5] * 1499]), api, 64)
    with pytest.raises(ValueError, match="exactly one"):
        runtime.enforce_context_budget(Tensor([[-200, -200, 5]]), api, 64)


@pytest.mark.parametrize("raw,label", [("Answer: B", "B"), ("(C)", "C"), ("", ""), ("undetermined", "")])
def test_candidate_copies_public_option_or_remains_unparsed(raw, label):
    question = {"question_id": "MM-0", "options": {"A": "one", "B": "two", "C": "three"}}
    actual = runtime.candidate_answer(question, raw, 1.25)
    assert actual["raw_model_output"] == raw
    assert actual["predicted_label"] == label
    assert actual["predicted_answer"] == question["options"].get(label, "")


@pytest.mark.parametrize("raw,has_smoke", [("  Answer: B\n", True), ("C", False), ("", False)])
def test_real_path_with_mock_model_preserves_decode_and_never_fakes_smoke(tmp_path, monkeypatch, raw, has_smoke):
    binding, output, question = fixture(tmp_path)
    api, calls = components(raw=raw)
    loads = []
    monkeypatch.setattr(runtime, "validate_assets", lambda *args: {})
    monkeypatch.setattr(runtime, "load_components", lambda *args: loads.append(True) or api)
    observed = []
    result = runtime.run_vqa(binding, SimpleNamespace(inference_mode=nullcontext, float16="float16"),
                             output_root=output, assets={}, cache_root=tmp_path,
                             multi_image_mode="montage", max_new_tokens=64, progress=observed.append)
    assert len(loads) == len(calls["generate"]) == len(result) == 1
    assert observed == result
    args, kwargs = calls["generate"][0]
    assert kwargs["do_sample"] is False and kwargs["max_new_tokens"] == 64
    assert len(calls["images"][0]) == 1  # All originals are in this montage.
    assert calls["decoded"] == [[[1, 52, 2]]]  # Not sliced by the 200-token prompt.
    response = json.loads((output / "MM-0/model-response.json").read_text())
    answer = json.loads((output / "MM-0/answer.json").read_text())
    assert response["raw_model_output"] == answer["raw_model_output"] == raw
    assert response["generation_token_ids"] == [1, 52, 2]
    assert (output / "smoke_forward.json").exists() is has_smoke
    if has_smoke:
        smoke = json.loads((output / "smoke_forward.json").read_text())
        assert set(smoke) == {"model_name", "device", "wall_s", "raw_output_sample", "success"}
        assert smoke["raw_output_sample"] == raw
    assert result[0]["authored_by"] == "provided_analysis_tool"
    assert result[0]["clinical_score"] is None
    with pytest.raises(FileExistsError):
        runtime.run_vqa(binding, None, output_root=output, assets={}, cache_root=tmp_path,
                        multi_image_mode="montage", max_new_tokens=64)


def test_context_failure_has_no_model_decode_or_smoke(tmp_path, monkeypatch):
    binding, output, _ = fixture(tmp_path)
    api, calls = components(token_count=1900)
    monkeypatch.setattr(runtime, "validate_assets", lambda *args: {})
    monkeypatch.setattr(runtime, "load_components", lambda *args: api)
    with pytest.raises(ValueError, match="unchanged model limit"):
        runtime.run_vqa(binding, SimpleNamespace(inference_mode=nullcontext), output_root=output,
                        assets={}, cache_root=tmp_path, multi_image_mode="montage", max_new_tokens=64)
    assert not calls["generate"]
    assert not (output / "smoke_forward.json").exists()
    assert not (output / "MM-0/model-response.json").exists()


def asset_fixture(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    assets = {}
    for key, repo, revision in (("vqa", runtime.MODEL_REPO, runtime.MODEL_REVISION),
                                ("vqa_vision", runtime.VISION_REPO, runtime.VISION_REVISION),
                                ("vqa_source", runtime.SOURCE_REPO, runtime.SOURCE_REVISION)):
        base = root / "hf-cache" / ("models--" + repo.replace("/", "--"))
        path = (base / "snapshots" / revision) if key != "vqa_source" else root / "sources/LLaVA-Med"
        path.mkdir(parents=True)
        data = json.dumps({"model_type": "llava_mistral", "mm_vision_tower": runtime.VISION_REPO,
                           "tokenizer_model_max_length": 2048}).encode()
        (path / "config.json").write_bytes(data)
        assets[key] = {"repository": repo, "revision": revision, "path": str(path),
                       "files": [{"path": "config.json", "bytes": len(data), "blake3": blake3(data).hexdigest()}]}
        if key == "vqa_vision":
            (base / "refs").mkdir()
            (base / "refs/main").write_text(revision)
            blob = base / "blobs/fixture"
            blob.parent.mkdir()
            (path / "config.json").rename(blob)
            (path / "config.json").symlink_to(blob)
    for key, value in {"HF_HUB_CACHE": str(root / "hf-cache"), "HF_HOME": str(root / "runtime-cache"),
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}.items():
        monkeypatch.setenv(key, value)
    workspace = tmp_path / "actor"
    workspace.mkdir()
    return {"workspace": workspace, "model": assets["vqa"]}, assets, root


def test_exact_assets_allow_only_same_repo_hf_blob_links(tmp_path, monkeypatch):
    binding, assets, cache = asset_fixture(tmp_path, monkeypatch)
    paths = runtime.validate_assets(binding, assets, cache)
    assert set(paths) == {"vqa", "vqa_vision", "vqa_source"}
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    with pytest.raises(RuntimeError, match="offline-only"):
        runtime.validate_assets(binding, assets, cache)


def test_changed_vision_alias_or_source_config_is_rejected(tmp_path, monkeypatch):
    binding, assets, cache = asset_fixture(tmp_path, monkeypatch)
    ref = cache / "hf-cache" / ("models--" + runtime.VISION_REPO.replace("/", "--")) / "refs/main"
    ref.write_text("wrong")
    with pytest.raises(ValueError, match="alias"):
        runtime.validate_assets(binding, assets, cache)
    ref.write_text(runtime.VISION_REVISION)
    config = Path(assets["vqa_source"]["path"]) / "config.json"
    data = config.read_bytes()
    config.write_bytes(b"x" * len(data))
    with pytest.raises(ValueError, match="source/config changed"):
        runtime.validate_assets(binding, assets, cache)


def test_original_loader_receives_true_name_not_snapshot_hash(tmp_path, monkeypatch):
    source = tmp_path / "source"
    cache = tmp_path / "cache"
    model_path = tmp_path / runtime.MODEL_REVISION
    origins = {"llava": source / "llava/__init__.py",
               "transformers": cache / "runtimes/llava-transformers436/transformers/__init__.py"}
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(origins[name])))
    calls = []
    model = SimpleNamespace(device="cuda:0", eval=lambda: None)
    def loader(**kwargs):
        calls.append(kwargs)
        return object(), model, object(), 2048
    for name, module in {
        "transformers": SimpleNamespace(__version__="4.36.2"),
        "llava.model.builder": SimpleNamespace(load_pretrained_model=loader),
        "llava.conversation": SimpleNamespace(conv_templates={"mistral_instruct": object()}),
        "llava.constants": SimpleNamespace(IMAGE_TOKEN_INDEX=-200, DEFAULT_IMAGE_TOKEN="<image>"),
        "llava.mm_utils": SimpleNamespace(tokenizer_image_token=object(), process_images=object()),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    actual = runtime.load_components({"vqa_source": source, "vqa": model_path}, cache)
    assert actual.model is model
    assert calls == [{"model_path": str(model_path), "model_base": None,
                      "model_name": "llava-med-v1.5-mistral-7b", "load_8bit": False,
                      "load_4bit": False, "device": "cuda"}]
