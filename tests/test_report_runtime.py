"""Report CPU fixtures: no actual Torch import, model load, GPU, or references."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from blake3 import blake3
import pytest

from training.benchmark_models import report_runtime as report


class Tensor:
    def __init__(self, value): self.value = value
    def __getitem__(self, key): return Tensor(self.value[key])
    def tolist(self): return self.value
    def to(self, device): assert device == "cuda:0"; return self


class Tokenizer:
    img_start_id, img_end_id, img_pad_id, eos_token_id = 11, 12, 13, 99
    def from_list_format(self, items):
        self.image = items[0]["image"]
        self.items = items
        return "official-image-query-fixture"
    def apply_chat_template(self, conversation, **kwargs):
        self.conversation = conversation
        self.template_options = kwargs
        return Tensor([[10, 11, *[1000 + ord(c) for c in self.image], 13, 12, 14]])
    def decode(self, tokens, **kwargs):
        if tokens and min(tokens) >= 1000:
            return "".join(chr(t - 1000) for t in tokens)
        return "".join({201: "Synthetic generated findings fixture with sufficient plain English characters.",
                        202: "FinalWord", 99: "<|endoftext|>"}.get(t, "?") for t in tokens)


def fixture(tmp_path, monkeypatch):
    workspace = tmp_path / "actor"
    image = workspace / "inputs/CXP0001/images/01.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"public JPEG fixture, never opened by model mocks")
    manifest = image.parent.parent / "manifest.json"
    manifest.write_text(json.dumps({"case_id": "CXP0001", "task_id": "chexpert-plus-cxr-task", "image_count": 1, "image_files": ["01.jpg"]}))
    cache = tmp_path / "cache"
    assets = {}
    for key, repo, revision in (("report", report.MODEL_REPO, report.MODEL_REVISION),
                                ("report_vision", report.VISION_REPO, report.VISION_REVISION)):
        root = cache / key
        root.mkdir(parents=True)
        config = {"architectures": ["CheXagentForCausalLM"], "max_position_embeddings": 2048,
                  "visual": {"vision_model_name_or_path": report.VISION_REPO}} if key == "report" else {}
        (root / "config.json").write_text(json.dumps(config))
        if key == "report":
            for name in report.SOURCE_PINS:
                (root / name).write_bytes(b"inert synthetic source")
        assets[key] = {"repository": repo, "revision": revision, "path": str(root), "files": [
            {"path": p.name, "bytes": p.stat().st_size, "blake3": blake3(p.read_bytes()).hexdigest()}
            for p in root.iterdir()]}
    monkeypatch.setattr(report, "SOURCE_PINS", {name: blake3(b"inert synthetic source").hexdigest() for name in report.SOURCE_PINS})
    binding = {"workspace": workspace, "task": {"track": "report", "no_private_reference_access": True,
        "public_config": {"task_id": "chexpert-plus-cxr-task"}}, "model": assets["report"],
        "selected": {"CXP0001": [{"path": str(p.relative_to(workspace))} for p in (manifest, image)]}}
    return binding, assets, cache, image


@pytest.mark.parametrize("prompt,budget", [(None, 100), ("short", 100), ("Generate <|img|> report", 100), ("Generate a report", None), ("Generate a report", 513)])
def test_explicit_controls(prompt, budget):
    with pytest.raises(ValueError): report.validate_controls(prompt, budget)


def test_public_single_frontal_contract_no_extra_image_locator(tmp_path, monkeypatch):
    binding, assets, cache, image = fixture(tmp_path, monkeypatch)
    assert report.public_frontal_image(binding, "CXP0001") == image
    path = image.parent.parent / "manifest.json"
    data = json.loads(path.read_text())
    data["image_files"] = ["https://example.invalid/image.jpg"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="single-frontal"):
        report.public_frontal_image(binding, "CXP0001")


def test_assets_are_exact_and_source_tampering_is_rejected(tmp_path, monkeypatch):
    binding, assets, cache, _ = fixture(tmp_path, monkeypatch)
    assert report.validate_assets(binding, assets, cache)["report"] == Path(assets["report"]["path"])
    assets["report_vision"]["revision"] = "main"
    with pytest.raises(ValueError, match="revision"): report.validate_assets(binding, assets, cache)
    assets["report_vision"]["revision"] = report.VISION_REVISION
    (Path(assets["report"]["path"]) / "modeling_visual.py").write_bytes(b"altered synthetic code")
    with pytest.raises(ValueError): report.validate_assets(binding, assets, cache)


def test_official_template_and_actual_model_context(tmp_path):
    tokenizer = Tokenizer()
    components = SimpleNamespace(tokenizer=tokenizer, context_length=2048)
    image = tmp_path / "images/01.jpg"
    ids, request = report.make_request(components, image, "Generate the findings section.", 512)
    assert tokenizer.items == [{"image": str(image)}, {"text": "Generate the findings section."}]
    assert tokenizer.conversation[0] == {"from": "system", "value": "You are a helpful assistant."}
    assert tokenizer.template_options == {"add_generation_prompt": True, "return_tensors": "pt"}
    components.context_length = len(request["input_token_ids"]) + 511
    with pytest.raises(ValueError, match="context"):
        report.make_request(components, image, "Generate the findings section.", 512)


def test_decode_never_drops_a_valid_last_token_or_repairs_text():
    raw, candidate, evidence = report.decode_generated(Tokenizer(), [10, 11, 201, 202], [10, 11])
    assert candidate.endswith("FinalWord") and raw == candidate and not evidence["terminal_eos_removed"]
    raw, candidate, evidence = report.decode_generated(Tokenizer(), [10, 11, 201, 99], [10, 11])
    assert raw.endswith("<|endoftext|>") and not candidate.endswith("<|endoftext|>") and evidence["terminal_eos_removed"]
    assert report.format_observation("short")["basic_format_valid"] is False
    with pytest.raises(ValueError, match="prefix"):
        report.decode_generated(Tokenizer(), [11, 10, 201], [10, 11])


def test_load_uses_local_vision_exact_source_bf16_and_complete_state(tmp_path, monkeypatch):
    events = {}
    model = SimpleNamespace(device="cuda:0")
    model.to = lambda **kwargs: model
    model.eval = lambda: model
    config = SimpleNamespace(visual={"vision_model_name_or_path": report.VISION_REPO})
    def load(path, **kwargs): events.update(kwargs); return model, {}
    module = ModuleType("transformers")
    module.__version__ = "4.40.0"
    module.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())
    module.AutoConfig = SimpleNamespace(from_pretrained=lambda *a, **k: config)
    module.AutoModelForCausalLM = SimpleNamespace(from_pretrained=load)
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setattr(report.importlib.util, "find_spec", lambda _: SimpleNamespace(origin=str(tmp_path / "runtimes/chexagent-transformers440/transformers/__init__.py")))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    paths = {"report": tmp_path / "report", "report_vision": tmp_path / "vision"}
    report.load_components(paths, tmp_path, SimpleNamespace(bfloat16="BF16-fixture"))
    assert config.visual["vision_model_name_or_path"] == str(paths["report_vision"])
    assert events["local_files_only"] and events["use_safetensors"] and events["output_loading_info"]
    assert events["device_map"] == {"": "cuda:0"} and events["torch_dtype"] == "BF16-fixture"
    module.AutoModelForCausalLM.from_pretrained = lambda *a, **k: (model, {"missing_keys": ["fixture.weight"]})
    with pytest.raises(RuntimeError, match="completely"):
        report.load_components(paths, tmp_path, SimpleNamespace(bfloat16="BF16-fixture"))


def test_actual_generation_branch_with_cpu_mocks_preserves_source_text(tmp_path, monkeypatch):
    binding, assets, cache, image = fixture(tmp_path, monkeypatch)
    events = {}
    def generate(ids, **kwargs):
        events.update(kwargs)
        return Tensor([ids[0].tolist() + [201, 99]])
    components = SimpleNamespace(tokenizer=Tokenizer(), model=SimpleNamespace(generate=generate), context_length=2048)
    monkeypatch.setattr(report, "load_components", lambda *a: components)
    progress = []
    rows = report.run_report(binding, SimpleNamespace(inference_mode=nullcontext),
        output_root=binding["workspace"] / "outputs/job", assets=assets, cache_root=cache,
        prompt="Generate the findings section.", max_new_tokens=512, progress=progress.append)
    assert progress == rows and events == {"do_sample": False, "num_beams": 1, "temperature": 1.0,
                                           "top_p": 1.0, "use_cache": True, "max_new_tokens": 512}
    row = rows[0]
    assert row["clinical_score"] is None and row["private_reference_access"] is False and not row["final_submission_written"]
    raw = json.loads((binding["workspace"] / row["raw_output"]["path"]).read_text())
    assert raw["generated_token_ids"] == [201, 99]
    text = (binding["workspace"] / row["candidate_output"]["path"]).read_text()
    assert text == Tokenizer().decode([201]) and not text.endswith("\n")
    assert row["format_observation"]["native_score_computed"] is False and row["data_notice"]
