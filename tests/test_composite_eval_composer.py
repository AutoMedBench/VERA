"""Provider-free composer fixtures; no real benchmark or provider claims."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from training.eva_rsi import composite_eval
from training.eva_rsi.evidence import commitment
from test_composite_eval import _source


@pytest.fixture
def composer():
    path = Path(__file__).resolve().parents[1] / "scripts/compose_eva_rsi_evaluation_v1.py"
    spec = importlib.util.spec_from_file_location("composite_composer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sources(tmp_path):
    paths = {}
    for name in ("original", "supplement"):
        root = tmp_path / name
        root.mkdir()
        skill = root / "skill-content-binding.json"
        skill.write_text(json.dumps({"fixture": name}))
        index = {"benchmark_run_root": str(root / "benchmark"), "skill_selection": None}
        if name == "supplement":
            index["skill_content_identity"] = commitment(skill)
        path = root / "index.json"
        path.write_text(json.dumps(index))
        paths[name] = path
    return {"original_index": paths["original"], "supplement_index": paths["supplement"],
            "original_skill_binding": paths["original"].parent / "skill-content-binding.json",
            "output": tmp_path / "new-output/evaluation-index.json"}


def verifier_fixture(monkeypatch, composer):
    # Reuse the existing real outer verifier; only benchmark-native source boundaries are fixtures.
    stable = {"same_hf": True}
    values = {"original": _source("original", composite_eval.ORIGINAL, stable),
              "supplement": _source("supplement", composite_eval.SUPPLEMENTED, stable)}
    monkeypatch.setattr(composite_eval, "_source",
        lambda *a, source_name, **kw: deepcopy(values[source_name]))
    monkeypatch.setattr(composite_eval, "read_document", lambda p: {"tracks": []})
    monkeypatch.setattr(composite_eval, "read", lambda p: {"exact_final_model_path": "/fixture/hf"})
    monkeypatch.setattr(composite_eval, "_zero_turn_eligibility", lambda *a: {"eligible_for_supplement": True})
    assert composer.verify_composite_index is composite_eval.verify_composite_index
    return values


def test_default_no_write_then_exact_final_binding_and_exclusive_publish(tmp_path, monkeypatch, composer):
    args = sources(tmp_path)
    verifier_fixture(monkeypatch, composer)
    old_bytes = {name: Path(args[name]).read_bytes() for name in
                 ("original_index", "supplement_index", "original_skill_binding")}
    result = composer.compose(**args)
    assert result["valid"] and not result["published"] and result["provider_calls"] == 0
    assert not args["output"].parent.exists()
    result = composer.compose(**args, execute=True)
    doc = json.loads(args["output"].read_bytes())
    assert result["published"] and result["verified_track_count"] == 7
    assert doc["supplement_skill_content_binding"] == json.loads(
        args["supplement_index"].read_bytes())["skill_content_identity"]
    assert doc["track_sources"]["classification"] == "supplement"
    assert doc["track_sources"]["report"] == "original" and doc["skill_selection"] is None
    assert all(Path(args[name]).read_bytes() == data for name, data in old_bytes.items())
    with pytest.raises(ValueError, match="output_must_be_fresh"):
        composer.compose(**args, execute=True)


@pytest.mark.parametrize("failure", ["missing_index", "missing_final_binding", "selection", "invalid_proof"])
def test_invalid_inputs_never_publish(tmp_path, monkeypatch, composer, failure):
    args = sources(tmp_path)
    values = verifier_fixture(monkeypatch, composer)
    if failure == "missing_index":
        args["supplement_index"] = tmp_path / "not-finished.json"
    elif failure in ("missing_final_binding", "selection"):
        path = args["supplement_index"]
        doc = json.loads(path.read_bytes())
        if failure == "missing_final_binding":
            doc.pop("skill_content_identity")
        else:
            doc["skill_selection"] = {"path": "/different", "blake3": "different"}
        path.write_text(json.dumps(doc))
    else:
        values["supplement"]["checkpoint"]["stable"] = {"different_hf": True}
    with pytest.raises((ValueError, FileNotFoundError)):
        composer.compose(**args, execute=True)
    assert not args["output"].exists() and not args["output"].parent.exists()


def test_cli_parses_exact_arguments_without_publishing(tmp_path, monkeypatch, composer, capsys):
    args = sources(tmp_path)
    verifier_fixture(monkeypatch, composer)
    argv = [item for key, value in args.items() for item in ("--" + key.replace("_", "-"), str(value))]
    assert composer.main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["verified_feedback_count"] == 2 and result["published"] is False
    assert not args["output"].exists()
