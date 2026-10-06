"""The detached command preserves runtime selectors, not inherited secrets."""
import json
from pathlib import Path
import runpy
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_detached_command_crosses_clean_parent_environment(tmp_path):
    build = runpy.run_path(str(ROOT / "scripts/run_eva_rsi_loop_v1.py"))["detached_command"]
    runtime = {"EVA_HARNESS_ROOT": str(tmp_path / "harness with spaces"),
               "EVA_MEDRESEARCH_DATA_ROOT": str(tmp_path / "data's directory"),
               "EVA_GRPO_CONTEXT_PROFILE": "evamed-grpo-native-24576-v1",
               "EVA_SLIME_NATIVE_AUTH_PATH": str(tmp_path / "auth.json"),
               "EVA_SLIME_JUDGE_CONCURRENCY": "4",
               "EVA_SLIME_LOSS_MEMORY": "fp32-chunked-reuse-v1",
               "OPENAI_API_KEY": "not-a-real-key-do-not-copy"}
    log = tmp_path / "supervisor log.json"
    program = "import os,json; print(json.dumps({k:v for k,v in os.environ.items() if k.startswith('EVA_') or k=='OPENAI_API_KEY'}))"
    command = build([sys.executable, "-c", program], log, runtime)
    assert "not-a-real-key-do-not-copy" not in command
    subprocess.run(["/bin/bash", "-c", command], env={"PATH": "/usr/bin:/bin"}, check=True)
    assert json.loads(log.read_text()) == {k: v for k, v in runtime.items() if k != "OPENAI_API_KEY"}


def test_detached_command_no_selected_harness_keeps_default(tmp_path):
    build = runpy.run_path(str(ROOT / "scripts/run_eva_rsi_loop_v1.py"))["detached_command"]
    command = build([sys.executable, "--version"], tmp_path / "log", {})
    assert "EVA_HARNESS_ROOT=" not in command
