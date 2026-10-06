from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_codex_sdk_capacity.py"


def _audit_module() -> ModuleType:
    name = "eva_test_codex_sdk_capacity"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module


def test_capacity_profiles_are_explicit_and_keep_eight_candidates_per_shard() -> None:
    audit = _audit_module()
    expected = {
        "64": (64, 128, 64, 8),
        "128": (128, 256, 128, 16),
        "256": (256, 512, 256, 32),
        "512": (512, 1_024, 512, 64),
    }
    assert tuple(audit.CAPACITY_PROFILES) == tuple(expected)
    for name, values in expected.items():
        profile = audit.CAPACITY_PROFILES[name]
        assert (
            profile.worker_width,
            profile.queue_capacity,
            profile.claim_batch_size,
            profile.app_server_shards,
        ) == values
        assert profile.candidate_turns_per_shard == 8
        assert profile.maximum_concurrent_model_turns == profile.worker_width * 3


def test_synthetic_probe_proves_reuse_isolation_mcp_and_parallel_tools(
    tmp_path: Path,
) -> None:
    audit = _audit_module()
    profile = audit.CapacityProfile("test", 8, 16, 8, 2)

    result = audit.benchmark_synthetic_profile(
        profile,
        cwd=tmp_path,
        turn_delay_seconds=0.001,
    )

    assert result["passed"] is True
    assert result["provider_calls_made"] == 0
    assert result["runtime_opens"] == result["runtime_closes"] == 2
    assert result["fresh_threads"] == result["candidate_mcp_inventories"] == 8
    assert result["resumed_threads"] == 0
    assert result["maximum_same_turn_parallel_tools_observed"] == 64
    assert result["actor_and_judge_read_only"] is True
    assert result["config_values_recorded"] is False


def test_app_server_command_is_credential_empty_and_control_plane_minimal(
    tmp_path: Path,
) -> None:
    audit = _audit_module()
    binary = tmp_path / "codex"
    binary.touch(mode=0o700)
    isolation_root = tmp_path / "isolated"
    isolation_root.mkdir(mode=0o700)

    command = audit._isolated_app_server_command(
        binary=binary,
        isolation_root=isolation_root,
        trace_prefix=None,
    )

    assert command[0] == "/usr/bin/env"
    assert "-i" in command
    assert "check_for_update_on_startup=false" in command
    assert "analytics.enabled=false" in command
    assert command[-4:] == ("app-server", "--strict-config", "--listen", "stdio://")
    assert not any("KEY=" in token or "TOKEN=" in token for token in command)


def test_network_trace_filter_excludes_non_inet_calls() -> None:
    audit = _audit_module()
    rows = (
        'connect(3, {sa_family=AF_UNIX, sun_path="/tmp/socket"}, 16) = 0',
        'connect(4, {sa_family=AF_INET, sin_port=htons(443)}, 16) = 0',
        'sendto(5, "dns", 3, 0, {sa_family=AF_INET6}, 28) = 3',
        'socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) = 6',
    )

    assert audit._inet_trace_calls(rows) == rows[1:3]


def test_recommendation_selects_signed_512_profile_on_large_host() -> None:
    audit = _audit_module()
    profiles = {
        name: {
            "supported": True,
            "worker_width": profile.worker_width,
            "app_server_shards": profile.app_server_shards,
        }
        for name, profile in audit.CAPACITY_PROFILES.items()
    }
    recommendation = audit.recommend_profile(
        host={
            "memory_available_gib": 700,
            "cpu_affinity": 72,
            "hard_nofile_supports_production": True,
        },
        production={"profiles": profiles},
        synthetic=tuple(
            {"name": name, "passed": True} for name in audit.CAPACITY_PROFILES
        ),
        app_server={"provider_request_boundary_pass": True},
    )

    assert recommendation["target_profile"] == "512"
    assert recommendation["worker_width"] == 512
    assert recommendation["app_server_shards"] == 64
    assert recommendation["provider_turn_ceiling"] == 1_536
    assert recommendation["candidate_turns_per_shard"] == 8


def test_exclusive_receipt_writer_is_read_only_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    audit = _audit_module()
    destination = tmp_path / "audit.json"
    core = {
        "schema": audit.AUDIT_SCHEMA,
        "provider_calls_made": 0,
        "capability_gates": {"fresh_threads": True},
        "synthetic_profile_results": [
            {"passed": True, "provider_calls_made": 0}
        ],
        "passed": True,
    }
    document = {**core, "audit_blake3": audit.blake3_hex(core)}
    payload = audit.canonical_json_bytes(document) + b"\n"
    audit._write_exclusive(destination, payload)

    assert json.loads(destination.read_bytes()) == document
    assert stat.S_IMODE(destination.stat().st_mode) == 0o400
    assert audit.verify_audit_document(destination)["passed"] is True
    with pytest.raises(FileExistsError):
        audit._write_exclusive(destination, b"replacement\n")
    os.chmod(destination, 0o600)
