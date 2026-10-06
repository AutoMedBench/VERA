from __future__ import annotations

import base64
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from eva_agent.deployment import CampaignDeploymentError
from eva_agent.codex_providers import (
    AdapterReceiptSigner,
    CANARY_PROMPT,
    CodexCanaryReceipt,
    receipts_document,
)
from eva_agent.pipeline.digests import (
    blake3_bytes,
    blake3_hex,
    canonical_json_bytes,
    is_blake3,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_campaign_v2.py"
DIRECT_CANARIES = ROOT / ".eva" / "codex-route-canaries-direct-v3.json"
ADAPTER_CANARY = ROOT / ".eva" / "codex-route-canaries-rollout-adapter-v3.json"
DIRECT_DOCUMENT_BLAKE3 = blake3_hex("direct-canaries-v2")
ADAPTER_DOCUMENT_BLAKE3 = blake3_hex("rollout-adapter-canaries-v2")
HOST_KEY_ID = "eva-test-host-v1"


def _launcher():
    name = "eva_test_campaign_v2_launcher"
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


def _write_config(tmp_path: Path, **changes) -> Path:
    launcher = _launcher()
    runs = tmp_path / "runs"
    runs.mkdir(parents=True)
    document = {
        "schema": launcher.LAUNCH_CONFIG_SCHEMA,
        "factory": launcher.REAL_FACTORY_SPEC,
        "ledger_path": str((runs / "campaign-v2.sqlite3").resolve()),
        "selection_blake3": blake3_hex("selection-v2"),
        "execution_binding_catalog_blake3": blake3_hex("v24-bindings"),
        "direct_canary_path": str(DIRECT_CANARIES.resolve()),
        "direct_canary_document_blake3": DIRECT_DOCUMENT_BLAKE3,
        "adapter_canary_path": str(ADAPTER_CANARY.resolve()),
        "adapter_canary_document_blake3": ADAPTER_DOCUMENT_BLAKE3,
        "adapter_canary_config_blake3": blake3_hex("adapter-config"),
        "adapter_signing_key_id": "eva-test-adapter-v1",
        "adapter_signing_public_key_blake3": blake3_hex("adapter-public-key"),
        **_host_trust_fields(tmp_path),
        "adapter_canary_acceptance_policy_blake3": (
            launcher.ADAPTER_CANARY_ACCEPTANCE_POLICY_BLAKE3
        ),
        "adapter_canary_acceptance_enforcement_blake3": (
            launcher.ADAPTER_CANARY_ACCEPTANCE_ENFORCEMENT_BLAKE3
        ),
        "production_factory_source_blake3": (
            launcher.REAL_FACTORY_SOURCE_BLAKE3
        ),
        "production_factory_package_blake3": (
            launcher.REAL_FACTORY_PACKAGE_BLAKE3
        ),
    }
    document.update(changes)
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    path.chmod(0o600)
    return path.resolve()


def _host_trust_fields(root: Path) -> dict[str, object]:
    material_root = root / "host-trust"
    material_root.mkdir(parents=True, exist_ok=True)
    private_path = material_root / "host-key.pem"
    trust_path = material_root / "host-trust.json"
    if private_path.exists():
        private = serialization.load_pem_private_key(
            private_path.read_bytes(), password=None
        )
        assert isinstance(private, Ed25519PrivateKey)
    else:
        private = Ed25519PrivateKey.generate()
        private_path.write_bytes(
            private.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        private_path.chmod(0o600)
    public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    trust_document = {
        "schema": "rlevo.med-research-host-trust-store.v1",
        "algorithm": "Ed25519",
        "created_at_utc": "2026-09-07T00:00:00Z",
        "keys": {HOST_KEY_ID: base64.b64encode(public).decode("ascii")},
        "status": "active",
    }
    trust_payload = canonical_json_bytes(trust_document)
    if not trust_path.exists() or trust_path.read_bytes() != trust_payload:
        trust_path.write_bytes(trust_payload)
    return {
        "host_private_key_path": str(private_path),
        "host_trust_store_path": str(trust_path),
        "host_key_id": HOST_KEY_ID,
        "host_public_key_blake3": blake3_bytes(public),
        "host_trust_store_blake3": blake3_bytes(trust_payload),
    }


def _trust_fields(launcher, adapter_document=None) -> dict[str, str]:
    if adapter_document is None:
        adapter_config_blake3 = blake3_hex("adapter-config")
        signing_key_id = "eva-test-adapter-v1"
        signing_public_key_blake3 = blake3_hex("adapter-public-key")
    else:
        signed_document = adapter_document["adapter_receipts"]
        first = signed_document["receipts"][0]
        adapter_config_blake3 = signed_document["adapter_config_blake3"]
        signing_key_id = first["key_id"]
        signing_public_key_blake3 = first["public_key_blake3"]
    return {
        "adapter_canary_config_blake3": adapter_config_blake3,
        "adapter_signing_key_id": signing_key_id,
        "adapter_signing_public_key_blake3": signing_public_key_blake3,
        "adapter_canary_acceptance_policy_blake3": (
            launcher.ADAPTER_CANARY_ACCEPTANCE_POLICY_BLAKE3
        ),
        "adapter_canary_acceptance_enforcement_blake3": (
            launcher.ADAPTER_CANARY_ACCEPTANCE_ENFORCEMENT_BLAKE3
        ),
        "production_factory_source_blake3": launcher.REAL_FACTORY_SOURCE_BLAKE3,
        "production_factory_package_blake3": (
            launcher.REAL_FACTORY_PACKAGE_BLAKE3
        ),
    }


def _launch_config(
    launcher,
    *,
    ledger_path: Path,
    direct_path: Path,
    direct_document_blake3: str,
    adapter_path: Path,
    adapter_document_blake3: str,
    adapter_document=None,
    trust_fields=None,
):
    host_trust_fields = _host_trust_fields(ledger_path.parent)
    return launcher.CampaignV2LaunchConfig(
        factory=launcher.REAL_FACTORY_SPEC,
        ledger_path=ledger_path.resolve(),
        selection_blake3=blake3_hex("selection-v2"),
        execution_binding_catalog_blake3=blake3_hex("v24-bindings"),
        direct_canary_path=direct_path.resolve(),
        direct_canary_document_blake3=direct_document_blake3,
        adapter_canary_path=adapter_path.resolve(),
        adapter_canary_document_blake3=adapter_document_blake3,
        host_private_key_path=Path(host_trust_fields["host_private_key_path"]),
        host_trust_store_path=Path(host_trust_fields["host_trust_store_path"]),
        host_key_id=host_trust_fields["host_key_id"],
        host_public_key_blake3=host_trust_fields["host_public_key_blake3"],
        host_trust_store_blake3=host_trust_fields["host_trust_store_blake3"],
        **(
            _trust_fields(launcher, adapter_document)
            if trust_fields is None
            else trust_fields
        ),
    )


def _canary_receipt(route_id: str, model_id: str, *, adapted: bool):
    return CodexCanaryReceipt.create(
        schema="eva.codex-responses-canary-receipt.v1",
        created_at_utc="2026-09-07T00:00:00Z",
        route_id=route_id,
        route_status="direct-pass",
        model_id=model_id,
        model_env_name=f"MODEL_{route_id.upper()}",
        registry_role_id=None,
        provider_id=(f"eva_adapter_{route_id}" if adapted else f"eva_{route_id}"),
        wire_api="responses",
        request_max_retries=0,
        stream_max_retries=0,
        semantic_retry_count=0,
        request_attempt_count=1,
        credential_env_name="NVIDIA_INFERENCE_API_KEY",
        endpoint_env_name="NVIDIA_INFERENCE_BASE_URL",
        credential_value_recorded=False,
        endpoint_value_recorded=False,
        raw_output_recorded=False,
        prompt_blake3=blake3_hex(CANARY_PROMPT),
        safe_config_blake3=blake3_hex({"route": route_id, "adapted": adapted}),
        safe_command_shape_blake3=blake3_hex("command"),
        status="passed",
        responses_accepted=True,
        semantic_exact_ok=True,
        no_tool_calls=True,
        turn_completed=True,
        exit_code=0,
        http_status=None,
        failure_class=None,
        latency_ms=1,
        json_event_count=1,
        malformed_jsonl_count=0,
        final_response_blake3=blake3_hex("EXACT_CANARY_OK"),
        diagnostic_blake3=blake3_hex("none"),
    )


def _replace_canary(receipt: CodexCanaryReceipt, **changes):
    values = receipt.to_dict()
    values.pop("receipt_blake3")
    values.update(changes)
    return CodexCanaryReceipt.create(**values)


def _canary_document(receipts, *, adapted: bool):
    route_ids = tuple(row.route_id for row in receipts)
    document = receipts_document(receipts)
    document.update(
        {
            "probe_transport": (
                "loopback-responses-to-chat-adapter"
                if adapted
                else "direct-responses"
            ),
            "effective_route_status": {
                route_id: "adapter-pass" if adapted else "direct-pass"
                for route_id in route_ids
            },
            "requested_routes": list(route_ids),
            "configured_routes": list(route_ids),
            "unconfigured_routes": [],
        }
    )
    if adapted:
        signer = AdapterReceiptSigner.ephemeral()
        families = {
            "deepseek_v4_flash": "deepseek",
            "gemini_3_1_pro": "google",
            "opus_5": "anthropic",
            "opus_4_8": "anthropic",
        }
        signed = [
            signer.sign(
                {
                    "schema": "eva.codex-responses-adapter-receipt-payload.v1",
                    "request_id": f"request-{row.route_id}",
                    "created_at_utc": "2026-09-07T00:00:00Z",
                    "route_id": row.route_id,
                    "model_id": row.model_id,
                    "provider_family": families[row.route_id],
                    "projection_version": (
                        "eva.codex-responses-chat-projection.v2-packed-namespaces"
                    ),
                    "status": "passed",
                    "failure_class": None,
                    "failure_message_blake3": None,
                    "adapter_http_status": 200,
                    "upstream_http_status": 200,
                    "upstream_latency_ms": 1,
                    "upstream_body_blake3": blake3_hex(
                        {"upstream": row.route_id}
                    ),
                    "request_shape": {
                        "parallel_tool_calls_requested": False,
                        "stream": True,
                        "input_item_count": 3,
                        "input_item_types": ["message", "message", "message"],
                        "function_call_output_count": 0,
                        "tool_count": 16,
                        "requested_max_output_tokens": None,
                        "effective_upstream_max_tokens": None,
                        "upstream_max_tokens_policy": "passthrough",
                    },
                    "response_shape": {
                        "parallel_function_call_count": 0,
                        "output_item_count": 1,
                        "output_item_types": ["message"],
                        "upstream_finish_reason": "stop",
                        "upstream_output_tokens": 1,
                        "upstream_reasoning_tokens": 0,
                    },
                    "request_max_retries": 0,
                    "stream_max_retries": 0,
                    "upstream_request_max_retries": 0,
                    "credential_value_recorded": False,
                    "endpoint_value_recorded": False,
                    "raw_request_recorded": False,
                    "raw_upstream_output_recorded": False,
                    "raw_response_recorded": False,
                }
            ).to_dict()
            for row in receipts
        ]
        safe_metadata = {
            "schema": "eva.codex-responses-adapter-config.v1",
            "projection_version": (
                "eva.codex-responses-chat-projection.v2-packed-namespaces"
            ),
            "route_bindings": [
                {
                    "route_id": row.route_id,
                    "model_id": row.model_id,
                    "provider_family": families[row.route_id],
                    "credential_env_name": row.credential_env_name,
                    "endpoint_env_name": row.endpoint_env_name,
                    "credential_value_recorded": False,
                    "endpoint_value_recorded": False,
                }
                for row in sorted(receipts, key=lambda item: item.route_id)
            ],
            "bind_host": "127.0.0.1",
            "max_concurrency": 32,
            "upstream_max_tokens_override": None,
            "request_max_retries": 0,
            "stream_max_retries": 0,
            "upstream_request_max_retries": 0,
            "loopback_token_recorded": False,
            "local_credential_env_name": receipts[0].credential_env_name,
            "raw_requests_recorded": False,
            "raw_upstream_outputs_recorded": False,
            "signing_key_id": signer.key_id,
            "signing_public_key_blake3": signer.public_key_blake3,
        }
        adapter_receipts = {
            "schema": "eva.codex-responses-adapter-receipts.v1",
            "adapter_config_blake3": blake3_hex(safe_metadata),
            "receipt_count": len(signed),
            "receipts": signed,
            "raw_requests_recorded": False,
            "raw_upstream_outputs_recorded": False,
            "credential_values_recorded": False,
            "endpoint_values_recorded": False,
        }
        adapter_receipts["document_blake3"] = blake3_hex(adapter_receipts)
        document["adapter_receipts"] = adapter_receipts
    document.pop("document_blake3")
    document["document_blake3"] = blake3_hex(document)
    return document


def test_profiles_are_explicit_canary_and_measured_ramp() -> None:
    launcher = _launcher()
    canary = launcher.concurrency_profile("canary")
    assert (
        canary.worker_width,
        canary.queue_capacity,
        canary.claim_batch_size,
        canary.app_server_shards,
        canary.max_candidates,
    ) == (1, 1, 1, 1, 1)
    expected = {
        "128": (128, 256, 128, 16),
        "256": (256, 512, 256, 32),
        "512": (512, 1_024, 512, 64),
    }
    for name, values in expected.items():
        profile = launcher.concurrency_profile(name)
        assert (
            profile.worker_width,
            profile.queue_capacity,
            profile.claim_batch_size,
            profile.app_server_shards,
        ) == values
        assert profile.maximum_parallel_models == 3
        assert profile.maximum_parallel_tools == 64
        assert profile.maximum_parallel_judge_tools == 64
        assert profile.required_process_soft_nofile == 65_536
        assert profile.max_candidates is None


def test_launch_config_requires_distinct_v2_ledger_and_exact_commitments(
    tmp_path: Path,
) -> None:
    launcher = _launcher()
    loaded = launcher.load_launch_config(_write_config(tmp_path))
    assert loaded.ledger_path.name == "campaign-v2.sqlite3"
    assert loaded.selection_blake3 == blake3_hex("selection-v2")

    legacy = _write_config(
        tmp_path / "legacy",
        ledger_path=str((tmp_path / "runs" / "campaign.sqlite3").resolve()),
    )
    with pytest.raises(CampaignDeploymentError, match="campaign-v2.sqlite3"):
        launcher.load_launch_config(legacy)


def test_launch_config_rejects_extra_keys_symlinks_and_legacy_factory(
    tmp_path: Path,
) -> None:
    launcher = _launcher()
    extra = _write_config(tmp_path / "extra", surprise=True)
    with pytest.raises(CampaignDeploymentError, match="keys differ"):
        launcher.load_launch_config(extra)

    legacy = _write_config(
        tmp_path / "factory", factory="scripts.run_campaign:build_components"
    )
    with pytest.raises(CampaignDeploymentError, match="REAL_FACTORY_SPEC"):
        launcher.load_launch_config(legacy)

    old_v2 = _write_config(
        tmp_path / "old-v2", schema="eva.campaign-v2-launch-config.v2"
    )
    with pytest.raises(CampaignDeploymentError, match="schema differs"):
        launcher.load_launch_config(old_v2)

    original = _write_config(tmp_path / "links")
    alias = tmp_path / "launch-link.json"
    alias.symlink_to(original)
    with pytest.raises(
        CampaignDeploymentError, match="normalized|non-symlink|unavailable|unsafe"
    ):
        launcher.load_launch_config(alias.absolute())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("ledger_path", "/tmp/eva-safe/../campaign-v2.sqlite3"),
        ("direct_canary_path", "/tmp/eva-safe/../direct.json"),
        ("adapter_canary_path", "/tmp/eva-safe/../adapter.json"),
        ("host_private_key_path", "/tmp/eva-safe/../host.pem"),
        ("host_trust_store_path", "/tmp/eva-safe/../trust.json"),
        ("direct_canary_path", "/tmp/eva-safe/\x00direct.json"),
    ),
)
def test_launch_config_rejects_traversal_and_nul_before_resolution(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    launcher = _launcher()
    config = _write_config(tmp_path / field.replace("_", "-"), **{field: value})

    with pytest.raises(CampaignDeploymentError, match="normalized|path differs"):
        launcher.load_launch_config(config)


def test_launch_config_rejects_relative_input_and_linked_parent(
    tmp_path: Path,
) -> None:
    launcher = _launcher()
    config = _write_config(tmp_path / "real")
    relative = Path(os.path.relpath(config, Path.cwd()))
    with pytest.raises(CampaignDeploymentError, match="absolute and normalized"):
        launcher.load_launch_config(relative)

    alias_parent = tmp_path / "linked-parent"
    alias_parent.symlink_to(config.parent, target_is_directory=True)
    with pytest.raises(CampaignDeploymentError, match="linked directory"):
        launcher.load_launch_config(alias_parent / config.name)


def test_all_trusted_json_rejects_top_level_and_nested_duplicate_keys(
    tmp_path: Path,
) -> None:
    launcher = _launcher()
    config = _write_config(tmp_path / "duplicate")
    raw = config.read_text(encoding="utf-8")
    config.write_text(
        raw.replace('"schema":', '"schema":"attacker","schema":', 1),
        encoding="utf-8",
    )
    config.chmod(0o600)
    with pytest.raises(CampaignDeploymentError, match="duplicate key"):
        launcher.load_launch_config(config)

    with pytest.raises(CampaignDeploymentError, match="duplicate key"):
        launcher._strict_json_object(
            b'{"outer":{"key":"first","key":"second"}}',
            label="nested contract",
        )


def test_provider_crossing_commands_require_exact_preflight_recipe() -> None:
    launcher = _launcher()
    actual = blake3_hex("recipe")
    launcher._expected_recipe(None, actual, command="preflight")
    with pytest.raises(CampaignDeploymentError, match="requires preflight"):
        launcher._expected_recipe(None, actual, command="canary")
    with pytest.raises(CampaignDeploymentError, match="changed after preflight"):
        launcher._expected_recipe(blake3_hex("other"), actual, command="production")
    launcher._expected_recipe(actual, actual, command="canary")


def test_ledger_and_canary_paths_reject_link_and_hardlink_topology(
    tmp_path: Path,
) -> None:
    launcher = _launcher()
    launch = launcher.load_launch_config(_write_config(tmp_path / "ledger-link"))
    ledger_target = launch.ledger_path.with_name("ledger-target.sqlite3")
    ledger_target.write_bytes(b"sqlite placeholder")
    launch.ledger_path.symlink_to(ledger_target)
    with pytest.raises(CampaignDeploymentError, match="unsafe"):
        launcher._assert_launch_trust(launch)

    canary = tmp_path / "canary.json"
    canary.write_text("{}", encoding="utf-8")
    canary_hardlink = tmp_path / "canary-hardlink.json"
    os.link(canary, canary_hardlink)
    with pytest.raises(CampaignDeploymentError, match="link count differs"):
        launcher._strict_document(canary, label="direct canary")


def test_exact_real_factory_module_and_callable_match_source() -> None:
    launcher = _launcher()
    factory = launcher.load_composition_factory(launcher.REAL_FACTORY_SPEC)
    assert factory.__module__ == "eva_agent.deployment.medresearch_v2"
    assert Path(launcher.inspect.getsourcefile(factory)) == (
        launcher.REAL_FACTORY_SOURCE_PATH
    )


def test_preflight_installs_schedule_without_starting_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    launcher = _launcher()
    launch = _launch_config(
        launcher,
        ledger_path=(tmp_path / "campaign-v2.sqlite3"),
        direct_path=DIRECT_CANARIES,
        direct_document_blake3=DIRECT_DOCUMENT_BLAKE3,
        adapter_path=ADAPTER_CANARY,
        adapter_document_blake3=ADAPTER_DOCUMENT_BLAKE3,
    )
    recipe = SimpleNamespace(
        selection_blake3=launch.selection_blake3,
        recipe_blake3=blake3_hex("recipe"),
        execution_binding_catalog_blake3=launch.execution_binding_catalog_blake3,
        executable_binding_count=1_344,
        build_provider_calls_made=0,
        concurrency={"worker_width": 1},
        turn_mcp_launch_blake3=blake3_hex("turn-mcp-launch"),
        turn_mcp_launch_metadata={
            "unix_socket_preflight": {
                "schema": "eva.turn-mcp-unix-socket-preflight.v1",
                "probed_socket_path_bytes": 56,
                "sockaddr_un_path_capacity_bytes": 108,
                "preflight_passed": True,
            }
        },
    )

    class FakeDeployment:
        def __init__(self):
            self.recipe = recipe
            self.ledger = SimpleNamespace(path=launch.ledger_path)
            self.ran = False

        def install_frozen_schedule(self, *, worker_width):
            assert worker_width == 1
            return 9_000

        def run_until_idle(self):
            self.ran = True
            raise AssertionError("provider-free preflight may not start runtime")

    deployment = FakeDeployment()
    monkeypatch.setattr(launcher, "load_launch_config", lambda _path: launch)
    monkeypatch.setattr(
        launcher,
        "build_from_launch_config",
        lambda _launch, *, profile, max_candidates=None: deployment,
    )
    args = SimpleNamespace(
        command="preflight",
        config=(tmp_path / "ignored.json"),
        profile="canary",
        expected_recipe_blake3=None,
    )
    assert launcher._run(args) == 0
    assert deployment.ran is False
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["installed_rows"] == 9_000
    assert receipt["runtime_started"] is False
    assert receipt["provider_calls_during_build"] == 0
    assert receipt["turn_mcp_launch_blake3"] == blake3_hex("turn-mcp-launch")
    assert receipt["turn_mcp_unix_socket_preflight"]["probed_socket_path_bytes"] == 56
    assert receipt["max_candidates"] is None
    assert receipt["launch_trust_blake3"] == launch.launch_trust_blake3
    assert receipt["production_factory_spec"] == launcher.REAL_FACTORY_SPEC
    assert receipt["production_factory_package_blake3"] == (
        launcher.REAL_FACTORY_PACKAGE_BLAKE3
    )
    assert receipt["host_key_id"] == HOST_KEY_ID
    assert receipt["host_public_key_blake3"] == launch.host_public_key_blake3
    assert receipt["host_trust_store_blake3"] == launch.host_trust_store_blake3
    receipt_text = json.dumps(receipt, sort_keys=True)
    assert str(launch.host_private_key_path) not in receipt_text
    assert str(launch.host_trust_store_path) not in receipt_text


def test_persisted_provider_health_is_strict_exact_and_qwen_absent() -> None:
    if not DIRECT_CANARIES.is_file() or not ADAPTER_CANARY.is_file():
        pytest.skip("local redacted provider canary receipts are not installed")
    launcher = _launcher()
    direct_document_blake3 = json.loads(
        DIRECT_CANARIES.read_text(encoding="utf-8")
    )["document_blake3"]
    adapter_document_blake3 = json.loads(
        ADAPTER_CANARY.read_text(encoding="utf-8")
    )["document_blake3"]
    adapter_document = json.loads(ADAPTER_CANARY.read_text(encoding="utf-8"))
    launch = _launch_config(
        launcher,
        ledger_path=(ROOT / "runs" / "campaign-v2.sqlite3"),
        direct_path=DIRECT_CANARIES,
        direct_document_blake3=direct_document_blake3,
        adapter_path=ADAPTER_CANARY,
        adapter_document_blake3=adapter_document_blake3,
        adapter_document=adapter_document,
    )
    health = launcher.load_verified_provider_health(launch)
    assert tuple(health) == (
        "deepseek_v4_flash",
        "gemini_3_1_pro",
        "opus_4_8",
        "gpt_5_6_sol",
        "opus_5",
    )
    assert all(
        health[route_id].status == "adapter-pass"
        for route_id in (
            "deepseek_v4_flash",
            "gemini_3_1_pro",
            "opus_5",
            "opus_4_8",
        )
    )
    assert all(
        health[route_id].status == "direct-pass"
        for route_id in ("gpt_5_6_sol",)
    )
    assert all(row.verified for row in health.values())
    assert "qwen_3_6_27b" not in health


def test_v2_health_requires_four_fresh_adapter_passes_and_one_direct_pass(
    tmp_path: Path,
) -> None:
    launcher = _launcher()
    direct_receipts = tuple(
        _canary_receipt(route_id, f"model/{route_id}", adapted=False)
        for route_id in ("gpt_5_6_sol",)
    )
    adapter_receipts = tuple(
        _canary_receipt(route_id, f"model/{route_id}", adapted=True)
        for route_id in (
            "deepseek_v4_flash",
            "gemini_3_1_pro",
            "opus_5",
            "opus_4_8",
        )
    )
    direct_document = _canary_document(direct_receipts, adapted=False)
    adapter_document = _canary_document(adapter_receipts, adapted=True)
    direct_path = (tmp_path / "direct-v2.json").resolve()
    adapter_path = (tmp_path / "adapter-v2.json").resolve()
    direct_path.write_text(json.dumps(direct_document), encoding="utf-8")
    adapter_path.write_text(json.dumps(adapter_document), encoding="utf-8")
    launch = _launch_config(
        launcher,
        ledger_path=(tmp_path / "campaign-v2.sqlite3"),
        direct_path=direct_path,
        direct_document_blake3=direct_document["document_blake3"],
        adapter_path=adapter_path,
        adapter_document_blake3=adapter_document["document_blake3"],
        adapter_document=adapter_document,
    )
    health = launcher.load_verified_provider_health(launch)
    assert tuple(health) == (
        "deepseek_v4_flash",
        "gemini_3_1_pro",
        "opus_4_8",
        "gpt_5_6_sol",
        "opus_5",
    )
    assert {
        route_id: row.status for route_id, row in health.items()
    } == {
        "deepseek_v4_flash": "adapter-pass",
        "gemini_3_1_pro": "adapter-pass",
        "opus_4_8": "adapter-pass",
        "gpt_5_6_sol": "direct-pass",
        "opus_5": "adapter-pass",
    }
    assert all(row.verified is True for row in health.values())
    for route_id in (
        "deepseek_v4_flash",
        "gemini_3_1_pro",
        "opus_5",
        "opus_4_8",
    ):
        assert health[route_id].canary_receipt_blake3 == launcher._adapter_health_commitment(
            launch=launch,
            canary_document_blake3=adapter_document["document_blake3"],
            canary={row.route_id: row for row in adapter_receipts}[route_id],
            signed_receipt=launcher._signed_adapter_receipt(
                next(
                    row
                    for row in adapter_document["adapter_receipts"]["receipts"]
                    if row["payload"]["route_id"] == route_id
                )
            ),
            adapter_safe_metadata=launcher._adapter_canary_safe_metadata(
                canaries={row.route_id: row for row in adapter_receipts},
                signed_receipts={
                    row["payload"]["route_id"]: launcher._signed_adapter_receipt(row)
                    for row in adapter_document["adapter_receipts"]["receipts"]
                },
            ),
        )

    adapter_document["effective_route_status"][
        "deepseek_v4_flash"
    ] = "direct-pass"
    adapter_document.pop("document_blake3")
    adapter_document["document_blake3"] = blake3_hex(adapter_document)
    adapter_path.write_text(json.dumps(adapter_document), encoding="utf-8")
    downgraded = _launch_config(
        launcher,
        ledger_path=launch.ledger_path,
        direct_path=direct_path,
        direct_document_blake3=direct_document["document_blake3"],
        adapter_path=adapter_path,
        adapter_document_blake3=adapter_document["document_blake3"],
        adapter_document=adapter_document,
    )
    with pytest.raises(CampaignDeploymentError, match="adapter canary inventory"):
        launcher.load_verified_provider_health(downgraded)


def test_adapter_semantic_probe_miss_keeps_original_failure_but_proves_transport(
    tmp_path: Path,
) -> None:
    launcher = _launcher()
    direct_receipts = tuple(
        _canary_receipt(route_id, f"model/{route_id}", adapted=False)
        for route_id in ("gpt_5_6_sol",)
    )
    adapter_receipts = list(
        _canary_receipt(route_id, f"model/{route_id}", adapted=True)
        for route_id in ("deepseek_v4_flash", "gemini_3_1_pro", "opus_5", "opus_4_8")
    )
    adapter_receipts[1] = _replace_canary(
        adapter_receipts[1],
        route_status="unavailable",
        status="semantic_failure",
        semantic_exact_ok=False,
        final_response_blake3=blake3_hex("a valid non-OK final response"),
    )
    direct_document = _canary_document(direct_receipts, adapted=False)
    adapter_document = _canary_document(tuple(adapter_receipts), adapted=True)
    adapter_document["effective_route_status"]["gemini_3_1_pro"] = (
        "adapter-failed"
    )
    adapter_document.pop("document_blake3")
    adapter_document["document_blake3"] = blake3_hex(adapter_document)
    direct_path = (tmp_path / "direct.json").resolve()
    adapter_path = (tmp_path / "adapter.json").resolve()
    direct_path.write_text(json.dumps(direct_document), encoding="utf-8")
    adapter_path.write_text(json.dumps(adapter_document), encoding="utf-8")
    launch = _launch_config(
        launcher,
        ledger_path=(tmp_path / "campaign-v2.sqlite3"),
        direct_path=direct_path,
        direct_document_blake3=direct_document["document_blake3"],
        adapter_path=adapter_path,
        adapter_document_blake3=adapter_document["document_blake3"],
        adapter_document=adapter_document,
    )

    health = launcher.load_verified_provider_health(launch)

    assert health["gemini_3_1_pro"].status == "adapter-pass"
    assert adapter_document["receipts"][1]["status"] == "semantic_failure"
    assert adapter_document["receipts"][1]["semantic_exact_ok"] is False
    assert launcher.ADAPTER_CANARY_ACCEPTANCE_POLICY[
        "semantic_exact_response_required"
    ] is False
    assert health["gemini_3_1_pro"].canary_receipt_blake3 not in {
        adapter_document["receipts"][1]["receipt_blake3"],
        adapter_document["adapter_receipts"]["receipts"][1]["envelope_blake3"],
    }


@pytest.mark.parametrize(
    ("changes", "match"),
    (
        ({"no_tool_calls": False}, "not production-ready"),
        ({"failure_class": "provider_or_transport_error"}, "not production-ready"),
        ({"json_event_count": 0}, "not production-ready"),
        ({"http_status": 503}, "not production-ready"),
        ({"latency_ms": -1}, "not production-ready"),
        ({"created_at_utc": "2026-09-07 00:00:00"}, "not production-ready"),
        ({"prompt_blake3": blake3_hex("different probe")}, "not production-ready"),
        (
            {"turn_completed": False, "responses_accepted": False},
            "not production-ready",
        ),
    ),
)
def test_adapter_semantic_probe_policy_rejects_structural_failures(
    tmp_path: Path, changes: dict[str, object], match: str
) -> None:
    launcher = _launcher()
    direct_receipts = tuple(
        _canary_receipt(route_id, f"model/{route_id}", adapted=False)
        for route_id in ("gpt_5_6_sol",)
    )
    adapter_receipts = list(
        _canary_receipt(route_id, f"model/{route_id}", adapted=True)
        for route_id in ("deepseek_v4_flash", "gemini_3_1_pro", "opus_5", "opus_4_8")
    )
    adapter_receipts[1] = _replace_canary(
        adapter_receipts[1],
        route_status="unavailable",
        status="semantic_failure",
        semantic_exact_ok=False,
        **changes,
    )
    direct_document = _canary_document(direct_receipts, adapted=False)
    adapter_document = _canary_document(tuple(adapter_receipts), adapted=True)
    adapter_document["effective_route_status"]["gemini_3_1_pro"] = (
        "adapter-failed"
    )
    adapter_document.pop("document_blake3")
    adapter_document["document_blake3"] = blake3_hex(adapter_document)
    direct_path = (tmp_path / "direct.json").resolve()
    adapter_path = (tmp_path / "adapter.json").resolve()
    direct_path.write_text(json.dumps(direct_document), encoding="utf-8")
    adapter_path.write_text(json.dumps(adapter_document), encoding="utf-8")
    launch = _launch_config(
        launcher,
        ledger_path=(tmp_path / "campaign-v2.sqlite3"),
        direct_path=direct_path,
        direct_document_blake3=direct_document["document_blake3"],
        adapter_path=adapter_path,
        adapter_document_blake3=adapter_document["document_blake3"],
        adapter_document=adapter_document,
    )

    with pytest.raises(CampaignDeploymentError, match=match):
        launcher.load_verified_provider_health(launch)


def test_direct_route_still_rejects_a_structurally_healthy_semantic_miss(
    tmp_path: Path,
) -> None:
    launcher = _launcher()
    direct_receipts = [
        _canary_receipt(route_id, f"model/{route_id}", adapted=False)
        for route_id in ("gpt_5_6_sol",)
    ]
    direct_receipts[0] = _replace_canary(
        direct_receipts[0],
        route_status="unavailable",
        status="semantic_failure",
        semantic_exact_ok=False,
    )
    adapter_receipts = tuple(
        _canary_receipt(route_id, f"model/{route_id}", adapted=True)
        for route_id in ("deepseek_v4_flash", "gemini_3_1_pro", "opus_5", "opus_4_8")
    )
    direct_document = _canary_document(tuple(direct_receipts), adapted=False)
    adapter_document = _canary_document(adapter_receipts, adapted=True)
    direct_path = (tmp_path / "direct.json").resolve()
    adapter_path = (tmp_path / "adapter.json").resolve()
    direct_path.write_text(json.dumps(direct_document), encoding="utf-8")
    adapter_path.write_text(json.dumps(adapter_document), encoding="utf-8")
    launch = _launch_config(
        launcher,
        ledger_path=(tmp_path / "campaign-v2.sqlite3"),
        direct_path=direct_path,
        direct_document_blake3=direct_document["document_blake3"],
        adapter_path=adapter_path,
        adapter_document_blake3=adapter_document["document_blake3"],
        adapter_document=adapter_document,
    )

    with pytest.raises(CampaignDeploymentError, match="direct provider route"):
        launcher.load_verified_provider_health(launch)


@pytest.mark.parametrize(
    "mutation",
    (
        "negative_upstream_latency",
        "bad_config_digest",
        "bad_signed_timestamp",
        "mixed_signing_identity",
        "duplicate_request_id",
        "extra_request_shape_key",
        "extra_response_shape_key",
        "extra_signed_document_key",
    ),
)
def test_adapter_signed_transport_evidence_must_be_structurally_valid(
    tmp_path: Path, mutation: str
) -> None:
    launcher = _launcher()
    direct_receipts = tuple(
        _canary_receipt(route_id, f"model/{route_id}", adapted=False)
        for route_id in ("gpt_5_6_sol",)
    )
    adapter_receipts = tuple(
        _canary_receipt(route_id, f"model/{route_id}", adapted=True)
        for route_id in ("deepseek_v4_flash", "gemini_3_1_pro", "opus_5", "opus_4_8")
    )
    direct_document = _canary_document(direct_receipts, adapted=False)
    adapter_document = _canary_document(adapter_receipts, adapted=True)
    trusted_fields = _trust_fields(launcher, adapter_document)
    signed_document = adapter_document["adapter_receipts"]
    if mutation == "bad_config_digest":
        signed_document["adapter_config_blake3"] = "not-a-digest"
    elif mutation == "extra_signed_document_key":
        signed_document["unexpected"] = False
    elif mutation == "mixed_signing_identity":
        original = signed_document["receipts"][0]
        signed_document["receipts"][0] = (
            AdapterReceiptSigner.ephemeral()
            .sign(dict(original["payload"]))
            .to_dict()
        )
    else:
        payloads = [dict(row["payload"]) for row in signed_document["receipts"]]
        if mutation == "negative_upstream_latency":
            payloads[0]["upstream_latency_ms"] = -1
        elif mutation == "duplicate_request_id":
            payloads[1]["request_id"] = payloads[0]["request_id"]
        elif mutation == "bad_signed_timestamp":
            payloads[0]["created_at_utc"] = "2026-09-07 00:00:00"
        elif mutation == "extra_request_shape_key":
            payloads[0]["request_shape"] = {
                **payloads[0]["request_shape"],
                "unexpected": False,
            }
        elif mutation == "extra_response_shape_key":
            payloads[0]["response_shape"] = {
                **payloads[0]["response_shape"],
                "unexpected": False,
            }
        signer = AdapterReceiptSigner.ephemeral()
        signed_document["receipts"] = [
            signer.sign(payload).to_dict() for payload in payloads
        ]
    signed_document.pop("document_blake3")
    signed_document["document_blake3"] = blake3_hex(signed_document)
    adapter_document.pop("document_blake3")
    adapter_document["document_blake3"] = blake3_hex(adapter_document)
    direct_path = (tmp_path / "direct.json").resolve()
    adapter_path = (tmp_path / "adapter.json").resolve()
    direct_path.write_text(json.dumps(direct_document), encoding="utf-8")
    adapter_path.write_text(json.dumps(adapter_document), encoding="utf-8")
    launch = _launch_config(
        launcher,
        ledger_path=(tmp_path / "campaign-v2.sqlite3"),
        direct_path=direct_path,
        direct_document_blake3=direct_document["document_blake3"],
        adapter_path=adapter_path,
        adapter_document_blake3=adapter_document["document_blake3"],
        trust_fields=trusted_fields,
    )

    with pytest.raises(CampaignDeploymentError, match="rollout adapter"):
        launcher.load_verified_provider_health(launch)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema", "eva.attacker-relabelled-envelope.v1"),
        ("signature_domain", "eva.attacker-controlled-domain.v1"),
        ("algorithm", "Ed448"),
    ),
)
def test_original_signed_envelope_literals_are_checked_before_reconstruction(
    field: str, value: str
) -> None:
    launcher = _launcher()
    canary = _canary_receipt(
        "opus_5", "model/opus_5", adapted=True
    )
    envelope = dict(
        _canary_document((canary,), adapted=True)["adapter_receipts"][
            "receipts"
        ][0]
    )
    # Keep the original signature and envelope commitment.  The old loader
    # discarded these two outer literals and reconstructed canonical values,
    # making this relabeling incorrectly verify.
    envelope[field] = value

    with pytest.raises(CampaignDeploymentError, match="envelope differs"):
        launcher._signed_adapter_receipt(envelope)


def test_valid_but_wrong_adapter_config_digest_cannot_be_operator_pinned(
    tmp_path: Path,
) -> None:
    launcher = _launcher()
    direct_receipts = tuple(
        _canary_receipt(route_id, f"model/{route_id}", adapted=False)
        for route_id in ("gpt_5_6_sol",)
    )
    adapter_receipts = tuple(
        _canary_receipt(route_id, f"model/{route_id}", adapted=True)
        for route_id in ("deepseek_v4_flash", "gemini_3_1_pro", "opus_5", "opus_4_8")
    )
    direct_document = _canary_document(direct_receipts, adapted=False)
    adapter_document = _canary_document(adapter_receipts, adapted=True)
    signed_document = adapter_document["adapter_receipts"]
    signed_document["adapter_config_blake3"] = blake3_hex(
        "valid-but-not-the-safe-gateway-config"
    )
    signed_document.pop("document_blake3")
    signed_document["document_blake3"] = blake3_hex(signed_document)
    adapter_document.pop("document_blake3")
    adapter_document["document_blake3"] = blake3_hex(adapter_document)
    direct_path = (tmp_path / "direct.json").resolve()
    adapter_path = (tmp_path / "adapter.json").resolve()
    direct_path.write_text(json.dumps(direct_document), encoding="utf-8")
    adapter_path.write_text(json.dumps(adapter_document), encoding="utf-8")
    launch = _launch_config(
        launcher,
        ledger_path=(tmp_path / "campaign-v2.sqlite3"),
        direct_path=direct_path,
        direct_document_blake3=direct_document["document_blake3"],
        adapter_path=adapter_path,
        adapter_document_blake3=adapter_document["document_blake3"],
        adapter_document=adapter_document,
    )

    with pytest.raises(CampaignDeploymentError, match="safe gateway config"):
        launcher.load_verified_provider_health(launch)


def test_fully_resigned_adapter_batch_is_rejected_by_launch_trust_anchor(
    tmp_path: Path,
) -> None:
    launcher = _launcher()
    direct_receipts = tuple(
        _canary_receipt(route_id, f"model/{route_id}", adapted=False)
        for route_id in ("gpt_5_6_sol",)
    )
    adapter_receipts = tuple(
        _canary_receipt(route_id, f"model/{route_id}", adapted=True)
        for route_id in ("deepseek_v4_flash", "gemini_3_1_pro", "opus_5", "opus_4_8")
    )
    direct_document = _canary_document(direct_receipts, adapted=False)
    adapter_document = _canary_document(adapter_receipts, adapted=True)
    trusted_fields = _trust_fields(launcher, adapter_document)
    signed_document = adapter_document["adapter_receipts"]
    attacker = AdapterReceiptSigner.ephemeral()
    signed_document["receipts"] = [
        attacker.sign(dict(row["payload"])).to_dict()
        for row in signed_document["receipts"]
    ]
    signed_document.pop("document_blake3")
    signed_document["document_blake3"] = blake3_hex(signed_document)
    adapter_document.pop("document_blake3")
    adapter_document["document_blake3"] = blake3_hex(adapter_document)
    direct_path = (tmp_path / "direct.json").resolve()
    adapter_path = (tmp_path / "adapter.json").resolve()
    direct_path.write_text(json.dumps(direct_document), encoding="utf-8")
    adapter_path.write_text(json.dumps(adapter_document), encoding="utf-8")
    launch = _launch_config(
        launcher,
        ledger_path=(tmp_path / "campaign-v2.sqlite3"),
        direct_path=direct_path,
        direct_document_blake3=direct_document["document_blake3"],
        adapter_path=adapter_path,
        adapter_document_blake3=adapter_document["document_blake3"],
        trust_fields=trusted_fields,
    )

    with pytest.raises(CampaignDeploymentError, match="signing inventory"):
        launcher.load_verified_provider_health(launch)


@pytest.mark.parametrize("document_name", ("direct", "adapter"))
def test_canary_outer_documents_reject_extra_keys(
    tmp_path: Path, document_name: str
) -> None:
    launcher = _launcher()
    direct_receipts = tuple(
        _canary_receipt(route_id, f"model/{route_id}", adapted=False)
        for route_id in ("gpt_5_6_sol",)
    )
    adapter_receipts = tuple(
        _canary_receipt(route_id, f"model/{route_id}", adapted=True)
        for route_id in ("deepseek_v4_flash", "gemini_3_1_pro", "opus_5", "opus_4_8")
    )
    direct_document = _canary_document(direct_receipts, adapted=False)
    adapter_document = _canary_document(adapter_receipts, adapted=True)
    trusted_fields = _trust_fields(launcher, adapter_document)
    target = direct_document if document_name == "direct" else adapter_document
    target["unexpected"] = False
    target.pop("document_blake3")
    target["document_blake3"] = blake3_hex(target)
    direct_path = (tmp_path / "direct.json").resolve()
    adapter_path = (tmp_path / "adapter.json").resolve()
    direct_path.write_text(json.dumps(direct_document), encoding="utf-8")
    adapter_path.write_text(json.dumps(adapter_document), encoding="utf-8")
    launch = _launch_config(
        launcher,
        ledger_path=(tmp_path / "campaign-v2.sqlite3"),
        direct_path=direct_path,
        direct_document_blake3=direct_document["document_blake3"],
        adapter_path=adapter_path,
        adapter_document_blake3=adapter_document["document_blake3"],
        trust_fields=trusted_fields,
    )

    with pytest.raises(CampaignDeploymentError, match="canary inventory differs"):
        launcher.load_verified_provider_health(launch)


def test_adapter_acceptance_enforcement_digest_binds_executable_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher()
    assert launcher.ADAPTER_CANARY_ACCEPTANCE_ENFORCEMENT_BLAKE3 == (
        launcher._adapter_acceptance_enforcement_blake3()
    )
    assert is_blake3(launcher.ADAPTER_CANARY_ACCEPTANCE_ENFORCEMENT_BLAKE3)
    canary = _canary_receipt(
        "gemini_3_1_pro", "model/gemini_3_1_pro", adapted=True
    )
    signed_document = _canary_document((canary,), adapted=True)["adapter_receipts"]
    signed = launcher._signed_adapter_receipt(signed_document["receipts"][0])
    launch = SimpleNamespace(
        launch_trust_blake3=blake3_hex("launch-trust"),
        adapter_canary_config_blake3=signed_document["adapter_config_blake3"],
        adapter_signing_key_id=signed.key_id,
        adapter_signing_public_key_blake3=signed.public_key_blake3,
    )
    safe_metadata = {"schema": "test-safe-metadata"}
    original = launcher._adapter_health_commitment(
        launch=launch,
        canary_document_blake3=blake3_hex("document"),
        canary=canary,
        signed_receipt=signed,
        adapter_safe_metadata=safe_metadata,
    )

    monkeypatch.setattr(
        launcher,
        "ADAPTER_CANARY_ACCEPTANCE_ENFORCEMENT_BLAKE3",
        blake3_hex("changed verifier source"),
    )

    assert launcher._adapter_health_commitment(
        launch=launch,
        canary_document_blake3=blake3_hex("document"),
        canary=canary,
        signed_receipt=signed,
        adapter_safe_metadata=safe_metadata,
    ) != original


def test_adapter_acceptance_enforcement_digest_is_import_name_independent() -> None:
    def load_as(name: str):
        spec = importlib.util.spec_from_file_location(name, SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(name, None)
        return module

    left = load_as("eva_campaign_v2_enforcement_left")
    right = load_as("eva_campaign_v2_enforcement_right")

    assert left.ADAPTER_CANARY_ACCEPTANCE_ENFORCEMENT_BLAKE3 == (
        right.ADAPTER_CANARY_ACCEPTANCE_ENFORCEMENT_BLAKE3
    )


def test_enforcement_digest_covers_strict_reader_and_transitive_verifier_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher()
    assert "strict_document" in inspect_enforcement_sources(launcher)
    assert set(launcher.ADAPTER_VERIFIER_DEPENDENCY_PATHS) == {
        "codex_providers.adapter",
        "codex_providers.canary",
        "codex_providers.routes",
        "pipeline.digests",
    }
    original_digest = launcher._adapter_acceptance_enforcement_blake3()
    original_getsource = launcher.inspect.getsource

    def changed_getsource(value):
        source = original_getsource(value)
        if value is launcher._strict_document:
            return source + "\n# simulated strict-reader drift\n"
        return source

    monkeypatch.setattr(launcher.inspect, "getsource", changed_getsource)
    assert launcher._adapter_acceptance_enforcement_blake3() != original_digest


def inspect_enforcement_sources(launcher) -> set[str]:
    """Test-only mirror of the stable logical enforcement labels."""

    source = launcher.inspect.getsource(
        launcher._adapter_acceptance_enforcement_blake3
    )
    return {
        label
        for label in (
            "strict_document",
            "provider_health_loader",
            "upstream_canary_verifier",
            "upstream_adapter_verifier",
        )
        if f'(\"{label}\",' in source
    }


def test_production_package_commitment_covers_runner_ledger_and_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher()
    observed: set[str] = set()

    def record(path: Path, *, label: str) -> str:
        del label
        observed.add(path.relative_to(launcher.REAL_FACTORY_PACKAGE_ROOT).as_posix())
        return blake3_hex(path.name)

    monkeypatch.setattr(launcher, "_literal_regular_file_blake3", record)
    assert is_blake3(
        launcher._source_package_blake3(launcher.REAL_FACTORY_PACKAGE_ROOT)
    )
    assert {
        "orchestration/runner.py",
        "orchestration/contracts.py",
        "campaign/ledger.py",
        "deployment/campaign.py",
    }.issubset(observed)


@pytest.mark.parametrize(
    "relative_path",
    (
        "orchestration/runner.py",
        "campaign/ledger.py",
        "orchestration/contracts.py",
    ),
)
def test_production_package_commitment_changes_on_execution_dependency_drift(
    tmp_path: Path, relative_path: str
) -> None:
    launcher = _launcher()
    package = tmp_path / "eva_agent"
    paths = tuple(
        package / value
        for value in (
            "deployment/medresearch_v2.py",
            "orchestration/runner.py",
            "campaign/ledger.py",
            "orchestration/contracts.py",
        )
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# committed {path.name}\n", encoding="utf-8")
    before = launcher._source_package_blake3(package)

    target = package / relative_path
    target.write_text(
        target.read_text(encoding="utf-8") + "# provider-boundary drift\n",
        encoding="utf-8",
    )

    assert launcher._source_package_blake3(package) != before


def test_production_package_commitment_rejects_symlink_topology(
    tmp_path: Path,
) -> None:
    launcher = _launcher()
    package = tmp_path / "eva_agent"
    package.mkdir()
    (package / "safe.py").write_text("# regular source\n", encoding="utf-8")
    (package / "alias.py").symlink_to(package / "safe.py")

    with pytest.raises(CampaignDeploymentError, match="topology differs"):
        launcher._source_package_blake3(package)


@pytest.mark.parametrize("shadow", ("cache.pyc", "cache.pyo", "payload.txt"))
def test_production_package_commitment_rejects_unexpected_file_shadows(
    tmp_path: Path,
    shadow: str,
) -> None:
    launcher = _launcher()
    package = tmp_path / "eva_agent"
    package.mkdir()
    (package / "safe.py").write_text("VALUE = 'safe'\n", encoding="utf-8")
    (package / shadow).write_bytes(b"uncommitted execution material")

    with pytest.raises(
        CampaignDeploymentError,
        match="executable source shadow|unexpected file",
    ):
        launcher._source_package_blake3(package)


def test_production_package_commitment_rejects_pycache_and_native_extension(
    tmp_path: Path,
) -> None:
    launcher = _launcher()
    package = tmp_path / "eva_agent"
    package.mkdir()
    (package / "safe.py").write_text("VALUE = 'safe'\n", encoding="utf-8")
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "safe.cpython-312.pyc").write_bytes(b"malicious bytecode")
    with pytest.raises(CampaignDeploymentError, match="executable source shadow"):
        launcher._source_package_blake3(package)

    (cache / "safe.cpython-312.pyc").unlink()
    cache.rmdir()
    extension = package / f"shadow{launcher.importlib.machinery.EXTENSION_SUFFIXES[0]}"
    extension.write_bytes(b"malicious extension")
    with pytest.raises(CampaignDeploymentError, match="executable source shadow"):
        launcher._source_package_blake3(package)


def test_loaded_module_and_factory_shadowing_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher()
    shadow = SimpleNamespace(__file__="/tmp/eva_agent/shadow.py")
    monkeypatch.setitem(sys.modules, "eva_agent.shadow", shadow)
    with pytest.raises(CampaignDeploymentError, match="origin differs"):
        launcher._assert_loaded_package_origins()
    monkeypatch.delitem(sys.modules, "eva_agent.shadow")

    def fake_factory(**_kwargs):
        raise AssertionError("shadow factory must never execute")

    fake_factory.__module__ = "eva_agent.deployment.medresearch_v2"
    fake_module = SimpleNamespace(
        __file__="/tmp/eva_agent/deployment/medresearch_v2.py",
        compose_campaign_v2=fake_factory,
    )
    monkeypatch.setattr(
        launcher.importlib,
        "import_module",
        lambda _name: fake_module,
    )
    with pytest.raises(CampaignDeploymentError, match="identity differs"):
        launcher.load_composition_factory(launcher.REAL_FACTORY_SPEC)


@pytest.mark.parametrize(
    ("field", "match"),
    (
        (
            "adapter_canary_acceptance_policy_blake3",
            "acceptance policy differs",
        ),
        (
            "adapter_canary_acceptance_enforcement_blake3",
            "acceptance enforcement differs",
        ),
        ("production_factory_source_blake3", "factory source differs"),
        ("production_factory_package_blake3", "factory package differs"),
    ),
)
def test_launch_config_rejects_unexpected_code_commitments(
    tmp_path: Path, field: str, match: str
) -> None:
    launcher = _launcher()
    config = _write_config(
        tmp_path / field, **{field: blake3_hex(f"wrong-{field}")}
    )
    with pytest.raises(CampaignDeploymentError, match=match):
        launcher.load_launch_config(config)


def test_jit_launch_gate_detects_enforcement_and_factory_package_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher()
    launch = launcher.load_launch_config(_write_config(tmp_path / "base"))

    monkeypatch.setattr(
        launcher,
        "_adapter_acceptance_enforcement_blake3",
        lambda: blake3_hex("changed-enforcement"),
    )
    with pytest.raises(CampaignDeploymentError, match="enforcement changed"):
        launcher._assert_launch_trust(launch)

    monkeypatch.setattr(
        launcher,
        "_adapter_acceptance_enforcement_blake3",
        lambda: launch.adapter_canary_acceptance_enforcement_blake3,
    )
    monkeypatch.setattr(
        launcher,
        "_source_package_blake3",
        lambda _root: blake3_hex("changed-production-package"),
    )
    with pytest.raises(CampaignDeploymentError, match="factory package changed"):
        launcher._assert_launch_trust(launch)


def test_host_signer_identity_is_recomputed_from_private_and_trust_bytes(
    tmp_path: Path,
) -> None:
    launcher = _launcher()
    launch = launcher.load_launch_config(_write_config(tmp_path / "valid"))
    launcher._reopen_host_signing_trust(launch)

    with pytest.raises(CampaignDeploymentError, match="public key commitment"):
        launcher._reopen_host_signing_trust(
            launcher.replace(
                launch,
                host_public_key_blake3=blake3_hex("attacker-public-key"),
            )
        )
    with pytest.raises(CampaignDeploymentError, match="trust store commitment"):
        launcher._reopen_host_signing_trust(
            launcher.replace(
                launch,
                host_trust_store_blake3=blake3_hex("attacker-trust-store"),
            )
        )
    with pytest.raises(CampaignDeploymentError, match="key ID is not trusted"):
        launcher._reopen_host_signing_trust(
            launcher.replace(launch, host_key_id="attacker-key-v1")
        )


def test_host_signer_rejects_changed_key_mode_bytes_links_and_symlinks(
    tmp_path: Path,
) -> None:
    launcher = _launcher()

    mode_launch = launcher.load_launch_config(_write_config(tmp_path / "mode"))
    mode_launch.host_private_key_path.chmod(0o644)
    with pytest.raises(CampaignDeploymentError, match="mode differs"):
        launcher._reopen_host_signing_trust(mode_launch)

    bytes_launch = launcher.load_launch_config(_write_config(tmp_path / "bytes"))
    replacement = Ed25519PrivateKey.generate()
    bytes_launch.host_private_key_path.write_bytes(
        replacement.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    bytes_launch.host_private_key_path.chmod(0o600)
    with pytest.raises(CampaignDeploymentError, match="public key commitment"):
        launcher._reopen_host_signing_trust(bytes_launch)

    link_launch = launcher.load_launch_config(_write_config(tmp_path / "hardlink"))
    hardlink = link_launch.host_private_key_path.with_name("host-key-hardlink.pem")
    os.link(link_launch.host_private_key_path, hardlink)
    with pytest.raises(CampaignDeploymentError, match="link count differs"):
        launcher._reopen_host_signing_trust(link_launch)

    symlink_launch = launcher.load_launch_config(_write_config(tmp_path / "symlink"))
    trust_alias = symlink_launch.host_trust_store_path.with_name("trust-alias.json")
    trust_alias.symlink_to(symlink_launch.host_trust_store_path)
    with pytest.raises(CampaignDeploymentError, match="unavailable|unsafe"):
        launcher._reopen_host_signing_trust(
            launcher.replace(symlink_launch, host_trust_store_path=trust_alias)
        )


def test_host_trust_store_duplicate_and_extra_keys_fail_even_when_repinned(
    tmp_path: Path,
) -> None:
    launcher = _launcher()
    duplicate_launch = launcher.load_launch_config(
        _write_config(tmp_path / "duplicate-trust")
    )
    encoded = json.loads(
        duplicate_launch.host_trust_store_path.read_text(encoding="utf-8")
    )["keys"][HOST_KEY_ID]
    duplicate_payload = (
        "{"
        '"algorithm":"Ed25519",'
        '"created_at_utc":"2026-09-07T00:00:00Z",'
        f'"keys":{{"{HOST_KEY_ID}":"{encoded}",'
        f'"{HOST_KEY_ID}":"{encoded}"}},'
        '"schema":"rlevo.med-research-host-trust-store.v1",'
        '"status":"active"'
        "}"
    ).encode("utf-8")
    duplicate_launch.host_trust_store_path.write_bytes(duplicate_payload)
    with pytest.raises(CampaignDeploymentError, match="duplicate key"):
        launcher._reopen_host_signing_trust(
            launcher.replace(
                duplicate_launch,
                host_trust_store_blake3=blake3_bytes(duplicate_payload),
            )
        )

    extra_launch = launcher.load_launch_config(_write_config(tmp_path / "extra-trust"))
    extra_document = json.loads(
        extra_launch.host_trust_store_path.read_text(encoding="utf-8")
    )
    extra_document["unexpected"] = False
    extra_payload = canonical_json_bytes(extra_document)
    extra_launch.host_trust_store_path.write_bytes(extra_payload)
    with pytest.raises(CampaignDeploymentError, match="keys differ"):
        launcher._reopen_host_signing_trust(
            launcher.replace(
                extra_launch,
                host_trust_store_blake3=blake3_bytes(extra_payload),
            )
        )


def test_launch_config_install_is_new_only_private_and_contains_no_secret(
    tmp_path: Path,
) -> None:
    launcher = _launcher()
    launch = _launch_config(
        launcher,
        ledger_path=(tmp_path / "campaign-v2.sqlite3"),
        direct_path=DIRECT_CANARIES,
        direct_document_blake3=DIRECT_DOCUMENT_BLAKE3,
        adapter_path=ADAPTER_CANARY,
        adapter_document_blake3=ADAPTER_DOCUMENT_BLAKE3,
    )
    output = tmp_path / "campaign-v2-launch.v3.json"
    digest = launcher._write_new_launch_config(output, launch)
    assert digest == blake3_hex(launch.to_document())
    assert os.stat(output).st_mode & 0o777 == 0o600
    raw = output.read_text(encoding="utf-8")
    assert "API_KEY" not in raw
    assert "TOKEN" not in raw
    assert launcher.load_launch_config(output.resolve()) == launch
    with pytest.raises(CampaignDeploymentError, match="already exists|unsafe"):
        launcher._write_new_launch_config(output, launch)


def test_launch_config_reader_and_writer_reject_mode_and_path_aliases(
    tmp_path: Path,
) -> None:
    launcher = _launcher()
    config = _write_config(tmp_path / "reader-mode")
    config.chmod(0o640)
    with pytest.raises(CampaignDeploymentError, match="mode differs"):
        launcher.load_launch_config(config)

    launch = _launch_config(
        launcher,
        ledger_path=tmp_path / "campaign-v2.sqlite3",
        direct_path=DIRECT_CANARIES,
        direct_document_blake3=DIRECT_DOCUMENT_BLAKE3,
        adapter_path=ADAPTER_CANARY,
        adapter_document_blake3=ADAPTER_DOCUMENT_BLAKE3,
    )
    with pytest.raises(CampaignDeploymentError, match="absolute and normalized"):
        launcher._write_new_launch_config(Path("relative-launch.json"), launch)
    traversal = tmp_path / "subdir" / ".." / "traversal-launch.json"
    with pytest.raises(CampaignDeploymentError, match="absolute and normalized"):
        launcher._write_new_launch_config(traversal, launch)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-output-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(CampaignDeploymentError, match="linked directory"):
        launcher._write_new_launch_config(linked_parent / "launch.json", launch)


def test_launch_receipt_is_committed_private_new_only_and_runs_scoped(
    tmp_path: Path,
) -> None:
    launcher = _launcher()
    runs = (tmp_path / "runs").resolve()
    runs.mkdir()
    output = (runs / "canary-receipt.v1.json").resolve()
    document = {
        "schema": "eva.campaign-v2-launch-receipt.v1",
        "command": "canary",
        "recipe_blake3": blake3_hex("recipe"),
    }
    document["document_blake3"] = blake3_hex(document)

    launcher._write_new_receipt(output, document, runs_root=runs)

    assert os.stat(output).st_mode & 0o777 == 0o600
    assert json.loads(output.read_text(encoding="utf-8")) == document
    with pytest.raises(CampaignDeploymentError, match="already exists|unsafe"):
        launcher._write_new_receipt(output, document, runs_root=runs)
    outside = (tmp_path / "outside.json").resolve()
    with pytest.raises(CampaignDeploymentError, match="under the runs root"):
        launcher._write_new_receipt(outside, document, runs_root=runs)
    with pytest.raises(CampaignDeploymentError, match="absolute and normalized"):
        launcher._write_new_receipt(
            Path("relative-receipt.json"), document, runs_root=runs
        )
    traversal = runs / "subdir" / ".." / "traversal-receipt.json"
    with pytest.raises(CampaignDeploymentError, match="absolute and normalized"):
        launcher._write_new_receipt(traversal, document, runs_root=runs)

    real_parent = runs / "real-parent"
    real_parent.mkdir()
    linked_parent = runs / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(CampaignDeploymentError, match="linked directory"):
        launcher._write_new_receipt(
            linked_parent / "receipt.json", document, runs_root=runs
        )


def test_all_launch_commands_accept_optional_durable_receipt_output() -> None:
    launcher = _launcher()
    parser = launcher.build_parser()
    config = str((ROOT / ".eva" / "campaign-v2-launch.v3.json").resolve())
    receipt = str((ROOT / "runs" / "receipt.json").resolve())
    for command in ("preflight", "canary", "production"):
        args = [command, "--config", config, "--receipt-output", receipt]
        if command == "canary":
            args += ["--expected-recipe-blake3", blake3_hex("recipe")]
        elif command == "production":
            args += [
                "--profile",
                "128",
                "--expected-recipe-blake3",
                blake3_hex("recipe"),
            ]
        parsed = parser.parse_args(args)
        assert parsed.receipt_output == Path(receipt)


def test_measured_ramp_candidate_cap_is_explicit_and_validated() -> None:
    launcher = _launcher()
    parser = launcher.build_parser()
    config = str((ROOT / ".eva" / "campaign-v2-launch.v3.json").resolve())
    recipe = blake3_hex("recipe")

    preflight = parser.parse_args(
        [
            "preflight",
            "--config",
            config,
            "--profile",
            "128",
            "--max-candidates",
            "16",
        ]
    )
    production = parser.parse_args(
        [
            "production",
            "--config",
            config,
            "--profile",
            "128",
            "--expected-recipe-blake3",
            recipe,
            "--max-candidates",
            "16",
        ]
    )
    assert preflight.max_candidates == production.max_candidates == 16

    for invalid in ("0", "9001", "not-an-int"):
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "production",
                    "--config",
                    config,
                    "--profile",
                    "128",
                    "--expected-recipe-blake3",
                    recipe,
                    "--max-candidates",
                    invalid,
                ]
            )


def test_compose_binds_candidate_cap_without_changing_profile_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher()
    launch = SimpleNamespace()
    observed = {}
    config = SimpleNamespace(concurrency=None, ledger_path=Path("/tmp/campaign-v2.sqlite3"))
    ports = SimpleNamespace()

    monkeypatch.setattr(launcher, "load_verified_provider_health", lambda _launch: {})
    monkeypatch.setattr(launcher, "_assert_launch_trust", lambda _launch: None)

    def factory(*, concurrency, provider_health, **host_trust):
        observed["concurrency"] = concurrency
        observed["host_trust"] = host_trust
        config.concurrency = concurrency
        return config, ports

    monkeypatch.setattr(launcher, "load_composition_factory", lambda _spec: factory)
    # The fixture swaps the loader only inside this test; the launch object
    # still carries the exact production factory spec and can never serialize
    # a test factory into a production config.
    launch.factory = launcher.REAL_FACTORY_SPEC
    launch.ledger_path = config.ledger_path
    launch.host_private_key_path = Path("/tmp/eva-test-host-key.pem")
    launch.host_trust_store_path = Path("/tmp/eva-test-host-trust.json")
    launch.host_key_id = HOST_KEY_ID
    launch.host_public_key_blake3 = blake3_hex("host-public")
    launch.host_trust_store_blake3 = blake3_hex("host-trust")

    # Stop before the later structural port checks; the factory observation is
    # sufficient to prove the immutable CampaignConcurrency replacement.
    with pytest.raises(CampaignDeploymentError):
        launcher._compose(launch, profile="128", max_candidates=16)
    assert observed["concurrency"].worker_width == 128
    assert observed["concurrency"].max_candidates == 16
    assert observed["host_trust"] == {
        "host_private_key_path": launch.host_private_key_path,
        "host_trust_store_path": launch.host_trust_store_path,
        "host_key_id": HOST_KEY_ID,
        "expected_host_public_key_blake3": launch.host_public_key_blake3,
        "expected_host_trust_store_blake3": launch.host_trust_store_blake3,
    }

    with pytest.raises(CampaignDeploymentError, match="fixed at one"):
        launcher._compose(launch, profile="canary", max_candidates=2)
