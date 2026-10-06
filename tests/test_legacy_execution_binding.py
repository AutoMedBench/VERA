from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from eva_agent.deployment.campaign import CampaignDeploymentError
from eva_agent.deployment.medresearch_v2 import _load_host_signer
from eva_agent.pipeline import (
    BenchmarkEpisode,
    BenchmarkSource,
    Cohort,
    DeterministicUUIDFactory,
    FilesystemSandbox,
    ModelTarget,
    OpenAIStyleRolloutAdapter,
    ParallelToolRuntime,
    ProviderRollout,
    RolloutRequest,
    Stage,
    TrajectoryEvent,
    ToolCall,
)
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_json_bytes
from eva_agent.codex_pipeline.native_policy_v2 import build_stage_tool_guidance_v1
from eva_agent.rubrics import load_and_compile_registry
from eva_agent.sources import LegacyExecutionBindingResolver, ProductionSourceBlocker
import eva_agent.training.teacher_worker as teacher_worker_module
from eva_agent.training.teacher_worker import (
    TeacherCandidateContext,
    execute_single_rollout,
    hydrate_teacher_stage_prerequisites,
)


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT.parent / "rlevo-med-research"
SUPERVISOR = LEGACY / "runs" / "evamed-campaign-supervisor-v24-attempt1"
SOURCE_ID = "rlevo-medres-agentclinic-s1-0129445e4049d288112e941e"
AUTOMEDBENCH_ALIAS_SOURCE_ID = "rlevo-medres-automedbench-e2e-7cea2e60b9a1586d"
AUTOMEDBENCH_ALIAS_CANDIDATE_ID = "c8b0645c-4701-539e-a1de-2392fb7c3f71"
AGENTCLINIC_S2_SOURCE_ID = "rlevo-medres-agentclinic-s2-408cc04a3dd49f33cc6e5d2c"
AUTOMEDBENCH_S3_SOURCE_ID = "rlevo-medres-automedbench-s3-13b7bd64ce7ed5fa"


def _legacy_host_key_source() -> Path:
    configured = os.environ.get("EVA_TEST_HOST_SIGNING_KEY_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path.home()
        / ".config"
        / "rlevo-med-research"
        / "host-signing-ed25519-v1.pem"
    ).resolve()


def _legacy_sha256(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _plain(value):
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    return value


@pytest.fixture(scope="module")
def resolver(tmp_path_factory):
    host_key_source = _legacy_host_key_source()
    required = (
        SUPERVISOR / "candidate-registry.v24.json",
        LEGACY / "config" / "host-trust-store.v1.json",
        LEGACY / "config" / "image-refs.v1.json",
        host_key_source,
    )
    if not all(path.is_file() and not path.is_symlink() for path in required):
        pytest.skip("pinned signed EvaMed v24 authority is not present")
    state = tmp_path_factory.mktemp("legacy-execution-state")
    host_key = state / "host-signing-ed25519-v1.pem"
    shutil.copyfile(host_key_source, host_key)
    host_key.chmod(0o600)
    return LegacyExecutionBindingResolver(
        authority_root=LEGACY,
        supervisor_root=SUPERVISOR,
        trust_store_path=LEGACY / "config" / "host-trust-store.v1.json",
        legacy_python_root=LEGACY / "src",
        runtime_state_root=state,
        host_private_key_path=host_key,
        host_key_id="rlevo-host-20260902-v1",
        image_refs_path=LEGACY / "config" / "image-refs.v1.json",
        worker_width=64,
    )


def _temporary_host_authority(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    key_id = "eva-test-host-v1"
    private = Ed25519PrivateKey.generate()
    private_path = (tmp_path / "host-signing.pem").resolve()
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
    trust_path = (tmp_path / "host-trust.json").resolve()
    trust_bytes = canonical_json_bytes(
        {
            "schema": "rlevo.med-research-host-trust-store.v1",
            "algorithm": "Ed25519",
            "status": "active",
            "created_at_utc": "2026-09-07T00:00:00Z",
            "keys": {key_id: base64.b64encode(public).decode("ascii")},
        }
    )
    trust_path.write_bytes(trust_bytes)
    values: dict[str, object] = {
        "private_key_path": private_path,
        "key_id": key_id,
        "trust_store_path": trust_path,
        "expected_public_key_blake3": blake3_bytes(public),
        "expected_trust_store_blake3": blake3_bytes(trust_bytes),
    }
    return values, private_path, trust_path


def test_host_signer_requires_exact_temporary_ed25519_authority(tmp_path: Path) -> None:
    values, private_path, trust_path = _temporary_host_authority(tmp_path)
    signer = _load_host_signer(**values)
    assert signer.key_id == values["key_id"]
    assert signer.public_key_blake3 == values["expected_public_key_blake3"]

    with pytest.raises(CampaignDeploymentError, match="key ID differs"):
        _load_host_signer(**{**values, "key_id": "missing"})
    with pytest.raises(CampaignDeploymentError, match="commitment differs"):
        _load_host_signer(
            **{**values, "expected_trust_store_blake3": "f" * 64}
        )
    with pytest.raises(CampaignDeploymentError, match="commitment differs"):
        _load_host_signer(
            **{**values, "expected_public_key_blake3": "e" * 64}
        )

    original_trust = trust_path.read_bytes()
    extra_trust = {**json.loads(original_trust), "unexpected": True}
    extra_trust_bytes = canonical_json_bytes(extra_trust)
    trust_path.write_bytes(extra_trust_bytes)
    with pytest.raises(CampaignDeploymentError, match="contract or key ID differs"):
        _load_host_signer(
            **{
                **values,
                "expected_trust_store_blake3": blake3_bytes(extra_trust_bytes),
            }
        )
    duplicate_trust_bytes = b'{"schema":"duplicate",' + original_trust[1:]
    trust_path.write_bytes(duplicate_trust_bytes)
    with pytest.raises(CampaignDeploymentError, match="duplicate JSON keys"):
        _load_host_signer(
            **{
                **values,
                "expected_trust_store_blake3": blake3_bytes(duplicate_trust_bytes),
            }
        )
    trust_path.write_bytes(original_trust)

    private_path.chmod(0o644)
    with pytest.raises(CampaignDeploymentError, match="mode differs"):
        _load_host_signer(**values)
    private_path.chmod(0o600)

    link = (tmp_path / "host-signing-link.pem").resolve()
    os.symlink(private_path, link)
    with pytest.raises(CampaignDeploymentError, match="path is unsafe"):
        _load_host_signer(**{**values, "private_key_path": link})

    other = Ed25519PrivateKey.generate()
    private_path.write_bytes(
        other.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    private_path.chmod(0o600)
    with pytest.raises(CampaignDeploymentError, match="differs from trust store"):
        _load_host_signer(**values)
    assert trust_path.is_file()


def _binding(resolver):
    candidate_id = str(uuid5(NAMESPACE_URL, "eva-test:" + SOURCE_ID))
    return resolver.resolve(candidate_id, source_candidate_id=SOURCE_ID)


def test_v24_catalog_and_candidate_policy_are_exact_and_candidate_scoped(resolver) -> None:
    binding = _binding(resolver)
    assert resolver.catalog.candidate_count == 1_344
    assert resolver.executable_candidate_count == 1_344
    assert resolver.catalog_inventory_blake3 == resolver.catalog_blake3
    assert tuple(row.source_candidate_id for row in resolver.inventory()) == tuple(
        sorted(row.source_candidate_id for row in resolver.inventory())
    )
    assert len(resolver.catalog_blake3) == 64
    assert binding.source_candidate_id == SOURCE_ID
    assert binding.source_policy_path.is_absolute()
    assert binding.construction_manifest_path.is_absolute()
    assert not binding.source_policy_path.is_symlink()
    assert binding.source_policy_blake3 == binding.source_policy_blake3.lower()
    assert len(binding.binding_blake3) == 64
    source_policy = json.loads(binding.source_policy_path.read_text(encoding="utf-8"))
    expected = {
        row["name"]: (row["description"], row["input_schema"])
        for row in source_policy["tools"]
    }
    observed = {
        row["function"]["name"]: (
            row["function"]["description"],
            _plain(row["function"]["parameters"]),
        )
        for row in binding.tool_registry.public_schemas()
    }
    assert observed == expected
    assert set(binding.initial_workspace_files) == {
        ".eva/runtime-context.json",
        ".eva/source-policy.json",
    }
    assert binding.initial_workspace_files[".eva/source-policy.json"] == (
        binding.source_policy_path.read_bytes()
    )


def test_automedbench_historical_episode_alias_is_provenance_bound(resolver) -> None:
    binding = resolver.resolve(
        AUTOMEDBENCH_ALIAS_CANDIDATE_ID,
        source_candidate_id=AUTOMEDBENCH_ALIAS_SOURCE_ID,
    )
    context = _plain(binding.public_runtime_context)
    active_episode = context["active_episode_id"]
    historical_episode = "evamed-automedb-e2e-7cea2e60b9a1586d"
    assert active_episode.startswith("panel-template-")
    assert active_episode != historical_episode
    assert context["s1_plan_contract"]["episode_id"] == active_episode
    assert context["s2_evidence_contract"]["episode_id"] == active_episode
    assert tuple(row.name for row in binding.tool_catalog) == (
        "execute_code",
        "materialize_evidence_selection",
        "materialize_plan",
        "retrieve_frozen_evidence",
        "submit_results",
    )
    source_policy = json.loads(binding.source_policy_path.read_text(encoding="utf-8"))
    assert any(
        historical_episode in message["content"]
        for message in source_policy["messages"]
    )
    core = binding.core_document()
    assert core["source_policy_blake3"] == (
        "d63b67bbea7b8391eef2f8f816cccf58c343814d1d4620576e8aabe6a641b46c"
    )
    assert core["construction_manifest_blake3"] == (
        "e4947cc09aac00f83b49d31dc851220dce1c83332402111a24f6c1b869d9880a"
    )
    assert core["public_runtime_context"] == context


class _RealHandlerFakeCompletions:
    """In-memory provider fixture; only the injected real handlers do work."""

    def __init__(self, plan: dict, evidence_id: str) -> None:
        self.plan = plan
        self.evidence_id = evidence_id
        self.calls = 0
        self.network_calls = 0

    def create(self, **request):
        assert request["parallel_tool_calls"] is True
        self.calls += 1
        if self.calls == 1:
            name = "materialize_plan"
            arguments = self.plan
        elif self.calls == 2:
            name = "retrieve_frozen_evidence"
            arguments = {"evidence_id": self.evidence_id}
        else:
            return {
                "choices": [
                    {"message": {"content": "fake provider stopped after observing signed S2 evidence"}}
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"fake-{self.calls}",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments, separators=(",", ":")),
                                },
                            }
                        ],
                    }
                }
            ]
        }


def test_fake_provider_executes_real_plan_and_frozen_evidence_handlers(
    resolver, tmp_path: Path
) -> None:
    binding = _binding(resolver)
    context = _plain(binding.public_runtime_context)
    s1_contract = context["s1_plan_contract"]
    s2_contract = context["s2_evidence_contract"]
    historical = binding.construction_root / (
        "construction-private/04-solver-rollout/stages/"
        "s1-attempt-1/submitted-plan.json"
    )
    plan = json.loads(historical.read_text(encoding="utf-8"))
    plan["episode_id"] = s1_contract["episode_id"]
    plan["contract_sha256"] = _legacy_sha256(s1_contract)
    evidence_id = s2_contract["evidence_objects"][0]["evidence_id"]

    ids = DeterministicUUIDFactory("legacy-real-handler-fake-provider")
    sandbox = FilesystemSandbox(
        tmp_path / "workspaces",
        ids.new("sandbox"),
        binding.initial_workspace_files,
    )
    runtime = ParallelToolRuntime(
        workspace=sandbox,
        registry=binding.tool_registry,
        id_factory=ids,
        maximum_parallel_calls=16,
    )
    fake = _RealHandlerFakeCompletions(plan, evidence_id)
    adapter = OpenAIStyleRolloutAdapter(
        SimpleNamespace(chat=SimpleNamespace(completions=fake)),
        id_factory=ids,
        maximum_model_turns=4,
    )
    request = RolloutRequest(
        rollout_id=ids.new("rollout"),
        sandbox=SimpleNamespace(manifest_blake3="0" * 64),
        model=ModelTarget(Cohort.STRONG, "fake-no-network", "in-memory"),
        policy_visible_context=binding.public_runtime_context,
        available_tools=binding.tool_registry.public_schemas(),
    )
    campaign_ledger = ROOT / "runs" / "campaign.sqlite3"
    ledger_before = campaign_ledger.read_bytes() if campaign_ledger.exists() else None
    rollout = adapter.run(request, runtime)
    ledger_after = campaign_ledger.read_bytes() if campaign_ledger.exists() else None

    assert rollout.assistant_output.startswith("fake provider stopped")
    assert fake.calls == 3
    assert fake.network_calls == 0
    trace = runtime.trace()
    assert [row.status for row in trace.results] == ["completed", "completed"]
    assert trace.results[0].output["stage"] == "S1"
    assert trace.results[0].output["gate_passed"] is True
    assert trace.results[1].output["stage"] == "S2"
    assert trace.results[1].output["gate_passed"] is True
    assert trace.results[1].output["evidence"]["evidence_id"] == evidence_id
    assert sandbox.read_bytes(s1_contract["stage_artifacts"]["S1"])
    assert ledger_after == ledger_before


def test_schema_invalid_plan_does_not_consume_single_s1_attempt(
    resolver, tmp_path: Path
) -> None:
    binding = _binding(resolver)
    context = _plain(binding.public_runtime_context)
    s1_contract = context["s1_plan_contract"]
    historical = binding.construction_root / (
        "construction-private/04-solver-rollout/stages/"
        "s1-attempt-1/submitted-plan.json"
    )
    plan = json.loads(historical.read_text(encoding="utf-8"))
    plan["episode_id"] = s1_contract["episode_id"]
    plan["contract_sha256"] = _legacy_sha256(s1_contract)
    ids = DeterministicUUIDFactory("legacy-schema-rejection-clean-retry")
    sandbox = FilesystemSandbox(
        tmp_path / "schema-retry-workspace",
        ids.new("sandbox"),
        binding.initial_workspace_files,
    )
    runtime = ParallelToolRuntime(
        workspace=sandbox,
        registry=binding.tool_registry,
        id_factory=ids,
        maximum_parallel_calls=1,
    )

    rejected = runtime.execute(
        [ToolCall(ids.new("call"), "materialize_plan", {})]
    )[0]
    assert rejected.status == "completed"
    assert rejected.output["attempt_consumed"] is False
    assert rejected.output["error"] == "schema_invalid_pre_effect"
    assert "stage-plan-artifact contract violation at $" in rejected.output["diagnostic"]

    accepted = runtime.execute(
        [ToolCall(ids.new("call"), "materialize_plan", plan)]
    )[0]
    assert accepted.status == "completed"
    assert accepted.output["gate_passed"] is True


def test_non_promoted_or_unknown_source_fails_before_provider(resolver) -> None:
    with pytest.raises(ProductionSourceBlocker, match="not v24-promoted"):
        resolver.resolve(
            str(uuid5(NAMESPACE_URL, "eva-test:unknown")),
            source_candidate_id="rlevo-medres-agentclinic-s1-does-not-exist",
        )


def test_same_frontier_retrievals_are_serialized_for_signed_contiguous_indices(
    resolver, tmp_path: Path
) -> None:
    source_id = "rlevo-medres-hbp-s1-01-v005"
    binding = resolver.resolve(
        str(uuid5(NAMESPACE_URL, "eva-test:" + source_id)),
        source_candidate_id=source_id,
    )
    context = _plain(binding.public_runtime_context)
    s1_contract = context["s1_plan_contract"]
    s2_contract = context["s2_evidence_contract"]
    historical = binding.construction_root / (
        "construction-private/04-solver-rollout/stages/"
        "s1-attempt-1/submitted-plan.json"
    )
    plan = json.loads(historical.read_text(encoding="utf-8"))
    plan["episode_id"] = s1_contract["episode_id"]
    plan["contract_sha256"] = _legacy_sha256(s1_contract)
    ids = DeterministicUUIDFactory("legacy-contiguous-retrievals")
    sandbox = FilesystemSandbox(
        tmp_path / "parallel-workspaces",
        ids.new("sandbox"),
        binding.initial_workspace_files,
    )
    runtime = ParallelToolRuntime(
        workspace=sandbox,
        registry=binding.tool_registry,
        id_factory=ids,
        maximum_parallel_calls=16,
    )
    s1 = runtime.execute(
        [ToolCall(ids.new("call"), "materialize_plan", plan)]
    )
    assert s1[0].status == "completed"
    assert s1[0].output["gate_passed"] is True
    evidence_ids = [row["evidence_id"] for row in s2_contract["evidence_objects"]]
    assert len(evidence_ids) == 2
    retrieved = runtime.execute(
        [
            ToolCall(
                ids.new("call"),
                "retrieve_frozen_evidence",
                {"evidence_id": evidence_id},
            )
            for evidence_id in evidence_ids
        ]
    )
    assert all(row.status == "completed" for row in retrieved)
    assert all(row.output["gate_passed"] is True for row in retrieved)
    assert {row.output["evidence_id"] for row in retrieved} == set(evidence_ids)
    # MCP may receive same-turn parallel calls, but the signed legacy service
    # requires contiguous indices; candidate-local execution is intentionally
    # serialized while different candidate/workspace bindings remain parallel.
    assert runtime.trace().max_parallelism_observed == 1


def test_s2_teacher_hydration_passes_real_host_retrieval_and_selection_gates(
    resolver, tmp_path: Path
) -> None:
    candidate_id = str(uuid5(NAMESPACE_URL, "eva-test:" + AGENTCLINIC_S2_SOURCE_ID))
    binding = resolver.resolve(
        candidate_id,
        source_candidate_id=AGENTCLINIC_S2_SOURCE_ID,
    )
    runtime_context = _plain(binding.public_runtime_context)
    guidance = build_stage_tool_guidance_v1(
        public_runtime_context=runtime_context,
        source_tool_catalog=binding.tool_registry.public_schemas(),
    )
    ids = DeterministicUUIDFactory("s2-teacher-prerequisite-hydration")
    sandbox = FilesystemSandbox(
        tmp_path / "s2-hydrated-workspaces",
        ids.new("sandbox"),
        binding.initial_workspace_files,
    )
    context = TeacherCandidateContext(
        episode=SimpleNamespace(
            policy_context={"execution_binding": binding.public_runtime_context}
        ),
        rubric=SimpleNamespace(),
        tool_registry=binding.tool_registry,
        turn_mcp_factory=None,
        skills_factory=None,
        stage_tool_guidance=guidance,
    )

    hydration = hydrate_teacher_stage_prerequisites(
        context=context,
        workspace=sandbox,
        id_factory=ids,
    )
    assert hydration is not None
    assert hydration["provider_calls"] == 0
    assert hydration["next_actor_frontier_index"] == 1
    assert hydration["tool_result"]["status"] == "completed"
    assert hydration["tool_result"]["output"]["gate_passed"] is True
    assert sandbox.read_bytes("work/stage-plan.json")

    runtime = ParallelToolRuntime(
        workspace=sandbox,
        registry=binding.tool_registry,
        id_factory=ids,
        maximum_parallel_calls=1,
    )
    s2_contract = runtime_context["s2_evidence_contract"]
    retrieval_results = []
    for evidence_id in s2_contract["required_evidence_ids"]:
        result = runtime.execute(
            (
                ToolCall(
                    ids.new("call"),
                    "retrieve_frozen_evidence",
                    {"evidence_id": evidence_id},
                ),
            )
        )[0]
        assert result.status == "completed"
        assert result.error_code is None
        assert result.output["gate_passed"] is True
        assert result.output["failed_check_ids"] == ()
        retrieval_results.append(result)

    by_id = {row["evidence_id"]: row for row in s2_contract["evidence_objects"]}
    selected_evidence = []
    all_statement_ids = []
    for result in retrieval_results:
        evidence_id = result.output["evidence_id"]
        source = by_id[evidence_id]
        all_statement_ids.extend(source["statement_ids"])
        selected_evidence.append(
            {
                "evidence_id": evidence_id,
                "source_id": source["source_id"],
                "source_revision": source["source_revision"],
                "statement_ids": source["statement_ids"],
                "retrieval_receipt_sha256": result.output[
                    "retrieval_receipt_sha256"
                ],
            }
        )
    selection = {
        "schema": "rlevo.med-research-evidence-selection.v1",
        "contract_sha256": hydration["tool_result"]["output"][
            "evidence_contract_sha256"
        ],
        "sandbox_id": s2_contract["sandbox_id"],
        "episode_id": s2_contract["episode_id"],
        "question": s2_contract["question"],
        "evidence_need": s2_contract["evidence_need"],
        "selected_evidence": selected_evidence,
        "inferences": [
            {
                "inference_id": claim_id,
                "text": "The retrieved frozen evidence supports this bound research inference.",
                "supporting_statement_ids": all_statement_ids,
            }
            for claim_id in s2_contract["required_claim_ids"]
        ],
        "unresolved_gaps": [
            "The frozen source does not resolve every research uncertainty."
        ],
        "limitations": ["Only the declared frozen source was available."],
        "care_directive": False,
    }
    selected = runtime.execute(
        (
            ToolCall(
                ids.new("call"),
                "materialize_evidence_selection",
                selection,
            ),
        )
    )[0]
    assert selected.status == "completed"
    assert selected.error_code is None
    assert selected.output["stage"] == "S2"
    assert selected.output["gate_passed"] is True
    assert selected.output["failed_check_ids"] == ()
    assert sandbox.read_bytes(s2_contract["selection_relative_path"])


def test_s3_hydration_finishes_before_fake_provider_and_rebuilds_manifest(
    resolver, tmp_path: Path, monkeypatch
) -> None:
    candidate_id = str(uuid5(NAMESPACE_URL, "eva-test:" + AUTOMEDBENCH_S3_SOURCE_ID))
    binding = resolver.resolve(
        candidate_id,
        source_candidate_id=AUTOMEDBENCH_S3_SOURCE_ID,
    )
    original_tool_schemas = binding.tool_registry.public_schemas()
    runtime_context = _plain(binding.public_runtime_context)
    guidance = build_stage_tool_guidance_v1(
        public_runtime_context=runtime_context,
        source_tool_catalog=binding.tool_registry.public_schemas(),
    )
    assert guidance.focus is Stage.S3
    rubric = load_and_compile_registry(
        ROOT / "rubrics/source/domain-stage-tables.v1.json"
    ).resolve("automedbench-segmentation", "S3")
    episode = BenchmarkEpisode(
        episode_id=runtime_context["active_episode_id"],
        source=BenchmarkSource(
            "AutoMedBench",
            "provider-free-s3-hydration-fixture.json",
            "signed-v24",
        ),
        domain="automedbench-segmentation",
        stage=Stage.S3,
        instruction="Continue the bound S3 pilot from verified prerequisites.",
        policy_context={"execution_binding": binding.public_runtime_context},
        initial_files=binding.initial_workspace_files,
    )
    context = TeacherCandidateContext(
        episode=episode,
        rubric=rubric,
        tool_registry=binding.tool_registry,
        turn_mcp_factory=None,
        skills_factory=None,
        stage_tool_guidance=guidance,
    )
    record = {
        "sandbox_id": str(uuid5(NAMESPACE_URL, "eva-test:s3-provider-order-sandbox")),
        "candidate_id": candidate_id,
        "episode_id": episode.episode_id,
        "domain": episode.domain,
        "stage": episode.stage.value,
        "reward_contract": {"rubric_table": rubric.to_document()},
        "workspace_initial_state": {
            "files": [],
            "file_count": 0,
            "byte_count": 0,
        },
    }
    ordering = []
    original_hydrate = teacher_worker_module.hydrate_teacher_stage_prerequisites

    def capture_hydration(**kwargs):
        ordering.append("hydration-start")
        hydration = original_hydrate(**kwargs)
        ordering.append("hydration-complete")
        return hydration

    monkeypatch.setattr(
        teacher_worker_module,
        "hydrate_teacher_stage_prerequisites",
        capture_hydration,
    )

    class CaptureProvider:
        calls = 0
        request = None

        def run(self, request, tools):
            ordering.append("provider")
            self.calls += 1
            self.request = request
            hydration = request.policy_visible_context[
                "teacher_prerequisite_hydration"
            ]
            assert hydration["focus"] == "S3"
            assert hydration["provider_calls"] == 0
            assert request.available_tools == original_tool_schemas
            assert binding.tool_registry.public_schemas() == original_tool_schemas
            assert hydration["next_actor_frontier_index"] == len(
                guidance.frontiers
            )
            assert tuple(hydration["artifact_relative_paths"]) == (
                "work/stage-plan.json",
                "work/evidence-selection.json",
            )
            assert tuple(hydration["hydrated_frontier_indices"]) == tuple(
                range(len(guidance.frontiers))
            )
            assert all(
                row["status"] == "completed"
                and row["output"]["gate_passed"] is True
                and tuple(row["output"]["failed_check_ids"]) == ()
                for row in hydration["tool_results"]
            )
            assert {
                "work/stage-plan.json",
                "work/evidence-selection.json",
            } <= set(request.sandbox.initial_files)
            assert json.loads(
                request.sandbox.initial_files["work/stage-plan.json"]
            ) == _plain(guidance.frontiers[0]["arguments"])
            selection_result = hydration["tool_results"][-1]
            assert selection_result["name"] == "materialize_evidence_selection"
            selection_artifact = json.loads(
                request.sandbox.initial_files["work/evidence-selection.json"]
            )
            assert selection_artifact["schema"] == (
                "rlevo.med-research-evidence-selection.v1"
            )
            assert selection_artifact["selected_evidence"]
            assert "$runtime" not in json.dumps(selection_artifact)
            assert "$author" not in json.dumps(selection_artifact)
            event_core = {
                "event_id": str(uuid5(NAMESPACE_URL, "eva-test:s3-provider-event")),
                "role": "assistant",
                "content": {"text": "S3 continuation observed."},
                "tool_call_ids": (),
            }
            event = TrajectoryEvent(
                **event_core, event_blake3=blake3_hex(event_core)
            )
            rollout_core = {
                "rollout_id": request.rollout_id,
                "model_id": request.model.model_id,
                "assistant_output": "S3 continuation observed.",
            }
            return ProviderRollout(
                assistant_output=rollout_core["assistant_output"],
                provider_receipt_blake3=blake3_hex(rollout_core),
                policy_events=(event,),
                safe_metadata={"provider_calls": 1},
            )

    provider = CaptureProvider()
    result = execute_single_rollout(
        record=record,
        context=context,
        provider=provider,
        target=ModelTarget(Cohort.STRONG, "fake-no-network", "in-memory"),
        workspace_root=tmp_path / "s3-provider-order-workspaces",
    )

    assert ordering == ["hydration-start", "hydration-complete", "provider"]
    assert provider.calls == 1
    assert provider.request is not None
    assert provider.request.sandbox.manifest_blake3
    for key in ("files", "file_count", "byte_count", "tree_blake3"):
        assert result["workspace_before"][key] == result["workspace_after"][key]
    assert result["tool_trace"]["results"] == []
