from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
import time

from eva_agent.pipeline import (
    BenchmarkEpisode,
    BenchmarkSource,
    Cohort,
    ModelTarget,
    ProviderRollout,
    Stage,
    TrajectoryEvent,
    ToolDefinition,
    ToolRegistry,
)
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_json_bytes
from eva_agent.rubrics import load_and_compile_registry
from eva_agent.training import (
    CampaignV2TeacherContextPool,
    TeacherCandidateContext,
    execute_single_rollout,
)


ROOT = Path(__file__).resolve().parents[1]


class _FakeCodexProvider:
    calls = 0

    def run(self, request, tools):
        self.calls += 1
        event_core = {
            "event_id": "40e4a94d-ec44-43b5-bbd2-9037e35c64c5",
            "role": "assistant",
            "content": {"text": "completed with workspace evidence"},
            "tool_call_ids": (),
        }
        event = TrajectoryEvent(**event_core, event_blake3=blake3_hex(event_core))
        core = {
            "rollout_id": request.rollout_id,
            "model_id": request.model.model_id,
            "assistant_output": "completed with workspace evidence",
        }
        return ProviderRollout(
            assistant_output=core["assistant_output"],
            provider_receipt_blake3=blake3_hex(core),
            policy_events=(event,),
            safe_metadata={"codex_turn_receipt": {"status": "completed"}},
        )


def test_fake_single_worker_emits_complete_receipt_without_judge_or_cascade(
    tmp_path: Path, monkeypatch
) -> None:
    rubric = load_and_compile_registry(
        ROOT / "rubrics/source/domain-stage-tables.v1.json"
    ).resolve("medxpertqa", "E2E")
    episode = BenchmarkEpisode(
        episode_id="episode-1",
        source=BenchmarkSource("MedXpertQA", "fixture.json", "fixture-v1"),
        domain="medxpertqa",
        stage=Stage.E2E,
        instruction="Use tools and workspace evidence.",
        policy_context={"question": "fixture"},
        initial_files={"seed.txt": b"seed\n"},
    )
    content = {"schema": "eva.bulk-workspace-fixture.v1"}
    payload = canonical_json_bytes(content)
    record = {
        "sandbox_id": "05381f58-7ed4-5314-8d43-3b098c2ee0d7",
        "candidate_id": "18cf11a4-76a4-5382-a0ad-d145824c88ac",
        "episode_id": episode.episode_id,
        "domain": episode.domain,
        "stage": episode.stage.value,
        "reward_contract": {"rubric_table": rubric.to_document()},
        "workspace_initial_state": {
            "files": [{
                "path": "input/task-contract.json",
                "content": content,
                "byte_count": len(payload),
                "content_blake3": blake3_bytes(payload),
            }],
            "file_count": 1,
            "byte_count": len(payload),
        },
    }
    monkeypatch.setenv("EVA_TEACHER_TASK_ID", "task-1")
    monkeypatch.setenv("EVA_TEACHER_ROUTE_ID", "gpt_5_6_sol")
    provider = _FakeCodexProvider()
    result = execute_single_rollout(
        record=record,
        context=TeacherCandidateContext(
            episode=episode,
            rubric=rubric,
            tool_registry=ToolRegistry(()),
            turn_mcp_factory=None,
            skills_factory=None,
        ),
        provider=provider,
        target=ModelTarget(Cohort.STRONG, "openai/gpt-5.6-sol", "eva_gpt"),
        workspace_root=tmp_path / "workspaces",
        task_id="explicit-task-1",
        route_id="gpt_5_6_sol",
    )
    assert provider.calls == 1
    assert result["score"] == 0.0
    assert result["selection_pending"] is True
    assert result["judge_calls"] == 0
    assert result["admission_writes"] == 0
    assert result["cascade_required"] is False
    assert result["semantic_retry_count"] == 0
    assert result["task_id"] == "explicit-task-1"
    assert result["route_id"] == "gpt_5_6_sol"
    assert result["messages"] and result["workspace_before"]["file_count"] == 2
    assert result["workspace_after"]["file_count"] == 2
    assert result["provider_metadata"]["codex_turn_receipt"]["status"] == "completed"


def test_context_pool_single_flights_routes_for_one_candidate(tmp_path: Path) -> None:
    rubric = load_and_compile_registry(
        ROOT / "rubrics/source/domain-stage-tables.v1.json"
    ).resolve("medxpertqa", "E2E")
    episode = BenchmarkEpisode(
        episode_id="episode-shared",
        source=BenchmarkSource("MedXpertQA", "fixture.json", "fixture-v1"),
        domain="medxpertqa",
        stage=Stage.E2E,
        instruction="Use tools without requesting the hidden answer key.",
        policy_context={"question": "fixture"},
        initial_files={"seed.txt": b"seed\n"},
    )

    class Source:
        def __init__(self):
            self.loads = 0
            self.lock = Lock()

        def load(self, _candidate_id):
            with self.lock:
                self.loads += 1
            time.sleep(0.03)
            return SimpleNamespace(episode=episode, rubric=rubric)

        def source_candidate_id(self, _candidate_id):
            return "source-candidate-1"

    class Resolver:
        def __init__(self):
            self.loads = 0

        def resolve(self, _candidate_id, *, source_candidate_id):
            assert source_candidate_id == "source-candidate-1"
            self.loads += 1
            return SimpleNamespace(
                initial_workspace_files={"binding.txt": b"bound\n"},
                public_runtime_context={"binding": "fixture"},
                tool_registry=ToolRegistry(()),
            )

    class Skills:
        def augment_registry(self, registry, stage):
            assert stage is Stage.E2E
            return registry

        def public_metadata(self, stage):
            return {"stage": stage.value, "delivery": "search-then-load"}

        def __call__(self, _request):
            return ()

    source = Source()
    resolver = Resolver()
    skills = Skills()
    turn_mcp = object()
    pool = CampaignV2TeacherContextPool(
        source=source,
        resolver=resolver,
        skills=skills,
        turn_mcp_factory=turn_mcp,
    )
    record = {
        "candidate_id": "candidate-1",
        "source_binding": {"source_candidate_id": "source-candidate-1"},
        "reward_contract": {"rubric_table": rubric.to_document()},
    }
    with ThreadPoolExecutor(max_workers=12) as executor:
        results = tuple(executor.map(lambda _: pool.load(record), range(12)))

    assert source.loads == resolver.loads == pool.context_build_count == 1
    assert len({id(context) for context, _binding in results}) == 1
    context = results[0][0]
    assert context.turn_mcp_factory is turn_mcp
    assert context.skill_delivery == {
        "stage": "E2E",
        "delivery": "search-then-load",
    }
    assert context.episode.initial_files["binding.txt"] == b"bound\n"
    assert "answer key" not in context.episode.instruction.casefold()
    assert canonical_json_bytes(
        context.episode.policy_context["public_reward_contract"]
    ) == canonical_json_bytes({"rubric_table": rubric.to_document()})
    assert context.episode.policy_context["teacher_actor_projection"] == {
        "judge_only_material_included": False,
        "public_reward_contract_included": True,
        "privacy_reserved_instruction_lexemes_rephrased": True,
    }


def test_s1_context_binds_exact_host_guidance_without_changing_tool_schemas(
    tmp_path: Path, monkeypatch
) -> None:
    rubric = load_and_compile_registry(
        ROOT / "rubrics/source/domain-stage-tables.v1.json"
    ).resolve("medxpertqa", "S1")
    episode = BenchmarkEpisode(
        episode_id="episode-s1-guidance",
        source=BenchmarkSource("MedXpertQA", "fixture.json", "fixture-v1"),
        domain="medxpertqa",
        stage=Stage.S1,
        instruction="Materialize the bound research plan.",
        policy_context={"question": "fixture"},
        initial_files={"seed.txt": b"seed\n"},
    )
    registry = ToolRegistry(
        (
            ToolDefinition(
                name="materialize_plan",
                description="Materialize the plan.",
                input_schema={"type": "object", "additionalProperties": False},
                handler=lambda _workspace, _arguments: {},
            ),
        )
    )
    original_schemas = registry.public_schemas()
    runtime_context = {"exact": "host-bound-runtime-context"}
    guidance = SimpleNamespace(
        focus=Stage.S1,
        source_candidate_id="source-candidate-s1",
        guidance_blake3="a" * 64,
        prompt_blake3="b" * 64,
        public_runtime_context_blake3="c" * 64,
        source_tool_catalog_blake3="d" * 64,
        prompt_text="exact FIRST_FRONTIER host-bound guidance",
    )
    observed = {}

    def build_guidance(*, public_runtime_context, source_tool_catalog):
        observed["context"] = public_runtime_context
        observed["catalog"] = source_tool_catalog
        return guidance

    monkeypatch.setattr(
        "eva_agent.training.teacher_worker.build_stage_tool_guidance_v1",
        build_guidance,
    )

    class Source:
        def load(self, _candidate_id):
            return SimpleNamespace(episode=episode, rubric=rubric)

        def source_candidate_id(self, _candidate_id):
            return "source-candidate-s1"

    class Resolver:
        def resolve(self, _candidate_id, *, source_candidate_id):
            assert source_candidate_id == "source-candidate-s1"
            return SimpleNamespace(
                initial_workspace_files={},
                public_runtime_context=runtime_context,
                tool_registry=registry,
            )

    class Skills:
        def augment_registry(self, supplied, stage):
            assert supplied is registry and stage is Stage.S1
            return supplied

        def public_metadata(self, stage):
            assert stage is Stage.S1
            return {"stage": "S1"}

    pool = CampaignV2TeacherContextPool(
        source=Source(),
        resolver=Resolver(),
        skills=Skills(),
        turn_mcp_factory=object(),
    )
    context, _binding = pool.load(
        {
            "candidate_id": "candidate-s1",
            "source_binding": {"source_candidate_id": "source-candidate-s1"},
            "reward_contract": {"rubric_table": rubric.to_document()},
        }
    )

    assert context.stage_tool_guidance is guidance
    assert observed == {"context": runtime_context, "catalog": original_schemas}
    assert registry.public_schemas() == original_schemas
    assert context.episode.policy_context["stage_tool_guidance_binding"] == {
        "schema": "eva.codex-stage-tool-guidance.v1",
        "source_candidate_id": "source-candidate-s1",
        "guidance_blake3": "a" * 64,
        "prompt_blake3": "b" * 64,
        "public_runtime_context_blake3": "c" * 64,
        "source_tool_catalog_blake3": "d" * 64,
        "delivery": "codex-developer-instruction-sidecar",
        "canonical_tool_schemas_changed": False,
    }
