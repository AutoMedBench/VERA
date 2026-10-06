from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from eva_agent.codex_providers import load_codex_provider_routes
from eva_agent.pipeline import Cohort, ModelTarget, Stage
from eva_agent.training import TeacherBatchError, TeacherTask
from eva_agent.training.persistent_teacher import (
    PersistentCodexTeacherPool,
    TEACHER_ADAPTER_UPSTREAM_MAX_TOKENS,
    TeacherRecordIndex,
    _isolated_shard_runtime_factory,
)


def _row() -> dict:
    return {
        "sandbox_id": "sandbox-0001",
        "candidate_id": "candidate-0001",
        "stage": "S3",
        "domain": "medxpertqa",
    }


def _task(**changes) -> TeacherTask:
    values = {
        "task_id": "sandbox-0001--qwen_3_5_397b_a17b",
        "sandbox_id": "sandbox-0001",
        "candidate_id": "candidate-0001",
        "route_id": "qwen_3_5_397b_a17b",
        "stage": "S3",
        "domain": "medxpertqa",
    }
    values.update(changes)
    return TeacherTask(**values)


def test_record_index_reopens_only_exact_scheduled_task() -> None:
    index = TeacherRecordIndex((_row(),))
    assert index.count == 1
    assert index.for_task(_task())["candidate_id"] == "candidate-0001"
    with pytest.raises(TeacherBatchError, match="identity"):
        index.for_task(_task(task_id="../wrong"))
    with pytest.raises(TeacherBatchError, match="binding"):
        index.for_task(_task(stage="E2E"))
    with pytest.raises(TeacherBatchError, match="absent"):
        index.for_task(
            _task(
                task_id="sandbox-9999--qwen_3_5_397b_a17b",
                sandbox_id="sandbox-9999",
            )
        )


def test_persistent_pool_construction_is_provider_free_and_bounded(
    tmp_path: Path,
) -> None:
    routes = load_codex_provider_routes(
        env_files=(),
        environment={
            "MODEL_QWEN_3_5_397B_A17B": "Qwen/Qwen3.5-397B-A17B",
            "NVIDIA_API_KEY": "fixture-secret",
            "NVIDIA_BASE_URL": "https://integrate.api.nvidia.com/v1",
        },
        route_ids=("qwen_3_5_397b_a17b",),
    )

    class ContextPool:
        def load(self, _record):  # pragma: no cover - construction must not call it
            raise AssertionError("provider-free construction loaded a candidate")

    pool = PersistentCodexTeacherPool(
        rows=(_row(),),
        routes=routes,
        output_root=tmp_path / "out",
        worker_width=64,
        app_server_shards=8,
        context_pool=ContextPool(),  # type: ignore[arg-type]
        codex_bin="/bin/true",
        launch_cwd=tmp_path,
    )
    assert pool.safe_metadata == {
        "schema": "eva.codex-teacher-persistent-pool.v1",
        "record_count": 1,
        "route_ids": ["qwen_3_5_397b_a17b"],
        "route_count": 1,
        "worker_width": 64,
        "app_server_shards": 8,
        "campaign_resource_opens": 1,
        "model_catalog_materializations": 1,
        "app_server_opens": 8,
        "fresh_thread_per_task": True,
        "workspace_per_task": True,
        "retry_count": 0,
        "provider_calls_during_construction": 0,
        "isolated_codex_user_config": True,
        "durable_adapter_receipts": True,
    }
    assert pool._gateway is not None
    assert (
        pool._gateway.safe_metadata["upstream_max_tokens_override"]
        == TEACHER_ADAPTER_UPSTREAM_MAX_TOKENS
        == 32_768
    )
    pool.close()
    with pytest.raises(TeacherBatchError, match="not open"):
        pool.run(_task())


def test_sharded_teacher_runtime_uses_one_isolated_codex_home_per_shard(
    tmp_path: Path,
) -> None:
    parent = (tmp_path / "private-runtimes").resolve()
    factory = _isolated_shard_runtime_factory(
        codex_bin=Path("/bin/true"),
        launch_cwd=tmp_path.resolve(),
        isolation_parent=parent,
        config_overrides=("project_doc_max_bytes=0",),
        child_env={"NVIDIA_API_KEY": "fixture-secret"},
        shard_count=16,
    )
    with ThreadPoolExecutor(max_workers=16) as executor:
        runtimes = tuple(executor.map(lambda _: factory(), range(16)))
    assert len({id(runtime) for runtime in runtimes}) == 16
    assert sorted(path.name for path in parent.iterdir()) == [
        f"shard-{index:03d}" for index in range(16)
    ]
    assert all((path.stat().st_mode & 0o077) == 0 for path in parent.iterdir())
    with pytest.raises(TeacherBatchError, match="over-consumed"):
        factory()


def test_s1_guidance_is_delivered_as_codex_developer_instruction() -> None:
    routes = load_codex_provider_routes(
        env_files=(),
        environment={
            "MODEL_QWEN_3_5_397B_A17B": "Qwen/Qwen3.5-397B-A17B",
            "NVIDIA_API_KEY": "fixture-secret",
            "NVIDIA_BASE_URL": "https://integrate.api.nvidia.com/v1",
        },
        route_ids=("qwen_3_5_397b_a17b",),
    )
    route = routes["qwen_3_5_397b_a17b"]
    context = SimpleNamespace(
        stage_tool_guidance=SimpleNamespace(
            focus=Stage.S1,
            prompt_text="exact FIRST_FRONTIER host-bound guidance",
        )
    )
    request = SimpleNamespace(
        model=ModelTarget(Cohort.STRONG, route.model_id, route.config.provider_id)
    )

    options = PersistentCodexTeacherPool._options_factory(route, context)(
        request, "/tmp/candidate", (), None
    )

    assert options.developer_instructions is not None
    assert options.developer_instructions.startswith(
        "exact FIRST_FRONTIER host-bound guidance\n\n"
    )
    assert "prioritize the next required tool call over long prose" in (
        options.developer_instructions
    )
    assert "Do not end the turn with an empty final_answer" in (
        options.developer_instructions
    )
