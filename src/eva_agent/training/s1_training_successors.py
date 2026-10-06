"""Training-only executable S1 successors for frozen source-only candidates.

The signed campaign remains immutable: a successor points back to one exact
primary/train/S1/source-only row and receives fresh application identities.
Only the public task, rubric, source binding, evidence manifest, canonical
legacy tool schemas, and a bounded workspace scaffold cross this boundary.
The lightweight S1 host effect is intentionally training-only; it is never a
promotion proof and cannot mutate the production admission state.
"""

from __future__ import annotations

from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path, PurePosixPath
from threading import RLock
from types import MappingProxyType, SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence
from uuid import UUID, uuid5

from jsonschema import Draft202012Validator

from eva_agent.codex_pipeline.native_policy_v2 import build_stage_tool_guidance_v1
from eva_agent.pipeline import Stage, ToolDefinition, ToolRegistry
from eva_agent.pipeline.digests import (
    blake3_bytes,
    blake3_hex,
    canonical_json_bytes,
    canonical_value,
)

from .teacher_batch import TeacherBatchError, iter_bulk_sandboxes
from .teacher_worker import (
    TeacherCandidateContext,
    _campaign_v2_context_resources,
    teacher_safe_actor_instruction,
)


BUNDLE_SCHEMA = "eva.training-only-s1-successor-bundle.v1"
ROW_SCHEMA = "eva.training-only-s1-successor.v1"
RUNTIME_CONTEXT_SCHEMA = "eva.legacy-candidate-runtime-context.v1"
CATALOG_SCHEMA = "eva.prospective-execution-binding-catalog.v3"
NAMESPACE = UUID("9e9d9424-6e5f-4e03-9ae1-acdeae2974f5")
SOURCE_ROOT = Path("/localhome/local-operator/operator_GB300-2/rlevo-med-research")
LEGACY_SCHEMA_ROOT = SOURCE_ROOT / "src/rlevo_med_research/schemas"
EXPECTED_FAMILIES = frozenset(
    {"agentclinic", "automedbench", "healthbench-professional", "medxpertqa"}
)


def _legacy_json_bytes(value: Any) -> bytes:
    return json.dumps(
        canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    """Use SHA-256 only where the unchanged legacy wire schema requires it."""

    return hashlib.sha256(_legacy_json_bytes(value)).hexdigest()


def _safe_relative_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise TeacherBatchError("training successor source path differs")
    path = root.joinpath(*pure.parts)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError:
        raise TeacherBatchError("training successor source path escapes") from None
    if path.is_symlink() or not resolved.is_file() or resolved.stat().st_nlink != 1:
        raise TeacherBatchError("training successor source topology differs")
    return resolved


def _strict_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        raise TeacherBatchError("training successor JSON differs") from None
    if not isinstance(value, Mapping):
        raise TeacherBatchError("training successor JSON must be an object")
    return value


def _catalog_rows(catalog_path: Path) -> Mapping[str, Mapping[str, Any]]:
    from eva_agent.admission.receipts import SignedEnvelope, verify_signed_envelope
    from eva_agent.deployment.medresearch_v2 import LEGACY_ROOT

    document = _strict_json(catalog_path.resolve(strict=True))
    envelope = SignedEnvelope.from_document(document)
    verify_signed_envelope(
        envelope,
        trust_store_path=LEGACY_ROOT / "config/host-trust-store.v1.json",
    )
    payload = envelope.payload
    rows = payload.get("rows")
    if payload.get("schema") != CATALOG_SCHEMA or not isinstance(rows, (list, tuple)):
        raise TeacherBatchError("signed v3.r1 catalog differs")
    selected: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("candidate_id"), str):
            raise TeacherBatchError("signed catalog row differs")
        selected[str(row["candidate_id"])] = row
    return MappingProxyType(selected)


def _eligible_rows(
    *, bulk_root: Path, catalog_path: Path
) -> tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...]:
    catalog = _catalog_rows(catalog_path)
    selected: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for record in iter_bulk_sandboxes(bulk_root):
        catalog_row = catalog.get(str(record["candidate_id"]))
        if catalog_row is None:
            raise TeacherBatchError("bulk candidate is absent from signed catalog")
        if (
            record.get("stage") == "S1"
            and record.get("split") == "train"
            and catalog_row.get("selection_tier") == "primary"
            and catalog_row.get("split") == "train"
            and catalog_row.get("stage") == "S1"
            and catalog_row.get("execution_status") == "source_only"
        ):
            selected.append((record, catalog_row))
    if len(selected) != 754:
        raise TeacherBatchError("signed primary/train/S1/source-only count differs")
    if {str(row[1]["source_family"]) for row in selected} != EXPECTED_FAMILIES:
        raise TeacherBatchError("training successor family coverage differs")
    return tuple(selected)


def _successor_id(kind: str, predecessor_id: str) -> str:
    return str(uuid5(NAMESPACE, f"training-only-s1-successor-v1:{kind}:{predecessor_id}"))


def _successor_source_id(predecessor_source_id: str) -> str:
    suffix = blake3_hex(
        {"schema": ROW_SCHEMA, "predecessor_source_candidate_id": predecessor_source_id}
    )[:20]
    return f"rlevo-medres-training-s1-{suffix}"


def _row_core(
    record: Mapping[str, Any], catalog_row: Mapping[str, Any]
) -> dict[str, Any]:
    source_binding = record.get("source_binding")
    reward = record.get("reward_contract")
    if not isinstance(source_binding, Mapping) or not isinstance(reward, Mapping):
        raise TeacherBatchError("training successor predecessor binding differs")
    source_path = _safe_relative_file(
        SOURCE_ROOT, str(source_binding.get("authority_relative_path", ""))
    )
    source_bytes = source_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != source_binding.get(
        "upstream_source_artifact_sha256"
    ):
        raise TeacherBatchError("training successor source artifact changed")
    construction = _strict_json(source_path)
    evidence_manifest_path = source_path.parent / "evidence-manifest.json"
    evidence_manifest_path = _safe_relative_file(
        SOURCE_ROOT, evidence_manifest_path.relative_to(SOURCE_ROOT).as_posix()
    )
    evidence_manifest_bytes = evidence_manifest_path.read_bytes()
    rubric = reward.get("rubric_table")
    items = rubric.get("items") if isinstance(rubric, Mapping) else None
    if (
        construction.get("schema")
        != "rlevo.med-research-candidate-construction-input.v3"
        or construction.get("sandbox", {}).get("focus") != "S1"
        or construction.get("sandbox", {}).get("sandbox_id")
        != source_binding.get("source_candidate_id")
        or not isinstance(items, (list, tuple))
        or not 5 <= len(items) <= 10
        or any(not isinstance(item, Mapping) or item.get("observable") is not True for item in items)
    ):
        raise TeacherBatchError("training successor source/rubric contract differs")
    candidate_id = _successor_id("candidate", str(record["candidate_id"]))
    sandbox_id = _successor_id("sandbox", str(record["sandbox_id"]))
    episode_id = _successor_id("episode", str(record["episode_id"]))
    source_id = _successor_source_id(str(source_binding["source_candidate_id"]))
    core = {
        "schema": ROW_SCHEMA,
        "ordinal": int(record["queue_ordinal"]),
        "candidate_id": candidate_id,
        "sandbox_id": sandbox_id,
        "episode_id": episode_id,
        "source_candidate_id": source_id,
        "source_family": str(record["source_family"]),
        "domain": str(record["domain"]),
        "stage": "S1",
        "split": "train",
        "predecessor": {
            "candidate_id": str(record["candidate_id"]),
            "sandbox_id": str(record["sandbox_id"]),
            "episode_id": str(record["episode_id"]),
            "source_candidate_id": str(source_binding["source_candidate_id"]),
            "bulk_record_blake3": str(record["record_blake3"]),
            "signed_catalog_row_blake3": str(catalog_row["row_blake3"]),
        },
        "public_materials": {
            "construction_input_relative_path": source_path.relative_to(SOURCE_ROOT).as_posix(),
            "construction_input_blake3": blake3_bytes(source_bytes),
            "evidence_manifest_relative_path": evidence_manifest_path.relative_to(SOURCE_ROOT).as_posix(),
            "evidence_manifest_blake3": blake3_bytes(evidence_manifest_bytes),
            "rubric_blake3": str(catalog_row["rubric_blake3"]),
            "rubric_item_count": len(items),
        },
        "controls": {
            "training_only": True,
            "production_promotion_claimed": False,
            "production_admission_writes": 0,
            "predecessor_schema_changed": False,
            "rubric_changed": False,
            "one_attempt_per_route": True,
            "provider_retry_count": 0,
        },
    }
    return {**core, "row_blake3": blake3_hex(core)}


def build_training_successor_bundle(
    *, bulk_root: Path, catalog_path: Path
) -> Mapping[str, Any]:
    rows = tuple(
        _row_core(record, catalog_row)
        for record, catalog_row in _eligible_rows(
            bulk_root=bulk_root, catalog_path=catalog_path
        )
    )
    identities = [
        str(row[key])
        for row in rows
        for key in ("candidate_id", "sandbox_id", "episode_id")
    ]
    predecessor_ids = {
        str(row["predecessor"][key])
        for row in rows
        for key in ("candidate_id", "sandbox_id", "episode_id")
    }
    if len(identities) != len(set(identities)) or set(identities) & predecessor_ids:
        raise TeacherBatchError("training successor identity isolation differs")
    family_counts: dict[str, int] = {}
    for row in rows:
        family = str(row["source_family"])
        family_counts[family] = family_counts.get(family, 0) + 1
    core = {
        "schema": BUNDLE_SCHEMA,
        "bundle_id": _successor_id("bundle", "signed-v3.r1-primary-train-s1-source-only"),
        "source_catalog": str(catalog_path),
        "source_bulk_root": str(bulk_root),
        "row_count": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "rows": rows,
        "controls": {
            "training_only": True,
            "external_provider_calls": 0,
            "canonical_campaign_schemas_mutated": False,
            "production_authority_mutated": False,
        },
    }
    return MappingProxyType({**core, "bundle_blake3": blake3_hex(core)})


def verify_training_successor_bundle(
    bundle: Mapping[str, Any], *, bulk_root: Path, catalog_path: Path
) -> str:
    expected = build_training_successor_bundle(
        bulk_root=bulk_root, catalog_path=catalog_path
    )
    if canonical_value(bundle) != canonical_value(expected):
        raise TeacherBatchError("training successor bundle differs")
    return str(expected["bundle_blake3"])


def write_training_successor_bundle(
    path: Path, *, bulk_root: Path, catalog_path: Path
) -> Mapping[str, Any]:
    bundle = build_training_successor_bundle(
        bulk_root=bulk_root, catalog_path=catalog_path
    )
    payload = canonical_json_bytes(bundle)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        if path.read_bytes() != payload:
            raise TeacherBatchError("existing training successor bundle differs")
    else:
        path.write_bytes(payload)
        path.chmod(0o400)
    return bundle


def load_training_successor_records(
    bundle_path: Path, *, bulk_root: Path, catalog_path: Path
) -> tuple[Mapping[str, Any], ...]:
    bundle = _strict_json(bundle_path.resolve(strict=True))
    verify_training_successor_bundle(
        bundle, bulk_root=bulk_root, catalog_path=catalog_path
    )
    predecessor_records = {
        str(row["candidate_id"]): row for row in iter_bulk_sandboxes(bulk_root)
    }
    records: list[Mapping[str, Any]] = []
    for descriptor in bundle["rows"]:
        predecessor = predecessor_records[str(descriptor["predecessor"]["candidate_id"])]
        record = deepcopy(dict(predecessor))
        record.update(
            {
                "schema": ROW_SCHEMA,
                "candidate_id": descriptor["candidate_id"],
                "sandbox_id": descriptor["sandbox_id"],
                "episode_id": descriptor["episode_id"],
                "training_successor": descriptor,
                "record_blake3": descriptor["row_blake3"],
            }
        )
        records.append(MappingProxyType(record))
    return tuple(records)


def _schema(name: str) -> dict[str, Any]:
    value = deepcopy(dict(_strict_json(LEGACY_SCHEMA_ROOT / name)))
    value.pop("$schema", None)
    value.pop("$id", None)
    return value


def _runtime_materials(
    descriptor: Mapping[str, Any], construction: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ToolRegistry]:
    source_id = str(descriptor["source_candidate_id"])
    episode_id = str(descriptor["episode_id"])
    source_family = str(descriptor["source_family"])
    task = construction["task_brief"]
    runtime = construction["runtime"]
    evidence_objects = construction["evidence_objects"]
    evidence_ids = [str(row["evidence_id"]) for row in evidence_objects]
    s4 = runtime["s4_artifact"]
    policy = {
        "schema": "eva.training-only-s1-tool-policy.v1",
        "source_candidate_id": source_id,
        "episode_id": episode_id,
        "stage": "S1",
        "source_family": source_family,
        "training_only": True,
        "production_promotion_claimed": False,
    }
    policy_sha = _sha256(policy)
    s1 = {
        "schema": "rlevo.med-research-stage-plan-contract.v1",
        "contract_id": f"training-s1-{source_id.rsplit('-', 1)[-1]}",
        "sandbox_id": source_id,
        "episode_id": episode_id,
        "focus": "S1",
        "policy_sha256": policy_sha,
        "task_contract": {
            "task_ids": list(task["task_ids"]),
            "final_artifact_relative_path": str(s4["relative_path"]),
            "final_artifact_schema_sha256": _sha256(s4["json_schema"]),
        },
        "host_inputs": [
            {"input_id": evidence_id, "inspection_tool": "retrieve_frozen_evidence"}
            for evidence_id in evidence_ids
        ],
        "stage_artifacts": {
            "S1": "work/stage-plan.json",
            "S2": "work/evidence-selection.json",
            "S3": str(runtime["s3_artifact"]["relative_path"]),
            "S4": str(s4["relative_path"]),
            "S5": ".eva/final-host-receipt.json",
        },
        "budgets": {
            "max_turns": int(runtime["policy_budgets"]["max_turns"]),
            "wall_time_seconds": int(runtime["policy_budgets"]["wall_time_seconds"]),
            "minimum_s5_reserved_turns": int(
                runtime["policy_budgets"]["minimum_s5_reserved_turns"]
            ),
            "max_clean_retries_per_execution_stage": 1,
        },
        "max_plan_bytes": 131072,
    }
    projected_evidence = []
    claim_ids = []
    for index, row in enumerate(evidence_objects):
        evidence_id = str(row["evidence_id"])
        statement_id = f"statement-{index + 1}-{evidence_id}"[:128]
        projected_evidence.append(
            {
                "evidence_id": evidence_id,
                "source_id": evidence_id,
                "source_revision": str(row["sha256"]),
                "statement_ids": [statement_id],
            }
        )
        claim_ids.append(f"claim-{index + 1}")
    s2_without_receipt = {
        "schema": "rlevo.med-research-evidence-service-contract.v1",
        "contract_id": f"training-s2-{source_id.rsplit('-', 1)[-1]}",
        "sandbox_id": source_id,
        "episode_id": episode_id,
        "focus": "S1",
        "policy_sha256": policy_sha,
        "s1_plan_contract_sha256": _sha256(s1),
        # The unchanged legacy context requires this field before S1 executes.
        # It is explicitly a commitment to the unmaterialized state, not a
        # claim that a signed predecessor receipt exists.
        "s1_receipt_sha256": _sha256(
            {"schema": "eva.training-s1-unmaterialized.v1", "episode_id": episode_id}
        ),
        "question": str(task["research_question"]),
        "evidence_need": str(task["evidence_need"]),
        "evidence_objects": projected_evidence,
        "required_evidence_ids": evidence_ids,
        "required_claim_ids": claim_ids,
        "limits": {
            "max_retrievals": len(evidence_ids),
            "min_selected": len(evidence_ids),
            "max_selected": len(evidence_ids),
            "max_selection_bytes": 131072,
            "max_selection_attempts": 1,
        },
        "selection_relative_path": "work/evidence-selection.json",
    }
    s2 = s2_without_receipt

    plan_schema = _schema("stage-plan-artifact.v1.schema.json")
    properties = plan_schema["properties"]
    properties["contract_sha256"] = {"const": _sha256(s1)}
    properties["sandbox_id"] = {"const": source_id}
    properties["episode_id"] = {"const": episode_id}
    properties["deliverable"] = {"const": s1["task_contract"] | {
        "relative_path": s1["task_contract"]["final_artifact_relative_path"],
        "schema_sha256": s1["task_contract"]["final_artifact_schema_sha256"],
    }}
    properties["deliverable"]["const"].pop("final_artifact_relative_path", None)
    properties["deliverable"]["const"].pop("final_artifact_schema_sha256", None)
    properties["host_inputs"] = {"const": s1["host_inputs"]}

    def out_of_scope(_workspace: Any, _arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "stage": "S1",
            "gate_passed": False,
            "error": "training_successor_s1_boundary_only",
        }

    def materialize_plan(workspace: Any, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        errors = sorted(
            Draft202012Validator(plan_schema).iter_errors(canonical_value(arguments)),
            key=lambda error: tuple(str(part) for part in error.path),
        )
        if errors:
            return {
                "stage": "S1",
                "effect": "plan_materialization",
                "gate_passed": False,
                "attempt_consumed": False,
                "error": "schema_invalid_pre_effect",
                "diagnostic": errors[0].message,
            }
        workspace.write_bytes(
            "work/stage-plan.json", canonical_json_bytes(arguments), create_only=True
        )
        return {
            "stage": "S1",
            "effect": "plan_materialization",
            "gate_passed": True,
            "failed_check_ids": [],
            "attempt_consumed": True,
            "next_stage": "S2",
            "evidence_contract_sha256": _sha256(s2),
            "training_only": True,
        }

    selection_schema = _schema("evidence-selection.v1.schema.json")
    definitions = (
        ToolDefinition(
            name="execute_code",
            description="Execute bounded stage code after verified prerequisites.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["stage", "code"],
                "properties": {
                    "stage": {"enum": ["S3", "S4"]},
                    "code": {"type": "string", "minLength": 1, "maxLength": 1048576},
                },
            },
            handler=out_of_scope,
        ),
        ToolDefinition(
            name="materialize_evidence_selection",
            description="Materialize the bound evidence selection after retrieval.",
            input_schema=selection_schema,
            handler=out_of_scope,
        ),
        ToolDefinition(
            name="materialize_plan",
            description=(
                "Materialize the exact episode-bound S1 plan once before every "
                "later stage; only the declared authored prose fields may vary."
            ),
            input_schema=plan_schema,
            handler=materialize_plan,
        ),
        ToolDefinition(
            name="retrieve_frozen_evidence",
            description="Retrieve one bound frozen evidence object by evidence_id.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["evidence_id"],
                "properties": {
                    "evidence_id": {
                        "type": "string",
                        "enum": evidence_ids,
                        "pattern": "^[a-z][a-z0-9_.-]{2,127}$",
                    }
                },
            },
            handler=out_of_scope,
        ),
        ToolDefinition(
            name="submit_results",
            description="Submit only after the complete verified stage chain.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["terminal"],
                "properties": {"terminal": deepcopy(runtime["terminal"]["json_schema"])},
            },
            handler=out_of_scope,
        ),
    )
    registry = ToolRegistry(definitions)
    catalog_rows = []
    for offered in registry.public_schemas():
        function = offered["function"]
        catalog_rows.append(
            {
                "name": function["name"],
                "description": function["description"],
                "input_schema": function["parameters"],
                "visibility": "actor_public",
                "handler_origin": "signed_legacy_evamed",
                "parallel_safe": function["x-eva-parallel-safe"],
            }
        )
    tool_catalog_blake3 = blake3_hex(sorted(catalog_rows, key=lambda row: row["name"]))
    context = {
        "schema": RUNTIME_CONTEXT_SCHEMA,
        "source_candidate_id": source_id,
        "source_family": source_family,
        "focus": "S1",
        "target_split": "train",
        "active_episode_id": episode_id,
        "policy_blake3": blake3_hex(policy),
        "tool_catalog_blake3": tool_catalog_blake3,
        "s1_plan_contract": s1,
        "s2_evidence_contract": s2,
        "execution_stages": {
            "S3": {
                "stage": "S3",
                "artifact_relative_path": str(runtime["s3_artifact"]["relative_path"]),
                "artifact_json_schema": runtime["s3_artifact"]["json_schema"],
                "limits": runtime["execution_limits"],
            },
            "S4": {
                "stage": "S4",
                "artifact_relative_path": str(s4["relative_path"]),
                "artifact_json_schema": s4["json_schema"],
                "limits": runtime["execution_limits"],
            },
        },
        "s5_terminal_json_schema": runtime["terminal"]["json_schema"],
    }
    return MappingProxyType(context), registry


@dataclass(frozen=True)
class TrainingSuccessorBinding:
    candidate_id: str
    source_candidate_id: str
    public_runtime_context: Mapping[str, Any]
    initial_workspace_files: Mapping[str, bytes]
    tool_registry: ToolRegistry
    binding_blake3: str


class TrainingS1SuccessorContextPool:
    """Single-flight source reopen plus training-only S1 execution binding."""

    def __init__(
        self,
        *,
        source: Any | None = None,
        skills: Any | None = None,
        turn_mcp_factory: Any | None = None,
    ) -> None:
        if source is None or skills is None or turn_mcp_factory is None:
            source, _unused_resolver, skills, turn_mcp_factory = (
                _campaign_v2_context_resources()
            )
        self._source = source
        self._skills = skills
        self._turn_mcp = turn_mcp_factory
        self._cache: dict[str, tuple[TeacherCandidateContext, TrainingSuccessorBinding]] = {}
        self._inflight: dict[
            str, Future[tuple[TeacherCandidateContext, TrainingSuccessorBinding]]
        ] = {}
        self._lock = RLock()
        self._build_count = 0

    @property
    def context_build_count(self) -> int:
        with self._lock:
            return self._build_count

    def _build(
        self, record: Mapping[str, Any]
    ) -> tuple[TeacherCandidateContext, TrainingSuccessorBinding]:
        descriptor = record.get("training_successor")
        if not isinstance(descriptor, Mapping) or descriptor.get("schema") != ROW_SCHEMA:
            raise TeacherBatchError("training successor descriptor differs")
        predecessor = descriptor["predecessor"]
        job = self._source.load(str(predecessor["candidate_id"]))
        if (
            job.episode.stage is not Stage.S1
            or job.episode.domain != descriptor["domain"]
            or canonical_value(job.rubric.to_document())
            != canonical_value(record["reward_contract"]["rubric_table"])
        ):
            raise TeacherBatchError("training successor public task/rubric differs")
        materials = descriptor["public_materials"]
        construction_path = _safe_relative_file(
            SOURCE_ROOT, str(materials["construction_input_relative_path"])
        )
        evidence_path = _safe_relative_file(
            SOURCE_ROOT, str(materials["evidence_manifest_relative_path"])
        )
        if (
            blake3_bytes(construction_path.read_bytes())
            != materials["construction_input_blake3"]
            or blake3_bytes(evidence_path.read_bytes())
            != materials["evidence_manifest_blake3"]
        ):
            raise TeacherBatchError("training successor public material changed")
        construction = _strict_json(construction_path)
        runtime_context, registry = _runtime_materials(descriptor, construction)
        initial = dict(job.episode.initial_files)
        explicit = {
            ".eva/runtime-context.json": canonical_json_bytes(runtime_context),
            ".eva/training-successor.json": canonical_json_bytes(descriptor),
            ".eva/evidence-manifest.json": evidence_path.read_bytes(),
        }
        for path, payload in explicit.items():
            if path in initial:
                raise TeacherBatchError("training successor workspace collision")
            initial[path] = payload
        policy = dict(job.episode.policy_context)
        policy["execution_binding"] = runtime_context
        policy["public_reward_contract"] = {
            "rubric_table": canonical_value(job.rubric.to_document())
        }
        policy["training_successor"] = {
            "schema": ROW_SCHEMA,
            "training_only": True,
            "production_promotion_claimed": False,
            "predecessor_candidate_id": predecessor["candidate_id"],
            "successor_candidate_id": descriptor["candidate_id"],
        }
        instruction, rephrased = teacher_safe_actor_instruction(job.episode.instruction)
        policy["teacher_actor_projection"] = {
            "judge_only_material_included": False,
            "public_reward_contract_included": True,
            "privacy_reserved_instruction_lexemes_rephrased": rephrased,
        }
        guidance = build_stage_tool_guidance_v1(
            public_runtime_context=runtime_context,
            source_tool_catalog=registry.public_schemas(),
        )
        policy["stage_tool_guidance_binding"] = {
            "schema": "eva.codex-stage-tool-guidance.v1",
            "source_candidate_id": guidance.source_candidate_id,
            "guidance_blake3": guidance.guidance_blake3,
            "prompt_blake3": guidance.prompt_blake3,
            "public_runtime_context_blake3": guidance.public_runtime_context_blake3,
            "source_tool_catalog_blake3": guidance.source_tool_catalog_blake3,
            "delivery": "codex-developer-instruction-sidecar",
            "canonical_tool_schemas_changed": False,
        }
        episode = replace(
            job.episode,
            episode_id=str(descriptor["episode_id"]),
            instruction=instruction,
            initial_files=initial,
            policy_context=policy,
        )
        teacher_registry = self._skills.augment_registry(registry, Stage.S1)
        core = {
            "schema": "eva.training-only-s1-execution-binding.v1",
            "candidate_id": descriptor["candidate_id"],
            "source_candidate_id": descriptor["source_candidate_id"],
            "runtime_context_blake3": blake3_hex(runtime_context),
            "tool_catalog_blake3": runtime_context["tool_catalog_blake3"],
            "initial_files": {
                path: blake3_bytes(payload) for path, payload in sorted(explicit.items())
            },
            "training_only": True,
            "production_promotion_claimed": False,
        }
        binding = TrainingSuccessorBinding(
            candidate_id=str(descriptor["candidate_id"]),
            source_candidate_id=str(descriptor["source_candidate_id"]),
            public_runtime_context=runtime_context,
            initial_workspace_files=MappingProxyType(explicit),
            tool_registry=registry,
            binding_blake3=blake3_hex(core),
        )
        return (
            TeacherCandidateContext(
                episode=episode,
                rubric=job.rubric,
                tool_registry=teacher_registry,
                turn_mcp_factory=self._turn_mcp,
                skills_factory=self._skills,
                skill_delivery=self._skills.public_metadata(Stage.S1),
                stage_tool_guidance=guidance,
            ),
            binding,
        )

    def load(
        self, record: Mapping[str, Any]
    ) -> tuple[TeacherCandidateContext, TrainingSuccessorBinding]:
        candidate_id = str(record.get("candidate_id", ""))
        if not candidate_id:
            raise TeacherBatchError("training successor identity differs")
        leader = False
        with self._lock:
            cached = self._cache.get(candidate_id)
            if cached is not None:
                return cached
            pending = self._inflight.get(candidate_id)
            if pending is None:
                pending = Future()
                self._inflight[candidate_id] = pending
                leader = True
        if not leader:
            return pending.result()
        try:
            result = self._build(record)
            with self._lock:
                self._cache[candidate_id] = result
                self._build_count += 1
            pending.set_result(result)
            return result
        except BaseException as exc:
            pending.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._inflight.pop(candidate_id, None)


def provider_free_execute_context(
    *, context: TeacherCandidateContext, workspace_root: Path, sandbox_id: str
) -> Mapping[str, Any]:
    """Execute the exact guided S1 frontier without crossing a provider boundary."""

    from eva_agent.pipeline import FilesystemSandbox, ParallelToolRuntime, RandomUUIDFactory, ToolCall

    guidance = context.stage_tool_guidance
    if guidance is None or guidance.focus is not Stage.S1:
        raise TeacherBatchError("training successor S1 guidance differs")
    frontier = guidance.frontiers[0]
    workspace = FilesystemSandbox(
        workspace_root,
        sandbox_id,
        context.episode.initial_files,
    )
    before = workspace.snapshot("before-fake-provider")
    ids = RandomUUIDFactory()
    runtime = ParallelToolRuntime(
        workspace=workspace,
        registry=context.tool_registry,
        id_factory=ids,
        maximum_parallel_calls=64,
    )
    results = runtime.execute(
        (
            ToolCall(
                call_id=ids.new("training-successor-fake-provider-call"),
                name="materialize_plan",
                arguments=frontier["arguments"],
            ),
        )
    )
    after = workspace.snapshot("after-fake-provider")
    result = results[0]
    output = result.output
    if (
        result.status != "completed"
        or result.error_code is not None
        or not isinstance(output, Mapping)
        or output.get("gate_passed") is not True
        or output.get("failed_check_ids") != ()
        or before.tree_blake3 == after.tree_blake3
        or not any(row.path == "work/stage-plan.json" for row in after.files)
    ):
        raise TeacherBatchError("training successor fake-provider S1 gate failed")
    return MappingProxyType(
        {
            "schema": "eva.training-only-s1-provider-free-check.v1",
            "external_provider_calls": 0,
            "fake_provider_calls": 1,
            "gate_passed": True,
            "workspace_before_blake3": before.tree_blake3,
            "workspace_after_blake3": after.tree_blake3,
            "tool_receipt_blake3": result.receipt_blake3,
        }
    )


__all__ = [
    "BUNDLE_SCHEMA",
    "ROW_SCHEMA",
    "TrainingS1SuccessorContextPool",
    "build_training_successor_bundle",
    "load_training_successor_records",
    "provider_free_execute_context",
    "verify_training_successor_bundle",
    "write_training_successor_bundle",
]
