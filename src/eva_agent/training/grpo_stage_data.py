"""Stage-targeted sampling from the released RL corpus; no generated rewards."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path


class StageDataCoverageError(ValueError):
    """An explicitly requested domain cannot be covered; no training rows emitted."""

    def __init__(self, coverage):
        self.coverage = coverage
        super().__init__("signed executable coverage is incomplete: " +
                         ", ".join(coverage["uncovered_requested_domains"]))


def select_stage_rows(rows, *, stage: str, limit: int, domains=()):
    if stage not in {"S1", "S2", "S3"} or type(limit) is not int or limit < 1:
        raise ValueError("only executable S1-S3 stage targets and positive limits are supported")
    selected, seen = defaultdict(list), set()
    for row in rows:
        if row.get("stage") != stage or row.get("split") != "train":
            continue
        if domains and row.get("domain") not in domains:
            continue
        sandbox_id = row.get("sandbox_id")
        if not isinstance(sandbox_id, str) or not sandbox_id or sandbox_id in seen:
            raise ValueError("selected stage contains duplicate or absent sandbox identities")
        rubric = row["reward_contract"]["rubric_table"]
        if rubric["stage"] != stage or rubric["domain"] != row["domain"]:
            raise ValueError("sandbox and exact reward rubric domain/stage differ")
        seen.add(sandbox_id)
        selected[row["domain"]].append(row)
    if not selected:
        raise ValueError("no training sandboxes match the observed stage/domain target")
    for members in selected.values():
        members.sort(key=lambda row: row["sandbox_id"])
    result = []
    # Round-robin domains: a high-volume source does not consume the entire batch.
    cursor = 0
    while len(result) < min(limit, len(seen)):
        for domain in sorted(selected):
            if cursor < len(selected[domain]) and len(result) < limit:
                result.append(selected[domain][cursor])
        cursor += 1
    return result


def select_executable_stage_rows(rows, *, stage: str, limit: int, domains=(),
                                 execution_catalog_path: Path, trust_store_path: Path):
    """Opt-in signed membership filter, not a live execution-success assertion.

    Only exact original primary/train/executable_legacy rows are supported by the
    current LegacyExecutionBindingResolver. No successor or premium substitution.
    Source artifacts are not opened or whole-corpus hashed here. The caller must
    use an already verified bulk release; per-target record commitments and all
    signed source/domain/stage/rubric identities are checked before selection.
    """
    from eva_agent.admission.receipts import SignedEnvelope, verify_signed_envelope
    from eva_agent.pipeline.digests import blake3_hex

    requested = tuple(domains)
    if (isinstance(domains, str) or len(set(requested)) != len(requested)
            or any(not isinstance(name, str) or not name for name in requested)):
        raise ValueError("requested domain inventory differs")
    if stage not in {"S1", "S2", "S3"} or type(limit) is not int or limit < 1:
        raise ValueError("only executable S1-S3 stage targets and positive limits are supported")
    envelope = SignedEnvelope.from_document(json.loads(Path(execution_catalog_path).read_text()))
    verify_signed_envelope(envelope, trust_store_path=Path(trust_store_path))
    payload = envelope.payload
    if (payload.get("schema") != "eva.prospective-execution-binding-catalog.v3"
            or not isinstance(payload.get("rows"), (tuple, list))
            or blake3_hex({key: value for key, value in payload.items() if key != "catalog_blake3"})
            != payload.get("catalog_blake3")):
        raise ValueError("signed executable catalog schema or content differs")
    catalog = {}
    for member in payload["rows"]:
        identity = member.get("candidate_id")
        if not isinstance(identity, str) or not identity or identity in catalog:
            raise ValueError("signed catalog candidate identity differs")
        catalog[identity] = member
    eligible, exclusions, seen = [], [], set()
    matched_counts, eligible_counts = Counter(), Counter()
    for row in rows:
        if (row.get("stage") != stage or row.get("split") != "train"
                or requested and row.get("domain") not in requested):
            continue
        sandbox = row.get("sandbox_id")
        if not isinstance(sandbox, str) or not sandbox or sandbox in seen:
            raise ValueError("selected stage contains duplicate or absent sandbox identities")
        seen.add(sandbox)
        member = catalog.get(row.get("candidate_id"))
        if member is None:
            raise ValueError("bulk candidate is absent from signed execution catalog")
        binding = row.get("source_binding", {})
        rubric = row.get("reward_contract", {}).get("rubric_table", {})
        expected = {"candidate_id": row.get("candidate_id"), "domain": row.get("domain"),
            "stage": stage, "split": "train", "source_family": row.get("source_family"),
            "source_candidate_id": binding.get("source_candidate_id"),
            "source_artifact_sha256": binding.get("upstream_source_artifact_sha256"),
            "source_registry_blake3": binding.get("source_registry_blake3"),
            "rubric_id": rubric.get("rubric_id"), "rubric_blake3": rubric.get("rubric_digest")}
        if (any(value is None or member.get(key) != value for key, value in expected.items())
                or any(binding.get(key) != row.get(key) for key in ("candidate_id", "domain", "stage", "source_family"))
                or rubric.get("stage") != stage or rubric.get("domain") != row.get("domain")
                or row.get("lineage", {}).get("selection_blake3") != payload.get("selection_blake3")
                or blake3_hex({key: value for key, value in row.items() if key != "record_blake3"}) != row.get("record_blake3")):
            raise ValueError("bulk record and signed execution source/rubric binding differ")
        domain = row["domain"]
        matched_counts[domain] += 1
        reason = None
        if member.get("selection_tier") != "primary":
            reason = "not_primary"
        elif member.get("execution_status") != "executable_legacy":
            reason = "execution_status:" + str(member.get("execution_status"))
        else:
            proof = member.get("execution_proof") or {}
            if (proof.get("kind") != "signed_v24_legacy_promoted"
                    or proof.get("proof_root_blake3") != member.get("readiness_proof_root_blake3")
                    or member.get("construction_readiness") != "promoted"):
                raise ValueError("signed legacy executable proof kind differs")
        if reason:
            exclusions.append({"sandbox_id": sandbox, "candidate_id": row["candidate_id"],
                               "domain": domain, "reason": reason})
        else:
            eligible.append(row)
            eligible_counts[domain] += 1
    selected = select_stage_rows(eligible, stage=stage, limit=limit, domains=requested) if eligible else []
    selected_counts = Counter(row["domain"] for row in selected)
    coverage = {"schema": "eva.grpo-signed-executable-selection.v1", "stage": stage,
        "catalog_envelope_blake3": envelope.envelope_blake3,
        "catalog_blake3": payload["catalog_blake3"], "catalog_signature_verified": True,
        "requested_domains": sorted(requested), "requested_limit": limit,
        "target_match_count": sum(matched_counts.values()), "eligible_count": len(eligible),
        "selected_count": len(selected), "limit_shortfall": max(0, limit - len(selected)),
        "domain_coverage": {domain: {"matching_rows": matched_counts[domain],
            "executable_legacy_rows": eligible_counts[domain], "selected_rows": selected_counts[domain]}
            for domain in sorted(set(requested) | set(matched_counts))},
        "uncovered_requested_domains": sorted(set(requested) - set(selected_counts)),
        "exclusions": exclusions, "excluded_count": len(exclusions),
        "excluded_reason_counts": dict(sorted(Counter(row["reason"] for row in exclusions).items())),
        "eligible_not_selected_due_to_limit": len(eligible) - len(selected),
        "selected_sandbox_ids": [row["sandbox_id"] for row in selected],
        "source_records_changed": False, "resolver_bindings_reopened": False,
        "actor_execution_verified": False, "provider_calls": 0}
    if coverage["uncovered_requested_domains"] or not selected:
        raise StageDataCoverageError(coverage)
    return selected, coverage


def prepare_stage_data(bulk_root: Path, output_root: Path, *, stage: str, limit: int = 128,
                       domains=(), actor_backend="native_codex_sglang", judge_backend="native_astra",
                       execution_catalog_path: Path | None = None,
                       trust_store_path: Path | None = None):
    if actor_backend not in {"native_codex_sglang", "sglang_eva_tools"}:
        raise ValueError("explicit supported actor backend is required")
    if judge_backend not in {"native_astra", "opus_5"}:
        raise ValueError("explicit supported workspace agent judge is required")
    bulk_root = bulk_root.resolve(strict=True)
    manifest = json.loads((bulk_root / "manifest.json").read_text())

    def records():
        for descriptor in manifest["payload"]["shards"]:
            path = (bulk_root / descriptor["path"]).resolve(strict=True)
            if not path.is_relative_to(bulk_root):
                raise ValueError("bulk shard is outside the release root")
            with path.open() as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)

    if (execution_catalog_path is None) != (trust_store_path is None):
        raise ValueError("signed filtering requires both execution catalog and trust store")
    selection = None
    if execution_catalog_path is not None:
        try:
            rows, selection = select_executable_stage_rows(records(), stage=stage,
                limit=limit, domains=domains, execution_catalog_path=execution_catalog_path,
                trust_store_path=trust_store_path)
        except StageDataCoverageError as exc:
            output_root.mkdir(parents=True, exist_ok=False)
            with (output_root / "preparation-receipt.json").open("x") as stream:
                json.dump({"schema": "eva.grpo-stage-target-data.v2", "status": "blocked",
                    "stage": stage, "sample_count": 0, "training_data_written": False,
                    "selection": exc.coverage, "provider_calls": 0}, stream, indent=2)
                stream.write("\n")
            raise
    else:
        rows = select_stage_rows(records(), stage=stage, limit=limit, domains=domains)
    output_root.mkdir(parents=True, exist_ok=False)
    with (output_root / "grpo.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps({
                "prompt": f"Complete the original {stage} medical research sandbox {row['sandbox_id']}.",
                "metadata": {"sandbox_id": row["sandbox_id"], "bulk_root": str(bulk_root),
                    "candidate_id": row["candidate_id"], "stage": stage, "domain": row["domain"],
                    "rubric_digest": row["reward_contract"]["rubric_table"]["rubric_digest"],
                    "actor_backend": actor_backend, "judge_backend": judge_backend},
            }, ensure_ascii=False) + "\n")
    receipt = {"schema": "eva.grpo-stage-target-data.v1", "stage": stage,
               "sample_count": len(rows), "domains": sorted({row["domain"] for row in rows}),
               "actor_backend_requested": actor_backend, "actor_execution_verified": False,
               "judge_backend": judge_backend, "online_generation_required": True,
               "shared_rubric_reward": True, "precomputed_rewards": False, "provider_calls": 0}
    if selection is not None:
        receipt.update(schema="eva.grpo-stage-target-data.v2", status="prepared",
                       selection=selection, training_data_written=True)
    with (output_root / "preparation-receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt
