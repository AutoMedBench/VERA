"""Build and reopen a self-contained release-bound execution data bundle."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4
from .contracts import PRIMARY_TOOLS, SUPPORTED_CHECKS, tools_for_case
from .integrity import byte_digest, canonical, contained_file, digest, file_digest, relative_path, strict_json, verify_upstream, write_json


def build(*, dataset, portable_root, source_tools_path, output, runtime_code_root, signer, support_files=None):
    dataset, portable_root, output = Path(dataset).resolve(), Path(portable_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "bundle.json").exists():
        raise ValueError("bundle_already_exists")
    source_inventory = strict_json(Path(source_tools_path).read_bytes())
    tools = {name: source_inventory["schemas"][name] for name in PRIMARY_TOOLS}
    write_json(output / "catalog/source-tools.json", tools)
    shutil.copy2(source_tools_path, output / "catalog/released-tool-schema-inventory.json")
    cases, strata, evidence_blobs = [], Counter(), {}
    revision = "5d11d71a6d0a9f0df3d3976a65d6f7351e55174b"
    for shard in sorted((dataset / "data").glob("train-*.jsonl")):
        for line_number, raw in enumerate(shard.open("rb"), 1):
            row = strict_json(raw)
            if row["record_blake3"] != digest({k: v for k, v in row.items() if k != "record_blake3"}):
                raise ValueError("record_commitment_mismatch")
            case_id = row["sandbox_id"]
            relative_path(case_id)
            source = contained_file(portable_root / "rlevo-med-research", row["source_binding"]["authority_relative_path"])
            source_raw = source.read_bytes()
            if hashlib.sha256(source_raw).hexdigest() != row["source_binding"]["upstream_source_artifact_sha256"]:
                raise ValueError("construction_source_commitment")
            construction = strict_json(source_raw)
            for stage in ("s3_artifact", "s4_artifact", "terminal"):
                if not {c["op"] for c in construction["runtime"][stage]["checks"]} <= SUPPORTED_CHECKS:
                    raise ValueError("unimplemented_source_check")
                if stage != "terminal" and relative_path(construction["runtime"][stage]["relative_path"]).parts[0] != "work":
                    raise ValueError("source_artifact_not_work_relative")
            case_root = output / "cases" / case_id
            case_root.mkdir(parents=True, exist_ok=True)
            (case_root / "record.json").write_bytes(raw)
            (case_root / "construction.json").write_bytes(source_raw)
            evidence = []
            for item in construction["evidence_objects"]:
                if len(relative_path(item["evidence_id"]).parts) != 1:
                    raise ValueError("source_evidence_id_not_single_component")
                path = contained_file(portable_root / "rlevo-med-research", item["source_relative_path"])
                data = verify_upstream(path, size=item["byte_count"], sha256=item["sha256"])
                blob = byte_digest(data)
                target = output / "evidence" / (blob + ".json")
                if blob not in evidence_blobs:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                    evidence_blobs[blob] = len(data)
                evidence.append({"evidence_id": item["evidence_id"], "path": str(target.relative_to(output)),
                                 "bytes": len(data), "blake3": blob, "upstream_sha256": item["sha256"],
                                 "content_kind": item["content_kind"], "original_source_path": item["source_relative_path"]})
            descriptor = {"case_id": case_id, "domain": row["domain"], "stage": row["stage"],
                          "source_family": row["source_family"], "record_path": str((case_root / "record.json").relative_to(output)),
                          "record_file_blake3": byte_digest(raw), "record_blake3": row["record_blake3"],
                          "construction_path": str((case_root / "construction.json").relative_to(output)),
                          "construction_blake3": byte_digest(source_raw), "evidence": evidence,
                          "tool_catalog_blake3": digest(tools_for_case(tools, construction)),
                          "rubric_digest": row["reward_contract"]["rubric_digest"],
                          "source_location": {"shard": shard.name, "line": line_number}}
            cases.append(descriptor)
            strata[(row["domain"], row["stage"])] += 1
        print(f"Bound {len(cases)} source cases", flush=True)
    if len(cases) != 6000 or len(strata) != 42:
        raise ValueError("released_case_inventory_differs")
    code_target = output / "runtime/evamed_portable"
    shutil.copytree(runtime_code_root, code_target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    code = [{"path": str(p.relative_to(output)), "blake3": file_digest(p), "bytes": p.stat().st_size}
            for p in sorted(code_target.rglob("*")) if p.is_file()]
    support = []
    for relative, source in (support_files or {}).items():
        relative_path(relative)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        support.append({"path": relative, "blake3": file_digest(target), "bytes": target.stat().st_size})
    manifest = {"schema": "eva.medresearch-portable-bundle.v1", "bundle_id": str(uuid4()),
                "name": "EVA-medresearch-v1", "source_dataset": "operator/EVA-Med-RL-data",
                "source_revision": revision, "fresh_execution_authority": {"key_id": signer.key_id, "public_key_base64": signer.public},
                "historical_authorities_modified": False, "historical_execution_reconstructed": False,
                "runtime_profile": "bubblewrap-seccomp-single-process-v1",
                "source_policy_limits_preserved": True, "multiprocessing_supported": False,
                "source_tool_catalog_path": "catalog/source-tools.json", "source_tool_catalog_blake3": digest(tools),
                "cases": cases, "case_count": len(cases), "unique_evidence_blobs": len(evidence_blobs),
                "evidence_bytes": sum(evidence_blobs.values()), "runtime_code": code, "support_files": support,
                "strata": [{"domain": d, "stage": s, "count": n} for (d, s), n in sorted(strata.items())],
                "admission": {"requires_backend_acceptance": True, "fixture_runs_are_not_api_rollouts": True,
                              "clinical_or_ability_scores_inferred": False,
                              "fresh_S1_S2_semantics": "Explicit versioned host sidecar; original historical effect handlers unavailable."}}
    manifest["document_blake3"] = digest(manifest)
    write_json(output / "bundle.json", manifest)
    write_json(output / "bundle-signature.json", signer.sign({"bundle_id": manifest["bundle_id"],
                                                              "manifest_blake3": manifest["document_blake3"]}))
    return manifest


class Bundle:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.manifest = strict_json(contained_file(self.root, "bundle.json").read_bytes())
        if self.manifest["document_blake3"] != digest({k: v for k, v in self.manifest.items() if k != "document_blake3"}):
            raise ValueError("bundle_manifest_commitment")
        from .integrity import verify_receipt
        signed = strict_json(contained_file(self.root, "bundle-signature.json").read_bytes())
        authority = self.manifest["fresh_execution_authority"]
        payload = verify_receipt(signed, authority["public_key_base64"])
        if payload != {"bundle_id": self.manifest["bundle_id"], "manifest_blake3": self.manifest["document_blake3"]}:
            raise ValueError("bundle_signature_binding")
        self.tools = strict_json(contained_file(self.root, self.manifest["source_tool_catalog_path"]).read_bytes())
        if digest(self.tools) != self.manifest["source_tool_catalog_blake3"]:
            raise ValueError("canonical_tool_catalog_commitment")
        self.cases = {c["case_id"]: c for c in self.manifest["cases"]}
        if len(self.cases) != self.manifest["case_count"]:
            raise ValueError("duplicate_source_case_identity")
        for item in self.manifest["runtime_code"] + self.manifest.get("support_files", []):
            path = contained_file(self.root, item["path"])
            if path.stat().st_size != item["bytes"] or file_digest(path) != item["blake3"]:
                raise ValueError("runtime_or_support_commitment")

    def case(self, case_id):
        descriptor = self.cases[case_id]
        row_bytes = contained_file(self.root, descriptor["record_path"]).read_bytes()
        construction_bytes = contained_file(self.root, descriptor["construction_path"]).read_bytes()
        if byte_digest(row_bytes) != descriptor["record_file_blake3"] or byte_digest(construction_bytes) != descriptor["construction_blake3"]:
            raise ValueError("case_source_tampered")
        row, construction = strict_json(row_bytes), strict_json(construction_bytes)
        definitions = tools_for_case(self.tools, construction)
        if digest(definitions) != descriptor["tool_catalog_blake3"] or row["reward_contract"]["rubric_digest"] != descriptor["rubric_digest"]:
            raise ValueError("case_contract_tampered")
        return descriptor, row, construction, definitions
