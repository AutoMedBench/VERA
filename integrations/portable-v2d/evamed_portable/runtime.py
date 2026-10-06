"""Five schema-preserving tools over a fresh, signed S1–S5 execution chain."""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
import time
from uuid import uuid4
from .bundle import Bundle
from .contracts import evaluate_artifact, public_schema_diagnostics, validate_arguments
from .integrity import (Signer, byte_digest, canonical, contained_file, digest, file_digest,
                        relative_path, strict_json, timestamp, verify_receipt, write_json)
from .sandbox import execute


FRESH_SEMANTICS = {
    "schema": "eva.medresearch-fresh-stage-semantics.v1",
    "historical_handler_equivalence_claimed": False,
    "canonical_input_schemas_modified": False,
    "completion_policy": "Every fresh execution completes S1,S2,S3,S4,S5; source stage is the focus label. No predecessor is fabricated.",
    "S1": "materialize_plan supplies a nonempty objective and five steps, each with stage S1-S5 and nonempty action. Host binds the supplied plan to this source case and fresh execution.",
    "S2": "Retrieve every immutable declared evidence_id, then materialize_evidence_selection with exactly those evidence_ids and a nonempty rationale. The host verifies all evidence hashes.",
    "S3": "execute_code(stage=S3) starts a fresh isolated CPU workspace, runs real code against public evidence, and validates its declared pilot artifact with unchanged source schema/checks.",
    "S4": "execute_code(stage=S4) starts another fresh isolated CPU workspace, receives verified read-only S1/S2 metadata and S3 artifact, and validates the new output against unchanged source schema/checks.",
    "S5": "submit_results independently reopens source-bound S3/S4 artifacts and signed prerequisites, validates the exact per-case terminal schema/checks, and closes the run once. No clinical or ability score is inferred.",
    "artifact_gate": "All unchanged source checks must pass; stricter than a critical-check-only gate. Public diagnostics never include hidden expected values.",
    "domain_tool_names": "The five released primary interfaces are the only implemented data-plane tools. Domain-specific names appearing in historical objectives are not advertised as implemented aliases.",
    "execution_profile": "Single-process CPU only; fork and thread creation denied, no network/GPU/host proc. Source limits are retained alongside actual stricter backend controls.",
}


def fresh_semantics(construction):
    if "prospective_public_artifact_bindings" not in construction:
        return FRESH_SEMANTICS
    value = deepcopy(FRESH_SEMANTICS)
    value.update({
        "schema": "eva.medresearch-fresh-stage-semantics.public-bindings.v1",
        "canonical_input_schemas_modified": True,
        "source_artifact_metadata_checks_revised": True,
        "S3": "Run real code in a fresh isolated CPU workspace; validate original numerical checks and declared prospective public metadata bindings.",
        "S4": "Independently recompute in a fresh CPU workspace with verified S1/S2/S3 handoffs; retain original numerical and invariant checks and prospective public metadata bindings.",
        "S5": "Independently reopen S3/S4, validate source terminal safety checks and declared prospective public metadata bindings, and close the run once.",
        "artifact_gate": "All numerical, invariant, confidentiality and isolation obligations are retained. Seven metadata equality checks across S3/S4/S5 use explicit public derivations. Hidden numerical expected values remain private.",
        "task_contract_revision": construction["prospective_public_artifact_bindings"]["schema"],
        "historical_benchmark_equivalence_claimed": False,
    })
    return value


class Runtime:
    def __init__(self, *, bundle, run_root, signer, bwrap, runtime_root):
        self.bundle = bundle if isinstance(bundle, Bundle) else Bundle(bundle)
        self.root = Path(run_root).resolve(strict=True)
        self.signer = signer if isinstance(signer, Signer) else Signer(signer)
        self.bwrap, self.runtime_root = Path(bwrap), Path(runtime_root)
        self.host = self.root / "host"
        state = self._load_state()
        self.descriptor, self.row, self.construction, self.definitions = self.bundle.case(state["case_id"])
        if state["bundle_blake3"] != self.bundle.manifest["document_blake3"]:
            raise ValueError("run_bundle_binding")
        self._verify_public(state)

    @classmethod
    def create(cls, *, bundle, case_id, runs_root, signer, bwrap, runtime_root):
        bundle = bundle if isinstance(bundle, Bundle) else Bundle(bundle)
        descriptor, row, construction, definitions = bundle.case(case_id)
        signer = signer if isinstance(signer, Signer) else Signer(signer)
        run_id = str(uuid4())
        root = Path(runs_root).resolve() / run_id
        root.mkdir(parents=True, mode=0o700)
        (root / "host").mkdir(mode=0o700)
        (root / "public").mkdir(mode=0o700)
        semantics = fresh_semantics(construction)
        write_json(root / "public/semantics.json", semantics)
        public = {"schema": "eva.medresearch-public-task.v1", "run_id": run_id,
                  "domain": row["domain"], "focus_stage": row["stage"], "completion_boundary": "fresh_S1_through_S5",
                  "source_stage_completion_boundary": next(f["content"]["completion_boundary"] for f in row["workspace_initial_state"]["files"] if f["path"] == "input/task-contract.json"),
                  "task_brief": construction["task_brief"], "source_sandbox": construction["sandbox"],
                  "template_episode_id": construction["runtime"]["template_episode_id"],
                  "source_policy_budgets": construction["runtime"]["policy_budgets"],
                  "source_execution_limits": construction["runtime"]["execution_limits"],
                  "evidence_inventory": [{k: e[k] for k in ("evidence_id", "bytes", "blake3", "upstream_sha256", "content_kind")} for e in descriptor["evidence"]],
                  "artifact_contracts": {stage: {k: v for k, v in construction["runtime"][name].items() if k in {"json_schema", "relative_path", "min_bytes", "max_bytes"}}
                                         for stage, name in (("S3", "s3_artifact"), ("S4", "s4_artifact"), ("S5", "terminal"))},
                  "primary_tool_definitions": list(definitions.values()),
                  "public_evidence_path_pattern": "input/evidence/<evidence_id>.json",
                  "judge_only_contracts_visible": False}
        if "prospective_public_artifact_bindings" in construction:
            public["prospective_artifact_bindings"] = construction["prospective_public_artifact_bindings"]
            public["source_checks_preserved_except_declared_metadata_bindings"] = True
        write_json(root / "public/task.json", public)
        (root / "public/task.json").chmod(0o400)
        (root / "public/semantics.json").chmod(0o400)
        write_json(root / "host/source-reward-contract.json", row["reward_contract"])
        state = {"schema": "eva.medresearch-fresh-run-state.v1", "run_id": run_id, "case_id": case_id,
                 "bundle_blake3": bundle.manifest["document_blake3"], "source_record_blake3": row["record_blake3"],
                 "source_rubric_digest": row["reward_contract"]["rubric_digest"], "source_policy_limits": construction["runtime"]["policy_budgets"],
                 "fresh_semantics_blake3": digest(semantics), "started_at": timestamp(), "started_epoch": time.time(),
                 "public_task_blake3": digest(public),
                 "next_stage": "S1", "completed": {}, "retrieved": [], "tool_calls": 0,
                 "submission_attempted": False, "events": [], "fresh_key_id": signer.key_id,
                 "clinical_or_ability_scores_computed": False}
        write_json(root / "host/state.json", signer.sign(state))
        return cls(bundle=bundle, run_root=root, signer=signer, bwrap=bwrap, runtime_root=runtime_root)

    def _load_state(self):
        return verify_receipt(strict_json((self.host / "state.json").read_bytes()), self.signer.public)

    def _verify_public(self, state):
        if file_digest(contained_file(self.root, "public/task.json")) != state["public_task_blake3"] or file_digest(contained_file(self.root, "public/semantics.json")) != state["fresh_semantics_blake3"]:
            raise ValueError("public_task_projection_tampered")

    @contextmanager
    def _locked(self):
        with (self.host / "runtime.lock").open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield self._load_state()
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _event(self, state, *, tool, arguments, result, details=None):
        identity = str(uuid4())
        payload = {"event_id": identity, "run_id": state["run_id"], "case_id": state["case_id"],
                   "ordinal": len(state["events"]) + 1, "at": timestamp(), "tool": tool,
                   "arguments_blake3": digest(arguments), "public_result": deepcopy(result), "details": details or {},
                   "previous_event_blake3": state["events"][-1]["blake3"] if state["events"] else None,
                   "source_record_blake3": state["source_record_blake3"], "source_rubric_digest": state["source_rubric_digest"],
                   "fresh_semantics_blake3": state["fresh_semantics_blake3"]}
        receipt = self.signer.sign(payload)
        write_json(self.host / "events" / (identity + ".json"), receipt, exclusive=True)
        state["events"].append({"event_id": identity, "blake3": digest(receipt)})
        result.update(receipt_id=identity, receipt_blake3=digest(receipt))
        write_json(self.host / "state.json", self.signer.sign(state))
        return result

    def _reopen(self, state, stage):
        self._verify_events(state)
        previous = state["completed"][stage]
        path = contained_file(self.host, previous["artifact_path"])
        if file_digest(path) != previous["artifact_file_blake3"]:
            raise ValueError("signed_stage_artifact_tampered")
        return strict_json(path.read_bytes())

    def _verify_events(self, state):
        previous = None
        for ordinal, item in enumerate(state["events"], 1):
            receipt = strict_json(contained_file(self.host, "events/" + item["event_id"] + ".json").read_bytes())
            payload = verify_receipt(receipt, self.signer.public)
            if digest(receipt) != item["blake3"] or payload["run_id"] != state["run_id"] or payload["ordinal"] != ordinal or payload["previous_event_blake3"] != previous:
                raise ValueError("fresh_event_chain_tampered")
            previous = item["blake3"]

    def public_snapshot(self):
        """Actor-visible state only; excludes rubric, private checks, paths, keys."""
        with self._locked() as state:
            self._verify_events(state)
            self._verify_public(state)
            return {"run_id": state["run_id"], "next_stage": state["next_stage"],
                    "completed_stages": sorted(state["completed"]), "tool_calls": state["tool_calls"],
                    "retrieved_evidence_ids": list(state["retrieved"]),
                    "submission_attempted": state["submission_attempted"],
                    "task": strict_json((self.root / "public/task.json").read_bytes()),
                    "semantics": strict_json((self.root / "public/semantics.json").read_bytes())}

    def finalize(self):
        """Host-only signed disposition; reports gates, never an inferred score."""
        with self._locked() as state:
            self._verify_events(state)
            reopened = {}
            for stage in sorted(state["completed"]):
                artifact = self._reopen(state, stage)
                if stage in {"S3", "S4", "S5"}:
                    spec = self.construction["runtime"][{"S3": "s3_artifact", "S4": "s4_artifact", "S5": "terminal"}[stage]]
                    reopened[stage] = evaluate_artifact(artifact, spec)
            complete = state["next_stage"] == "complete" and set(state["completed"]) == {"S1", "S2", "S3", "S4", "S5"} and all(v["gate_passed"] for v in reopened.values())
            receipt = self.signer.sign({"schema": "eva.medresearch-fresh-final-disposition.v1", "run_id": state["run_id"],
                "case_id": state["case_id"], "bundle_blake3": state["bundle_blake3"],
                "source_record_blake3": state["source_record_blake3"], "source_rubric_digest": state["source_rubric_digest"],
                "fresh_semantics_blake3": state["fresh_semantics_blake3"], "event_count": len(state["events"]),
                "last_event_blake3": state["events"][-1]["blake3"] if state["events"] else None,
                "all_stage_gates_passed": complete, "disposition": "completed" if complete else "incomplete",
                "completed_stages": sorted(state["completed"]), "independent_reopen": reopened,
                "clinical_or_ability_score": None, "s_target_evidence": False})
            write_json(self.host / "final-disposition.json", receipt)
            return receipt

    def call(self, tool, arguments):
        if tool not in self.definitions:
            raise ValueError("unknown_canonical_tool")
        validate_arguments(self.definitions[tool], arguments)
        if len(canonical(arguments)) > self.construction["runtime"]["execution_limits"]["max_submission_bytes"]:
            raise ValueError("source_submission_bytes_exceeded")
        with self._locked() as state:
            budget = self.construction["runtime"]["policy_budgets"]
            if state["next_stage"] == "complete" or state["tool_calls"] >= budget["max_turns"] or time.time() - state["started_epoch"] > budget["wall_time_seconds"]:
                raise ValueError("fresh_run_closed_or_budget_exhausted")
            if tool != "submit_results" and budget["max_turns"] - state["tool_calls"] <= budget["minimum_s5_reserved_turns"]:
                raise ValueError("source_S5_reserved_turns")
            state["tool_calls"] += 1
            try:
                result, details = getattr(self, "_" + tool)(state, arguments)
            except (ValueError, OSError, RuntimeError) as error:
                result, details = {"gate_passed": False, "error": str(error) if type(error) is ValueError else type(error).__name__,
                                   "attempt_consumed": True, "next_stage": state["next_stage"]}, {"failure_type": type(error).__name__}
            return self._event(state, tool=tool, arguments=arguments, result=result, details=details)

    def _store_stage(self, state, stage, document, *, raw=None):
        relative = "accepted/" + stage + ".json"
        path = self.host / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw if raw is not None else canonical(document))
        path.chmod(0o400)
        state["completed"][stage] = {"artifact_path": relative, "artifact_file_blake3": file_digest(path)}

    def _materialize_plan(self, state, args):
        if state["next_stage"] != "S1":
            raise ValueError("S1_prerequisite_or_once_boundary")
        steps = args.get("steps")
        if not isinstance(args.get("objective"), str) or not args["objective"].strip() or not isinstance(steps, list) or len(steps) != 5:
            raise ValueError("fresh_plan_requires_objective_and_five_steps")
        for stage, step in zip(("S1", "S2", "S3", "S4", "S5"), steps):
            if not isinstance(step, dict) or step.get("stage") != stage or not isinstance(step.get("action"), str) or not step["action"].strip():
                raise ValueError("fresh_plan_stage_action_required")
        self._store_stage(state, "S1", args)
        state["next_stage"] = "S2"
        return {"stage": "S1", "gate_passed": True, "attempt_consumed": True, "next_stage": "S2"}, {"fresh_semantics": True}

    def _retrieve_frozen_evidence(self, state, args):
        if state["next_stage"] not in {"S2", "S3", "S4", "S5"}:
            raise ValueError("S1_plan_required_before_evidence")
        item = next((e for e in self.descriptor["evidence"] if e["evidence_id"] == args["evidence_id"]), None)
        if item is None:
            raise ValueError("undeclared_evidence_id")
        path = contained_file(self.bundle.root, item["path"])
        raw = path.read_bytes()
        if len(raw) != item["bytes"] or byte_digest(raw) != item["blake3"]:
            raise ValueError("frozen_evidence_changed")
        if item["evidence_id"] not in state["retrieved"]:
            state["retrieved"].append(item["evidence_id"])
        return {"stage": "S2", "gate_passed": True, "evidence_id": item["evidence_id"],
                "content_blake3": item["blake3"], "upstream_sha256": item["upstream_sha256"],
                "content": strict_json(raw), "next_stage": state["next_stage"]}, {"bytes": len(raw)}

    def _materialize_evidence_selection(self, state, args):
        if state["next_stage"] != "S2":
            raise ValueError("S2_prerequisite_boundary")
        expected = {e["evidence_id"] for e in self.descriptor["evidence"]}
        chosen = args.get("evidence_ids")
        if not isinstance(chosen, list) or not all(isinstance(v, str) for v in chosen) or len(chosen) != len(set(chosen)) or set(chosen) != expected or set(state["retrieved"]) != expected:
            raise ValueError("retrieve_and_select_exact_declared_evidence")
        if not isinstance(args.get("rationale"), str) or not args["rationale"].strip():
            raise ValueError("fresh_selection_rationale_required")
        self._reopen(state, "S1")
        self._store_stage(state, "S2", args)
        state["next_stage"] = "S3"
        return {"stage": "S2", "gate_passed": True, "attempt_consumed": True, "next_stage": "S3"}, {"selected_evidence": sorted(expected)}

    def _execute_code(self, state, args):
        stage = args["stage"]
        if state["next_stage"] != stage:
            raise ValueError("execution_stage_prerequisite_missing")
        self._verify_public(state)
        for previous in ("S1", "S2") + (("S3",) if stage == "S4" else ()):
            self._reopen(state, previous)
        stage_root = self.root / "executions" / (stage + "-" + str(uuid4()))
        inputs = stage_root / "input"
        inputs.mkdir(parents=True)
        shutil.copy2(self.root / "public/task.json", inputs / "public-task.json")
        shutil.copy2(self.root / "public/semantics.json", inputs / "fresh-semantics.json")
        for previous in ("S1", "S2") + (("S3",) if stage == "S4" else ()):
            write_json(inputs / (previous + "-handoff.json"), self._reopen(state, previous))
        for item in self.descriptor["evidence"]:
            relative_path(item["evidence_id"])
            source = contained_file(self.bundle.root, item["path"])
            if file_digest(source) != item["blake3"]:
                raise ValueError("evidence_changed_before_execution")
            target = inputs / "evidence" / (item["evidence_id"] + ".json")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        execution = execute(bwrap=self.bwrap, runtime_root=self.runtime_root, stage_root=stage_root,
                            code=args["code"], limits=self.construction["runtime"]["execution_limits"])
        spec = self.construction["runtime"]["s3_artifact" if stage == "S3" else "s4_artifact"]
        if not execution["successful_exit"]:
            return {"stage": stage, "gate_passed": False, "attempt_consumed": True, "error": "bounded_execution_failed",
                    "next_stage": stage, "stdout": execution["stdout"], "stderr": execution["stderr"]}, {"execution": execution}
        path = contained_file(stage_root, spec["relative_path"])
        raw = path.read_bytes()
        if not int(spec.get("min_bytes", 0)) <= len(raw) <= int(spec["max_bytes"]):
            raise ValueError("source_artifact_size_boundary")
        document = strict_json(raw)
        validation = evaluate_artifact(document, spec)
        if validation["gate_passed"]:
            self._store_stage(state, stage, document, raw=raw)
            state["next_stage"] = "S4" if stage == "S3" else "S5"
        public_result = {"stage": stage, "gate_passed": validation["gate_passed"], "attempt_consumed": True,
                "next_stage": state["next_stage"], "artifact_blake3": byte_digest(raw),
                "stdout": execution["stdout"], "stderr": execution["stderr"],
                "error": None if validation["gate_passed"] else "source_artifact_contract_failed"}
        if not validation["schema_valid"]:
            public_result["public_schema_validation"] = public_schema_diagnostics(document, spec["json_schema"])
        return public_result, {"execution": execution, "private_validation": validation}

    def _submit_results(self, state, args):
        if state["next_stage"] != "S5" or state["submission_attempted"]:
            raise ValueError("S5_once_or_prerequisite_boundary")
        state["submission_attempted"] = True
        reopened = {}
        for previous, key in (("S3", "s3_artifact"), ("S4", "s4_artifact")):
            artifact = self._reopen(state, previous)
            reopened[previous] = evaluate_artifact(artifact, self.construction["runtime"][key])
        terminal = args["terminal"]
        validation = evaluate_artifact(terminal, self.construction["runtime"]["terminal"])
        if len(canonical(terminal)) > self.construction["runtime"]["terminal"]["max_bytes"]:
            raise ValueError("terminal_size_boundary")
        passed = validation["gate_passed"] and all(r["gate_passed"] for r in reopened.values())
        if passed:
            self._store_stage(state, "S5", terminal)
            state["next_stage"] = "complete"
        return {"stage": "S5", "gate_passed": passed, "attempt_consumed": True, "next_stage": state["next_stage"],
                "completed_stages": sorted(state["completed"]), "fresh_execution_only": True,
                "clinical_or_ability_score": None}, {"private_terminal_validation": validation, "independent_reopen": reopened}

    def native_registry(self):
        """Optional transport integration; schema values are unchanged."""
        from eva_agent.pipeline.tools import ToolDefinition, ToolRegistry
        definitions = []
        for name, row in self.definitions.items():
            definitions.append(ToolDefinition(name=name, description=row["description"], input_schema=row["parameters"],
                handler=lambda _workspace, args, name=name: self.call(name, dict(args)),
                parallel_safe=name == "retrieve_frozen_evidence", read_only=name == "retrieve_frozen_evidence"))
        return ToolRegistry(definitions)
