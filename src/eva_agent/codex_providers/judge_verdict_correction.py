"""Opt-in, one-shot Judge format feedback; never a host-authored verdict.

The original assistant message is appended to the same upstream history. Only
the model's second response can be emitted. Private signed operational records
are separate from the immutable, raw-provider-free adapter receipt schema.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import json
import math
from pathlib import Path
from threading import Lock
import time
from typing import Any, Callable, Mapping
from uuid import uuid4

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from jsonschema import Draft202012Validator

from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, canonical_json_bytes


EXTRA_ROW_FIELDS_POLICY = SELECTOR = "extra-row-fields-once-v1"
SIGNATURE_DOMAIN = b"eva.judge-verdict-correction.v1\x00"
SIGNED_SCHEMA = "eva.judge-verdict-correction-signed.v1"
EXTRA_KEYS = frozenset({"rationale_note", "score_note", "tool_calls", "evidence_refs_extra", "{}"})
ROW_KEYS = frozenset({"item_id", "score", "evidence_refs", "rationale"})
TOP_KEYS = frozenset({"item_scores", "hard_gates_passed", "summary"})


class JudgeVerdictCorrectionError(ValueError):
    pass


def _json(text: str) -> Any:
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value
    def constant(value):
        raise ValueError("nonfinite JSON")
    def floating(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("nonfinite JSON")
        return result
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=floating)


def correction_candidate(text: str, schema: Mapping[str, Any]) -> tuple[dict, list[dict]] | None:
    """Return a validation-only projection, never an emitted model response."""
    try:
        value = _json(text)
        if not isinstance(value, dict) or set(value) != TOP_KEYS:
            return None
        rows = value["item_scores"]
        if not isinstance(rows, list) or not rows:
            return None
        projected, errors = deepcopy(value), []
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not ROW_KEYS <= row.keys():
                return None
            extras = row.keys() - ROW_KEYS
            if not extras <= EXTRA_KEYS or ("tool_calls" in extras and row["tool_calls"] != []):
                return None
            for key in sorted(extras):
                errors.append({"path": ["item_scores", index, key], "keyword": "additionalProperties"})
                del projected["item_scores"][index][key]
        if not errors or not Draft202012Validator(schema).is_valid(projected):
            return None
        # The supplied schema must reject exactly the actual extras, too.
        actual_errors = list(Draft202012Validator(schema).iter_errors(value))
        if not actual_errors or any(row.validator != "additionalProperties" for row in actual_errors):
            return None
        return projected, errors
    except (ValueError, TypeError, KeyError):
        return None


def _terminal(upstream: Mapping[str, Any], model: str) -> tuple[str, dict] | None:
    choices = upstream.get("choices")
    if upstream.get("model") != model or not isinstance(choices, list) or len(choices) != 1:
        return None
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if (not isinstance(message, dict) or choice.get("finish_reason") != "stop"
            or message.get("role") != "assistant" or message.get("refusal") not in (None, "")
            or message.get("tool_calls") not in (None, [])
            or message.get("function_call") is not None
            or not isinstance(message.get("content"), str) or not message["content"]):
        return None
    return message["content"], message


def _usage(upstream: Mapping[str, Any]) -> dict | None:
    # Operational accounting only; never rewrite the native response's usage.
    from .adapter import _response_usage
    value = upstream.get("usage")
    return _response_usage(value) if isinstance(value, Mapping) else None


def verify_judge_verdict_correction(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """Reopen the exact signed texts and deterministic non-semantic contract.

    The consumer must additionally reopen source_binding and the Judge's actual
    native receipt/read coverage. A self-contained signature is not that proof.
    """
    try:
        if (document["schema"] != SIGNED_SCHEMA or document["signature_domain"] != SIGNATURE_DOMAIN[:-1].decode()
                or document["algorithm"] != "Ed25519"):
            raise ValueError("signature contract")
        payload = document["payload"]
        public = base64.b64decode(document["public_key_base64"], validate=True)
        signature = base64.b64decode(document["signature_base64"], validate=True)
        if (blake3_bytes(public) != document["public_key_blake3"]
                or blake3_hex(payload) != document["payload_blake3"]
                or blake3_hex({k: v for k, v in document.items() if k != "envelope_blake3"}) != document["envelope_blake3"]):
            raise ValueError("commitment")
        Ed25519PublicKey.from_public_bytes(public).verify(signature, SIGNATURE_DOMAIN + canonical_json_bytes(payload))
        if (payload["schema"] != "eva.judge-verdict-correction.v1" or payload["selector"] != SELECTOR
                or payload["event"] not in {"prepared", "completed", "failed"}
                or payload["schema_blake3"] != blake3_hex(payload["output_schema"])
                or not payload["source_binding"] or not payload["eligibility_proof"]
                or payload["maximum_corrective_calls"] != 1 or payload["native_turn_restarted"] is not False
                or payload["tools_reexecuted"] is not False):
            raise ValueError("policy")
        original = payload["original"]
        if blake3_bytes(original["final_text"].encode()) != original["final_text_blake3"]:
            raise ValueError("original text")
        candidate = correction_candidate(original["final_text"], payload["output_schema"])
        if candidate is None or candidate[1] != payload["schema_errors"]:
            raise ValueError("eligible schema")
        if payload["required_fields_blake3"] != blake3_hex(candidate[0]):
            raise ValueError("original required fields")
        corrected = payload["corrected"]
        if corrected is not None and corrected.get("final_text") is not None:
            if blake3_bytes(corrected["final_text"].encode()) != corrected["final_text_blake3"]:
                raise ValueError("corrected text")
        budget = payload["budget"]
        if not 0 < budget["total_seconds"] <= 600 or budget["elapsed_seconds"] < 0:
            raise ValueError("budget")
        if payload["accepted"]:
            if (payload["event"] != "completed" or corrected is None
                    or corrected["http_status"] != 200 or corrected["terminal_tool_free"] is not True
                    or budget["elapsed_seconds"] > budget["total_seconds"]):
                raise ValueError("accepted outcome")
            final = _json(corrected["final_text"])
            if (not Draft202012Validator(payload["output_schema"]).is_valid(final)
                    or canonical_json_bytes(final) != canonical_json_bytes(candidate[0])):
                raise ValueError("corrected required fields")
        return payload
    except Exception as exc:
        raise JudgeVerdictCorrectionError("Judge correction proof differs") from exc


class JudgeVerdictCorrectionPolicy:
    """One instance per Judge; bind once just before the original native turn."""

    def __init__(self, *, judgment_id: str, source_binding: Mapping[str, Any], sink: Callable[[dict], None]):
        if not judgment_id or not source_binding or not callable(sink):
            raise JudgeVerdictCorrectionError("Judge correction binding differs")
        self.judgment_id = judgment_id
        self.source_binding = deepcopy(dict(source_binding))
        self._sink, self._lock = sink, Lock()
        self._eligibility = None
        self._deadline = self._started = None
        self._claimed = False
        self.outcome = "not_needed"
        self._receipts: list[str] = []

    def bind(self, *, eligibility: Callable[[str], Mapping[str, Any] | None], deadline_monotonic: float) -> None:
        now = time.monotonic()
        if (not callable(eligibility) or type(deadline_monotonic) not in {int, float}
                or not math.isfinite(deadline_monotonic) or not 0 < deadline_monotonic - now <= 600):
            raise JudgeVerdictCorrectionError("Judge correction deadline differs")
        with self._lock:
            if self._eligibility is not None:
                raise JudgeVerdictCorrectionError("Judge correction already bound")
            self._eligibility, self._deadline, self._started = eligibility, deadline_monotonic, now

    def remaining_seconds(self) -> float:
        if self._deadline is None:
            raise JudgeVerdictCorrectionError("Judge correction is not bound")
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise JudgeVerdictCorrectionError("Judge correction deadline exhausted")
        return remaining

    @property
    def receipt_blake3s(self) -> tuple[str, ...]:
        return tuple(self._receipts)

    def summary(self) -> dict:
        return {"selector": SELECTOR, "judgment_id": self.judgment_id,
                "outcome": self.outcome, "receipt_blake3s": list(self.receipt_blake3s)}

    def maybe_correct(self, *, upstream: Any, upstream_object: Mapping[str, Any], chat_body: Mapping[str, Any],
                      binding: Any, request_id: str, signer: Any, call_upstream: Callable[[Mapping[str, Any]], Any]) -> Any:
        self.remaining_seconds()  # Late original responses cannot restart the deadline, even if valid.
        terminal = _terminal(upstream_object, binding.model_id)
        fmt = chat_body.get("response_format", {})
        schema = fmt.get("json_schema", {}).get("schema") if fmt.get("type") == "json_schema" else None
        if terminal is None or not isinstance(schema, Mapping):
            return upstream
        candidate = correction_candidate(terminal[0], schema)
        if candidate is None:
            return upstream
        with self._lock:
            if self._claimed:
                return upstream
            if self._eligibility is None:
                raise JudgeVerdictCorrectionError("Judge correction is not bound")
            try:
                proof = self._eligibility(terminal[0])
            except Exception:
                proof = None  # A checker failure never authorizes correction.
            if not isinstance(proof, Mapping) or not proof:
                self.outcome = "ineligible"
                return upstream
            self.remaining_seconds()
            self._claimed = True
        corrected_body = deepcopy(dict(chat_body))
        diagnostic = {"instruction": "Correct only the JSON schema errors below. Remove the unexpected row properties. "
                      "Keep every required field value exactly unchanged. Do not rescore, add evidence, or call tools. "
                      "Return only the corrected JSON object.", "schema_errors": candidate[1]}
        corrected_body["messages"] = [*corrected_body["messages"], deepcopy(terminal[1]),
                                      {"role": "user", "content": canonical_json_bytes(diagnostic).decode()}]
        if corrected_body.get("tools"):
            corrected_body["tool_choice"] = "none"
        now = time.monotonic()
        payload = {"schema": "eva.judge-verdict-correction.v1", "selector": SELECTOR,
            "correction_id": str(uuid4()), "judgment_id": self.judgment_id, "request_id": request_id,
            "source_binding": deepcopy(self.source_binding), "route_id": binding.route_id, "model_id": binding.model_id,
            "provider_family": binding.provider_family, "implementation": {
                "policy_blake3": blake3_bytes(Path(__file__).read_bytes()),
                "adapter_blake3": blake3_bytes(Path(__file__).with_name("adapter.py").read_bytes())},
            "output_schema": deepcopy(dict(schema)), "schema_blake3": blake3_hex(schema),
            "schema_errors": candidate[1], "required_fields_blake3": blake3_hex(candidate[0]),
            "eligibility_proof": deepcopy(dict(proof)), "maximum_corrective_calls": 1,
            "native_turn_restarted": False, "tools_reexecuted": False,
            "original": {"final_text": terminal[0], "final_text_blake3": blake3_bytes(terminal[0].encode()),
                "upstream_body_blake3": blake3_bytes(upstream.body), "http_status": upstream.status,
                "latency_ms": upstream.latency_ms, "request_blake3": blake3_hex(chat_body),
                "usage": _usage(upstream_object)},
            "correction_request_blake3": blake3_hex(corrected_body),
            "history_append": {"assistant": {"role": "assistant", "content": terminal[0]}, "feedback": diagnostic},
            "corrected": None, "event": "prepared", "accepted": False,
            "budget": {"total_seconds": self._deadline - self._started, "elapsed_seconds": now - self._started},
            "previous_receipt_blake3": None, "failure_type": None}

        def persist():
            payload["budget"]["elapsed_seconds"] = time.monotonic() - self._started
            document = signer.sign_judge_verdict_correction(payload)
            self._sink(document)
            self._receipts.append(document["envelope_blake3"])
            return document["envelope_blake3"]

        self.outcome = "failed"
        payload["previous_receipt_blake3"] = persist()
        try:
            correction = call_upstream(corrected_body)
            payload["corrected"] = {"final_text": None, "final_text_blake3": None,
                "upstream_body_blake3": blake3_bytes(correction.body), "http_status": correction.status,
                "latency_ms": correction.latency_ms, "terminal_tool_free": False, "usage": None}
            parsed = _json(correction.body.decode()) if correction.status == 200 else {}
            last = _terminal(parsed, binding.model_id) if isinstance(parsed, dict) else None
            # Preserve any textual assistant output even when it is refused or
            # proposes a tool. Such an output can never be accepted or executed.
            choices = parsed.get("choices", []) if isinstance(parsed, dict) else []
            message = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
            text = message.get("content") if isinstance(message, dict) else None
            payload["corrected"] = {"final_text": text if isinstance(text, str) else None,
                "final_text_blake3": blake3_bytes(text.encode()) if isinstance(text, str) else None,
                "upstream_body_blake3": blake3_bytes(correction.body), "http_status": correction.status,
                "latency_ms": correction.latency_ms, "terminal_tool_free": last is not None,
                "usage": _usage(parsed) if isinstance(parsed, dict) else None}
            self.remaining_seconds()
            if (last is None or not Draft202012Validator(schema).is_valid(_json(last[0]))
                    or canonical_json_bytes(_json(last[0])) != canonical_json_bytes(candidate[0])):
                raise JudgeVerdictCorrectionError("Judge correction changed values or remains invalid")
            payload.update(event="completed", accepted=True)
            persist()
            self.outcome = "corrected"
            return correction
        except Exception as exc:
            payload.update(event="failed", accepted=False, failure_type=type(exc).__name__)
            persist()
            raise JudgeVerdictCorrectionError("Judge format correction failed") from exc
