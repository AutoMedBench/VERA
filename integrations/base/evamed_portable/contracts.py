"""Preserve source schemas; evaluate artifact checks without exposing hidden values."""
from copy import deepcopy
import math
from jsonschema import Draft202012Validator
from .integrity import canonical, digest

PRIMARY_TOOLS = ("materialize_plan", "retrieve_frozen_evidence", "materialize_evidence_selection", "execute_code", "submit_results")
SUPPORTED_CHECKS = {"eq", "numeric_eq", "true", "false", "nonempty", "gte", "lte"}


def tools_for_case(source_tools, construction):
    result = {name: deepcopy(source_tools[name]) for name in PRIMARY_TOOLS}
    # Only the instance-specific terminal schema comes from this exact source.
    result["submit_results"]["parameters"]["properties"]["terminal"] = deepcopy(construction["runtime"]["terminal"]["json_schema"])
    return result


def validate_arguments(definition, arguments):
    errors = list(Draft202012Validator(definition["parameters"]).iter_errors(arguments))
    if errors:
        raise ValueError("canonical_tool_arguments_invalid")


def pointer(value, path):
    if path == "":
        return value
    if not path.startswith("/"):
        raise ValueError("invalid_check_pointer")
    for part in path[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def is_number(value):
    return type(value) in {int, float} and math.isfinite(value)


def evaluate_artifact(document, specification):
    schema_errors = list(Draft202012Validator(specification["json_schema"]).iter_errors(document))
    failures = []
    for check in specification["checks"]:
        op = check["op"]
        if op not in SUPPORTED_CHECKS:
            raise ValueError("unsupported_source_check_operation")
        try:
            actual = pointer(document, check["pointer"])
            expected = check.get("value")
            if op == "eq":
                passed = canonical(actual) == canonical(expected)
            elif op == "numeric_eq":
                passed = is_number(actual) and is_number(expected) and actual == expected
            elif op == "gte":
                passed = is_number(actual) and is_number(expected) and actual >= expected
            elif op == "lte":
                passed = is_number(actual) and is_number(expected) and actual <= expected
            elif op == "true":
                passed = actual is True
            elif op == "false":
                passed = actual is False
            else:
                passed = bool(actual.strip()) if isinstance(actual, str) else isinstance(actual, (list, dict)) and bool(actual)
        except (KeyError, IndexError, TypeError, ValueError):
            passed = False
        if not passed:
            failures.append(check["check_id"])
    critical = set(specification["critical_check_ids"])
    result = {"schema_valid": not schema_errors, "all_checks_passed": not failures,
              "critical_checks_passed": not (set(failures) & critical), "failed_check_ids": failures,
              "schema_error_count": len(schema_errors), "checks_evaluated": len(specification["checks"]),
              "artifact_blake3": digest(document), "source_contract_blake3": digest(specification),
              "target_metric_check_ids": specification.get("target_metric_check_ids", []),
              "clinical_score_computed": False}
    result["gate_passed"] = result["schema_valid"] and result["all_checks_passed"]
    return result
