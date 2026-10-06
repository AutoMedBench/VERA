"""Lossless projection of *completed* Codex argument-parser failures.

No tool is executed here. Original Responses items are untouched. A completed
parallel group is transcribed in its original item order, so changing one bad
call cannot leave an orphan formal Chat tool result from a good sibling.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Mapping


_PARSER_ERROR = re.compile(
    r"(?:Wall time: [0-9]+(?:\.[0-9]+)? seconds\nOutput:\n)?"
    r"err: (?:expected value|key must be a string|trailing characters|number out of range|"
    r"EOF while parsing (?:a value|an object|a list|a string)|"
    r"expected `:`|expected `,` or `}`|expected `,` or `]`|invalid escape)"
    r" at line [1-9][0-9]* column [1-9][0-9]*"
)


def _not_constant(value: str):
    raise ValueError("nonfinite JSON constant")


def _invalid_arguments(arguments: Any) -> bool:
    if not isinstance(arguments, str):
        return True
    try:
        value = json.loads(arguments, parse_constant=_not_constant)
    except (ValueError, TypeError):
        return True
    def finite(item):
        if isinstance(item, float):
            return math.isfinite(item)
        if isinstance(item, dict):
            return all(finite(child) for child in item.values())
        if isinstance(item, list):
            return all(finite(child) for child in item)
        return True
    return not isinstance(value, dict) or not finite(value)


def project_completed_invalid_history(
    items: Any, *, transcript: Callable[[Mapping[str, Any]], str],
    output_text: Callable[[Any], str],
) -> Any:
    """Only an exact existing parser-error result can close an invalid call.

    Valid histories return the original value. Pending, duplicate,
    orphan, interleaved-message or ambiguously failed invalid histories raise
    before an upstream request. Plain success/error prose is not parser proof.
    Current stage tools govern future execution, not historical observations.
    A closed parser-error pair remains text evidence after its tool is retired;
    this does not expose the retired tool or grant permission to execute it.
    """
    if not isinstance(items, list):
        return items
    invalid = {index for index, item in enumerate(items) if isinstance(item, dict)
        and item.get("type") == "function_call" and _invalid_arguments(item.get("arguments"))}
    if not invalid:
        return items
    calls: dict[str, tuple[int, Mapping[str, Any]]] = {}
    results: dict[str, tuple[int, Mapping[str, Any]]] = {}
    pending: set[str] = set()
    group_start: int | None = None
    groups: list[tuple[int, int]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError("completed invalid history item differs")
        kind = item.get("type", "message")
        if kind == "function_call":
            call_id, name, namespace = item.get("call_id"), item.get("name"), item.get("namespace")
            if (not isinstance(call_id, str) or not call_id or call_id in calls
                    or not isinstance(name, str) or not name
                    or namespace is not None and not isinstance(namespace, str)):
                raise ValueError("completed invalid history call identity differs")
            if not pending:
                group_start = index
            calls[call_id] = (index, item)
            pending.add(call_id)
        elif kind == "function_call_output":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or call_id not in pending or call_id in results:
                raise ValueError("completed invalid history result is duplicate or orphan")
            results[call_id] = (index, item)
            pending.remove(call_id)
            if not pending:
                assert group_start is not None
                groups.append((group_start, index))
                group_start = None
        elif pending and kind != "reasoning":
            raise ValueError("completed invalid history group is interrupted by a message")
    selected: set[int] = set()
    for index in invalid:
        item = items[index]
        call_id = item["call_id"]
        if call_id not in results or not isinstance(item.get("arguments"), str):
            raise ValueError("completed invalid history call is pending or has non-text arguments")
        result_index, result = results[call_id]
        text = output_text(result.get("output"))
        if result_index <= index or _PARSER_ERROR.fullmatch(text.strip()) is None:
            raise ValueError("completed invalid history lacks an exact Codex parser error")
        group = next(((start, end) for start, end in groups if start <= index <= end), None)
        if group is None:
            raise ValueError("completed invalid history parallel group is still pending")
        selected.update(range(group[0], group[1] + 1))
    projected = []
    for index, item in enumerate(items):
        kind = item.get("type", "message")
        if index in selected and kind in {"function_call", "function_call_output"}:
            projected.append({"type": "message", "role": "assistant" if kind == "function_call" else "user",
                "content": transcript(item)})
        else:
            projected.append(item)
    return projected
