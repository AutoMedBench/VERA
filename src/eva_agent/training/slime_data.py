"""EVA's observable assistant decisions as Slime full-parameter SFT samples.

Tokenization uses the deployed Qwen tokenizer and its complete tool template.
Only the released final assistant decision receives loss; context, old assistant
messages, and tool results are conditioning tokens. No reasoning is synthesized.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping


def render_sft_row(row: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
    messages = deepcopy(row["messages"])
    tools = deepcopy(row.get("tools", []))
    contract = row["loss_contract"]
    if (
        contract.get("supervise_exactly_one_assistant_message") is not True
        or contract.get("assistant_message_index") != len(messages) - 1
        or messages[-1].get("role") != "assistant"
    ):
        raise ValueError("EVA SFT requires exactly one final assistant target")
    for index, message in enumerate(messages):
        if message.get("content") is None:
            message["content"] = ""
        message["step_loss_mask"] = int(index == len(messages) - 1)
        for call in message.get("tool_calls", []):
            function = call["function"]
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"])

    kwargs = dict(tokenize=False, tools=tools, return_dict=False)
    rendered = tokenizer.apply_chat_template(messages, **kwargs)
    history = tokenizer.apply_chat_template(messages[:-1], **kwargs)
    header = "<|im_start|>assistant\n"
    if not rendered.startswith(history + header):
        raise ValueError("Qwen final assistant template does not extend exact history")
    body_start = len(history) + len(header)
    target_end = rendered.find("<|im_end|>", body_start)
    if target_end < 0 or rendered[target_end + len("<|im_end|>"):].strip():
        raise ValueError("Qwen target has unexpected message boundary")
    # Empty template reasoning wrappers are context. Explicit visible summaries
    # remain ordinary released assistant content and can be learned as supplied.
    empty_thinking = "<think>\n\n</think>\n\n"
    if rendered.startswith(empty_thinking, body_start):
        body_start += len(empty_thinking)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    tokens = list(encoded["input_ids"])
    mask = [int(end > start and end > body_start) for start, end in encoded["offset_mapping"]]
    if not any(mask):
        raise ValueError("Qwen target contains no loss-bearing token")
    first = mask.index(1)
    # A token crossing the boundary would train context bytes; reject rather
    # than silently changing the published single-assistant loss contract.
    if encoded["offset_mapping"][first][0] < body_start:
        raise ValueError("token crosses the assistant loss boundary")
    return {
        "prompt": history + header,
        "label": rendered[body_start:],
        "metadata": {
            "schema": "eva.slime-observable-sft-sample.v1",
            "row_id": row["row_id"],
            "stage": row.get("metadata", {}).get("stage"),
            "quality_tier": row.get("metadata", {}).get("quality_tier"),
            "tokens": tokens,
            "loss_mask": mask,
            "response_length": len(tokens) - first,
            "tools": tools,
            "source_loss_contract": dict(contract),
            "history_loss_masked": True,
            "tool_observations_loss_masked": True,
        },
    }


def generate_sft_rollout(args: Any, rollout_id: int, data_buffer: Any, evaluation: bool = False):
    """Slime's synchronous rollout hook for verified pre-tokenized SFT."""
    if evaluation:
        raise ValueError("SFT hook does not generate evaluation completions")
    grouped = data_buffer.get_samples(args.rollout_batch_size)
    result = []
    for group in grouped:
        if len(group) != 1:
            raise ValueError("SFT requires one sample per prompt")
        sample = group[0]
        meta = sample.metadata
        tokens, mask = meta["tokens"], meta["loss_mask"]
        if len(tokens) != len(mask) or not any(mask) or any(x not in (0, 1) for x in mask):
            raise ValueError("EVA token/loss-mask alignment differs")
        response_length = len(tokens) - mask.index(1)
        if response_length != meta["response_length"]:
            raise ValueError("EVA response boundary differs")
        sample.tokens = list(tokens)
        sample.response_length = response_length
        sample.loss_mask = list(mask[-response_length:])
        sample.reward = 0.0
        # Slime uses this ID to reduce all compact segments of ONE trajectory.
        # Independent SFT samples must not share the outer training iteration.
        sample.rollout_id = getattr(sample, "index", None)
        sample.metadata["training_iteration"] = rollout_id
        result.append(sample)
    return result


def score_workspace_rubric(rubric_table: Mapping[str, Any], verdict: Mapping[str, Any]) -> float:
    """Use exactly the benchmark rubric score after workspace judge execution.

    The host calls this with a trusted judge verdict, never policy-authored
    scores. Presence of evidence is required for every positive atomic score.
    """
    from eva_agent.rubrics.models import CompiledRubric

    if verdict.get("workspace_inspected") is not True:
        raise ValueError("reward requires a workspace-inspecting judge")
    if verdict.get("rubric_digest") != rubric_table["rubric_digest"]:
        raise ValueError("judge used a different rubric")
    scores = verdict["item_scores_bps"]
    evidence = verdict["evidence_by_item"]
    if any(score > 0 and not evidence.get(item_id) for item_id, score in scores.items()):
        raise ValueError("positive rubric score has no observed workspace/context evidence")
    return CompiledRubric.from_document(rubric_table).score(scores).reward_bps / 10_000
