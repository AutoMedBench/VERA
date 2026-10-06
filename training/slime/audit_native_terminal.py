"""Inspect retained terminal parser behavior without model/Judge/tool execution."""
import argparse
import ast
import json
import math
import os
from pathlib import Path
from typing import List, Optional

from blake3 import blake3


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--sample-root', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    repo = Path(__file__).resolve().parents[2]
    source = repo.parent / '.venv/lib/python3.12/site-packages/sglang/srt/parser/reasoning_parser.py'
    syntax = ast.parse(source.read_text())
    classes = []
    for node in syntax.body:
        if isinstance(node, ast.ClassDef) and node.name in {
                'StreamingParseResult', 'BaseReasoningFormatDetector', 'Qwen3Detector'}:
            node.body = [item for item in node.body if isinstance(item, ast.FunctionDef)
                         and item.name in {'__init__', 'detect_and_parse'}]
            classes.append(node)
    pure = ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[]))
    namespace = {'Optional': Optional, 'List': List}
    exec(compile(pure, str(source), 'exec'), namespace)
    detector = namespace['Qwen3Detector'](force_reasoning=True, stream_reasoning=False)
    segments = sorted((json.loads(path.read_bytes()) for path in
        (args.sample_root / 'private-provider-tokens').glob('segment-*.json')),
        key=lambda value: value['segment_index'])
    last = segments[-1]
    parsed = detector.detect_and_parse(last['output_text'])
    failure_path = args.sample_root / 'failure.json'
    failure = json.loads(failure_path.read_bytes())
    receipt = failure['codex_turn_receipt']
    calls = []
    for call in receipt['tool_calls']:
        result = call['output']['result']['structuredContent']['tool_result']
        output = result.get('output') or {}
        error = output.get('error') or {}
        calls.append({'name': call['name'], 'mcp_status': call['status'],
            'tool_status': result['status'], 'tool_error_code': result.get('error_code'),
            'gate_passed': output.get('gate_passed'), 'stage': output.get('stage'),
            'policy_error_category': error.get('category') if isinstance(error, dict) else None,
            'workspace_before_blake3': result['workspace_before_blake3'],
            'workspace_after_blake3': result['workspace_after_blake3']})
    alignment = all(len(value['output_token_ids']) == len(value['output_token_logprobs'])
        == len(value['loss_mask']) and all(item == 1 for item in value['loss_mask'])
        and all(math.isfinite(item) and item <= 0 for item in value['output_token_logprobs'])
        for value in segments)
    document = {'schema': 'eva.native-codex-terminal-audit.v1',
        'provider_calls': 0, 'judge_calls': 0, 'tool_reexecutions': 0, 'gpu_model_loads': 0,
        'source_failure_blake3': blake3(failure_path.read_bytes()).hexdigest(),
        'installed_parser_source_blake3': blake3(source.read_bytes()).hexdigest(),
        'codex_status': receipt['status'], 'codex_visible_final_length': len(receipt['final_response']),
        'segments': len(segments), 'sampled_tokens': sum(len(value['output_token_ids']) for value in segments),
        'all_output_ids_logprobs_masks_aligned': alignment,
        'last_segment_index': last['segment_index'], 'last_finish_reason': last['finish_reason'],
        'last_raw_text_blake3': blake3(last['output_text'].encode()).hexdigest(),
        'last_has_thinking_close': '</think>' in last['output_text'],
        'last_has_tool_marker': '<tool_call>' in last['output_text'],
        'installed_parser_visible_characters': len(parsed.normal_text),
        'installed_parser_reasoning_characters': len(parsed.reasoning_text),
        'private_reasoning_exported': False, 'tools': calls,
        'original_attempt_mutated': False, 'reward_emitted': False}
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(document, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(document), flush=True)


if __name__ == '__main__':
    main()
