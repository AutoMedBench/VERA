"""CPU-only reproduction of retained Qwen XML parameter type projection.

Decodes the exact retained prompt through tokenizers (no Torch/model import),
then executes only the installed parser's two pure argument-conversion methods
via AST extraction. It never prints private model reasoning or alters receipts.
"""
import argparse
import ast
import json
import logging
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Optional

from blake3 import blake3
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--host-typed', action='store_true', help='Offline host-schema decoding and validation only')
    cli = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    detector_path = repo.parent / '.venv/lib/python3.12/site-packages/sglang/srt/function_call/qwen3_coder_detector.py'
    syntax = ast.parse(detector_path.read_text())
    klass = next(node for node in syntax.body if isinstance(node, ast.ClassDef) and node.name == 'Qwen3CoderDetector')
    methods = [node for node in klass.body if isinstance(node, ast.FunctionDef)
               and node.name in {'_get_arguments_config', '_convert_param_value'}]
    pure = ast.ClassDef(name='PureDetector', bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[pure], type_ignores=[]))
    namespace = dict(ast=ast, json=json, Any=Any, Optional=Optional, Tool=Any, logger=logging.getLogger('parser-audit'))
    exec(compile(module, str(detector_path), 'exec'), namespace)
    detector = namespace['PureDetector']()
    projector = None
    if cli.host_typed:
        from eva_agent.sources.legacy_execution import _load_legacy_modules
        from eva_agent.training.qwen_tool_types import HostArgumentProjector
        from jsonschema import Draft202012Validator
        _load_legacy_modules(repo.parent / 'rlevo-med-research/src')
        policy_path = next((cli.sample_root / 'workspace').glob('*/.eva/source-policy.json'))
        runtime = json.loads((policy_path.parent / 'runtime-context.json').read_bytes())
        context = SimpleNamespace(episode=SimpleNamespace(
            initial_files={'.eva/source-policy.json': policy_path.read_bytes()},
            policy_context={'execution_binding': runtime}))
        projector = HostArgumentProjector.from_context(context)
    tokenizer = Tokenizer.from_file(str(repo.parent / 'Qwen3.5-9B/tokenizer.json'))
    segments = []
    for path in (cli.sample_root / 'private-provider-tokens').glob('segment-*.json'):
        segments.append((json.loads(path.read_text()), path))
    segments.sort(key=lambda pair: pair[0]['segment_index'])
    report = {'schema': 'eva.native-qwen-xml-parser-audit.v1', 'provider_calls': 0, 'gpu_model_loads': 0,
              'parser_source_blake3': blake3(detector_path.read_bytes()).hexdigest(), 'segments': []}
    if projector is not None:
        report['host_type_binding'] = projector.binding
        report['historical_calls_reexecuted'] = False
    for segment, path in segments:
        prompt = tokenizer.decode(segment['prompt_token_ids'], skip_special_tokens=False)
        tool_blocks = re.findall(r'<tools>\s*(.*?)\s*</tools>', prompt, re.S)
        signatures = []
        for block in tool_blocks:
            for line in block.splitlines():
                try:
                    signature = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(signature, dict):
                    signature = signature.get('function', signature)
                    if isinstance(signature, dict) and 'name' in signature:
                        signatures.append(signature)
        tools = [SimpleNamespace(type='function', function=SimpleNamespace(**row)) for row in signatures]
        calls = []
        for name, content in re.findall(r'<function=([^>]+)>(.*?)</function>', segment['output_text'], re.S):
            config = detector._get_arguments_config(name, tools)
            parameters = []
            installed_values = {}
            for field, raw in re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', content, re.S):
                value = detector._convert_param_value(raw.strip(), field, config, name)
                installed_values[field] = value
                try:
                    literal = json.loads(raw.strip())
                    valid_json, literal_type = True, type(literal).__name__
                except json.JSONDecodeError:
                    valid_json, literal_type = False, None
                parameters.append({'name': field, 'offered_schema': config.get(field),
                    'raw_value_blake3': blake3(raw.encode()).hexdigest(),
                    'raw_valid_json': valid_json, 'raw_json_type': literal_type,
                    'installed_result_type': type(value).__name__})
            signature = next((row for row in signatures if row['name'] == name), None)
            calls.append({'name': name, 'tool_signature_found': signature is not None,
                          'offered_parameters_schema': signature.get('parameters') if signature else None,
                          'parameters': parameters})
            if projector is not None and signature is not None:
                canonical_name = projector._name(name)
                message = {'role': 'assistant', 'tool_calls': [{'id': 'offline-not-executed', 'type': 'function',
                    'function': {'name': name, 'arguments': json.dumps(installed_values)}}]}
                projected, audit = projector.project_message(message, [{'type': 'function', 'function': signature}])
                values = json.loads(projected['tool_calls'][0]['function']['arguments'])
                errors = sorted(Draft202012Validator(projector.host_schemas[canonical_name]).iter_errors(values),
                                key=lambda error: str(list(error.path))) if canonical_name else []
                calls[-1]['typed_projection'] = audit
                calls[-1]['host_json_schema_valid'] = not errors
                calls[-1]['host_json_schema_errors'] = [{'path': list(error.path), 'validator': error.validator}
                                                       for error in errors]
        report['segments'].append({'segment_index': segment['segment_index'],
            'segment_file_blake3': blake3(path.read_bytes()).hexdigest(), 'advertised_tools': len(tools), 'calls': calls})
    import os
    fd = os.open(cli.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
