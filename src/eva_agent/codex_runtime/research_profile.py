"""Opt-in Codex research defaults; never alter the benchmark's locked profile."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from typing import Mapping

from .backend import CodexLaunchOptions

PROFILE_NAME = 'evamed-codex-v1.1-research'
PINNED_CODEX_VERSION = 'codex-cli 0.153.4'
DEFAULT_PROFILE = Path(__file__).resolve().parents[3] / 'config' / (PROFILE_NAME + '.toml')
FEATURE_LINE = re.compile(r'^(\S+)\s+(stable|experimental|under development|deprecated|removed)\s+(true|false)$')


class ResearchProfileError(ValueError):
    """Fixed diagnostics; never include inherited config, credentials or stderr."""


def load_profile(path: Path = DEFAULT_PROFILE) -> dict[str, bool]:
    value = tomllib.loads(Path(path).read_text())
    features = value.get('features')
    if (set(value) != {'features'} or not isinstance(features, dict) or not features
            or any(not re.fullmatch(r'[a-z][a-z0-9_]*', name) or type(enabled) is not bool
                   for name, enabled in features.items())):
        raise ResearchProfileError('profile_must_contain_only_boolean_features')
    return dict(features)


def profile_overrides(path: Path = DEFAULT_PROFILE) -> tuple[str, ...]:
    return tuple(f'features.{key}={str(value).lower()}' for key, value in load_profile(path).items())


def research_launch_options(base: CodexLaunchOptions | None = None, *,
                            profile_path: Path = DEFAULT_PROFILE) -> CodexLaunchOptions:
    """Add defaults before explicit caller overrides; preserve every other field.

    No process/provider call occurs here. The caller must use check_profile once
    for the selected binary. Existing benchmark policy overrides still win.
    """
    source = base or CodexLaunchOptions()
    return replace(source, config_overrides=(*profile_overrides(profile_path), *source.config_overrides))


def command(binary: str, arguments: tuple[str, ...], *,
            profile_path: Path = DEFAULT_PROFILE) -> list[str]:
    argv = [binary]
    for override in profile_overrides(profile_path):
        argv.extend(('--config', override))
    return [*argv, *arguments]


def inspect_profile(path: Path = DEFAULT_PROFILE) -> dict:
    return {'schema': 'eva.codex-research-profile-inspection.v1', 'profile': PROFILE_NAME,
            'profile_path': str(Path(path).resolve()), 'pinned_codex_version': PINNED_CODEX_VERSION,
            'features': load_profile(path), 'precedence': 'profile defaults before explicit caller overrides',
            'provider_model_approval_sandbox_environment_overridden': False,
            'canonical_tool_skill_schema_changes': False, 'provider_calls': 0,
            'speed_gain_measured': False, 'memory_behavior_verified': False,
            'benchmark_admission_claimed': False}


def resolve_binary(value: str | None) -> str:
    selected = value or shutil.which('codex')
    if not selected:
        raise ResearchProfileError('codex_binary_unavailable')
    path = Path(selected).absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ResearchProfileError('codex_binary_unavailable')
    return str(path)


def _private_check_environment(directory: Path) -> dict[str, str]:
    # Only the provider-free checker gets an empty private Codex home. Normal
    # launch/helper paths preserve the caller's full environment unchanged.
    safe = {key: os.environ[key] for key in ('PATH', 'LANG', 'LC_ALL', 'SYSTEMROOT') if key in os.environ}
    return {**safe, 'CODEX_HOME': str(directory), 'RUST_LOG': 'off'}


def _version(binary: str, *, env: Mapping[str, str], cwd: Path) -> str:
    try:
        value = subprocess.run([binary, '--version'], cwd=cwd, env=dict(env),
            capture_output=True, text=True, check=True, timeout=15).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise ResearchProfileError('codex_version_check_failed') from error
    if value != PINNED_CODEX_VERSION:
        raise ResearchProfileError('codex_version_not_pinned_0_153_4')
    return value


def _strict_initialize(binary: str, *, profile_path: Path, env: Mapping[str, str], cwd: Path) -> None:
    """Parse strict config and initialize only; never start/resume a thread."""
    argv = command(binary, ('app-server', '--strict-config', '--listen', 'stdio://'), profile_path=profile_path)
    request = {'id': 1, 'method': 'initialize', 'params': {
        'clientInfo': {'name': 'evamed_research_profile_check', 'version': '1.1.0'},
        'capabilities': {'experimentalApi': True}}}
    process = subprocess.Popen(argv, cwd=cwd, env=dict(env), stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        process.stdin.write(json.dumps(request).encode() + b'\n')
        process.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + 20
            pending = b''
            while time.monotonic() < deadline:
                if not selector.select(max(0, deadline - time.monotonic())):
                    break
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    break
                pending += chunk
                if len(pending) > 1024 * 1024:
                    raise ResearchProfileError('strict_config_response_too_large')
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    value = json.loads(line)
                    if value.get('id') == 1:
                        if 'error' in value or not isinstance(value.get('result'), dict):
                            raise ResearchProfileError('strict_config_initialize_rejected')
                        return
        raise ResearchProfileError('strict_config_initialize_unavailable')
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        process.stdout.close()


def check_profile(binary: str, *, profile_path: Path = DEFAULT_PROFILE) -> dict:
    features = load_profile(profile_path)
    with tempfile.TemporaryDirectory(prefix='evamed-profile-check-') as temporary:
        directory = Path(temporary)
        env = _private_check_environment(directory)
        version = _version(binary, env=env, cwd=directory)
        try:
            result = subprocess.run(command(binary, ('features', 'list'), profile_path=profile_path),
                cwd=directory, env=env, capture_output=True, text=True, check=True, timeout=15)
        except (OSError, subprocess.SubprocessError) as error:
            raise ResearchProfileError('codex_features_check_failed') from error
        observed = {}
        for line in result.stdout.splitlines():
            match = FEATURE_LINE.fullmatch(line.strip())
            if match:
                name, lifecycle, enabled = match.groups()
                observed[name] = (lifecycle, enabled == 'true')
        for name, enabled in features.items():
            row = observed.get(name)
            if row is None or row[0] in ('removed', 'deprecated') or row[1] != enabled:
                raise ResearchProfileError('profile_feature_unsupported_or_not_applied')
            if enabled and row[0] != 'stable':
                raise ResearchProfileError('profile_enables_unstable_feature')
        _strict_initialize(binary, profile_path=profile_path, env=env, cwd=directory)
    return {**inspect_profile(profile_path), 'schema': 'eva.codex-research-profile-check.v1',
            'valid': True, 'actual_codex_version': version, 'feature_count_checked': len(features),
            'strict_config_initialize_passed': True, 'thread_requests': 0, 'turn_requests': 0,
            'auth_material_copied': False, 'global_config_writes': False,
            'check_scope': 'isolated profile defaults, not effective caller configuration or model behavior'}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codex-bin')
    parser.add_argument('--profile-path', type=Path, default=DEFAULT_PROFILE)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--inspect', action='store_true')
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('codex_arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    rest = args.codex_arguments
    if rest[:1] == ['--']:
        rest = rest[1:]
    try:
        if args.inspect:
            result = inspect_profile(args.profile_path)
        else:
            binary = resolve_binary(args.codex_bin)
            result = check_profile(binary, profile_path=args.profile_path)
            if args.run:
                # No env/cwd arguments: preserve the caller's actual setup.
                return subprocess.call(command(binary, tuple(rest), profile_path=args.profile_path))
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        code = str(error) if isinstance(error, ResearchProfileError) else type(error).__name__
        print(json.dumps({'profile': PROFILE_NAME, 'valid': False, 'error': code}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
