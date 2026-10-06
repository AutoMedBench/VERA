# EVRA — Codex-based medical research harness

This repository is an owner-authorized, privacy-filtered source export for
agentic medical research. It is research software, not a clinical device.

## Branches

| Branch | Purpose |
| --- | --- |
| `vera-coevolve` | Model/harness co-evolution, evaluation feedback, skill selection and resumable controllers |
| `vera-harness` | Codex-based runtime, MCP tools, skill catalogs and versioned harness profiles |

The default branch is `vera-coevolve`. These are the repository's only two branches.

Internal `eva_agent` / `evamed` module and schema names are retained for
compatibility. VERA / EVRA are the release-facing names; this is not a renamed
Codex binary or a claim of new benchmark results.

See [privacy boundaries](PRIVACY.md) and [third-party notices](THIRD_PARTY_NOTICES.md).
No credentials, private datasets, benchmark answers, live-job configuration,
model weights or training logs are shipped.

## EVRA harness and skills

`src/eva_agent/codex_runtime` provides persistent threads, context management,
policy boundaries and restart/recovery. `plugins/evamed-codex` exposes the
schema-preserving MCP adapter, five plugin skills, nine project skills, and
24 distinct legacy skill capabilities with stage-specific access.

The versioned profiles v1.1, v1.2 and v1.3/Supra are documented under `docs/`.
The retained v1.3 profile targets the pinned upstream Codex 0.153.4 executable;
the Python SDK dependency retains its separate pinned version in `pyproject.toml`.
Codex is an external Apache-2.0 dependency, not bundled or rebranded here.

Python 3.11+ is required. Install with `python -m pip install -e '.[dev]'` in
your own workspace environment. The provider-free skill check is:

```sh
python plugins/evamed-codex/scripts/verify_skill_readiness.py --output /absolute/workspace/skill-check
```

The output must be a new directory outside the checkout, within the operator's
allowed workspace. This verifies loading/stage boundaries, not model quality.
Select authorized external sandbox assets and provider credentials separately.
Source snapshots under `integrations/` are libraries, not a live-cluster launcher.
