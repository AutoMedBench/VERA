# EvaMed Codex v1.1 research profile

This is an opt-in, features-only profile for **Codex CLI 0.153.4**. It does not
fork or rebuild Codex, install a plugin, mutate global configuration, choose a
model/provider, or override approvals, filesystem/network permissions or process
environment. Explicit caller configuration comes **after** these defaults and
keeps precedence, including stricter benchmark feature restrictions.

The profile disables unrelated connector/UI, dictation, automation, remote-plugin
discovery, image-generation and personality features. Experimental code/context
modes stay off. It retains native shell/unified execution (including ordinary
code editing), image viewing, local plugins/skill discovery, multi-agent work,
memories and image-aware compaction. It uses no removed feature names; there is
no `apply_patch_freeform` toggle because that flag is removed in this build.

## Inspect and validate without model inference

Use the exact locally pinned binary, not an unverified `codex` on `PATH`:

```bash
python scripts/evamed_codex_research_v1_1.py --inspect
python scripts/evamed_codex_research_v1_1.py --codex-bin /absolute/path/to/codex-0.153.4 --check
```

`--inspect` reads only the profile. `--check` verifies the exact binary version,
its actual feature names/lifecycle/effective booleans, then initializes one
`app-server --strict-config` in an empty temporary Codex home. It sends no thread,
turn, model-generation or MCP tool request, copies no credentials, and closes its
own child. The private check home is discarded; global config/auth are untouched.
This checks isolated profile defaults, **not** a user's effective provider setup,
MCP readiness, medical accuracy, memory quality or speed.

## Explicit research launch

```bash
python scripts/evamed_codex_research_v1_1.py --codex-bin /absolute/path/to/codex-0.153.4 --run --
python scripts/evamed_codex_research_v1_1.py --codex-bin /absolute/path/to/codex-0.153.4 --run -- exec "Inspect this research repository"
```

The wrapper validates first, then passes remaining arguments directly to the
selected Codex binary with profile defaults. Normal launch inherits the caller's
environment and working directory. Caller `--config`, model/provider/security
and feature choices remain available and take precedence. It does not silently
enable sandbox bypass, network access or a model service.

Repository-based SDK callers can construct launch options without starting any
process:

```python
from eva_agent.codex_runtime import CodexLaunchOptions, OpenAICodexBackend
from eva_agent.codex_runtime.research_profile import research_launch_options

base = CodexLaunchOptions(codex_bin="/absolute/path/to/codex-0.153.4",
                         cwd="/absolute/research/workspace",
                         config_overrides=("features.shell_tool=false",))
launch = research_launch_options(base)
backend = OpenAICodexBackend(launch)  # Still not started; caller opens it explicitly.
# The caller's shell_tool=false remains effective.
```

The profile TOML is repository-local, not a new installed console entry. If the
helper is packaged independently, supply its retained `profile_path` explicitly.
The existing SDK dependency is unchanged; this helper does not claim that every
native research event is admitted by the existing benchmark receipt validator.

## Medical tools, skills and memory

Keep the existing `evamed` MCP server and byte-verified stage-permitted medical
skills. The profile does not replace their schemas, bodies, visibility policies
or parallel-safety metadata. Plugin loading remains enabled; the plugin still
requires its real registry/handlers and policy binding. Native `skill_search` is
separate from canonical MCP `search_skills`/`load_skill`. Native image viewing
remains available for authorized medical images; image *generation* is disabled.

Medical/literature search must remain available through the intended research
MCP tools or the caller's supported search configuration. This profile does not
override `web_search`; disabling GUI browser automation is not a substitute for
disabling medical research search. Preserve actual safe parallel MCP execution;
multi-agent enablement does not override tool-level serialization requirements.

Keep a persistent research workspace and Codex session state, `AGENTS.md`, and
concise project-local notes such as `.codex/task-memory.md`. Use actual thread
resume and compaction for continuation; the profile does not zero project-doc
context or force ephemeral sessions. `memories=true` enables a feature, not proof
that a memory behavior check passed. For a local model, retain the separately
verified context-window/compaction settings; never copy a context size from a
different model. No memory-quality or speed gain has been measured here.

## Research is not benchmark admission

AutoMedBench's public-only actor intentionally disables native shell/image/search
surfaces and exposes its fixed Docker/model tools and canonical skills through
MCP. **Do not replace that profile with this one.** Research native shell, image,
plugins and multi-agent events need a separately versioned evidence projection
before they can be credited as benchmark trajectories or training samples.
Existing policy/rubric/tool/skill/evidence schemas remain unchanged.

The implementation follows the official [configuration reference](https://developers.openai.com/codex/config-reference/)
and [app-server interface](https://developers.openai.com/codex/app-server/), but
the actual pinned binary's feature list and strict initializer determine build
compatibility; newer documentation is not assumed to describe 0.153.4 exactly.

Local validation on 2026-09-10: 14 focused tests passed, including caller-disabled
shell/image/multi-agent precedence. The real pinned 0.153.4 checker accepted all
29 configured features and strict initialization, with zero thread/turn requests.
No research task, model-quality, memory-behavior or speed experiment was run.
