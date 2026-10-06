"""Opt-in v1.2 task-memory composition over the unchanged Codex/MCP contracts."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from eva_agent.pipeline.digests import blake3_bytes
from .backend import CodexLaunchOptions
from .contracts import CodexSkill, CodexThreadOptions, CodexTurnInput
from .research_profile import research_launch_options

PROFILE_NAME = 'evamed-codex-v1.2-research'
SKILL_PATH = Path(__file__).resolve().parents[3] / 'plugins/evamed-codex/skills/summary-failures/SKILL.md'
MEMORY_INSTRUCTIONS = '''Task-memory protocol (public workspace only):
Keep the same task thread across stages when available. At each stage transition
save a concise notes/task-state.md: objective, decisions, artifact paths, pending
jobs, unresolved issues and next action. Read it on resume and after compaction;
verify referenced files and running jobs before relying on the summary.
Apply the summary_failures skill after meaningful errors; before a similar action,
read notes/failures.md and perform the recorded preventive check. Do not import
hidden references, judge material, or another model's answers into memory.
Use only offered tools with their actual argument schemas. Prefer targeted public
reads (around 2048 characters initially), query/search or Python summaries over
dumping logs and datasets. Expand only the relevant section. Persist full evidence
in workspace files and carry compact paths/results in conversation. Parallelize
independent reads/computation, but coordinate writes to the same note or artifact.
Task notes remain task-scoped; memory flags and compaction events are not proof
of successful recall. Record actual reads and later verified use in tool receipts.
'''


@dataclass(frozen=True)
class ResearchContextPolicy:
    """Use the real serving capacity; never invent a larger model context."""
    context_tokens: int = 32768
    output_tokens: int = 4096
    reserve_tokens: int = 2048
    compact_at_tokens: int = 12288
    compaction_headroom_mode: str = 'conservative_double'

    def __post_init__(self):
        values = (self.context_tokens, self.output_tokens, self.reserve_tokens, self.compact_at_tokens)
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError('context policy requires positive integer token budgets')
        if self.compaction_headroom_mode not in {'conservative_double', 'exact_request_guard'}:
            raise ValueError('unsupported compaction headroom mode')
        # Preserve historical defaults. The opt-in mode requires the caller's
        # exact post-projection text-token guard and the serving context guard;
        # multimodal token counts remain explicitly unverified by text tokenize.
        # This mode
        # does not infer that serialized history expands by an arbitrary 2x.
        factor = 2 if self.compaction_headroom_mode == 'conservative_double' else 1
        if factor * self.compact_at_tokens + self.output_tokens + self.reserve_tokens > self.context_tokens:
            raise ValueError('context policy leaves insufficient compaction headroom')

    def config_defaults(self):
        return {'model_auto_compact_token_limit': self.compact_at_tokens}


def summary_failures_skill(path: Path = SKILL_PATH) -> CodexSkill:
    target = Path(path).resolve(strict=True)
    body = target.read_bytes()
    if len(body) > 16384:
        raise ValueError('summary_failures skill exceeds its bounded context budget')
    return CodexSkill(skill_id='summary_failures', name='summary-failures',
                      path=str(target), content_blake3=blake3_bytes(body))


def memory_launch_options(base: CodexLaunchOptions | None = None) -> CodexLaunchOptions:
    """Keep model/provider/security/environment and explicit caller overrides."""
    return research_launch_options(base)


def memory_thread_options(base: CodexThreadOptions, *,
                          policy: ResearchContextPolicy = ResearchContextPolicy()) -> CodexThreadOptions:
    if not base.role.is_actor:
        raise ValueError('actor task-memory profile must not be applied to a judge')
    instructions = (base.developer_instructions or '') + '\n\n' + MEMORY_INSTRUCTIONS
    # Explicit caller settings win; inspection exposes the effective threshold.
    config = {**policy.config_defaults(), **dict(base.config)}
    return replace(base, config=config, developer_instructions=instructions, service_name=PROFILE_NAME)


def memory_turn_input(base: CodexTurnInput, *, skill_path: Path = SKILL_PATH) -> CodexTurnInput:
    """Mount the new public skill without changing existing tools or skill bytes."""
    added = summary_failures_skill(skill_path)
    current = [skill for skill in base.skills if skill.skill_id == added.skill_id]
    if current and current != [added]:
        raise ValueError('summary_failures skill identity conflicts with an existing mount')
    return base if current else replace(base, skills=(*base.skills, added))


def inspect_memory_profile(policy: ResearchContextPolicy = ResearchContextPolicy()) -> dict:
    skill = summary_failures_skill()
    return {'schema': 'eva.codex-research-memory-profile.v1', 'profile': PROFILE_NAME,
            'inspection_scope': 'profile defaults, not effective caller overrides or observed model capacity',
            'inherits': 'evamed-codex-v1.1-research', 'context_tokens': policy.context_tokens,
            'output_tokens': policy.output_tokens, 'reserve_tokens': policy.reserve_tokens,
            'compact_at_tokens': policy.compact_at_tokens, 'skill': dict(skill.catalog_entry()),
            'compaction_headroom_mode': policy.compaction_headroom_mode,
            'memory_scope': 'task public workspace', 'canonical_tool_schema_changes': False,
            'provider_calls': 0, 'behavioral_memory_verified': False,
            'caller_overrides_take_precedence': True}
