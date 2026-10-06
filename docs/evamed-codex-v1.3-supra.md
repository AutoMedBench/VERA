# EvaMed Codex v1.3 Supra

This prospective profile composes pinned Codex 0.153.4, existing medical MCP
tools/skills and v1.2 task memory. It does not patch the binary, rewrite frozen
trajectories, or replace native benchmark scoring rules.

## Modes and provider controls

| Mode | Scope | Actual requested control |
| --- | --- | --- |
| instant | Explicit loopback local Qwen only | Chat chat_template_kwargs.enable_thinking=false |
| think | Direct native GPT routes | Codex turn effort xhigh |
| think | Qwen Chat routes | Chat chat_template_kwargs.enable_thinking=true; not claimed equivalent to native xhigh |
| think | Other Chat routes | Explicit endpoint-supported reasoning protocol, verified per endpoint |

Requested cloud cohorts are gpt-6-astra, gpt-5.6-sol, gpt-5.6-luna, gpt-5.5,
Claude Opus 5 and Qwen3.6/3.5 API models with at least 27B total parameters.
Retain actual provider registry IDs exactly. Requested coverage does not mean
every endpoint is available. Local trained Qwen3.5-9B is the explicit exception
to the cloud size bound. Native instant is not part of the requested profile.

The existing Responses-to-Chat adapter accepts reasoning but does not forward
effort automatically. Supply SupraChatTransport through its upstream_transport
hook: this sets actual Chat controls without modifying messages/tools/schemas.
Do not add the older thinking-always-true wrapper around instant. Omitting a
thinking override does not disable thinking. chat_reasoning_effort is an explicit
route contract, not an assertion that an arbitrary Opus endpoint accepts it.
No silent mode downgrade, automatic retry or quota-bypassing key rotation.

## SDK use

Create SupraProfile with the exact model/provider, real ResearchContextPolicy,
protocol and mode. Apply supra_launch_options to the launch, thread_options to
the actor thread, and turn_input to each turn. Caller security and canonical
tools stay unchanged; the verified summary_failures skill is appended. Record
the extra skill alongside the original catalog: a 25-file catalog alone does
not describe a 26-skill effective mount. Inspection is not behavioral proof.

```sh
PYTHONPATH=src python scripts/evamed_codex_supra_v1_3.py \
  --model Qwen/Qwen3.5-9B --provider eva_local_qwen \
  --mode instant --protocol qwen_template \
  --local-qwen-endpoint http://127.0.0.1:30910/v1 \
  --context-tokens 32768 --compact-at-tokens 12288
```

Use actual provider capacity. Native Astra's observed post-compaction floor
exceeded 13k; a 12,288 trigger repeatedly recompacted already compacted context.
The subsequent native profile used 131,072 capacity and 49,152 compaction. This
is not an equal-budget comparison with the local 32,768-token Qwen server.
A supplied measured floor is checked against both defaults and caller overrides.

The prospective local seven-track configuration uses context32768, compact20480,
output4096, reserve2048 and measured compacted-input floor13100. Construct its
`ResearchContextPolicy` with `compaction_headroom_mode="exact_request_guard"`.
The historical `conservative_double` default is unchanged. The explicit mode
checks compact+output+reserve against capacity instead of assuming every history
doubles. It requires the caller's post-projection token-budget transport and
the real server context limit; it is not a declaration of additional capacity.
SGLang text-only Chat tokenization is exact; multimodal requests remain explicitly
unverified by that tokenizer and are forwarded unchanged to the server's checks.
Do not present text token counts as image-processor counts.

## Memory and medical tasks

Keep task-state separate from final answers. After resume/compaction, check the
current stage against the latest request. Read the final artifact before ending
and preserve drafts separately. These address observed stage drift/overwrites;
adding instructions alone does not prove those model failures are fixed.
Failure notes are public, task-scoped and evidence-linked; actual later reads
and correct use, not flags, establish memory behavior.

- AutoMedBench-Lite: existing seven tracks, S1-S5 process rubrics, prescribed
  model jobs, native metrics and same-thread restart/resume.
- AgentClinic: actual patient/measurement boundaries, no invented simulator
  tools and no hidden diagnoses in actor memory.
- HealthBench Professional: original conversation/rubric; answer-only E2E
  execution does not imply nonexistent S1-S4 stages.
- [HealthAgentBench](https://github.com/microsoft/HealthAgentBench): preserve
  task environments and native verifiers; public input separate from hidden
  gold/tests. Gated datasets require existing authorized access. Derived stages
  are observed process views, not replacement native metrics.

Parallelize independent trajectories and safe tools; coordinate shared writes
and GPU jobs separately. Retain exact model/mode/capacity, tool/skill mounts,
public traces, original failures, final workspace and actual Judge reads. The
30 focused CPU tests establish mechanics, not clinical quality or speed gains.
