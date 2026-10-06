# MCP usage

Keep discovery small and bind data tools to one verified candidate policy.

1. Call `capabilities` with the task domain and stage, then `search` and `load`
   only when workflow guidance is needed.
2. Call canonical `search_skills` with `{"query":"evidence","stage":"S2"}`;
   load only the selected result with `load_skill`. The schemas and return
   contracts are the existing EvaMed `SkillCatalog` contracts.
3. Start a candidate-bound bridge with both `EVAMED_MCP_POLICY_PATH` and the
   literal-file `EVAMED_MCP_POLICY_BLAKE3`. Inject handlers through
   `EVAMED_MCP_REGISTRY_FACTORY=module:function`. A policy-only tool is listed
   as unavailable and its call fails closed.
4. Independent tools may be sent in the same turn. Respect
   `_meta.evamed.parallelSafe`; an unsafe tool receives an exclusive frontier.

Never paste private rubric items, answers, or judge-only references into any
MCP argument. Never retry a provider, judge, or tool failure through the
bridge. Preserve the returned typed failure and its BLAKE3 receipt.
