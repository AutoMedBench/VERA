# Schema-preserving MCP compatibility

`eva_agent.mcp_compat` is a transport-only boundary between canonical EvaMed
policy tools and MCP. It does not define or modify any tool, skill, rubric,
policy, or evidence contract.

For every canonical policy row:

```json
{"name": "...", "description": "...", "input_schema": {}}
```

the adapter emits exactly one MCP list entry:

```json
{"name": "...", "description": "...", "inputSchema": {}}
```

Only the outer protocol key is renamed. The schema is detached from the source,
validated as Draft 2020-12 JSON Schema, and required to have identical canonical
JSON bytes and BLAKE3 content before and after projection. Input ordering is
retained; duplicate tool names fail closed.

## Commitments and sidecars

`build_mcp_catalog()` returns an immutable `MCPToolCatalog`. Its wire document
contains:

- `tools`: the ordered MCP tool list;
- `schema_commitments`: an ordered per-tool `input_schema_blake3` list;
- `annotations`: ordered `visibility`, `read_only`, `mutating`, and
  `parallel_safe` rows;
- `schema_catalog_blake3`: a BLAKE3 commitment over the complete ordered MCP
  tool list;
- `catalog_blake3`: a BLAKE3 commitment over the tools, schema commitments,
  annotations, catalog schema identifier, and schema-catalog commitment.

Annotations never enter `inputSchema`. Unspecified annotations use conservative
defaults: public, mutating, and not parallel-safe. A tool must be exactly one of
read-only or mutating.

Use `verify_mcp_catalog()` before provider execution. Verification rejects
unknown or missing fields, invalid schemas, duplicate names, sidecar/name/order
mismatches, schema or catalog tampering, and an optional mismatch against a
previously pinned `expected_catalog_blake3`. Passing `canonical_rows=` also
reopens the exact policy-to-MCP projection and catches source catalog reordering.

## Per-policy binding and multi-variant inventories

Serving is bound to one policy/candidate, not to a process-wide map keyed by
tool name. `bind_policy_catalog()` takes the required BLAKE3 of the exact source
policy bytes plus its policy and candidate identities, then commits those to the
exact catalog. Before execution, reopen both the required source-policy digest
and the canonical rows with `PolicyCatalogBinding.verify()`. The source file is
read-only: measure its literal bytes before decoding, and never write a
normalized representation back to it.

`build_policy_catalog_inventory()` creates a deterministic, digest-only view
across bindings. It sorts by `(policy_id, candidate_id)` and counts variants by
`(tool name, exact input-schema BLAKE3)`. It deliberately does not produce a
merged MCP serving list. Duplicate names are forbidden within one policy but
are expected across policies, including when their exact schemas differ.

The inspected legacy snapshot contains 3,287 `sandbox-policy.v2` policies, six
tool names, and twelve exact name/schema variants: `execute_code` has three,
`submit_results` has five, and the other four names have one each. These are
inventory facts, not hard-coded schemas or validation constants. Runtime code
must load the selected policy instance and verify its required BLAKE3; it must
never select a global schema by tool name. No private policy payload belongs in
the repository.

## Existing definitions and skills

`tool_definitions_to_mcp_catalog()` accepts both pipeline `ToolDefinition`
objects (`input_schema`) and harness `ToolDefinition` objects (`parameters`). It
reads their metadata without replacing handlers or changing argument/result
contracts.

`skill_catalog_to_mcp_catalog()` obtains the existing `search_skills` and
`load_skill` definitions from a harness `SkillCatalog`, preserves their names,
descriptions, and parameter schemas, and records both as read-only sidecar
entries. Skill search/load results remain produced by the original handlers.

Minimal usage:

```python
from eva_agent.mcp_compat import ToolAnnotations, build_mcp_catalog

catalog = build_mcp_catalog(
    policy_tool_rows,
    annotations={
        "read_evidence": ToolAnnotations(
            read_only=True, mutating=False, parallel_safe=True
        )
    },
)
mcp_tools = catalog.to_document()["tools"]
catalog.verify(expected_catalog_blake3=catalog.catalog_blake3)
```
