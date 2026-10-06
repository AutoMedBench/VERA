# Third-party notices

EVA-Agent is distributed under the Apache License 2.0 in [`LICENSE`](LICENSE).
That license applies to this repository's original code and documentation; it
does not relicense third-party software, benchmark material, datasets, model
outputs, or source-derived rubric text.

## Python and Codex dependencies

The direct dependencies declared in `pyproject.toml` are distributed under
their own terms. The versions inspected for this release audit were:

| Distribution | Inspected version | Declared upstream license expression |
| --- | ---: | --- |
| `blake3` | 1.0.9 | CC0-1.0 OR Apache-2.0 |
| `cryptography` | 50.0.1 | Apache-2.0 OR BSD-3-Clause |
| `jsonschema` | 4.26.0 | MIT |
| `openai-codex` | 0.147.0 | Apache-2.0 |
| `openai-codex-cli-bin` | 0.147.0 | Apache-2.0 |
| `openai` | 2.54.0 | Apache-2.0 |
| `pytest` (development only) | 9.1.1 | MIT |

The installed distributions contain their authoritative license texts and
notices. Transitive dependencies remain governed by their respective terms.

## Benchmark-as-Teacher material

The rubric provenance catalog identifies the pinned
`benchmark-as-teacher-v2` revision
`27124a2a35fa9ee68ef890bd7e4a36ffb5802860`. The checkout inspected during
this audit did not contain a `LICENSE`, `COPYING`, or equivalent grant.
Some files under `rubrics/source/` describe criteria adapted from that
revision. **Rights review is required before public redistribution of those
source-derived rubric materials.** The EVA-Agent Apache-2.0 license is not a
grant of rights in them.

## Medical benchmarks and legacy authorities

AutoMedBench, MedXpertQA, AgentClinic, HealthBench, and the pinned legacy
`rlevo-med-research` authorities are external inputs. Their benchmark records,
datasets, images, and historical run bodies are not licensed by this
repository. EVA-Agent stores provenance identifiers, digests, schemas, and
integration logic so an operator can supply authorized copies. Each operator
must review and comply with the applicable upstream dataset and benchmark
terms before ingestion, training, publication, or redistribution.

The September 2026 harness source sync includes the owner's requested pinned
legacy **skill instructions and catalogs only** under
`plugins/evamed-codex/vendor/`. Their original `license: other` metadata and
source provenance are retained. See that directory's `NOTICE.md`; this
repository's Apache license does not supply a new license for those materials.

Generated trajectories, model responses, and sandbox publications are runtime
artifacts and are intentionally excluded from the source distribution. Their
use may also be governed by provider terms and the licenses of their source
benchmarks.
