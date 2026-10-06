---
name: fhir-interop-mini
description: Use host-bound FHIR research tools for bounded reads, searches, draft writes, and authorized commits with schema and privacy checks.
---

# FHIR interoperability

1. Use only host-provided FHIR identifiers and named operations; never infer a
   server endpoint, patient record, or write authority.
2. Validate resource type, required fields, terminology assumptions, and
   de-identification boundaries after every material result.
3. Treat draft writes as reversible artifacts and commit only after the
   declared validator and governance checks pass.
4. Do not expose raw protected records, credentials, or unsupported clinical
   claims in a FHIR draft or summary.

