---
name: expert-consultation-mini
description: Request one host-authorized stronger-model research consultation through an opaque bounded request and integrate it as fallible evidence.
---

# Expert consultation

1. Select only a host-admitted expert and consultation mode that matches the
   stated gap. The current declared routing is deliberately narrow:
   - `gemini-3.1-pro-preview` is for **web-grounded evidence** only, when the
     host has separately approved Google Search grounding and citation capture.
     It is useful for locating current sources, not for replacing verification.
   - `gpt-5.5` is for **research-plan review** and **method review**. It is
     useful for critiquing a tool plan, study design, or validation approach;
     do not imply that it performed an external search unless the host supplied
     a separately governed evidence result.
   - `claude-opus-4-6` has no active verified declaration in this runtime.
     Do not request it unless a future host adds an explicit reviewed provider
     declaration, transport preflight, and consultation grant.
   Treat every expert response as fallible evidence, never as an answer key.
2. Send only the opaque host-issued consultation request identifier; do not
   place credentials, protected data, or unbounded prompts in the request.
3. Treat the consultation as a fallible second opinion. Compare it with
   approved evidence and record disagreements or uncertainty.
4. Do not use an expert consultation to bypass stage, data, network, or
   governance restrictions.
