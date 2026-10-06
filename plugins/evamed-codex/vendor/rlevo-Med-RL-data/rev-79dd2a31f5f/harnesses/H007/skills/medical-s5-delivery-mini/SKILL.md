---
name: medical-s5-delivery-mini
description: Validate and deliver bounded medical-research artifacts with independent checks rather than treating an execution success as completion.
---

# Delivery validation

1. Run the host-registered validator against the declared artifact identifier.
2. Check schema, counts, non-empty required outputs, finite values, and
   declared delivery format independently.
3. Repair the earliest failed invariant and rerun validation before delivery.
4. Submit only a verified host-approved artifact. Do not fabricate a pass,
   suppress a failed check, or expose protected data in the delivery receipt.

