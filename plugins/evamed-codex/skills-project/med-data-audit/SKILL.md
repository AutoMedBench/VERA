---
name: med-data-audit
description: Audit medical datasets for identity, leakage, label integrity, file/schema consistency, and reproducible train/evaluation boundaries.
---

Identify the dataset version and hash before analysis. Inspect schema, sample
across shards and classes, and test joins between inputs, labels, splits, and
derived artifacts. Check patient or study leakage across splits and distinguish
missing, malformed, and clinically implausible values. Write machine-readable
findings with evidence references and do not repair source data silently.

