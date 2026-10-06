---
name: med-image-synthesis
description: Solve medical image synthesis or enhancement tasks while preserving geometry, intensity conventions, metadata, and evaluation-compatible outputs.
---

Inspect modality, affine/orientation, voxel spacing, dtype, intensity range, and
the evaluation metric before transforming images. Prefer a measured baseline
and verify that saved outputs preserve case identity and required geometry.
Check for NaN, clipping, shape drift, missing cases, and suspicious metric gains
that indicate leakage or format mistakes.

