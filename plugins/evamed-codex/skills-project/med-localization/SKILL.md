---
name: med-localization
description: Handle medical detection and segmentation tasks with coordinate, geometry, label, and empty-case checks tied to the official metric.
---

Determine the coordinate and label conventions from the task files. Preserve
image geometry and case IDs through preprocessing and postprocessing. For
detection, verify box order, scale, bounds, scores, and duplicate suppression.
For segmentation, verify label IDs, shape, affine/orientation, and empty masks.
Run the supplied format checker and metric on a small validation slice before
full inference.

