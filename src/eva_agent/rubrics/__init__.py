"""Versioned, shared rubric contracts for EVA-Agent evaluation and reward."""

from .compiler import (
    COMPILED_SCHEMA,
    COMPILER_VERSION,
    SOURCE_SCHEMA,
    RubricValidationError,
    compile_registry,
    load_and_compile_registry,
)
from .integrity import IntegrityDependencyError, blake3_document, canonical_json_bytes
from .models import CompiledRubric, RubricScore, RubricScoreError, SandboxRubricBinding
from .registry import CompiledRubricRegistry, RubricRegistryError

__all__ = [
    "COMPILED_SCHEMA",
    "COMPILER_VERSION",
    "SOURCE_SCHEMA",
    "CompiledRubric",
    "CompiledRubricRegistry",
    "IntegrityDependencyError",
    "RubricRegistryError",
    "RubricScore",
    "RubricScoreError",
    "RubricValidationError",
    "SandboxRubricBinding",
    "blake3_document",
    "canonical_json_bytes",
    "compile_registry",
    "load_and_compile_registry",
]
