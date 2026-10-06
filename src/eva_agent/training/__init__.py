"""Training-data materialization surfaces."""

from .bulk_rl_factory import (
    BULK_RL_CATALOG_SCHEMA,
    BULK_RL_SANDBOX_SCHEMA,
    BulkRLSandboxCatalog,
    BulkRLSandboxFactoryError,
    build_bulk_rl_sandbox_catalog,
    verify_bulk_rl_sandbox_catalog,
)
from .teacher_batch import (
    TEACHER_ROUTES,
    TeacherBatchError,
    TeacherCheckpoint,
    TeacherTask,
    iter_bulk_sandboxes,
    run_batch,
    run_persistent_batch,
)
from .teacher_worker import (
    CampaignV2TeacherContextPool,
    TeacherCandidateContext,
    bulk_initial_files,
    campaign_v2_candidate_context,
    execute_single_rollout,
    hydrate_teacher_stage_prerequisites,
    load_bulk_record,
)
from .progressive_skills import ProgressiveTeacherSkillSurface
from .persistent_teacher import PersistentCodexTeacherPool, TeacherRecordIndex
from .bulk_rl_release_overlay import (
    BulkRLReleaseOverlayError,
    derive_overlay_rows,
    fixed_evaluator_recipe,
    load_verified_release_view,
    release_manifest_core,
    release_view_records,
)
from .execution_verified_sft import (
    CANDIDATE_SFT_DATASET_SCHEMA,
    CANDIDATE_SFT_SLICE_SCHEMA,
    CANDIDATE_SFT_VERIFY_SCHEMA,
    DEFAULT_CANDIDATE_ROUTES,
    ExecutionVerifiedSFTBuild,
    ExecutionVerifiedSFTError,
    build_execution_verified_candidate_sft_dataset,
    verify_execution_verified_candidate_sft_dataset,
)

__all__ = [
    "BULK_RL_CATALOG_SCHEMA",
    "BULK_RL_SANDBOX_SCHEMA",
    "BulkRLSandboxCatalog",
    "BulkRLSandboxFactoryError",
    "build_bulk_rl_sandbox_catalog",
    "verify_bulk_rl_sandbox_catalog",
    "TEACHER_ROUTES",
    "TeacherBatchError",
    "TeacherCheckpoint",
    "TeacherTask",
    "iter_bulk_sandboxes",
    "run_batch",
    "run_persistent_batch",
    "CampaignV2TeacherContextPool",
    "TeacherCandidateContext",
    "bulk_initial_files",
    "campaign_v2_candidate_context",
    "execute_single_rollout",
    "hydrate_teacher_stage_prerequisites",
    "load_bulk_record",
    "ProgressiveTeacherSkillSurface",
    "PersistentCodexTeacherPool",
    "TeacherRecordIndex",
    "BulkRLReleaseOverlayError",
    "derive_overlay_rows",
    "fixed_evaluator_recipe",
    "load_verified_release_view",
    "release_manifest_core",
    "release_view_records",
    "CANDIDATE_SFT_DATASET_SCHEMA",
    "CANDIDATE_SFT_SLICE_SCHEMA",
    "CANDIDATE_SFT_VERIFY_SCHEMA",
    "DEFAULT_CANDIDATE_ROUTES",
    "ExecutionVerifiedSFTBuild",
    "ExecutionVerifiedSFTError",
    "build_execution_verified_candidate_sft_dataset",
    "verify_execution_verified_candidate_sft_dataset",
]
