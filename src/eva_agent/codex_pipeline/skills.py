"""Verified candidate-scoped skill mounts for the Codex SDK execution path.

The legacy skill bodies are intentionally not copied into EVA-Agent's source
tree because their pinned source snapshot is marked ``license: other``.  This
catalog reopens the external files, verifies their manifest and literal bytes,
and materializes those same bytes into a private content-addressed runtime
tree.  Codex receives only read-only runtime paths explicitly permitted for
the candidate stage.  The repo-owned stage-rollout skill follows the same
materialization boundary.
"""

from __future__ import annotations

import json
import fcntl
import os
from pathlib import Path, PurePosixPath
import stat
from types import MappingProxyType
from typing import Any, Mapping

from eva_agent.codex_runtime import CodexSkill
from eva_agent.pipeline.contracts import RolloutRequest, Stage
from eva_agent.pipeline.digests import blake3_bytes, blake3_hex, is_blake3

from .adapter import CodexPipelineError


LEGACY_SKILL_MANIFEST_SCHEMA = "eva.evamed-legacy-skill-manifest.v1"
LEGACY_SKILL_SOURCE_REVISION = "79dd2a31f5f"
# This value is deliberately independent of the writable manifest it anchors.
# The manifest's embedded digest proves only internal consistency; without a
# separately pinned value, a rewritten body/stage grant plus a recomputed
# embedded digest would be accepted as the historical source snapshot.
LEGACY_SKILL_MANIFEST_BLAKE3 = (
    "1d981263023cff2b0d1817081247cc4f511c65c79e399be3d4e443d6d610bc64"
)
LEGACY_SKILL_OCCURRENCES = 174
LEGACY_SKILL_UNIQUE_CONTENTS = 24
ACTOR_SKILL_MATERIALIZATION_SCHEMA = "eva.codex-actor-skill-materialization.v1"
ACTOR_SKILL_PATH_POLICY = "blake3-root/ordinal-content-blake3/SKILL.md"
_MAXIMUM_SKILL_BYTES = 4 * 1024 * 1024


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _normalized_absolute(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise CodexPipelineError(f"{label} must be an absolute normalized path")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CodexPipelineError(f"{label} is unavailable") from exc
    if resolved != path:
        raise CodexPipelineError(f"{label} uses a symlink or normalization alias")
    return resolved


def _stable_regular_bytes(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CodexPipelineError(f"{label} cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 1
            or before.st_size > _MAXIMUM_SKILL_BYTES
        ):
            raise CodexPipelineError(f"{label} topology or size differs")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise CodexPipelineError(f"{label} ended before its committed size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise CodexPipelineError(f"{label} exceeds its committed size")
        after = os.fstat(descriptor)
        try:
            entry = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise CodexPipelineError(f"{label} changed while reopening") from exc
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) != (
        entry.st_dev,
        entry.st_ino,
        entry.st_mode,
        entry.st_nlink,
        entry.st_size,
        entry.st_mtime_ns,
        entry.st_ctime_ns,
    ):
        raise CodexPipelineError(f"{label} changed while reopening")
    return b"".join(chunks)


def _open_private_runtime_root(path: Path) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        entry = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        try:
            os.close(descriptor)
        except (OSError, UnboundLocalError):
            pass
        raise CodexPipelineError("actor skill runtime root cannot be opened safely") from exc
    if (
        not stat.S_ISDIR(opened.st_mode)
        or _stat_identity(opened) != _stat_identity(entry)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o700
    ):
        os.close(descriptor)
        raise CodexPipelineError("actor skill runtime root must be a private owned directory")
    return descriptor


def _open_materialized_directory(
    parent_descriptor: int,
    name: str,
    *,
    expected_mode: int,
    label: str,
) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        opened = os.fstat(descriptor)
        entry = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as exc:
        try:
            os.close(descriptor)
        except (OSError, UnboundLocalError):
            pass
        raise CodexPipelineError(f"{label} cannot be opened safely") from exc
    if (
        not stat.S_ISDIR(opened.st_mode)
        or _stat_identity(opened) != _stat_identity(entry)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != expected_mode
    ):
        os.close(descriptor)
        raise CodexPipelineError(f"{label} topology or permissions differ")
    return descriptor


def _write_all(descriptor: int, payload: bytes, *, label: str) -> None:
    offset = 0
    try:
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise CodexPipelineError(f"{label} ended before its committed size")
            offset += written
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    except OSError as exc:
        raise CodexPipelineError(f"{label} could not be materialized") from exc


def _stable_materialized_bytes(
    parent_descriptor: int,
    name: str,
    *,
    expected_blake3: str,
    expected_size: int,
    label: str,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise CodexPipelineError(f"{label} cannot be reopened safely") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o400
            or before.st_size != expected_size
        ):
            raise CodexPipelineError(f"{label} topology or permissions differ")
        chunks: list[bytes] = []
        remaining = expected_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise CodexPipelineError(f"{label} ended before its committed size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise CodexPipelineError(f"{label} exceeds its committed size")
        after = os.fstat(descriptor)
        entry = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as exc:
        raise CodexPipelineError(f"{label} changed while reopening") from exc
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if (
        _stat_identity(before) != _stat_identity(after)
        or _stat_identity(after) != _stat_identity(entry)
        or blake3_bytes(payload) != expected_blake3
    ):
        raise CodexPipelineError(f"{label} bytes differ")
    return payload


def _materialize_skill_payloads(
    runtime_root: Path,
    payloads: tuple[Mapping[str, Any], ...],
) -> tuple[Path, str, Mapping[str, Path]]:
    rows = tuple(
        {
            "ordinal": index,
            "skill_id": row["skill_id"],
            "name": row["name"],
            "relative_path": (
                f"{index:03d}-{row['content_blake3']}/SKILL.md"
            ),
            "content_blake3": row["content_blake3"],
            "bytes": len(row["payload"]),
        }
        for index, row in enumerate(payloads)
    )
    core = {
        "schema": ACTOR_SKILL_MATERIALIZATION_SCHEMA,
        "path_policy": ACTOR_SKILL_PATH_POLICY,
        "skills": rows,
    }
    materialization_blake3 = blake3_hex(core)
    root_name = f"blake3-{materialization_blake3}"
    materialization_root = runtime_root / root_name
    runtime_descriptor = _open_private_runtime_root(runtime_root)
    try:
        # A directory descriptor lock serializes first materialization and
        # idempotent reopens without a writable lock artifact.
        fcntl.flock(runtime_descriptor, fcntl.LOCK_EX)
        created = False
        try:
            os.mkdir(root_name, mode=0o700, dir_fd=runtime_descriptor)
            created = True
        except FileExistsError:
            pass
        materialization_descriptor = _open_materialized_directory(
            runtime_descriptor,
            root_name,
            expected_mode=0o700 if created else 0o500,
            label="actor skill materialization root",
        )
        try:
            if created:
                for row, source in zip(rows, payloads, strict=True):
                    directory_name = PurePosixPath(row["relative_path"]).parts[0]
                    try:
                        os.mkdir(
                            directory_name,
                            mode=0o700,
                            dir_fd=materialization_descriptor,
                        )
                    except OSError as exc:
                        raise CodexPipelineError(
                            "actor skill materialization directory could not be created"
                        ) from exc
                    skill_descriptor = _open_materialized_directory(
                        materialization_descriptor,
                        directory_name,
                        expected_mode=0o700,
                        label=f"materialized skill {row['skill_id']}",
                    )
                    try:
                        flags = (
                            os.O_WRONLY
                            | os.O_CREAT
                            | os.O_EXCL
                            | getattr(os, "O_CLOEXEC", 0)
                            | getattr(os, "O_NOFOLLOW", 0)
                        )
                        try:
                            file_descriptor = os.open(
                                "SKILL.md",
                                flags,
                                0o400,
                                dir_fd=skill_descriptor,
                            )
                        except OSError as exc:
                            raise CodexPipelineError(
                                f"materialized skill {row['skill_id']} already exists"
                            ) from exc
                        try:
                            _write_all(
                                file_descriptor,
                                source["payload"],
                                label=f"materialized skill {row['skill_id']}",
                            )
                        finally:
                            os.close(file_descriptor)
                        os.fchmod(skill_descriptor, 0o500)
                        os.fsync(skill_descriptor)
                    finally:
                        os.close(skill_descriptor)
                os.fchmod(materialization_descriptor, 0o500)
                os.fsync(materialization_descriptor)

            expected_directories = {
                PurePosixPath(row["relative_path"]).parts[0] for row in rows
            }
            if set(os.listdir(materialization_descriptor)) != expected_directories:
                raise CodexPipelineError("actor skill materialization inventory differs")
            resolved: dict[str, Path] = {}
            for row, source in zip(rows, payloads, strict=True):
                directory_name = PurePosixPath(row["relative_path"]).parts[0]
                skill_descriptor = _open_materialized_directory(
                    materialization_descriptor,
                    directory_name,
                    expected_mode=0o500,
                    label=f"materialized skill {row['skill_id']}",
                )
                try:
                    if set(os.listdir(skill_descriptor)) != {"SKILL.md"}:
                        raise CodexPipelineError(
                            f"materialized skill {row['skill_id']} inventory differs"
                        )
                    reopened = _stable_materialized_bytes(
                        skill_descriptor,
                        "SKILL.md",
                        expected_blake3=row["content_blake3"],
                        expected_size=row["bytes"],
                        label=f"materialized skill {row['skill_id']}",
                    )
                    if reopened != source["payload"]:
                        raise CodexPipelineError(
                            f"materialized skill {row['skill_id']} literal bytes differ"
                        )
                finally:
                    os.close(skill_descriptor)
                resolved[row["skill_id"]] = (
                    materialization_root / directory_name / "SKILL.md"
                )
        finally:
            os.close(materialization_descriptor)
    finally:
        fcntl.flock(runtime_descriptor, fcntl.LOCK_UN)
        os.close(runtime_descriptor)
    return materialization_root, materialization_blake3, MappingProxyType(resolved)


def _relative_skill_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str):
        raise CodexPipelineError("legacy skill source path differs")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.name != "SKILL.md"
    ):
        raise CodexPipelineError("legacy skill source path is unsafe")
    return path


class VerifiedActorSkillCatalog:
    """Callable ``ActorSkillsFactory`` backed by verified local source bytes.

    Construction performs no provider call.  It verifies the complete
    24-skill external inventory up front and writes the literal verified bytes
    once into a private content-addressed runtime tree.  Existing trees are
    reopened byte-for-byte and permission-for-permission before reuse.
    """

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        legacy_source_root: str | Path,
        native_stage_skill_path: str | Path,
        runtime_root: str | Path,
    ) -> None:
        manifest = _normalized_absolute(manifest_path, label="legacy skill manifest")
        source = _normalized_absolute(legacy_source_root, label="legacy skill source root")
        native = _normalized_absolute(
            native_stage_skill_path, label="native stage-rollout skill"
        )
        runtime = _normalized_absolute(runtime_root, label="actor skill runtime root")
        if not source.is_dir():
            raise CodexPipelineError("legacy skill source root is not a directory")
        if not manifest.is_file() or not native.is_file() or not runtime.is_dir():
            raise CodexPipelineError("skill catalog inputs must be regular files")
        if (
            runtime == source
            or runtime.is_relative_to(source)
            or runtime == native.parent
            or runtime.is_relative_to(native.parent)
        ):
            raise CodexPipelineError("actor skill runtime root overlaps source material")

        raw_manifest = _stable_regular_bytes(manifest, label="legacy skill manifest")
        try:
            document = json.loads(raw_manifest)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CodexPipelineError("legacy skill manifest is not canonical JSON") from exc
        if not isinstance(document, dict):
            raise CodexPipelineError("legacy skill manifest must be an object")
        core = dict(document)
        manifest_blake3 = core.pop("manifest_blake3", None)
        if (
            document.get("schema") != LEGACY_SKILL_MANIFEST_SCHEMA
            or document.get("source_revision") != LEGACY_SKILL_SOURCE_REVISION
            or document.get("source_license_marker") != "other"
            or document.get("redistribution") != "external-only-license-unresolved"
            or type(document.get("occurrence_count")) is not int
            or document.get("occurrence_count") != LEGACY_SKILL_OCCURRENCES
            or type(document.get("unique_content_count")) is not int
            or document.get("unique_content_count") != LEGACY_SKILL_UNIQUE_CONTENTS
            or not isinstance(manifest_blake3, str)
            or not is_blake3(manifest_blake3)
            or manifest_blake3 != LEGACY_SKILL_MANIFEST_BLAKE3
            or manifest_blake3 != blake3_hex(core)
        ):
            raise CodexPipelineError("legacy skill manifest commitment differs")

        readme = source / "README.md"
        if readme.resolve(strict=True) != readme:
            raise CodexPipelineError("legacy skill README uses a symlink")
        readme_bytes = _stable_regular_bytes(readme, label="legacy skill README")
        if (
            blake3_bytes(readme_bytes) != document.get("source_readme_blake3")
            or b"license: other" not in readme_bytes
        ):
            raise CodexPipelineError("legacy skill source identity differs")

        rows = document.get("skills")
        if not isinstance(rows, list) or len(rows) != LEGACY_SKILL_UNIQUE_CONTENTS:
            raise CodexPipelineError("legacy skill inventory differs")
        stage_skill_ids: dict[str, list[str]] = {stage.value: [] for stage in Stage}
        source_inventory: list[Mapping[str, Any]] = []
        skill_payloads: list[Mapping[str, Any]] = []
        prior_skill_id = ""
        content_digests: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise CodexPipelineError("legacy skill row differs")
            skill_id = row.get("skill_id")
            description = row.get("description")
            digest = row.get("content_blake3")
            byte_count = row.get("bytes")
            stages = row.get("allowed_stages")
            if (
                not isinstance(skill_id, str)
                or not skill_id
                or skill_id <= prior_skill_id
                or not isinstance(description, str)
                or not description
                or not isinstance(digest, str)
                or not is_blake3(digest)
                or digest in content_digests
                or type(byte_count) is not int
                or not 1 <= byte_count <= _MAXIMUM_SKILL_BYTES
                or not isinstance(stages, list)
                or not stages
                or any(
                    not isinstance(stage, str) or stage not in stage_skill_ids
                    for stage in stages
                )
                or stages != sorted(set(stages))
            ):
                raise CodexPipelineError("legacy skill row commitment differs")
            relative = _relative_skill_path(row.get("canonical_source_path"))
            skill_path = source.joinpath(*relative.parts)
            try:
                resolved_skill_path = skill_path.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise CodexPipelineError("legacy skill source is unavailable") from exc
            if resolved_skill_path != skill_path or not skill_path.is_file():
                raise CodexPipelineError("legacy skill source uses unsafe topology")
            payload = _stable_regular_bytes(skill_path, label=f"legacy skill {skill_id}")
            if len(payload) != byte_count or blake3_bytes(payload) != digest:
                raise CodexPipelineError("legacy skill source bytes differ")
            for stage in stages:
                stage_skill_ids[stage].append(skill_id)
            skill_payloads.append(
                MappingProxyType(
                    {
                        "skill_id": skill_id,
                        "name": skill_id,
                        "payload": payload,
                        "content_blake3": digest,
                    }
                )
            )
            source_inventory.append(
                MappingProxyType(
                    {
                        "skill_id": skill_id,
                        "description": description,
                        "allowed_stages": tuple(stages),
                        "source_path": str(skill_path),
                        "content_blake3": digest,
                        "bytes": byte_count,
                    }
                )
            )
            prior_skill_id = skill_id
            content_digests.add(digest)

        native_bytes = _stable_regular_bytes(native, label="native stage-rollout skill")
        native_id = "eva-workflow/stage-rollout"
        materialization_root, materialization_blake3, materialized_paths = (
            _materialize_skill_payloads(
                runtime,
                (
                    MappingProxyType(
                        {
                            "skill_id": native_id,
                            "name": "stage-rollout",
                            "payload": native_bytes,
                            "content_blake3": blake3_bytes(native_bytes),
                        }
                    ),
                    *tuple(skill_payloads),
                ),
            )
        )
        native_skill = CodexSkill(
            skill_id=native_id,
            name="stage-rollout",
            path=str(materialized_paths[native_id]),
            content_blake3=blake3_bytes(native_bytes),
        )
        legacy_skills = {
            row["skill_id"]: CodexSkill(
                skill_id=row["skill_id"],
                name=row["name"],
                path=str(materialized_paths[row["skill_id"]]),
                content_blake3=row["content_blake3"],
            )
            for row in skill_payloads
        }
        self._by_stage = MappingProxyType(
            {
                stage: (
                    native_skill,
                    *(legacy_skills[skill_id] for skill_id in stage_skill_ids[stage]),
                )
                for stage in sorted(stage_skill_ids)
            }
        )
        self._inventory = tuple(
            MappingProxyType(
                {
                    **dict(row),
                    "path": str(materialized_paths[row["skill_id"]]),
                }
            )
            for row in source_inventory
        )
        catalog_core = {
            "schema": "eva.codex-actor-skill-mount-catalog.v1",
            "legacy_manifest_blake3": manifest_blake3,
            "materialization": {
                "schema": ACTOR_SKILL_MATERIALIZATION_SCHEMA,
                "root": str(materialization_root),
                "blake3": materialization_blake3,
                "path_policy": ACTOR_SKILL_PATH_POLICY,
            },
            "native_skill": native_skill.catalog_entry(),
            "stage_mounts": {
                stage: tuple(skill.catalog_entry() for skill in skills)
                for stage, skills in self._by_stage.items()
            },
        }
        self.manifest_blake3 = manifest_blake3
        self.materialization_root = materialization_root
        self.materialization_blake3 = materialization_blake3
        self.materialization_path_policy = ACTOR_SKILL_PATH_POLICY
        self.catalog_blake3 = blake3_hex(catalog_core)

    def for_stage(self, stage: Stage | str) -> tuple[CodexSkill, ...]:
        value = stage.value if isinstance(stage, Stage) else stage
        if not isinstance(value, str) or value not in self._by_stage:
            raise CodexPipelineError("actor skill stage differs")
        return self._by_stage[value]

    def __call__(self, request: RolloutRequest) -> tuple[CodexSkill, ...]:
        if not isinstance(request, RolloutRequest):
            raise CodexPipelineError("actor skill factory requires a RolloutRequest")
        return self.for_stage(request.sandbox.stage)

    def inventory(self) -> tuple[Mapping[str, Any], ...]:
        return self._inventory


__all__ = [
    "ACTOR_SKILL_MATERIALIZATION_SCHEMA",
    "ACTOR_SKILL_PATH_POLICY",
    "LEGACY_SKILL_MANIFEST_SCHEMA",
    "LEGACY_SKILL_MANIFEST_BLAKE3",
    "LEGACY_SKILL_OCCURRENCES",
    "LEGACY_SKILL_SOURCE_REVISION",
    "LEGACY_SKILL_UNIQUE_CONTENTS",
    "VerifiedActorSkillCatalog",
]
