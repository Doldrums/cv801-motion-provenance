"""Explicit storage roots without importing numerical packages or creating files."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def _root(name: str, env: Mapping[str, str], cwd: Path) -> Path | None:
    if name not in env:
        return None
    value = env[name]
    if not value.strip():
        raise ValueError(f'{name} must name a directory; an explicitly empty root is invalid')
    path = Path(value).expanduser()
    return (path if path.is_absolute() else cwd/path).resolve()


@dataclass(frozen=True)
class AssetRoots:
    comotion: Path
    smpl: Path


def asset_roots(legacy_assets: str | Path, *, environ: Mapping[str, str] | None = None,
                cwd: str | Path | None = None) -> AssetRoots:
    """Choose one asset tree; explicit roots never fall back based on contents."""
    env = os.environ if environ is None else environ
    launch = Path.cwd() if cwd is None else Path(cwd)
    launch = launch.resolve()
    selected = _root('CHOREO_COMOTION_ASSETS_ROOT', env, launch)
    if selected is None:
        generic = _root('CHOREO_ARTIFACTS_ROOT', env, launch)
        if generic is not None:
            selected = generic/'comotion'
    if selected is None:
        legacy = Path(legacy_assets)
        legacy = (legacy if legacy.is_absolute() else launch/legacy).resolve()
        selected = legacy
        # A wheel may share the src/motionprov.core/comotion/data suffix. Only the caller's
        # actual checkout, never launch cwd, can activate this new-layout default.
        if legacy.parts[-4:] == ('src', 'motionprov.core', 'comotion', 'data'):
            checkout = legacy.parents[3]
            modern = checkout/'data/artifacts/comotion'
            if (checkout/'pyproject.toml').is_file() and modern.is_dir():
                selected = modern
    smpl = _root('CHOREO_SMPL_ROOT', env, launch)
    return AssetRoots(selected, smpl if smpl is not None else selected/'smpl')


@dataclass(frozen=True)
class DemoStorage:
    gallery: Path
    checkpoints: Path
    uploads: Path
    logs: Path


def demo_storage(*, environ: Mapping[str, str] | None = None,
                 cwd: str | Path | None = None) -> DemoStorage:
    """Resolve a launch snapshot; read-only gallery and checkpoints stay distinct."""
    env = os.environ if environ is None else environ
    launch = (Path.cwd() if cwd is None else Path(cwd)).resolve()
    mutable = _root('CHOREO_VAR_ROOT', env, launch)
    return DemoStorage(
        gallery=_root('CHOREO_DEMO_GALLERY_ROOT', env, launch) or launch/'out',
        checkpoints=_root('CHOREO_CHECKPOINTS_ROOT', env, launch) or launch/'checkpoints',
        uploads=_root('CHOREO_DEMO_UPLOADS_ROOT', env, launch) or
                (mutable/'demo/uploads' if mutable is not None else launch/'uploads'),
        logs=_root('CHOREO_LOG_ROOT', env, launch) or
             (mutable/'logs' if mutable is not None else launch),
    )
