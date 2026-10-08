"""Authentic observation-aware CoMotion adapters and explicit legacy cache import."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from motionprov.core.pose.local_records import RECORD_SCHEMA_VERSION, SKELETON_SCHEMA, PoseRecord, _json_bytes
from motionprov.core.pose.preprocessing import PREPROCESSING_VERSION, process_comotion_data

POSE_CACHE_VERSION = 1


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _source_digest(data):
    h = hashlib.sha256()
    for key in sorted(data):
        value = data[key]
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            a = np.ascontiguousarray(_numpy(value))
            if a.dtype.hasobject:
                raise ValueError(f"Source field {key} cannot contain Python objects")
            h.update(_json_bytes([key, a.dtype.str, a.shape]))
            h.update(a.tobytes())
        elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
            h.update(_json_bytes([key, value]))
        else:
            raise ValueError(f"Unsupported source metadata field {key}")
    return h.hexdigest()


def _source_times(data, metadata, length, fps):
    if fps is not None:
        rate, source = float(fps), "explicit_fps_override"
    else:
        for key, factor in (("time_us", 1), ("timestamps_us", 1), ("timestamps_s", 1_000_000)):
            value = data.get(key, metadata.get(key))
            if value is not None:
                values = np.asarray(_numpy(value), dtype=np.float64)
                if values.shape != (length,) or not np.isfinite(values).all():
                    raise ValueError(f"{key} must contain one finite timestamp per dense source frame")
                times = np.rint(values * factor).astype(np.int64)
                if length > 1 and np.any(np.diff(times) <= 0):
                    raise ValueError("Source timestamps must be strictly increasing")
                return times, {"time_source": key, "fps_override": None}
        rate = data.get("fps", metadata.get("fps"))
        if rate is None:
            raise ValueError("Source timestamps/fps are missing; supply an explicit fps override")
        rate, source = float(rate), "source_fps_metadata"
    if not np.isfinite(rate) or rate <= 0 or rate > 1_000_000:
        raise ValueError("fps must be finite and in (0, 1000000]")
    times = np.rint(np.arange(length, dtype=np.float64) * (1_000_000 / rate)).astype(np.int64)
    return times, {"time_source": source, "source_fps": rate,
                   "fps_override": rate if fps is not None else None}


def _source_length(data, metadata, minimum):
    """Keep the full video clock, including leading/trailing absent detections."""
    declared = []
    for owner in (data, metadata):
        for key in ("num_frames", "source_frame_count"):
            if key in owner:
                value = _numpy(owner[key])
                if (value.shape != () or not np.issubdtype(value.dtype, np.integer)
                        or int(value) < 0):
                    raise ValueError(f"{key} must be a nonnegative integer frame count")
                declared.append(int(value))
        for key in ("time_us", "timestamps_us", "timestamps_s"):
            if key in owner:
                value = _numpy(owner[key])
                if value.ndim != 1:
                    raise ValueError(f"{key} must be a dense timestamp vector")
                declared.append(len(value))
    if declared and (min(declared) < minimum or len(set(declared)) != 1):
        raise ValueError("Declared source frame counts/timestamps conflict or are shorter than detections")
    return declared[0] if declared else minimum


def pose_record_from_comotion(data: dict, *, metadata: dict, fps: float | None = None,
                              target_fps: float | None = 30,
                              smpl_decoder=None) -> PoseRecord:
    """Decode root-relative SMPL24 with genuine detection/finite masks.

    Missing detection rows retain their time slots. Ready joints require the
    SMPL24 layout; another skeleton must be mapped explicitly before this adapter.
    Source FPS is never guessed. Only SMPL data instantiate a lazy decoder.
    """
    metadata = dict(metadata)
    source_hash = metadata.get("source_hash") or _source_digest(data)
    clean_data = dict(data)
    if "joints3d" in data:
        joints = np.asarray(_numpy(data["joints3d"]), dtype=np.float32)
        if joints.ndim != 3 or joints.shape[1:] != (24, 3):
            raise ValueError("Ready joints must explicitly use the SMPL24 XYZ layout")
        n = len(joints)
        source_index = _numpy(data.get("frame_idx", np.arange(n, dtype=np.int64)))
        if source_index.shape != (n,) or not np.issubdtype(source_index.dtype, np.integer):
            raise ValueError("Ready joints require integer frame_idx for each source row")
        source_index = source_index.astype(np.int64)
        if np.any(source_index < 0) or len(np.unique(source_index)) != n:
            raise ValueError("Ready joints must identify unique nonnegative source frames")
        length = _source_length(data, metadata, int(source_index.max()) + 1 if n else 0)
        present = np.isfinite(joints).all(-1)
        if "observed" in data:
            original = _numpy(data["observed"])
            if original.shape == (n,):
                original = np.broadcast_to(original[:, None], (n, 24))
            if original.shape != (n, 24) or original.dtype != np.bool_:
                raise ValueError("Ready joints observed must be bool[N] or bool[N,24]")
            present &= original
        present &= present[:, :1]  # Root-relative coordinates need the same frame's root.
        dense = np.zeros((length, 24, 3), np.float32)
        dense_observed = np.zeros((length, 24), bool)
        dense[source_index] = np.where(present[..., None], joints, 0)
        dense_observed[source_index] = present
        clean_data["joints3d"] = torch.from_numpy(dense)
        # This adapter has already admitted the generic ready-joint source and
        # densified its mask. Do not reinterpret legitimate zero coordinates or
        # reuse the original sparse observed array against the dense tensor.
        clean_data["observed"] = torch.from_numpy(dense_observed)
        clean_data.pop("frame_valid", None)  # Applied on the full clock below.
        skeleton, _ = process_comotion_data(clean_data, None, 0, length, return_mask=True)
        observed = dense_observed[:, 1:]
        provenance = "ready_joints_source_presence_and_finite"
        dropped = int((~present.all(-1)).sum())
    else:
        if "frame_idx" not in data:
            raise ValueError("CoMotion detections require frame_idx")
        frame_indices = _numpy(data["frame_idx"])
        if frame_indices.ndim != 1 or not np.issubdtype(frame_indices.dtype, np.integer):
            raise ValueError("frame_idx must be an integer vector")
        if np.any(frame_indices < 0):
            raise ValueError("frame_idx must be nonnegative")
        length = _source_length(data, metadata,
                                int(frame_indices.max()) + 1 if len(frame_indices) else 0)
        keep = np.ones(len(frame_indices), bool)
        for key in ("pose", "trans", "betas"):
            values = _numpy(data[key])
            if values.shape[0] != len(frame_indices):
                raise ValueError(f"Detection rows in {key} do not match frame_idx")
            keep &= np.isfinite(values).reshape(len(values), -1).all(-1) if len(values) else keep
        for key in ("frame_idx", "pose", "trans", "betas", "id"):
            if key in data:
                clean_data[key] = torch.as_tensor(_numpy(data[key])[keep])
        if keep.any():
            if smpl_decoder is None:
                from motionprov.core.comotion.utils.smpl_kinematics import SMPLKinematics
                smpl_decoder = SMPLKinematics()
            skeleton, frame_observed = process_comotion_data(
                clean_data, smpl_decoder, 0, length, return_mask=True)
        else:
            skeleton = torch.zeros(3, length, 24)
            frame_observed = torch.zeros(length, dtype=torch.bool)
        xyz = _numpy(skeleton).transpose(1, 2, 0)
        if xyz.shape != (length, 24, 3):
            raise ValueError("SMPL decoder must return the SMPL24 layout")
        observed = np.broadcast_to(_numpy(frame_observed)[:, None], (length, 23)).copy()
        observed &= np.isfinite(xyz[:, 1:]).all(-1)
        observed &= np.isfinite(xyz[:, :1]).all(-1)
        provenance = "selected_comotion_detection_frame_mask"
        dropped = int((~keep).sum())
    xyz = np.asarray(_numpy(skeleton).transpose(1, 2, 0)[:, 1:], dtype=np.float32)
    times, clock = _source_times(data, metadata, length, fps)
    metadata.update(
        source_hash=source_hash, skeleton_schema=SKELETON_SCHEMA,
        preprocessing_id=f"comotion-preprocess-{PREPROCESSING_VERSION}/local-mask-v1",
        observation_provenance=provenance, assumed_dense=False,
        nonfinite_source_rows=dropped, coordinate_frame="root_relative",
        source_frame_count=length, **clock)
    frame_valid = np.ones(length, bool)
    if "frame_valid" in data:
        frame_valid = _numpy(data["frame_valid"])
        if frame_valid.shape != (length,) or frame_valid.dtype != np.bool_:
            raise ValueError("frame_valid must be bool[T] on the dense source grid")
        observed &= frame_valid[:, None]
    record = PoseRecord(xyz, observed.astype(bool), frame_valid, times,
                        np.arange(length, dtype=np.int64), metadata)
    return resample_pose_record(record, target_fps) if target_fps is not None else record


def resample_pose_record(record: PoseRecord, target_fps: float) -> PoseRecord:
    """Nearest-source grid; a copied source sample is observed at most once.

    No interpolated XYZ is called an observation. The output clock is the
    declared target grid; exact selected source times/indices remain in metadata.
    Upsampling leaves duplicate-source slots invalid instead of fabricating data.
    """
    record.validate()
    rate = float(target_fps)
    if not np.isfinite(rate) or rate <= 0 or rate > 1_000_000:
        raise ValueError("target_fps must be finite and in (0,1000000]")
    if not len(record.time_us):
        metadata = dict(record.metadata, target_fps=rate,
                        resampling_policy="nearest_unique_source_v1")
        return PoseRecord(record.positions.copy(), record.observed.copy(),
                          record.frame_valid.copy(), record.time_us.copy(),
                          record.frame_index.copy(), metadata)
    if len(record.time_us) == 1:
        grid = record.time_us.copy()
    else:
        duration = int(record.time_us[-1] - record.time_us[0])
        count = int(np.ceil(duration * rate / 1_000_000)) + 1
        grid = int(record.time_us[0]) + np.rint(
            np.arange(count, dtype=np.float64) * 1_000_000 / rate).astype(np.int64)
        grid = grid[grid <= record.time_us[-1]]
    right = np.searchsorted(record.time_us, grid).clip(0, len(record.time_us) - 1)
    left = (right - 1).clip(0)
    nearest = np.where(np.abs(record.time_us[left] - grid) <=
                       np.abs(record.time_us[right] - grid), left, right)
    unique = np.ones(len(nearest), bool)
    if len(nearest) > 1:
        unique[1:] = nearest[1:] != nearest[:-1]
    # Reused samples are absent, not newly measured repeated poses.
    observed = record.observed[nearest] & unique[:, None]
    metadata = dict(record.metadata)
    metadata.update(
        target_fps=rate, resampling_policy="nearest_unique_source_v1",
        resampling_parent_hash=record.content_hash(),
        selected_source_frame_index=record.frame_index[nearest].tolist(),
        selected_source_time_us=record.time_us[nearest].tolist(),
        repeated_source_slots=int((~unique).sum()), output_time_source="target_grid")
    return PoseRecord(record.positions[nearest].copy(), observed,
                      record.frame_valid[nearest].copy(), grid,
                      np.arange(len(grid), dtype=np.int64), metadata)


def pose_record_from_legacy_xyz(xyz, *, metadata: dict, fps: float,
                                allow_assumed_dense: bool = False) -> PoseRecord:
    """Explicitly import an XYZ-only cache without claiming authentic visibility."""
    if not allow_assumed_dense:
        raise ValueError("Legacy XYZ has no observation provenance; allow_assumed_dense explicitly")
    x = np.asarray(xyz)
    if x.ndim != 3 or x.shape[1:] != (23, 3):
        raise ValueError("Legacy adapter requires explicit frame-major [T,23,3] XYZ")
    if not np.isfinite(x).all():
        raise ValueError("Legacy assumed-dense coordinates must be finite")
    x = x.astype(np.float32)
    metadata = dict(metadata)
    metadata.update(
        skeleton_schema=SKELETON_SCHEMA, preprocessing_id="legacy_xyz_explicit_v1",
        source_hash=metadata.get("source_hash") or hashlib.sha256(x.tobytes()).hexdigest(),
        observation_provenance="assumed_dense", assumed_dense=True)
    times, clock = _source_times({}, metadata, len(x), fps)
    metadata.update(clock)
    return PoseRecord(x, np.ones((len(x), 23), bool), np.ones(len(x), bool),
                      times, np.arange(len(x), dtype=np.int64), metadata)


def save_pose_cache(records: Mapping[str, PoseRecord], directory: str | Path) -> Path:
    """Write content-addressed pose records, then atomically publish their manifest."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for video_id in sorted(records):
        record = records[video_id]
        record.validate()
        if video_id != record.metadata["video_id"]:
            raise ValueError("Cache mapping key must equal record video_id")
        digest = record.content_hash()
        relative = f"records/{digest}.npz"
        record.save(directory / relative)
        entries.append({"video_id": video_id, "path": relative, "content_hash": digest,
                        "observation_provenance": record.metadata.get("observation_provenance"),
                        "assumed_dense": record.metadata.get("assumed_dense", False)})
    manifest = {"cache_version": POSE_CACHE_VERSION,
                "record_schema_version": RECORD_SCHEMA_VERSION, "records": entries}
    manifest["cache_hash"] = hashlib.sha256(_json_bytes(manifest)).hexdigest()
    path = directory / "manifest.json"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".manifest.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(_json_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_pose_cache(directory: str | Path, *, allow_assumed_dense: bool = False,
                    expected_manifest_sha256: str | None = None) -> dict[str, PoseRecord]:
    """Load records bound to the exact manifest snapshot admitted by the caller."""
    directory = Path(directory).resolve()
    raw = (directory / "manifest.json").read_bytes()
    if (expected_manifest_sha256 is not None
            and hashlib.sha256(raw).hexdigest() != expected_manifest_sha256):
        raise ValueError("Pose cache manifest raw checksum mismatch")
    manifest = json.loads(raw)
    digest = manifest.pop("cache_hash", None)
    if digest != hashlib.sha256(_json_bytes(manifest)).hexdigest():
        raise ValueError("Pose cache manifest checksum mismatch")
    if (manifest.get("cache_version") != POSE_CACHE_VERSION
            or manifest.get("record_schema_version") != RECORD_SCHEMA_VERSION):
        raise ValueError("Unsupported pose cache version")
    records = {}
    for entry in manifest["records"]:
        path = (directory / entry["path"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Pose record path escapes its cache")
        record = PoseRecord.load(path)
        if record.content_hash() != entry["content_hash"]:
            raise ValueError("Pose cache record hash mismatch")
        video_id = entry["video_id"]
        if video_id in records or video_id != record.metadata["video_id"]:
            raise ValueError("Duplicate or mismatched video ID in cache")
        if record.metadata.get("assumed_dense", False) and not allow_assumed_dense:
            raise ValueError("Cache contains explicitly assumed-dense legacy observations")
        records[video_id] = record
    return records
