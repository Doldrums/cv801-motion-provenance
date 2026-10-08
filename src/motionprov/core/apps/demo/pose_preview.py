"""Read-only previews of the authentic PoseRecords behind indexed references."""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np

from motionprov.core.datasets.local_pose_cache import POSE_CACHE_VERSION
from motionprov.core.pose.local_records import RECORD_SCHEMA_VERSION, PoseRecord, _json_bytes

# Same SMPL tree as scripts/dataset_browser_figures.py. Stored coordinates are
# original joints 1..23; the root and its three incident edges are not observed
# points in these records and are deliberately absent from this topology.
SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
JOINT_NAMES = ("left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
               "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
               "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
               "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand")
EDGES = [[parent-1, child-1] for child, parent in enumerate(SMPL_PARENTS) if parent > 0]
MAX_PREVIEW_FRAMES = 900


def valid_record_id(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _inside_file(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Pose preview path escapes its trusted storage")
    return path if path.is_file() else None


def _same_source(pose, record):
    return (pose.metadata["video_id"] == record.metadata["video_id"]
            and pose.metadata["source_hash"] == record.metadata.get("source_hash"))


def resolve_reference_pose(record, *, cache_root=None, uploads_root=None):
    """Load just the selected pose, checking its binding to the indexed codes.

    A named upload changes only enrollment labels/source-group identifiers.
    Reconstruct those exact authorized metadata changes and compare the full
    expected pose hash; matching a filename or source hash alone is insufficient.
    """
    expected = record.metadata.get("pose_record_hash")
    if not valid_record_id(expected):
        return None
    if record.metadata.get("video_storage") == "uploads" and uploads_root is not None:
        video_id = record.metadata["video_id"]
        try:
            if str(uuid.UUID(video_id)) != video_id:
                return None
        except (ValueError, AttributeError):
            return None
        path = _inside_file(uploads_root, f"{video_id}.pose.npz")
        if path is None:
            return None
        pose = PoseRecord.load(path)
        if not _same_source(pose, record) or any(
                pose.metadata.get(key) != record.metadata.get(key)
                for key in ("video_storage", "video_filename")):
            raise ValueError("Uploaded pose source differs from the indexed reference")
        if pose.content_hash() != expected:
            metadata = dict(pose.metadata)
            for field in ("choreography_id", "source_group_id", "performance_id"):
                metadata[field] = record.metadata[field]
            if replace(pose, metadata=metadata).content_hash() != expected:
                raise ValueError("Uploaded pose content differs from the enrolled reference")
        return pose
    if cache_root is None:
        return None
    root = Path(cache_root).resolve()
    manifest_path = _inside_file(root, "manifest.json")
    if manifest_path is None:
        return None
    manifest = json.loads(manifest_path.read_text())
    digest = manifest.pop("cache_hash", None)
    if digest != hashlib.sha256(_json_bytes(manifest)).hexdigest():
        raise ValueError("Pose preview cache manifest checksum mismatch")
    if (manifest.get("cache_version") != POSE_CACHE_VERSION
            or manifest.get("record_schema_version") != RECORD_SCHEMA_VERSION):
        raise ValueError("Unsupported pose preview cache version")
    entries = [entry for entry in manifest["records"]
               if entry.get("content_hash") == expected
               and entry.get("video_id") == record.metadata["video_id"]]
    if not entries:
        return None
    if len(entries) != 1 or entries[0].get("path") != f"records/{expected}.npz":
        raise ValueError("Ambiguous or unsafe pose preview cache entry")
    path = _inside_file(root, entries[0]["path"])
    if path is None:
        return None
    pose = PoseRecord.load(path)
    if pose.content_hash() != expected or not _same_source(pose, record):
        raise ValueError("Pose preview cache content differs from the indexed reference")
    return pose


def load_query_pose(uploads_root, task_id):
    task_id = str(uuid.UUID(str(task_id)))
    path = _inside_file(uploads_root, f"{task_id}.pose.npz")
    if path is None:
        return None
    pose = PoseRecord.load(path)
    if pose.metadata["video_id"] != task_id:
        raise ValueError("Saved preview pose belongs to another upload")
    return pose


def preview_payload(pose, *, record_id=None, task_id=None, choreography_id=None,
                    indexed_pose_hash=None, start_us=None, end_us=None, display_up_axis="y"):
    """Keep a bounded contiguous interval of native frames, including all gaps."""
    pose.validate()
    display_up_axis = pose.metadata.get("display_up_axis", display_up_axis)
    if display_up_axis not in ("y", "-y"):
        raise ValueError("Pose preview display_up_axis must be y or -y")
    if (start_us is not None and (type(start_us) is not int or start_us < 0)
            or end_us is not None and (type(end_us) is not int or end_us < 1)
            or start_us is not None and end_us is not None and end_us <= start_us):
        raise ValueError("Preview requires a nonempty half-open timestamp interval")
    times = pose.time_us
    start = 0 if start_us is None else int(np.searchsorted(times, start_us, side="left"))
    stop = len(times) if end_us is None else int(np.searchsorted(times, end_us, side="left"))
    stop = max(start, stop)
    selected_stop = min(stop, start+MAX_PREVIEW_FRAMES)
    selected = slice(start, selected_stop)
    selected_times = times[selected]
    interval = None
    if len(selected_times):
        # A final recorded frame still has one native-frame exposure interval.
        period = (round(float(np.median(np.diff(times)))) if len(times) > 1 else
                  round(1_000_000 / float(pose.metadata.get("target_fps", 30))))
        right = int(times[selected_stop]) if selected_stop < len(times) else int(times[-1])+period
        interval = [int(selected_times[0]), min(right, end_us) if end_us is not None else right]
    return {"schema_version": 1, "kind": "reference" if record_id is not None else "query",
        "record_id": record_id, "task_id": task_id, "pose_hash": pose.content_hash(),
        "indexed_pose_hash": indexed_pose_hash, "video_id": pose.metadata["video_id"],
        "choreography_id": choreography_id or pose.metadata["choreography_id"],
        "skeleton_schema": pose.metadata["skeleton_schema"],
        "coordinate_frame": pose.metadata.get("coordinate_frame", "unspecified"),
        "display_up_axis": display_up_axis,
        "units": "source_pose_units", "root_included": False,
        "joint_ids": list(range(1, 24)), "joint_names": list(JOINT_NAMES), "edges": EDGES,
        "positions": pose.positions[selected].tolist(), "observed": pose.observed[selected].tolist(),
        "frame_valid": pose.frame_valid[selected].tolist(), "time_us": selected_times.tolist(),
        "frame_index": pose.frame_index[selected].tolist(), "interval_us": interval,
        "requested_interval_us": [start_us, end_us], "total_frames": len(times),
        "source_frame_count": int(pose.metadata.get("source_frame_count", len(times))), "observed_frames": int(pose.observed.any(-1).sum()),
        "returned_frame_count": len(selected_times),
        "observation_provenance": pose.metadata.get("observation_provenance"),
        "assumed_dense": pose.metadata.get("assumed_dense", False),
        "sampling": {"strategy": "contiguous_native_frames", "max_frames": MAX_PREVIEW_FRAMES,
                     "source_fps": pose.metadata.get("target_fps"), "interpolation": False,
                     "truncated": selected_stop < stop}}
