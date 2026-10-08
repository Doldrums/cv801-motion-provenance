"""Versioned NumPy records for local fingerprints, with explicit time and validity.

Files contain only numeric arrays and UTF-8 JSON, never pickled Python objects.
Content identity excludes container timestamps and is independent of JSON key order.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

RECORD_SCHEMA_VERSION = 1
SKELETON_SCHEMA = "smpl24_nonroot23_v1"
SOURCE_FIELDS = ("video_id", "choreography_id", "source_group_id", "performance_id")


def code_record_metadata(metadata, **extra):
    """Keep retrieval provenance by content reference, not duplicated frame arrays.

    The detailed PoseRecord remains an optional audit archive addressed by
    pose_record_hash. Core timestamps and validity remain in the code record.
    This changes metadata storage only, never assignments or quality values.
    """
    detailed = {"selected_source_time_us", "selected_source_frame_index", "timestamps_us",
                "time_us", "timestamps_s", "source_file", "source_manifest"}
    result = {key: value for key, value in metadata.items() if key not in detailed}
    result.update(extra)
    result["detailed_pose_provenance"] = "external_pose_record_by_content_hash"
    return result


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _array(name, value, dtype, shape):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype):
        raise ValueError(f"{name} must be a NumPy {np.dtype(dtype)} array")
    if len(value.shape) != len(shape) or any(
            expected is not None and actual != expected
            for actual, expected in zip(value.shape, shape)):
        raise ValueError(f"{name}: expected shape {shape}, got {value.shape}")


def _metadata(metadata, extra=()):
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON dictionary")
    for key in (*SOURCE_FIELDS, *extra):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"metadata requires nonempty {key}")
    try:
        _json_bytes(metadata)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain finite JSON values") from exc


def _increasing(name, values):
    if len(values) > 1 and np.any(values[1:] <= values[:-1]):
        raise ValueError(f"{name} must be strictly increasing")


def _content_digest(record_type, metadata, arrays):
    h = hashlib.sha256()
    h.update(_json_bytes({"record_type": record_type,
                          "schema_version": RECORD_SCHEMA_VERSION, "metadata": metadata}))
    for name, value in arrays.items():
        arr = np.ascontiguousarray(value)
        header = _json_bytes({"name": name, "dtype": arr.dtype.str,
                              "shape": arr.shape, "nbytes": arr.nbytes})
        h.update(len(header).to_bytes(8, "little"))
        h.update(header)
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()


class _Record:
    ARRAY_FIELDS: ClassVar[tuple[str, ...]]
    RECORD_TYPE: ClassVar[str]

    def validate(self):
        raise NotImplementedError

    @property
    def array_nbytes(self) -> int:
        """Actual in-memory numeric payload, excluding metadata/container overhead."""
        return sum(getattr(self, name).nbytes for name in self.ARRAY_FIELDS)

    def content_hash(self) -> str:
        self.validate()
        return _content_digest(self.RECORD_TYPE, self.metadata,
                               {name: getattr(self, name) for name in self.ARRAY_FIELDS})

    def save(self, path: str | Path) -> Path:
        """Atomically save an NPZ record, preserving a primary write failure.

        If failure cleanup also fails, retain the temporary file and attach the
        cleanup error to the primary exception for diagnosis.
        """
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {"schema_version": RECORD_SCHEMA_VERSION,
                    "record_type": self.RECORD_TYPE, "metadata": self.metadata,
                    "content_hash": self.content_hash()}
        arrays = {name: getattr(self, name) for name in self.ARRAY_FIELDS}
        arrays["manifest_json"] = np.frombuffer(_json_bytes(manifest), dtype=np.uint8)
        temporary = None
        primary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    np.savez(stream, **arrays)
                    stream.flush()
                    os.fsync(stream.fileno())
                except BaseException as exc:
                    primary = exc
                    raise
            os.replace(temporary, path)
        except BaseException as failure:
            if primary is None:
                primary = failure
            elif failure is not primary:
                try:
                    primary.add_note(f"Record stream cleanup failed: {failure!r}")
                except BaseException:
                    pass
            try:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            except BaseException as secondary:
                try:
                    primary.add_note(f"Record temporary cleanup failed: {secondary!r}")
                except BaseException:
                    pass
            raise primary from None
        else:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def load(cls, path: str | Path):
        return cls._load_npz(Path(path))

    @classmethod
    def load_bytes(cls, raw: bytes):
        """Decode exactly one immutable NPZ snapshot with the native validation."""
        if type(raw) is not bytes:
            raise ValueError("Record snapshot must be immutable bytes")
        return cls._load_npz(io.BytesIO(raw))

    @classmethod
    def _load_npz(cls, source):
        with np.load(source, allow_pickle=False) as archive:
            expected = {*cls.ARRAY_FIELDS, "manifest_json"}
            if len(archive.files) != len(expected) or set(archive.files) != expected:
                raise ValueError("Unexpected or missing record arrays")
            raw = archive["manifest_json"]
            _array("manifest_json", raw, np.uint8, (None,))
            try:
                manifest = json.loads(raw.tobytes().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Invalid record manifest") from exc
            if not isinstance(manifest, dict) or "metadata" not in manifest:
                raise ValueError("Invalid record manifest structure")
            if (type(manifest.get("schema_version")) is not int
                    or manifest.get("schema_version") != RECORD_SCHEMA_VERSION
                    or manifest.get("record_type") != cls.RECORD_TYPE):
                raise ValueError("Incompatible record schema or record type")
            arrays = {name: archive[name].copy() for name in cls.ARRAY_FIELDS}
            # Check stored bytes before construction can sanitize hidden values.
            if _content_digest(cls.RECORD_TYPE, manifest["metadata"], arrays) != manifest.get("content_hash"):
                raise ValueError("Record content checksum mismatch")
            record = cls(**arrays, metadata=manifest["metadata"])
        if record.content_hash() != manifest.get("content_hash"):
            raise ValueError("Record content checksum mismatch")
        return record


@dataclass
class PoseRecord(_Record):
    positions: np.ndarray
    observed: np.ndarray
    frame_valid: np.ndarray
    time_us: np.ndarray
    frame_index: np.ndarray
    metadata: dict
    ARRAY_FIELDS: ClassVar = ("positions", "observed", "frame_valid", "time_us", "frame_index")
    RECORD_TYPE: ClassVar = "pose"

    def __post_init__(self):
        _array("positions", self.positions, np.float32, (None, 23, 3))
        t = len(self.positions)
        _array("observed", self.observed, np.bool_, (t, 23))
        _array("frame_valid", self.frame_valid, np.bool_, (t,))
        # Hidden values are irrelevant. Sanitize before any arithmetic and hashing.
        hidden = ~(self.observed & self.frame_valid[:, None])
        self.positions = np.where(hidden[..., None], np.float32(0), self.positions)
        self.validate()

    def validate(self):
        _array("positions", self.positions, np.float32, (None, 23, 3))
        t = len(self.positions)
        _array("observed", self.observed, np.bool_, (t, 23))
        _array("frame_valid", self.frame_valid, np.bool_, (t,))
        _array("time_us", self.time_us, np.int64, (t,))
        _array("frame_index", self.frame_index, np.int64, (t,))
        _metadata(self.metadata, ("skeleton_schema", "preprocessing_id", "source_hash"))
        if self.metadata["skeleton_schema"] != SKELETON_SCHEMA:
            raise ValueError(f"Unsupported skeleton_schema; expected {SKELETON_SCHEMA}")
        if not np.isfinite(self.positions).all():
            raise ValueError("Observed pose coordinates must be finite")
        if np.any(self.observed & ~self.frame_valid[:, None]):
            raise ValueError("An invalid frame cannot contain observed joints")
        if np.any(self.positions[~self.observed] != 0):
            raise ValueError("Hidden pose coordinates must remain sanitized to zero")
        _increasing("time_us", self.time_us)
        _increasing("frame_index", self.frame_index)
        if t and self.frame_index[0] < 0:
            raise ValueError("frame_index must be nonnegative")
        if t > 1 and np.any(np.diff(self.frame_index) != 1):
            raise ValueError("Pose records require a dense grid; retain missing rows with masks")
        return self


class _WindowRecord(_Record):
    def _validate_windows(self, w):
        _array("core_index", self.core_index, np.uint32, (w,))
        for name in ("center_us", "core_start_us", "core_end_us"):
            _array(name, getattr(self, name), np.int64, (w,))
            _increasing(name, getattr(self, name))
        for name in ("core_coverage", "context_coverage"):
            a = getattr(self, name)
            _array(name, a, np.float32, (w,))
            if not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
                raise ValueError(f"{name} must be finite in [0,1]")
        _array("valid", self.valid, np.bool_, (w,))
        _increasing("core_index", self.core_index)
        if np.any(self.core_end_us <= self.core_start_us):
            raise ValueError("Core intervals must be nonempty and half-open")
        if np.any((self.center_us < self.core_start_us) | (self.center_us >= self.core_end_us)):
            raise ValueError("Center timestamps must lie inside their core")
        if np.any(self.valid & (self.core_coverage == 0)):
            raise ValueError("A valid window needs observed core evidence")


@dataclass
class DescriptorRecord(_WindowRecord):
    z: np.ndarray
    core_index: np.ndarray
    center_us: np.ndarray
    core_start_us: np.ndarray
    core_end_us: np.ndarray
    core_coverage: np.ndarray
    context_coverage: np.ndarray
    valid: np.ndarray
    metadata: dict
    ARRAY_FIELDS: ClassVar = ("z", "core_index", "center_us", "core_start_us",
                              "core_end_us", "core_coverage", "context_coverage", "valid")
    RECORD_TYPE: ClassVar = "descriptors"

    def __post_init__(self):
        self.validate()

    def validate(self):
        _array("z", self.z, np.float32, (None, None))
        if self.z.shape[1] < 1 or not np.isfinite(self.z).all():
            raise ValueError("Descriptors require a positive dimension and finite values")
        self._validate_windows(len(self.z))
        _metadata(self.metadata, ("encoder_id", "pose_record_hash"))
        if np.any(self.valid & (np.linalg.norm(self.z.astype(np.float64), axis=1) <= 1e-12)):
            raise ValueError("A valid cosine descriptor must have nonzero norm")
        return self


@dataclass
class CodeRecord(_WindowRecord):
    codes: np.ndarray
    core_index: np.ndarray
    center_us: np.ndarray
    core_start_us: np.ndarray
    core_end_us: np.ndarray
    core_coverage: np.ndarray
    context_coverage: np.ndarray
    valid: np.ndarray
    metadata: dict
    ARRAY_FIELDS: ClassVar = ("codes", "core_index", "center_us", "core_start_us",
                              "core_end_us", "core_coverage", "context_coverage", "valid")
    RECORD_TYPE: ClassVar = "codes"

    def __post_init__(self):
        self.validate()

    def validate(self):
        _array("codes", self.codes, np.uint16, (None, None))
        if self.codes.shape[1] < 1:
            raise ValueError("At least one codebook is required")
        self._validate_windows(len(self.codes))
        _metadata(self.metadata, ("tokenizer_id",))
        if "num_words" in self.metadata:
            k = self.metadata["num_words"]
            if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 65536:
                raise ValueError("num_words must fit uint16 code IDs")
            if np.any(self.codes >= k):
                raise ValueError("Code ID lies outside its vocabulary")
        return self
