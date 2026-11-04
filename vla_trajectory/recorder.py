"""Atomic, pickle-free episode storage for CPU observations and actions.

The manifest count is the commit boundary. A step becomes visible only after
both files have been replaced successfully and the new manifest is committed.
Files after that count may be left by interruption and are deliberately ignored.
Each recorder owns a new episode; existing episodes are never reopened to write.
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, Mapping
import uuid
import zipfile

import numpy as np


_STATUSES = {"recording", "complete", "error", "interrupted"}
_OUTCOMES = {"success", "failure", "unknown"}
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}


def _safe_name(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not _NAME.fullmatch(value)
        or value.endswith(".")
        or value.split(".")[0].upper() in _WINDOWS_RESERVED
    ):
        raise ValueError(f"{label} must be a safe nonempty name of at most 128 characters")
    return value


def _json_copy(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    try:
        # Round-tripping both detaches nested data and rejects NumPy/device values.
        return json.loads(json.dumps(dict(value), ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain finite JSON-serializable values: {exc}") from exc


def _numeric_array(value: Any, label: str, *, action: bool = False) -> np.ndarray:
    try:
        array = np.array(value, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric array") from exc
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{label} must be a real numeric array without objects or strings")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite numeric values")
    if action and array.size == 0:
        raise ValueError("action must not be empty")
    return array


def _add_arrays(target: dict[str, np.ndarray], prefix: str, values: Any) -> None:
    if not isinstance(values, Mapping):
        raise TypeError(f"{prefix} must be a mapping of names to arrays")
    for key, value in values.items():
        name = _safe_name(key, f"{prefix} array key")
        target[f"{prefix}/{name}"] = _numeric_array(value, f"{prefix}/{name}")


def _reward(value: Any) -> float:
    if isinstance(value, (bool, np.bool_, str, bytes)) or not np.isscalar(value) or np.iscomplexobj(value):
        raise ValueError("reward must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("reward must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError("reward must be a finite real number")
    return result


def _atomic_write(path: Path, write: Any) -> None:
    """Flush a same-directory temporary file before atomically replacing a file."""
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = stream.name
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    encoded = (json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
    _atomic_write(path, lambda stream: stream.write(encoded))


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream, parse_constant=reject_constant)
    if not isinstance(document, dict):
        raise ValueError("expected a JSON object")
    # Also rejects overflowed JSON numbers such as 1e999 in nested data.
    return _json_copy(document, "document")


class EpisodeRecorder:
    """Write one new episode, preserving raw transitions and optional extra arrays.

    Calls on one instance are serialized. There is no resume API or competing
    writer: the episode directory must not already exist. ``manifest`` returns a
    detached snapshot so callers cannot alter the writer's commit count.
    """

    def __init__(self, root: str | os.PathLike[str], metadata: Mapping[str, Any], episode_id: str | None = None) -> None:
        copied_metadata = _json_copy(metadata, "metadata")
        identifier = _safe_name(uuid.uuid4().hex if episode_id is None else episode_id, "episode_id")
        self.path = Path(root) / identifier
        self._lock = threading.RLock()
        self._manifest: dict[str, Any] = {
            "schema_version": 1,
            "episode_id": identifier,
            "metadata": copied_metadata,
            "status": "recording",
            "num_steps": 0,
            "outcome": "unknown",
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.mkdir(exist_ok=False)
        (self.path / "steps").mkdir()
        _write_json(self.path / "manifest.json", self._manifest)

    @property
    def manifest(self) -> dict[str, Any]:
        with self._lock:
            self._manifest = read_manifest(self.path)
            return copy.deepcopy(self._manifest)

    def append(
        self,
        observation: Mapping[str, Any],
        action: Any,
        next_observation: Mapping[str, Any],
        *,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        context: Mapping[str, Any] | None = None,
        arrays: Mapping[str, Any] | None = None,
    ) -> int:
        """Commit one transition and return its zero-based step index.

        Values are copied before writing. The disk manifest is authoritative,
        including if an interruption arrives just after its atomic replacement.
        Retrying may replace only a step that has not committed on disk.
        """
        with self._lock:
            # A prior atomic replace may have committed before Python received
            # an interrupt, leaving the cached manifest behind the disk state.
            self._manifest = read_manifest(self.path)
            if self._manifest["status"] != "recording":
                raise RuntimeError("Cannot append to a finished episode")
            if not isinstance(terminated, (bool, np.bool_)) or not isinstance(truncated, (bool, np.bool_)):
                raise TypeError("terminated and truncated must be booleans")
            payload = {"action": _numeric_array(action, "action", action=True)}
            _add_arrays(payload, "obs", observation)
            _add_arrays(payload, "next_obs", next_observation)
            if arrays is not None:
                _add_arrays(payload, "extra", arrays)
            index = self._manifest["num_steps"]
            record = {
                "index": index,
                "timestamp_ns": time.time_ns(),
                "reward": _reward(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "context": _json_copy({} if context is None else context, "context"),
                "array_file": f"{index:06d}.npz",
            }
            _atomic_write(self.path / "steps" / record["array_file"], lambda stream: np.savez_compressed(stream, **payload))
            _write_json(self.path / "steps" / f"{index:06d}.json", record)
            manifest = dict(self._manifest, num_steps=index + 1)
            _write_json(self.path / "manifest.json", manifest)
            self._manifest = manifest
            return index

    def finish(self, outcome: str = "unknown", status: str = "complete", error: str | None = None) -> None:
        """Commit an explicit terminal status; identical repeated calls are safe."""
        if status not in _STATUSES - {"recording"}:
            raise ValueError("finish status must be complete, error, or interrupted")
        if outcome not in _OUTCOMES:
            raise ValueError("outcome must be success, failure, or unknown")
        if error is not None and not isinstance(error, str):
            raise TypeError("error must be a string or None")
        with self._lock:
            self._manifest = read_manifest(self.path)
            manifest = dict(self._manifest, status=status, outcome=outcome)
            manifest.pop("error", None)
            if error is not None:
                manifest["error"] = error
            if self._manifest["status"] != "recording":
                if manifest == self._manifest:
                    return
                raise RuntimeError("Episode already finished with a different result")
            _write_json(self.path / "manifest.json", manifest)
            self._manifest = manifest


def read_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and validate an episode manifest without inferring completion."""
    manifest_path = Path(path) / "manifest.json"
    try:
        manifest = _read_json(manifest_path)
        if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
            raise ValueError("unsupported schema_version")
        _safe_name(manifest.get("episode_id"), "episode_id")
        if not isinstance(manifest.get("metadata"), dict):
            raise ValueError("metadata must be an object")
        if type(manifest.get("num_steps")) is not int or manifest["num_steps"] < 0:
            raise ValueError("num_steps must be a nonnegative integer")
        if manifest.get("status") not in _STATUSES:
            raise ValueError("invalid status")
        if manifest.get("outcome") not in _OUTCOMES:
            raise ValueError("invalid outcome")
        if "error" in manifest and not isinstance(manifest["error"], str):
            raise ValueError("error must be a string")
        return manifest
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Cannot read manifest at {manifest_path}: {exc}") from exc


def _read_step(path: Path, index: int) -> dict[str, Any]:
    try:
        step = _read_json(path / "steps" / f"{index:06d}.json")
        if type(step.get("index")) is not int or step["index"] != index:
            raise ValueError("index does not match committed position")
        if type(step.get("timestamp_ns")) is not int or step["timestamp_ns"] < 0:
            raise ValueError("timestamp_ns must be a nonnegative integer")
        _reward(step.get("reward"))
        if type(step.get("terminated")) is not bool or type(step.get("truncated")) is not bool:
            raise ValueError("terminated and truncated must be booleans")
        if not isinstance(step.get("context"), dict):
            raise ValueError("context must be a JSON object")
        if step.get("array_file") != f"{index:06d}.npz":
            raise ValueError("array_file must match its committed step")
        if not (path / "steps" / step["array_file"]).is_file():
            raise ValueError("committed array file is missing")
        return step
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Cannot read committed step {index} in {path}: {exc}") from exc


def read_steps(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read committed JSON records; arrays are validated by ``load_arrays``.

    Orphan files beyond the manifest count are ignored. Missing files inside the
    committed range are corruption, even when the episode is still recording.
    """
    episode = Path(path)
    manifest = read_manifest(episode)
    return [_read_step(episode, index) for index in range(manifest["num_steps"])]


def load_arrays(path: str | os.PathLike[str], index: int) -> dict[str, np.ndarray]:
    """Load detached numeric arrays from one committed step, with pickle disabled."""
    episode = Path(path)
    manifest = read_manifest(episode)
    if type(index) is not int:
        raise TypeError("step index must be an integer")
    if index < 0 or index >= manifest["num_steps"]:
        raise IndexError(f"Step {index} is outside the committed range [0, {manifest['num_steps']})")
    step = _read_step(episode, index)
    try:
        with np.load(episode / "steps" / step["array_file"], allow_pickle=False) as archive:
            if not isinstance(archive, np.lib.npyio.NpzFile):
                raise ValueError("expected an NPZ archive")
            if "action" not in archive.files or len(set(archive.files)) != len(archive.files):
                raise ValueError("archive must contain one action and unique array keys")
            arrays = {}
            for key in archive.files:
                if key != "action":
                    prefix, separator, name = key.partition("/")
                    if not separator or prefix not in {"obs", "next_obs", "extra"}:
                        raise ValueError(f"invalid array key {key!r}")
                    _safe_name(name, "array key")
                arrays[key] = _numeric_array(archive[key], key, action=key == "action")
            return arrays
    except (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Cannot load arrays for committed step {index} in {episode}: {exc}") from exc
