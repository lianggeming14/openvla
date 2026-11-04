"""Predicate evidence and human annotations for recorded episodes.

Events describe observations, never causal or task-required ordering. ``initial``
marks the start of a known truth interval, including observations after a gap;
only consecutive known truth values can establish ``achieved`` or ``reverted``.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
from typing import Any


SOURCE = "libero_predicates"
_FIELDS = {"schema_version", "episode_id", "num_steps", "source", "events",
           "segments", "outcome", "notes", "revision", "object_attempts", "object_reviewed"}
_IMMUTABLE = ("schema_version", "episode_id", "num_steps", "source", "events")
_ATTEMPT_FRAMES = ("start_frame", "grasp_start_frame", "grasp_frame",
                   "place_start_frame", "place_frame", "end_frame")
_OPERATION_STARTS = {"grasp_start_frame", "place_start_frame"}
_ATTEMPT_FIELDS = set(_ATTEMPT_FRAMES) | {"attempt_id", "object", "subtask_success", "source", "notes"}


def _predicate_snapshot(context: dict, name: str) -> dict:
    snapshot = context.get(name)
    if snapshot is None:
        return {}
    if not isinstance(snapshot, dict):
        raise ValueError(f"{name} must be a predicate mapping")
    for predicate, value in snapshot.items():
        if not isinstance(predicate, str) or not predicate:
            raise ValueError("predicate names must be nonempty strings")
        if value is not None and type(value) is not bool:
            raise ValueError(f"predicate {predicate!r} must be boolean or null")
    return snapshot


def generate_annotations(episode_path: str | Path) -> dict[str, Any]:
    """Generate events from committed before/after predicate observations.

    A missing key or snapshot means unknown. Names discovered later in the
    episode are consequently unknown from the first frame. Initial truth does
    not count as an observed achievement, nor does truth after unknown evidence.
    """
    return _generate_annotations(episode_path)


def _generate_annotations(
    episode_path: str | Path, *, require_terminal: bool = False,
) -> dict[str, Any]:
    from .recorder import read_manifest, read_steps

    manifest = read_manifest(episode_path)
    if require_terminal and manifest["status"] == "recording":
        raise ValueError(
            "Cannot save annotations while episode is recording; "
            "a terminal status (complete, error, or interrupted) is required"
        )
    steps = read_steps(episode_path)
    if read_manifest(episode_path) != manifest or len(steps) != manifest["num_steps"]:
        raise ValueError("Episode changed while reading annotation evidence; retry with a stable snapshot")
    source = "synthetic_demo" if manifest["metadata"].get("synthetic") is True else SOURCE
    snapshots = []
    predicates: set[str] = set()
    for step in steps:
        context = step["context"]
        for name in ("predicates_before", "predicates_after"):
            snapshot = _predicate_snapshot(context, name)
            snapshots.append((step["index"], snapshot))
            predicates.update(snapshot)
    events = []
    previous: dict[str, bool | None] = {}
    for frame, snapshot in snapshots:
        for predicate in sorted(predicates):
            value = snapshot.get(predicate)
            if predicate in previous and previous[predicate] is value:
                continue
            old = previous.get(predicate)
            if value is None:
                kind = "unknown"
            elif old is None:
                kind = "initial"
            elif value:
                kind = "achieved"
            else:
                kind = "reverted"
            events.append({"frame": frame, "predicate": predicate, "value": value,
                           "kind": kind, "source": source})
            previous[predicate] = value
    return {"schema_version": 2, "episode_id": manifest["episode_id"],
            "num_steps": len(steps), "source": source, "events": events,
            "segments": [], "outcome": manifest["outcome"], "notes": "",
            "revision": 0, "object_attempts": [], "object_reviewed": False}


def _nonnegative_integer(value: Any, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _migrate(document: dict[str, Any]) -> dict[str, Any]:
    """Copy legacy documents in memory; never silently repair malformed v2."""
    if not isinstance(document, dict):
        raise ValueError("annotation document must be an object")
    document = copy.deepcopy(document)
    version = document.get("schema_version")
    if type(version) is not int or version not in (1, 2):
        raise ValueError("unsupported annotation schema_version")
    if version == 1:
        if {"object_attempts", "object_reviewed"} & document.keys():
            raise ValueError("v1 annotation document cannot contain v2 object fields")
        document.update(schema_version=2, object_attempts=[], object_reviewed=False)
    attempts = document.get("object_attempts")
    if isinstance(attempts, list):
        for attempt in attempts:
            if isinstance(attempt, dict):
                for field in _OPERATION_STARTS:
                    attempt.setdefault(field, None)
    return document


def _validate_document(document: dict[str, Any], evidence: dict[str, Any],
                       phases: list[str | None]) -> None:
    if not isinstance(document, dict):
        raise ValueError("annotation document must be an object")
    if not _FIELDS <= document.keys() or document.keys() - _FIELDS - {"base_revision"}:
        raise ValueError("annotation document has missing or unexpected fields")
    try:
        json.dumps(document, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("annotation document must contain finite JSON values") from exc
    for field in _IMMUTABLE:
        # JSON comparison distinguishes 1/true and prevents weakening the original
        # evidence through Python's permissive numeric equality.
        if json.dumps(document[field], sort_keys=True) != json.dumps(evidence[field], sort_keys=True):
            raise ValueError(f"immutable annotation field changed: {field}")
    _nonnegative_integer(document["revision"], "revision")
    if "base_revision" in document:
        _nonnegative_integer(document["base_revision"], "base_revision")
        if document["base_revision"] != document["revision"]:
            raise ValueError("base_revision differs from document revision")
    if document["outcome"] not in ("success", "failure", "unknown"):
        raise ValueError("outcome must be success, failure, or unknown")
    if not isinstance(document["notes"], str):
        raise ValueError("notes must be a string")
    if not isinstance(document["segments"], list):
        raise ValueError("segments must be a list")
    for segment in document["segments"]:
        if not isinstance(segment, dict) or set(segment) != {"start", "end", "label", "notes"}:
            raise ValueError("segment must contain start, end, label, and notes")
        for field in ("start", "end"):
            _nonnegative_integer(segment[field], f"segment {field}")
        if not segment["start"] <= segment["end"] < evidence["num_steps"]:
            raise ValueError("segment inclusive range is outside the episode")
        if not isinstance(segment["label"], str) or not segment["label"].strip():
            raise ValueError("segment label must be a nonempty string")
        if not isinstance(segment["notes"], str):
            raise ValueError("segment notes must be a string")
    if type(document["object_reviewed"]) is not bool:
        raise ValueError("object_reviewed must be boolean")
    if not isinstance(document["object_attempts"], list):
        raise ValueError("object_attempts must be a list")
    ids = set()
    for attempt in document["object_attempts"]:
        if not isinstance(attempt, dict) or set(attempt) != _ATTEMPT_FIELDS:
            raise ValueError("object attempt has missing or unexpected fields")
        for field in ("attempt_id", "source"):
            if not isinstance(attempt[field], str) or not attempt[field].strip():
                raise ValueError(f"attempt {field} must be a nonempty string")
        if attempt["attempt_id"] in ids:
            raise ValueError("attempt_id must be unique within the episode")
        ids.add(attempt["attempt_id"])
        if attempt["object"] is not None and (not isinstance(attempt["object"], str)
                                                or not attempt["object"].strip()):
            raise ValueError("attempt object must be a nonempty string or null")
        if attempt["subtask_success"] is not None and type(attempt["subtask_success"]) is not bool:
            raise ValueError("attempt subtask_success must be boolean or null")
        if not isinstance(attempt["notes"], str):
            raise ValueError("attempt notes must be a string")
        previous = None
        for field in _ATTEMPT_FRAMES:
            frame = attempt[field]
            if frame is None:
                continue
            _nonnegative_integer(frame, f"attempt {field}")
            if frame >= evidence["num_steps"]:
                raise ValueError("attempt frame is outside the episode")
            if previous is not None and frame < previous:
                raise ValueError("attempt frame boundaries must be in chronological order")
            previous = frame
            if field not in ("start_frame", "end_frame") and phases[frame] == "settling":
                raise ValueError("attempt grasp/place events cannot reference settling frames")


def _phases(path: Path, evidence: dict[str, Any]) -> list[str | None]:
    from .recorder import read_steps
    steps = read_steps(path)
    if len(steps) != evidence["num_steps"]:
        raise ValueError("Episode changed while reading annotation evidence")
    return [step.get("context", {}).get("phase") for step in steps]


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number: {value}")


def _load_existing(path: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    try:
        with (path / "annotations.json").open(encoding="utf-8") as stream:
            document = json.load(stream, parse_constant=_reject_constant)
    except FileNotFoundError:
        return copy.deepcopy(evidence)
    document = _migrate(document)
    _validate_document(document, evidence, _phases(path, evidence))
    if "base_revision" in document:
        raise ValueError("saved annotations must not contain base_revision")
    return document


def load_annotations(episode_path: str | Path) -> dict[str, Any]:
    """Load validated annotations, or generate them in memory without writing."""
    path = Path(episode_path)
    return _load_existing(path, generate_annotations(path))


def save_annotations(
    episode_path: str | Path,
    document: dict[str, Any],
    expected_revision: int | None = None,
) -> dict[str, Any]:
    """Atomically save human fields and return the incremented document.

    Only terminal episodes can be saved; recording episodes may be previewed.
    Only segments, object attempts/review status, outcome and notes are editable. Automatic evidence is checked
    against committed steps and copied from that trusted source. A process-wide
    exclusive file lock serializes revision checks and replacement. An abandoned
    lock is never reclaimed automatically: after a crashed writer, verify that no
    writer remains before manually removing ``.annotations.lock``.
    """
    path = Path(episode_path)
    lock_path = path / ".annotations.lock"
    try:
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"annotation lock is held: {lock_path}") from exc
    temporary_path = None
    try:
        with os.fdopen(lock_fd, "w", encoding="ascii") as stream:
            stream.write(str(os.getpid()))
        evidence = _generate_annotations(path, require_terminal=True)
        current = _load_existing(path, evidence)
        legacy_import = isinstance(document, dict) and document.get("schema_version") == 1
        document = _migrate(document)
        if legacy_import:
            for field in ("object_attempts", "object_reviewed"):
                document[field] = copy.deepcopy(current[field])
        _validate_document(document, evidence, _phases(path, evidence))
        supplied_revision = document.get("base_revision", document["revision"])
        if expected_revision is None:
            expected_revision = supplied_revision
        _nonnegative_integer(expected_revision, "expected_revision")
        if expected_revision != supplied_revision or expected_revision != current["revision"]:
            raise ValueError(
                f"annotation revision conflict: expected {expected_revision}, "
                f"current {current['revision']}, document {supplied_revision}"
            )
        saved = copy.deepcopy(evidence)
        for field in ("segments", "outcome", "notes", "object_attempts", "object_reviewed"):
            saved[field] = copy.deepcopy(document[field])
        saved["revision"] = current["revision"] + 1
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".annotations-", suffix=".tmp",
            dir=path, delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(saved, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path / "annotations.json")
        temporary_path = None
        return saved
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        lock_path.unlink()


def summarize_annotations(document: dict[str, Any]) -> dict[str, Any]:
    """Report first observed truth and transitions without causal claims.

    Equal-frame entries are simultaneous at the available frame resolution;
    their listing order does not establish an ordering within that frame.
    """
    first_true = []
    first_true_frames = {}
    achieved = []
    achievement_frames = {}
    final_state = {}
    for event in document["events"]:
        predicate = event["predicate"]
        final_state[predicate] = event["value"]
        if event["value"] is True and predicate not in first_true:
            first_true.append(predicate)
            first_true_frames[predicate] = event["frame"]
        if event["kind"] == "achieved" and predicate not in achieved:
            achieved.append(predicate)
            achievement_frames[predicate] = event["frame"]
    return {"episode_id": document["episode_id"], "num_steps": document["num_steps"],
            "source": document["source"], "outcome": document["outcome"],
            "first_observed_true_order": first_true,
            "first_observed_true_frames": first_true_frames,
            "observed_achievement_order": achieved,
            "observed_achievement_frames": achievement_frames,
            "order_note": "Observed frame order only; equal-frame entries are unordered.",
            "final_state": final_state}
