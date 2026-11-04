"""Evidence, revision, and validation tests for trajectory annotations."""

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


def make_episode(tmp_path, snapshots, outcome="unknown"):
    """Write a valid committed episode without depending on recorder helpers."""
    episode = tmp_path / "episode-one"
    (episode / "steps").mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "episode_id": "episode-one",
        "metadata": {"task": "test fixture"},
        "status": "complete",
        "num_steps": len(snapshots),
        "outcome": outcome,
    }
    (episode / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for index, (before, after) in enumerate(snapshots):
        context = {"phase": "policy"}
        if before is not ...:
            context["predicates_before"] = before
        if after is not ...:
            context["predicates_after"] = after
        step = {
            "index": index,
            "timestamp_ns": 1000 + index,
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "context": context,
            "array_file": f"{index:06d}.npz",
        }
        (episode / "steps" / f"{index:06d}.json").write_text(
            json.dumps(step), encoding="utf-8"
        )
        np.savez_compressed(
            episode / "steps" / f"{index:06d}.npz",
            **{"action": np.zeros(7), "obs/rgb": np.zeros((2, 2, 3), dtype=np.uint8),
               "next_obs/rgb": np.zeros((2, 2, 3), dtype=np.uint8)},
        )
    return episode


def api():
    return importlib.import_module("vla_trajectory.annotations")


def compact_events(document):
    return [(event["frame"], event["predicate"], event["value"], event["kind"])
            for event in document["events"]]


def test_initial_truth_achievement_and_reversal_use_both_snapshots(tmp_path):
    episode = make_episode(tmp_path, [
        ({"a": True, "b": False}, {"a": False, "b": True}),
        ({"a": False, "b": True}, {"a": True, "b": True}),
    ])
    document = api().generate_annotations(episode)
    assert compact_events(document) == [
        (0, "a", True, "initial"), (0, "b", False, "initial"),
        (0, "a", False, "reverted"), (0, "b", True, "achieved"),
        (1, "a", True, "achieved"),
    ]
    assert all(event["source"] == "libero_predicates" for event in document["events"])
    assert document["revision"] == 0
    assert document["outcome"] == "unknown"
    assert document["segments"] == []
    assert not (episode / "annotations.json").exists()


def test_missing_and_null_predicates_break_transition_evidence(tmp_path):
    episode = make_episode(tmp_path, [
        ({"goal": False}, {}),
        (..., {"goal": True}),
        ({"goal": True}, {"goal": None}),
        ({"goal": False}, {"goal": True}),
    ])
    assert compact_events(api().generate_annotations(episode)) == [
        (0, "goal", False, "initial"), (0, "goal", None, "unknown"),
        (1, "goal", True, "initial"), (2, "goal", None, "unknown"),
        (3, "goal", False, "initial"), (3, "goal", True, "achieved"),
    ]


def test_late_predicate_is_unknown_at_episode_start(tmp_path):
    episode = make_episode(tmp_path, [({}, {}), ({}, {"late": True})])
    assert compact_events(api().generate_annotations(episode)) == [
        (0, "late", None, "unknown"), (1, "late", True, "initial"),
    ]


@pytest.mark.parametrize("invalid", [{"goal": 1}, {"goal": "yes"}, [True]])
def test_invalid_predicate_evidence_is_rejected(tmp_path, invalid):
    episode = make_episode(tmp_path, [(invalid, {})])
    with pytest.raises(ValueError, match="predicate"):
        api().generate_annotations(episode)


def test_empty_episode_retains_explicit_outcome_and_has_no_events(tmp_path):
    episode = make_episode(tmp_path, [], outcome="failure")
    document = api().generate_annotations(episode)
    assert document["events"] == []
    assert document["num_steps"] == 0
    assert document["outcome"] == "failure"


def test_synthetic_demo_preserves_distinct_evidence_provenance(tmp_path):
    episode = make_episode(tmp_path, [({"goal": False}, {"goal": True})])
    manifest_path = episode / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["synthetic"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    document = api().generate_annotations(episode)
    assert document["source"] == "synthetic_demo"
    assert all(event["source"] == "synthetic_demo" for event in document["events"])
    saved = api().save_annotations(episode, document)
    assert api().load_annotations(episode) == saved


def test_summary_reports_observations_without_implying_causal_order(tmp_path):
    episode = make_episode(tmp_path, [
        ({"already": True, "a": False, "b": False},
         {"already": True, "a": False, "b": True}),
        ({"already": True, "a": False, "b": True},
         {"already": None, "a": True, "b": False}),
    ])
    summary = api().summarize_annotations(api().generate_annotations(episode))
    assert summary["first_observed_true_order"] == ["already", "b", "a"]
    assert summary["observed_achievement_order"] == ["b", "a"]
    assert summary["final_state"] == {"already": None, "a": True, "b": False}
    assert "causal" not in summary
    json.dumps(summary, allow_nan=False)


def test_summary_keeps_provenance_and_same_frame_ties_visible(tmp_path):
    episode = make_episode(tmp_path, [
        ({"a": False, "b": False}, {"a": True, "b": True}),
    ])
    summary = api().summarize_annotations(api().generate_annotations(episode))
    assert summary["source"] == "libero_predicates"
    assert summary["first_observed_true_frames"] == {"a": 0, "b": 0}
    assert summary["observed_achievement_frames"] == {"a": 0, "b": 0}


def blank_document(num_steps=2):
    return {"schema_version": 1, "episode_id": "episode-one", "num_steps": num_steps,
            "source": "libero_predicates", "events": [], "segments": [],
            "outcome": "unknown", "notes": "", "revision": 0}


def test_save_manual_fields_with_inclusive_ranges_and_reload(tmp_path):
    episode = make_episode(tmp_path, [({}, {}), ({}, {})])
    original_files = {path: path.read_bytes() for path in episode.rglob("*") if path.is_file()}
    document = blank_document()
    document.update(segments=[{"start": 0, "end": 1, "label": "grasp", "notes": "复核"}],
                    outcome="failure", notes="Human reviewed")
    saved = api().save_annotations(episode, document, expected_revision=0)
    assert saved["revision"] == 1
    assert saved["segments"] == [{"start": 0, "end": 1, "label": "grasp", "notes": "复核"}]
    assert saved["outcome"] == "failure"
    assert saved["notes"] == "Human reviewed"
    assert document["revision"] == 0
    assert api().load_annotations(episode) == saved
    assert all(path.read_bytes() == content for path, content in original_files.items())
    assert not (episode / ".annotations.lock").exists()


def test_loading_without_saved_document_is_read_only(tmp_path):
    episode = make_episode(tmp_path, [({}, {})])
    assert api().load_annotations(episode) == blank_document(1)
    assert not (episode / "annotations.json").exists()


@pytest.mark.parametrize("status", ["complete", "error", "interrupted"])
def test_recording_preview_cannot_be_saved_until_episode_is_terminal(tmp_path, status):
    from vla_trajectory.recorder import EpisodeRecorder

    recorder = EpisodeRecorder(tmp_path, {"task": "still running"}, episode_id="live")
    recorder.append({}, [0.0], {}, context={
        "predicates_before": {"goal": False}, "predicates_after": {"goal": False},
    })
    preview = api().load_annotations(recorder.path)
    assert preview["num_steps"] == 1
    with pytest.raises(ValueError, match="recording.*terminal|terminal.*recording"):
        api().save_annotations(recorder.path, preview)
    assert not (recorder.path / "annotations.json").exists()
    assert not (recorder.path / ".annotations.lock").exists()

    recorder.append({}, [0.0], {}, context={
        "predicates_before": {"goal": False}, "predicates_after": {"goal": True},
    })
    recorder.finish(status=status)
    complete = api().load_annotations(recorder.path)
    complete["notes"] = "Reviewed after finalization"
    saved = api().save_annotations(recorder.path, complete)
    assert saved["num_steps"] == 2
    assert saved["revision"] == 1
    assert api().load_annotations(recorder.path) == saved


def test_generation_rejects_manifest_change_during_evidence_read(tmp_path, monkeypatch):
    from vla_trajectory import recorder as storage

    recorder = storage.EpisodeRecorder(tmp_path, {}, episode_id="live")
    recorder.append({}, [0.0], {})
    original_read = storage.read_steps

    def append_after_read(path):
        steps = original_read(path)
        recorder.append({}, [1.0], {})
        return steps

    monkeypatch.setattr(storage, "read_steps", append_after_read)
    with pytest.raises(ValueError, match="changed|snapshot"):
        api().generate_annotations(recorder.path)


@pytest.mark.parametrize("segment", [
    {"start": -1, "end": 1, "label": "x", "notes": ""},
    {"start": 0, "end": 2, "label": "x", "notes": ""},
    {"start": 1, "end": 0, "label": "x", "notes": ""},
    {"start": False, "end": 1, "label": "x", "notes": ""},
    {"start": 0, "end": 1, "label": "", "notes": ""},
    {"start": 0, "end": 1, "label": "x", "notes": []},
    {"start": 0, "end": 1, "label": "x", "notes": "", "events": []},
])
def test_invalid_segment_ranges_or_strings_are_rejected_without_write(tmp_path, segment):
    episode = make_episode(tmp_path, [({}, {}), ({}, {})])
    document = blank_document()
    document["segments"] = [segment]
    with pytest.raises(ValueError, match="segment"):
        api().save_annotations(episode, document)
    assert not (episode / "annotations.json").exists()


@pytest.mark.parametrize("field,value", [
    ("episode_id", "different-episode"), ("num_steps", 3), ("schema_version", 2),
    ("source", "human"), ("events", [{"frame": 0, "predicate": "forged"}]),
    ("notes", []), ("outcome", "done"), ("revision", True),
    ("revision", -1), ("extra", "unexpected"),
])
def test_invalid_import_or_immutable_evidence_change_is_rejected(tmp_path, field, value):
    episode = make_episode(tmp_path, [({}, {}), ({}, {})])
    document = blank_document()
    document[field] = value
    with pytest.raises(ValueError):
        api().save_annotations(episode, document)
    assert not (episode / "annotations.json").exists()


def test_import_preserves_events_and_provenance(tmp_path):
    episode = make_episode(tmp_path, [({"goal": False}, {"goal": True})])
    document = api().generate_annotations(episode)
    original_events = copy.deepcopy(document["events"])
    document["notes"] = "An observed achievement"
    saved = api().save_annotations(episode, document)
    assert saved["events"] == original_events
    assert saved["source"] == "libero_predicates"
    changed = copy.deepcopy(saved)
    changed["events"][0]["source"] = "human"
    with pytest.raises(ValueError, match="immutable"):
        api().save_annotations(episode, changed)
    assert api().load_annotations(episode) == saved


def test_stale_revision_cannot_overwrite_newer_annotations(tmp_path):
    episode = make_episode(tmp_path, [({}, {}), ({}, {})])
    first = api().save_annotations(episode, blank_document())
    stale = blank_document()
    stale["notes"] = "stale edit"
    with pytest.raises(ValueError, match="revision"):
        api().save_annotations(episode, stale)
    with pytest.raises(ValueError, match="revision"):
        api().save_annotations(episode, stale, expected_revision=0)
    first["base_revision"] = 1
    first["notes"] = "new edit"
    second = api().save_annotations(episode, first, expected_revision=1)
    assert second["revision"] == 2
    assert second["notes"] == "new edit"
    assert "base_revision" not in second
    assert api().load_annotations(episode) == second


def test_conflicting_base_revision_is_rejected(tmp_path):
    episode = make_episode(tmp_path, [({}, {}), ({}, {})])
    document = blank_document()
    document["base_revision"] = 5
    with pytest.raises(ValueError, match="revision"):
        api().save_annotations(episode, document, expected_revision=0)


def test_existing_process_lock_blocks_writes_without_removing_lock(tmp_path):
    episode = make_episode(tmp_path, [({}, {}), ({}, {})])
    lock = episode / ".annotations.lock"
    lock.write_text("another writer", encoding="utf-8")
    with pytest.raises(RuntimeError, match="lock"):
        api().save_annotations(episode, blank_document())
    assert lock.read_text(encoding="utf-8") == "another writer"
    assert not (episode / "annotations.json").exists()


def test_atomic_replace_failure_preserves_previous_document_and_cleans_lock(tmp_path, monkeypatch):
    episode = make_episode(tmp_path, [({}, {}), ({}, {})])
    saved = api().save_annotations(episode, blank_document())
    previous_bytes = (episode / "annotations.json").read_bytes()
    saved["notes"] = "must not be partially written"

    def fail_replace(source, destination):
        raise OSError("simulated disk replace failure")

    monkeypatch.setattr(api().os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failure"):
        api().save_annotations(episode, saved)
    assert (episode / "annotations.json").read_bytes() == previous_bytes
    assert not (episode / ".annotations.lock").exists()
    assert not list(episode.glob(".annotations-*.tmp"))


def test_load_rejects_corrupt_or_tampered_saved_document(tmp_path):
    episode = make_episode(tmp_path, [({}, {}), ({}, {})])
    annotation_file = episode / "annotations.json"
    annotation_file.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError):
        api().load_annotations(episode)
    tampered = blank_document()
    tampered["source"] = "forged"
    annotation_file.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError):
        api().load_annotations(episode)
