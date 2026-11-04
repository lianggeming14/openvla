"""Storage contract tests using real episode files and deliberate corruption."""

import importlib
import json

import numpy as np
import pytest


@pytest.fixture
def api():
    try:
        return importlib.import_module("vla_trajectory.recorder")
    except ModuleNotFoundError:
        pytest.fail("The trajectory recorder API has not been implemented")


def append_sample(recorder, **kwargs):
    return recorder.append(
        {"camera_image": np.full((2, 3, 3), 10, dtype=np.uint8)},
        np.array([0.25, -0.5, 1.0], dtype=np.float32),
        {"camera_image": np.full((2, 3, 3), 20, dtype=np.uint8)},
        **kwargs,
    )


def test_round_trip_preserves_transition_arrays_and_json(api, tmp_path):
    recorder = api.EpisodeRecorder(tmp_path, {"seed": 7}, episode_id="episode-1")
    assert append_sample(
        recorder,
        reward=1.5,
        terminated=True,
        context={"phase": "policy", "chunk_id": 2, "predicates_after": {"open": True}},
        arrays={"state": np.array([3.0, 4.0])},
    ) == 0
    assert append_sample(recorder) == 1
    recorder.finish(outcome="success")

    manifest = api.read_manifest(recorder.path)
    assert manifest == {
        "schema_version": 1,
        "episode_id": "episode-1",
        "metadata": {"seed": 7},
        "status": "complete",
        "num_steps": 2,
        "outcome": "success",
    }
    steps = api.read_steps(recorder.path)
    assert [step["index"] for step in steps] == [0, 1]
    assert steps[0]["reward"] == 1.5
    assert steps[0]["terminated"] is True
    assert steps[0]["truncated"] is False
    assert steps[0]["context"]["predicates_after"] == {"open": True}
    assert steps[0]["timestamp_ns"] > 0
    assert steps[0]["array_file"] == "000000.npz"
    arrays = api.load_arrays(recorder.path, 0)
    assert set(arrays) == {"obs/camera_image", "next_obs/camera_image", "action", "extra/state"}
    np.testing.assert_array_equal(arrays["obs/camera_image"], np.full((2, 3, 3), 10))
    np.testing.assert_array_equal(arrays["next_obs/camera_image"], np.full((2, 3, 3), 20))
    np.testing.assert_array_equal(arrays["action"], [0.25, -0.5, 1.0])
    np.testing.assert_array_equal(arrays["extra/state"], [3.0, 4.0])


def test_copies_metadata_context_arrays_without_mutating_inputs(api, tmp_path):
    metadata = {"configuration": {"seed": 3}}
    recorder = api.EpisodeRecorder(tmp_path, metadata)
    metadata["configuration"]["seed"] = 90
    context = {"chunk": {"offset": 1}}
    observation = np.arange(6, dtype=np.float32).reshape(2, 3)
    action = np.array([0.2, 0.4])
    before = observation.copy()
    recorder.append({"state": observation}, action, {"state": observation + 1}, context=context)
    np.testing.assert_array_equal(observation, before)
    np.testing.assert_array_equal(action, [0.2, 0.4])
    context["chunk"]["offset"] = 90
    observation[:] = 99
    action[:] = 99
    exposed = recorder.manifest
    exposed["metadata"]["configuration"]["seed"] = 91
    exposed["num_steps"] = 100
    assert recorder.manifest["metadata"]["configuration"]["seed"] == 3
    assert api.read_manifest(recorder.path)["num_steps"] == 1
    assert api.read_steps(recorder.path)[0]["context"] == {"chunk": {"offset": 1}}
    np.testing.assert_array_equal(api.load_arrays(recorder.path, 0)["obs/state"], before)
    np.testing.assert_array_equal(api.load_arrays(recorder.path, 0)["action"], [0.2, 0.4])


@pytest.mark.parametrize("metadata", [{"bad": object()}, {"bad": float("nan")}, ["not-a-mapping"]])
def test_rejects_invalid_metadata_before_creating_episode(api, tmp_path, metadata):
    with pytest.raises((TypeError, ValueError)):
        api.EpisodeRecorder(tmp_path, metadata, episode_id="invalid")
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("episode_id", ["../outside", "a/b", "a\\b", "", ".", "..", "CON", "NUL.txt", "trailing."])
def test_rejects_unsafe_episode_names(api, tmp_path, episode_id):
    with pytest.raises(ValueError):
        api.EpisodeRecorder(tmp_path, {}, episode_id=episode_id)


def test_duplicate_episode_never_overwrites_original(api, tmp_path):
    recorder = api.EpisodeRecorder(tmp_path, {"original": True}, episode_id="same")
    append_sample(recorder)
    with pytest.raises(FileExistsError):
        api.EpisodeRecorder(tmp_path, {"original": False}, episode_id="same")
    assert api.read_manifest(recorder.path)["metadata"] == {"original": True}
    assert len(api.read_steps(recorder.path)) == 1


@pytest.mark.parametrize("status", ["complete", "error", "interrupted"])
def test_empty_episode_finish_is_idempotent_only_for_same_result(api, tmp_path, status):
    recorder = api.EpisodeRecorder(tmp_path, {})
    assert api.read_manifest(recorder.path)["status"] == "recording"
    assert api.read_steps(recorder.path) == []
    recorder.finish(status=status, outcome="unknown", error="stopped")
    recorder.finish(status=status, outcome="unknown", error="stopped")
    assert api.read_manifest(recorder.path)["status"] == status
    assert api.read_manifest(recorder.path)["error"] == "stopped"
    with pytest.raises((RuntimeError, ValueError)):
        recorder.finish(status=status, outcome="failure", error="stopped")
    with pytest.raises(RuntimeError):
        append_sample(recorder)


@pytest.mark.parametrize("changes", [
    {"action": [float("nan")]},
    {"action": [float("inf")]},
    {"action": []},
    {"action": ["0.4"]},
    {"observation": {"state": np.array([object()], dtype=object)}},
    {"next_observation": {"state": [float("inf")]}},
    {"arrays": {"state": np.array([object()], dtype=object)}},
    {"arrays": {"../../escape": [1.0]}},
    {"context": {"bad": object()}},
    {"context": {"bad": float("nan")}},
    {"reward": float("inf")},
    {"reward": np.complex64(1 + 2j)},
    {"terminated": "yes"},
])
def test_invalid_append_does_not_commit_a_step(api, tmp_path, changes):
    recorder = api.EpisodeRecorder(tmp_path, {})
    arguments = {"observation": {}, "action": [1.0], "next_observation": {}}
    arguments.update(changes)
    with pytest.raises((TypeError, ValueError)):
        recorder.append(**arguments)
    assert api.read_manifest(recorder.path)["num_steps"] == 0
    assert api.read_steps(recorder.path) == []


@pytest.mark.parametrize("failed_file", ["000001.npz", "000001.json", "manifest.json"])
def test_commit_failure_keeps_previous_steps_readable_and_can_retry(api, tmp_path, monkeypatch, failed_file):
    recorder = api.EpisodeRecorder(tmp_path, {})
    append_sample(recorder)
    replace = api.os.replace

    def fail_manifest(source, destination):
        if str(destination).endswith(failed_file):
            raise OSError("simulated full disk during step commit")
        return replace(source, destination)

    monkeypatch.setattr(api.os, "replace", fail_manifest)
    with pytest.raises(OSError, match="simulated full disk"):
        append_sample(recorder)
    assert recorder.manifest["num_steps"] == 1
    assert len(api.read_steps(recorder.path)) == 1
    with pytest.raises(IndexError):
        api.load_arrays(recorder.path, 1)
    monkeypatch.setattr(api.os, "replace", replace)
    assert append_sample(recorder, reward=9) == 1
    assert api.read_steps(recorder.path)[1]["reward"] == 9


@pytest.mark.parametrize("changes", [{"status": "recording"}, {"status": "done"}, {"outcome": "done"}, {"error": {"message": "bad"}}])
def test_invalid_finish_leaves_episode_recording(api, tmp_path, changes):
    recorder = api.EpisodeRecorder(tmp_path, {})
    with pytest.raises((ValueError, TypeError)):
        recorder.finish(**changes)
    assert api.read_manifest(recorder.path)["status"] == "recording"
    assert append_sample(recorder) == 0


def test_failed_finish_preserves_manifest_and_can_be_retried(api, tmp_path, monkeypatch):
    recorder = api.EpisodeRecorder(tmp_path, {})
    append_sample(recorder)
    replace = api.os.replace

    def fail_commit(source, destination):
        raise OSError("simulated finish failure")

    monkeypatch.setattr(api.os, "replace", fail_commit)
    with pytest.raises(OSError, match="simulated finish failure"):
        recorder.finish(status="interrupted", error="keyboard interrupt")
    assert recorder.manifest["status"] == "recording"
    assert api.read_manifest(recorder.path)["status"] == "recording"
    monkeypatch.setattr(api.os, "replace", replace)
    recorder.finish(status="interrupted", error="keyboard interrupt")
    assert api.read_manifest(recorder.path)["status"] == "interrupted"


def interrupt_after_manifest_replace(api, monkeypatch):
    """Inject interruption after the filesystem commit, before Python resumes."""
    replace = api.os.replace

    def replace_then_interrupt(source, destination):
        replace(source, destination)
        if str(destination).endswith("manifest.json"):
            raise KeyboardInterrupt("interrupted after manifest commit")

    monkeypatch.setattr(api.os, "replace", replace_then_interrupt)


@pytest.mark.parametrize("next_operation", ["finish", "append", "manifest"])
def test_post_commit_append_interruption_preserves_committed_step(api, tmp_path, monkeypatch, next_operation):
    recorder = api.EpisodeRecorder(tmp_path, {})
    with monkeypatch.context() as patch:
        interrupt_after_manifest_replace(api, patch)
        with pytest.raises(KeyboardInterrupt, match="after manifest commit"):
            append_sample(recorder, reward=4)
    assert api.read_manifest(recorder.path)["num_steps"] == 1
    first_json = (recorder.path / "steps" / "000000.json").read_bytes()
    first_npz = (recorder.path / "steps" / "000000.npz").read_bytes()

    if next_operation == "finish":
        recorder.finish(status="interrupted")
        assert api.read_manifest(recorder.path)["status"] == "interrupted"
        assert len(api.read_steps(recorder.path)) == 1
    elif next_operation == "append":
        assert append_sample(recorder, reward=9) == 1
        assert [step["reward"] for step in api.read_steps(recorder.path)] == [4, 9]
    else:
        assert recorder.manifest["num_steps"] == 1
    assert (recorder.path / "steps" / "000000.json").read_bytes() == first_json
    assert (recorder.path / "steps" / "000000.npz").read_bytes() == first_npz


@pytest.mark.parametrize("next_operation", ["finish", "append", "manifest"])
def test_post_commit_finish_interruption_preserves_terminal_status(api, tmp_path, monkeypatch, next_operation):
    recorder = api.EpisodeRecorder(tmp_path, {})
    append_sample(recorder)
    with monkeypatch.context() as patch:
        interrupt_after_manifest_replace(api, patch)
        with pytest.raises(KeyboardInterrupt, match="after manifest commit"):
            recorder.finish(outcome="success")
    assert api.read_manifest(recorder.path)["status"] == "complete"
    committed = (recorder.path / "manifest.json").read_bytes()

    if next_operation == "finish":
        with pytest.raises(RuntimeError, match="different result"):
            recorder.finish(status="interrupted")
        recorder.finish(outcome="success")
    elif next_operation == "append":
        with pytest.raises(RuntimeError, match="finished episode"):
            append_sample(recorder)
    else:
        assert recorder.manifest["status"] == "complete"
        assert recorder.manifest["outcome"] == "success"
    assert (recorder.path / "manifest.json").read_bytes() == committed
    assert len(api.read_steps(recorder.path)) == 1


def test_array_names_cannot_collide_with_numpy_save_arguments(api, tmp_path):
    recorder = api.EpisodeRecorder(tmp_path, {})
    recorder.append({"file": [1]}, [0.0], {"allow_pickle": [2]}, arrays={"file": [3]})
    arrays = api.load_arrays(recorder.path, 0)
    np.testing.assert_array_equal(arrays["obs/file"], [1])
    np.testing.assert_array_equal(arrays["next_obs/allow_pickle"], [2])
    np.testing.assert_array_equal(arrays["extra/file"], [3])


def test_orphan_files_are_ignored_without_marking_recording_complete(api, tmp_path):
    recorder = api.EpisodeRecorder(tmp_path, {})
    append_sample(recorder)
    (recorder.path / "steps" / "000001.json").write_text("corrupt", encoding="utf-8")
    (recorder.path / "steps" / "000001.npz").write_bytes(b"orphan")
    assert len(api.read_steps(recorder.path)) == 1
    assert api.read_manifest(recorder.path)["status"] == "recording"


@pytest.mark.parametrize("damage", ["missing_json", "invalid_json", "wrong_index", "path_escape", "missing_npz"])
def test_committed_step_corruption_fails_with_step_context(api, tmp_path, damage):
    recorder = api.EpisodeRecorder(tmp_path, {})
    append_sample(recorder)
    path = recorder.path / "steps" / "000000.json"
    if damage == "missing_json":
        path.unlink()
    elif damage == "invalid_json":
        path.write_text("{broken", encoding="utf-8")
    elif damage == "missing_npz":
        (recorder.path / "steps" / "000000.npz").unlink()
    else:
        record = json.loads(path.read_text(encoding="utf-8"))
        record["index" if damage == "wrong_index" else "array_file"] = 9 if damage == "wrong_index" else "../../outside.npz"
        path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="[Ss]tep.*0"):
        api.read_steps(recorder.path)


@pytest.mark.parametrize("damage", ["invalid_archive", "object_array", "nonfinite_action", "missing_action"])
def test_array_reader_rejects_corrupt_or_pickle_dependent_committed_data(api, tmp_path, damage):
    recorder = api.EpisodeRecorder(tmp_path, {})
    append_sample(recorder)
    path = recorder.path / "steps" / "000000.npz"
    if damage == "invalid_archive":
        path.write_bytes(b"invalid archive")
    elif damage == "object_array":
        np.savez(path, action=np.array([object()], dtype=object))
    elif damage == "nonfinite_action":
        np.savez(path, action=np.array([np.nan]))
    else:
        np.savez(path, other=np.array([1.0]))
    with pytest.raises(ValueError, match="[Ss]tep.*0"):
        api.load_arrays(recorder.path, 0)


@pytest.mark.parametrize("index", [-1, 0, True, 0.5])
def test_array_reader_rejects_uncommitted_or_invalid_index(api, tmp_path, index):
    recorder = api.EpisodeRecorder(tmp_path, {})
    with pytest.raises((IndexError, TypeError)):
        api.load_arrays(recorder.path, index)


@pytest.mark.parametrize("field,value", [("schema_version", 2), ("num_steps", -1), ("num_steps", True), ("status", "finished"), ("outcome", "maybe")])
def test_manifest_reader_rejects_invalid_contract(api, tmp_path, field, value):
    recorder = api.EpisodeRecorder(tmp_path, {})
    manifest = recorder.manifest
    manifest[field] = value
    (recorder.path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="[Mm]anifest"):
        api.read_manifest(recorder.path)
