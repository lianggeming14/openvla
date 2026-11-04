"""Analysis contracts using committed recorder files and validated human v2 data."""

import csv
import importlib
import json
import shutil

import numpy as np
import pytest

from vla_trajectory.annotations import generate_annotations, save_annotations
from vla_trajectory.recorder import EpisodeRecorder


def api():
    try:
        return importlib.import_module("vla_trajectory.analysis")
    except ModuleNotFoundError:
        pytest.fail("episode analysis is not implemented")


def attempt(identifier, obj, grasp, place=None, **changes):
    row = dict(attempt_id=identifier, object=obj, start_frame=grasp,
               end_frame=place if place is not None else grasp,
               grasp_start_frame=None, grasp_frame=grasp,
               place_start_frame=None, place_frame=place,
               subtask_success=None, source="human_review", notes="")
    row.update(changes)
    return row


def episode(root, identifier="episode-1", *, outcome="success", status="complete",
            metadata=None, transitions=None, attempts=None, reviewed=True,
            manual_outcome=None):
    meta = dict(task_id="Task_01", task_name="双物体任务", init_state_id="S001",
                stage="formal", analysis_object_pair=["汤罐", "番茄罐"])
    meta.update(metadata or {})
    recorder = EpisodeRecorder(root, meta, episode_id=identifier)
    if transitions is None:
        transitions = [
            ("settling", [0, 0, 0], [100, 0, 0]),
            ("policy", [100, 0, 0], [103, 4, 0]),
            ("policy", [103, 4, 0], [103, 4, 2]),
            ("policy", [103, 4, 2], [103, 4, 5]),
        ]
    for phase, before, after in transitions:
        obs = {} if before is None else {"robot0_eef_pos": before}
        nxt = {} if after is None else {"robot0_eef_pos": after}
        obs["agentview_image"] = np.full((16, 16, 3), 30, dtype=np.uint8)
        nxt["agentview_image"] = np.full((16, 16, 3), 90, dtype=np.uint8)
        context = dict(chunk_id=0, predicates_before={"goal": False},
                       predicates_after={"goal": True})
        if phase is not None:
            context["phase"] = phase
        recorder.append(obs, [0.0] * 7, nxt, context=context)
    recorder.finish(status=status, outcome=outcome)
    document = generate_annotations(recorder.path)
    document.update(object_attempts=attempts or [], object_reviewed=reviewed)
    if manual_outcome is not None:
        document["outcome"] = manual_outcome
    save_annotations(recorder.path, document)
    return recorder.path


def write_execution(path, *, executed=3, recorded=3, count_complete=True,
                    recording_complete=True, executed_settling=1, recorded_settling=1):
    document = dict(schema_version=1, episode_id=path.name,
                    executed_policy_steps=executed, recorded_policy_steps=recorded,
                    executed_settling_steps=executed_settling, recorded_settling_steps=recorded_settling,
                    recording_complete=recording_complete,
                    step_count_complete=count_complete)
    (path / "execution.json").write_text(json.dumps(document), encoding="utf-8")


def fixture_video(path, monkeypatch):
    """Generate a real video API sidecar, substituting only the absent encoder.

    The placeholder is not playable media. These tests validate provenance and
    completeness logic, not MP4 encoding or the external FFmpeg executable.
    """
    from vla_trajectory import video

    class FixtureEncoder:
        def __init__(self, output):
            self.output = output

        def append_data(self, frame):
            pass

        def close(self):
            self.output.write_bytes(b"SYNTHETIC TEST ENCODER PLACEHOLDER")

    monkeypatch.setattr(video, "_open_writer", lambda output, fps: FixtureEncoder(output))
    return video.export_episode_video(path)


def test_policy_transition_lengths_include_first_action_exclude_settling_and_chunks(tmp_path):
    path = episode(tmp_path, outcome="failure", manual_outcome="success",
                   attempts=[attempt("a", "汤罐", 1), attempt("b", "番茄罐", 2)])
    row = api().analyze_episode(path)
    assert row["total_steps"] == 3  # three env.step calls in one inference chunk
    assert row["recorded_policy_steps"] == 3
    assert row["frame_to_action_step"] == {0: None, 1: 1, 2: 2, 3: 3}
    assert row["eef_path_length"] == pytest.approx(10.0)  # 5 + 2 + 3
    assert row["observed_eef_path_length"] == pytest.approx(10.0)
    assert row["eef_coverage"] == 1
    assert row["success"] is False
    assert row["official_outcome"] == "failure"
    assert row["manual_outcome"] == "success"
    assert row["pair_order"] == "A→B"


def test_missing_coordinate_marks_total_unknown_without_bridging_discontinuities(tmp_path):
    path = episode(tmp_path, transitions=[
        ("policy", [0, 0, 0], [1, 0, 0]),
        ("policy", None, [10, 0, 0]),
        ("policy", [20, 0, 0], [22, 0, 0]),
    ])
    row = api().analyze_episode(path)
    assert row["eef_path_length"] is None
    assert row["observed_eef_path_length"] == 3
    assert row["eef_coverage"] == pytest.approx(2 / 3)
    assert row["missing_eef_steps"] == 1
    assert row["trajectory"][1]["obs_xyz"] is None


def test_attempt_rows_keep_regrasp_failure_and_inclusive_operation_boundaries(tmp_path):
    path = episode(tmp_path, attempts=[
        attempt("a1", "汤罐", 1, 2, grasp_start_frame=1, place_start_frame=2,
                start_frame=1, end_frame=2, subtask_success=False),
        attempt("b1", "番茄罐", 2, 3, start_frame=2, end_frame=3),
        attempt("a2", "汤罐", 3, subtask_success=True),
    ])
    row = api().analyze_episode(path)
    assert row["grasp_order"] == "汤罐→番茄罐→汤罐"
    assert row["pair_order"] == "A→B"
    assert len(row["subtask_events"]) == 3
    first = row["subtask_events"][0]
    assert first["grasp_step"] == 1
    assert first["place_step"] == 2
    assert first["attempt_steps"] == 2
    assert first["grasp_operation_steps"] == 1
    assert first["place_operation_steps"] == 1
    assert first["grasp_to_place_interval_steps"] == 1
    assert first["subtask_success"] is False
    assert row["subtask_events"][1]["grasp_operation_steps"] is None
    assert row["subtask_events"][2]["place_step"] is None


def test_attempt_range_counts_policy_actions_even_when_start_is_settling(tmp_path):
    path = episode(tmp_path, attempts=[
        attempt("a", "汤罐", 2, 3, start_frame=0, end_frame=3,
                grasp_start_frame=1, place_start_frame=2),
    ])
    event = api().analyze_episode(path)["subtask_events"][0]
    assert event["start_step"] is None  # settling has no action number
    assert event["attempt_steps"] == 3  # three policy actions in inclusive frame range
    assert event["grasp_operation_steps"] == 2
    assert event["place_operation_steps"] == 2
    assert event["grasp_to_place_interval_steps"] == 1


def test_automatic_goal_truth_never_becomes_grasp_order_or_default_object_pair(tmp_path):
    row = api().analyze_episode(episode(tmp_path, metadata={"analysis_object_pair": None}))
    assert row["subtask_events"] == []
    assert row["object_pair"] is None
    assert row["pair_order"] is None
    assert row["grasp_order"] == "unknown"


@pytest.mark.parametrize("attempts,reviewed,status", [
    ([attempt("a", "汤罐", 1), attempt("b", "番茄罐", 1)], True, "tied"),
    ([attempt("a", "汤罐", 1), attempt("b", "番茄罐", None)], True, "unknown"),
    ([attempt("a", "汤罐", 1), attempt("b", "番茄罐", 2)], False, "unreviewed"),
])
def test_unknown_ties_and_unreviewed_are_not_forced_into_pair_order(tmp_path, attempts, reviewed, status):
    row = api().analyze_episode(episode(tmp_path, attempts=attempts, reviewed=reviewed))
    assert row["order_status"] == status
    assert row["pair_order"] is None
    assert len(row["subtask_events"]) == 2


@pytest.mark.parametrize("status,with_execution,count_complete,total,coverage", [
    ("error", False, True, None, None),
    ("error", True, True, 5, 0.6),
    ("error", True, False, None, None),
])
def test_failed_recording_never_reports_recorded_steps_as_executed_total(
        tmp_path, status, with_execution, count_complete, total, coverage):
    path = episode(tmp_path, status=status)
    if with_execution:
        write_execution(path, executed=5, count_complete=count_complete,
                        recording_complete=False)
    row = api().analyze_episode(path)
    assert row["total_steps"] == total
    assert row["recorded_policy_steps"] == 3
    assert row["eef_path_length"] is None
    assert row["eef_coverage"] == coverage
    assert row["recording_complete"] is False


def test_unknown_phase_prevents_unproven_action_mapping_and_total(tmp_path):
    row = api().analyze_episode(episode(tmp_path, transitions=[
        (None, [0, 0, 0], [1, 0, 0]),
        ("policy", [1, 0, 0], [2, 0, 0]),
    ], attempts=[attempt("a", "汤罐", 1)]))
    assert row["total_steps"] is None
    assert row["frame_to_action_step"][1] is None
    assert row["subtask_events"][0]["grasp_step"] is None
    assert row["phase_unknown_steps"] == 1


def test_execution_identity_and_count_mismatch_rejected(tmp_path):
    path = episode(tmp_path)
    write_execution(path, recorded=2)
    with pytest.raises(ValueError, match="execution.*recorded|recorded.*execution"):
        api().analyze_episode(path)


def test_unknown_phase_total_can_be_proven_by_execution_but_frame_mapping_cannot(tmp_path):
    path = episode(tmp_path, transitions=[
        (None, [0, 0, 0], [1, 0, 0]),
        ("policy", [1, 0, 0], [3, 0, 0]),
    ])
    write_execution(path, executed=1, recorded=1)
    row = api().analyze_episode(path)
    assert row["total_steps"] == 1
    assert row["frame_to_action_step"] == {0: None, 1: None}
    assert row["eef_path_length"] is None  # unknown phase could carry missing policy evidence


def test_execution_recorded_count_covers_unclassified_frames_without_inventing_positions(tmp_path):
    path = episode(tmp_path, transitions=[
        (None, [0, 0, 0], [1, 0, 0]),
        ("policy", [1, 0, 0], [3, 0, 0]),
    ])
    write_execution(path, executed=2, recorded=2, executed_settling=0, recorded_settling=0)
    row = api().analyze_episode(path)
    assert row["recorded_policy_steps"] == 2
    assert row["known_phase_policy_steps"] == 1
    assert row["executed_settling_steps"] == 0
    assert row["recorded_eef_coverage"] == .5
    assert row["eef_coverage"] == .5
    assert row["eef_path_length"] is None


def test_exports_three_csvs_group_by_task_stage_and_use_explicit_denominators(tmp_path):
    root = tmp_path / "episodes"
    pair = [attempt("a", "汤罐", 1), attempt("b", "番茄罐", 2)]
    episode(root, "success", attempts=pair)
    episode(root, "failed", outcome="failure", metadata={"init_state_id": "S002"},
            attempts=[attempt("a", "汤罐", 1), attempt("b", "番茄罐", 1)])
    episode(root, "unknown", outcome="unknown", reviewed=False,
            metadata={"init_state_id": "S003"})
    episode(root, "pilot", attempts=pair, metadata={"stage": "pilot"})
    episode(root, "other-task", attempts=pair, metadata={"task_id": "Task_02"})
    episode(root, "demo", attempts=pair, metadata={"synthetic": True})
    result = api().export_analysis(root, tmp_path / "out", plots=False)
    assert len(result["episodes"]) == 5
    formal = next(row for row in result["task_summary"]
                  if row["task_id"] == "Task_01" and row["stage"] == "formal")
    assert formal["episodes"] == 3
    assert formal["known_outcome_episodes"] == 2
    assert formal["unknown_outcome_episodes"] == 1
    assert formal["success_rate"] == 0.5
    assert formal["comparable_order_episodes"] == 1
    assert formal["tied_order_episodes"] == 1
    assert formal["unreviewed_episodes"] == 1
    assert formal["a_to_b_ratio"] == 1
    assert formal["b_to_a_ratio"] == 0
    assert formal["distinct_initial_states"] == 3
    assert formal["mean_successful_steps"] == 3
    assert set(result["paths"]) == {"episode_statistics", "subtask_events", "task_summary", "report"}
    for name in ("episode_statistics", "subtask_events", "task_summary"):
        with result["paths"][name].open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert rows
    report = result["paths"]["report"].read_text(encoding="utf-8")
    assert "正式" in report and "因果" in report and "视频" in report
    assert "20 个不同初态" in report


def test_synthetic_opt_in_is_visible_and_duplicate_episode_ids_are_rejected(tmp_path):
    root = tmp_path / "episodes"
    source = episode(root, metadata={"synthetic": True})
    excluded = api().export_analysis(root, tmp_path / "real", plots=False)
    assert excluded["episodes"] == []
    result = api().export_analysis(root, tmp_path / "demo", plots=False, include_synthetic=True)
    assert "SYNTHETIC" in result["paths"]["report"].read_text(encoding="utf-8")
    shutil.copytree(source, root / "copied")
    with pytest.raises(ValueError, match="[Dd]uplicate episode"):
        api().export_analysis(root, tmp_path / "duplicated", plots=False, include_synthetic=True)


def test_all_unknown_official_outcomes_keep_success_rate_unknown_and_reverse_order_visible(tmp_path):
    root = tmp_path / "episodes"
    episode(root, outcome="unknown", attempts=[
        attempt("a", "汤罐", 3), attempt("b", "番茄罐", 1),
    ])
    result = api().export_analysis(root, tmp_path / "out", plots=False)
    summary = result["task_summary"][0]
    assert summary["success_rate"] is None
    assert summary["mean_successful_steps"] is None
    assert summary["known_outcome_episodes"] == 0
    assert summary["unknown_outcome_episodes"] == 1
    assert summary["b_to_a_ratio"] == 1
    assert summary["a_to_b_ratio"] == 0
    assert result["episodes"][0]["grasp_order"] == "番茄罐→汤罐"


def test_csv_preserves_chinese_commas_newlines_and_neutralizes_formula_strings(tmp_path):
    path = episode(tmp_path / "episodes", metadata={"task_id": " =1+1"},
                   attempts=[attempt("a", "汤罐", 1, notes='@SUM(1,2)\n复核,"保留"')])
    result = api().export_analysis(path, tmp_path / "out", plots=False)
    with result["paths"]["episode_statistics"].open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["Task ID"] == "' =1+1"
    with result["paths"]["subtask_events"].open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["Object"] == "汤罐"
    assert row["Notes"] == '\'@SUM(1,2)\n复核,"保留"'


def test_arbitrary_nonempty_mp4_is_not_counted_as_complete_episode_video(tmp_path):
    path = episode(tmp_path)
    (path / "unrelated.mp4").write_bytes(b"nonempty but not provenance evidence")
    row = api().analyze_episode(path)
    assert row["video_present"] is False
    assert row["video_files"] == []
    assert row["video_issues"]


def test_independent_video_export_counts_despite_historical_execution_video_error(tmp_path, monkeypatch):
    path = episode(tmp_path)
    write_execution(path)
    document = json.loads((path / "execution.json").read_text(encoding="utf-8"))
    document.update(video_status="error", video_error="encoder was unavailable")
    (path / "execution.json").write_text(json.dumps(document), encoding="utf-8")
    video = fixture_video(path, monkeypatch)
    row = api().analyze_episode(path)
    assert row["video_present"] is True
    assert row["video_files"] == [str(video)]
    assert row["video_status"] == "error"
    assert row["video_error"] == "encoder was unavailable"


@pytest.mark.parametrize("damage", ["wrong_episode", "missing_settling", "missing_final_frame", "wrong_action_step"])
def test_video_sidecar_identity_and_complete_transition_mapping_are_required(tmp_path, damage, monkeypatch):
    path = episode(tmp_path)
    video = fixture_video(path, monkeypatch)
    sidecar = video.with_suffix(".json")
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    if damage == "wrong_episode":
        document["episode_id"] = "other"
    elif damage == "missing_settling":
        document["include_settling"] = False
    elif damage == "missing_final_frame":
        document["frames"].pop()
    else:
        document["frames"][-1]["action_step_id"] = 900
    sidecar.write_text(json.dumps(document), encoding="utf-8")
    row = api().analyze_episode(path)
    assert row["video_present"] is False
    assert row["video_issues"]


def test_plot_artifacts_and_event_markers_use_post_action_coordinates_without_gap_bridge(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    path = episode(tmp_path / "episodes", transitions=[
        ("policy", [0, 0, 0], [1, 0, 0]),
        ("policy", None, [10, 0, 0]),
        ("policy", [20, 0, 0], [22, 0, 0]),
    ], attempts=[attempt("a", "汤罐", 0, 2)])
    row = api().analyze_episode(path)
    plotting = importlib.import_module("vla_trajectory.plots")
    fig = plotting.create_episode_figure(row)
    spatial = fig.axes[0]
    segments = [line.get_data_3d() for line in spatial.lines]
    assert len(segments) == 2
    assert [list(segment[0]) for segment in segments] == [[0, 1], [20, 22]]
    assert len(spatial.collections) == 2  # grasp/place markers
    assert list(spatial.collections[0]._offsets3d[0]) == [1]
    assert list(spatial.collections[1]._offsets3d[0]) == [22]
    import matplotlib.pyplot as plt
    plt.close(fig)
    result = api().export_analysis(path, tmp_path / "out", plots=True)
    artifacts = result["plots"]["episode-1"]
    assert {p.suffix for p in artifacts} == {".svg", ".png"}
    assert all(p.stat().st_size > 1000 for p in artifacts)


def test_same_position_plot_events_share_one_label_and_keep_attempt_identity(tmp_path):
    import matplotlib.pyplot as plt
    row = api().analyze_episode(episode(tmp_path, attempts=[
        attempt("first", "汤罐", 1, 2), attempt("second", "番茄罐", 2, 3),
    ]))
    plotting = importlib.import_module("vla_trajectory.plots")
    fig = plotting.create_episode_figure(row)
    try:
        # place(first) and grasp(second) share next_obs at frame 2; one grouped
        # caption prevents drawing two labels on top of each other.
        assert len(fig.axes[0].texts) == 3
        legend = " ".join(text.get_text() for chart in fig.axes[1:] for text in chart.texts)
        assert "first" in legend and "second" in legend
    finally:
        plt.close(fig)
