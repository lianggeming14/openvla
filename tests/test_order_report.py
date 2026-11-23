"""Order reports keep simulator evidence, delivered prompts and outcomes distinct."""

import csv
import hashlib
import importlib
import json

import pytest

from vla_trajectory.annotations import generate_annotations, save_annotations
from vla_trajectory.recorder import EpisodeRecorder


KEYS = {"A": "in(cheese, basket)", "B": "in(butter, basket)"}


def api():
    try:
        return importlib.import_module("vla_trajectory.order_report")
    except ModuleNotFoundError:
        pytest.fail("completion-order report is not implemented")


def steps(snapshots):
    return [dict(index=index, context={"phase": "policy",
                                      "predicates_before": before,
                                      "predicates_after": after})
            for index, (before, after) in enumerate(snapshots)]


def state(a, b):
    return {KEYS["A"]: a, KEYS["B"]: b}


AB = [(state(False, False), state(True, False)),
      (state(True, False), state(True, True))]
BA = [(state(False, False), state(False, True)),
      (state(False, True), state(True, True))]


@pytest.mark.parametrize("snapshots,want,a,b", [
    (AB, "AB", 0, 1),
    (BA, "BA", 1, 0),
    ([(state(False, False), state(True, True))], "tie", 0, 0),
])
def test_only_observed_transitions_establish_completion_order(snapshots, want, a, b):
    result = api().classify_completion_order(steps(snapshots), KEYS)
    assert result["actual_completion_order"] == want
    assert result["first_A_step"] == a
    assert result["first_B_step"] == b
    assert result["evidence_status"] == "complete"


@pytest.mark.parametrize("snapshots,reason", [
    ([(state(True, False), state(False, True)),
      (state(False, True), state(True, True))], "initially_true"),
    ([({}, state(False, False)), *AB], "missing"),
    ([(state(False, False), state(None, False)),
      *AB], "missing"),
    ([(state(False, False), state(False, False)),
      (state(True, False), state(False, True)),
      (state(False, True), state(True, True))], "discontinuous"),
])
def test_uncertain_early_evidence_never_promotes_later_transition(snapshots, reason):
    result = api().classify_completion_order(steps(snapshots), KEYS)
    assert result["actual_completion_order"] == "unknown"
    assert result["first_A_step"] is None
    assert result["evidence_status"] == "uncertain"
    assert reason in result["reason"]


def test_later_missing_predicates_do_not_erase_proven_first_completions():
    result = api().classify_completion_order(steps([*AB, ({}, {})]), KEYS)
    assert result["actual_completion_order"] == "AB"
    assert result["first_A_step"] == 0
    assert result["first_B_step"] == 1


@pytest.mark.parametrize("snapshots", [[], [(state(False, False), state(True, False))]])
def test_unfinished_subgoals_remain_unknown(snapshots):
    result = api().classify_completion_order(steps(snapshots), KEYS)
    assert result["actual_completion_order"] == "unknown"
    assert result["evidence_status"] == "incomplete"
    assert result["first_B_step"] is None


@pytest.mark.parametrize("keys", [{"A": "x"}, {"A": "x", "B": "x"}, ["a", "b"]])
def test_invalid_subgoal_definition_is_rejected(keys):
    with pytest.raises(ValueError, match="subgoal_keys"):
        api().classify_completion_order([], keys)


def test_invalid_predicate_value_and_noncontiguous_records_are_rejected():
    with pytest.raises(ValueError, match="boolean"):
        api().classify_completion_order(steps([(state(1, False), state(True, True))]), KEYS)
    records = steps(AB)
    records[1]["index"] = 4
    with pytest.raises(ValueError, match="contiguous"):
        api().classify_completion_order(records, KEYS)


def episode(root, name="episode-1", *, snapshots=AB, outcome="success",
            status="complete", metadata=None, queries=True, video=True):
    instruction = "First cheese, then butter."
    meta = dict(task_id=2, task_name="cheese and butter", group_id=1,
                condition_id="AB", task_description=instruction,
                instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
                init_state_id="S001", state_sha256="state-fingerprint",
                config={"pretrained_checkpoint": "local/original-oft", "load_in_4bit": True},
                model_repository="moojink/openvla-7b-oft-finetuned-libero-10",
                analysis_object_pair=["cheese", "butter"])
    meta.update(metadata or {})
    recorder = EpisodeRecorder(root, meta, episode_id=name)
    for index, record in enumerate(steps(snapshots)):
        recorder.append({}, [0.0] * 7, {}, context={
            **record["context"], "chunk_id": index, "action_step_id": index + 1,
            "instruction": instruction,
            "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
        })
    recorder.finish(outcome=outcome, status=status)
    execution = dict(schema_version=1, episode_id=name,
                     executed_policy_steps=len(snapshots), recorded_policy_steps=len(snapshots),
                     executed_settling_steps=0, recorded_settling_steps=0,
                     step_count_complete=True, recording_complete=True,
                     video_status="complete" if video else "not_requested")
    if video:
        (recorder.path / "evidence.mp4").write_bytes(b"fixture video presence")
        (recorder.path / "evidence.json").write_text(json.dumps({
            "schema_version": 1, "episode_id": name, "frames": [
                {"video_frame": 0, "record_step": 0, "observation": "obs", "action_step_id": None},
                *[{"video_frame": index + 1, "record_step": index, "observation": "next_obs",
                   "action_step_id": index + 1} for index in range(len(snapshots))],
            ],
        }), encoding="utf-8")
        execution["video_path"] = "evidence.mp4"
    (recorder.path / "execution.json").write_text(json.dumps(execution), encoding="utf-8")
    if queries:
        (recorder.path / "queries.json").write_text(json.dumps({
            "schema_version": 1, "episode_id": name,
            "queries": [dict(chunk_id=index, instruction=instruction,
                             instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
                             status="complete", error=None)
                        for index in range(len(snapshots))],
        }), encoding="utf-8")
    return recorder.path


def test_success_and_order_compliance_are_distinct_with_auditable_paths(tmp_path):
    path = episode(tmp_path, snapshots=BA)
    result = api().episode_order_result(path, subgoal_keys=KEYS, expected_order="AB")
    assert result["official_success"] is True
    assert result["outcome"] == "success"
    assert result["actual_completion_order"] == "BA"
    assert result["order_compliance"] is False
    assert result["recording_complete"] is True
    assert result["evidence_complete"] is True
    assert result["trajectory_path"] == str(path)
    assert result["queries_path"] == str(path / "queries.json")
    assert result["video_paths"] == [str(path / "evidence.mp4")]
    assert result["grasp_order"] == "unknown"


@pytest.mark.parametrize("status", ["recording", "error", "interrupted"])
def test_noncomplete_episode_never_counts_as_success_or_compliant(tmp_path, status):
    path = episode(tmp_path, outcome="unknown", status="complete" if status == "recording" else status)
    if status == "recording":
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "recording"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = api().episode_order_result(path, subgoal_keys=KEYS, expected_order="AB")
    assert result["status"] == status
    assert result["official_success"] is None
    assert result["recording_complete"] is False
    assert result["order_compliance"] is None
    assert result["actual_completion_order"] == "AB"


@pytest.mark.parametrize("snapshots", [
    [(state(False, False), state(True, True))],
    [(state(True, False), state(True, True))],
])
def test_tie_or_initial_truth_cannot_prove_requested_sequence(tmp_path, snapshots):
    result = api().episode_order_result(episode(tmp_path, snapshots=snapshots),
                                        subgoal_keys=KEYS, expected_order="AB")
    assert result["official_success"] is True
    assert result["order_compliance"] is None


def test_order_compliance_can_hold_when_official_task_fails(tmp_path):
    result = api().episode_order_result(episode(tmp_path, outcome="failure"),
                                        subgoal_keys=KEYS, expected_order=["A", "B"])
    assert result["official_success"] is False
    assert result["order_compliance"] is True


@pytest.mark.parametrize("queries,video", [(False, True), (True, False), (False, False)])
def test_missing_query_or_video_remains_an_explicit_evidence_gap(tmp_path, queries, video):
    result = api().episode_order_result(episode(tmp_path, queries=queries, video=video),
                                        subgoal_keys=KEYS, expected_order="AB")
    assert result["evidence_complete"] is False
    assert result["query_evidence_status"] == ("available" if queries else "missing")
    assert result["video_evidence_status"] == ("available" if video else "missing")
    assert result["actual_completion_order"] == "AB"
    if not queries:
        assert result["order_compliance"] is None


def test_mismatched_query_cannot_be_called_instruction_compliance(tmp_path):
    path = episode(tmp_path)
    query_path = path / "queries.json"
    document = json.loads(query_path.read_text(encoding="utf-8"))
    document["queries"][0]["instruction"] = "different actual prompt"
    query_path.write_text(json.dumps(document), encoding="utf-8")
    result = api().episode_order_result(path, subgoal_keys=KEYS, expected_order="AB")
    assert result["query_evidence_status"] == "invalid"
    assert result["order_compliance"] is None
    assert result["evidence_complete"] is False


@pytest.mark.parametrize("query_status,evidence_status", [
    ("pending", "invalid"), ("error", "invalid"), (None, "unknown"),
])
def test_only_completed_model_queries_are_available_instruction_evidence(tmp_path, query_status, evidence_status):
    path = episode(tmp_path)
    query_path = path / "queries.json"
    document = json.loads(query_path.read_text(encoding="utf-8"))
    if query_status is None:
        del document["queries"][0]["status"]
    else:
        document["queries"][0]["status"] = query_status
    query_path.write_text(json.dumps(document), encoding="utf-8")
    result = api().episode_order_result(path, subgoal_keys=KEYS, expected_order="AB")
    assert result["query_evidence_status"] == evidence_status
    assert result["actual_model_instruction"] is None
    assert result["order_compliance"] is None
    assert result["evidence_complete"] is False


def test_report_preserves_model_identity_and_actual_delivered_instruction(tmp_path):
    result = api().episode_order_result(episode(tmp_path), subgoal_keys=KEYS, expected_order="AB")
    assert result["pretrained_checkpoint"] == "local/original-oft"
    assert result["model_repository"] == "moojink/openvla-7b-oft-finetuned-libero-10"
    assert result["load_in_4bit"] is True
    assert result["instruction"] == "First cheese, then butter."
    assert result["instruction_sha256"] == hashlib.sha256(b"First cheese, then butter.").hexdigest()
    assert result["actual_model_instruction"] == "First cheese, then butter."


def test_completion_record_policy_and_video_positions_use_their_own_evidence(tmp_path):
    path = episode(tmp_path)
    sidecar = path / "evidence.json"
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    document["frames"][1]["video_frame"] = 17
    document["frames"][2]["video_frame"] = 23
    sidecar.write_text(json.dumps(document), encoding="utf-8")
    result = api().episode_order_result(path, subgoal_keys=KEYS, expected_order="AB")
    assert result["first_A_step"] == 0
    assert result["first_B_step"] == 1
    assert result["first_A_policy_step"] == 1
    assert result["first_B_policy_step"] == 2
    assert result["first_A_video_frame"] == 17
    assert result["first_B_video_frame"] == 23


def test_unavailable_completion_position_mappings_stay_null(tmp_path):
    path = episode(tmp_path, video=False)
    record_path = path / "steps" / "000000.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    del record["context"]["action_step_id"]
    record_path.write_text(json.dumps(record), encoding="utf-8")
    result = api().episode_order_result(path, subgoal_keys=KEYS, expected_order="AB")
    assert result["first_A_step"] == 0
    assert result["first_A_policy_step"] is None
    assert result["first_A_video_frame"] is None
    assert result["first_B_video_frame"] is None


def test_missing_committed_arrays_mark_recording_corrupt(tmp_path):
    path = episode(tmp_path)
    (path / "steps" / "000001.npz").write_bytes(b"not an archive")
    result = api().episode_order_result(path, subgoal_keys=KEYS, expected_order="AB")
    assert result["recording_complete"] is False
    assert result["evidence_status"] == "corrupt"
    assert result["order_compliance"] is None
    assert result["official_success"] is None


def test_false_execution_completeness_is_preserved(tmp_path):
    path = episode(tmp_path)
    execution_path = path / "execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution.update(recording_complete=False, step_count_complete=False)
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    result = api().episode_order_result(path, subgoal_keys=KEYS, expected_order="AB")
    assert result["recording_complete"] is False
    assert result["order_compliance"] is None
    assert result["official_success"] is None


def test_explicit_null_object_pair_keeps_stove_group_out_of_grasp_comparison(tmp_path):
    path = episode(tmp_path, metadata={"group_id": 6, "analysis_object_pair": None})
    result = api().episode_order_result(path, subgoal_keys=KEYS, expected_order="AB")
    assert result["actual_completion_order"] == "AB"
    assert result["grasp_order"] == "unknown"
    assert result["grasp_evidence_status"] == "not_applicable"


def test_grasp_order_requires_reviewed_manual_grasp_frames(tmp_path):
    path = episode(tmp_path)
    document = generate_annotations(path)
    document["object_reviewed"] = True
    for name, frame in [("butter", 0), ("cheese", 1)]:
        document["object_attempts"].append(dict(
            attempt_id=name, object=name, start_frame=frame, end_frame=frame,
            grasp_start_frame=None, grasp_frame=frame, place_start_frame=None,
            place_frame=None, subtask_success=None, source="human_review", notes=""))
    save_annotations(path, document)
    result = api().episode_order_result(path, subgoal_keys=KEYS, expected_order="AB")
    assert result["actual_completion_order"] == "AB"
    assert result["grasp_order"] == "BA"
    assert result["grasp_evidence_status"] == "manual_review"


def test_exports_only_recorded_nonsynthetic_attempts_and_groups_known_denominators(tmp_path):
    paths = [episode(tmp_path, "success"),
             episode(tmp_path, "failure", snapshots=BA, outcome="failure"),
             episode(tmp_path, "unknown", outcome="unknown"),
             episode(tmp_path, "condition-BA", metadata={"condition_id": "BA"}),
             episode(tmp_path, "synthetic", metadata={"synthetic": True}),
             tmp_path / "planned-only"]
    specs = [dict(episode=path, subgoal_keys=KEYS, expected_order="AB") for path in paths]
    output = tmp_path / "report"
    report = api().export_order_results(specs, output)
    assert len(report["episodes"]) == 4
    assert report["excluded_synthetic"] == 1
    assert report["excluded_unrecorded"] == 1
    group = next(row for row in report["summary"] if row["condition_id"] == "AB")
    assert group["recorded_episodes"] == 3
    assert group["known_outcome_episodes"] == 2
    assert group["unknown_outcome_episodes"] == 1
    assert group["success_rate"] == 0.5
    assert group["successful_and_compliant_episodes"] == 1
    assert group["comparable_order_episodes"] == 3
    assert group["order_compliance_rate"] == pytest.approx(2 / 3)
    with (output / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert {row["episode_id"] for row in rows} == {"success", "failure", "unknown", "condition-BA"}
    assert json.loads((output / "results.json").read_text(encoding="utf-8")) == report
    assert (output / "summary.csv").is_file()


def test_opted_in_synthetic_is_kept_in_a_separate_summary(tmp_path):
    real = episode(tmp_path, "real")
    synthetic = episode(tmp_path, "synthetic", metadata={"synthetic": True})
    report = api().export_order_results([
        dict(episode=path, subgoal_keys=KEYS, expected_order="AB") for path in (real, synthetic)
    ], tmp_path / "report", include_synthetic=True)
    assert len(report["episodes"]) == 2
    assert len(report["summary"]) == 2
    assert sorted(row["synthetic"] for row in report["summary"]) == [False, True]
