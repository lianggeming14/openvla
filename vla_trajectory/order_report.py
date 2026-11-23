"""Report observed completion order without turning schedules into results.

Completion steps are zero-based committed record indices, including settling.
They describe the first proven simulator false-to-true events, not grasps or
causal effects of instructions. Query and video availability are separate
evidence fields; video availability is a file/sidecar check, not codec QA.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import csv
import hashlib
from pathlib import Path
from typing import Any

from .annotations import load_annotations
from .recorder import _read_json, _write_json, load_arrays, read_manifest, read_steps


def _subgoals(value):
    if (not isinstance(value, Mapping) or set(value) != {"A", "B"}
            or any(not isinstance(key, str) or not key.strip() for key in value.values())
            or value["A"] == value["B"]):
        raise ValueError("subgoal_keys must map A and B to two distinct nonempty predicate keys")
    return dict(value)


def _expected(value):
    if isinstance(value, (list, tuple)):
        value = "".join(value) if all(isinstance(item, str) for item in value) else value
    if value not in (None, "AB", "BA"):
        raise ValueError("expected_order must be AB, BA, a corresponding pair, or None")
    return value


def _snapshot(context, field, keys):
    value = context.get(field)
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a predicate mapping")
    result = {}
    for label, key in keys.items():
        observed = value.get(key)
        if observed is not None and type(observed) is not bool:
            raise ValueError(f"predicate {key!r} must be boolean or null")
        result[label] = observed
    return result


def classify_completion_order(steps, subgoal_keys) -> dict[str, Any]:
    """Classify first completions only when their evidence is continuous.

    A missing/initially true observation before a goal's first completion can
    hide an earlier completion, so a later transition cannot repair it. Known
    prior-after/current-before contradictions also break that evidence. Missing
    observations after a proven first completion do not erase its timestamp.
    A same-record completion is a tie at the available temporal resolution.
    """
    keys = _subgoals(subgoal_keys)
    first = {"A": None, "B": None}
    problems = {}
    previous = None
    count = 0
    for index, step in enumerate(steps):
        count += 1
        if (not isinstance(step, Mapping) or type(step.get("index")) is not int
                or step["index"] != index):
            raise ValueError("steps must have contiguous zero-based committed indices")
        context = step.get("context")
        if not isinstance(context, Mapping):
            raise ValueError("step context must be a mapping")
        before = _snapshot(context, "predicates_before", keys)
        after = _snapshot(context, "predicates_after", keys)
        for label in ("A", "B"):
            if first[label] is not None or label in problems:
                continue
            start, end = before[label], after[label]
            if start is None or end is None:
                problems[label] = "missing_predicate"
            elif index == 0 and start is True:
                problems[label] = "initially_true"
            elif previous is not None and previous[label] is not start:
                problems[label] = "discontinuous_snapshots"
            elif start is False and end is True:
                first[label] = index
        previous = after
    a, b = first["A"], first["B"]
    if problems:
        order, status = "unknown", "uncertain"
        reason = "; ".join(f"{label}:{reason}" for label, reason in sorted(problems.items()))
    elif a is None or b is None:
        order, status = "unknown", "incomplete"
        reason = "no_recorded_steps" if not count else "; ".join(
            f"{label}:no_observed_completion" for label in ("A", "B") if first[label] is None)
    else:
        order = "tie" if a == b else ("AB" if a < b else "BA")
        status = "complete"
        reason = "same_record_step" if order == "tie" else "observed_false_to_true_transitions"
    return dict(actual_completion_order=order, first_A_step=a, first_B_step=b,
                evidence_status=status, reason=reason)


def _execution(path, manifest, steps):
    sidecar = path / "execution.json"
    if not sidecar.is_file():
        return False, "missing", {}, "execution.json is missing"
    try:
        document = _read_json(sidecar)
        if (type(document.get("schema_version")) is not int or document["schema_version"] != 1
                or document.get("episode_id") != manifest["episode_id"]):
            raise ValueError("execution schema or episode identity mismatch")
        for field in ("recording_complete", "step_count_complete"):
            if type(document.get(field)) is not bool:
                raise ValueError(f"execution {field} must be boolean")
        counts = Counter(step["context"].get("phase") for step in steps)
        if counts["policy"] + counts["settling"] != len(steps):
            raise ValueError("recorded phase is unknown")
        for phase in ("policy", "settling"):
            recorded, executed = (document.get(prefix + phase + "_steps")
                                  for prefix in ("recorded_", "executed_"))
            if (type(recorded) is not int or type(executed) is not int
                    or recorded != counts[phase] or executed < recorded):
                raise ValueError("execution counts disagree with committed transitions")
            if document["recording_complete"] and recorded != executed:
                raise ValueError("execution completeness contradicts counts")
        complete = document["recording_complete"] and document["step_count_complete"]
        return complete, "available", document, None
    except (ValueError, OSError, TypeError) as exc:
        return False, "invalid", {}, str(exc)


def _queries(path, manifest, steps):
    sidecar = path / "queries.json"
    if not sidecar.is_file():
        return "missing", "queries.json is missing"
    try:
        document = _read_json(sidecar)
        if (type(document.get("schema_version")) is not int or document["schema_version"] != 1
                or document.get("episode_id") != manifest["episode_id"]
                or not isinstance(document.get("queries"), list)):
            raise ValueError("query schema or episode identity mismatch")
        instruction = manifest["metadata"].get("task_description")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("effective episode instruction is unavailable")
        digest = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        recorded_digest = manifest["metadata"].get("instruction_sha256")
        if recorded_digest is not None and recorded_digest != digest:
            raise ValueError("episode instruction hash mismatch")
        chunks = set()
        legacy_status = False
        for row in document["queries"]:
            if (not isinstance(row, dict) or type(row.get("chunk_id")) is not int
                    or row["chunk_id"] < 0 or row["chunk_id"] in chunks
                    or row.get("instruction") != instruction or row.get("instruction_sha256") != digest):
                raise ValueError("query instruction, hash or chunk identity mismatch")
            if "status" not in row:
                legacy_status = True
            elif row["status"] != "complete" or row.get("error") is not None:
                raise ValueError("model query did not complete successfully")
            chunks.add(row["chunk_id"])
        if not chunks:
            return "missing", "no actual model query was recorded"
        for step in steps:
            context = step["context"]
            if context.get("phase") != "policy":
                continue
            chunk = context.get("chunk_id")
            if type(chunk) is not int or chunk not in chunks:
                raise ValueError("policy transition has no recorded model query")
            for field in ("instruction", "query_instruction"):
                if field in context and context[field] != instruction:
                    raise ValueError("committed query instruction mismatch")
            for field in ("instruction_sha256", "query_instruction_sha256"):
                if field in context and context[field] != digest:
                    raise ValueError("committed query hash mismatch")
        if legacy_status:
            return "unknown", "legacy query rows do not record inference completion status"
        return "available", None
    except (ValueError, OSError, TypeError) as exc:
        return "invalid", str(exc)


def _videos(path, manifest, execution):
    declared = execution.get("video_path")
    if execution.get("video_status") != "complete" or not isinstance(declared, str):
        return "missing", [], "no completed video is declared"
    candidate = Path(declared)
    if candidate.is_absolute() or ".." in candidate.parts or candidate.suffix.lower() != ".mp4":
        return "invalid", [], "video path must reference an MP4 within the episode"
    video = path / candidate
    try:
        if not video.is_file() or video.stat().st_size == 0:
            return "missing", [], "declared video is missing or empty"
        sidecar = _read_json(video.with_suffix(".json"))
        if (type(sidecar.get("schema_version")) is not int or sidecar["schema_version"] != 1
                or sidecar.get("episode_id") != manifest["episode_id"]
                or not isinstance(sidecar.get("frames"), list)):
            raise ValueError("video sidecar schema or episode identity mismatch")
        return "available", [str(video)], None
    except (ValueError, OSError, TypeError) as exc:
        return "invalid", [str(video)] if video.is_file() else [], str(exc)


def _completion_positions(steps, classification, videos):
    """Map record indices only through explicit policy/video provenance."""
    result = {}
    video_frames = []
    for video in videos:
        try:
            video_frames.extend(_read_json(Path(video).with_suffix(".json"))["frames"])
        except (ValueError, OSError, TypeError, KeyError):
            continue
    for label in ("A", "B"):
        record_index = classification["first_" + label + "_step"]
        policy_step = video_frame = None
        if record_index is not None and record_index < len(steps):
            context = steps[record_index]["context"]
            value = context.get("action_step_id")
            if context.get("phase") == "policy" and type(value) is int and value > 0:
                policy_step = value
            matches = [frame["video_frame"] for frame in video_frames
                       if isinstance(frame, dict) and frame.get("observation") == "next_obs"
                       and type(frame.get("record_step")) is int and frame["record_step"] == record_index
                       and type(frame.get("video_frame")) is int and frame["video_frame"] >= 0]
            if len(matches) == 1:
                video_frame = matches[0]
        result["first_" + label + "_policy_step"] = policy_step
        result["first_" + label + "_video_frame"] = video_frame
    return result


def _grasp(path, metadata):
    result = dict(grasp_order="unknown", grasp_evidence_status="unknown",
                  first_grasp_A_step=None, first_grasp_B_step=None)
    pair = metadata.get("analysis_object_pair")
    if pair is None:
        if "analysis_object_pair" in metadata:
            result["grasp_evidence_status"] = "not_applicable"
        return result
    if (not isinstance(pair, (list, tuple)) or len(pair) != 2 or pair[0] == pair[1]
            or any(not isinstance(item, str) or not item.strip() for item in pair)):
        return dict(result, grasp_evidence_status="invalid")
    if not (path / "annotations.json").is_file():
        return result
    try:
        document = load_annotations(path)
    except (ValueError, OSError, TypeError):
        return dict(result, grasp_evidence_status="invalid")
    if not document["object_reviewed"]:
        return dict(result, grasp_evidence_status="unreviewed")
    attempts = [row for row in document["object_attempts"] if row["grasp_frame"] is not None]
    if any(row["source"] != "human_review" or row["object"] is None for row in attempts):
        return result
    first = [min((row["grasp_frame"] for row in attempts if row["object"] == obj), default=None)
             for obj in pair]
    result.update(first_grasp_A_step=first[0], first_grasp_B_step=first[1])
    if None not in first:
        result.update(grasp_order="tie" if first[0] == first[1] else ("AB" if first[0] < first[1] else "BA"),
                      grasp_evidence_status="manual_review")
    return result


def episode_order_result(episode, *, subgoal_keys, expected_order) -> dict[str, Any]:
    """Read an actual episode, preserving outcome and evidence gaps separately.

    ``official_success`` uses the recorder's official outcome, never annotation
    overrides or goal-order inference. Corrupt/incomplete recordings cannot
    supply a known official outcome or instruction compliance. Ties and unknown
    completion evidence have null compliance. Query availability establishes the
    recorded instruction; video availability is reported separately.
    """
    keys, expected = _subgoals(subgoal_keys), _expected(expected_order)
    path = Path(episode)
    manifest = read_manifest(path)
    metadata = manifest["metadata"]
    config = metadata.get("config")
    config = config if isinstance(config, Mapping) else {}
    instruction = metadata.get("task_description")
    instruction = instruction if isinstance(instruction, str) else None
    instruction_hash = None if instruction is None else hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    recording_error = None
    try:
        steps = read_steps(path)
        for step in steps:
            load_arrays(path, step["index"])
        classification = classify_completion_order(steps, keys)
    except (ValueError, OSError, TypeError) as exc:
        steps = []
        recording_error = str(exc)
        classification = dict(actual_completion_order="unknown", first_A_step=None,
                              first_B_step=None, evidence_status="corrupt", reason=recording_error)
    counted, execution_status, execution, execution_error = _execution(path, manifest, steps)
    query_status, query_error = _queries(path, manifest, steps)
    video_status, videos, video_error = _videos(path, manifest, execution)
    if read_manifest(path) != manifest:
        recording_error = "episode changed while reading report; retry a stable recording"
        classification = dict(actual_completion_order="unknown", first_A_step=None,
                              first_B_step=None, evidence_status="corrupt", reason=recording_error)
    complete = (manifest["status"] == "complete" and not manifest.get("error")
                and recording_error is None and counted)
    official = {"success": True, "failure": False, "unknown": None}[manifest["outcome"]] if complete else None
    order = classification["actual_completion_order"]
    compliance, compliance_reason = None, "completion_order_is_not_determinable"
    if not complete:
        compliance_reason = "episode_recording_is_not_complete"
    elif expected is None:
        compliance_reason = "no_expected_order"
    elif query_status != "available":
        compliance_reason = "actual_instruction_evidence_is_unavailable"
    elif order in ("AB", "BA"):
        completion_steps = (classification["first_A_step"], classification["first_B_step"])
        if any(steps[index]["context"].get("phase") != "policy" for index in completion_steps):
            compliance_reason = "completion_outside_policy_actions"
        else:
            compliance, compliance_reason = order == expected, "observed_completion_order"
    row = dict(episode_id=manifest["episode_id"], task_id=metadata.get("task_id"),
               task_name=metadata.get("task_name", metadata.get("task_description")),
               group_id=metadata.get("group_id"), condition_id=metadata.get("condition_id"),
               pretrained_checkpoint=config.get("pretrained_checkpoint"),
               model_repository=metadata.get("model_repository"),
               load_in_4bit=config.get("load_in_4bit") if type(config.get("load_in_4bit")) is bool else None,
               instruction=instruction, instruction_sha256=instruction_hash,
               actual_model_instruction=instruction if query_status == "available" else None,
               experiment_identity=metadata.get("experiment_identity"),
               init_state_id=metadata.get("init_state_id"), init_state_index=metadata.get("init_state_index"),
               state_sha256=metadata.get("state_sha256"), synthetic=metadata.get("synthetic") is True,
               status=manifest["status"], outcome=manifest["outcome"], official_success=official,
               error=manifest.get("error"), recorded_steps=manifest["num_steps"],
               recording_complete=bool(complete), recording_error=recording_error,
               subgoal_keys=keys, expected_order=expected, order_compliance=compliance,
               compliance_reason=compliance_reason, **classification, **_grasp(path, metadata),
               **_completion_positions(steps, classification, videos if video_status == "available" else []),
               query_evidence_status=query_status, query_evidence_error=query_error,
               video_evidence_status=video_status, video_evidence_error=video_error,
               execution_evidence_status=execution_status, execution_evidence_error=execution_error,
               evidence_complete=bool(complete and classification["evidence_status"] == "complete"
                                      and query_status == "available" and video_status == "available"),
               trajectory_path=str(path), manifest_path=str(path / "manifest.json"),
               queries_path=str(path / "queries.json") if (path / "queries.json").is_file() else None,
               execution_path=str(path / "execution.json") if (path / "execution.json").is_file() else None,
               annotations_path=str(path / "annotations.json") if (path / "annotations.json").is_file() else None,
               video_paths=videos)
    return row


def _summaries(episodes):
    groups = defaultdict(list)
    for row in episodes:
        key = (str(row["task_id"]), str(row["task_name"]), str(row["condition_id"]), row["synthetic"])
        groups[key].append(row)
    summaries = []
    for _, rows in sorted(groups.items()):
        successful = sum(row["official_success"] is True for row in rows)
        failed = sum(row["official_success"] is False for row in rows)
        known = successful + failed
        compliant = sum(row["order_compliance"] is True for row in rows)
        noncompliant = sum(row["order_compliance"] is False for row in rows)
        comparable = compliant + noncompliant
        summaries.append(dict(
            task_id=rows[0]["task_id"], task_name=rows[0]["task_name"],
            group_id=rows[0]["group_id"], condition_id=rows[0]["condition_id"],
            synthetic=rows[0]["synthetic"], recorded_episodes=len(rows),
            successful_episodes=successful, failed_episodes=failed,
            known_outcome_episodes=known, unknown_outcome_episodes=len(rows) - known,
            success_rate=successful / known if known else None,
            comparable_order_episodes=comparable, compliant_episodes=compliant,
            successful_and_compliant_episodes=sum(row["official_success"] is True
                                                  and row["order_compliance"] is True for row in rows),
            noncompliant_episodes=noncompliant, unknown_compliance_episodes=len(rows) - comparable,
            order_compliance_rate=compliant / comparable if comparable else None,
            AB_episodes=sum(row["actual_completion_order"] == "AB" for row in rows),
            BA_episodes=sum(row["actual_completion_order"] == "BA" for row in rows),
            tied_order_episodes=sum(row["actual_completion_order"] == "tie" for row in rows),
            unknown_order_episodes=sum(row["actual_completion_order"] == "unknown" for row in rows),
            complete_evidence_episodes=sum(row["evidence_complete"] for row in rows)))
    return summaries


def _write_csv(path, rows, empty_columns):
    import json

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else empty_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list))
                             else value for key, value in row.items()})


def export_order_results(episode_specs, output_dir, *, include_synthetic=False) -> dict[str, Any]:
    """Write episodes.csv, summary.csv and results.json for recorded attempts.

    Each spec is ``{episode: path, subgoal_keys: {A: key, B: key},
    expected_order: AB|BA|None}``. Missing manifests are counted as excluded
    unrecorded inputs, never materialized as episode results. Duplicate supplied
    attempts remain distinct rows; callers select a latest-attempt policy.
    Synthetic rows require an explicit opt-in and have separate summary groups.
    """
    rows = []
    synthetic = unrecorded = 0
    for spec in episode_specs:
        if not isinstance(spec, Mapping) or not {"episode", "subgoal_keys", "expected_order"} <= spec.keys():
            raise ValueError("episode spec requires episode, subgoal_keys and expected_order")
        path = Path(spec["episode"])
        if not (path / "manifest.json").is_file():
            unrecorded += 1
            continue
        manifest = read_manifest(path)
        if manifest["metadata"].get("synthetic") is True and not include_synthetic:
            synthetic += 1
            continue
        rows.append(episode_order_result(path, subgoal_keys=spec["subgoal_keys"],
                                         expected_order=spec["expected_order"]))
    report = dict(schema_version=1, artifact_type="recorded_episode_order_results",
                  step_index_basis="zero_based_committed_record_including_settling",
                  contains_experiment_results=any(not row["synthetic"] for row in rows),
                  episodes=rows, summary=_summaries(rows), excluded_synthetic=synthetic,
                  excluded_unrecorded=unrecorded)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "episodes.csv", rows, ["episode_id", "task_name", "condition_id"])
    _write_csv(output / "summary.csv", report["summary"], ["task_name", "condition_id", "recorded_episodes"])
    _write_json(output / "results.json", report)
    return report
