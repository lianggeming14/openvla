"""Evidence-preserving statistics for recorded transitions and human attempts.

Action Steps are 1-based policy environment transitions, not inference chunks.
All unknown measurements remain ``None``; automatic predicates never supply
human grasp/place events. Matplotlib is imported only when plots are requested.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from .annotations import load_annotations
from .recorder import load_arrays, read_manifest, read_steps


def _pair(value):
    if value is None:
        return None
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(not isinstance(item, str) or not item.strip() for item in value)
            or value[0] == value[1]):
        raise ValueError("object_pair must contain two different nonempty object names")
    return list(value)


def _xyz(arrays, key):
    value = arrays.get(key)
    if value is None:
        return None
    value = np.asarray(value)
    if value.shape != (3,) or not np.isfinite(value).all():
        return None
    return value.astype(float).tolist()


def _execution(path, manifest, policy, settling, unknown):
    sidecar = path / "execution.json"
    if not sidecar.exists():
        complete = manifest["status"] == "complete" and unknown == 0
        return dict(total_steps=policy if complete else None,
                    executed_policy_steps=None, executed_settling_steps=None,
                    recorded_policy_steps=policy, recorded_settling_steps=settling,
                    recording_complete=complete,
                    step_count_complete=complete, execution_source="legacy_phase_count",
                    video_status=None, video_error=None)
    try:
        document = json.loads(sidecar.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError(f"Invalid execution.json: {exc}") from exc
    if (not isinstance(document, dict) or type(document.get("schema_version")) is not int
            or document["schema_version"] != 1
            or document.get("episode_id") != manifest["episode_id"]):
        raise ValueError("execution.json schema or episode identity mismatch")
    counts = ["executed_policy_steps", "recorded_policy_steps",
              "executed_settling_steps", "recorded_settling_steps"]
    for name in counts:
        if type(document.get(name)) is not int or document[name] < 0:
            raise ValueError(f"execution.json {name} must be a nonnegative integer")
    for name in ("step_count_complete", "recording_complete"):
        if type(document.get(name)) is not bool:
            raise ValueError(f"execution.json {name} must be boolean")
    rp, rs = document["recorded_policy_steps"], document["recorded_settling_steps"]
    ep, es = document["executed_policy_steps"], document["executed_settling_steps"]
    if (rp < policy or rs < settling or rp + rs != policy + settling + unknown
            or (unknown == 0 and (rp != policy or rs != settling))
            or ep < rp or es < rs):
        raise ValueError("execution.json recorded counts disagree with committed transitions")
    if document["recording_complete"] and (ep != rp or es != rs):
        raise ValueError("execution.json recording_complete contradicts recorded counts")
    return dict(total_steps=ep if document["step_count_complete"] else None,
                executed_policy_steps=ep, executed_settling_steps=es,
                recorded_policy_steps=rp, recorded_settling_steps=rs,
                recording_complete=document["recording_complete"],
                step_count_complete=document["step_count_complete"],
                execution_source="execution.json",
                video_status=document.get("video_status"),
                video_error=document.get("video_error"))


def _span(phases, start, end):
    if start is None or end is None:
        return None
    interval = [phases[frame] for frame in range(start, end + 1)]
    if any(phase not in ("policy", "settling") for phase in interval):
        return None
    return interval.count("policy")


def _orders(attempts, reviewed, pair):
    grouped = defaultdict(list)
    unresolved = []
    for row in attempts:
        if row["grasp_step"] is None or row["object"] is None:
            unresolved.append(row)
        else:
            grouped[row["grasp_step"]].append(row["object"])
    # Braces mean same-resolution events, not a list-order tiebreak.
    groups = [objects[0] if len(objects) == 1 else "{" + " = ".join(objects) + "}"
              for _, objects in sorted(grouped.items())]
    sequence = "→".join(groups) or "unknown"
    if unresolved:
        sequence += " | unknown: " + ", ".join(row["attempt_id"] for row in unresolved)
    if not reviewed:
        return sequence, "unreviewed", None
    if pair is None:
        return sequence, "unknown", None
    relevant = [row for row in attempts if row["object"] in pair or row["object"] is None]
    if any(row["grasp_step"] is None or row["object"] is None for row in relevant):
        return sequence, "unknown", None
    first = {}
    for row in relevant:
        name = row["object"]
        first[name] = min(first.get(name, row["grasp_step"]), row["grasp_step"])
    if any(obj not in first for obj in pair):
        return sequence, "unknown", None
    if first[pair[0]] == first[pair[1]]:
        return sequence, "tied", None
    return sequence, "comparable", "A→B" if first[pair[0]] < first[pair[1]] else "B→A"


def _videos(path, manifest, steps, mapping, execution):
    """Validate episode-local video provenance, including later standalone exports.

    A nonempty media file alone is not evidence of complete coverage. Validation
    checks the exporter sidecar against every committed transition. It does not
    decode the MP4 or certify its codec; encoding remains the video API's job.
    """
    valid, issues = [], []
    expected = []
    if steps:
        first = steps[0]
        expected.append(dict(video_frame=0, record_step=first["index"],
                             action_step_id=None, phase=first["context"].get("phase"), observation="obs"))
        for step in steps:
            expected.append(dict(video_frame=len(expected), record_step=step["index"],
                                 action_step_id=mapping[step["index"]],
                                 phase=step["context"].get("phase"), observation="next_obs"))
    for candidate in sorted(path.rglob("*.mp4")):
        try:
            if not candidate.is_file() or candidate.stat().st_size == 0:
                raise ValueError("empty video")
            sidecar = candidate.with_suffix(".json")
            document = json.loads(sidecar.read_text(encoding="utf-8"))
            if not isinstance(document, dict) or type(document.get("schema_version")) is not int or document["schema_version"] != 1:
                raise ValueError("invalid sidecar schema")
            for key in ("episode_id", "task_id", "init_state_id"):
                wanted = manifest["episode_id"] if key == "episode_id" else manifest["metadata"].get(key)
                if json.dumps(document.get(key), sort_keys=True) != json.dumps(wanted, sort_keys=True):
                    raise ValueError(f"sidecar {key} mismatch")
            if document.get("include_settling") is not True:
                raise ValueError("video does not include settling")
            if (not expected or any(step["context"].get("phase") not in ("policy", "settling") for step in steps)
                    or not execution["recording_complete"] or not execution["step_count_complete"]
                    or manifest["status"] == "recording"):
                raise ValueError("complete executed transition coverage is unproven")
            if json.dumps(document.get("frames"), sort_keys=True) != json.dumps(expected, sort_keys=True):
                raise ValueError("sidecar frame mapping does not cover first obs and every next_obs")
            fps = document.get("playback_fps")
            if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
                raise ValueError("sidecar playback_fps must be finite and positive")
            if not isinstance(document.get("camera"), str) or not document["camera"]:
                raise ValueError("missing sidecar camera")
            valid.append(str(candidate))
        except (OSError, ValueError, TypeError) as exc:
            issues.append(f"{candidate.name}: {exc}")
    return valid, issues


def analyze_episode(path, *, object_pair=None, eef_key="robot0_eef_pos") -> dict[str, Any]:
    """Read one stable episode; return measurements, attempted events and XYZ edges.

    ``subtask_events`` retains one row per annotation attempt. Operation durations
    include both boundary actions; the grasp-to-place interval excludes the
    grasp action. ``trajectory`` contains separate policy edges, never bridges.
    """
    path = Path(path)
    manifest = read_manifest(path)
    steps = read_steps(path)
    annotations = load_annotations(path)
    metadata = manifest["metadata"]
    pair = _pair(object_pair if object_pair is not None else metadata.get("analysis_object_pair"))
    mapping, trajectory = {}, []
    phases = {step["index"]: step["context"].get("phase") for step in steps}
    policy = settling = unknown = observed = 0
    observed_length = 0.0
    for step in steps:
        frame = step["index"]
        phase = step["context"].get("phase")
        if phase == "settling":
            settling += 1
            mapping[frame] = None
            continue
        if phase != "policy":
            unknown += 1
            mapping[frame] = None
            continue
        policy += 1
        mapping[frame] = policy if unknown == 0 else None
        arrays = load_arrays(path, frame)
        before, after = _xyz(arrays, f"obs/{eef_key}"), _xyz(arrays, f"next_obs/{eef_key}")
        edge = dict(frame=frame, action_step=mapping[frame], obs_xyz=before, next_xyz=after)
        trajectory.append(edge)
        if before is not None and after is not None:
            observed += 1
            observed_length += float(np.linalg.norm(np.asarray(after) - np.asarray(before)))
    execution = _execution(path, manifest, policy, settling, unknown)
    total = execution["total_steps"]
    complete_xyz = (total is not None and observed == total and unknown == 0
                    and execution["recording_complete"])
    coverage = (observed / total if total else 1.0) if total is not None else None
    attempts = []
    for original in annotations["object_attempts"]:
        row = dict(original)
        row.update(episode_id=manifest["episode_id"],
                   task_id=metadata.get("task_id", metadata.get("task_name", metadata.get("task_description"))),
                   stage=metadata.get("stage", "unknown"),
                   object_reviewed=annotations["object_reviewed"])
        for field in ("start", "end", "grasp_start", "grasp", "place_start", "place"):
            row[field + "_step"] = mapping.get(row.get(field + "_frame"))
        row["attempt_steps"] = _span(phases, row["start_frame"], row["end_frame"])
        row["grasp_operation_steps"] = _span(phases, row["grasp_start_frame"], row["grasp_frame"])
        row["place_operation_steps"] = _span(phases, row["place_start_frame"], row["place_frame"])
        row["grasp_to_place_interval_steps"] = (
            row["place_step"] - row["grasp_step"]
            if row["place_step"] is not None and row["grasp_step"] is not None else None)
        attempts.append(row)
    order, order_status, pair_order = _orders(attempts, annotations["object_reviewed"], pair)
    videos, video_issues = _videos(path, manifest, steps, mapping, execution)
    row = dict(episode_id=manifest["episode_id"], task_id=metadata.get("task_id"),
               task_name=metadata.get("task_name", metadata.get("task_description")),
               init_state_id=metadata.get("init_state_id"),
               init_state_index=metadata.get("init_state_index"),
               state_sha256=metadata.get("state_sha256"), stage=metadata.get("stage", "unknown"),
               synthetic=metadata.get("synthetic") is True, object_pair=pair,
               official_outcome=manifest["outcome"], manual_outcome=annotations["outcome"],
               success={"success": True, "failure": False, "unknown": None}[manifest["outcome"]],
               status=manifest["status"], known_phase_policy_steps=policy,
               recorded_steps=manifest["num_steps"], phase_unknown_steps=unknown,
               eef_path_length=observed_length if complete_xyz else None,
               observed_eef_path_length=observed_length, eef_coverage=coverage,
               observed_eef_steps=observed, missing_eef_steps=(total - observed if total is not None else None),
               recorded_eef_coverage=(observed / execution["recorded_policy_steps"]
                                     if execution["recorded_policy_steps"] else None),
               grasp_order=order, order_status=order_status, pair_order=pair_order,
               object_reviewed=annotations["object_reviewed"], subtask_events=attempts,
               frame_to_action_step=mapping, trajectory=trajectory,
               video_present=bool(videos), video_files=videos, video_issues=video_issues,
               notes=annotations["notes"], **execution)
    if row["task_id"] is None:
        row["task_id"] = row["task_name"]  # old recordings remain grouped by actual task text
    for event in attempts:
        event["task_id"] = row["task_id"]
    if read_manifest(path) != manifest:
        raise ValueError("Episode changed while reading analysis; retry a stable snapshot")
    return row


def _summaries(episodes):
    groups = defaultdict(list)
    for row in episodes:
        key = (str(row["task_id"]), str(row["stage"]), tuple(row["object_pair"] or []), row["synthetic"])
        groups[key].append(row)
    result = []
    for (_, _, _, synthetic), rows in sorted(groups.items()):
        outcomes = Counter(row["official_outcome"] for row in rows)
        orders = Counter(row["order_status"] for row in rows)
        known = outcomes["success"] + outcomes["failure"]
        comparable = orders["comparable"]
        success_steps = [row["total_steps"] for row in rows
                         if row["success"] is True and row["total_steps"] is not None]
        steps = [row["total_steps"] for row in rows if row["total_steps"] is not None]
        identities = {str(row["init_state_id"]) for row in rows if row["init_state_id"] is not None}
        fingerprints = {row["state_sha256"] for row in rows if row["state_sha256"] is not None}
        result.append(dict(
            task_id=rows[0]["task_id"], stage=rows[0]["stage"], object_pair=rows[0]["object_pair"],
            synthetic=synthetic, episodes=len(rows), successful_episodes=outcomes["success"],
            failed_episodes=outcomes["failure"], known_outcome_episodes=known,
            unknown_outcome_episodes=outcomes["unknown"], success_rate=outcomes["success"] / known if known else None,
            comparable_order_episodes=comparable, unknown_order_episodes=orders["unknown"],
            tied_order_episodes=orders["tied"], unreviewed_episodes=orders["unreviewed"],
            a_to_b_ratio=sum(row["pair_order"] == "A→B" for row in rows) / comparable if comparable else None,
            b_to_a_ratio=sum(row["pair_order"] == "B→A" for row in rows) / comparable if comparable else None,
            mean_successful_steps=mean(success_steps) if success_steps else None,
            successful_steps_samples=len(success_steps), known_steps_episodes=len(steps),
            min_steps=min(steps) if steps else None, median_steps=median(steps) if steps else None,
            max_steps=max(steps) if steps else None, mean_steps=mean(steps) if steps else None,
            distinct_initial_states=len(identities), distinct_initial_state_hashes=len(fingerprints),
            missing_initial_state_episodes=sum(row["init_state_id"] is None for row in rows),
            incomplete_recording_episodes=sum(not row["recording_complete"] for row in rows),
            missing_eef_episodes=sum(row["eef_path_length"] is None for row in rows),
            video_episodes=sum(row["video_present"] for row in rows),
            failed_attempts=sum(event["subtask_success"] is False for row in rows for event in row["subtask_events"]),
            unknown_attempt_results=sum(event["subtask_success"] is None for row in rows for event in row["subtask_events"]),
        ))
    return result


_EPISODE_COLUMNS = [
    ("Task ID", "task_id"), ("Init State ID", "init_state_id"), ("Episode ID", "episode_id"),
    ("Grasp Order", "grasp_order"), ("Success", "success"), ("Total Steps", "total_steps"),
    ("EEF Path Length", "eef_path_length"), ("Stage", "stage"), ("Task Name", "task_name"),
    ("Official Outcome", "official_outcome"), ("Manual Outcome", "manual_outcome"),
    ("Object Pair", "object_pair"), ("Object Reviewed", "object_reviewed"),
    ("Pair Order", "pair_order"), ("Order Status", "order_status"),
    ("Recorded Policy Steps", "recorded_policy_steps"), ("Executed Policy Steps", "executed_policy_steps"),
    ("Known Phase Policy Steps", "known_phase_policy_steps"), ("Recorded Steps", "recorded_steps"),
    ("Executed Settling Steps", "executed_settling_steps"),
    ("Recorded Settling Steps", "recorded_settling_steps"), ("Unknown Phase Steps", "phase_unknown_steps"),
    ("Step Count Complete", "step_count_complete"), ("Recording Complete", "recording_complete"),
    ("Execution Source", "execution_source"), ("Observed EEF Path Length", "observed_eef_path_length"),
    ("EEF Coverage", "eef_coverage"), ("Observed EEF Steps", "observed_eef_steps"),
    ("Missing EEF Steps", "missing_eef_steps"), ("Recorded EEF Coverage", "recorded_eef_coverage"),
    ("Status", "status"), ("Synthetic", "synthetic"), ("State SHA256", "state_sha256"),
    ("Video Present", "video_present"), ("Video Status", "video_status"), ("Video Error", "video_error"),
    ("Video Files", "video_files"), ("Video Issues", "video_issues"),
    ("Notes", "notes")]
_EVENT_COLUMNS = [
    ("Episode ID", "episode_id"), ("Object", "object"), ("Grasp Step", "grasp_step"),
    ("Place Step", "place_step"), ("Subtask Success", "subtask_success"),
    ("Task ID", "task_id"), ("Stage", "stage"), ("Attempt ID", "attempt_id"),
    ("Object Reviewed", "object_reviewed"), ("Start Frame", "start_frame"), ("End Frame", "end_frame"),
    ("Grasp Start Frame", "grasp_start_frame"), ("Grasp Frame", "grasp_frame"),
    ("Place Start Frame", "place_start_frame"), ("Place Frame", "place_frame"),
    ("Start Step", "start_step"), ("End Step", "end_step"),
    ("Grasp Start Step", "grasp_start_step"), ("Place Start Step", "place_start_step"),
    ("Grasp Operation Steps", "grasp_operation_steps"), ("Place Operation Steps", "place_operation_steps"),
    ("Attempt Steps", "attempt_steps"), ("Grasp To Place Interval Steps", "grasp_to_place_interval_steps"),
    ("Source", "source"), ("Notes", "notes")]
_SUMMARY_COLUMNS = [
    ("Task ID", "task_id"), ("Episodes", "episodes"), ("Success Rate", "success_rate"),
    ("A→B Ratio", "a_to_b_ratio"), ("B→A Ratio", "b_to_a_ratio"),
    ("Mean Successful Steps", "mean_successful_steps"), ("Stage", "stage"), ("Object Pair", "object_pair"),
    ("Synthetic", "synthetic"), ("Successful Episodes", "successful_episodes"), ("Failed Episodes", "failed_episodes"),
    ("Known Outcome Episodes", "known_outcome_episodes"), ("Unknown Outcome Episodes", "unknown_outcome_episodes"),
    ("Comparable Order Episodes", "comparable_order_episodes"), ("Unknown Order Episodes", "unknown_order_episodes"),
    ("Tied Order Episodes", "tied_order_episodes"), ("Unreviewed Episodes", "unreviewed_episodes"),
    ("Successful Steps Samples", "successful_steps_samples"), ("Known Steps Episodes", "known_steps_episodes"),
    ("Min Steps", "min_steps"), ("Median Steps", "median_steps"), ("Max Steps", "max_steps"),
    ("Mean Steps", "mean_steps"), ("Distinct Initial States", "distinct_initial_states"),
    ("Distinct Initial State Hashes", "distinct_initial_state_hashes"),
    ("Missing Initial State Episodes", "missing_initial_state_episodes"),
    ("Incomplete Recording Episodes", "incomplete_recording_episodes"), ("Missing EEF Episodes", "missing_eef_episodes"),
    ("Video Episodes", "video_episodes"), ("Failed Attempts", "failed_attempts"),
    ("Unknown Attempt Results", "unknown_attempt_results")]


def _csv_value(value):
    if value is None:
        return ""
    if type(value) is bool:
        return int(value)
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False)
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([title for title, _ in columns])
        for row in rows:
            writer.writerow([_csv_value(row.get(key)) for _, key in columns])


def _md(value):
    return str(value if value is not None else "unknown").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _report(episodes, summaries, excluded):
    lines = ["# VLA 执行行为分析报告", ""]
    synthetic = sum(row["synthetic"] for row in episodes)
    if synthetic:
        lines += [f"> **SYNTHETIC 离线演示：包含 {synthetic} 个合成回合，不是真实模型实验结果。**", ""]
    lines += [f"纳入 {len(episodes)} 个回合；默认规则下排除 {excluded} 个 synthetic 回合。按任务、阶段、对象对和数据来源分别统计。",
              "官方成功来自 manifest.outcome，人工判断另列。成功率分母仅含官方已知成功/失败，unknown 不混入失败。",
              "动作步为从 1 开始的 policy env.step，不含 settling，不按推理块计数。未知数值在 CSV 中留空。",
              "完整抓取序列保留重抓；A/B 为对象对中第一/第二个名字，比例仅比较已复核、抓取事件已知且首次先后可区分的回合。并列、未知和未复核分别列出。",
              "抓取和放置操作步数均包含两端；尝试步数为 start 至 end 包含两端。抓取至放置间隔为 place_step−grasp_step，不代表纯放置耗时。",
              "EEF 总长逐个 policy transition 计算 Σ‖next_xyz−obs_xyz‖，包含首个动作；任何应计步缺失则总长 unknown，另报已观测长度和覆盖率，图不跨缺口连线。", ""]
    lines += ["视频覆盖需非空 MP4 与同名 sidecar 同时存在，身份一致、包含 settling、首个 obs 和全部 next_obs 的映射与轨迹一致，且执行计数证明无落盘缺失。支持后续独立导出；仅有 MP4 文件不计完整覆盖，媒体解码有效性需视频验收另行确认。", ""]
    if not episodes:
        lines += ["没有可用于统计的回合，不能形成行为结论。", ""]
    for row in summaries:
        lines += [f"## 任务 {_md(row['task_id'])} / 阶段 {_md(row['stage'])}", "",
                  f"对象定义：{_md(row['object_pair'])}；来源：{'SYNTHETIC' if row['synthetic'] else '记录数据'}。",
                  f"样本 {row['episodes']}；官方成功 {row['successful_episodes']}，失败 {row['failed_episodes']}，未知 {row['unknown_outcome_episodes']}；成功率 {_md(row['success_rate'])}（分母 {row['known_outcome_episodes']}）。",
                  f"首次对象对顺序：A→B 比例 {_md(row['a_to_b_ratio'])}，B→A 比例 {_md(row['b_to_a_ratio'])}；可比较分母 {row['comparable_order_episodes']}，未知 {row['unknown_order_episodes']}，并列 {row['tied_order_episodes']}，未复核 {row['unreviewed_episodes']}。",
                  f"可信总步数样本 {row['known_steps_episodes']}；最小/中位/最大 {_md(row['min_steps'])}/{_md(row['median_steps'])}/{_md(row['max_steps'])}；平均 {_md(row['mean_steps'])}。成功回合平均步数 {_md(row['mean_successful_steps'])}（样本 {row['successful_steps_samples']}）。",
                  f"不同初态 ID {row['distinct_initial_states']}，已知不同内容指纹 {row['distinct_initial_state_hashes']}，缺初态 ID 回合 {row['missing_initial_state_episodes']}；ID 不同本身不证明初态内容互异。",
                  f"记录不完整 {row['incomplete_recording_episodes']}；EEF 总长未知 {row['missing_eef_episodes']}；视频文件覆盖 {row['video_episodes']}/{row['episodes']}。失败尝试 {row['failed_attempts']}；子任务结果未知 {row['unknown_attempt_results']}。", ""]
    lines += ["## 各初态的顺序观察", "", "|任务|阶段|初态|Episode|完整抓取序列|官方结果|总步数|", "|---|---|---|---|---|---|---|"]
    for row in episodes:
        lines.append("|" + "|".join(_md(row[key]) for key in ("task_id", "stage", "init_state_id", "episode_id", "grasp_order", "official_outcome", "total_steps")) + "|")
    video_notes = [row for row in episodes if row["video_error"] or row["video_issues"]]
    if video_notes:
        lines += ["", "## 视频问题与历史导出状态", ""]
        for row in video_notes:
            lines.append(f"- {_md(row['episode_id'])}：完整 sidecar 覆盖已验证={row['video_present']}；历史 execution 视频错误={_md(row['video_error'])}；未通过核验的文件={_md(row['video_issues'])}。")
    lines += ["", "## 解释范围与待验事项", "",
              "以上只描述已观察的顺序和轨迹差异。有限样本不能据此推断主动规划、因果关系、训练示范的影响或最优路径；初态与顺序的对应关系不等于布局导致顺序改变。",
              "正式基础实验要求同一适合的双物体任务至少 20 个不同初态，并保存完整轨迹、视频和人工顺序；此前先做 5 个不同初态试采。当前表格不自动证明真实 A100 模型验收已完成。", ""]
    return "\n".join(lines)


def export_analysis(root, output, *, object_pair=None, include_synthetic=False, plots=True) -> dict[str, Any]:
    """Export three UTF-8 CSVs, a Chinese report and optional PNG/SVG figures.

    Duplicate episode IDs are rejected before writing. Synthetic recordings are
    excluded by default and explicitly flagged and grouped separately on opt-in.
    Return data rows plus paths; no publication or model execution is performed.
    """
    root, output = Path(root), Path(output)
    if not root.exists():
        raise FileNotFoundError(root)
    paths = [root / "manifest.json"] if (root / "manifest.json").is_file() else sorted(root.rglob("manifest.json"))
    identifiers, episodes, excluded = set(), [], 0
    for manifest_path in paths:
        manifest = read_manifest(manifest_path.parent)
        identifier = manifest["episode_id"]
        if identifier in identifiers:
            raise ValueError(f"Duplicate episode ID: {identifier}")
        identifiers.add(identifier)
        if manifest["metadata"].get("synthetic") is True and not include_synthetic:
            excluded += 1
            continue
        episodes.append(analyze_episode(manifest_path.parent, object_pair=object_pair))
    summary = _summaries(episodes)
    subtasks = [event for row in episodes for event in row["subtask_events"]]
    output.mkdir(parents=True, exist_ok=True)
    outputs = {name: output / (name + ".csv") for name in
               ("episode_statistics", "subtask_events", "task_summary")}
    outputs["report"] = output / "analysis_report.md"
    artifacts = {}
    if plots:
        from .plots import export_episode_plots
        for row in episodes:
            artifacts[row["episode_id"]] = export_episode_plots(row, output / "plots")
    _write_csv(outputs["episode_statistics"], _EPISODE_COLUMNS, episodes)
    _write_csv(outputs["subtask_events"], _EVENT_COLUMNS, subtasks)
    _write_csv(outputs["task_summary"], _SUMMARY_COLUMNS, summary)
    outputs["report"].write_text(_report(episodes, summary, excluded), encoding="utf-8")
    return dict(episodes=episodes, subtasks=subtasks, task_summary=summary,
                paths=outputs, plots=artifacts, excluded_synthetic=excluded)
