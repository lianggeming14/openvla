"""Exercise exported review artifacts and their downloaded annotation contract."""

import base64
import importlib
import io
import json
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


class ExportParser(HTMLParser):
    def __init__(self, content):
        super().__init__()
        self.scripts = []
        self.resources = []
        self.current = None
        self.ids = []
        self.feed(content)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "script":
            self.current = {"attrs": attrs, "text": ""}
            self.scripts.append(self.current)
        for key in ("src", "href"):
            if attrs.get(key):
                self.resources.append(attrs[key])

    def handle_endtag(self, tag):
        if tag == "script":
            self.current = None

    def handle_data(self, data):
        if self.current is not None:
            self.current["text"] += data

    @property
    def payload(self):
        return json.loads(next(s["text"] for s in self.scripts
                               if s["attrs"].get("id") == "review-data"))

    @property
    def program(self):
        return next(s["text"] for s in self.scripts
                    if s["attrs"].get("type") != "application/json")


def episode(tmp_path, *, images=True, empty=False, metadata=None):
    from vla_trajectory.recorder import EpisodeRecorder

    recorder = EpisodeRecorder(tmp_path / "episodes", metadata or {}, episode_id="review-test")
    if not empty:
        before = {}
        after = {}
        if images:
            # Distinct top/bottom colors reveal an accidental flip or rotation.
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            image[:360, :, 0] = 255
            image[360:, :, 2] = 255
            before["agentview_image"] = image
            before["robot0_eye_in_hand_image"] = np.full((20, 30, 3), 90, dtype=np.uint8)
            after["agentview_image"] = np.full((720, 1280, 3), 30, dtype=np.uint8)
        recorder.append(before, np.array([0.2, -0.4, 1.0]), after, reward=1,
                        context={"phase": "policy", "chunk_id": 2, "chunk_index": 0,
                                 "predicates_before": {"open": False},
                                 "predicates_after": {"open": True}})
    recorder.finish(outcome="unknown")
    return recorder.path


def export(path, tmp_path):
    # Import inside the test so absence is a focused failing feature test.
    review = importlib.import_module("vla_trajectory.review")
    output = tmp_path / "review.html"
    review.export_review(path, output)
    return ExportParser(output.read_text(encoding="utf-8"))


def test_export_embeds_bounded_dual_camera_previews_with_original_orientation(tmp_path):
    review = importlib.import_module("vla_trajectory.review")
    path = episode(tmp_path)
    output = tmp_path / "nested" / "review.html"
    review.export_review(path, output)
    parsed = ExportParser(output.read_text(encoding="utf-8"))
    data = parsed.payload
    assert data["manifest"]["episode_id"] == "review-test"
    frame = data["frames"][0]
    assert frame["action"] == [0.2, -0.4, 1.0]
    assert frame["context"]["predicates_after"] == {"open": True}
    assert frame["context"]["chunk_id"] == 2
    camera = frame["cameras"]["before"]["agentview_image"]
    assert camera["raw_shape"] == [720, 1280, 3]
    assert camera["orientation"] == "raw"
    with Image.open(io.BytesIO(base64.b64decode(camera["src"].split(",", 1)[1]))) as image:
        assert image.width <= 640 and image.height <= 480
        assert image.getpixel((image.width // 2, 0))[0] > 240
        assert image.getpixel((image.width // 2, image.height - 1))[2] > 240
    assert frame["cameras"]["before"]["robot0_eye_in_hand_image"]["src"].startswith("data:image/")
    assert frame["cameras"]["after"]["agentview_image"]["src"].startswith("data:image/")
    assert all(not value.startswith(("http:", "https:", "//")) for value in parsed.resources)


def test_untrusted_text_round_trips_without_creating_script_or_markup(tmp_path):
    attack = '</script><script>alert("x")</script><img src=x onerror=alert(1)>&\u2028'
    parsed = export(episode(tmp_path, metadata={"instruction": attack}), tmp_path)
    assert parsed.payload["manifest"]["metadata"]["instruction"] == attack
    assert len(parsed.scripts) == 2
    assert "x" not in parsed.resources


def test_missing_cameras_and_empty_episode_export(tmp_path):
    parsed = export(episode(tmp_path, images=False), tmp_path)
    assert parsed.payload["frames"][0]["cameras"]["before"]["agentview_image"]["src"] is None
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    parsed = export(episode(empty_dir, empty=True), empty_dir)
    assert parsed.payload["frames"] == []
    assert parsed.payload["annotations"]["segments"] == []


def test_export_preserves_existing_manual_annotation_and_revision(tmp_path):
    from vla_trajectory.annotations import load_annotations, save_annotations

    path = episode(tmp_path)
    annotation = load_annotations(path)
    annotation["segments"] = [{"start": 0, "end": 0, "label": "打开", "notes": "检查"}]
    annotation["notes"] = "人工备注"
    saved = save_annotations(path, annotation, expected_revision=0)
    parsed = export(path, tmp_path)
    assert parsed.payload["annotations"]["revision"] == saved["revision"] == 1
    assert parsed.payload["annotations"]["notes"] == "人工备注"
    assert parsed.payload["annotations"]["segments"][0]["label"] == "打开"


def run_review_script(parsed, scenario):
    """Execute the actual exported program against a small, strict DOM boundary.

    Node is an optional developer check, never a runtime package dependency.
    The stub forbids unsafe markup writes and captures real Blob downloads.
    """
    node = shutil.which("node")
    if node is None:
        candidate = Path("C:/Users/robot/.conda/envs/direct/node.exe")
        node = str(candidate) if candidate.is_file() else None
    if node is None:
        pytest.skip("Node is unavailable for optional exported JavaScript execution")
    harness = r'''
const fs = require('fs'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
class Element {
  constructor() { this.children = []; this.value = ''; this.checked = false;
    this.disabled = false; this.hidden = false; this.style = {}; this.listeners = {}; }
  set innerHTML(_) { throw Error('Unsafe HTML insertion'); }
  set textContent(value) { this.text = String(value); this.children = []; }
  get textContent() { return this.text || ''; }
  appendChild(child) { this.children.push(child); return child; }
  append(...children) { children.forEach(child => this.appendChild(child)); }
  replaceChildren(...children) { this.children = [...children]; }
  addEventListener(type, callback) { this.listeners[type] = callback; }
  removeAttribute(key) { delete this[key]; }
  setAttribute(key, value) { this[key] = value; }
  click() { if (!this.disabled && this.listeners.click) this.listeners.click({target: this}); }
  dispatch(type) { if (this.listeners[type]) this.listeners[type]({target: this}); }
  remove() {}
}
const elements = Object.fromEntries(input.ids.map(id => [id, new Element()]));
elements['review-data'].textContent = JSON.stringify(input.data);
let blob = null;
const timerCallbacks = [];
const sandbox = {
 document: {getElementById: id => {if (!elements[id]) throw Error('Unknown id ' + id); return elements[id];},
   createElement: () => new Element(), body: new Element()},
 Blob, URL: {createObjectURL: b => {blob = b; return 'blob:local';}, revokeObjectURL: () => {}},
 timerCallbacks, setTimeout: callback => {timerCallbacks.push(callback); return timerCallbacks.length;},
};
vm.createContext(sandbox);
vm.runInContext(input.program, sandbox);
vm.runInContext(input.scenario, sandbox);
(async () => console.log(JSON.stringify({
 downloaded: blob ? JSON.parse(await blob.text()) : null,
 error: elements['save-status'].textContent,
 frame: elements['frame-label'].textContent,
 scrubDisabled: elements['frame-slider'].disabled,
 addDisabled: elements['add-segment'].disabled,
 cameraSource: elements['camera-agentview_image'].src || null,
})))().catch(error => {console.error(error); process.exit(1);});
'''
    result = subprocess.run([node, "-e", harness], input=json.dumps({
        "ids": parsed.ids, "data": parsed.payload,
        "program": parsed.program, "scenario": scenario,
    }), text=True, encoding="utf-8", capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_review_script_downloads_edits_and_base_revision_without_mutating_evidence(tmp_path):
    from vla_trajectory.annotations import load_annotations, save_annotations

    path = episode(tmp_path)
    save_annotations(path, load_annotations(path), expected_revision=0)
    parsed = export(path, tmp_path)
    result = run_review_script(parsed, r'''
document.getElementById('add-segment').click();
segmentRows[0].label.value = '</script><img onerror=alert(1)>';
segmentRows[0].notes.value = '人工区间';
document.getElementById('outcome').value = 'failure';
document.getElementById('episode-notes').value = '人工备注';
document.getElementById('show-after').checked = true;
document.getElementById('show-after').dispatch('change');
document.getElementById('download').click();
''')
    downloaded = result["downloaded"]
    assert downloaded["base_revision"] == 1
    assert downloaded["episode_id"] == "review-test"
    assert downloaded["events"] == parsed.payload["annotations"]["events"]
    assert downloaded["segments"] == [{"start": 0, "end": 0,
        "label": "</script><img onerror=alert(1)>", "notes": "人工区间"}]
    assert downloaded["outcome"] == "failure"
    assert downloaded["notes"] == "人工备注"
    assert result["cameraSource"] == parsed.payload["frames"][0]["cameras"]["after"]["agentview_image"]["src"]


@pytest.mark.parametrize("start,end", [(0, 1), (-1, 0), (0.5, 0), ("", 0)])
def test_review_script_blocks_invalid_inclusive_frame_range(tmp_path, start, end):
    parsed = export(episode(tmp_path, images=False), tmp_path)
    result = run_review_script(parsed, "document.getElementById('add-segment').click();\n"
        f"segmentRows[0].start.value = {json.dumps(str(start))};\n"
        f"segmentRows[0].end.value = {json.dumps(str(end))};\n"
        "segmentRows[0].label.value = '范围';\n"
        "document.getElementById('download').click();")
    assert result["downloaded"] is None
    assert result["error"]


def test_review_script_handles_empty_episode_and_downloads_no_segments(tmp_path):
    parsed = export(episode(tmp_path, empty=True), tmp_path)
    result = run_review_script(parsed, "document.getElementById('download').click();")
    assert result["scrubDisabled"] and result["addDisabled"]
    assert result["downloaded"]["segments"] == []
    assert result["downloaded"]["num_steps"] == 0


@pytest.mark.parametrize("relative", ["manifest.json", "annotations.json", "steps/000000.json", "steps/000000.npz"])
def test_export_cannot_overwrite_original_episode_files(tmp_path, relative):
    from vla_trajectory.annotations import load_annotations, save_annotations
    from vla_trajectory.review import export_review

    path = episode(tmp_path, images=False)
    save_annotations(path, load_annotations(path))
    destination = path / relative
    original = destination.read_bytes()
    with pytest.raises(ValueError, match="episode|轨迹|source|original"):
        export_review(path, destination)
    assert destination.read_bytes() == original


@pytest.mark.parametrize("relative,original", [
    ("execution.json", b'{"executed_policy_steps": 7}\r\n'),
    ("videos/task-init-episode.mp4", b"\x00\x00\x00\x18ftypmp42\x00original-video"),
    ("videos/task-init-episode.json", b'{"frames": [{"record_step": 0}]}\n'),
    ("notes.txt", b"original non-HTML evidence\r\n"),
])
def test_review_rejects_non_html_evidence_without_changing_its_bytes(tmp_path, relative, original):
    from vla_trajectory.review import export_review

    path = episode(tmp_path, images=False)
    destination = path / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(original)
    with pytest.raises(ValueError, match="HTML|html"):
        export_review(path, destination)
    assert destination.read_bytes() == original
    assert not list(destination.parent.glob(".review-*.tmp"))


@pytest.mark.parametrize("suffix", [".html", ".htm", ".HTML", ".HtM"])
def test_review_allows_html_export_and_refresh_with_case_insensitive_suffix(tmp_path, suffix):
    from vla_trajectory.annotations import load_annotations, save_annotations
    from vla_trajectory.review import export_review

    path = episode(tmp_path, images=False)
    destination = tmp_path / "exports" / ("review" + suffix)
    assert export_review(path, destination) == destination
    original = destination.read_bytes()
    assert original.startswith(b"<!doctype html>")
    annotation = load_annotations(path)
    annotation["notes"] = "Updated HTML review"
    save_annotations(path, annotation)
    assert export_review(path, destination) == destination
    updated = destination.read_bytes()
    assert updated != original
    assert ExportParser(updated.decode("utf-8")).payload["annotations"]["notes"] == annotation["notes"]


def test_review_script_navigation_updates_the_selected_transition(tmp_path):
    from vla_trajectory.recorder import EpisodeRecorder

    recorder = EpisodeRecorder(tmp_path / "episodes", {}, episode_id="two-frames")
    recorder.append({}, np.array([1]), {}, context={"phase": "settling"})
    recorder.append({}, np.array([2]), {}, context={"phase": "policy"})
    recorder.finish()
    parsed = export(recorder.path, tmp_path)
    result = run_review_script(parsed, r'''
document.getElementById('next').click();
if (document.getElementById('action').textContent !== '[\n  2\n]') throw Error('Action did not follow frame');
document.getElementById('add-segment').click();
segmentRows[0].label.value = 'second';
document.getElementById('download').click();
''')
    assert result["downloaded"]["segments"] == [{"start": 1, "end": 1, "label": "second", "notes": ""}]


def test_unusual_image_shapes_fail_gracefully_and_float_previews_are_scaled(tmp_path):
    from vla_trajectory.recorder import EpisodeRecorder

    recorder = EpisodeRecorder(tmp_path / "episodes", {}, episode_id="image-formats")
    recorder.append({"agentview_image": np.full((5, 10, 1), 0.5, dtype=np.float32),
                     "robot0_eye_in_hand_image": np.zeros((3, 4, 5), dtype=np.uint8)},
                    np.array([0]), {})
    recorder.finish()
    parsed = export(recorder.path, tmp_path)
    cameras = parsed.payload["frames"][0]["cameras"]["before"]
    assert cameras["robot0_eye_in_hand_image"]["src"] is None
    camera = cameras["agentview_image"]
    with Image.open(io.BytesIO(base64.b64decode(camera["src"].split(",", 1)[1]))) as image:
        assert image.getpixel((0, 0)) == (128, 128, 128)


def test_export_rejects_episode_that_changes_during_snapshot(tmp_path, monkeypatch):
    from vla_trajectory.recorder import EpisodeRecorder
    from vla_trajectory import review

    recorder = EpisodeRecorder(tmp_path / "episodes", {}, episode_id="still-recording")
    recorder.append({}, np.array([1]), {})
    read_steps = review.read_steps

    def append_before_reading(path):
        recorder.append({}, np.array([2]), {})
        return read_steps(path)

    monkeypatch.setattr(review, "read_steps", append_before_reading)
    output = tmp_path / "review.html"
    with pytest.raises(ValueError, match="changed|snapshot"):
        review.export_review(recorder.path, output)
    assert not output.exists()


def attempt_episode(tmp_path):
    from vla_trajectory.recorder import EpisodeRecorder
    recorder = EpisodeRecorder(tmp_path / "episodes", {}, episode_id="attempt-review")
    for index in range(7):
        recorder.append({}, np.array([index]), {},
                        context={"phase": "settling" if index == 0 else "policy"})
    recorder.finish(outcome="success")
    return recorder.path


def test_review_v2_maps_action_steps_and_downloads_attempts_for_real_import(tmp_path):
    from vla_trajectory.annotations import save_annotations, load_annotations
    path = attempt_episode(tmp_path)
    parsed = export(path, tmp_path)
    assert [frame["action_step"] for frame in parsed.payload["frames"]] == [None, 1, 2, 3, 4, 5, 6]
    result = run_review_script(parsed, r'''
navigate(1);
document.getElementById('add-attempt').click();
attemptRows[0].object.value = '汤罐';
attemptRows[0].start_frame.value = '1';
attemptRows[0].grasp_start_frame.value = '1';
navigate(2); attemptRows[0].setButtons.grasp_frame.click();
attemptRows[0].place_start_frame.value = '4';
navigate(5); attemptRows[0].setButtons.place_frame.click();
attemptRows[0].end_frame.value = '6';
attemptRows[0].subtask_success.value = 'false';
document.getElementById('add-attempt').click();
attemptRows[1].object.value = '汤罐';
attemptRows[1].start_frame.value = '';
attemptRows[1].end_frame.value = '';
document.getElementById('object-reviewed').checked = true;
document.getElementById('download').click();
''')
    downloaded = result["downloaded"]
    assert downloaded["schema_version"] == 2 and downloaded["object_reviewed"] is True
    first, second = downloaded["object_attempts"]
    assert first["object"] == second["object"] == "汤罐"
    assert [first[field] for field in ("start_frame", "grasp_start_frame", "grasp_frame",
                                     "place_start_frame", "place_frame", "end_frame")] == [1, 1, 2, 4, 5, 6]
    assert first["subtask_success"] is False and second["subtask_success"] is None
    assert first["attempt_id"] != second["attempt_id"]
    assert "Action Step 5" in result["frame"]
    assert downloaded["events"] == parsed.payload["annotations"]["events"]
    saved = save_annotations(path, downloaded)
    assert load_annotations(path) == saved


def test_review_attempt_rejects_invalid_action_step_and_can_mark_empty_review_complete(tmp_path):
    parsed = export(attempt_episode(tmp_path), tmp_path)
    result = run_review_script(parsed, r'''
document.getElementById('add-attempt').click();
attemptRows[0].grasp_frame.value = '0';
document.getElementById('download').click();
''')
    assert result["downloaded"] is None and result["error"]
    result = run_review_script(parsed, r'''
document.getElementById('object-reviewed').checked = true;
document.getElementById('download').click();
''')
    assert result["downloaded"]["object_attempts"] == []
    assert result["downloaded"]["object_reviewed"] is True


def test_review_continuous_playback_advances_and_pause_stops_pending_timer(tmp_path):
    parsed = export(attempt_episode(tmp_path), tmp_path)
    result = run_review_script(parsed, r'''
document.getElementById('play-fps').value = '10';
document.getElementById('play-pause').click();
timerCallbacks.shift()();
if (currentFrame !== 1) throw Error('Playback did not advance');
document.getElementById('play-pause').click();
timerCallbacks.shift()();
if (currentFrame !== 1) throw Error('Pause did not stop playback');
''')
    assert "Action Step 1" in result["frame"]


def test_review_unknown_phase_blocks_later_action_numbering_and_preserves_raw_attempt(tmp_path):
    from vla_trajectory.recorder import EpisodeRecorder
    from vla_trajectory.annotations import load_annotations, save_annotations
    recorder = EpisodeRecorder(tmp_path / "episodes", {}, episode_id="unknown-phase")
    for phase in ("policy", None, "policy"):
        recorder.append({}, np.array([0]), {}, context={} if phase is None else {"phase": phase})
    recorder.finish()
    annotation = load_annotations(recorder.path)
    annotation["object_attempts"] = [{"attempt_id": "old-attempt", "object": None,
        "start_frame": None, "end_frame": None, "grasp_start_frame": None,
        "grasp_frame": 2, "place_start_frame": None, "place_frame": None,
        "subtask_success": None, "source": "human_review", "notes": "旧记录"}]
    save_annotations(recorder.path, annotation)
    parsed = export(recorder.path, tmp_path)
    assert [frame["action_step"] for frame in parsed.payload["frames"]] == [1, None, None]
    result = run_review_script(parsed, r'''
if (attemptRows[0].units.value !== 'frame') throw Error('Unknown mapping must use raw frame');
navigate(2);
document.getElementById('download').click();
''')
    assert result["downloaded"]["object_attempts"] == annotation["object_attempts"]
    assert "Action Step 未知" in result["frame"]


def test_review_raw_settling_grasp_and_reversed_operation_boundaries_are_rejected(tmp_path):
    parsed = export(attempt_episode(tmp_path), tmp_path)
    for scenario in (r'''
document.getElementById('add-attempt').click();
attemptRows[0].units.value = 'frame';
attemptRows[0].grasp_frame.value = '0';
''', r'''
document.getElementById('add-attempt').click();
attemptRows[0].grasp_start_frame.value = '3';
attemptRows[0].grasp_frame.value = '2';
'''):
        result = run_review_script(parsed, scenario + "document.getElementById('download').click();")
        assert result["downloaded"] is None and result["error"]
