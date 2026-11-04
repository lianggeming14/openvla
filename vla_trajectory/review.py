"""Export a self-contained, offline trajectory and annotation review page."""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .annotations import load_annotations
from .recorder import load_arrays, read_manifest, read_steps


_CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
_PREVIEW_SIZE = (640, 480)


def _preview(array: np.ndarray | None, source_key: str) -> dict:
    """Encode a bounded preview without changing the stored camera orientation."""
    preview = {"src": None, "source_key": source_key, "raw_shape": None,
               "orientation": "raw", "message": "此帧没有该相机图像"}
    if array is None:
        return preview
    preview["raw_shape"] = list(array.shape)
    preview["raw_dtype"] = str(array.dtype)
    if (array.ndim not in (2, 3) or min(array.shape[:2], default=0) == 0
            or (array.ndim == 3 and array.shape[2] not in (1, 3, 4))):
        preview["message"] = "无法预览：需要 H×W、H×W×1、H×W×3 或 H×W×4 图像"
        return preview
    if array.dtype.kind not in "buif" or not np.isfinite(array).all():
        preview["message"] = "无法预览：图像必须是有限数值数组"
        return preview
    if array.dtype == np.uint8:
        pixels = array
        conversion = "uint8 原始像素"
    elif array.dtype.kind == "b" or (array.dtype.kind == "f" and array.min() >= 0 and array.max() <= 1):
        pixels = np.rint(array * 255).astype(np.uint8)
        conversion = "[0, 1] 像素缩放到 [0, 255]"
    else:
        pixels = np.clip(array, 0, 255).astype(np.uint8)
        conversion = "像素裁剪到 [0, 255] 并转换为 uint8"
    if pixels.ndim == 3 and pixels.shape[2] == 1:
        pixels = pixels[:, :, 0]
    image = Image.fromarray(pixels).convert("RGB")
    image.thumbnail(_PREVIEW_SIZE, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    preview.update(src="data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
                   preview_shape=[image.height, image.width, 3], conversion=conversion,
                   message="保持原始数组方向；未旋转或翻转。JPEG 仅用于审阅，原始数组保持不变。")
    return preview


def _safe_json(value: object) -> str:
    """JSON in a script element must never contain HTML delimiters."""
    serialized = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return (serialized.replace("&", "\\u0026").replace("<", "\\u003c")
            .replace(">", "\\u003e").replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029"))


def export_review(episode_path: str | Path, output_html: str | Path) -> Path:
    """Write an offline HTML review; editing it never writes to episode storage.

    Downloaded annotations include the original ``base_revision`` so the CLI
    can reject stale edits when importing them. Images are embedded JPEGs with
    maximum dimensions 640×480; raw observations and automatic evidence remain
    in their original files.
    """
    episode_path = Path(episode_path)
    output_html = Path(output_html)
    episode_root, destination = episode_path.resolve(), output_html.resolve()
    protected = {episode_root / "manifest.json", episode_root / "annotations.json"}
    if destination in protected or (episode_root / "steps") in destination.parents:
        raise ValueError("review output cannot overwrite original episode files")
    manifest = read_manifest(episode_path)
    annotations = load_annotations(episode_path)
    frames = []
    for step in read_steps(episode_path):
        arrays = load_arrays(episode_path, step["index"])
        frames.append({
            "index": step["index"], "reward": step["reward"],
            "terminated": step["terminated"], "truncated": step["truncated"],
            "context": step.get("context", {}), "action": arrays["action"].tolist(),
            "cameras": {
                phase: {name: _preview(arrays.get(f"{prefix}/{name}"), f"{prefix}/{name}")
                        for name in _CAMERAS}
                for phase, prefix in (("before", "obs"), ("after", "next_obs"))
            },
        })
    if (len(frames) != manifest["num_steps"]
            or annotations["num_steps"] != manifest["num_steps"]
            or read_manifest(episode_path) != manifest):
        raise ValueError("episode changed during review snapshot; retry after recording has stopped")
    payload = {"manifest": manifest, "annotations": annotations, "frames": frames}
    html = _HTML.replace("__REVIEW_DATA__", _safe_json(payload))
    output_html.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=output_html.parent, prefix=".review-", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(html)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_html)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return output_html


_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VLA 轨迹审阅</title>
<style>
:root{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#172637;background:#f2f5f8;line-height:1.5}
*{box-sizing:border-box}body{margin:0}main{max-width:1440px;margin:auto;padding:24px}
header{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin-bottom:20px}
h1{font-size:26px;margin:0 0 6px}h2{font-size:18px;margin:0 0 14px}h3{font-size:15px;margin:0}
p{margin:8px 0}.muted,small{color:#526778}.badge{display:inline-block;background:#d9eee9;color:#155b50;padding:5px 12px;border-radius:20px;white-space:nowrap}
.panel{background:white;border:1px solid #dce3e9;border-radius:12px;padding:20px;margin-bottom:18px}
.notice{border-left:4px solid #238474;background:#eef7f4;padding:12px 16px;font-size:14px}
.cameras,.columns{display:grid;grid-template-columns:1fr 1fr;gap:18px}.camera{min-width:0}
.image-wrap{margin-top:8px;background:#101d29;aspect-ratio:4/3;border-radius:8px;display:flex;align-items:center;justify-content:center;overflow:hidden;color:#c6d3dd}
.image-wrap img{width:100%;height:100%;object-fit:contain}.image-wrap p{padding:24px}.image-info{font-size:12px;overflow-wrap:anywhere;margin-top:8px}
.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:16px}.controls input[type=range]{flex:1;min-width:160px}
button{border:1px solid #bac9d5;border-radius:6px;background:#fff;color:#243e50;padding:7px 12px;font:inherit;cursor:pointer}
button:hover{background:#edf4f7}button:disabled{cursor:default;opacity:.45}.primary{background:#166c60;color:white;border-color:#166c60}.primary:hover{background:#11564c}
input,select,textarea{font:inherit;border:1px solid #b9c8d3;border-radius:5px;padding:7px;color:#172637;background:#fff;max-width:100%}
input[type=number]{width:82px}input[type=checkbox]{accent-color:#166c60}textarea{width:100%;resize:vertical;min-height:68px}label{font-size:14px}
pre{background:#f3f6f8;padding:12px;border-radius:6px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:250px;overflow:auto;font-size:12px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #e1e7ec;padding:8px;text-align:left;overflow-wrap:anywhere}
.segment{border:1px solid #dce3e9;border-radius:8px;padding:12px;margin:12px 0}.segment-fields{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:8px}.segment-label{flex:1;min-width:120px}
.segment-tools{display:flex;gap:6px;margin-top:6px;flex-wrap:wrap}.segment-tools button{font-size:12px}.stack{display:grid;gap:10px}.status{min-height:24px;font-size:14px;color:#8f3426;overflow-wrap:anywhere}
summary{cursor:pointer;font-weight:600}code{font-size:12px;overflow-wrap:anywhere}.wide{grid-column:1/-1}
@media(max-width:850px){main{padding:14px}.cameras,.columns{grid-template-columns:1fr}header{flex-direction:column;gap:8px}.panel{padding:14px}}
</style>
</head>
<body><main>
<header><div><h1>VLA 轨迹审阅</h1><div id="episode-heading" class="muted"></div></div><span class="badge">离线 · 本地审阅</span></header>
<div class="panel notice"><strong>下载后仍需通过 CLI 导入。</strong>此页面完全离线，不连接服务器；修改只保存在当前页面内存中，关闭或刷新页面会丢失未下载的修改。下载 JSON 后请运行 CLI 的 import 命令导入对应 episode，才能保存标注。原始轨迹和自动谓词证据不会被页面修改。</div>
<section class="panel" aria-label="帧和相机">
<div class="controls"><button id="previous" type="button">← 上一帧</button><input id="frame-slider" aria-label="轨迹帧" type="range" min="0" value="0" step="1"><button id="next" type="button">下一帧 →</button><label>跳转 <input id="frame-number" type="number" min="0" value="0" step="1"></label><strong id="frame-label"></strong><label><input id="show-after" type="checkbox">显示动作执行后的观测</label></div>
<p class="muted" id="frame-context"></p>
<div class="cameras">
<div class="camera"><h3>场景相机 · agentview</h3><div class="image-wrap"><img id="camera-agentview_image" alt="场景相机预览" hidden><p id="missing-agentview_image"></p></div><p id="info-agentview_image" class="image-info muted"></p></div>
<div class="camera"><h3>腕部相机 · eye in hand</h3><div class="image-wrap"><img id="camera-robot0_eye_in_hand_image" alt="腕部相机预览" hidden><p id="missing-robot0_eye_in_hand_image"></p></div><p id="info-robot0_eye_in_hand_image" class="image-info muted"></p></div>
</div>
</section>
<div class="columns">
<section class="panel"><h2>动作与谓词证据</h2><p class="muted">帧编号对应一次 transition。区间的起止编号均包含在内；after 图像仍属于当前 transition。</p>
<h3>实际执行的动作</h3><pre id="action"></pre>
<h3>当前帧谓词</h3><table><thead><tr><th>谓词</th><th>动作前</th><th>动作后</th></tr></thead><tbody id="predicates"></tbody></table>
<p id="predicate-note" class="muted"></p><h3>当前帧自动事件</h3><pre id="events"></pre>
<details><summary>上下文与分块信息</summary><pre id="context"></pre></details>
<p class="muted">自动事件记录观测到的真值变化；缺失证据表示未知。事件先后不表示因果关系或任务必需顺序。</p></section>
<section class="panel"><h2>人工标注</h2><p class="muted">区间使用从 0 开始的帧编号。可重叠；每个区间都需要标签。</p><div id="segments"></div><button id="add-segment" type="button">＋ 添加当前帧区间</button>
<hr style="border:0;border-top:1px solid #e1e7ec;margin:20px 0">
<div class="stack"><label>人工审阅结果 <select id="outcome"><option value="unknown">未知 / 尚未确定</option><option value="success">成功</option><option value="failure">失败</option></select></label><label for="episode-notes">Episode 备注</label><textarea id="episode-notes" placeholder="记录判断依据或需要回看的问题"></textarea><button class="primary" id="download" type="button">下载标注 JSON</button><div id="save-status" class="status" role="status" aria-live="polite"></div></div>
<p class="muted" id="revision"></p><code id="import-command"></code></section>
<section class="panel wide"><details><summary>Episode 元数据与录制状态</summary><pre id="manifest"></pre></details></section>
</div></main>
<script id="review-data" type="application/json">__REVIEW_DATA__</script>
<script>
'use strict';
const data = JSON.parse(document.getElementById('review-data').textContent);
const $ = id => document.getElementById(id);
const frames = data.frames;
const original = data.annotations;
let currentFrame = 0;
const segmentRows = [];
const pretty = value => JSON.stringify(value, null, 2);
const truth = value => value === true ? '真' : value === false ? '假' : '未知';
function textElement(tag, text) { const node = document.createElement(tag); node.textContent = text; return node; }
function renderFrame() {
  const frame = frames[currentFrame];
  const phase = $('show-after').checked ? 'after' : 'before';
  $('frame-label').textContent = frame ? `帧 ${currentFrame} / ${frames.length - 1}` : '无已提交帧';
  $('frame-slider').value = String(currentFrame); $('frame-number').value = String(currentFrame);
  $('previous').disabled = !frame || currentFrame === 0;
  $('next').disabled = !frame || currentFrame === frames.length - 1;
  $('frame-context').textContent = frame ? `观测：${phase === 'after' ? '动作后 next_obs' : '动作前 obs'} · 阶段 ${frame.context.phase || '未知'} · reward ${frame.reward} · terminated ${frame.terminated} · truncated ${frame.truncated}` : '该 episode 没有已提交的 transition，仍可审阅结果与备注。';
  for (const camera of ['agentview_image', 'robot0_eye_in_hand_image']) {
    const preview = frame ? frame.cameras[phase][camera] : null;
    const image = $('camera-' + camera), missing = $('missing-' + camera);
    image.hidden = !preview || !preview.src;
    missing.hidden = !!(preview && preview.src);
    if (preview && preview.src) image.src = preview.src;
    else image.removeAttribute('src');
    missing.textContent = preview ? preview.message : '没有可显示的帧';
    $('info-' + camera).textContent = preview ? `${preview.source_key} · 原始形状 ${pretty(preview.raw_shape)} · ${preview.raw_dtype || ''} · ${preview.message}${preview.conversion ? ' ' + preview.conversion : ''}` : '';
  }
  $('action').textContent = frame ? pretty(frame.action) : '无动作';
  $('context').textContent = frame ? pretty(frame.context) : '{}';
  $('events').textContent = pretty(frame ? original.events.filter(event => event.frame === frame.index) : []);
  const before = frame ? frame.context.predicates_before || {} : {};
  const after = frame ? frame.context.predicates_after || {} : {};
  const keys = [...new Set([...Object.keys(before), ...Object.keys(after)])].sort();
  $('predicates').replaceChildren();
  for (const key of keys) {
    const row = document.createElement('tr');
    row.append(textElement('td', key), textElement('td', truth(before[key])), textElement('td', truth(after[key])));
    $('predicates').appendChild(row);
  }
  $('predicate-note').textContent = keys.length ? '未知：当前观测没有可用的谓词真值。' : '当前帧没有谓词证据，不能据此推断成功或失败。';
}
function navigate(value) {
  const number = Number(value);
  if (frames.length && Number.isFinite(number)) currentFrame = Math.max(0, Math.min(frames.length - 1, Math.trunc(number)));
  renderFrame();
}
function addSegment(segment) {
  if (!frames.length) return;
  const row = document.createElement('div'); row.className = 'segment';
  const fields = document.createElement('div'); fields.className = 'segment-fields';
  const start = document.createElement('input'), end = document.createElement('input');
  for (const input of [start, end]) { input.type = 'number'; input.min = '0'; input.max = String(frames.length - 1); input.step = '1'; }
  start.value = String(segment.start); end.value = String(segment.end);
  const startLabel = textElement('label', '起始 '), endLabel = textElement('label', '结束 ');
  startLabel.appendChild(start); endLabel.appendChild(end);
  const label = document.createElement('input'); label.type = 'text'; label.value = segment.label; label.placeholder = '区间标签（必填）'; label.className = 'segment-label'; label.setAttribute('aria-label', '区间标签');
  fields.append(startLabel, endLabel, label);
  const notes = document.createElement('textarea'); notes.value = segment.notes; notes.placeholder = '区间备注'; notes.setAttribute('aria-label', '区间备注');
  const tools = document.createElement('div'); tools.className = 'segment-tools';
  const setStart = textElement('button', '当前帧 → 起始'), setEnd = textElement('button', '当前帧 → 结束'), remove = textElement('button', '删除区间');
  for (const button of [setStart, setEnd, remove]) button.type = 'button';
  const record = {row, start, end, label, notes};
  setStart.addEventListener('click', () => { start.value = String(currentFrame); });
  setEnd.addEventListener('click', () => { end.value = String(currentFrame); });
  remove.addEventListener('click', () => { segmentRows.splice(segmentRows.indexOf(record), 1); row.remove(); });
  tools.append(setStart, setEnd, remove); row.append(fields, notes, tools);
  segmentRows.push(record); $('segments').appendChild(row);
}
function buildReviewDocument() {
  const segments = segmentRows.map((row, index) => {
    const startText = String(row.start.value).trim(), endText = String(row.end.value).trim();
    const start = Number(startText), end = Number(endText);
    if (!startText || !endText || !Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start || end >= frames.length)
      throw new Error(`区间 ${index + 1}：起止帧必须是 0 到 ${frames.length - 1} 之间的整数，且起始不晚于结束。`);
    const label = String(row.label.value), notes = String(row.notes.value);
    if (!label.trim()) throw new Error(`区间 ${index + 1}：请填写标签。`);
    return {start, end, label, notes};
  });
  const outcome = $('outcome').value;
  if (!['success', 'failure', 'unknown'].includes(outcome)) throw new Error('请选择有效的审阅结果。');
  const document = JSON.parse(JSON.stringify(original));
  document.base_revision = original.revision;
  document.segments = segments; document.outcome = outcome; document.notes = String($('episode-notes').value);
  return document;
}
$('previous').addEventListener('click', () => navigate(currentFrame - 1));
$('next').addEventListener('click', () => navigate(currentFrame + 1));
$('frame-slider').addEventListener('input', event => navigate(event.target.value));
$('frame-number').addEventListener('change', event => navigate(event.target.value));
$('show-after').addEventListener('change', renderFrame);
$('add-segment').addEventListener('click', () => addSegment({start: currentFrame, end: currentFrame, label: '', notes: ''}));
$('download').addEventListener('click', () => {
  try {
    const annotation = buildReviewDocument();
    const blob = new Blob([pretty(annotation) + '\n'], {type: 'application/json;charset=utf-8'});
    const url = URL.createObjectURL(blob), link = document.createElement('a');
    link.href = url; link.download = String(data.manifest.episode_id).replace(/[^a-zA-Z0-9_-]/g, '_') + '-annotations.json';
    document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    $('save-status').textContent = '已发起 JSON 下载。请通过 CLI 导入对应 episode；浏览器下载不会直接保存到轨迹目录。';
  } catch (error) { $('save-status').textContent = error.message; }
});
$('episode-heading').textContent = `${data.manifest.episode_id} · ${frames.length} 帧 · 录制状态 ${data.manifest.status} · 原始结果 ${data.manifest.outcome}`;
$('manifest').textContent = pretty(data.manifest);
$('frame-slider').max = String(Math.max(0, frames.length - 1)); $('frame-number').max = $('frame-slider').max;
for (const id of ['frame-slider', 'frame-number', 'show-after', 'add-segment']) $(id).disabled = !frames.length;
$('outcome').value = original.outcome; $('episode-notes').value = original.notes;
$('revision').textContent = `本页基于标注修订号 ${original.revision}。若其他审阅已保存，请重新导出页面，避免覆盖新修订。`;
$('import-command').textContent = 'vla-trace import <episode目录> <下载的JSON文件>';
for (const segment of original.segments) addSegment(segment);
renderFrame();
</script>
</body></html>
'''
