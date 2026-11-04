"""Stream complete recorded transitions to MP4 with explicit frame provenance."""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from .recorder import _write_json, load_arrays, read_manifest, read_steps


def _open_writer(path, fps):
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401 - the optional executable provider
        return imageio.get_writer(str(path), format='FFMPEG', mode='I', fps=fps,
                                  codec='libx264', macro_block_size=1)
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError('MP4 encoder unavailable; install the video extra: ' + str(exc)) from exc


def _component(value):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', str(value))[:100]


def export_episode_video(episode_path, output=None, *, fps=None, include_settling=True,
                         camera='agentview_image', rotate180=True):
    """Export first obs plus all next_obs, including the final terminal image.

    A sidecar with the same stem and ``.json`` extension maps every encoded
    frame to a raw transition and a 1-based policy action step (or null).
    An encoder failure raises and never leaves a newly published MP4.
    """
    episode = Path(episode_path)
    manifest = read_manifest(episode)
    metadata = manifest['metadata']
    steps = read_steps(episode)
    selected = []
    action_step = 0
    for step in steps:
        phase = step['context'].get('phase')
        if phase == 'policy':
            action_step += 1
        elif phase != 'settling':
            raise ValueError('video requires an explicit policy/settling phase for every transition')
        if include_settling or phase == 'policy':
            selected.append((step['index'], phase, action_step if phase == 'policy' else None))
    if not selected:
        raise ValueError('episode has no selected recorded video transitions')
    control_freq = metadata.get('control_freq')
    rate = (control_freq if control_freq is not None else 20) if fps is None else fps
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('fps must be a finite positive number')
    if output is None:
        output = episode / ('task-' + _component(metadata.get('task_id', 'unknown'))
                            + '_init-' + _component(metadata.get('init_state_id', 'unknown'))
                            + '_episode-' + _component(manifest['episode_id']) + '.mp4')
    output = Path(output)
    if output.suffix.lower() != '.mp4':
        raise ValueError('video output must have an .mp4 extension')
    sidecar = output.with_suffix('.json')
    if output.exists() or sidecar.exists():
        raise FileExistsError(f'video output already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix='.' + output.stem, suffix='.mp4', dir=output.parent)
    os.close(descriptor)
    temporary = Path(temp_name)
    writer = None
    mappings = []
    shape = None
    try:
        writer = _open_writer(temporary, float(rate))
        for position, (index, phase, action_id) in enumerate(selected):
            arrays = load_arrays(episode, index)
            for observation in (('obs', 'next_obs') if position == 0 else ('next_obs',)):
                key = observation + '/' + camera
                if key not in arrays:
                    raise ValueError(f'missing camera {key} at record_step {index}')
                frame = arrays[key]
                if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                    raise ValueError('video camera must contain uint8 HxWx3 RGB images')
                if shape is None:
                    shape = frame.shape
                if frame.shape != shape:
                    raise ValueError('video image shape changed during episode')
                writer.append_data(np.ascontiguousarray(frame[::-1, ::-1] if rotate180 else frame))
                mappings.append({'video_frame': len(mappings), 'record_step': index,
                                 'action_step_id': action_id if observation == 'next_obs' else None,
                                 'phase': phase, 'observation': observation})
        writer.close()
        writer = None
        if temporary.stat().st_size == 0:
            raise RuntimeError('MP4 encoder produced an empty video')
        _write_json(sidecar, {'schema_version': 1, 'episode_id': manifest['episode_id'],
                             'task_id': metadata.get('task_id'), 'init_state_id': metadata.get('init_state_id'),
                             'playback_fps': rate, 'control_freq': control_freq,
                             'include_settling': include_settling, 'camera': camera,
                             'rotate180': rotate180, 'frames': mappings})
        os.replace(temporary, output)
        return output
    except BaseException:
        sidecar.unlink(missing_ok=True)
        raise
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            temporary.unlink(missing_ok=True)
