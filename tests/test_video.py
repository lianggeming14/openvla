import json

import numpy as np
import pytest

from vla_trajectory.recorder import EpisodeRecorder


def episode(tmp_path):
    rec = EpisodeRecorder(tmp_path, {'task_id': 3, 'init_state_id': 'S004', 'control_freq': 20}, 'E001')
    for i, phase in enumerate(['settling', 'policy', 'policy']):
        image = np.full((16, 16, 3), i * 50, dtype=np.uint8)
        nxt = np.full((16, 16, 3), (i + 1) * 50, dtype=np.uint8)
        rec.append({'agentview_image': image}, [0], {'agentview_image': nxt},
                   context={'phase': phase, 'action_step_id': None if i == 0 else i})
    rec.finish('success')
    return rec.path


class Sink:
    def __init__(self, path):
        self.path, self.frames = path, []
    def append_data(self, frame):
        self.frames.append(frame.copy())
    def close(self):
        self.path.write_bytes(b'encoded-video')


def test_video_includes_settling_and_terminal_next_obs(tmp_path, monkeypatch):
    from vla_trajectory import video
    ep = episode(tmp_path)
    writers = []
    def writer(path, fps):
        sink = Sink(path)
        writers.append(sink)
        return sink
    monkeypatch.setattr(video, '_open_writer', writer)
    output = video.export_episode_video(ep, fps=8)
    assert output.name == 'task-3_init-S004_episode-E001.mp4'
    assert [int(f[0, 0, 0]) for f in writers[0].frames] == [0, 50, 100, 150]
    sidecar = json.loads(output.with_suffix('.json').read_text())
    assert sidecar['playback_fps'] == 8
    assert sidecar['control_freq'] == 20
    assert [(f['record_step'], f['action_step_id'], f['observation']) for f in sidecar['frames']] == [
        (0, None, 'obs'), (0, None, 'next_obs'), (1, 1, 'next_obs'), (2, 2, 'next_obs')]
    policy = video.export_episode_video(ep, ep / 'policy.mp4', include_settling=False)
    assert [int(f[0, 0, 0]) for f in writers[1].frames] == [50, 100, 150]
    assert len(json.loads(policy.with_suffix('.json').read_text())['frames']) == 3


def test_video_rotation_and_codec_failure_are_explicit(tmp_path, monkeypatch):
    from vla_trajectory import video
    ep = episode(tmp_path)
    def fail(path, fps):
        raise RuntimeError('encoder unavailable')
    monkeypatch.setattr(video, '_open_writer', fail)
    with pytest.raises(RuntimeError, match='encoder unavailable'):
        video.export_episode_video(ep)
    assert not list(ep.glob('*.mp4'))
    assert not list(ep.glob('task-*.json'))


def test_rotation_uses_raw_rgb_orientation_and_missing_frame_aborts(tmp_path, monkeypatch):
    from vla_trajectory import video
    rec = EpisodeRecorder(tmp_path, {}, 'rotation')
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]
    rec.append({'agentview_image': image}, [0], {'agentview_image': image + 1}, context={'phase': 'policy'})
    rec.finish()
    writers = []
    def writer(path, fps):
        sink = Sink(path)
        writers.append(sink)
        return sink
    monkeypatch.setattr(video, '_open_writer', writer)
    video.export_episode_video(rec.path)
    assert writers[0].frames[0][-1, -1].tolist() == [10, 20, 30]
    assert writers[0].frames[-1][-1, -1].tolist() == [11, 21, 31]
    video.export_episode_video(rec.path, rec.path / 'raw.mp4', rotate180=False)
    assert writers[1].frames[0][0, 0].tolist() == [10, 20, 30]
    with pytest.raises(ValueError, match='missing camera'):
        video.export_episode_video(rec.path, rec.path / 'missing.mp4', camera='missing')
    assert not (rec.path / 'missing.mp4').exists()
    assert not (rec.path / 'missing.json').exists()


def test_late_codec_failure_and_unknown_phase_do_not_publish(tmp_path, monkeypatch):
    from vla_trajectory import video
    ep = episode(tmp_path)
    class BrokenSink(Sink):
        def close(self):
            super().close()
            raise RuntimeError('codec close failed')
    monkeypatch.setattr(video, '_open_writer', lambda path, fps: BrokenSink(path))
    with pytest.raises(RuntimeError, match='codec close failed'):
        video.export_episode_video(ep)
    assert not list(ep.glob('*.mp4'))
    assert not list(ep.glob('task-*.json'))
    rec = EpisodeRecorder(tmp_path, {}, 'legacy')
    rec.append({'agentview_image': np.zeros((2, 2, 3), dtype=np.uint8)}, [0], {}, context={})
    rec.finish()
    with pytest.raises(ValueError, match='explicit'):
        video.export_episode_video(rec.path)


def test_actual_mp4_encode_decode_when_optional_encoder_installed(tmp_path):
    pytest.importorskip('imageio_ffmpeg', reason='video extra required for real MP4 encoding')
    import imageio.v2 as imageio
    from vla_trajectory.video import export_episode_video
    ep = episode(tmp_path)
    output = export_episode_video(ep, fps=10)
    reader = imageio.get_reader(output)
    try:
        frames = list(reader.iter_data())
    finally:
        reader.close()
    assert len(frames) == 4
    assert abs(float(frames[-1].mean()) - 150) < 5
