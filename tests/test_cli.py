import json
from pathlib import Path

import pytest


def test_demo_annotation_review_import_roundtrip(tmp_path):
    from vla_trajectory.cli import main
    assert main(['demo','--output',str(tmp_path / 'data')]) == 0
    ep=next((tmp_path/'data').iterdir())
    assert main(['validate',str(ep)]) == 0
    assert main(['annotate',str(ep)]) == 0
    annotation=json.loads((ep/'annotations.json').read_text(encoding='utf-8'))
    annotation['base_revision']=annotation['revision']
    annotation['segments']=[{'start':1,'end':3,'label':'reach','notes':'manual review'}]
    manual=tmp_path/'manual.json'
    manual.write_text(json.dumps(annotation),encoding='utf-8')
    assert main(['import',str(ep),str(manual)]) == 0
    assert json.loads((ep/'annotations.json').read_text(encoding='utf-8'))['segments'][0]['label']=='reach'
    # Same browser export cannot silently overwrite a newer revision.
    assert main(['import',str(ep),str(manual)]) == 1
    assert main(['review',str(ep),'--output',str(tmp_path/'review.html')]) == 0
    assert (tmp_path/'review.html').stat().st_size>1000
    assert main(['summary',str(ep)]) == 0


def test_collect_rejects_wrong_root_before_import(tmp_path):
    from vla_trajectory.cli import main
    assert main(['collect','--oft-root',str(tmp_path),'--output',str(tmp_path/'out'),'--','--seed','7']) == 1
    assert not (tmp_path/'out').exists()


def test_user_libero10_4bit_flags_reach_import_and_evaluation_unchanged(tmp_path):
    """Upstream imports inspect argv before eval_libero parses its options."""
    import subprocess
    import sys

    oft_root = tmp_path / 'oft checkout'
    package = oft_root / 'experiments' / 'robot' / 'libero'
    package.mkdir(parents=True)
    for directory in (oft_root/'experiments', oft_root/'experiments/robot', package):
        (directory/'__init__.py').write_text('', encoding='utf-8')
    (package/'run_libero_eval.py').write_text('''
import json
import sys
from pathlib import Path
import_argv = sys.argv[1:]
def run_episode(cfg, env, task_description):
    pass
def get_action(*args, **kwargs):
    pass
def process_action(*args, **kwargs):
    pass
def log_message(*args, **kwargs):
    pass
def eval_libero():
    Path('forwarded.json').write_text(json.dumps({
        'import_argv': import_argv, 'eval_argv': sys.argv[1:], 'cwd': str(Path.cwd())
    }), encoding='utf-8')
''', encoding='utf-8')
    options = ['--pretrained_checkpoint', 'moojink/openvla-7b-oft-finetuned-libero-10',
               '--task_suite_name', 'libero_10', '--num_trials_per_task', '1',
               '--load_in_4bit', 'True']
    completed = subprocess.run([sys.executable, '-m', 'vla_trajectory', 'collect',
        '--oft-root', str(oft_root), '--output', str(tmp_path/'recordings'), '--', *options],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    forwarded = json.loads((oft_root/'forwarded.json').read_text(encoding='utf-8'))
    assert forwarded['import_argv'] == options
    assert forwarded['eval_argv'] == options
    assert Path(forwarded['cwd']) == oft_root


@pytest.mark.parametrize('stage,count', [('pilot', 5), ('formal', 20)])
def test_plan_writes_reproducible_default_task_states(tmp_path, capsys, stage, count):
    from vla_trajectory.cli import main
    target = tmp_path / 'plans' / f'{stage}.json'
    assert main(['plan', '--stage', stage, '--output', str(target)]) == 0
    plan = json.loads(target.read_text(encoding='utf-8'))
    assert plan['task_name'] == 'LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket'
    assert plan['init_state_indices'] == list(range(count))
    assert plan['object_pair'] == ['alphabet_soup_1', 'tomato_sauce_1']
    output = json.loads(capsys.readouterr().out)
    assert output['states'] == count
    assert Path(output['path']) == target.resolve()


def test_plan_custom_states_and_objects_and_invalid_duplicate(tmp_path, capsys):
    from vla_trajectory.cli import main
    target = tmp_path / 'custom.json'
    assert main(['plan', '--stage', 'pilot', '--task-name', 'custom-task',
                 '--state-ids', '9', '7', '5', '3', '1', '--objects', 'cup', 'plate',
                 '--output', str(target)]) == 0
    plan = json.loads(target.read_text(encoding='utf-8'))
    assert plan['init_state_indices'] == [9, 7, 5, 3, 1]
    assert plan['object_pair'] == ['cup', 'plate']
    before = target.read_bytes()
    assert main(['plan', '--stage', 'pilot', '--state-ids', '0', '0', '1', '2', '3',
                 '--output', str(target)]) == 1
    assert target.read_bytes() == before
    assert 'unique' in capsys.readouterr().err


def test_collect_plan_drives_real_instrumentation_without_changing_model_flags(tmp_path):
    """A tiny external evaluator exposes selected states without loading a GPU."""
    import subprocess
    import sys
    from vla_trajectory.collection import create_plan, DEFAULT_TASK_NAME
    oft = tmp_path / 'oft'
    package = oft / 'experiments/robot/libero'
    package.mkdir(parents=True)
    for directory in (oft/'experiments', oft/'experiments/robot', package):
        (directory/'__init__.py').write_text('', encoding='utf-8')
    (package/'run_libero_eval.py').write_text('''
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import_argv = sys.argv[1:]
states_seen = []
def run_episode(cfg, env, task_description): pass
def get_action(*args, **kwargs): pass
def process_action(*args, **kwargs): pass
def log_message(*args, **kwargs): pass
class Suite:
    n_tasks = 1
    def get_task(self, task_id):
        return SimpleNamespace(name='LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket')
    def get_task_init_states(self, task_id):
        return np.arange(60).reshape(30, 2)
def run_task(cfg, task_suite, task_id, total_episodes=0, total_successes=0):
    states_seen.append(task_suite.get_task_init_states(task_id)[0].tolist())
    return total_episodes + 1, total_successes
def eval_libero():
    cfg = SimpleNamespace(task_suite_name='libero_10', num_trials_per_task=1, initial_states_path='DEFAULT')
    totals = run_task(cfg, Suite(), 0)
    Path('forwarded.json').write_text(json.dumps({'states': states_seen, 'totals': totals,
        'import_argv': import_argv, 'eval_argv': sys.argv[1:]}), encoding='utf-8')
''', encoding='utf-8')
    plan = tmp_path/'pilot.json'
    plan.write_text(json.dumps(create_plan(DEFAULT_TASK_NAME, 'pilot', [9, 7, 5, 3, 1])), encoding='utf-8')
    options = ['--pretrained_checkpoint', 'moojink/openvla-7b-oft-finetuned-libero-10',
               '--task_suite_name', 'libero_10', '--num_trials_per_task', '1', '--load_in_4bit', 'True']
    completed = subprocess.run([sys.executable, '-m', 'vla_trajectory', 'collect',
        '--oft-root', str(oft), '--output', str(tmp_path/'data'), '--plan', str(plan), '--', *options],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    forwarded = json.loads((oft/'forwarded.json').read_text())
    assert forwarded['states'] == [[18, 19], [14, 15], [10, 11], [6, 7], [2, 3]]
    assert forwarded['totals'] == [5, 0]
    assert forwarded['import_argv'] == forwarded['eval_argv'] == options
    assert len(list((tmp_path/'data').glob('collection_plan_*.json'))) == 1


def test_analyze_exports_imported_attempts_and_serializable_path_summary(tmp_path, capsys):
    import csv
    from vla_trajectory.cli import main
    from vla_trajectory.demo import create_demo
    from vla_trajectory.annotations import load_annotations
    ep = create_demo(tmp_path/'data')
    document = load_annotations(ep)
    document['base_revision'] = document['revision']
    document['object_reviewed'] = True
    document['object_attempts'] = [{'attempt_id': 'first', 'object': 'cup',
        'start_frame': 1, 'end_frame': 5, 'grasp_start_frame': 1, 'grasp_frame': 2,
        'place_start_frame': 4, 'place_frame': 5, 'subtask_success': True,
        'source': 'human_review', 'notes': '人工复核'}]
    manual = tmp_path/'manual.json'
    manual.write_text(json.dumps(document), encoding='utf-8')
    assert main(['import', str(ep), str(manual)]) == 0
    capsys.readouterr()
    assert main(['analyze', str(ep), '--output', str(tmp_path/'excluded'), '--no-plots']) == 0
    excluded = json.loads(capsys.readouterr().out)
    assert excluded['episodes'] == 0 and excluded['excluded_synthetic'] == 1
    output = tmp_path/'analysis'
    assert main(['analyze', str(ep), '--output', str(output), '--objects', 'cup', 'plate',
                 '--include-synthetic', '--no-plots']) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['episodes'] == 1 and summary['subtasks'] == 1
    assert summary['task_summary'] == 1 and summary['plots'] == {}
    assert Path(summary['paths']['report']).is_file()
    with (output/'subtask_events.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['Object'] == 'cup'
    assert rows[0]['Grasp Step'] == '3' and rows[0]['Place Step'] == '6'
    assert rows[0]['Notes'] == '人工复核'


def test_video_cli_preserves_real_frame_mapping_and_reports_codec_errors(tmp_path, capsys, monkeypatch):
    from vla_trajectory.cli import main
    from vla_trajectory.demo import create_demo
    from vla_trajectory import video
    ep = create_demo(tmp_path/'data')
    frames = []
    class EncoderBoundary:
        def __init__(self, path): self.path = path
        def append_data(self, frame): frames.append(frame.copy())
        def close(self): self.path.write_bytes(b'boundary-only-not-real-mp4')
    monkeypatch.setattr(video, '_open_writer', lambda path, fps: EncoderBoundary(path))
    target = tmp_path/'video.mp4'
    assert main(['video', str(ep), '--output', str(target), '--fps', '8',
                 '--camera', 'robot0_eye_in_hand_image', '--no-rotate180', '--exclude-settling']) == 0
    summary = json.loads(capsys.readouterr().out)
    assert Path(summary['path']) == target
    assert len(frames) == 25
    sidecar = json.loads(Path(summary['sidecar']).read_text())
    assert sidecar['rotate180'] is False and sidecar['include_settling'] is False
    assert sidecar['camera'] == 'robot0_eye_in_hand_image'
    assert sidecar['playback_fps'] == 8 and sidecar['frames'][-1]['action_step_id'] == 24
    def unavailable(path, fps): raise RuntimeError('encoder unavailable')
    monkeypatch.setattr(video, '_open_writer', unavailable)
    assert main(['video', str(ep), '--output', str(tmp_path/'failed.mp4')]) == 1
    assert 'encoder unavailable' in capsys.readouterr().err
    assert not (tmp_path/'failed.mp4').exists()

