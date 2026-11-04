import json
from pathlib import Path


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

