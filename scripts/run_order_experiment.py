"""Prepare/run the six-task protocol; this script never downloads weights.

The run command requires the original merged checkpoint already downloaded
locally and a working OFT installation. Each condition gets a fresh process.
"""
from __future__ import annotations

import argparse
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from vla_trajectory.collection import _hash
from vla_trajectory.order_batch import (create_cases, execute_case, file_sha256,
    fingerprint_checkpoint, numpy_init_state_context, resolve_cases)
from vla_trajectory.recorder import _read_json, _write_json


def source_fingerprint(root):
    evaluator = root / 'experiments/robot/libero/run_libero_eval.py'
    if not evaluator.is_file():
        raise ValueError(f'OFT evaluator missing: {evaluator}')
    paths = set(root.glob('*.py'))
    for directory in ('experiments', 'prismatic', 'openvla'):
        paths.update((root / directory).rglob('*.py'))
    if (root / 'pyproject.toml').is_file():
        paths.add(root / 'pyproject.toml')
    return _hash({p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(paths)
                  if '__pycache__' not in p.parts})


def worker(args):
    document = _read_json(Path(args.case_file))
    root = Path(document['oft_root'])
    if source_fingerprint(root) != document['provenance']['oft_source_sha256']:
        raise ValueError('OFT source changed after protocol creation')
    output = Path(document['output'])
    cache = output / '.numba-cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['NUMBA_CACHE_DIR'] = str(cache)
    if document.get('libero_config'):
        os.environ['LIBERO_CONFIG_PATH'] = document['libero_config']
    sys.dont_write_bytecode = True
    evaluator = root / 'experiments/robot/libero/run_libero_eval.py'
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.argv = [str(evaluator), '--pretrained_checkpoint', document['checkpoint_dir'],
                '--task_suite_name', 'libero_10', '--num_trials_per_task', '1',
                '--load_in_4bit', 'True', '--seed', str(document['seed'])]
    module = importlib.import_module('experiments.robot.libero.run_libero_eval')
    if Path(module.__file__).resolve() != evaluator:
        raise ValueError('A different OFT checkout is already imported')
    with numpy_init_state_context():
        result = execute_case(module, document['case'], output,
                              experiment_identity=document['experiment_identity'],
                              max_new_episodes=document.get('max_new_episodes'))
    print(json.dumps(result, ensure_ascii=True))


def export_run(root, cases):
    from vla_trajectory.order_report import export_order_results
    selected, attempts = [], []
    for case in cases:
        directory = root / case['case_id']
        for manifest in sorted(directory.glob('*/manifest.json')):
            attempts.append({'episode': manifest.parent, 'subgoal_keys': case['subgoal_keys'],
                             'expected_order': case['expected_order']})
        status_path = directory / 'case_status.json'
        if status_path.is_file():
            status = _read_json(status_path)
            for row in status['episodes']:
                selected.append({'episode': directory / row['episode'], 'subgoal_keys': case['subgoal_keys'],
                                 'expected_order': case['expected_order']})
    # Attempts include interruptions; selected tables contain one verified
    # completed attempt per planned row. Neither includes unrun planned rows.
    export_order_results(attempts, root / 'attempts')
    return export_order_results(selected, root / 'results')


def run(args, cases):
    root, checkpoint = Path(args.oft_root).expanduser().resolve(), Path(args.checkpoint_dir).expanduser().resolve()
    local_cache = Path(args.output).expanduser().resolve() / '.numba-cache'
    local_cache.mkdir(parents=True, exist_ok=True)
    os.environ['NUMBA_CACHE_DIR'] = str(local_cache)
    if args.libero_config:
        args.libero_config = str(Path(args.libero_config).expanduser().resolve())
        os.environ['LIBERO_CONFIG_PATH'] = args.libero_config
    sys.dont_write_bytecode = True
    packages = {}
    for name in ('torch', 'transformers', 'bitsandbytes', 'mujoco', 'robosuite', 'libero'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    if any(value is None for value in packages.values()):
        raise ValueError('Inference dependencies are missing: ' + ', '.join(k for k, v in packages.items() if v is None))
    from libero.libero import benchmark
    with numpy_init_state_context():
        cases = resolve_cases(cases, benchmark.get_benchmark_dict()['libero_10']())
    print('Hashing downloaded checkpoint files before evaluation...', flush=True)
    provenance = {'checkpoint': fingerprint_checkpoint(checkpoint), 'oft_source_sha256': source_fingerprint(root),
                  'python': sys.version, 'packages': packages, 'seed': args.seed}
    identity = _hash({'cases': cases, 'provenance': provenance})
    output = Path(args.output).expanduser().resolve() / identity
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / 'protocol.json', {'schema_version': 1, 'experiment_identity': identity,
                'planned_episodes': 240, 'provenance': provenance, 'cases': cases})
    remaining = args.max_new_episodes
    for case in cases:
        if remaining == 0:
            break
        case_root = output / case['case_id']
        case_root.mkdir(exist_ok=True)
        document = {'case': case, 'oft_root': str(root), 'checkpoint_dir': str(checkpoint),
                    'output': str(case_root), 'experiment_identity': identity, 'provenance': provenance,
                    'seed': args.seed, 'libero_config': args.libero_config, 'max_new_episodes': remaining}
        case_file = case_root / 'invocation.json'
        _write_json(case_file, document)
        print(f"Running {case['case_id']}; logs: {case_root / 'worker.log'}", flush=True)
        with (case_root / 'worker.log').open('ab') as log:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'worker',
                                      '--case-file', str(case_file)], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            export_run(output, cases)
            raise RuntimeError(f"{case['case_id']} failed; inspect {case_root / 'worker.log'}")
        status = _read_json(case_root / 'case_status.json')
        if remaining is not None:
            remaining -= status['executed']
    report = export_run(output, cases)
    print(json.dumps({'output': str(output), 'recorded_completed_episodes': len(report['episodes']),
                      'planned_episodes': 240}, ensure_ascii=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('describe', 'run'):
        sub = commands.add_parser(command)
        sub.add_argument('--registry', default=str(REPO / 'plans/libero10-order-six.json'))
        if command == 'run':
            sub.add_argument('--oft-root', required=True)
            sub.add_argument('--checkpoint-dir', required=True)
            sub.add_argument('--output', required=True)
            sub.add_argument('--libero-config')
            sub.add_argument('--seed', type=int, default=7)
            sub.add_argument('--max-new-episodes', type=int, help='Use 1 for a real-model smoke run, then omit to resume')
    child = commands.add_parser('worker', help=argparse.SUPPRESS)
    child.add_argument('--case-file', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'worker':
            worker(args)
            return 0
        cases = create_cases(_read_json(Path(args.registry)))
        if args.command == 'describe':
            print(json.dumps({'contains_experiment_results': False, 'planned_episodes': 240,
                              'cases': cases}, ensure_ascii=True))
        else:
            if args.max_new_episodes is not None and args.max_new_episodes < 1:
                raise ValueError('--max-new-episodes must be positive')
            run(args, cases)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
