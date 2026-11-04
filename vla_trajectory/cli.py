"""CLI entry points; collect delegates all model options to the installed OFT."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def _json(value):
    def encode(item):
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f'Cannot encode {type(item).__name__} as JSON')
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=encode))


def _collect(args):
    root = Path(args.oft_root).expanduser().resolve()
    evaluator_path = root / 'experiments/robot/libero/run_libero_eval.py'
    if not evaluator_path.is_file():
        raise ValueError(f'OFT evaluator not found: {evaluator_path}')
    output = Path(args.output).expanduser().resolve()
    options = args.oft_args[1:] if args.oft_args[:1] == ['--'] else args.oft_args
    plan = None
    if args.plan:
        from .collection import validate_plan
        plan = validate_plan(json.loads(Path(args.plan).expanduser().read_text(encoding='utf-8')))
    try:
        revision = subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unavailable'
    previous_cwd, previous_path, previous_argv = Path.cwd(), list(sys.path), sys.argv[:]
    try:
        os.chdir(root)
        sys.path.insert(0, str(root))
        # OFT modules inspect argv at import time to select robot constants.
        # Hide our transport options before any upstream module is imported.
        sys.argv = [str(evaluator_path)] + options
        module = importlib.import_module('experiments.robot.libero.run_libero_eval')
        if Path(module.__file__).resolve() != evaluator_path:
            raise ValueError('Another OFT checkout is already imported; launch collect in a fresh Python process')
        from .oft import instrument
        with instrument(module, output, run_metadata={'oft_root':str(root), 'oft_revision':revision},
                        collection_plan=plan):
            module.eval_libero()
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path
        sys.argv[:] = previous_argv


def _doctor(args):
    from importlib.metadata import version, PackageNotFoundError
    packages = {}
    for package in ('numpy','Pillow','torch','transformers','mujoco','robosuite','draccus'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    report = {'python':sys.version, 'executable':sys.executable, 'packages':packages,
        'note':'Preflight only; does not prove model loading, rendering or task success.'}
    try:
        gpu = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],
            capture_output=True,text=True,timeout=15)
        report['gpu'] = gpu.stdout.strip() if gpu.returncode==0 else gpu.stderr.strip()
    except (OSError, subprocess.TimeoutExpired):
        report['gpu'] = 'nvidia-smi unavailable'
    ready = True
    if args.oft_root:
        root = Path(args.oft_root).expanduser()
        report['evaluator_exists'] = (root/'experiments/robot/libero/run_libero_eval.py').is_file()
        ready = report['evaluator_exists'] and all(packages[p] for p in packages)
    report['basic_checks_passed'] = bool(ready)
    _json(report)
    return 0 if ready else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description='OpenVLA-OFT + LIBERO trajectory tools')
    sub = parser.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('demo', help='Generate a clearly marked synthetic CPU episode')
    demo.add_argument('--output', default='data/demo')
    doctor = sub.add_parser('doctor', help='Print installed package and GPU information')
    doctor.add_argument('--oft-root')
    collect = sub.add_parser('collect', help='Run existing OFT eval with recording')
    collect.add_argument('--oft-root', required=True)
    collect.add_argument('--output', required=True)
    collect.add_argument('--plan', help='Validated pilot/formal collection plan; place before --')
    collect.add_argument('oft_args', nargs=argparse.REMAINDER)
    plan = sub.add_parser('plan', help='Write a five-state pilot or twenty-state formal plan')
    plan.add_argument('--stage', required=True, choices=('pilot', 'formal'))
    plan.add_argument('--output', required=True)
    from .collection import DEFAULT_TASK_NAME
    plan.add_argument('--task-name', default=DEFAULT_TASK_NAME)
    plan.add_argument('--task-suite-name', default='libero_10')
    plan.add_argument('--state-ids', nargs='+', type=int, help='Initial-state bank indices')
    plan.add_argument('--objects', nargs=2, metavar=('A', 'B'))
    video = sub.add_parser('video', help='Export recorded RGB frames including the final next_obs')
    video.add_argument('episode')
    video.add_argument('--output')
    video.add_argument('--fps', type=float)
    video.add_argument('--camera', default='agentview_image')
    video.add_argument('--exclude-settling', action='store_true')
    video.add_argument('--no-rotate180', action='store_true')
    analyze = sub.add_parser('analyze', help='Export three statistics tables, figures and report')
    analyze.add_argument('root')
    analyze.add_argument('--output', required=True)
    analyze.add_argument('--objects', nargs=2, metavar=('A', 'B'))
    analyze.add_argument('--include-synthetic', action='store_true')
    analyze.add_argument('--no-plots', action='store_true')
    for name in ('annotate','validate','summary'):
        sub.add_parser(name).add_argument('episode')
    review = sub.add_parser('review')
    review.add_argument('episode')
    review.add_argument('--output', required=True)
    imp = sub.add_parser('import', help='Validate and save downloaded manual annotations')
    imp.add_argument('episode')
    imp.add_argument('document')
    args = parser.parse_args(argv)
    try:
        if args.command == 'doctor':
            return _doctor(args)
        if args.command == 'collect':
            _collect(args)
        elif args.command == 'plan':
            from .collection import create_plan
            document = create_plan(args.task_name, args.stage, args.state_ids,
                                   task_suite_name=args.task_suite_name, object_pair=args.objects)
            target = Path(args.output).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                              encoding='utf-8')
            _json({'path': target, 'stage': document['stage'],
                   'states': len(document['init_state_indices']), 'plan_hash': document['plan_hash']})
        elif args.command == 'video':
            from .video import export_episode_video
            output = None if args.output is None else Path(args.output).expanduser().resolve()
            path = export_episode_video(Path(args.episode).expanduser().resolve(), output,
                                        fps=args.fps, camera=args.camera,
                                        include_settling=not args.exclude_settling,
                                        rotate180=not args.no_rotate180)
            _json({'path': path, 'sidecar': path.with_suffix('.json')})
        elif args.command == 'analyze':
            from .analysis import export_analysis
            result = export_analysis(Path(args.root).expanduser().resolve(),
                                     Path(args.output).expanduser().resolve(),
                                     object_pair=args.objects, include_synthetic=args.include_synthetic,
                                     plots=not args.no_plots)
            _json({'episodes': len(result['episodes']), 'subtasks': len(result['subtasks']),
                   'task_summary': len(result['task_summary']), 'paths': result['paths'],
                   'plots': result['plots'], 'excluded_synthetic': result['excluded_synthetic']})
        elif args.command == 'demo':
            from .demo import create_demo
            print(create_demo(args.output))
        elif args.command == 'validate':
            from .recorder import read_manifest, read_steps, load_arrays
            manifest = read_manifest(args.episode)
            steps = read_steps(args.episode)
            for step in steps:
                load_arrays(args.episode, step['index'])
            _json({'valid':True, 'status':manifest['status'], 'num_steps':len(steps),
                'outcome':manifest['outcome'], 'complete':manifest['status']=='complete'})
        elif args.command == 'review':
            from .review import export_review
            export_review(args.episode, args.output)
            print(Path(args.output).resolve())
        else:
            from .annotations import load_annotations, save_annotations, summarize_annotations
            if args.command == 'annotate':
                # Idempotent: do not overwrite human work on repeated annotation.
                document = load_annotations(args.episode)
                if not (Path(args.episode)/'annotations.json').exists():
                    save_annotations(args.episode, document, expected_revision=0)
                _json(load_annotations(args.episode))
            elif args.command == 'summary':
                _json(summarize_annotations(load_annotations(args.episode)))
            else:
                document = json.loads(Path(args.document).read_text(encoding='utf-8'))
                if 'base_revision' not in document:
                    raise ValueError('Import requires base_revision from the review page export')
                save_annotations(args.episode, document, expected_revision=document['base_revision'])
                _json({'saved':True, 'revision':load_annotations(args.episode)['revision']})
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
