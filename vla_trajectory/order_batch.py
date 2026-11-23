"""Paired order-experiment plans and conservative recorded-episode recovery.

No model or network imports. Checkpoint fingerprints prove byte identity, not
that a file came from the advertised repository; retain download provenance too.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

from .collection import _hash, create_plan, resolve_plan
from .recorder import _read_json, load_arrays, read_manifest, read_steps


CHECKPOINT = 'moojink/openvla-7b-oft-finetuned-libero-10'


def create_cases(registry):
    """Validate the six-task protocol and produce twelve paired formal plans."""
    if registry.get('schema_version') != 1 or registry.get('task_suite_name') != 'libero_10':
        raise ValueError('expected libero_10 order registry schema 1')
    fixed = registry.get('fixed_model_configuration', {})
    if (fixed.get('pretrained_checkpoint') != CHECKPOINT or fixed.get('load_in_4bit') is not True
            or type(fixed.get('num_trials_per_task')) is not int or fixed['num_trials_per_task'] != 1):
        raise ValueError('registry conflicts with the fixed model configuration')
    groups = registry.get('groups')
    if not isinstance(groups, list) or len(groups) != 6:
        raise ValueError('six task groups are required')
    cases, names, identifiers = [], set(), set()
    for group in groups:
        gid, name = group.get('group_id'), group.get('task_name')
        if type(gid) is not int or gid < 1 or gid in identifiers or name in names:
            raise ValueError('task groups require unique IDs and names')
        identifiers.add(gid)
        names.add(name)
        keys = group.get('completion_order', {}).get('simulator_subgoal_keys', {})
        if (set(keys) != {'A', 'B'} or any(not isinstance(v, str) or not v.strip() for v in keys.values())
                or keys['A'] == keys['B']):
            raise ValueError('two distinct runtime subgoal keys are required')
        conditions = [c for c in group.get('conditions', []) if c.get('enabled') is True]
        if len(conditions) != 2 or {c.get('condition_id') for c in conditions} != {'AB', 'BA'}:
            raise ValueError('exactly one AB and one BA condition must be enabled')
        paired = None
        for condition in sorted(conditions, key=lambda c: c['condition_id']):
            cid, instruction = condition['condition_id'], condition.get('instruction')
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError('instruction must be nonempty')
            indices = condition.get('state_indices')
            if not isinstance(indices, list) or len(indices) != 20 or condition.get('episodes_planned') != 20:
                raise ValueError('each condition requires twenty initial states')
            if condition.get('expected_order') != list(cid):
                raise ValueError('expected_order conflicts with condition')
            plan = create_plan(name, 'formal', indices, object_pair=group.get('grasp_order', {}).get('object_pair'))
            if paired is not None and indices != paired:
                raise ValueError('AB and BA must use the same ordered initial-state indices')
            paired = indices
            cases.append({'case_id': f'g{gid}-{cid}', 'group_id': gid, 'task_name': name,
                          'condition_id': cid, 'instruction': instruction, 'expected_order': cid,
                          'subgoal_keys': dict(keys), 'plan': plan})
    return cases


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_checkpoint(directory):
    """Fingerprint inference inputs, excluding OFT-generated backup files.

    OFT overwrites config.auto_map and the two copied model/configuration Python
    files at load time. Normalize that overwritten config field, and identify
    those two modules through the separate OFT source fingerprint. Other Python
    files (including the processor) still contribute their actual bytes.
    """
    root = Path(directory).expanduser().resolve()
    index_path = root / 'model.safetensors.index.json'
    if not index_path.is_file():
        raise ValueError('downloaded checkpoint shard index is missing')
    index = _read_json(index_path)
    shards = set(index.get('weight_map', {}).values())
    if not shards:
        raise ValueError('checkpoint index contains no shards')
    for shard in shards:
        if not isinstance(shard, str) or Path(shard).name != shard or not (root / shard).is_file():
            raise ValueError('checkpoint shard missing or invalid')
    for prefix in ('action_head--', 'proprio_projector--'):
        if not list(root.glob(prefix + '*.pt')):
            raise ValueError(f'checkpoint inference component missing: {prefix}')
    for name in ('config.json', 'dataset_statistics.json'):
        if not (root / name).is_file():
            raise ValueError(f'checkpoint support file missing: {name}')
    files = {}
    for path in sorted(root.iterdir()):
        if (not path.is_file() or path.name.startswith('.') or path.suffix not in
                {'.json', '.safetensors', '.pt', '.model', '.txt', '.py'}):
            continue
        if path.name in {'modeling_prismatic.py', 'configuration_prismatic.py'}:
            continue
        if path.name == 'config.json':
            config = _read_json(path)
            config.pop('auto_map', None)
            files[path.name] = _hash(config)
        else:
            files[path.name] = file_sha256(path)
    return {'sha256': _hash(files), 'files': files,
            'normalization': 'config.json canonical JSON without OFT-overwritten auto_map; synced model/config Python identified by OFT source'}


@contextmanager
def numpy_init_state_context():
    """Allow numeric LIBERO state banks with PyTorch's weights-only loader.

    PyTorch >=2.6 defaults to weights_only=True. Official LIBERO banks contain
    NumPy arrays. Allow only their numeric reconstruction types, without globally
    switching off the weights-only loader. This is a single-thread runner.
    """
    import importlib
    import numpy as np
    import torch
    if not hasattr(torch.serialization, 'safe_globals'):
        yield
        return
    reconstruct = importlib.import_module('numpy.core.multiarray')._reconstruct
    allowed = [reconstruct, np.ndarray, np.dtype,
               type(np.dtype('float64')), type(np.dtype('float32'))]
    if reconstruct.__module__ != 'numpy.core.multiarray':
        allowed.append((reconstruct, 'numpy.core.multiarray._reconstruct'))
    with torch.serialization.safe_globals(allowed):
        yield


def resolve_cases(cases, task_suite):
    """Freeze every selected state's contents before either condition runs."""
    result = []
    for source in cases:
        case = json.loads(json.dumps(source, allow_nan=False))
        resolved, _ = resolve_plan(case['plan'], task_suite)
        case.update(resolved_rows=resolved['rows'], state_bank_sha256=resolved['state_bank_sha256'])
        result.append(case)
    return result


def _complete_episode(episode, row, cfg, identity, instruction):
    from .oft import _config
    manifest = read_manifest(episode)
    metadata = manifest['metadata']
    if manifest['status'] != 'complete' or manifest['outcome'] not in ('success', 'failure') or manifest.get('error'):
        return False
    if metadata.get('synthetic') or metadata.get('experiment_identity') != identity or metadata.get('config') != _config(cfg):
        return False
    for key in ('task_name', 'init_state_index', 'state_sha256', 'plan_hash'):
        if metadata.get(key) != row.get(key):
            return False
    prompt_hash = hashlib.sha256(instruction.encode('utf-8')).hexdigest()
    if (metadata.get('instruction_override') is not True or metadata.get('task_description') != instruction
            or metadata.get('instruction_sha256') != prompt_hash):
        return False
    execution, queries = _read_json(episode / 'execution.json'), _read_json(episode / 'queries.json')
    if (execution.get('episode_id') != manifest['episode_id'] or execution.get('recording_complete') is not True
            or execution.get('step_count_complete') is not True or execution.get('video_status') != 'complete'
            or queries.get('episode_id') != manifest['episode_id']):
        return False
    steps = read_steps(episode)
    counts = {phase: sum(s['context'].get('phase') == phase for s in steps) for phase in ('policy', 'settling')}
    if not counts['policy'] or sum(counts.values()) != len(steps):
        return False
    for phase, count in counts.items():
        if execution.get('executed_' + phase + '_steps') != count or execution.get('recorded_' + phase + '_steps') != count:
            return False
    consumed = {step['context'].get('chunk_id') for step in steps if step['context'].get('phase') == 'policy'}
    query_rows = queries.get('queries', [])
    if not query_rows or len({q.get('chunk_id') for q in query_rows}) != len(query_rows):
        return False
    if {q.get('chunk_id') for q in query_rows} != consumed:
        return False
    if any(q.get('status') != 'complete' or q.get('instruction') != instruction
           or q.get('instruction_sha256') != prompt_hash for q in query_rows):
        return False
    for step in steps:
        arrays = load_arrays(episode, step['index'])
        if any(f'{prefix}/{camera}' not in arrays for prefix in ('obs', 'next_obs')
               for camera in ('agentview_image', 'robot0_eye_in_hand_image')):
            return False
    video_name = execution.get('video_path')
    if not isinstance(video_name, str) or Path(video_name).name != video_name:
        return False
    video = episode / video_name
    if not video.is_file() or video.stat().st_size == 0:
        return False
    sidecar = _read_json(video.with_suffix('.json'))
    frames = sidecar.get('frames', [])
    if sidecar.get('episode_id') != manifest['episode_id'] or len(frames) != len(steps) + 1:
        return False
    expected = [(0, 'obs')] + [(s['index'], 'next_obs') for s in steps]
    return all(frame.get('video_frame') == i and (frame.get('record_step'), frame.get('observation')) == pair
               for i, (frame, pair) in enumerate(zip(frames, expected)))


def find_completed_episode(root, *, row, cfg, experiment_identity, instruction):
    """Return the latest complete matching attempt; incomplete trials rerun.

    A verified official failure is complete. Corrupt/missing evidence remains on
    disk, but cannot cause a trial to be skipped. This validates stored video
    export evidence; it does not independently decode or infer actions from MP4.
    """
    matches = []
    for path in Path(root).glob('*/manifest.json'):
        try:
            if _complete_episode(path.parent, row, cfg, experiment_identity, instruction):
                matches.append((path.stat().st_mtime_ns, str(path.parent), path.parent))
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    return max(matches)[2] if matches else None


def execute_case(module, case, output_root, *, experiment_identity, max_new_episodes=None):
    """Run an already imported OFT evaluator, retaining only verified resumes.

    A model/environment exception propagates after preserving the recorder's
    partial episode. A deliberately limited smoke run returns incomplete state.
    This function does not load a model itself; ``module.eval_libero`` does.
    """
    from .oft import instrument
    from .recorder import _write_json
    if max_new_episodes is not None and (type(max_new_episodes) is not int or max_new_episodes < 0):
        raise ValueError('max_new_episodes must be nonnegative or None')
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    rows, configurations = [], []
    counters = {'executed': 0, 'skipped': 0}

    def select(cfg, row):
        if 'resolved_rows' in case:
            expected = next((item for item in case['resolved_rows']
                             if item['init_state_index'] == row['init_state_index']), None)
            if expected is None or any(row.get(key) != expected.get(key) for key in
                    ('task_id', 'task_name', 'init_state_index', 'state_sha256', 'plan_hash')):
                raise ValueError('initial state fingerprint differs from the paired protocol')
        rows.append(dict(row))
        configurations.append(cfg)
        previous = find_completed_episode(root, row=row, cfg=cfg,
            experiment_identity=experiment_identity, instruction=case['instruction'])
        if previous is not None:
            counters['skipped'] += 1
            return False
        if max_new_episodes is not None and counters['executed'] >= max_new_episodes:
            return False
        import shutil
        if shutil.disk_usage(root).free < 2 * 1024**3:
            raise RuntimeError('Insufficient disk space for another episode; retain existing recordings and choose more storage')
        counters['executed'] += 1
        return True

    metadata = {k: case[k] for k in ('case_id', 'group_id', 'task_name', 'condition_id',
                                     'expected_order', 'subgoal_keys')}
    metadata.update(experiment_identity=experiment_identity, model_repository=CHECKPOINT)
    with instrument(module, root, run_metadata=metadata, collection_plan=case['plan'],
                    instruction_override=case['instruction'], row_filter=select):
        module.eval_libero()
    completed = []
    for row, cfg in zip(rows, configurations):
        episode = find_completed_episode(root, row=row, cfg=cfg,
            experiment_identity=experiment_identity, instruction=case['instruction'])
        if episode is not None:
            completed.append({'episode': episode.name, 'init_state_index': row['init_state_index'],
                              'state_sha256': row['state_sha256']})
    result = {'schema_version': 1, 'case_id': case['case_id'], 'experiment_identity': experiment_identity,
              'planned': 20, 'completed': len(completed), 'complete': len(completed) == 20,
              **counters, 'episodes': completed}
    _write_json(root / 'case_status.json', result)
    if max_new_episodes is None and not result['complete']:
        raise RuntimeError('Case has incomplete or invalid episode evidence; see recorded attempts')
    return result
