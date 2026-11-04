"""Validated, reproducible single-task collection plans (no model imports)."""
from __future__ import annotations

import hashlib
import json

import numpy as np


DEFAULT_TASK_NAME = 'LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket'
DEFAULT_OBJECT_PAIR = ['alphabet_soup_1', 'tomato_sauce_1']


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(',', ':'), allow_nan=False).encode('ascii')).hexdigest()


def create_plan(task_name, stage, init_state_indices=None, *, task_suite_name='libero_10', object_pair=None):
    """Create a pilot (five) or formal (at least twenty) distinct-index plan."""
    indices = list(range(5 if stage == 'pilot' else 20)) if init_state_indices is None else list(init_state_indices)
    plan = {'schema_version': 1, 'task_name': task_name, 'stage': stage,
            'task_suite_name': task_suite_name, 'init_state_indices': indices,
            'object_pair': (list(DEFAULT_OBJECT_PAIR) if task_name == DEFAULT_TASK_NAME else None)
            if object_pair is None else list(object_pair)}
    return validate_plan(plan)


def validate_plan(plan):
    """Return a detached normalized plan; reject stale hashes and ambiguous input."""
    try:
        result = json.loads(json.dumps(plan, allow_nan=False))
    except (ValueError, TypeError) as exc:
        raise ValueError('plan must be a finite JSON object') from exc
    if not isinstance(result, dict) or result.get('schema_version') != 1:
        raise ValueError('unsupported collection plan schema_version')
    for key in ('task_name', 'task_suite_name'):
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ValueError(f'{key} must be a nonempty string')
    stage, indices = result.get('stage'), result.get('init_state_indices')
    if stage not in ('pilot', 'formal'):
        raise ValueError('stage must be pilot or formal')
    if not isinstance(indices, list) or any(type(i) is not int or i < 0 for i in indices):
        raise ValueError('init_state_indices must contain nonnegative integers')
    if len(set(indices)) != len(indices):
        raise ValueError('init_state_indices must be unique')
    if (stage == 'pilot' and len(indices) != 5) or (stage == 'formal' and len(indices) < 20):
        raise ValueError('pilot requires five states; formal requires at least twenty')
    pair = result.get('object_pair')
    if pair is not None and (not isinstance(pair, list) or len(pair) != 2
                            or any(not isinstance(x, str) or not x.strip() for x in pair)
                            or pair[0] == pair[1]):
        raise ValueError('object_pair must contain two distinct nonempty object names')
    old_hash = result.pop('plan_hash', None)
    digest = _hash(result)
    if old_hash is not None and old_hash != digest:
        raise ValueError('collection plan hash does not match content')
    result['plan_hash'] = digest
    return result


def resolve_plan(plan, task_suite):
    """Resolve exact task identity and fingerprint the complete initial-state bank."""
    plan = validate_plan(plan)
    matches = [i for i in range(task_suite.n_tasks)
               if task_suite.get_task(i).name == plan['task_name']]
    if len(matches) != 1:
        raise ValueError('collection task name must resolve to exactly one task')
    task_id = matches[0]
    bank = task_suite.get_task_init_states(task_id)
    hashes = []
    for state in bank:
        value = np.asarray(state)
        if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
            raise ValueError('initial state must be a finite numeric array')
        hashes.append(_hash({'shape': list(value.shape), 'values': value.astype(float).tolist()}))
    indices = plan['init_state_indices']
    if any(i >= len(bank) for i in indices):
        raise ValueError('initial state index out of range')
    if len({hashes[i] for i in indices}) != len(indices):
        raise ValueError('selected initial states have duplicate content')
    rows = []
    for trial, index in enumerate(indices):
        rows.append({'task_id': task_id, 'task_name': plan['task_name'],
                     'init_state_id': f'S{index:03d}', 'init_state_index': index,
                     'trial_index': trial, 'stage': plan['stage'],
                     'plan_row_id': f'{plan["stage"]}-{trial:03d}',
                     'plan_hash': plan['plan_hash'], 'state_sha256': hashes[index],
                     'analysis_object_pair': plan['object_pair']})
    return dict(plan, task_id=task_id, state_bank_sha256=_hash(hashes), rows=rows), bank
