"""Read simulator evidence without importing LIBERO or changing its state."""
from __future__ import annotations

import numpy as np


def _layers(env):
    seen = set()
    while env is not None and id(env) not in seen:
        seen.add(id(env))
        yield env
        env = getattr(env, 'env', None)


def _boolean(value):
    return bool(value) if isinstance(value, (bool, np.bool_)) else None


def snapshot(env):
    """Unavailable checks remain unknown; errors are preserved as diagnostics."""
    layers = list(_layers(env))
    result = {'predicates': {}, 'success': None, 'errors': []}
    for layer in layers:
        check = getattr(layer, 'check_success', None) or getattr(layer, '_check_success', None)
        if callable(check):
            try:
                result['success'] = _boolean(check())
            except Exception as exc:
                result['errors'].append(f'success: {type(exc).__name__}: {exc}')
            break
    for layer in layers:
        parsed = getattr(layer, 'parsed_problem', None)
        if not isinstance(parsed, dict):
            continue
        evaluate = getattr(layer, '_eval_predicate', None)
        for state in parsed.get('goal_state', []):
            if not isinstance(state, (list, tuple)) or not state:
                result['errors'].append('Malformed goal predicate')
                continue
            name = f'{state[0]}({", ".join(map(str, state[1:]))})'
            value = None
            if callable(evaluate):
                try:
                    value = _boolean(evaluate(state))
                except Exception as exc:
                    result['errors'].append(f'{name}: {type(exc).__name__}: {exc}')
            result['predicates'][name] = value
        break
    return result


def sim_state(env):
    """Return a copied numeric simulator state, or None if unsupported."""
    for layer in _layers(env):
        getter = getattr(layer, 'get_sim_state', None)
        if callable(getter):
            try:
                return np.asarray(getter()).copy()
            except Exception:
                return None
    return None
