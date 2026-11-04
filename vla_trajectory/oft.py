"""Reversible instrumentation of the existing OFT evaluator (single thread)."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from functools import wraps
import inspect
from pathlib import Path
import time

import numpy as np

from .libero import snapshot, sim_state
from .recorder import EpisodeRecorder


def _copy_observation(observation):
    if not isinstance(observation, dict):
        raise ValueError('Expected observation dictionary from LIBERO')
    result = {}
    for key, value in observation.items():
        array = np.asarray(value)
        if array.dtype.kind in 'biuf':
            result[key] = array.copy()
    return result


def _config(cfg):
    fields = asdict(cfg) if is_dataclass(cfg) else vars(cfg)
    result = {}
    for key, value in fields.items():
        if any(word in key.lower() for word in ('password', 'token', 'secret', 'api_key')):
            continue
        if value is None or isinstance(value, (str, bool, int, float)):
            result[key] = value
        elif isinstance(value, Path):
            result[key] = str(value)
    return result


class _Proxy:
    def __init__(self, env, state, module, cfg):
        self._env, self._state, self._module, self._cfg = env, state, module, cfg
        self._obs = None
        self._steps = 0

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _observe(self, method, *args, **kwargs):
        try:
            obs = getattr(self._env, method)(*args, **kwargs)
            self._obs = _copy_observation(obs)
            return obs
        except BaseException as exc:
            self._state['error'] = f'{type(exc).__name__}: {exc}'
            raise

    def reset(self, *args, **kwargs):
        return self._observe('reset', *args, **kwargs)

    def set_init_state(self, *args, **kwargs):
        return self._observe('set_init_state', *args, **kwargs)

    def get_observation(self):
        if callable(getattr(self._env, 'get_observation', None)):
            return self._observe('get_observation')
        # Upstream ControlEnv lacks a public getter; reset returned a valid obs.
        if self._obs is None:
            raise ValueError('Environment has no observation yet')
        return _copy_observation(self._obs)

    def step(self, action):
        state = self._state
        try:
            before = _copy_observation(self._obs)
            executed = np.asarray(action).copy()
            evidence_before, sim_before = snapshot(self._env), sim_state(self._env)
            result = self._env.step(action)
            if len(result) == 4:
                after, reward, terminated, info = result
                truncated = False
            elif len(result) == 5:
                after, reward, terminated, truncated, info = result
            else:
                raise ValueError('Expected 4 or 5 environment step results')
            evidence_after, sim_after = snapshot(self._env), sim_state(self._env)
            phase = 'settling' if self._steps < self._cfg.num_steps_wait else 'policy'
            limit = getattr(self._module, 'TASK_MAX_STEPS', {}).get(self._cfg.task_suite_name)
            if limit is not None and self._steps + 1 >= limit + self._cfg.num_steps_wait and not terminated:
                truncated = True
            context = {
                'phase': phase,
                'chunk_id': state['chunk_id'] if phase == 'policy' else None,
                'chunk_index': state['chunk_index'] if phase == 'policy' else None,
                'predicates_before': evidence_before['predicates'],
                'predicates_after': evidence_after['predicates'],
                'success_before': evidence_before['success'],
                'success_after': evidence_after['success'],
                'evidence_errors': evidence_before['errors'] + evidence_after['errors'],
                'inference_ms': state.get('inference_ms') if state['query_pending'] else None,
            }
            extra = {}
            for name, value in [('sim_state', sim_before), ('next_sim_state', sim_after)]:
                if value is not None:
                    extra[name] = value
            if self._steps == 0 and state['initial_state'] is not None:
                extra['initial_state'] = state['initial_state']
            if phase == 'policy':
                if state.get('raw_action') is not None:
                    extra['raw_action'] = state['raw_action']
                if state['query_pending']:
                    extra['query_actions'] = state['query_actions']
                    for key, value in state['query_observation'].items():
                        extra['policy_' + key] = value
            try:
                state['recorder'].append(before, executed, _copy_observation(after),
                    reward=float(reward), terminated=bool(terminated), truncated=bool(truncated),
                    context=context, arrays=extra)
            except Exception as exc:
                state['recording_error'] = exc
                raise
            self._obs = _copy_observation(after)
            self._steps += 1
            if phase == 'policy':
                state['chunk_index'] += 1
                state['query_pending'] = False
                state['raw_action'] = None
            return result
        except BaseException as exc:
            state['error'] = f'{type(exc).__name__}: {exc}'
            raise


@contextmanager
def instrument(module, output_root, *, run_metadata=None):
    """Wrap supported OFT functions for the lifetime of this context only.

    OFT's episode loop catches ordinary exceptions. Those errors are retained
    explicitly; recording failures are raised again after that loop returns.
    """
    required = ('run_episode', 'get_action', 'process_action', 'log_message')
    for name in required:
        if not callable(getattr(module, name, None)):
            raise ValueError(f'OFT evaluator missing required callable: {name}')
    signature = inspect.signature(module.run_episode)
    for name in ('cfg', 'env', 'task_description'):
        if name not in signature.parameters:
            raise ValueError(f'Unsupported run_episode signature: missing {name}')
    if getattr(module, '_vla_trace_active', False):
        raise ValueError('OFT instrumentation is already active')
    original = {name: getattr(module, name) for name in required}
    active = [None]

    @wraps(original['get_action'])
    def get_action(*args, **kwargs):
        state = active[0]
        if state is None:
            return original['get_action'](*args, **kwargs)
        try:
            observation = kwargs.get('observation', args[2] if len(args) > 2 else None)
            copied = _copy_observation(observation)
            start = time.perf_counter_ns()
            actions = original['get_action'](*args, **kwargs)
            # Official OFT extends a bounded deque; excess leading rows are
            # discarded. Keep indices relative to the full model prediction.
            cfg = kwargs.get('cfg', args[0] if args else None)
            capacity = getattr(cfg, 'num_open_loop_steps', len(actions))
            offset = max(0, len(actions) - capacity)
            state.update(chunk_id=state['chunk_id'] + 1, chunk_index=offset,
                query_pending=True, query_actions=np.asarray(actions).copy(),
                query_observation=copied, inference_ms=(time.perf_counter_ns() - start) / 1e6)
            return actions
        except BaseException as exc:
            state['error'] = f'{type(exc).__name__}: {exc}'
            raise

    @wraps(original['process_action'])
    def process_action(action, *args, **kwargs):
        state = active[0]
        if state is not None:
            state['raw_action'] = np.asarray(action).copy()
        try:
            return original['process_action'](action, *args, **kwargs)
        except BaseException as exc:
            if state is not None:
                state['error'] = f'{type(exc).__name__}: {exc}'
            raise

    @wraps(original['log_message'])
    def log_message(message, *args, **kwargs):
        if active[0] is not None and str(message).startswith('Episode error:'):
            active[0]['error'] = str(message)
        return original['log_message'](message, *args, **kwargs)

    @wraps(original['run_episode'])
    def run_episode(*args, **kwargs):
        if active[0] is not None:
            raise RuntimeError('Concurrent/nested episodes are unsupported; use separate processes/output roots')
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        cfg, env = bound.arguments['cfg'], bound.arguments['env']
        metadata = dict(run_metadata or {})
        metadata.update(config=_config(cfg), task_description=bound.arguments['task_description'],
            recorder='openvla-trajectory-tools/0.1.0', raw_image_orientation='libero_unmodified',
            bddl_file=str(getattr(env, 'bddl_file_name', getattr(getattr(env, 'env', None), 'bddl_file_name', ''))))
        supplied_initial = bound.arguments.get('initial_state')
        metadata['initial_state'] = None if supplied_initial is None else np.asarray(supplied_initial).tolist()
        recorder = EpisodeRecorder(output_root, metadata)
        state = {'recorder': recorder, 'chunk_id': -1, 'chunk_index': 0,
            'query_pending': False, 'initial_state': bound.arguments.get('initial_state'),
            'error': None, 'recording_error': None}
        active[0] = state
        bound.arguments['env'] = _Proxy(env, state, module, cfg)
        status = 'complete'
        try:
            result = original['run_episode'](*bound.args, **bound.kwargs)
            if state['recording_error'] is not None:
                raise RuntimeError('Trajectory recording failed; aborting evaluation') from state['recording_error']
            return result
        except (KeyboardInterrupt, SystemExit) as exc:
            status = 'interrupted'
            state['error'] = f'{type(exc).__name__}: {exc}'
            raise
        except BaseException as exc:
            status = 'error'
            state['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            try:
                if state['error'] and status == 'complete':
                    status = 'error'
                success = snapshot(env)['success']
                outcome = 'unknown' if status != 'complete' or success is None else ('success' if success else 'failure')
                recorder.finish(outcome=outcome, status=status, error=state['error'])
            finally:
                active[0] = None

    module._vla_trace_active = True
    module.run_episode, module.get_action = run_episode, get_action
    module.process_action, module.log_message = process_action, log_message
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(module, name, value)
        del module._vla_trace_active
