"""Reversible instrumentation of the existing OFT evaluator (single thread)."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from functools import wraps
import inspect
from pathlib import Path
import time

import numpy as np

from . import __version__
from .libero import snapshot, sim_state
from .recorder import EpisodeRecorder, _read_json, _write_json, read_steps
from .collection import resolve_plan, validate_plan


def _observation_fields(observation, object_pair):
    """Describe explicit sensors only; never guess poses from flattened sim state."""
    def field(key, unit=None, order=None):
        value = observation.get(key)
        result = {'key': key, 'present': value is not None,
                  'shape': None if value is None else list(np.asarray(value).shape)}
        if unit:
            result['unit'] = unit
        if order:
            result['order'] = order
        return result
    objects = list(object_pair or [])
    if not objects:
        objects = [key[:-4] for key in observation if key.endswith('_pos')
                   and not key.startswith('robot') and '_to_' not in key]
    return {'eef_position': field('robot0_eef_pos', 'm'),
            'eef_quaternion': field('robot0_eef_quat', 'unit_quaternion', 'xyzw'),
            'gripper_position': field('robot0_gripper_qpos', 'model_joint_native'),
            'gripper_velocity': field('robot0_gripper_qvel', 'model_joint_native_per_s'),
            'objects': {name: {'position': field(name + '_pos', 'm'),
                               'quaternion': field(name + '_quat', 'unit_quaternion', 'xyzw')}
                        for name in objects},
            'available_numeric_keys': sorted(observation),
            'pose_reference': 'world',
            'source': 'LIBERO BDDLBaseDomain object sensors and robosuite SingleArm observables'}


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
            if self._steps == 0:
                self._state['observation_fields'] = _observation_fields(
                    self._obs, self._state['metadata'].get('analysis_object_pair'))
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
            phase = 'settling' if self._steps < self._cfg.num_steps_wait else 'policy'
            try:
                result = self._env.step(action)
            except BaseException:
                # The simulator may have advanced before throwing: counts are a
                # verified lower bound, never an invented completed transition.
                state['step_count_complete'] = False
                raise
            state['executed_' + phase + '_steps'] += 1
            record_step = self._steps
            self._steps += 1
            if len(result) == 4:
                after, reward, terminated, info = result
                truncated = False
            elif len(result) == 5:
                after, reward, terminated, truncated, info = result
            else:
                raise ValueError('Expected 4 or 5 environment step results')
            evidence_after, sim_after = snapshot(self._env), sim_state(self._env)
            limit = getattr(self._module, 'TASK_MAX_STEPS', {}).get(self._cfg.task_suite_name)
            if limit is not None and self._steps >= limit + self._cfg.num_steps_wait and not terminated:
                truncated = True
            context = {
                'phase': phase,
                'action_step_id': state['executed_policy_steps'] if phase == 'policy' else None,
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
            if record_step == 0 and state['initial_state'] is not None:
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
            state['recorded_' + phase + '_steps'] += 1
            self._obs = _copy_observation(after)
            if phase == 'policy':
                state['chunk_index'] += 1
                state['query_pending'] = False
                state['raw_action'] = None
            return result
        except BaseException as exc:
            state['error'] = f'{type(exc).__name__}: {exc}'
            raise


@contextmanager
def instrument(module, output_root, *, run_metadata=None, collection_plan=None):
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
    plan = None if collection_plan is None else validate_plan(collection_plan)
    resolved = [None]
    selected_states = [None]
    row_metadata = [None]
    row_environment = [None]
    task_signature = None
    if plan is not None:
        if not callable(getattr(module, 'run_task', None)):
            raise ValueError('OFT evaluator missing required callable: run_task')
        original['run_task'] = module.run_task
        if callable(getattr(module, 'get_libero_env', None)):
            original['get_libero_env'] = module.get_libero_env
        task_signature = inspect.signature(module.run_task)
        for name in ('cfg', 'task_suite', 'task_id', 'total_episodes', 'total_successes'):
            if name not in task_signature.parameters:
                raise ValueError(f'Unsupported run_task signature: missing {name}')

    class SelectedStateSuite:
        def __init__(self, suite, task_id, initial_state):
            self._suite, self._task_id, self._initial_state = suite, task_id, initial_state
        def __getattr__(self, name):
            return getattr(self._suite, name)
        def get_task_init_states(self, task_id):
            if task_id != self._task_id:
                raise ValueError('unexpected task in selected-state suite')
            return [self._initial_state]

    def run_task(*args, **kwargs):
        bound = task_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        cfg, suite, task_id = (bound.arguments[name] for name in ('cfg', 'task_suite', 'task_id'))
        if cfg.task_suite_name != plan['task_suite_name']:
            raise ValueError('collection task_suite_name conflicts with evaluator config')
        if type(cfg.num_trials_per_task) is not int or cfg.num_trials_per_task != 1:
            raise ValueError('collection requires num_trials_per_task=1')
        if cfg.initial_states_path != 'DEFAULT':
            raise ValueError('collection requires initial_states_path=DEFAULT')
        if resolved[0] is None:
            candidate, bank = resolve_plan(plan, suite)
            # Snapshot selected states so later task/environment mutation cannot
            # alter a fingerprinted initial condition.
            states = [np.asarray(bank[i]).copy() for i in plan['init_state_indices']]
            Path(output_root).mkdir(parents=True, exist_ok=True)
            plan_path = Path(output_root) / f'collection_plan_{plan["plan_hash"]}.json'
            if plan_path.exists() and _read_json(plan_path) != candidate:
                raise ValueError('collection plan state bank differs from the persisted fingerprint')
            _write_json(plan_path, candidate)
            resolved[0], selected_states[0] = candidate, states
        if task_id != resolved[0]['task_id']:
            return bound.arguments['total_episodes'], bound.arguments['total_successes']
        totals = bound.arguments['total_episodes'], bound.arguments['total_successes']
        for row, initial_state in zip(resolved[0]['rows'], selected_states[0]):
            bound.arguments['task_suite'] = SelectedStateSuite(suite, task_id, initial_state.copy())
            bound.arguments['total_episodes'], bound.arguments['total_successes'] = totals
            row_metadata[0] = row
            primary_error = None
            try:
                totals = original['run_task'](*bound.args, **bound.kwargs)
            except BaseException as exc:
                primary_error = exc
                raise
            finally:
                row_metadata[0] = None
                env, row_environment[0] = row_environment[0], None
                if env is not None and callable(getattr(env, 'close', None)):
                    try:
                        env.close()
                    except BaseException as cleanup_error:
                        if primary_error is None:
                            raise
                        # Keep the original exception object/type, including
                        # KeyboardInterrupt/SystemExit. Attach cleanup evidence
                        # on Python 3.10 too; 3.11+ also prints it in tracebacks.
                        detail = f'{type(cleanup_error).__name__}: {cleanup_error}'
                        primary_error.cleanup_errors = [*getattr(primary_error, 'cleanup_errors', []), detail]
                        if callable(getattr(primary_error, 'add_note', None)):
                            primary_error.add_note(f'Environment cleanup also failed: {detail}')
        return totals

    def get_libero_env(*args, **kwargs):
        result = original['get_libero_env'](*args, **kwargs)
        # Capture factory ownership before upstream pre-episode logging. The
        # run_episode fallback still supports evaluator variants without a
        # public factory, and the single row owns/ closes its environment once.
        if row_metadata[0] is not None:
            row_environment[0] = result[0]
        return result

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
        if plan is not None and row_metadata[0] is None:
            raise ValueError('collection episodes must be invoked through run_task')
        if plan is not None:
            row_environment[0] = env
        metadata = dict(run_metadata or {})
        metadata.update(row_metadata[0] or {})
        frequency = getattr(env, 'control_freq', getattr(getattr(env, 'env', None), 'control_freq', None))
        metadata['control_freq'] = None if frequency is None else float(frequency)
        metadata.update(config=_config(cfg), task_description=bound.arguments['task_description'],
            recorder=f'openvla-trajectory-tools/{__version__}', raw_image_orientation='libero_unmodified',
            bddl_file=str(getattr(env, 'bddl_file_name', getattr(getattr(env, 'env', None), 'bddl_file_name', ''))))
        supplied_initial = bound.arguments.get('initial_state')
        metadata['initial_state'] = None if supplied_initial is None else np.asarray(supplied_initial).tolist()
        recorder = EpisodeRecorder(output_root, metadata)
        state = {'recorder': recorder, 'chunk_id': -1, 'chunk_index': 0,
            'query_pending': False, 'initial_state': bound.arguments.get('initial_state'),
            'error': None, 'recording_error': None, 'metadata': metadata,
            'executed_settling_steps': 0, 'executed_policy_steps': 0,
            'recorded_settling_steps': 0, 'recorded_policy_steps': 0,
            'step_count_complete': True, 'observation_fields': {}}
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
                # The manifest, rather than append's return, is the commit
                # boundary even when an exception arrived just after replace.
                committed = read_steps(recorder.path)
                for phase in ('settling', 'policy'):
                    state['recorded_' + phase + '_steps'] = sum(
                        step['context'].get('phase') == phase for step in committed)
                execution = {'schema_version': 1, 'episode_id': recorder.manifest['episode_id'],
                             **{key: state[key] for key in (
                                 'executed_settling_steps', 'executed_policy_steps',
                                 'recorded_settling_steps', 'recorded_policy_steps',
                                 'step_count_complete', 'observation_fields')},
                             'recording_complete': (state['step_count_complete']
                                 and state['executed_settling_steps'] == state['recorded_settling_steps']
                                 and state['executed_policy_steps'] == state['recorded_policy_steps']),
                             'video_status': 'not_requested', 'video_error': None}
                # Persist counts before invoking an optional codec, including
                # when recording failed or the evaluator swallowed a sim error.
                _write_json(recorder.path / 'execution.json', execution)
                if plan is not None:
                    try:
                        from .video import export_episode_video
                        output = export_episode_video(recorder.path)
                        execution.update(video_status='complete', video_path=output.name)
                    except Exception as exc:
                        execution.update(video_status='error', video_error=f'{type(exc).__name__}: {exc}')
                        _write_json(recorder.path / 'execution.json', execution)
                        # Preserve any already-active episode/recording failure.
                        if status == 'complete':
                            raise RuntimeError(f'Episode video export failed: {exc}') from exc
                    _write_json(recorder.path / 'execution.json', execution)
            finally:
                active[0] = None

    module._vla_trace_active = True
    module.run_episode, module.get_action = run_episode, get_action
    module.process_action, module.log_message = process_action, log_message
    if plan is not None:
        module.run_task = run_task
        if 'get_libero_env' in original:
            module.get_libero_env = wraps(original['get_libero_env'])(get_libero_env)
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(module, name, value)
        del module._vla_trace_active
