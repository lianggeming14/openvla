from types import ModuleType, SimpleNamespace
import hashlib
import json
import numpy as np
import pytest


def evaluator():
    m = ModuleType('fake_oft')
    exec('''
import numpy as np
from collections import deque
TASK_MAX_STEPS = {'test': 3}
def get_action(cfg, model, observation, task_description, **kwargs):
    return np.array([[1,0,0,0,0,0,.2],[2,0,0,0,0,0,.8]])
def process_action(action, model_family):
    action[-1] = -1 if action[-1] > .5 else 1
    return action
def log_message(message, log_file=None):
    pass
def run_episode(cfg, env, task_description, model=None, resize_size=8, initial_state=None, log_file=None):
    env.reset()
    obs = env.set_init_state(initial_state)
    images=[]
    try:
        for i in range(cfg.num_steps_wait):
            obs, reward, done, info = env.step([0]*7)
        queue=deque(maxlen=cfg.num_open_loop_steps)
        for i in range(TASK_MAX_STEPS[cfg.task_suite_name]):
            if not queue:
                queue.extend(get_action(cfg,model,{'full_image':obs['agentview_image'], 'state':obs['state']},task_description))
            a=process_action(queue.popleft(),cfg.model_family)
            obs,reward,done,info=env.step(a.tolist())
            if done:
                return True,images
    except Exception as e:
        log_message(f'Episode error: {e}')
    return False,images
''', m.__dict__)
    return m


class Env:
    closed_count = 0
    def __init__(self, done_at=4, success=True, fail_at=None):
        self.i = 0
        self.done_at, self.success, self.fail_at = done_at, success, fail_at
        self.obs = {'agentview_image': np.zeros((8,8,3), dtype=np.uint8), 'state':np.zeros(2)}
        self.parsed_problem = {'goal_state': [['on','cup','plate']]}
    def reset(self):
        self.i=0
        return self.obs
    def set_init_state(self,state):
        return self.obs
    def step(self,action):
        self.i+=1
        if self.i == self.fail_at:
            raise RuntimeError('sim failed')
        self.obs['agentview_image'][:] = self.i
        self.obs['state'][:] = self.i
        return self.obs,0.0,self.i == self.done_at,{}
    def check_success(self):
        return self.success and self.i >= self.done_at
    def _eval_predicate(self,state):
        return self.check_success()
    def get_sim_state(self):
        return np.array([self.i],dtype=float)
    def close(self):
        type(self).closed_count += 1


def config():
    return SimpleNamespace(num_steps_wait=1, num_open_loop_steps=2, task_suite_name='test', model_family='openvla',seed=7,pretrained_checkpoint='local/model')


def test_chunk_alignment_inplace_mutation_and_restore(tmp_path):
    from vla_trajectory.oft import instrument
    from vla_trajectory.recorder import read_manifest, read_steps, load_arrays
    m=evaluator()
    old=m.run_episode
    with instrument(m,tmp_path):
        result=m.run_episode(config(),Env(),'put cup on plate',initial_state=np.array([0.]))
    assert result[0] is True
    assert m.run_episode is old
    ep=next(tmp_path.iterdir())
    assert read_manifest(ep)['outcome']=='success'
    steps=read_steps(ep)
    assert [s['context']['phase'] for s in steps]==['settling','policy','policy','policy']
    assert [(s['context']['chunk_id'],s['context']['chunk_index']) for s in steps[1:]]==[(0,0),(0,1),(1,0)]
    a=load_arrays(ep,1)
    assert np.all(a['obs/agentview_image']==1)
    assert np.all(a['next_obs/agentview_image']==2)
    assert a['extra/raw_action'][-1] == .2
    assert a['action'][-1] == 1
    assert a['extra/query_actions'][0,-1] == .2
    assert a['extra/sim_state'][0] == 1
    assert a['extra/next_sim_state'][0] == 2


def test_done_does_not_override_explicit_failure(tmp_path):
    from vla_trajectory.oft import instrument
    from vla_trajectory.recorder import read_manifest
    m=evaluator()
    with instrument(m,tmp_path):
        m.run_episode(config(),Env(success=False),'test')
    assert read_manifest(next(tmp_path.iterdir()))['outcome']=='failure'


def test_timeout_and_swallowed_sim_error(tmp_path):
    from vla_trajectory.oft import instrument
    from vla_trajectory.recorder import read_manifest,read_steps
    m=evaluator()
    with instrument(m,tmp_path):
        m.run_episode(config(),Env(done_at=100),'timeout')
        m.run_episode(config(),Env(fail_at=3),'error')
    episodes=[(p,read_manifest(p)) for p in tmp_path.iterdir()]
    timeout=next(p for p,manifest in episodes if manifest['metadata']['task_description']=='timeout')
    error=next(manifest for p,manifest in episodes if manifest['metadata']['task_description']=='error')
    assert read_steps(timeout)[-1]['truncated'] is True
    assert error['status']=='error'
    assert error['num_steps']==2
    assert 'sim failed' in error['error']


def test_keyboard_interrupt_saved_and_propagated(tmp_path):
    from vla_trajectory.oft import instrument
    from vla_trajectory.recorder import read_manifest
    m=evaluator()
    def interrupt(*args,**kwargs):
        raise KeyboardInterrupt()
    m.get_action=interrupt
    with pytest.raises(KeyboardInterrupt):
        with instrument(m,tmp_path):
            m.run_episode(config(),Env(),'interrupted')
    assert read_manifest(next(tmp_path.iterdir()))['status']=='interrupted'


def test_missing_interface_fails_before_recording(tmp_path):
    from vla_trajectory.oft import instrument
    with pytest.raises(ValueError,match='run_episode'):
        with instrument(ModuleType('bad'),tmp_path):
            pass
    assert not list(tmp_path.iterdir())


def test_libero_predicate_errors_are_unknown():
    from vla_trajectory.libero import snapshot
    env=Env()
    env._eval_predicate=lambda _: None
    data=snapshot(SimpleNamespace(env=env))
    assert data['predicates']=={'on(cup, plate)': None}
    assert data['success'] is False


def test_disk_failure_cannot_be_swallowed_by_oft(tmp_path, monkeypatch):
    from vla_trajectory.oft import instrument
    from vla_trajectory.recorder import EpisodeRecorder, read_manifest
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(EpisodeRecorder, 'append', fail)
    m=evaluator()
    with pytest.raises(RuntimeError, match='recording failed'):
        with instrument(m,tmp_path):
            m.run_episode(config(),Env(),'disk test')
    manifest=read_manifest(next(tmp_path.iterdir()))
    assert manifest['status']=='error'
    assert manifest['num_steps']==0


def test_unknown_success_and_missing_predicates_stay_unknown(tmp_path):
    from vla_trajectory.oft import instrument
    from vla_trajectory.recorder import read_manifest
    env=Env()
    env.check_success=lambda: None
    env.parsed_problem={}
    m=evaluator()
    with instrument(m,tmp_path):
        m.run_episode(config(),env,'unknown')
    assert read_manifest(next(tmp_path.iterdir()))['outcome']=='unknown'


def test_truncated_action_queue_preserves_source_chunk_index(tmp_path):
    from vla_trajectory.oft import instrument
    from vla_trajectory.recorder import read_steps, load_arrays
    cfg=config()
    cfg.num_open_loop_steps=1
    m=evaluator()
    with instrument(m,tmp_path):
        m.run_episode(cfg,Env(),'short queue')
    ep=next(tmp_path.iterdir())
    step=read_steps(ep)[1]
    arrays=load_arrays(ep,1)
    assert step['context']['chunk_index']==1
    np.testing.assert_array_equal(arrays['extra/raw_action'], arrays['extra/query_actions'][1])


def test_initial_condition_survives_first_step_failure(tmp_path):
    from vla_trajectory.oft import instrument
    from vla_trajectory.recorder import read_manifest
    m=evaluator()
    with instrument(m,tmp_path):
        m.run_episode(config(),Env(fail_at=1),'initial state', initial_state=np.array([123.]))
    manifest=read_manifest(next(tmp_path.iterdir()))
    assert manifest['num_steps']==0
    assert manifest['status']=='error'
    assert manifest['metadata']['initial_state']==[123.]


def task_evaluator():
    """Exercise the official run_task/load_initial_states call boundaries."""
    m = evaluator()
    m.Env = Env
    exec('''
TASK_MAX_STEPS['libero_10'] = 3
model_loads = 0
def initialize_model(cfg):
    global model_loads
    model_loads += 1
    return object()
def load_initial_states(cfg, task_suite, task_id, log_file=None):
    return task_suite.get_task_init_states(task_id), None
def get_libero_env(task, model_family, resolution):
    return Env(), task.name
def run_task(cfg, task_suite, task_id, model, resize_size, processor=None,
             action_head=None, proprio_projector=None, noisy_action_projector=None,
             total_episodes=0, total_successes=0, log_file=None):
    task = task_suite.get_task(task_id)
    states, custom = load_initial_states(cfg, task_suite, task_id, log_file)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
    for episode_idx in range(cfg.num_trials_per_task):
        success, images = run_episode(cfg, env, task_description, model, resize_size,
                                     initial_state=states[episode_idx])
        total_episodes += 1
        total_successes += int(success)
    return total_episodes, total_successes
def evaluate(cfg, suite):
    model = initialize_model(cfg)
    episodes, successes = 0, 0
    for task_id in range(suite.n_tasks):
        episodes, successes = run_task(cfg, suite, task_id, model, 8,
                                     total_episodes=episodes, total_successes=successes)
    return episodes, successes
''', m.__dict__)
    return m


class Suite:
    n_tasks = 3
    def __init__(self, states=None):
        self.states = np.arange(60, dtype=float).reshape(30, 2) if states is None else states
    def get_task(self, task_id):
        from vla_trajectory.collection import DEFAULT_TASK_NAME
        return SimpleNamespace(name=DEFAULT_TASK_NAME if task_id == 1 else f'task-{task_id}')
    def get_task_init_states(self, task_id):
        return self.states


def plan_config():
    cfg = config()
    cfg.task_suite_name = 'libero_10'
    cfg.num_trials_per_task = 1
    cfg.initial_states_path = 'DEFAULT'
    cfg.pretrained_checkpoint = 'moojink/openvla-7b-oft-finetuned-libero-10'
    cfg.load_in_4bit = True
    return cfg


@pytest.mark.parametrize('stage,indices', [('pilot', [5, 1, 9, 3, 7]), ('formal', list(range(20)))])
def test_collection_filters_task_replays_each_state_once_and_loads_model_once(tmp_path, monkeypatch, stage, indices):
    from vla_trajectory.collection import create_plan, DEFAULT_TASK_NAME
    from vla_trajectory.oft import instrument
    from vla_trajectory.recorder import read_manifest, read_steps
    from vla_trajectory import video
    # Encoding is an external optional dependency; frame semantics have dedicated tests.
    def exported(path, **kwargs):
        target = path / 'complete.mp4'
        target.write_bytes(b'video')
        return target
    monkeypatch.setattr(video, 'export_episode_video', exported)
    m, cfg, suite = task_evaluator(), plan_config(), Suite()
    old_task, old_loader, old_factory = m.run_task, m.load_initial_states, m.get_libero_env
    closed_before = Env.closed_count
    original_config = vars(cfg).copy()
    plan = create_plan(DEFAULT_TASK_NAME, stage, indices)
    with instrument(m, tmp_path, collection_plan=plan):
        assert m.evaluate(cfg, suite) == (len(indices), len(indices))
    assert m.model_loads == 1
    assert vars(cfg) == original_config
    assert m.run_task is old_task and m.load_initial_states is old_loader
    assert m.get_libero_env is old_factory
    assert Env.closed_count == closed_before + len(indices)
    episodes = [p for p in tmp_path.iterdir() if p.is_dir()]
    manifests = [read_manifest(p) for p in episodes]
    assert sorted(m['metadata']['init_state_index'] for m in manifests) == sorted(indices)
    assert len({m['metadata']['state_sha256'] for m in manifests}) == len(indices)
    assert all(m['metadata']['task_id'] == 1 for m in manifests)
    resolved = json.loads(next(tmp_path.glob('collection_plan_*.json')).read_text())
    assert [r['init_state_index'] for r in resolved['rows']] == indices
    assert len(resolved['state_bank_sha256']) == 64
    for ep in episodes:
        data = json.loads((ep / 'execution.json').read_text())
        assert data['executed_settling_steps'] == data['recorded_settling_steps'] == 1
        assert data['executed_policy_steps'] == data['recorded_policy_steps'] == 3
        assert data['recording_complete'] is True and data['step_count_complete'] is True
        assert data['video_status'] == 'complete'
        assert [s['context']['action_step_id'] for s in read_steps(ep)] == [None, 1, 2, 3]


@pytest.mark.parametrize('mutation', ['duplicate', 'range', 'trials', 'suite', 'custom'])
def test_collection_rejects_bad_states_or_config_before_episodes(tmp_path, mutation):
    from vla_trajectory.collection import create_plan, DEFAULT_TASK_NAME
    from vla_trajectory.oft import instrument
    cfg, suite, indices = plan_config(), Suite(), list(range(5))
    if mutation == 'duplicate':
        suite.states[1] = suite.states[0]
    elif mutation == 'range':
        indices = [0, 1, 2, 3, 99]
    elif mutation == 'trials':
        cfg.num_trials_per_task = 5
    elif mutation == 'suite':
        cfg.task_suite_name = 'wrong'
    else:
        cfg.initial_states_path = 'custom.json'
    m = task_evaluator()
    original = m.run_task
    with pytest.raises(ValueError):
        with instrument(m, tmp_path, collection_plan=create_plan(DEFAULT_TASK_NAME, 'pilot', indices)):
            m.evaluate(cfg, suite)
    assert m.run_task is original
    assert not list(tmp_path.glob('*/manifest.json'))


def test_execution_count_survives_disk_failure_and_uncertain_env_step(tmp_path, monkeypatch):
    from vla_trajectory.oft import instrument
    from vla_trajectory.recorder import EpisodeRecorder
    original_append = EpisodeRecorder.append
    def fail_policy(self, *args, **kwargs):
        if kwargs['context']['phase'] == 'policy':
            raise OSError('disk full')
        return original_append(self, *args, **kwargs)
    monkeypatch.setattr(EpisodeRecorder, 'append', fail_policy)
    with pytest.raises(RuntimeError, match='recording failed'):
        m = evaluator()
        with instrument(m, tmp_path / 'disk'):
            m.run_episode(config(), Env(), 'disk')
    data = json.loads(next((tmp_path / 'disk').glob('*/execution.json')).read_text())
    assert data['executed_policy_steps'] == 1 and data['recorded_policy_steps'] == 0
    assert data['recording_complete'] is False and data['step_count_complete'] is True
    monkeypatch.setattr(EpisodeRecorder, 'append', original_append)
    m = evaluator()
    with instrument(m, tmp_path / 'sim'):
        m.run_episode(config(), Env(fail_at=3), 'sim')
    data = json.loads(next((tmp_path / 'sim').glob('*/execution.json')).read_text())
    assert data['executed_policy_steps'] == data['recorded_policy_steps'] == 1
    assert data['step_count_complete'] is False


def test_plan_video_failure_is_reported_and_propagated(tmp_path, monkeypatch):
    from vla_trajectory.collection import create_plan, DEFAULT_TASK_NAME
    from vla_trajectory.oft import instrument
    from vla_trajectory import video
    def fail(*args, **kwargs):
        raise RuntimeError('codec failed')
    monkeypatch.setattr(video, 'export_episode_video', fail)
    m = task_evaluator()
    with pytest.raises(RuntimeError, match='video'):
        with instrument(m, tmp_path, collection_plan=create_plan(DEFAULT_TASK_NAME, 'pilot')):
            m.evaluate(plan_config(), Suite())
    data = json.loads(next(tmp_path.glob('*/execution.json')).read_text())
    assert data['video_status'] == 'error' and 'codec failed' in data['video_error']


def test_first_observation_field_mapping_reports_missing_target_pose(tmp_path):
    from vla_trajectory.oft import instrument
    env = Env()
    env.obs.update(robot0_eef_pos=np.array([1., 2., 3.]), robot0_eef_quat=np.array([0., 0., 0., 1.]),
                   robot0_gripper_qpos=np.array([.01, -.01]), alphabet_soup_1_pos=np.array([.1, .2, .3]),
                   alphabet_soup_1_quat=np.array([0., 0., 0., 1.]))
    m = evaluator()
    with instrument(m, tmp_path, run_metadata={'analysis_object_pair': ['alphabet_soup_1', 'tomato_sauce_1']}):
        m.run_episode(config(), env, 'fields')
    data = json.loads(next(tmp_path.glob('*/execution.json')).read_text())
    fields = data['observation_fields']
    assert fields['eef_position']['key'] == 'robot0_eef_pos'
    assert fields['eef_quaternion']['order'] == 'xyzw'
    assert fields['objects']['alphabet_soup_1']['position']['present'] is True
    assert fields['objects']['tomato_sauce_1']['position']['present'] is False


def test_recorded_counts_follow_manifest_commit_even_if_append_raises_after_commit(tmp_path, monkeypatch):
    from vla_trajectory.oft import instrument
    from vla_trajectory.recorder import EpisodeRecorder
    original = EpisodeRecorder.append
    def committed_then_failed(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise OSError('post-commit interruption')
    monkeypatch.setattr(EpisodeRecorder, 'append', committed_then_failed)
    m = evaluator()
    with pytest.raises(RuntimeError, match='recording failed'):
        with instrument(m, tmp_path):
            m.run_episode(config(), Env(), 'committed')
    data = json.loads(next(tmp_path.glob('*/execution.json')).read_text())
    assert data['executed_settling_steps'] == data['recorded_settling_steps'] == 1
    assert data['recording_complete'] is True


def test_plan_fingerprint_cannot_overwrite_different_state_bank(tmp_path, monkeypatch):
    from vla_trajectory.collection import create_plan, DEFAULT_TASK_NAME
    from vla_trajectory.oft import instrument
    from vla_trajectory import video
    monkeypatch.setattr(video, 'export_episode_video', lambda path: path / 'test.mp4')
    plan = create_plan(DEFAULT_TASK_NAME, 'pilot')
    m, cfg = task_evaluator(), plan_config()
    with instrument(m, tmp_path, collection_plan=plan):
        m.evaluate(cfg, Suite())
    before = next(tmp_path.glob('collection_plan_*.json')).read_bytes()
    changed = Suite()
    changed.states += 1
    with instrument(m, tmp_path, collection_plan=plan):
        # A caller catching validation errors cannot bypass the persisted-bank
        # check by trying a second task invocation in the same context.
        for unused in range(2):
            with pytest.raises(ValueError, match='state bank'):
                m.evaluate(cfg, changed)
    assert next(tmp_path.glob('collection_plan_*.json')).read_bytes() == before


def test_plan_releases_environment_when_export_fails(tmp_path, monkeypatch):
    from vla_trajectory.collection import create_plan, DEFAULT_TASK_NAME
    from vla_trajectory.oft import instrument
    from vla_trajectory import video
    def fail(path):
        raise RuntimeError('codec failed')
    monkeypatch.setattr(video, 'export_episode_video', fail)
    before = Env.closed_count
    m = task_evaluator()
    with pytest.raises(RuntimeError, match='video'):
        with instrument(m, tmp_path, collection_plan=create_plan(DEFAULT_TASK_NAME, 'pilot')):
            m.evaluate(plan_config(), Suite())
    assert Env.closed_count == before + 1


@pytest.mark.parametrize('failure', ['video', 'recording', 'interrupt', 'system_exit', 'none'])
def test_cleanup_preserves_primary_failure_and_attaches_secondary_error(tmp_path, monkeypatch, failure):
    from vla_trajectory.collection import create_plan, DEFAULT_TASK_NAME
    from vla_trajectory.oft import instrument
    from vla_trajectory.recorder import EpisodeRecorder
    from vla_trajectory import video
    m = task_evaluator()
    originals = (m.run_task, m.run_episode)
    def close_fails(self):
        raise OSError('close failed')
    monkeypatch.setattr(Env, 'close', close_fails)
    monkeypatch.setattr(video, 'export_episode_video', lambda path: path / 'complete.mp4')
    if failure == 'video':
        def video_fails(path):
            raise RuntimeError('codec failed')
        monkeypatch.setattr(video, 'export_episode_video', video_fails)
        expected, match = RuntimeError, 'Episode video export failed: codec failed'
    elif failure == 'recording':
        def recording_fails(*args, **kwargs):
            raise OSError('disk full')
        monkeypatch.setattr(EpisodeRecorder, 'append', recording_fails)
        expected, match = RuntimeError, 'Trajectory recording failed'
    elif failure in ('interrupt', 'system_exit'):
        primary = KeyboardInterrupt('user stop') if failure == 'interrupt' else SystemExit('shutdown')
        def interrupted(*args, **kwargs):
            raise primary
        m.get_action = interrupted
        expected, match = type(primary), str(primary)
    else:
        expected, match = OSError, 'close failed'
    with pytest.raises(expected, match=match) as captured:
        with instrument(m, tmp_path, collection_plan=create_plan(DEFAULT_TASK_NAME, 'pilot')):
            m.evaluate(plan_config(), Suite())
    if failure != 'none':
        assert captured.value.cleanup_errors == ['OSError: close failed']
        if hasattr(captured.value, 'add_note'):
            assert any('OSError: close failed' in note for note in captured.value.__notes__)
    if failure in ('interrupt', 'system_exit'):
        assert captured.value is primary
    assert (m.run_task, m.run_episode) == originals
    assert not hasattr(m, '_vla_trace_active')


def test_factory_environment_is_closed_when_pre_episode_logging_fails(tmp_path):
    from vla_trajectory.collection import create_plan, DEFAULT_TASK_NAME
    from vla_trajectory.oft import instrument
    m = task_evaluator()
    m.get_libero_env = lambda task, family, resolution: (Env(), task.name)
    def log_fails(message, *args, **kwargs):
        raise OSError('log full')
    m.log_message = log_fails
    exec('''
def run_task(cfg, task_suite, task_id, model, resize_size, processor=None,
             action_head=None, proprio_projector=None, noisy_action_projector=None,
             total_episodes=0, total_successes=0, log_file=None):
    task = task_suite.get_task(task_id)
    states, custom = load_initial_states(cfg, task_suite, task_id, log_file)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
    log_message('Starting episode...', log_file)
    raise AssertionError('pre-episode logging should have failed')
''', m.__dict__)
    factory = m.get_libero_env
    before = Env.closed_count
    with pytest.raises(OSError, match='log full'):
        with instrument(m, tmp_path, collection_plan=create_plan(DEFAULT_TASK_NAME, 'pilot')):
            m.evaluate(plan_config(), Suite())
    assert Env.closed_count == before + 1
    assert m.get_libero_env is factory
    assert not list(tmp_path.glob('*/manifest.json'))


def instruction_evaluator(*, keyword_query=False, query_text=None, inference_failure=False):
    """Keep the real wrapper/storage boundary while replacing GPU inference."""
    m = evaluator()
    m.episode_instructions = []
    m.model_instructions = []
    m.pending_queries_seen = []
    m.query_text = query_text
    m.inference_failure = inference_failure
    call = ('get_action(cfg=cfg, model=model, observation=obs, task_description=instruction)'
            if keyword_query else 'get_action(cfg, model, obs, instruction)')
    exec('''
def get_action(cfg, model, observation, task_description, **kwargs):
    model_instructions.append(task_description)
    from pathlib import Path
    import json
    pending_queries_seen.append(json.loads(next(Path(output_root).glob('*/queries.json')).read_text(encoding='utf-8')))
    if inference_failure:
        raise RuntimeError('model unavailable')
    return np.zeros((1, 7))
def run_episode(cfg, env, task_description, model=None, initial_state=None):
    episode_instructions.append(task_description)
    obs = env.reset()
    try:
        env.step([0] * 7)
        instruction = task_description if query_text is None else query_text
        actions = QUERY_CALL
        env.step(actions[0])
    except Exception as exc:
        log_message(f'Episode error: {exc}')
    return False, []
'''.replace('QUERY_CALL', call), m.__dict__)
    return m


@pytest.mark.parametrize('keyword_query', [False, True])
def test_instruction_override_reaches_episode_and_model_with_query_evidence(tmp_path, keyword_query):
    from vla_trajectory.oft import instrument
    from vla_trajectory.recorder import read_manifest, read_steps
    m = instruction_evaluator(keyword_query=keyword_query)
    m.output_root = str(tmp_path)
    originals = m.run_episode, m.get_action, m.process_action, m.log_message
    prompt = 'First put cheese away; then move butter.'
    digest = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    with instrument(m, tmp_path, instruction_override=prompt):
        m.run_episode(config(), Env(), 'original task')
    assert m.episode_instructions == m.model_instructions == [prompt]
    assert (m.run_episode, m.get_action, m.process_action, m.log_message) == originals
    assert not hasattr(m, '_vla_trace_active')
    ep = next(tmp_path.iterdir())
    manifest = read_manifest(ep)
    assert manifest['metadata']['task_description'] == prompt
    assert manifest['metadata']['original_task_description'] == 'original task'
    assert manifest['metadata']['original_task_description_sha256'] == hashlib.sha256(b'original task').hexdigest()
    assert manifest['metadata']['instruction_sha256'] == digest
    assert manifest['metadata']['instruction_override'] is True
    evidence = json.loads((ep / 'queries.json').read_text(encoding='utf-8'))
    assert evidence['schema_version'] == 1
    assert evidence['episode_id'] == manifest['episode_id']
    assert evidence['queries'] == [{'chunk_id': 0, 'instruction': prompt,
        'instruction_sha256': digest, 'status': 'complete', 'error': None}]
    assert m.pending_queries_seen[0]['queries'][0]['status'] == 'pending'
    context = read_steps(ep)[1]['context']
    assert context['instruction'] == prompt and context['instruction_sha256'] == digest
    assert read_steps(ep)[0]['context'].get('instruction') is None


def test_instruction_mismatch_is_not_hidden_by_upstream_exception_handler(tmp_path):
    from vla_trajectory.oft import instrument
    from vla_trajectory.recorder import read_manifest
    m = instruction_evaluator(query_text='stale task text')
    m.output_root = str(tmp_path)
    original = m.run_episode
    with pytest.raises(RuntimeError, match='instruction'):
        with instrument(m, tmp_path, instruction_override='first cheese then butter'):
            m.run_episode(config(), Env(), 'original task')
    assert m.model_instructions == []
    assert m.run_episode is original
    ep = next(tmp_path.iterdir())
    manifest = read_manifest(ep)
    assert manifest['status'] == 'error' and manifest['outcome'] == 'unknown'
    assert 'instruction' in manifest['error']
    row = json.loads((ep / 'queries.json').read_text())['queries'][0]
    assert row['instruction'] == 'stale task text'
    assert row['instruction_sha256'] == hashlib.sha256(b'stale task text').hexdigest()
    assert row['status'] == 'error' and 'instruction' in row['error']


def test_failed_model_query_still_has_a_failed_query_record(tmp_path):
    from vla_trajectory.oft import instrument
    from vla_trajectory.recorder import read_manifest, read_steps
    m = instruction_evaluator(inference_failure=True)
    m.output_root = str(tmp_path)
    with instrument(m, tmp_path, instruction_override='first A then B'):
        m.run_episode(config(), Env(), 'original')
    ep = next(tmp_path.iterdir())
    assert read_manifest(ep)['status'] == 'error'
    assert len(read_steps(ep)) == 1
    row = json.loads((ep / 'queries.json').read_text())['queries'][0]
    assert row['chunk_id'] == 0 and row['instruction'] == 'first A then B'
    assert row['status'] == 'error' and 'model unavailable' in row['error']


@pytest.mark.parametrize('value', ['', '  \t\n', 7, False])
def test_invalid_instruction_override_rejected_before_writing(tmp_path, value):
    from vla_trajectory.oft import instrument
    m = evaluator()
    original = m.run_episode
    with pytest.raises(ValueError, match='instruction_override'):
        with instrument(m, tmp_path, instruction_override=value):
            m.run_episode(config(), Env(), 'original')
    assert not list(tmp_path.iterdir())
    assert m.run_episode is original and not hasattr(m, '_vla_trace_active')


@pytest.mark.parametrize('value', ['', '  ', 7])
def test_invalid_original_task_description_rejected_before_writing(tmp_path, value):
    from vla_trajectory.oft import instrument
    m = evaluator()
    with pytest.raises(ValueError, match='task_description'):
        with instrument(m, tmp_path, instruction_override='valid instruction'):
            m.run_episode(config(), Env(), value)
    assert not list(tmp_path.iterdir())


def test_opaque_get_action_signature_only_rejected_when_override_needed(tmp_path):
    from vla_trajectory.oft import instrument
    from vla_trajectory.recorder import read_manifest
    m = evaluator()
    action = m.get_action
    m.get_action = lambda *args, **kwargs: action(*args, **kwargs)
    with pytest.raises(ValueError, match='get_action.*task_description'):
        with instrument(m, tmp_path / 'override', instruction_override='first A then B'):
            pass
    assert not (tmp_path / 'override').exists()
    with instrument(m, tmp_path / 'native'):
        m.run_episode(config(), Env(), 'native instruction')
    assert read_manifest(next((tmp_path / 'native').iterdir()))['status'] == 'complete'


def test_query_storage_failure_is_not_swallowed_and_prevents_inference(tmp_path, monkeypatch):
    from vla_trajectory import oft
    from vla_trajectory.recorder import read_manifest
    original_write = oft._write_json
    def fail_query_write(path, document):
        if path.name == 'queries.json' and document['queries']:
            raise OSError('query storage full')
        return original_write(path, document)
    monkeypatch.setattr(oft, '_write_json', fail_query_write)
    m = instruction_evaluator()
    m.output_root = str(tmp_path)
    with pytest.raises(RuntimeError, match='query recording'):
        with oft.instrument(m, tmp_path, instruction_override='first A then B'):
            m.run_episode(config(), Env(), 'original')
    assert m.model_instructions == []
    assert read_manifest(next(tmp_path.iterdir()))['status'] == 'error'


@pytest.mark.parametrize('skipped', [set(range(5)), {0, 2, 4}])
def test_collection_row_filter_skips_before_environment_creation(tmp_path, monkeypatch, skipped):
    from vla_trajectory.collection import create_plan, DEFAULT_TASK_NAME
    from vla_trajectory.oft import instrument
    from vla_trajectory.recorder import read_manifest
    from vla_trajectory import video
    monkeypatch.setattr(video, 'export_episode_video', lambda path: path / 'complete.mp4')
    m, cfg = task_evaluator(), plan_config()
    created = []
    factory = m.get_libero_env
    def counted_factory(*args, **kwargs):
        result = factory(*args, **kwargs)
        created.append(result[0])
        return result
    m.get_libero_env = counted_factory
    closed_before = Env.closed_count
    seen_rows = []
    def include_row(actual_cfg, row):
        assert actual_cfg is cfg
        seen_rows.append(row['init_state_index'])
        include = row['init_state_index'] not in skipped
        row['init_state_index'] = 99  # Callback receives a detached row.
        return include
    with instrument(m, tmp_path, collection_plan=create_plan(DEFAULT_TASK_NAME, 'pilot'),
                    row_filter=include_row):
        totals = m.run_task(cfg, Suite(), 1, object(), 8, total_episodes=8, total_successes=3)
    assert seen_rows == [0, 1, 2, 3, 4]
    assert totals == (8 + 5 - len(skipped), 3 + 5 - len(skipped))
    assert Env.closed_count == closed_before + 5 - len(skipped)
    assert len(created) == 5 - len(skipped)
    manifests = [read_manifest(path.parent) for path in tmp_path.glob('*/manifest.json')]
    assert sorted(item['metadata']['init_state_index'] for item in manifests) == sorted(set(range(5)) - skipped)


def test_invalid_row_filter_rejected_before_writing(tmp_path):
    from vla_trajectory.oft import instrument
    with pytest.raises(ValueError, match='row_filter'):
        with instrument(evaluator(), tmp_path, row_filter=True):
            pass
    assert not list(tmp_path.iterdir())


def official_instruction_evaluator(*, keyword_query, stale_query=False):
    """Mirror robot_utils.get_action's obs/task_label parameter names."""
    m = instruction_evaluator(query_text='stale instruction' if stale_query else None)
    call = ('get_action(cfg=cfg, model=model, obs=obs, task_label=instruction)'
            if keyword_query else 'get_action(cfg, model, obs, instruction)')
    exec('''
def get_action(cfg, model, obs, task_label, processor=None, action_head=None,
               proprio_projector=None, noisy_action_projector=None, use_film=False):
    model_instructions.append(task_label)
    # Read the observation so an unbound obs alias cannot silently pass.
    return np.zeros((1, 7)) + float(obs['state'][0])
def run_episode(cfg, env, task_description, model=None, initial_state=None):
    episode_instructions.append(task_description)
    obs = env.reset()
    try:
        env.step([0] * 7)
        instruction = task_description if query_text is None else query_text
        actions = QUERY_CALL
        env.step(actions[0])
    except Exception as exc:
        log_message(f'Episode error: {exc}')
    return False, []
'''.replace('QUERY_CALL', call), m.__dict__)
    return m


@pytest.mark.parametrize('keyword_query', [False, True])
@pytest.mark.parametrize('stale_query', [False, True])
def test_official_obs_task_label_signature_verifies_exact_instruction(tmp_path, keyword_query, stale_query):
    from vla_trajectory.oft import instrument
    from vla_trajectory.recorder import read_manifest, read_steps
    m = official_instruction_evaluator(keyword_query=keyword_query, stale_query=stale_query)
    originals = m.run_episode, m.get_action
    prompt = 'first cheese then butter'
    if stale_query:
        with pytest.raises(RuntimeError, match='instruction'):
            with instrument(m, tmp_path, instruction_override=prompt):
                m.run_episode(config(), Env(), 'official original')
    else:
        with instrument(m, tmp_path, instruction_override=prompt):
            m.run_episode(config(), Env(), 'official original')
    assert (m.run_episode, m.get_action) == originals
    ep = next(tmp_path.iterdir())
    manifest = read_manifest(ep)
    assert manifest['metadata']['original_task_description'] == 'official original'
    assert manifest['metadata']['task_description'] == prompt
    query = json.loads((ep / 'queries.json').read_text())['queries'][0]
    actual = 'stale instruction' if stale_query else prompt
    assert query['instruction'] == actual
    assert query['instruction_sha256'] == hashlib.sha256(actual.encode()).hexdigest()
    assert query['status'] == ('error' if stale_query else 'complete')
    assert manifest['status'] == ('error' if stale_query else 'complete')
    assert m.model_instructions == ([] if stale_query else [prompt])
    if not stale_query:
        assert read_steps(ep)[1]['context']['instruction'] == prompt


def test_official_keyword_query_without_override_preserves_native_instruction(tmp_path):
    from vla_trajectory.oft import instrument
    from vla_trajectory.recorder import read_manifest
    m = official_instruction_evaluator(keyword_query=True)
    with instrument(m, tmp_path):
        m.run_episode(config(), Env(), 'native official instruction')
    ep = next(tmp_path.iterdir())
    assert read_manifest(ep)['status'] == 'complete'
    query = json.loads((ep / 'queries.json').read_text())['queries'][0]
    assert query['instruction'] == 'native official instruction'
    assert query['status'] == 'complete'

