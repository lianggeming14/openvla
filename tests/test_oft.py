from types import ModuleType, SimpleNamespace
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

