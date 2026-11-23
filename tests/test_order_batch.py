"""Protect pairing, actual checkpoint identity and conservative recovery."""
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest


def registry():
    return {
        'schema_version': 1, 'task_suite_name': 'libero_10',
        'fixed_model_configuration': {
            'pretrained_checkpoint': 'moojink/openvla-7b-oft-finetuned-libero-10',
            'load_in_4bit': True, 'num_trials_per_task': 1},
        'groups': [
            {'group_id': i, 'task_name': f'task-{i}',
             'completion_order': {'simulator_subgoal_keys': {'A': 'on(a, x)', 'B': 'on(b, y)'}},
             'grasp_order': {'object_pair': ['a', 'b'] if i < 6 else None},
             'conditions': [
                 {'condition_id': name, 'enabled': True, 'instruction': text,
                  'expected_order': order, 'episodes_planned': 20,
                  'state_indices': list(range(20))}
                 for name, text, order in [('AB', 'First A, then B.', ['A', 'B']),
                                           ('BA', 'First B, then A.', ['B', 'A'])]]}
            for i in range(1, 7)]}


def test_cases_keep_twenty_states_paired_and_reverse_only_instruction():
    from vla_trajectory.order_batch import create_cases
    cases = create_cases(registry())
    assert len(cases) == 12
    assert sum(len(c['plan']['init_state_indices']) for c in cases) == 240
    assert cases[0]['plan']['init_state_indices'] == cases[1]['plan']['init_state_indices'] == list(range(20))
    assert cases[0]['instruction'] == 'First A, then B.'
    assert cases[1]['instruction'] == 'First B, then A.'
    assert cases[0]['expected_order'] == 'AB'
    assert cases[1]['expected_order'] == 'BA'
    assert cases[-1]['plan']['object_pair'] is None


@pytest.mark.parametrize('mutation', ['unpaired', 'duplicate', 'wrong_model', 'blank', 'extra_condition'])
def test_registry_errors_do_not_produce_an_ambiguous_batch(mutation):
    from vla_trajectory.order_batch import create_cases
    value = registry()
    if mutation == 'unpaired':
        value['groups'][0]['conditions'][1]['state_indices'][-1] = 22
    elif mutation == 'duplicate':
        value['groups'][0]['conditions'][0]['state_indices'][-1] = 0
    elif mutation == 'wrong_model':
        value['fixed_model_configuration']['load_in_4bit'] = False
    elif mutation == 'blank':
        value['groups'][0]['conditions'][0]['instruction'] = '  '
    else:
        value['groups'][0]['conditions'].append(copy.deepcopy(value['groups'][0]['conditions'][0]))
    with pytest.raises(ValueError):
        create_cases(value)


def checkpoint(path):
    path.mkdir()
    (path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': 'model-00001-of-00001.safetensors'}}))
    for name in ('model-00001-of-00001.safetensors', 'action_head--150000_checkpoint.pt',
                 'proprio_projector--150000_checkpoint.pt', 'dataset_statistics.json', 'config.json'):
        (path / name).write_bytes(b'checkpoint-fixture')
    (path / 'config.json').write_text('{"hidden_size": 16}')
    return path


def test_checkpoint_identity_detects_changed_weight_bytes(tmp_path):
    from vla_trajectory.order_batch import fingerprint_checkpoint
    root = checkpoint(tmp_path / 'checkpoint')
    first = fingerprint_checkpoint(root)
    (root / 'model-00001-of-00001.safetensors').write_bytes(b'changed')
    assert fingerprint_checkpoint(root)['sha256'] != first['sha256']
    assert len(first['files']) == 6


def test_checkpoint_missing_shard_cannot_be_treated_as_downloaded(tmp_path):
    from vla_trajectory.order_batch import fingerprint_checkpoint
    root = checkpoint(tmp_path / 'checkpoint')
    (root / 'model-00001-of-00001.safetensors').unlink()
    with pytest.raises(ValueError, match='shard'):
        fingerprint_checkpoint(root)


def test_oft_generated_backups_and_auto_map_do_not_restart_verified_experiment(tmp_path):
    from vla_trajectory.order_batch import fingerprint_checkpoint
    root = checkpoint(tmp_path / 'checkpoint')
    first = fingerprint_checkpoint(root)['sha256']
    (root / 'config.json.back.20261123_120000').write_text('original backup')
    (root / 'modeling_prismatic.py').write_text('synced from fingerprinted OFT source')
    config = {'hidden_size': 16, 'auto_map': {
        'AutoConfig': 'configuration_prismatic.OpenVLAConfig',
        'AutoModelForVision2Seq': 'modeling_prismatic.OpenVLAForActionPrediction'}}
    (root / 'config.json').write_text(json.dumps(config, indent=2))
    assert fingerprint_checkpoint(root)['sha256'] == first
    config['hidden_size'] = 32
    (root / 'config.json').write_text(json.dumps(config))
    assert fingerprint_checkpoint(root)['sha256'] != first


def test_resolved_pairs_reject_bank_change_before_any_environment(tmp_path):
    from vla_trajectory.order_batch import create_cases, resolve_cases, execute_case
    from test_oft import task_evaluator, plan_config, Suite
    from vla_trajectory.collection import DEFAULT_TASK_NAME
    value = registry()
    value['groups'][0]['task_name'] = DEFAULT_TASK_NAME
    # This test resolves only one real-shaped task pair, since our external
    # suite double has three tasks. Production resolves all twelve cases.
    pair = resolve_cases(create_cases(value)[:2], Suite())
    assert pair[0]['resolved_rows'][0]['state_sha256'] == pair[1]['resolved_rows'][0]['state_sha256']
    module, cfg, changed = task_evaluator(), plan_config(), Suite()
    changed.states[0, 0] = 900
    module.eval_libero = lambda: module.evaluate(cfg, changed)
    with pytest.raises(ValueError, match='fingerprint'):
        execute_case(module, pair[1], tmp_path, experiment_identity='changed-bank')
    assert not list(tmp_path.glob('*/manifest.json'))


def test_safe_numpy_context_loads_numeric_libero_style_bank(tmp_path):
    torch = pytest.importorskip('torch')
    from vla_trajectory.order_batch import numpy_init_state_context
    path = tmp_path / 'trusted_numeric_bank.pt'
    torch.save(np.array([[1., 2.], [3., 4.]]), path)
    with numpy_init_state_context():
        states = torch.load(path)
    assert np.array_equal(states, [[1., 2.], [3., 4.]])


def recorded_episode(tmp_path):
    from vla_trajectory.recorder import EpisodeRecorder, _write_json
    prompt = 'First A, then B.'
    cfg = SimpleNamespace(task_suite_name='libero_10', num_trials_per_task=1,
                          load_in_4bit=True, pretrained_checkpoint='local/model', seed=7)
    row = {'task_name': 'task-1', 'init_state_index': 0, 'state_sha256': 'abc', 'plan_hash': 'plan'}
    recorder = EpisodeRecorder(tmp_path, {
        **row, 'experiment_identity': 'identity', 'config': vars(cfg),
        'task_description': prompt, 'instruction_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
        'instruction_override': True}, episode_id='attempt-one')
    obs = {'agentview_image': np.zeros((2, 2, 3), dtype=np.uint8),
           'robot0_eye_in_hand_image': np.zeros((2, 2, 3), dtype=np.uint8)}
    recorder.append(obs, np.zeros(7), obs, context={'phase': 'policy', 'chunk_id': 0})
    recorder.finish(outcome='failure')
    _write_json(recorder.path / 'execution.json', {
        'episode_id': 'attempt-one', 'recording_complete': True, 'step_count_complete': True,
        'executed_policy_steps': 1, 'recorded_policy_steps': 1,
        'executed_settling_steps': 0, 'recorded_settling_steps': 0,
        'video_status': 'complete', 'video_path': 'episode.mp4'})
    _write_json(recorder.path / 'queries.json', {'schema_version': 1, 'episode_id': 'attempt-one',
        'queries': [{'chunk_id': 0, 'instruction': prompt, 'instruction_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'status': 'complete'}]})
    (recorder.path / 'episode.mp4').write_bytes(b'encoder-output-fixture')
    _write_json(recorder.path / 'episode.json', {'episode_id': 'attempt-one', 'frames': [
        {'video_frame': 0, 'record_step': 0, 'observation': 'obs'},
        {'video_frame': 1, 'record_step': 0, 'observation': 'next_obs'}]})
    return recorder.path, row, cfg, prompt


def test_verified_failure_is_a_completed_trial_and_is_not_rerun(tmp_path):
    from vla_trajectory.order_batch import find_completed_episode
    episode, row, cfg, prompt = recorded_episode(tmp_path)
    assert find_completed_episode(tmp_path, row=row, cfg=cfg, experiment_identity='identity', instruction=prompt) == episode


@pytest.mark.parametrize('mutation', ['query', 'video', 'arrays', 'execution', 'config', 'identity', 'state'])
def test_resume_rejects_missing_or_mismatched_evidence(tmp_path, mutation):
    from vla_trajectory.order_batch import find_completed_episode
    episode, row, cfg, prompt = recorded_episode(tmp_path)
    identity = 'identity'
    if mutation == 'query':
        value = json.loads((episode / 'queries.json').read_text())
        value['queries'][0]['instruction'] = 'Wrong actual prompt'
        (episode / 'queries.json').write_text(json.dumps(value))
    elif mutation == 'video':
        (episode / 'episode.mp4').unlink()
    elif mutation == 'arrays':
        (episode / 'steps/000000.npz').write_bytes(b'corrupt')
    elif mutation == 'execution':
        value = json.loads((episode / 'execution.json').read_text())
        value['recording_complete'] = False
        (episode / 'execution.json').write_text(json.dumps(value))
    elif mutation == 'config':
        cfg.seed = 8
    elif mutation == 'identity':
        identity = 'other-checkpoint-identity'
    else:
        row['state_sha256'] = 'different-state'
    assert find_completed_episode(tmp_path, row=row, cfg=cfg, experiment_identity=identity, instruction=prompt) is None


def test_execute_case_uses_real_transport_and_skips_verified_rows(tmp_path, monkeypatch):
    from vla_trajectory.order_batch import create_cases, execute_case
    from vla_trajectory.recorder import _write_json, read_manifest, read_steps
    from vla_trajectory import video
    from test_oft import task_evaluator, plan_config, Suite
    from vla_trajectory.collection import DEFAULT_TASK_NAME

    # Only the external video encoder/model/environment are doubles; the plan,
    # instruction transport, raw recording and recovery run as production code.
    def encode(episode):
        manifest, steps = read_manifest(episode), read_steps(episode)
        path = episode / 'encoded.mp4'
        path.write_bytes(b'external-encoder-fixture')
        frames = [{'video_frame': 0, 'record_step': 0, 'observation': 'obs'}]
        frames.extend({'video_frame': i + 1, 'record_step': i, 'observation': 'next_obs'}
                      for i in range(len(steps)))
        _write_json(path.with_suffix('.json'), {'episode_id': manifest['episode_id'], 'frames': frames})
        return path

    monkeypatch.setattr(video, 'export_episode_video', encode)
    value = registry()
    value['groups'][0]['task_name'] = DEFAULT_TASK_NAME
    case = create_cases(value)[0]
    module, cfg, suite = task_evaluator(), plan_config(), Suite()
    original_factory = module.get_libero_env
    def factory(*args, **kwargs):
        env, description = original_factory(*args, **kwargs)
        env.obs['robot0_eye_in_hand_image'] = np.zeros((8, 8, 3), dtype=np.uint8)
        return env, description
    module.get_libero_env = factory
    module.eval_libero = lambda: module.evaluate(cfg, suite)
    first = execute_case(module, case, tmp_path, experiment_identity='fixture-identity')
    assert first['completed'] == first['executed'] == 20
    assert first['skipped'] == 0
    second = execute_case(module, case, tmp_path, experiment_identity='fixture-identity')
    assert second['completed'] == second['skipped'] == 20
    assert second['executed'] == 0
    assert len(list(tmp_path.glob('*/manifest.json'))) == 20


def test_script_describe_does_not_import_or_run_a_policy(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    source = tmp_path / 'registry.json'
    source.write_text(json.dumps(registry()), encoding='utf-8')
    result = subprocess.run([sys.executable, 'scripts/run_order_experiment.py', 'describe',
                             '--registry', str(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert document['planned_episodes'] == 240
    assert document['contains_experiment_results'] is False
    assert len(document['cases']) == 12


def test_disk_guard_stops_before_constructing_another_environment(tmp_path, monkeypatch):
    import shutil
    from vla_trajectory.order_batch import create_cases, execute_case
    from test_oft import task_evaluator, plan_config, Suite
    from vla_trajectory.collection import DEFAULT_TASK_NAME
    value = registry()
    value['groups'][0]['task_name'] = DEFAULT_TASK_NAME
    module, cfg, suite = task_evaluator(), plan_config(), Suite()
    module.eval_libero = lambda: module.evaluate(cfg, suite)
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: SimpleNamespace(free=1))
    with pytest.raises(RuntimeError, match='disk space'):
        execute_case(module, create_cases(value)[0], tmp_path, experiment_identity='disk-test')
    assert not list(tmp_path.glob('*/manifest.json'))
