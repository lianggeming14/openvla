import pytest


def test_plan_defaults_and_detached_validation():
    from vla_trajectory.collection import create_plan, validate_plan, DEFAULT_TASK_NAME
    pilot = create_plan(DEFAULT_TASK_NAME, 'pilot')
    assert pilot['init_state_indices'] == [0, 1, 2, 3, 4]
    assert pilot['object_pair'] == ['alphabet_soup_1', 'tomato_sauce_1']
    formal = create_plan('another_task', 'formal')
    assert formal['init_state_indices'] == list(range(20))
    assert formal['object_pair'] is None
    checked = validate_plan(pilot)
    checked['init_state_indices'][0] = 99
    assert pilot['init_state_indices'][0] == 0


@pytest.mark.parametrize('stage,indices', [
    ('pilot', [0, 1, 2, 3]), ('formal', list(range(19))),
    ('pilot', [0, 1, 2, 3, 3]), ('pilot', [-1, 1, 2, 3, 4]),
    ('pilot', [True, 1, 2, 3, 4]), ('bad', list(range(5))),
])
def test_invalid_plan_rejected(stage, indices):
    from vla_trajectory.collection import create_plan
    with pytest.raises(ValueError):
        create_plan('task', stage, indices)


def test_tampered_plan_hash_rejected():
    from vla_trajectory.collection import create_plan, validate_plan
    plan = create_plan('task', 'pilot')
    plan['task_name'] = 'different'
    with pytest.raises(ValueError, match='hash'):
        validate_plan(plan)


@pytest.mark.parametrize('pair', [['same', 'same'], ['one'], ['', 'two']])
def test_ambiguous_object_pair_rejected(pair):
    from vla_trajectory.collection import create_plan
    with pytest.raises(ValueError, match='object_pair'):
        create_plan('task', 'pilot', object_pair=pair)
