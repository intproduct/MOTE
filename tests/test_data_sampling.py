from __future__ import annotations

import pytest

from MOTE.data.sampling import DeterministicSequenceSampler


def _draw(sampler, count):
    return [sampler.next_record()["id"] for _ in range(count)]


def test_source_epoch_is_deterministic_without_replacement():
    records = [{"id": index} for index in range(20)]
    first = DeterministicSequenceSampler(records, seed=7, namespace="math")
    second = DeterministicSequenceSampler(records, seed=7, namespace="math")
    first_epoch = _draw(first, 20)
    assert first_epoch == _draw(second, 20)
    assert sorted(first_epoch) == list(range(20))
    assert first.statistics()["repeat_count"] == 0
    next_epoch = _draw(first, 20)
    assert sorted(next_epoch) == list(range(20))
    assert next_epoch != first_epoch
    assert first.statistics()["repeat_count"] == 20


def test_shards_are_disjoint_and_cover_the_global_permutation():
    records = [{"id": index} for index in range(17)]
    shards = [
        DeterministicSequenceSampler(records, seed=11, namespace="source", shard_id=index, shard_count=3)
        for index in range(3)
    ]
    drawn = [set(_draw(sampler, len(sampler.order))) for sampler in shards]
    assert not (drawn[0] & drawn[1] or drawn[0] & drawn[2] or drawn[1] & drawn[2])
    assert set.union(*drawn) == set(range(17))


def test_max_samples_selects_a_deterministic_subset_not_a_fixed_prefix():
    records = [{"id": index} for index in range(100)]
    sampler = DeterministicSequenceSampler(records, seed=3, namespace="large", max_samples=10)
    selected = _draw(sampler, 10)
    assert len(set(selected)) == 10
    assert set(selected) != set(range(10))


def test_sampler_resume_matches_uninterrupted_and_rejects_changed_data():
    records = [{"id": index} for index in range(12)]
    uninterrupted = DeterministicSequenceSampler(records, seed=5, namespace="resume")
    _draw(uninterrupted, 19)
    state = uninterrupted.state_dict()
    expected = _draw(uninterrupted, 25)
    resumed = DeterministicSequenceSampler.from_state(records, state)
    assert _draw(resumed, 25) == expected
    with pytest.raises(RuntimeError, match="record count changed"):
        DeterministicSequenceSampler.from_state(records + [{"id": 99}], state)


def test_source_epoch_limit_can_fail_closed_before_repeated_exposure():
    records = [{"id": index} for index in range(5)]
    sampler = DeterministicSequenceSampler(records, seed=1, namespace="bounded", max_epochs=1)
    assert len(set(_draw(sampler, 5))) == 5
    with pytest.raises(RuntimeError, match="max_epochs=1"):
        sampler.next_record()
