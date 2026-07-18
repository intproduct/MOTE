from __future__ import annotations

import pytest

from MOTE.rl.sampler import StatefulRLRecordSampler


def _records(count: int = 7):
    return [
        {"idx": index, "source": "test", "question": f"q{index}", "answer": f"a{index}"}
        for index in range(count)
    ]


def _ids(batch):
    return [row["idx"] for row in batch]


def test_sampler_shuffle_is_reproducible_and_not_dataset_order():
    records = _records()
    first = StatefulRLRecordSampler(records, shuffle=True, seed=1234)
    second = StatefulRLRecordSampler(records, shuffle=True, seed=1234)

    first_ids = _ids(first.next_batch(len(records)))
    second_ids = _ids(second.next_batch(len(records)))

    assert first_ids == second_ids
    assert first_ids != list(range(len(records)))
    assert sorted(first_ids) == list(range(len(records)))


def test_sampler_exact_resume_matches_uninterrupted_across_epochs():
    records = _records(5)
    uninterrupted = StatefulRLRecordSampler(records, shuffle=True, seed=17)
    prefix = _ids(uninterrupted.next_batch(7))
    state = uninterrupted.state_dict()
    expected_suffix = _ids(uninterrupted.next_batch(11))

    resumed = StatefulRLRecordSampler.from_state(records, state)
    actual_suffix = _ids(resumed.next_batch(11))

    assert len(prefix) == 7
    assert actual_suffix == expected_suffix
    assert resumed.state_dict()["samples_seen"] == 18
    assert resumed.state_dict()["epoch"] == uninterrupted.state_dict()["epoch"]


def test_sampler_resume_rejects_changed_dataset_content():
    records = _records()
    sampler = StatefulRLRecordSampler(records, shuffle=True, seed=3)
    sampler.next_batch(2)
    changed = _records()
    changed[0] = {**changed[0], "question": "changed"}

    with pytest.raises(RuntimeError, match="content/order changed"):
        StatefulRLRecordSampler.from_state(changed, sampler.state_dict())


def test_sampler_can_preserve_sequential_traversal():
    records = _records(3)
    sampler = StatefulRLRecordSampler(records, shuffle=False, seed=999)

    assert _ids(sampler.next_batch(5)) == [0, 1, 2, 0, 1]
    assert sampler.epoch == 1
    assert sampler.position == 2

