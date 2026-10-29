from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterator, List


def _format_money(value: int) -> str:
    return f"${value}"


def _word_choice(rng: random.Random, values: List[str]) -> str:
    return values[rng.randrange(len(values))]


def _render_sample(seed: int, index: int) -> Dict[str, object]:
    rng = random.Random(int(seed) + int(index) * 7919)
    template = rng.randrange(5)

    if template == 0:
        start = rng.randint(18, 95)
        add = rng.randint(6, 45)
        sub = rng.randint(3, min(25, start + add - 1))
        item = _word_choice(rng, ["stickers", "marbles", "pens", "bookmarks"])
        owner = _word_choice(rng, ["Lena", "Mia", "Noah", "Ethan"])
        question = f"{owner} has {start} {item}. {owner} buys {add} more and gives away {sub}. How many {item} does {owner} have now?"
        total = start + add - sub
        solution = (
            f"Start with {start} {item}. After buying {add} more, the total is {start} + {add} = {start + add}. "
            f"Then subtract the {sub} given away: {start + add} - {sub} = {total}."
        )
        answer = str(total)
    elif template == 1:
        groups = rng.randint(3, 12)
        each = rng.randint(4, 16)
        item = _word_choice(rng, ["boxes", "trays", "bags", "shelves"])
        obj = _word_choice(rng, ["cookies", "bolts", "erasers", "cards"])
        question = f"There are {groups} {item}, and each {item[:-1] if item.endswith('s') else item} holds {each} {obj}. How many {obj} are there in total?"
        total = groups * each
        solution = f"Each of the {groups} {item} holds {each} {obj}, so multiply {groups} x {each} = {total}."
        answer = str(total)
    elif template == 2:
        divisor = rng.randint(2, 12)
        quotient = rng.randint(4, 18)
        total = divisor * quotient
        item = _word_choice(rng, ["apples", "tickets", "shells", "toy cars"])
        question = f"{total} {item} are shared equally among {divisor} friends. How many does each friend get?"
        solution = f"Equal sharing means division. Compute {total} / {divisor} = {quotient}, so each friend gets {quotient} {item}."
        answer = str(quotient)
    elif template == 3:
        price = rng.randint(2, 15)
        count = rng.randint(2, 9)
        extra = rng.randint(1, 12)
        person = _word_choice(rng, ["Ava", "Lucas", "Zoe", "Henry"])
        item = _word_choice(rng, ["notebook", "juice", "sandwich", "pencil case"])
        total = price * count + extra
        question = (
            f"{person} buys {count} {item}s for {_format_money(price)} each and also pays {_format_money(extra)} for shipping. "
            f"How much does {person} pay in total?"
        )
        solution = (
            f"The {count} {item}s cost {count} x {price} = {_format_money(price * count)}. "
            f"Add the shipping cost: {price * count} + {extra} = {_format_money(total)}."
        )
        answer = _format_money(total)
    else:
        start = rng.randint(20, 70)
        bought_each = rng.randint(2, 8)
        bought_days = rng.randint(2, 4)
        used = rng.randint(3, 15)
        item = _word_choice(rng, ["markers", "cupcakes", "beads", "lemons"])
        person = _word_choice(rng, ["Olivia", "Daniel", "Grace", "Jack"])
        added = bought_each * bought_days
        total = start + added - used
        question = (
            f"{person} starts with {start} {item}. For {bought_days} days, {person} gets {bought_each} more {item} each day. "
            f"After that, {person} uses {used} {item}. How many {item} are left?"
        )
        solution = (
            f"In {bought_days} days, {person} gets {bought_each} x {bought_days} = {added} {item}. "
            f"That makes {start} + {added} = {start + added}. Then subtract the {used} used: {start + added} - {used} = {total}."
        )
        answer = str(total)

    return {
        "question": question,
        "solution": solution,
        "answer": answer,
        "program_answer": answer,
        "generation_seed": int(seed),
        "generation_index": int(index),
        "template_type": f"arithmetic_template_{template}",
        "difficulty": {"template_id": int(template)},
    }


@dataclass
class SyntheticArithmeticDataset:
    num_samples: int
    seed: int

    def __len__(self) -> int:
        return max(0, int(self.num_samples))

    def __iter__(self) -> Iterator[Dict[str, object]]:
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, index: int) -> Dict[str, object]:
        size = len(self)
        if size <= 0:
            raise IndexError("SyntheticArithmeticDataset is empty")
        if index < 0 or index >= size:
            raise IndexError(index)
        return _render_sample(self.seed, index)


def build_synthetic_arithmetic_dataset(num_samples: int, seed: int) -> SyntheticArithmeticDataset:
    return SyntheticArithmeticDataset(num_samples=int(num_samples), seed=int(seed))
