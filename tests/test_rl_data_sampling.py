from __future__ import annotations

from collections import Counter

import pytest

from rl_training.data import RLSample, select_proportional_prefix


EXPECTED = {
    "2wiki": {
        "source": {
            "compositional": 831,
            "comparison": 486,
            "bridge_comparison": 440,
            "inference": 243,
        },
        500: {
            "compositional": 208,
            "comparison": 121,
            "bridge_comparison": 110,
            "inference": 61,
        },
        1000: {
            "compositional": 416,
            "comparison": 243,
            "bridge_comparison": 220,
            "inference": 121,
        },
    },
    "hotpotqa": {
        "source": {"hard/bridge": 1596, "hard/comparison": 404},
        500: {"hard/bridge": 399, "hard/comparison": 101},
        1000: {"hard/bridge": 798, "hard/comparison": 202},
    },
    "musique": {
        "source": {
            "2hop": 1036,
            "3hop1": 470,
            "3hop2": 159,
            "4hop1": 203,
            "4hop2": 53,
            "4hop3": 79,
        },
        500: {
            "2hop": 259,
            "3hop1": 117,
            "3hop2": 40,
            "4hop1": 51,
            "4hop2": 13,
            "4hop3": 20,
        },
        1000: {
            "2hop": 518,
            "3hop1": 235,
            "3hop2": 80,
            "4hop1": 101,
            "4hop2": 27,
            "4hop3": 39,
        },
    },
}


def make_samples(dataset: str, counts: dict[str, int]) -> list[RLSample]:
    return [
        RLSample(
            qid=f"{dataset}:{stratum}:{index}",
            dataset=dataset,
            question=f"question {dataset} {stratum} {index}",
            answer="answer",
            answer_aliases=[],
            supporting_facts=[],
            context_doc_ids=[],
            metadata={},
            sampling_stratum=stratum,
        )
        for stratum, count in counts.items()
        for index in range(count)
    ]


def stratum_counts(samples: list[RLSample]) -> dict[str, int]:
    return dict(Counter(sample.sampling_stratum for sample in samples))


def qids(samples: list[RLSample]) -> list[str]:
    return [sample.qid for sample in samples]


def test_proportional_prefix_matches_v2_quotas_and_is_nested() -> None:
    for dataset, contract in EXPECTED.items():
        source = make_samples(dataset, contract["source"])
        selected_500 = select_proportional_prefix(
            source,
            max_samples=500,
            seed=20260826,
        )
        selected_1000 = select_proportional_prefix(
            source,
            max_samples=1000,
            seed=20260826,
        )

        assert stratum_counts(selected_500) == contract[500]
        assert stratum_counts(selected_1000) == contract[1000]
        assert set(qids(selected_500)) <= set(qids(selected_1000))


def test_proportional_prefix_is_reproducible_and_seed_changes_membership() -> None:
    source = make_samples("hotpotqa", EXPECTED["hotpotqa"]["source"])

    first = select_proportional_prefix(source, max_samples=500, seed=17)
    repeated = select_proportional_prefix(source, max_samples=500, seed=17)
    changed = select_proportional_prefix(source, max_samples=500, seed=19)

    assert qids(first) == qids(repeated)
    assert set(qids(first)) != set(qids(changed))
    assert stratum_counts(first) == stratum_counts(changed)


@pytest.mark.parametrize("limit", [None, 2000, 3000])
def test_proportional_prefix_full_limits_keep_all_members(limit: int | None) -> None:
    source = make_samples("2wiki", EXPECTED["2wiki"]["source"])

    selected = select_proportional_prefix(source, max_samples=limit, seed=20260826)

    assert len(selected) == len(source)
    assert set(qids(selected)) == set(qids(source))


def test_proportional_prefix_rejects_invalid_inputs() -> None:
    source = make_samples("2wiki", {"compositional": 2})
    with pytest.raises(ValueError, match="max_samples"):
        select_proportional_prefix(source, max_samples=-1, seed=1)

    mixed = source + make_samples("hotpotqa", {"hard/bridge": 1})
    with pytest.raises(ValueError, match="exactly one dataset"):
        select_proportional_prefix(mixed, max_samples=1, seed=1)

    unsupported = make_samples("unknown", {"other": 1})
    with pytest.raises(ValueError, match="Unsupported"):
        select_proportional_prefix(unsupported, max_samples=1, seed=1)
