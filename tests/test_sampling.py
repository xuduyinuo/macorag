from macorag.sampling import sample_examples


def test_sample_examples_balances_question_type():
    examples = []
    for index in range(10):
        examples.append({"qid": f"b{index}", "question_type": "bridge", "hop_count": 2})
        examples.append(
            {"qid": f"c{index}", "question_type": "comparison", "hop_count": 2}
        )

    sampled = sample_examples(examples, target_count=6, seed=7)

    types = [item["question_type"] for item in sampled]
    assert len(sampled) == 6
    assert "bridge" in types
    assert "comparison" in types


def test_sample_examples_is_stable_for_same_seed():
    examples = [
        {"qid": f"b{index}", "question_type": "bridge", "hop_count": 2}
        for index in range(6)
    ] + [
        {"qid": f"c{index}", "question_type": "comparison", "hop_count": 2}
        for index in range(6)
    ]

    first = sample_examples(examples, target_count=8, seed=3)
    second = sample_examples(examples, target_count=8, seed=3)

    assert [item["qid"] for item in first] == [item["qid"] for item in second]


def test_sample_examples_returns_all_when_target_exceeds_size():
    examples = [
        {"qid": "a", "question_type": "bridge", "hop_count": 2},
        {"qid": "b", "question_type": "comparison", "hop_count": 2},
        {"qid": "c", "question_type": "bridge", "hop_count": 2},
    ]

    sampled = sample_examples(examples, target_count=10, seed=11)

    assert len(sampled) == len(examples)
    assert {item["qid"] for item in sampled} == {"a", "b", "c"}


def test_sample_examples_buckets_by_hop_count_without_question_type():
    examples = [
        {"qid": f"two-{index}", "hop_count": 2}
        for index in range(5)
    ] + [
        {"qid": f"three-{index}", "hop_count": 3}
        for index in range(5)
    ]

    sampled = sample_examples(examples, target_count=4, seed=5)

    hop_counts = {item["hop_count"] for item in sampled}
    assert len(sampled) == 4
    assert hop_counts == {2, 3}


def test_sample_examples_does_not_reorder_input_examples():
    examples = [
        {"qid": f"b{index}", "question_type": "bridge", "hop_count": 2}
        for index in range(5)
    ] + [
        {"qid": f"c{index}", "question_type": "comparison", "hop_count": 2}
        for index in range(5)
    ]
    original_order = [item["qid"] for item in examples]

    sample_examples(examples, target_count=6, seed=7)

    assert [item["qid"] for item in examples] == original_order
