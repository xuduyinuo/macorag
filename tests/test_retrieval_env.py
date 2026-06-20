import pytest

from macorag.retrieval_env import InMemoryRetrievalEnv


def _questions():
    return [
        {
            "qid": "q1",
            "dataset": "hotpotqa",
            "question": "Where was Alice born?",
        }
    ]


def _chunks():
    return [
        {
            "chunk_id": "c1",
            "title": "Alice",
            "text": "Alice was born in Paris.",
        },
        {
            "chunk_id": "c2",
            "title": "Bob",
            "text": "Bob wrote a book.",
        },
    ]


def test_in_memory_retrieval_env_tracks_stepwise_retrieval_state():
    questions = _questions()
    chunks = _chunks()
    env = InMemoryRetrievalEnv(questions, chunks, retrieval_budget=2)

    initial_state = env.reset("q1")
    assert initial_state["retrieval_count"] == 0

    observation = env.step("Alice born", top_k=1)

    assert observation[0]["chunk_id"] == "c1"
    state = env.get_state()
    assert state["retrieval_count"] == 1
    assert state["retrieval_history"][0]["query"] == "Alice born"


def test_init_copies_questions_and_chunks_before_external_mutation():
    questions = _questions()
    chunks = _chunks()
    env = InMemoryRetrievalEnv(questions, chunks, retrieval_budget=2)

    questions[0]["dataset"] = "mutated"
    questions[0]["question"] = "Mutated question?"
    chunks[0]["chunk_id"] = "mutated"
    chunks[0]["title"] = "Carol"
    chunks[0]["text"] = "Carol was born in Rome."

    state = env.reset("q1")
    observation = env.step("Alice born", top_k=1)

    assert state["dataset"] == "hotpotqa"
    assert state["question"] == "Where was Alice born?"
    assert observation[0]["chunk_id"] == "c1"
    assert observation[0]["title"] == "Alice"


def test_get_state_returns_deep_copy():
    env = InMemoryRetrievalEnv(_questions(), _chunks(), retrieval_budget=2)
    env.reset("q1")
    env.step("Alice born", top_k=1)

    state = env.get_state()
    state["evidence"][0]["chunk_id"] = "mutated"
    state["retrieval_history"][0]["query"] = "mutated"
    state["retrieval_history"][0]["results"][0]["chunk_id"] = "mutated"

    fresh_state = env.get_state()
    assert fresh_state["evidence"][0]["chunk_id"] == "c1"
    assert fresh_state["retrieval_history"][0]["query"] == "Alice born"
    assert fresh_state["retrieval_history"][0]["results"][0]["chunk_id"] == "c1"


def test_step_observation_is_isolated_from_history():
    env = InMemoryRetrievalEnv(_questions(), _chunks(), retrieval_budget=2)
    env.reset("q1")

    observation = env.step("Alice born", top_k=1)
    observation[0]["chunk_id"] = "mutated"

    state = env.get_state()
    assert state["retrieval_history"][0]["results"][0]["chunk_id"] == "c1"


def test_empty_query_and_empty_doc_score_zero():
    env = InMemoryRetrievalEnv(
        _questions(),
        [
            {
                "chunk_id": "empty",
                "title": "",
                "text": "",
            }
        ],
        retrieval_budget=2,
    )
    env.reset("q1")

    observation = env.step("", top_k=1)

    assert observation[0]["chunk_id"] == "empty"
    assert observation[0]["score"] == 0.0


def test_budget_limit_and_reset_clears_history_and_count():
    env = InMemoryRetrievalEnv(_questions(), _chunks(), retrieval_budget=1)
    env.reset("q1")
    env.step("Alice born", top_k=1)

    with pytest.raises(RuntimeError):
        env.step("Alice born", top_k=1)

    state = env.reset("q1")
    assert state["retrieval_count"] == 0
    assert state["retrieval_history"] == []
