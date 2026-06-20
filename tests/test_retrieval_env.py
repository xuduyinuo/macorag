from macorag.retrieval_env import InMemoryRetrievalEnv


def test_in_memory_retrieval_env_tracks_stepwise_retrieval_state():
    questions = [
        {
            "qid": "q1",
            "dataset": "hotpotqa",
            "question": "Where was Alice born?",
        }
    ]
    chunks = [
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
    env = InMemoryRetrievalEnv(questions, chunks, retrieval_budget=2)

    initial_state = env.reset("q1")
    assert initial_state["retrieval_count"] == 0

    observation = env.step("Alice born", top_k=1)

    assert observation[0]["chunk_id"] == "c1"
    state = env.get_state()
    assert state["retrieval_count"] == 1
    assert state["retrieval_history"][0]["query"] == "Alice born"
