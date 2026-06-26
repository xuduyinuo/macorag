from __future__ import annotations

from typing import Any

from .executor import RAGLoopExecutor
from .reward import compute_reward_terms


def rollout_with_rewards(
    *,
    executor: RAGLoopExecutor,
    question: str,
    dataset: str,
    gold_answer: str | None,
    answer_aliases: list[str],
) -> dict[str, Any]:
    result = executor.run(question=question, dataset=dataset)
    reward_terms = compute_reward_terms(
        trajectory=result.trajectory,
        final_answer=result.final_answer,
        gold_answer=gold_answer,
        answer_aliases=answer_aliases,
        parse_errors=result.parse_errors,
    )
    return {
        "result": result,
        "trajectory": result.trajectory,
        "reward_terms": reward_terms,
    }
