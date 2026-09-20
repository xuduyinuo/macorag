from __future__ import annotations

import inspect
import re
import threading
import time
from typing import Any

from .models import (
    evidence_action_layout, evidence_completion_budget, evidence_guided_regex,
)
from .protocol import build_prompt, central_state, parse_action
from .rewards import answer_decision_reward, evidence_reward, query_reward, terminal_reward
from .mappo_types import AgentRole, Episode, MAPPOTransition, RAGState, RLSample


def _repair_action_format(text: str, role: AgentRole) -> str:
    """Repair only known serialization artifacts, never action semantics."""
    tag = {
        AgentRole.QUERY: "query-retriever",
        AgentRole.EVIDENCE: "update-evidence",
        AgentRole.ANSWER: "answer",
    }[role]
    opening, closing = f"<{tag}>", f"</{tag}>"
    start, end = text.find(opening), text.rfind(closing)
    if start < 0 or end < start:
        return text
    body_start = start + len(opening)
    body = text[body_start:end]
    repaired_body = body.replace(closing, "").replace("\\'", "'")
    if repaired_body == body:
        return text
    return opening + repaired_body + closing


def _is_recoverable_format_error(error: ValueError) -> bool:
    message = str(error)
    return message.startswith((
        "Missing required tag:",
        "Expected exactly one required tag:",
        "Invalid JSON in tag ",
        "Generated Evidence action violated guided regex",
    ))


class RolloutCollector:
    def __init__(self, *, actor: Any, critic: Any, retrieval: Any, config: Any) -> None:
        self.actor = actor
        self.critic = critic
        self.retrieval = retrieval
        self.config = config
        # The learner model and critic share CUDA state and are not invoked
        # concurrently. vLLM HTTP generation and retrieval remain parallel.
        self._actor_rescore_lock = threading.Lock()
        self._critic_lock = threading.Lock()
        self.evidence_completion_budget = None
        if (
            hasattr(actor, "tokenizer")
            and bool(getattr(config, "force_evidence_guided_decoding", False))
        ):
            self.evidence_completion_budget = evidence_completion_budget(
                actor.tokenizer,
                passage_ids=list(range(int(getattr(config, "retrieval_top_k", 5)))),
                max_completion_length=int(getattr(config, "evidence_max_completion_length", 192)),
            )

    @staticmethod
    def _add_timing(episode: Episode, name: str, seconds: float) -> None:
        episode.timing[name] = episode.timing.get(name, 0.0) + float(seconds)

    def _act(self, episode: Episode, role: AgentRole, state: RAGState, *, observation: dict[str, Any] | None, final_round: bool) -> tuple[dict[str, Any] | None, MAPPOTransition]:
        prompt = build_prompt(
            role, question=state.question, state=state,
            observation=observation, final_round=final_round,
            max_rounds=self.config.max_rounds,
            max_evidence_items=getattr(self.config, "prompt_max_evidence_items", 6),
            max_history_items=getattr(self.config, "prompt_max_history_items", 3),
            evidence_text_chars=getattr(self.config, "prompt_evidence_text_chars", 160),
            observation_text_chars=getattr(self.config, "prompt_observation_text_chars", 200),
            evidence_min_selected_passages=getattr(
                self.config, "evidence_min_selected_passages", 1,
            ),
        )
        guidance = None
        if (
            role is AgentRole.EVIDENCE
            and observation is not None
            and bool(getattr(self.config, "force_evidence_guided_decoding", False))
        ):
            guidance = evidence_guided_regex(
                [
                    int(item["passage_id"])
                    for item in observation.get("passages", [])
                ],
                min_selected=getattr(
                    self.config, "evidence_min_selected_passages", 1,
                ),
            )
        def _generate_action():
            if guidance is None:
                parameters = inspect.signature(self.actor.generate).parameters
                accepts_max_tokens = (
                    "max_tokens" in parameters
                    or any(
                        item.kind is inspect.Parameter.VAR_KEYWORD
                        for item in parameters.values()
                    )
                )
                if role is AgentRole.ANSWER and accepts_max_tokens:
                    return self.actor.generate(
                        role, prompt,
                        max_tokens=getattr(
                            self.config, "answer_max_completion_length", 192,
                        ),
                    )
                return self.actor.generate(role, prompt)
            return self.actor.generate(
                role, prompt, guided_regex=guidance,
                max_tokens=getattr(self.config, "evidence_max_completion_length", 192),
            )

        def generate_action():
            started = time.perf_counter()
            try:
                return _generate_action()
            finally:
                self._add_timing(
                    episode, f"{role.value}_generation_seconds",
                    time.perf_counter() - started,
                )
                episode.timing["generation_calls"] = (
                    episode.timing.get("generation_calls", 0.0) + 1.0
                )

        def parse_generated(candidate: str) -> dict[str, Any]:
            parsed_action = parse_action(candidate, role, final_round=final_round)
            if guidance is not None and re.fullmatch(guidance, candidate) is None:
                raise ValueError("Generated Evidence action violated guided regex")
            return parsed_action

        response, prompt_ids, action_ids, old_logprobs = generate_action()
        recovery = None
        failure: ValueError | None = None
        try:
            parsed = parse_generated(response)
        except ValueError as first_error:
            failure = first_error
            if _is_recoverable_format_error(first_error):
                repaired = _repair_action_format(response, role)
                if repaired != response and hasattr(self.actor, "rescore_response"):
                    try:
                        parsed = parse_generated(repaired)
                        with self._actor_rescore_lock:
                            action_ids, old_logprobs = self.actor.rescore_response(
                                prompt_ids, repaired,
                            )
                        response, recovery = repaired, "repaired"
                    except ValueError:
                        parsed = None
                else:
                    parsed = None
                if parsed is None:
                    try:
                        response, prompt_ids, action_ids, old_logprobs = generate_action()
                        parsed = parse_generated(response)
                        recovery = "retried"
                    except ValueError as retry_error:
                        parsed = None
                        failure = retry_error
            else:
                parsed = None
        tokenization_recovery = None
        if guidance is not None and parsed is not None and hasattr(self.actor, "tokenizer"):
            # Guided decoding may emit a valid text through a non-canonical BPE
            # path.  The learner and its counterfactual pointer trie must use
            # one canonical tokenizer path, so normalize only when the vLLM
            # token IDs do not round-trip to the parsed response.
            canonical_ids = list(self.actor.tokenizer.encode(
                response, add_special_tokens=False,
            ))
            prefix_matches = action_ids[:len(canonical_ids)] == canonical_ids
            trailing_ids = action_ids[len(canonical_ids):] if prefix_matches else []
            special_ids = set(
                getattr(self.actor.tokenizer, "all_special_ids", []) or []
            )
            canonical_with_optional_stop = prefix_matches and (
                not trailing_ids
                or bool(special_ids) and all(item in special_ids for item in trailing_ids)
            )
            if not canonical_with_optional_stop:
                if getattr(
                    self.config, "ppo_old_logprob_source", "local_actor",
                ) == "local_actor":
                    # The trainer deterministically replaces these placeholders
                    # before PPO. Avoid an unnecessary 7B learner forward pass
                    # for every initial-validation Evidence action.
                    action_ids = canonical_ids
                    old_logprobs = old_logprobs.new_zeros(len(canonical_ids))
                elif hasattr(self.actor, "rescore_response"):
                    with self._actor_rescore_lock:
                        action_ids, old_logprobs = self.actor.rescore_response(
                            prompt_ids, response,
                        )
                else:
                    raise RuntimeError(
                        "Evidence guided decoding produced a non-canonical token path, "
                        "but the actor cannot rescore its canonical response"
                    )
                tokenization_recovery = "canonicalized"
        joint = central_state(state, role=role, observation=observation, max_rounds=self.config.max_rounds)
        critic_started = time.perf_counter()
        with self._critic_lock, self.critic.torch.no_grad():
            value = float(self.critic([joint]).item())
        self._add_timing(episode, "critic_seconds", time.perf_counter() - critic_started)
        transition = MAPPOTransition(
            role=role, round_index=state.round_index, prompt=prompt,
            prompt_ids=prompt_ids, action_ids=action_ids,
            old_token_logprobs=old_logprobs, central_state=joint,
            next_central_state=joint, old_value=value, response=response,
            format_recovery=recovery,
            tokenization_recovery=tokenization_recovery,
        )
        if guidance is not None and parsed is not None and hasattr(self.actor, "tokenizer"):
            constraints, optimize, segments = evidence_action_layout(
                self.actor.tokenizer,
                response=response,
                action_ids=action_ids,
                passage_ids=[int(item["passage_id"]) for item in observation["passages"]],
                min_selected=getattr(self.config, "evidence_min_selected_passages", 1),
            )
            transition.token_constraints = constraints
            transition.optimization_token_mask = optimize
            transition.token_segments = segments
        if parsed is not None:
            transition.parsed_action = parsed
            transition.reward = float(getattr(self.config, "format_reward_weight", 0.0))
        else:
            transition.valid = False
            if failure is None:
                raise RuntimeError("Invalid action is missing its parse failure")
            transition.parse_error = str(failure)
            penalty_name = (
                "final_answer_invalid_penalty"
                if role is AgentRole.ANSWER and final_round
                else "invalid_action_penalty"
            )
            transition.reward = float(getattr(
                self.config, penalty_name,
                getattr(self.config, "invalid_action_penalty", -1.0),
            ))
            episode.parse_errors.append(str(failure))
        episode.transitions.append(transition)
        return parsed, transition

    def collect(self, sample: RLSample) -> Episode:
        episode_started = time.perf_counter()
        episode = Episode(qid=sample.qid, dataset=sample.dataset)
        state = RAGState(question=sample.question)
        previous_retrieved: list[dict[str, Any]] = []
        for round_index in range(self.config.max_rounds):
            state.round_index = round_index
            final_round = round_index + 1 == self.config.max_rounds
            query_action, query_transition = self._act(episode, AgentRole.QUERY, state, observation=None, final_round=False)
            if query_action is None:
                break
            retrieval_started = time.perf_counter()
            observation = self.retrieval.query(sample.dataset, query_action["query"])
            self._add_timing(
                episode, "retrieval_seconds", time.perf_counter() - retrieval_started,
            )
            episode.timing["retrieval_calls"] = episode.timing.get("retrieval_calls", 0.0) + 1.0
            retrieved = list(observation["passages"])
            query_transition.reward += query_reward(retrieved, previous_retrieved, sample.supporting_facts, self.config.eta_query)
            previous_retrieved.extend(retrieved)

            updater_state = RAGState(
                question=state.question, sub_goal=query_action["sub_goal"],
                evidence=list(state.evidence), retrieval_history=list(state.retrieval_history),
                round_index=round_index,
            )
            # Match evaluation exactly: expose the top-k FAISS result on every
            # round, even when a document was selected in an earlier round.
            candidate_passages = retrieved
            candidate_observation = {**observation, "passages": candidate_passages}
            evidence_action, evidence_transition = self._act(
                episode, AgentRole.EVIDENCE, updater_state,
                observation=candidate_observation, final_round=False,
            )
            if evidence_action is None:
                break
            by_id = {int(item["passage_id"]): item for item in candidate_passages}
            selected = [dict(by_id[item]) for item in evidence_action["selected_passage_ids"] if item in by_id]
            requested_ids = evidence_action["selected_passage_ids"]
            ids_valid = (
                len(requested_ids) == len(set(requested_ids))
                and len(selected) == len(requested_ids)
            )
            if ids_valid:
                evidence_transition.reward += evidence_reward(retrieved, selected, sample.supporting_facts, self.config.eta_evidence)
            else:
                evidence_transition.reward = float(getattr(self.config, "invalid_action_penalty", -1.0))
            evidence_transition.valid = ids_valid
            next_state = RAGState(
                question=state.question, sub_goal=query_action["sub_goal"],
                evidence=[*state.evidence, *selected],
                retrieval_history=[*state.retrieval_history, {
                    "query": query_action["query"], "sub_goal": query_action["sub_goal"],
                    "passage_ids": [item["passage_id"] for item in selected],
                }], round_index=round_index,
            )
            # Retrieval/evidence state exists independently of whether the
            # answer action parses. Preserve it for diagnostics and gated
            # terminal reward computation on final-answer failures.
            state = next_state
            answer_action, answer_transition = self._act(episode, AgentRole.ANSWER, next_state, observation=None, final_round=final_round)
            if answer_action is None:
                break
            answer_transition.reward += self.config.answer_local_reward_weight * answer_decision_reward(
                answer_action["can_answer"], next_state.evidence, sample.supporting_facts,
                final_round=final_round,
                non_final_wait_reward=float(getattr(self.config, "non_final_wait_reward", 0.2)),
                final_answer_bonus=float(getattr(self.config, "final_answer_bonus", 1.0)),
            )
            episode.trajectory.append({
                "round": round_index, "query_retriever": query_action,
                "observation": observation, "update_evidence": evidence_action,
                "answer": answer_action,
            })
            if answer_action["can_answer"]:
                episode.final_answer = answer_action["answer"]
                break

        score, f1, coverage = terminal_reward(
            episode.final_answer, state.evidence,
            (sample.answer, *sample.answer_aliases), sample.supporting_facts,
            self.config.omega_answer, self.config.omega_evidence,
            gate_evidence_on_valid_answer=bool(getattr(
                self.config, "gate_terminal_evidence_on_valid_answer", False,
            )),
        )
        episode.global_reward, episode.answer_f1, episode.evidence_coverage = score, f1, coverage
        for transition in episode.transitions:
            transition.local_reward = transition.reward
        last_by_role: dict[AgentRole, MAPPOTransition] = {}
        for transition in episode.transitions:
            last_by_role[transition.role] = transition
        for transition in last_by_role.values():
            # A malformed/refusing final action must never be rescued by a
            # positive team reward from already-retrieved evidence.
            if transition.valid:
                transition.team_reward = self.config.terminal_reward_weight * score
                transition.reward += transition.team_reward
            transition.done = True
            transition.next_value = 0.0
        episode.timing["episode_seconds"] = time.perf_counter() - episode_started
        return episode
