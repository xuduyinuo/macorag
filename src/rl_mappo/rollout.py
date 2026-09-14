from __future__ import annotations

from typing import Any

from .protocol import build_prompt, central_state, parse_action
from .rewards import answer_decision_reward, evidence_reward, query_reward, terminal_reward
from .mappo_types import AgentRole, Episode, MAPPOTransition, RAGState, RLSample


class RolloutCollector:
    def __init__(self, *, actor: Any, critic: Any, retrieval: Any, config: Any) -> None:
        self.actor = actor
        self.critic = critic
        self.retrieval = retrieval
        self.config = config

    def _act(self, episode: Episode, role: AgentRole, state: RAGState, *, observation: dict[str, Any] | None, final_round: bool) -> tuple[dict[str, Any] | None, MAPPOTransition]:
        prompt = build_prompt(
            role, question=state.question, state=state,
            observation=observation, final_round=final_round,
            max_evidence_items=getattr(self.config, "prompt_max_evidence_items", 6),
            max_history_items=getattr(self.config, "prompt_max_history_items", 3),
            evidence_text_chars=getattr(self.config, "prompt_evidence_text_chars", 160),
            observation_text_chars=getattr(self.config, "prompt_observation_text_chars", 200),
        )
        response, prompt_ids, action_ids, old_logprobs = self.actor.generate(role, prompt)
        joint = central_state(state, role=role, observation=observation, max_rounds=self.config.max_rounds)
        with self.critic.torch.no_grad():
            value = float(self.critic([joint]).item())
        transition = MAPPOTransition(
            role=role, round_index=state.round_index, prompt=prompt,
            prompt_ids=prompt_ids, action_ids=action_ids,
            old_token_logprobs=old_logprobs, central_state=joint,
            next_central_state=joint, old_value=value, response=response,
        )
        try:
            parsed = parse_action(response, role, final_round=final_round)
            transition.reward = float(getattr(self.config, "format_reward_weight", 0.0))
        except ValueError as exc:
            transition.valid = False
            transition.parse_error = str(exc)
            penalty_name = (
                "final_answer_invalid_penalty"
                if role is AgentRole.ANSWER and final_round
                else "invalid_action_penalty"
            )
            transition.reward = float(getattr(
                self.config, penalty_name,
                getattr(self.config, "invalid_action_penalty", -1.0),
            ))
            episode.parse_errors.append(str(exc))
            parsed = None
        episode.transitions.append(transition)
        return parsed, transition

    def collect(self, sample: RLSample) -> Episode:
        episode = Episode(qid=sample.qid, dataset=sample.dataset)
        state = RAGState(question=sample.question)
        previous_retrieved: list[dict[str, Any]] = []
        for round_index in range(self.config.max_rounds):
            state.round_index = round_index
            final_round = round_index + 1 == self.config.max_rounds
            query_action, query_transition = self._act(episode, AgentRole.QUERY, state, observation=None, final_round=False)
            if query_action is None:
                break
            observation = self.retrieval.query(sample.dataset, query_action["query"])
            retrieved = list(observation["passages"])
            query_transition.reward += query_reward(retrieved, previous_retrieved, sample.supporting_facts, self.config.eta_query)
            previous_retrieved.extend(retrieved)

            updater_state = RAGState(
                question=state.question, sub_goal=query_action["sub_goal"],
                evidence=list(state.evidence), retrieval_history=list(state.retrieval_history),
                round_index=round_index,
            )
            evidence_action, evidence_transition = self._act(episode, AgentRole.EVIDENCE, updater_state, observation=observation, final_round=False)
            if evidence_action is None:
                break
            by_id = {int(item["passage_id"]): item for item in retrieved}
            selected = [dict(by_id[item]) for item in evidence_action["selected_passage_ids"] if item in by_id]
            ids_valid = len(selected) == len(evidence_action["selected_passage_ids"])
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
            episode.final_answer, state.evidence, sample.answer, sample.supporting_facts,
            self.config.omega_answer, self.config.omega_evidence,
            gate_evidence_on_valid_answer=bool(getattr(
                self.config, "gate_terminal_evidence_on_valid_answer", False,
            )),
        )
        episode.global_reward, episode.answer_f1, episode.evidence_coverage = score, f1, coverage
        last_by_role: dict[AgentRole, MAPPOTransition] = {}
        for transition in episode.transitions:
            last_by_role[transition.role] = transition
        for transition in last_by_role.values():
            # A malformed/refusing final action must never be rescued by a
            # positive team reward from already-retrieved evidence.
            if transition.valid:
                transition.reward += self.config.terminal_reward_weight * score
            transition.done = True
            transition.next_value = 0.0
        return episode
