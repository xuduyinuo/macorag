# Teacher Trajectory Metric-Aligned Validation Design

## Goal

Reduce false-positive trajectory filtering without weakening the answer contract used by evaluation. Teacher answers retained for SFT must optimize the existing normalized exact match, bidirectional contain, and token F1 metrics; semantic equivalence is allowed only when deciding whether selected evidence supports an already metric-correct answer.

## Fixed Metric Contract

The existing evaluator remains unchanged:

- exact match compares normalized prediction and normalized primary gold;
- contain is bidirectional: normalized gold contains normalized prediction or normalized prediction contains normalized gold;
- F1 is normalized token overlap.

Teacher admission uses the primary gold answer and the same normalization as evaluation. A trajectory is retained only when its final answer has normalized exact match equal to 1 against the primary gold. This implies contain 1 and F1 1 for the admitted answer and deliberately excludes a merely semantic equivalent string that evaluation would score as incorrect. Dataset aliases do not replace the primary-gold admission rule unless the evaluator is separately changed to score aliases in the future.

## Evidence Support Contract

Evidence support is independent from answer-label correctness. After the final answer passes normalized exact match, selected evidence may support it through any deterministic equivalent form drawn from:

- the primary gold answer;
- dataset-provided answer aliases;
- normalization-equivalent variants produced by the evaluator's case, punctuation, article, and whitespace normalization.

No embedding threshold or LLM semantic judge is used. This permits evidence wording that uses a known alias while preventing unconstrained semantic similarity from admitting hallucinated support.

## Query Leakage Validation

The capitalization heuristic is removed as a hard filter. A query is rejected for answer leakage only when all of the following are true:

1. it contains a normalized primary-gold or dataset-alias answer form;
2. that form is not already present in the question;
3. that form is not present in evidence selected before the current round.

Generic title-cased phrases such as `Birth Place` are therefore allowed. The teacher prompt continues to instruct the query role not to invent unsupported intermediate facts.

## Forbidden Training-Label Validation

Forbidden terms are checked only in assistant-authored structured fields that can become SFT targets:

- query-retriever sub-goal and query;
- evidence-updater rationale;
- answer text and rationale.

Observation passages, selected evidence copied from passages, state, question, metadata, and raw diagnostic responses are excluded. Thus a source title such as `Gold Coast` cannot invalidate an otherwise correct trajectory.

## Filter Diagnostics

Every filtered candidate is appended to `teacher_filtered.jsonl`. Each record contains:

- dataset, qid, question, stage, and exact validation errors;
- normalized metric components `answer_exact_match`, `answer_contain`, and `answer_f1` against the primary gold;
- evidence-support result;
- the complete candidate trajectory, including observations and raw role responses already present in each turn;
- prompt contract version/fingerprint and retrieval index fingerprint.

The existing `teacher_errors.jsonl` remains reserved for runtime/API/parser failures. `summary.json` reports filtered records by candidate as well as error-reason occurrences so multiple validation errors on one candidate are not mistaken for multiple samples.

## Prompt and Evaluation Boundaries

This change does not modify `calculate_contain`, exact match, F1, evaluation outputs, RL rewards, retrieval behavior, or the shared final-round fallback protocol. It changes only teacher-trajectory validation and diagnostics. The final-round role may still make a fallback guess, but a guessed trajectory enters SFT only if its final answer has normalized exact match 1 and selected evidence deterministically supports the accepted answer forms.

## Verification

Tests must demonstrate:

- `Gold Coast` in an observation no longer triggers a forbidden-term failure;
- forbidden terms in authored rationale still fail;
- a generic title-cased query is accepted;
- an unseen gold/alias leaked into a query is rejected;
- a final answer semantically equivalent but metric-unequal to gold is rejected;
- an EM-correct final answer can be supported by a dataset alias in evidence;
- contain remains bidirectional and unchanged;
- filtered records preserve raw responses, exact errors, metric components, and trajectory;
- the previous three valid smoke trajectories still validate successfully.

