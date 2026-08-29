# Prompt Budget JSON Tag Selection Fix

## Problem

GRPO prompt builders refer to protocol tags in prose before emitting the real
JSON blocks, for example `Use selected evidence in <state>.` followed later by
`<state>{...}</state>`. When a prompt exceeds `max_prompt_length`,
`compact_tagged_json_prompt()` uses a non-overlapping regular expression that
starts at the prose reference and ends at the real closing tag. The captured
text is not JSON, so no removable state or observation content is found and the
function raises `PromptBudgetError`.

The first sampled question has a small fixed prompt. The failure appears only
after evidence accumulates across rounds, which is why the progress bar remains
at zero while the first sample is still being rolled out.

## Design

Change the internal tag matcher in `src/rag/prompt_budget.py` to enumerate
overlapping `<tag>...</tag>` candidates. For each candidate, attempt to parse
the enclosed content as a JSON object and select the first parseable candidate.
This skips prose-only opening tags while preserving the existing state and
observation compaction order.

The matcher will use a zero-width lookahead with capture groups so that the
inner real opening tag remains discoverable after an outer invalid candidate.
The replacement helper will replace the span of the full captured tag rather
than the zero-width lookahead match.

Do not change prompt templates, `max_prompt_length`, retrieval content,
evidence selection, or the order in which retrieval history, evidence, and
observation passages are removed.

## Testing

Add a focused regression test in `tests/test_prompt_budget.py` containing prose
references to both `<state>` and `<observation>` before real JSON blocks. Set
the character-count budget to require removal of retrieval history, evidence,
and an observation passage. The current implementation must raise the observed
`PromptBudgetError`; the fixed implementation must return the compacted prompt
with protocol edges intact and all three removal counters equal to one.

Run the focused prompt-budget tests, relevant RAG/RL tests, Python compilation,
and `git diff --check`. Then run GRPO with `--max-steps 1` against the existing
vLLM service to verify that the first sample completes rather than failing in
answer prompt encoding.

## Scope and Safety

Production changes are limited to the internal tag-matching and replacement
spans in `src/rag/prompt_budget.py`. Existing unrelated working-tree changes
remain untouched. The earlier failed output directory contains no checkpoint
and will not be used for resume.
