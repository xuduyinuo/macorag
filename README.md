# MACORAG

Multi-agent RAG data processing and teacher trajectory construction for HotpotQA, 2Wiki, and MuSiQue.

## First Milestone

- Build canonical `data/processed/<dataset>/examples.*.jsonl` and `corpus.jsonl`.
- Build LinearRAG-compatible `linearrag_dataset/<dataset>/questions.json` and `chunks.json`.
- Generate full-API teacher trajectories with fixed tags:
  `<plan>`, `<retrieval>`, `<update-evidence>`, `<answer>`.
- Filter to 3K SFT trajectories for Qwen2.5-7B warm-up:
  1K HotpotQA, 1K 2Wiki, 1K MuSiQue.

## Safety Rule

Gold answers and gold evidence are used only by verifiers and filters. They must not appear in teacher prompts.
