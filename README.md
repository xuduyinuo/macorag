# MACORAG

Multi-agent RAG data processing and teacher trajectory construction for HotpotQA, 2Wiki, and MuSiQue.

## First Milestone

- Build canonical `data/processed/<dataset>/examples.*.jsonl` and `corpus.jsonl`.
- Build LinearRAG-compatible `data/retrieval_env/<dataset>/questions.json` and `chunks.json`.
- Generate full-API teacher trajectories with fixed tags:
  `<plan>`, `<retrieval>`, `<update-evidence>`, `<answer>`.
- Filter to 3K SFT trajectories for Qwen2.5-7B warm-up:
  1K HotpotQA, 1K 2Wiki, 1K MuSiQue.

## Safety Rule

Gold answers and gold evidence are used only by verifiers and filters. They must not appear in teacher prompts.

## macorag 环境依赖说明

当前执行 `pip` 安装时，如果网络受限会在构建环境里报 `BackendUnavailable: Cannot import 'setuptools.build_meta'`。
这是因为 pip 的 PEP517 隔离构建会尝试从索引重新拉取 `setuptools`。

建议在 `conda activate macorag` 后用以下方式安装依赖：

```bash
PIP_NO_BUILD_ISOLATION=1 python -m pip install --no-deps -e .
PIP_NO_BUILD_ISOLATION=1 python -m pip install -r LinearRAG/requirements.txt
```

也可直接执行仓库脚本（同样会带上该配置）：

```bash
bash scripts/setup_macorag_env.sh
```
