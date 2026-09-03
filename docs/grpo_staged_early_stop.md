# GRPO 分阶段自动验证与早停

该流程在同一个 LoRA 热同步 vLLM 服务上交替执行训练和固定验证，不需要每个阶段重启服务。

## 默认策略

- 当前训练上限：3000 条，每个数据集最多 1000 条；`global_step` 表示处理一个问题样本，每个问题固定生成 4 条 rollout。
- 早期验证集：`data/eval_90_grpo_fixed/manifest.jsonl`，三个数据集各 30 条；用于 200、400、600 步的快速验证。
- 后期验证集：`data/eval_300_grpo_fixed/manifest.jsonl`，三个数据集各 100 条；从 1000 步开始每 500 步验证一次，即 1000、1500、2000、2500、3000 步。
- 两套验证清单均来自同一固定分层候选集、使用同一随机种子，并且与 RL 训练 QID 不重叠。
- 并发：`eval_request_workers: 4`。
- 200 步只检查训练稳定性、生成错误和显著回退；早期层级从 400 步开始计算 patience，因此 400、600 连续两次无改善时可以在 600 步早停。
- 早期 90 条使用 SFT 的 90 条结果作为基线；后期完整 300 条在 step 1000 建立该层级的首个 RL 基线，之后每 500 步独立维护 best 与 patience，避免混用不同样本规模的 F1。
- 相对历史最佳 macro-F1 至少提高 0.005 才算一次有效改善；连续两次未改善则早停。
- macro-F1 相对 SFT 回退超过 0.01，或任一数据集 F1 回退超过 0.02，会立即早停。
- `best_checkpoint.json` 始终指向当前最佳的 SFT adapter 或 RL checkpoint。
- 3000 条训练完成且未触发早停时，控制器只标记 `complete`，不会自动增加训练样本；扩大数据规模需要显式修改配置后开启新一轮训练。

阈值和阶段表统一配置在 `config/grpo_staged_early_stop.yml`。

## 奖励与优势口径

- 支持覆盖按唯一支持文档计数，与 passage/doc 级检索和证据选择保持一致；同一文档中的多条 supporting-fact 句子不会重复增加目标数。
- `reward_total` 表示实际进入动作信用分配的 terminal reward；额外的过程 shaping 汇总单独记录为 `monitor_reward_total`。
- 主优势仍在同一 `(role, round)` 的候选之间计算；若该桶完全相同，回退也只比较该桶内候选的 terminal reward，不跨轮次比较。
- 每个动作先对自身 completion token 求平均，再在动作间等权聚合，避免长 Evidence 输出仅因 token 更多而获得更大梯度权重。

## 启动

先在一个终端启动训练用的 LoRA 热同步服务：

```bash
CONFIG_PATH=config/train_grpo.yml bash scripts/run_grpo_vllm_server.sh
```

服务就绪后，在另一个终端启动控制器：

```bash
bash scripts/run_grpo_staged_early_stop.sh
```

控制器会自动完成“SFT 基线验证 -> 分阶段训练 -> checkpoint 验证 -> 判定续训或早停”。状态写入
`outputs/grpo_Meta-Llama-3-8B/staged_early_stop/controller_state.json`。命令中断后再次执行即可恢复；若训练已经完成而验证尚未完成，会复用刚生成的 checkpoint 和已有验证进度。

只检查命令和配置而不启动任务：

```bash
bash scripts/run_grpo_staged_early_stop.sh --dry-run
```
