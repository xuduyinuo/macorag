# RL 优化验证顺序

运行入口（需要两张空闲 GPU；已接受的新协议实验）：

```bash
bash scripts/run_grpo_optimization_validation.sh --stage all
```

运行目录默认是 `outputs/grpo_Meta-Llama-3-8B/optimization_validation_300_dual_gpu_accepted`。
同目录有排他锁；重复启动不会接管已有服务。再次运行会校验冻结的代码、配置、
数据和 adapter，复用完整评估、续跑部分预测，并从完整训练 checkpoint 续训。
不要在正在运行时更改训练或评估源码；改变实验条件应使用新的 `--run-dir`。

## 当前自动执行范围

1. 冒烟：固定 300 条中每类前 4 条，单 GPU/4 workers 对比双 GPU/8 workers。
   两边的最终答案、完整轨迹、解析错误必须一致，且无请求错误，否则停止。
   用户已明确接受此前双副本输出差异时，可用下述显式接受流程替代严格等价检查。
2. 固定 300 条：同一口径依次评估 SFT、aligned_control、no_answer_local。
   仅当 `off > control` 且 `off >= SFT`，并且三组均无请求错误时继续。
   这是筛选下一轮试验的门槛，不等于统计显著性结论。
3. 单变量学习率试验：从相同 SFT 开始，Answer 局部奖励 0，学习率 3e-6，
   KL beta 0.02；200 步，保留原 3000 步 scheduler horizon、3000 条训练池
   和相同打乱前缀。完成后重新从 checkpoint 加载双副本，验证固定 300 条。
4. 写出 `low_lr.json`，在弱效率优势试验之前停下检查稳定性与正确率。
   本脚本尚不自动实现或启动效率优势试验，也不会扩容数据集。

可单独运行 `--stage prepare`（无 GPU）、`smoke`、`confirm`、`low-lr`。

## 评估并行与一致性

正式方案是两个独立单请求推理副本，端口 8002/GPU 1 和 8003/GPU 0，
`eval_request_workers=8`，`eval_generate_batch_size=1`。
训练时先释放评估副本，再使用 GPU 0 训练、GPU 1 生成；训练后再启动评估副本。

增加 workers 不能让当前阻塞式 `/generate/` 自动变成 GPU 批处理。
可选的请求合批实现已经添加，但不是本轮正式默认值：此前 12 条 GPU 冒烟中，
合批使总耗时从约 99 秒变成 71 秒，却改变了 1 条最终答案、另 1 条的解释。
因此合批检查未通过，结果保存在 `optimization_validation_300/smoke.json`，
不可作为输出等价的加速方案。双副本也必须通过上述检查才可进入正式评估。

### 接受前的实测状态（2026-09-03）

双副本检查已经完成：12 条总耗时约 101 秒 → 65 秒（约 1.55 倍吞吐）。
逐题 F1、EM、检索轮数、支持文档覆盖计数和解析错误数全部相同，但 5 条
存在原始文本或轨迹差异，所以当前严格门槛仍然失败。
两个自有 GPU 服务已退出，固定 300 条评估和低学习率训练均尚未启动。
这些只是 12 条冒烟结果，不证明完整验证集指标不会受影响。
接受并行方案为新的统一推理口径，需要显式决定后才能改变门槛并重建三组基线；
不能直接把现有 `smoke.json` 的 `passed` 改为 true。

### 已确认采用新口径（2026-09-03）

用户明确确认“接受”。初始化方式如下（本次已经执行）：

```bash
bash scripts/run_grpo_optimization_validation.sh --stage prepare \
  --accept-parallel-smoke outputs/grpo_Meta-Llama-3-8B/optimization_validation_300_dual_gpu
```

该操作读取并校验原始 12 条预测、QID、模型身份、逐题 F1/EM、轮数、支持文档覆盖
和解析错误数，将证据文件哈希及已知差异记录到新计划的
`parallel_protocol_acceptance` 和 `parallel_protocol_acceptance.json`。
它不修改原始严格冒烟结果，不重新跑耗时冒烟；后续 `--stage all` 自动沿用已冻结的接受记录。
新协议标记为 `dual_gpu_single_prompt_8workers_v1`，固定 300 条结果还检查每组实际
workers、合批大小和服务端点与计划相符。其后质量门槛保持原样，不因接受并行而放宽。

## 输出

- `plan.json`、`configs/`：冻结的条件与来源指纹。
- `smoke.json`：一致性与计时检查。
- `confirmation.json`：三组固定 300 条分数、覆盖、轮数、逐题改善/退化数和门槛。
- `low_lr.json`：学习率试验结果、完整 checkpoint 路径和稳定性检查。
- `evaluations/*/throughput.json`：预测阶段用时（不含模型服务启动时间）。
- `logs/*-evaluation.log`、`logs/low_lr-training.log`：持续运行日志。

中断续跑后的计时仅表示本次进程用时，不能拿它估算从零开始的吞吐。
目前兼容原稳定性检查口径，零优势跳过步的 KL=0 尚未改成缺失值，
因此分析低学习率结果时还需单独看实际计算/更新步的 KL。
