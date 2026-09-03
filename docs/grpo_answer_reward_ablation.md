# 固定评分与 Answer 局部奖励消融

训练正确性奖励统一调用 `answer_metrics.calculate_f1`。验证指标不变：删除标点、去冠词、规范空白后计算 token F1，只对标准答案评分，不取别名最高分。空答案得 0。该规则同时用于监控 F1、Answer 当轮正确性奖励和 terminal reward。

## 对照设计

| 条件 | aligned_control | no_answer_local |
| --- | --- | --- |
| 正确性评分 | 固定验证 gold-only F1 | 相同 |
| Answer 局部行为奖励权重 | 1.0 | 0.0 |
| 初始化 | 同一个 SFT adapter | 相同，非 control checkpoint |
| 训练问题、顺序、随机种子 | 固定前 200 题，seed 42 | 相同 |
| 验证 | 固定 90 题，各数据集 30，4 workers | 相同 |

`answer_local_reward_weight` 缩放当轮正确性、正确等待、错误拒答及过早作答四项；0 表示全部移除，1 保留当前方案。格式错误惩罚不缩放；终局 F1、终局支持覆盖/过早作答惩罚、Query/Evidence 局部奖励、三个角色的终局权重均不改变。它不是移除 Answer 的全部训练信号，也不保证与轮数有关的其他偏好全部消失。

保留 `max_samples=1000`、`max_total_samples=3000`、`max_steps=3000` 和原 90 次更新的 warmup。仅 `run_until_step=200`，不把 scheduler horizon 缩到 200。首次 200 题为 2Wiki 62、HotpotQA 64、MuSiQue 74；与上一轮实验的实际问题及顺序一致。生成轨迹在权重更新后可以不同，不能保证两组全程生成相同候选。

这是一轮单种子诊断，不是显著性/最终模型选择结论。两组固定跑到 200 题后评估，不复用分阶段早停控制器的质量门禁；训练器原有的非有限梯度防护保留。不要把两组优化器更新次数不同误认成不同训练问题预算。

## 运行

先只核对计划（不写文件、不启动服务或占用 GPU）：

```bash
cd /data/xudu/macorag
bash scripts/run_grpo_answer_reward_ablation.sh --dry-run
```

冻结配置和实验身份、不开始训练：

```bash
bash scripts/run_grpo_answer_reward_ablation.sh --prepare-only
```

在 macorag 环境运行实验：

```bash
bash scripts/run_grpo_answer_reward_ablation.sh
```

脚本自行启动训练用 LoRA 服务，默认 GPU 1、端口 8002；训练用 GPU 0。**请先确保两张 GPU 都空闲，不需要额外手动启动 vLLM。** 端口被占用会报错，不接管或停止已有进程。流程为 SFT 基线验证 → 新 control 训练/验证 → 新 no-answer-local 训练/验证 → 对照报告；每段只关闭自己创建的服务进程组。

默认目录：`outputs/grpo_Meta-Llama-3-8B/answer_local_ablation_200/`。

- `plan.json`：SFT 内容哈希、评分版本、源码哈希、完整数据身份、前 200 题顺序。
- `configs/`：冻结的两组训练配置与验证配置。
- `logs/`：服务、训练、验证日志；运行时终端会打印各日志路径。
- `<variant>/training/<timestamp>/`：各组独立训练产物，每 50 题保存完整 checkpoint。
- `evaluations/{sft,aligned_control,no_answer_local}/`：三份固定验证结果。
- `comparison.json`：每数据集 F1、相对 SFT 和 B−A 的变化、平均检索轮数、标注支持文档覆盖、格式指标、优化器更新/零优势跳过次数，以及实际训练序列一致性检查。

用同一命令重启可恢复到本组最近的完整 checkpoint；中断 checkpoint 之后的未提交工作会重做，旧日志保留，报告仅沿实际恢复链统计。若只中断验证，会加载该组的确切 checkpoint 后续评，不误用其他组权重。旧实验目录 `staged_early_stop` 不受影响。

计划被冻结后若代码、配置、SFT 或数据身份变化，会拒绝混跑；另起实验请用 `--run-dir outputs/grpo_Meta-Llama-3-8B/answer_local_ablation_200_v2`。奖励口径及局部权重进入 checkpoint 配置指纹，**旧奖励口径的 RL checkpoint 不再允许直接全状态续训**。旧文件不删除，本次两组从 SFT 开始。

## 如何判读

主要比较 `no_answer_local` 与 `aligned_control`，而非将旧奖励的 RL-200 直接作为对照。优先看 F1 是否恢复，同时检查支持文档覆盖、提前错误答案与检索轮数；检索次数增加本身不是成功。文档覆盖是文档级代理指标，不等于事实已完整理解。若去除局部偏好后只有轮数变长、正确率未提高，不支持其为主要退化原因。不要仅凭 90 题单种子结果决定正式扩大训练。
