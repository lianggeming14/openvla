# A100 接入与验收

本地开发机没有用户的 A100、checkpoint 或完整 OFT/LIBERO 环境。这里给出可执行验收步骤，不能把它当作已完成的 GPU 测试报告。

用户已提供现用命令：checkpoint=`moojink/openvla-7b-oft-finetuned-libero-10`、suite=`libero_10`、trials/task=`1`、`load_in_4bit=True`。对应采集命令见 README，启动脚本为 `scripts/collect_libero10_4bit.sh`。上述参数原样保留，其余沿用用户现有 OFT 默认设置。

1. 在同一个 conda 环境、同一个 checkpoint、相同 seed 和初始状态下，先确认原始 OFT 命令成功运行。
2. 执行 `python -m vla_trajectory doctor --oft-root /path/to/openvla-oft`。此命令只检查包版本/文件/GPU 信息，不加载权重、不验证渲染。
3. 将原入口替换成 README 的 `collect`，其余参数保持不变。保留已设置好的 `MUJOCO_GL`、`PYOPENGL_PLATFORM`、CUDA 环境配置。
4. 用少量 trial 采集，确认 episode 目录中有主视角、腕部图像，`manifest.metadata.config` 与本次配置一致。两套命令都保存日志，比较执行动作、成功结果及初始状态；记录器本身会增加磁盘写入延迟。
5. 对一个 episode 执行 `validate`，检查每步 `obs`/`next_obs` 连续、action 为 7 维、chunk_id 从 0 开始且索引对应已执行动作；settling 步骤不计作策略动作。`chunk_index` 是完整输出中的源索引，短 deque 会丢弃前部动作，首个索引可能大于 0。
6. 执行 `annotate`、`review`，人工核对至少一个目标谓词的变化是否与图像一致。自动标签来自目标谓词，不包含所有中间抓取/接触事件；需要这类标签时在复核页添加人工片段。
7. 下载并 import 人工 JSON，确认新 revision；再次导入旧文件必须报冲突而不是覆盖。
8. 做一次 Ctrl+C 验收：已提交步骤可 validate，回合状态为 interrupted。外部 SIGKILL、断电等不运行 Python finally 的情况会保持 recording；这些也必须排除完整成功集。recording 回合只允许标注预览，不允许保存；强杀后的数据需要先确认写入进程已经退出，再人工将 manifest 的 status 改为 interrupted、outcome 改为 unknown，保留 num_steps 原值，之后 validate 再标注。
9. 再扩大 trial 数量，统计每回合磁盘容量和采集速度。保留失败和超时回合，训练/分析时显式筛选，不删除证据。

## 用户适配版接口

当前 collect 会 import `experiments.robot.libero.run_libero_eval` 并调用它装饰后的 `eval_libero()`；命令行参数由其原 draccus 解析。要求：

- `run_episode` 参数包含 `cfg, env, task_description`，以原来的方式调用模块全局 `get_action`、`process_action` 和 `log_message`。
- cfg 提供 `num_steps_wait, task_suite_name`；`TASK_MAX_STEPS` 存在时记录最后一步的 timeout/truncated。若你删除此常量，原程序照常停止，但不会有推断出的 truncated 标签。
- LIBERO env 返回观测字典，step 返回 `(obs,reward,done,info)`；记录器代理也可读 5 元组，但官方 OFT 自身的解包是否兼容仍由原代码决定。
- success 优先调用环境 `check_success/_check_success`；没有接口就存 unknown，绝不只凭 done 标成功。
- 谓词从内层环境 `parsed_problem['goal_state']` 和 `_eval_predicate` 读取。不可用或求值失败均存 unknown/diagnostic。

若你的代码改变了这些接口，应提供该评测脚本以做针对性适配。不要为了兼容记录器改动模型动作语义。

## 当前范围

保存可用的初始与逐步模拟器状态，便于在同版本环境分析；没有额外归档场景 XML、assets 和整个模型环境，因此不承诺跨版本精确重放。当前钩子仅支持同进程串行 episode；并行采集请每个进程独立启动并使用不同输出目录。

待核对：私有仓库原方案中的字段、阶段分类及任务顺序要求；真实 A100 的一次小规模采集及人工复核；GitHub 远程发布状态。
