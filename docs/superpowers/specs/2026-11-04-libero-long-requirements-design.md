# LIBERO-Long 原文要求对齐设计

## 来源与边界

已完整读取用户私有仓库 `lianggeming14/upload-files` 中《VLA 模型执行轨迹采集与任务顺序分析.md》全部七节、171 行、5645 字节，Git blob 为 `ff1ac6625ee3ff3917cdddcbe9ded101881c1a62`。本地原文副本位于被 Git 忽略的 `.requirements-source/`，不随公开源码包发布。本文为实现与验收说明，不复制私有正文。

本设计取代 2026-11-03 实现中尚未按原文核对的假设。用户已明确要求按原文执行并只要结果，沿用现有 Python/OFT 集成方式；具体接口、统计口径和可选依赖为本次实现选择。

本次功能及标注格式升级发布为 `0.2.0`，同步 pyproject、包内版本与下载链接，避免安装器将新发行包误判为原 `0.1.0`。旧 wheel 保留为历史文件。

保持官方模型，不训练或修改模型。模型 `moojink/openvla-7b-oft-finetuned-libero-10`，任务套件 `libero_10`（LIBERO-Long），保留用户已验证的 `num_trials_per_task=1`、`load_in_4bit=True`。

## 交付及统一数据约定

1. 在原有逐步 NPZ/JSON 记录上补充明确的任务、初态、试验计划身份。记录执行动作、EEF 位姿、夹爪、物体位姿及字段缺失情况；保留官方成功判断。
2. 原始 `record_step` 为包括 settling 的零基 transition 索引。对外 `Action Step` 从 1 开始，仅为实际 policy 环境控制步骤，settling 不编号，模型推理次数不参与计数。人工标注仍保存原始 frame 引用，表格按映射导出动作步。
3. 人工抓放尝试独立于自动目标谓词。每次尝试保存独立 ID、对象、起止/抓取/放置 frame、三态子任务成功、来源及备注。保留重复对象和失败重试，未知为 null；没有证据时不推断抓取。
4. `manifest.outcome` 为官方回合结果，人工 outcome 另列，不覆盖正式成功率。保存已执行与已记录步数的区别；发生落盘错误时不得把缺失轨迹当完整数据。

### 人工标注接口

annotation schema v2 新增 `object_attempts` 和 `object_reviewed`，原始 recorder schema 保持 v1。每条尝试的完整字段：

```json
{"attempt_id":"attempt-001","object":"alphabet_soup_1","start_frame":12,"end_frame":28,"grasp_start_frame":12,"grasp_frame":15,"place_start_frame":25,"place_frame":28,"subtask_success":true,"source":"human_review","notes":""}
```

对象及六个 frame 可为 null，成功为 true/false/null，frame 为原始零基 transition 索引；新增操作起点也兼容缺省为 null。已知起止/抓放时需满足范围、先后及回合边界；拒绝明确落在 settling 的抓放事件。允许未知阶段的旧记录，但分析必须标明无法转换动作步。`object_reviewed=false` 表示尚未完成抓放核对。

抓取/放置事件对应当前 transition 的动作后 next_obs。抓取操作耗时为人工定义的 `grasp_start_frame` 至 `grasp_frame` 的 policy 动作数（包含两端），放置操作同理使用 `place_start_frame` 至 `place_frame`；缺边界时不估算。尝试总步数使用 start/end 包含两端；抓取至放置事件间隔使用 `place_action_step-grasp_action_step`，不把该间隔宣传成纯放置耗时。

旧 v1 文档在内存迁移到 v2，补空尝试和未复核标志。旧导入缺少新字段时保留已保存的尝试；只有明确提交的 v2 空数组才表示清空。自动 events、source、episode ID 和步数继续受原始证据、终态、快照、文件锁和修订号保护。

离线页面提供逐帧播放、policy 动作步显示、结构化尝试表、当前帧设置抓/放、重复尝试与三态结果，下载后用现有 import 写回。

### 试采、固定初态与视频接口

新增 `collection.create_plan(task_name, stage, init_state_indices=None, *, task_suite_name='libero_10', object_pair=None) -> dict`、`validate_plan(plan) -> dict`。`pilot` 默认五个索引 0–4，`formal` 默认二十个索引 0–19；显式 pilot 必须五项，formal 至少二十项。索引唯一、非负，正式执行时验证范围和初态内容互异。

默认双物体任务准确名称：`LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`。按名称解析真实任务编号；默认分析对象为 `alphabet_soup_1`、`tomato_sauce_1`，不把其他任务强行映射到这两个物体。

`oft.instrument(module, output_root, *, run_metadata=None, collection_plan=None)` 保持旧调用兼容。计划模式可逆包装官方 `run_task`，跳过其他任务，为每个指定初态调用原 run_task 一次，保留配置中的 `num_trials_per_task=1`；官方 eval 只加载模型一次。计划模式先要求官方 `initial_states_path=DEFAULT`，不复制推理循环。

每个 episode 元信息包含 task_id/task_name/init_state_id/init_state_index/trial_index/stage/plan_row_id/plan_hash/state_sha256/analysis_object_pair；运行前持久化解析后的初态清单及状态库指纹。执行计数写 `execution.json`，含 episode_id、已执行/已记录 settling 和 policy 步数，采集异常也保留可核验的计数。

`execution.json` 统一字段为 `schema_version=1`、`episode_id`、`executed_policy_steps`、`recorded_policy_steps`、`executed_settling_steps`、`recorded_settling_steps`、`recording_complete`、`step_count_complete`，可附视频状态/错误。环境步已返回而落盘失败时执行计数仍可可信；环境步本身异常导致无法确认是否完成时 `step_count_complete=false`，执行数只是下界，正式 Total Steps 留空。

新增 `video.export_episode_video(episode_path, output=None, *, fps=None, include_settling=True, camera='agentview_image', rotate180=True) -> Path`。完整视频取第一个 obs 和每一步 next_obs，包含终态；默认包含 settling，视频帧与原始 record_step、动作步的映射写入 sidecar。文件名带任务 ID、初态 ID、episode ID。播放 fps 和实际 control_freq 分开记录。计划采集自动导出视频，失败必须报告，不能把缺视频回合算成完整实验交付。

## 三张统计表、图与报告

新增 `analysis.analyze_episode(path, *, object_pair=None, eef_key='robot0_eef_pos') -> dict` 和 `analysis.export_analysis(root, output, *, object_pair=None, include_synthetic=False, plots=True) -> dict`。使用 stdlib CSV；Matplotlib 为可选 analysis 依赖，视频编码依赖单列，不升级模型依赖。

- `episode_statistics.csv`：Task ID、Init State ID、Episode ID、Grasp Order、Success、Total Steps、EEF Path Length；附阶段、官方/人工结果、记录完整性和缺失指标。
- `subtask_events.csv`：Episode ID、Object、Grasp Step、Place Step、Subtask Success；逐尝试一行，附 Attempt ID、原始 frame、尝试范围、Grasp Operation Steps、Place Operation Steps、Attempt Steps、Grasp To Place Interval Steps 和备注，不能按对象去重。
- `task_summary.csv`：Task ID、Episodes、Success Rate、A→B Ratio、B→A Ratio、Mean Successful Steps；附对象定义、已知/未知结果分母、可比较/未知/并列顺序样本数及不同初态数。不同任务、pilot/formal 不混合。

正式成功率分母为官方已知成功和失败；未知另计。平均成功步数只纳入官方成功且步数可信的回合。完整抓取序列保留 A→B→A 等重抓；对象对比例只使用已复核且顺序可可靠比较的样本，未知和同一步并列不硬分配顺序。

EEF Path Length 是所有 policy transition 的 `sum(norm(next_xyz-obs_xyz))`，包含第一个 policy 动作、排除 settling。若有任何应计步缺失坐标，总长为未知，同时提供已观测长度和覆盖率；不跨缺口连线。绘制等比例 XYZ 轨迹及坐标随动作步变化图，用不同标记标出抓取和放置。

生成中文 `analysis_report.md`，描述样本范围、顺序观测、成功率和步数分布，并列出未知、失败、缺失、不同初态及视频覆盖。分析只报告观察到的偏好和变化，不由有限样本推断主动规划、因果关系或最优路径。

默认排除 synthetic；演示必须显式开启并在报告醒目标记。重复 episode ID 必须拒绝，以免复制文件重复计数。CSV 文本正确处理中文、逗号、换行和表格公式前缀。

## 验收

代码侧：原有测试、结构化标注往返与迁移、policy 计数、5/20 初态计划、最终视频帧、三张 CSV、路径长度、未知分母、SVG/PNG 科学图和报告均有可运行 CPU 验证；对新增重要行为先记录失败再实现。独立审查对照本设计及私有原文。

实机侧：至少一个双物体任务，先 5 次试采，再 20 个不同初态；保存完整 NPZ/视频/人工顺序，生成表、图和报告，才能宣称基础实验验收完成。当前未获得 A100 SSH 主机和目录，相关真实模型结果保持待验，不生成假实验数据。飞书发送仍需可用连接。
