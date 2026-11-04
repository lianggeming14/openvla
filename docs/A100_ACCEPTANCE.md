# A100 接入与基础实验验收

当前没有 A100 SSH 主机/OFT 目录连接，未执行真实权重加载、仿真 rollout、五次试采或二十初态正式评测。本文为可执行清单；CPU 测试、synthetic 演示、接口核对不能代替实机结果。私有原方案已完整读取，逐项证据见 [验收矩阵](REQUIREMENTS_TRACEABILITY.md)。

## 1. 环境预检

激活用户原本可运行 OFT 的环境，保留已有 MUJOCO_GL、PYOPENGL_PLATFORM、CUDA 配置。按 [README](../README.md) 安装工具及 analysis/video 依赖，设置两个绝对路径：

```bash
OFT_ROOT=/path/to/openvla-oft
DATA_ROOT=/path/to/trajectory-data/libero10-order
python -m vla_trajectory doctor --oft-root "$OFT_ROOT"
```

保存 GPU/驱动、Python/依赖版本、OFT revision 和预检日志。doctor 不加载模型；先确认用户原命令在相同环境正常运行。保留 checkpoint=`moojink/openvla-7b-oft-finetuned-libero-10`、suite=`libero_10`、`num_trials_per_task=1`、`load_in_4bit=True`，其余使用现有 OFT 设置，不训练或修改模型。

## 2. 五个不同初态试采

```bash
python -m vla_trajectory plan --stage pilot --output "$DATA_ROOT/plans/pilot.json"
python -m vla_trajectory collect \
  --oft-root "$OFT_ROOT" --output "$DATA_ROOT/pilot" \
  --plan "$DATA_ROOT/plans/pilot.json" \
  -- \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --num_trials_per_task 1 \
  --load_in_4bit True
```

默认只选 Soup + Tomato → Basket 的准确任务名。核对执行前保存的 `collection_plan_<hash>.json`：真实 Task ID、五个互异状态索引/内容指纹、状态库指纹、行 ID、对象对。每个初态调用原 run_task 一次，由原 eval 初始化模型一次；`num_trials_per_task` 仍为 1。计划模式要求 `initial_states_path=DEFAULT`，不能靠重复初态 0 替代五个初态。

逐个 episode 核验：

|项目|通过条件|证据|
|---|---|---|
|任务/状态|manifest 的任务、初态索引/指纹与计划逐行一致|计划、manifest、日志|
|逐步字段|validate 成功；保存动作前后观测、实际动作、原始模型动作、EEF/夹爪/可用物体位姿|NPZ/JSON、observation_fields|
|缺失字段|明确键/维度/缺失，不从 sim_state 猜值；必需字段缺失时核对环境配置|execution.observation_fields|
|Action Step|1-based 实际 policy env.step；settling 为 null；不按模型查询计数|steps.context|
|记录完整|执行/记录的 policy/settling 分别相等，recording_complete 和 step_count_complete 均 true|execution.json|
|官方结果|保存 success/failure/unknown、异常/超时；人工结果另列|manifest、日志|
|视频|各回合可播放 MP4+匹配 sidecar，首 obs 至末 next_obs，含 settling，命名含任务/初态|MP4、帧映射、人工播放|
|人工尝试|逐次抓放/重试/失败/未知均核对，导入修订成功|annotations.json、review HTML|

对每个真实目录执行：

```bash
python -m vla_trajectory validate EPISODE_DIR
python -m vla_trajectory annotate EPISODE_DIR
python -m vla_trajectory review EPISODE_DIR --output outputs/pilot-review.html
# 浏览器打开 HTML，逐帧复核后下载 JSON
python -m vla_trajectory import EPISODE_DIR /path/to/downloaded-annotations.json
python -m vla_trajectory analyze "$DATA_ROOT/pilot" --output outputs/libero10-pilot
```

每个尝试填写独立 ID、对象、尝试起止、抓取操作起点/事件、放置操作起点/事件、true/false/null 子任务结果；frame 是原始 0-based transition，事件对应 next_obs。未知留空，不以夹爪闭合或目标谓词推断抓取。全部核对后勾选 object_reviewed。重新导入旧 JSON 必须冲突。下载本身不写回数据集。

五次均核对后记录容量、吞吐、视频可播放与事件对齐情况，再进入正式阶段。

## 3. 二十不同初态正式评测

```bash
python -m vla_trajectory plan --stage formal --output "$DATA_ROOT/plans/formal.json"
python -m vla_trajectory collect \
  --oft-root "$OFT_ROOT" --output "$DATA_ROOT/formal" \
  --plan "$DATA_ROOT/plans/formal.json" \
  -- \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --num_trials_per_task 1 \
  --load_in_4bit True
```

默认索引 0–19，可显式指定至少二十个互异非负索引；执行前校验范围及状态内容互异。pilot/formal 可复用部分索引，但分别统计，正式阶段自身至少二十种状态，不用试采凑数。正式每回合重复第 2 节所有核验与人工导入，保留失败/未知/异常，不只选成功样本。

独立实验使用新目录；重跑计划生成新 episode，不自动去重/续跑。补采需记录原因及原始失败，报告列实际回合数、不同状态数和各类分母。

## 4. 三表、图、报告与结论

```bash
python -m vla_trajectory analyze "$DATA_ROOT/formal" \
  --output outputs/libero10-formal \
  --objects alphabet_soup_1 tomato_sauce_1
```

逐项核验 episode_statistics.csv、subtask_events.csv、task_summary.csv、每回合 PNG/SVG 和 analysis_report.md。抽查已知顺序、重试/未知/并列及失败样本；某种情况未出现就注明未观察到，不造数据。

- 成功率仅用官方已知成功/失败，未知另列；成功平均步数仅用官方成功且步数可信的回合。
- 完整顺序保留重抓；对象对比例只用已复核可比较样本，明确对象、分母、未知/并列数。
- 总步数排除 settling；执行不确定时留空，执行数只作下界。缺失记录不算完整轨迹。
- 抓取/放置操作各用人工起点至事件的 policy 步数，包含两端，缺边界留空；抓到放的事件间隔另列。
- EEF 长度含第一个 policy 动作，逐 transition 计算；缺坐标令完整长度未知，报告覆盖率、图不跨缺口连线，抓/放标记对齐 next_obs。
- 核对视频覆盖及匹配帧映射，实际播放每条 MP4；文件存在和映射匹配不等于内容已人工核验。

真实数据报告回答：对象顺序是否偏向固定方向，不同初态下顺序与轨迹怎样变化；给出样本量、已知分母、失败/未知及局限。观察不能直接证明主动规划、布局导致顺序改变或最优路径。多任务扩展、自动抓放识别与布局因果实验另作进阶验收。

## 5. 异常与接口

另在非正式目录做 Ctrl+C 验收：已提交步骤可读，状态 interrupted；环境步未返回时 step_count_complete=false。SIGKILL/断电不执行 finally，可能留下 recording，不能算完整成功。需恢复成可复核终态时，先确认写入进程退出并备份原 manifest，记录人工处置与原状态，保持 num_steps 不变、outcome=unknown，不把数据改成 complete 来通过验收。

缺编码器/视频失败应保留错误并返回失败。补导可用 `python -m vla_trajectory video EPISODE_DIR`，不覆盖已有输出；补齐后重新分析，保留原 execution 错误，不能掩盖缺失控制步。

模块接口需有 `run_episode(cfg, env, task_description, ...)` 及模块级 get_action/process_action/log_message。计划模式还需 run_task 的 cfg/task_suite/task_id/total_episodes/total_successes 参数，cfg 的 num_trials_per_task、initial_states_path；通常 cfg 还提供 num_steps_wait/task_suite_name。适配版更改接口时先核对，不改模型动作语义。短 deque 丢弃块前部时 chunk_index 可大于 0。

官方成功取环境 check_success/_check_success，无接口保留 unknown；自动目标谓词不等于中间抓取。保存模拟器状态便于同版本分析，不承诺跨版本精确重放。钩子仅同进程串行，多进程需各用独立目录。

仅在真实单个双物体任务的二十不同初态、完整轨迹/视频/人工顺序、三表/图/报告均核验后，才能宣布基础实验完成。原始数据保留用户服务器/私有存储，不上传公开 GitHub；代码和文档交付 GitHub，飞书仍待可用连接与明确接收位置。
