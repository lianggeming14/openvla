# openvla · OFT / LIBERO 轨迹采集与顺序分析

本工具接入已经搭好的 **OpenVLA-OFT + LIBERO**，保存实际控制步骤，人工复核抓放尝试，导出三张 CSV、末端轨迹图和中文分析报告。不训练或改动模型。

已于 2026-11-04 完整读取私有原方案并补齐实现，证据见 [要求与验收矩阵](docs/REQUIREMENTS_TRACEABILITY.md)。当前代码版本 **0.2.0**。CPU 验证与真实实验分开：A100 上的五次试采、二十不同初态正式评测、真实视频及实验结论仍待执行。当前没有服务器连接或飞书发送认证，不能声称已验收或发送。

## 安装到现有环境

仓库：[lianggeming14/openvla](https://github.com/lianggeming14/openvla)。发行文件使用 [源码 ZIP](https://raw.githubusercontent.com/lianggeming14/openvla/main/downloads/openvla-source.zip) 与 [0.2.0 wheel](https://raw.githubusercontent.com/lianggeming14/openvla/main/downloads/openvla_trajectory_tools-0.2.0-py3-none-any.whl)；是否已发布以 [交付记录](docs/IMPLEMENTATION_STATUS.md) 为准。历史 0.1.0 wheel 保留，但不含本轮新增功能。

在原本能运行 OFT 的 conda 环境中：

```bash
git clone https://github.com/lianggeming14/openvla.git
cd openvla
python -m pip install -e . --no-deps --no-build-isolation
python -m pip install 'matplotlib>=3.7' 'imageio>=2.31' 'imageio-ffmpeg>=0.4.9'
python -m vla_trajectory doctor --oft-root /path/to/openvla-oft
```

基础依赖为 Python 3.10+、NumPy、Pillow，安装要求已有 setuptools>=64、wheel。`analysis` extra 提供 Matplotlib，`video` extra 提供 imageio 与 FFmpeg；也可运行 `python -m pip install -e '.[analysis,video]' --no-build-isolation`。工具不声明或升级 torch、CUDA、transformers、LIBERO。`doctor` 只做预检，不证明模型加载、渲染或任务成功。

## 双物体任务：先五次试采，再二十初态正式采集

在本工具库根目录执行，替换两个绝对路径：

```bash
OFT_ROOT=/path/to/openvla-oft
DATA_ROOT=/path/to/trajectory-data/libero10-order
python -m vla_trajectory plan --stage pilot --output "$DATA_ROOT/plans/pilot.json"
python -m vla_trajectory plan --stage formal --output "$DATA_ROOT/plans/formal.json"

python -m vla_trajectory collect \
  --oft-root "$OFT_ROOT" --output "$DATA_ROOT/pilot" \
  --plan "$DATA_ROOT/plans/pilot.json" \
  -- \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --num_trials_per_task 1 \
  --load_in_4bit True
```

默认任务准确名称为 `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`，对象为 `alphabet_soup_1`、`tomato_sauce_1`。计划按名称解析真实 Task ID；试采索引 0–4，正式索引 0–19。执行前校验状态范围与内容互异，保存状态库指纹。`S000` 表示状态库索引 0。可用 `--state-ids 9 7 5 3 1` 选择试采初态；其他任务加 `--task-name EXACT_NAME --objects OBJECT_A OBJECT_B`，不会继承汤罐/番茄罐对象。模板位于 [plans/](plans/)。

计划模式仅运行所选任务，每个初态调用原 `run_task` 一次，保留 `num_trials_per_task=1`，由原 eval 初始化模型一次；要求 `initial_states_path=DEFAULT`。**全部模型参数放在 `--` 后原样转发**，保留用户四项参数，其余沿用现有 OFT 默认设置。钩子退出时恢复原函数，不改写 OFT 源文件；上游自身的 checkpoint 配置更新行为保持原样。

先完成下一节五个回合的核验，再正式采集：

```bash
python -m vla_trajectory collect \
  --oft-root "$OFT_ROOT" --output "$DATA_ROOT/formal" \
  --plan "$DATA_ROOT/plans/formal.json" \
  -- \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --num_trials_per_task 1 \
  --load_in_4bit True
```

不带 `--plan` 仍支持原全部套件采集与 `scripts/collect_libero10_4bit.sh`。该模式下 `libero_10`、每任务一次通常为十个任务各一回合，不能代替单任务二十初态实验；自动完整视频导出仅在计划采集模式触发。

## 逐回合复核与视频

用实际 episode 路径替换 `EPISODE_DIR`：

```bash
python -m vla_trajectory validate EPISODE_DIR
python -m vla_trajectory annotate EPISODE_DIR
python -m vla_trajectory review EPISODE_DIR --output outputs/review.html
# 将 HTML 复制到本机双击，复核后下载 JSON
python -m vla_trajectory import EPISODE_DIR /path/to/downloaded-annotations.json
python -m vla_trajectory summary EPISODE_DIR
```

离线 HTML 支持两路相机、连续播放、逐帧/动作步定位、结构化尝试表和当前帧设置抓/放。每次尝试使用独立 ID，保留重复对象、失败重试与顺序变化。填写对象、尝试起止、抓取操作起点/事件、放置操作起点/事件与 true/false/null 子任务结果；未知留空，不能以目标谓词或夹爪闭合替代抓取证据。完成核对后勾选 `object_reviewed`。自动谓词与人工抓放分别保存。

原始 frame 从 0 开始并包含 settling；**Action Step 从 1 开始，只计实际 policy `env.step`，不计推理查询或 settling**。事件对应动作后 `next_obs`。操作步数用人工起点至事件的 policy 步数，包含两端；抓到放的事件间隔另列，不当作纯放置耗时。缺边界或未知阶段不估算。

下载后必须 `import` 才写回数据集；修订冲突或证据变化会拒绝导入，继续编辑前重新导出 HTML。旧 v1 标注在内存迁移到 v2，旧导入保留已有抓放尝试。`recording` 回合仅预览，终态才能保存。

计划采集自动生成带任务、初态、episode ID 的 MP4 与同名 JSON 帧映射。完整视频为首个 `obs` 加每一步 `next_obs`，包含终态及默认的 settling。编码失败写入 `execution.json` 并返回错误，不能算完整交付。独立导出或补导视频：

```bash
python -m vla_trajectory video EPISODE_DIR
python -m vla_trajectory video EPISODE_DIR --output outputs/check.mp4 --fps 10
```

不覆盖已有 MP4/sidecar。默认主视角并旋转 180°；可用 `--camera robot0_eye_in_hand_image`、`--no-rotate180`、`--exclude-settling`。播放 fps 与真实 `control_freq` 分别记录。补导后重新分析并核验完整性，保留原 execution 错误作为运行证据。

## 三张表、图与报告

完成正式阶段的逐回合复核导入后：

```bash
python -m vla_trajectory analyze "$DATA_ROOT/formal" \
  --output outputs/libero10-formal \
  --objects alphabet_soup_1 tomato_sauce_1
```

命令返回简洁 JSON 计数与路径，生成：

- `episode_statistics.csv`：任务/初态/回合、完整抓取序列、官方成功、总控制步数、EEF 长度及覆盖/缺失指标。
- `subtask_events.csv`：每次尝试一行，抓/放动作步、独立操作步数、尝试范围、结果与备注。
- `task_summary.csv`：分任务、阶段、对象对汇总成功率、A→B/B→A 比例、平均成功步数与不同初态数。
- `plots/*_eef.png`、`plots/*_eef.svg`：等比例 XYZ 路径、坐标随动作步变化，标记人工抓/放事件。
- `analysis_report.md`：中文样本范围、顺序观察、成功率与步数分布，列出未知、缺失及视频覆盖。

成功率分母只含 `manifest.outcome` 的官方已知成功/失败，未知另计；人工结果不覆盖官方结果。平均成功步数只用官方成功且执行计数可信的回合。对象对顺序比例只用已复核、可比较样本，并列/未知单列；完整序列保留 A→B→A。pilot/formal 分组，不拿试采补正式数量。

EEF 长度为所有 policy transition 的 `norm(next_xyz-obs_xyz)` 之和，包括第一个动作、排除 settling。任何应计坐标缺失令完整长度未知，提供已观测长度/覆盖率，图不跨缺口连线。已执行与已记录步数分开，环境步异常无法确认时 Total Steps 留空，执行数只作下界。观察到顺序偏好或变化不能证明主动规划、因果关系或最优路径。

## Synthetic 开发演示

```bash
python -m vla_trajectory demo --output data/demo
# 用 demo 输出路径替换 EPISODE_DIR
python -m vla_trajectory review EPISODE_DIR --output outputs/demo-review.html
python -m vla_trajectory analyze data/demo --output outputs/demo-analysis --include-synthetic
```

默认排除 synthetic，显式包含后报告/图均标演示；`--no-plots` 可只导表与报告。演示不产生真实模型结论，不算 A100 验收。

## 数据与验证边界

```text
trajectory-data/
  collection_plan_<hash>.json   # 解析任务、状态指纹、行清单
  <episode-id>/
    manifest.json              # 元信息、官方结果、已提交步数
    execution.json             # 执行/记录计数、完整性、传感器映射、视频状态
    steps/000000.json          # 时间、phase、动作块、谓词证据
    steps/000000.npz           # obs/*、next_obs/*、action、extra/*，禁用 pickle
    annotations.json           # v2 人工尝试与受保护的自动证据
    task-*_init-*_episode-*.mp4
    task-*_init-*_episode-*.json
```

`execution.observation_fields` 报告 EEF/夹爪/目标物体的实际键、维度、存在情况、单位/四元数顺序，不从扁平 `sim_state` 猜位姿。每步原子落盘，manifest 范围外孤立文件不读入。同步图像/NPZ 增加耗时与容量，先实测五次试采。完整模型动作块只存于首次执行步，未执行动作不计 transition；`chunk_index` 保留真实源索引，短 deque 丢弃块前部时可不从 0 开始。

```bash
python -m pip install -e '.[test,analysis,video]'
python -m pytest -q
```

CI 安装全部 extras 并运行真实 MP4 编码/解码测试。本地没有 imageio_ffmpeg/FFmpeg，该项明确 skip；帧序/终态/映射/错误测试使用编码边界替身，不等于实际编码通过。真实验收见 [A100 清单](docs/A100_ACCEPTANCE.md)，发布验证见 [交付记录](docs/IMPLEMENTATION_STATUS.md)。

公开交付仅含代码、测试、脚本、正式文档与明确的源码/安装包；私有原文、权重、原始轨迹、缓存与凭据不上传。GitHub 更新以远程核验为准；飞书待可用连接和明确接收位置，当前尚未发送。
