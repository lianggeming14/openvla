# openvla · OFT / LIBERO 轨迹采集与顺序分析

接入你已经搭好的 OpenVLA-OFT + LIBERO 环境，逐步记录真实执行动作，人工复核抓放事件，再生成三张 CSV、末端轨迹图和中文分析报告。代码版本 **0.2.0**，不训练或修改模型。

## 从这里开始

**按顺序执行 [A100 详细操作手册](docs/A100_ACCEPTANCE.md)。** 每一步说明在哪里执行、命令作用、预期文件和检查方法，包括新终端恢复变量、逐回合标注、JSON 上传导入及正式采集。

- **蓝色替换项版本：**[下载 HTML 操作手册](https://raw.githubusercontent.com/lianggeming14/openvla/main/docs/A100_RUNBOOK.html)，保存为 `A100_RUNBOOK.html` 后用本机浏览器打开。GitHub 文件页显示 HTML 源码，下载打开才显示排版及蓝字。
- 工程根目录已填为 `/HUBU-AI096/lgm/ICASSP/peft-in-openvla`，不用替换。
- 只有 `REPLACE_WITH_*` 是需要手动替换的值，例如原 OFT conda 环境名、实际回合目录、上传后的标注 JSON 路径。Markdown 用 **🔵** 提示；HTML 在命令中直接标蓝。替换后再执行。
- 使用 Bash，在原本能运行 OFT 的 conda 环境中操作。工具库建议放在 `/HUBU-AI096/lgm/ICASSP/openvla-trajectory-tools`，与 OFT 工程分开。

本轮说明已使用你提供的服务器路径；尚未连接 A100 执行真实实验。CPU/CI 测试不能代替这台服务器上的模型、渲染、传感器与实验验收。

## 已确定的实验

|项目|约定|
|---|---|
|任务集|官方 LIBERO-Long，即 `libero_10`，含 10 个任务|
|先执行的任务|将汤罐和番茄酱罐都放进篮子|
|准确任务名|`LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`|
|对象|`alphabet_soup_1`、`tomato_sauce_1`|
|模型|`moojink/openvla-7b-oft-finetuned-libero-10`|
|保留参数|`--num_trials_per_task 1 --load_in_4bit True`|
|试采|同一任务，官方初态索引 0–4，共 5 回合|
|正式采集|同一任务，官方初态索引 0–19，共 20 回合，阶段内状态内容互异|

先检查试采的字段、视频与标注，再进入正式阶段。pilot/formal 分开，试采不计入正式 20 个不同初态。计划执行前检查初态范围及内容指纹，按任务名称解析真实 Task ID，避免依赖任务排序。

官方已提供场景、成功条件和初态库。本次运行现成 checkpoint，采集模型自己的轨迹，无需为此次评测额外下载训练用 RLDS 示范数据。成功条件要求两物体最终都进篮子，没有规定先后，适合观察顺序偏好。该 checkpoint 已在 LIBERO-10 上微调过，20 初态用于已有任务上的行为分析，不能单凭此称为未见任务泛化测试。

来源：[官方任务清单](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/benchmark/libero_suite_task_map.py)、[本任务定义](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/bddl_files/libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl)、[OFT 评测说明](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md)。

## 数据怎样产生

**从 `python -m vla_trajectory collect` 启动才会开启本工具的记录。** 它在同一进程接入原 OFT 评测流程，退出时恢复包装的函数，不改写 OFT 源文件。其余模型参数放在 `--` 后转发；适配版接口需要先按手册预检。

1. 开始回合：创建独立目录，保存任务、初态、配置和计划身份。
2. 每个实际 `env.step` 返回后：将动作前后观测、实际动作及上下文写入 NPZ/JSON，再更新 manifest 提交计数。
3. 回合结束：保存官方结果、执行与记录计数。带 `--plan` 的采集自动导出从首个 obs 到末次 next_obs 的完整视频及帧映射。
4. 运行后：用离线 HTML 逐帧标注抓放和重试，下载 JSON、上传服务器，再用 `import` 写回对应回合。
5. 完成复核：用 `analyze` 生成三表、图和报告。

Action Step 从 1 开始，只计实际 policy 控制步，排除 settling 和推理查询。原始 frame 从 0 开始且包含 settling；事件对应动作后的 `next_obs`。抓取/放置操作耗时需要分别标记操作起点与完成点，抓到放的事件间隔不等于纯放置耗时。

不带 `--plan` 的普通采集仍支持原全部套件流程，但不自动导出本工具的完整 MP4；可另用 `video` 导出。旧 `scripts/collect_libero10_4bit.sh` 是全套件模式，**本次单任务 5/20 实验请使用详细手册中的计划模式命令**。仅运行原 eval 命令不会开启本工具记录，未保存的精确动作/状态无法事后从视频补回。

## 运行后得到什么

本轮数据位于工程内 `trajectory-data/libero10-order/run-日期时间/`。其中 `run-env.sh` 保存本轮目录变量，`plans/` 保存计划，`pilot/` 与 `formal/` 保存回合，`reviews/` 保存离线审阅页，`annotation-inbox/` 接收上传的标注 JSON，`analysis/` 保存分析产物。

每个回合包含 `manifest.json`、`execution.json`、逐步 `steps/*.npz` 和 `steps/*.json`、MP4 及同名帧映射 JSON。`annotations.json` 在 annotate/import 后持久保存。

|分析产物|内容|
|---|---|
|`episode_statistics.csv`|任务/初态/回合、抓取顺序、官方结果、总步数、EEF 长度及缺失指标|
|`subtask_events.csv`|每次抓放尝试一行，保留重复抓取、失败与未知|
|`task_summary.csv`|成功率、两种顺序比例、成功回合平均步数、不同初态数|
|`plots/*_eef.png`、`plots/*_eef.svg`|XYZ 轨迹和坐标随动作步变化，标记抓放事件|
|`analysis_report.md`|样本范围、顺序观察、成功率、步数分布及未知/缺失|

成功率使用官方已知成功/失败；人工结果不覆盖官方结果。顺序比例只用已复核、可比较的样本，保留 A→B→A 等重抓序列。EEF 长度逐 policy transition 计算、包含首个动作，坐标缺失时完整长度未知、图不跨缺口连线。失败、异常和未知回合均保留。

## 安装包与验证

[源码 ZIP](https://raw.githubusercontent.com/lianggeming14/openvla/main/downloads/openvla-source.zip) 包含正式文档和代码；[0.2.0 wheel](https://raw.githubusercontent.com/lianggeming14/openvla/main/downloads/openvla_trajectory_tools-0.2.0-py3-none-any.whl) 是运行包。本次仅更新说明，不改变运行版本；旧 0.1.0 wheel 保留历史。

基础依赖为 Python 3.10+、NumPy、Pillow。`analysis` extra 提供 Matplotlib，`video` extra 提供 imageio/FFmpeg；工具不声明 torch、CUDA、transformers、LIBERO 为安装依赖，使用现有 OFT 环境。

开发验证：安装 `.[test,analysis,video]` 后运行 `python -m pytest -q`。离线演示可用 `python -m vla_trajectory demo --output data/demo`，分析时须显式 `--include-synthetic`，演示不计入真实实验。

原始要求已完整读取，证据见 [要求与验收矩阵](docs/REQUIREMENTS_TRACEABILITY.md) 和 [交付记录](docs/IMPLEMENTATION_STATUS.md)。基础实验验收需完成真实单个双物体任务的 20 个不同初态、完整轨迹/视频、人工标注和分析产物。

公开 GitHub 只保存代码、正式文档与发行包；私有原文、权重、原始轨迹和凭据不上传。飞书同步尚缺可用连接和明确接收位置，当前未发送。
