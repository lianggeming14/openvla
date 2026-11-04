# openvla — OFT / LIBERO 轨迹记录与标注

这是独立的采集工具库，接入你已经搭好的 **OpenVLA-OFT + LIBERO**。仓库名为 `openvla`，不是上游 OpenVLA 模型仓库，也不重复安装模型环境。

当前实现依据你确认的 OFT + LIBERO 环境。私有仓库中的《VLA 模型执行轨迹采集与任务顺序分析》正文尚未读取，不能宣称已逐项符合该方案。开发机的 CPU 测试不等于你的 A100 实测。

## 已实现的功能

- **完整逐步记录**：主视角/腕部相机、数值观测（末端/夹爪/物体等，以环境实际提供字段为准）、动作前后状态、原始模型动作、实际执行动作、动作块编号与块内位置、查询时模型输入、初始状态及可用的模拟器状态。
- **每回合独立保存**：配置、种子、任务文本、BDDL 路径、OFT Git 版本；成功、失败、异常、中断分开。稳定场景的等待步骤标记为 `settling`，策略步骤为 `policy`。
- **自动标注**：使用 LIBERO 的 BDDL 目标谓词，记录初始状态、达成、撤销、未知。不是视觉模型猜标签；夹爪闭合不会直接标成抓取成功。
- **人工复核**：生成可离线双击打开的 HTML，两路相机逐帧查看、前后观测切换、编辑时间段标签和结果、下载 JSON，再用命令导入。支持版本冲突检测，保留原始自动证据。
- **顺序汇总**：首次观测为真、实际观测到的达成顺序和最终状态。该顺序仅是执行中观测到的顺序，不代表任务必须遵守的因果顺序。

## 1. 安装到现有环境

仓库地址：[lianggeming14/openvla](https://github.com/lianggeming14/openvla)。可直接下载 [源码包](https://raw.githubusercontent.com/lianggeming14/openvla/main/downloads/openvla-source.zip) 和 [Python 安装包](https://raw.githubusercontent.com/lianggeming14/openvla/main/downloads/openvla_trajectory_tools-0.1.0-py3-none-any.whl)。

在 A100 上激活你原本能运行 OFT 的 conda 环境，然后克隆并安装本工具库：

```bash
git clone https://github.com/lianggeming14/openvla.git
cd openvla
python -m pip install -e . --no-deps --no-build-isolation
python -m vla_trajectory doctor --oft-root /path/to/openvla-oft
```

本工具需要 Python 3.10+、NumPy、Pillow；不安装或升级 torch、CUDA、transformers、LIBERO。若已有环境缺少 NumPy/Pillow，再单独安装。`--no-build-isolation` 要求已有 setuptools>=64 和 wheel。

采集钩子不改写 OFT 源码。它会调用原评测入口；[上游模型加载器](https://github.com/moojink/openvla-oft/blob/main/experiments/robot/openvla_utils.py) 本身可能更新本地 checkpoint 的配置和建模文件，这部分行为与直接运行原评测相同。

## 2. 先验证记录、标注链路

```bash
python -m vla_trajectory demo --output data/demo
# 上条命令输出 EPISODE_DIR；下文用实际路径替换
python -m vla_trajectory validate EPISODE_DIR
python -m vla_trajectory annotate EPISODE_DIR
python -m vla_trajectory review EPISODE_DIR --output outputs/review.html
```

将 HTML 复制到本机并双击打开。它不依赖服务器、网络或 GPU。编辑后点击下载，再执行：

```bash
python -m vla_trajectory import EPISODE_DIR /path/to/downloaded-annotations.json
python -m vla_trajectory summary EPISODE_DIR
```

**下载不是保存到数据集**：只有 `import` 成功后，`annotations.json` 才更新。导入后若继续编辑，请重新导出 HTML，避免使用旧版本。重复 `annotate` 不覆盖人工标签。演示数据明确标记 `synthetic=true`，不能用作模型评测结果。

只能向 `complete/error/interrupted` 回合保存标注；`recording` 状态只允许预览，以免新步骤让已保存标注失效。若标注写入进程崩溃留下 `.annotations.lock`，确认没有写入进程后再手动移除该锁文件。

## 3. 接入你现有的 OFT 测试命令

### 你提供的 LIBERO-10 / 4-bit 命令

安装本工具后，在原来执行评测的 **OFT 仓库根目录**运行以下命令。模型及评测参数与你提供的命令完全相同，其他参数仍使用你当前 OFT 代码的默认值：

```bash
python -m vla_trajectory collect \
  --oft-root "$PWD" \
  --output ./trajectory-data/libero10-4bit \
  -- \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --num_trials_per_task 1 \
  --load_in_4bit True
```

也可以从本工具库运行现成的启动脚本：

```bash
bash scripts/collect_libero10_4bit.sh /path/to/openvla-oft /path/to/trajectory-data/libero10-4bit
```

标准 `libero_10` 含 10 个任务，`num_trials_per_task=1` 是每个任务一回合，正常完整运行约为 10 回合；不是只采一条轨迹。4-bit 设置由原 OFT 加载器处理，记录器不改动量化；本地只验证了参数转发，尚未在 A100 上加载这个 checkpoint。

原始命令已收到并适配；仍需你 A100 上实际执行以验证模型加载、渲染及采集。采集完成后，按第 2 节对输出中的具体 episode 目录运行 annotate/review。

### 通用接入方式

假设原来在 OFT 目录执行：

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /path/to/checkpoint \
  --task_suite_name libero_spatial \
  --num_trials_per_task 1 \
  --use_l1_regression True \
  --use_proprio True \
  --num_images_in_input 2 \
  --num_open_loop_steps 8
```

现在把入口替换为下面这样，**`--` 后保留你原来已经验证可用的所有参数**，包括与你 checkpoint 匹配的 action head、proprio、FiLM、图像数等。上述参数只是结构示例，不能替代你 checkpoint 的配置。

```bash
python -m vla_trajectory collect \
  --oft-root /path/to/openvla-oft \
  --output /path/to/trajectory-data \
  -- \
  --pretrained_checkpoint /path/to/checkpoint \
  --task_suite_name libero_spatial \
  --num_trials_per_task 1 \
  --use_l1_regression True \
  --use_proprio True \
  --num_images_in_input 2 \
  --num_open_loop_steps 8
```

采集入口临时包裹 OFT 的评测函数，仍由原脚本处理 checkpoint、推理、动作队列、gripper 转换、初始化和任务循环；退出后恢复函数，不改写 OFT 源文件。当前支持官方具有 `run_episode/get_action/process_action/log_message` 的评测模块。你的适配版若改了接口，先对照 [接入与验收说明](docs/A100_ACCEPTANCE.md)。

`num_trials_per_task=1` 通常仍会运行任务套件的多个任务，不等于只运行一个 episode。先用已有的单任务配置试采；需要中途停止时 Ctrl+C 会保留已提交的步骤及 interrupted 状态。

## 数据目录

```text
trajectory-data/
  <episode-id>/
    manifest.json           # 元信息、状态、已提交步数和结果
    steps/
      000000.json           # 时间、奖励、终止标志、动作块、谓词证据
      000000.npz            # obs/*、next_obs/*、action、extra/*
    annotations.json        # 自动事件及经导入的人工标注
```

NPZ 禁止 pickle；每步先原子写入数组和记录，再提交 manifest 步数。记录期间异常退出的目录不能当作完整回合。工具不恢复未执行的动作，也不保证断电/磁盘故障下零数据丢失；manifest 范围外的孤立文件不会被读入。

原始相机方向保持 LIBERO 输出，查看时可能需要翻转；策略预处理后的图像另存在查询步骤的 `extra/policy_*` 中，避免混淆。RGB 和前后状态会占用磁盘，同步压缩/落盘也会增加评测耗时；先测 1 个 trial 的容量和时间，再扩大规模。每次模型查询的完整动作块仅存于首次执行步 `extra/query_actions`，该查询后未执行的动作不算 transition。若查询后执行失败，失败前的完整 transition 保留，未完成查询/动作不作为独立事件持久化。

`chunk_index` 指向完整模型输出中的行。官方 OFT 使用有容量上限的 deque：当 `num_open_loop_steps` 小于模型输出长度时，会丢弃块前部动作，因此首个执行索引可能不为 0；记录器保留这一原有行为。若你的适配版更改了动作队列规则，需要一并调整记录索引。

## 测试与发布

```bash
python -m pip install pytest
python -m pytest -q
```

CI 仅覆盖 CPU 工具链；真实 A100 验收见 [清单](docs/A100_ACCEPTANCE.md)。代码与数据分离，`data/outputs/`、权重、缓存和 `.env` 被 Git 忽略。

更新本仓库时，在自己的可写克隆中提交更改并推送至现有 remote。下面的发布脚本保留为首次创建新仓库时的工具；当前 `lianggeming14/openvla` 已存在，脚本会按设计拒绝重复创建。

首次建库时，具备 GitHub CLI、GitHub 登录及联网权限后，可在本库完成提交并运行：

```bash
git init -b main
git add .github .gitignore docs pyproject.toml README.md scripts tests vla_trajectory
git commit -m "Add OFT LIBERO trajectory recording and annotation"
python scripts/publish_github.py --owner lianggeming14
```

提交使用你已有的 Git 身份设置。若当前工作区的 `.git` 为只读，把 `dist/openvla-source.zip` 解压到普通可写目录再执行。

脚本核对当前登录账号，只创建新的私有 `lianggeming14/openvla`，拒绝覆盖已有同名库或已有 remote。它不会创建凭据、上传数据目录或替你修改现有仓库。

## 上游接口依据

- [OpenVLA-OFT 评测入口](https://github.com/moojink/openvla-oft/blob/main/experiments/robot/libero/run_libero_eval.py)：动作分块、预处理和评测函数。
- [LIBERO 环境封装](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/env_wrapper.py)：success、初始状态和模拟器状态接口。
- [LIBERO 谓词检查](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/problems/libero_tabletop_manipulation.py)：从目标表达式逐项求值。

接口在 2026-11-03 查阅。真实运行会记录本机 OFT commit；若上游或你的适配发生变化，以你实际代码为准。
