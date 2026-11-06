# A100 实验操作手册：Soup + Tomato → Basket

可下载[配套离线 HTML 手册](https://raw.githubusercontent.com/lianggeming14/openvla/main/docs/A100_RUNBOOK.html)，保存为 `.html` 后在本机浏览器打开；其中需要手动替换的内容显示为蓝色。命令以本文为准。

本手册从服务器上已经能运行的 OpenVLA-OFT 环境出发，先做 **5 个不同初态的试采**，全部检查通过后，再做 **20 个不同初态的正式采集**。第一轮只研究双物体任务 Soup + Tomato → Basket 的操作顺序，暂不扩展到其他任务。

已知 OFT 根目录为 `/HUBU-AI096/lgm/ICASSP/peft-in-openvla`。建议把轨迹工具单独放在 `/HUBU-AI096/lgm/ICASSP/openvla-trajectory-tools`，把实验数据放在 OFT 目录下的 `trajectory-data/libero10-order`。当前会话尚未连接这台服务器，以下命令已按本工具的接口核对，但真实模型加载、渲染、5 次试采和 20 初态评测仍需在服务器执行；CPU 测试或 synthetic 演示不能作为真实实验结果。

**阅读约定：**本 Markdown 是命令事实源；配套离线 HTML 将手动替换项显示为蓝色。🔵 形如 `REPLACE_WITH_OFT_ENV` 的内容必须先手动替换，不能带着占位符直接执行。给定的 OFT、工具库和数据根目录已经写成真实绝对路径，无需替换。以下服务器命令使用 **Bash**，不要直接粘贴到本机 Windows PowerShell。激活环境后启用 `set -euo pipefail`：任何命令或管道失败都会以非零状态退出当前 Bash，未设置变量也会报错；不会继续后续安装、采集或分析命令。重新 SSH 登录后，按第 1 节激活同一原 OFT 环境并 `source` 本轮 `run-env.sh`，修复错误后从失败步骤继续，**不要重新创建 `RUN_ROOT`**。首次尚未成功创建 `run-env.sh` 时，修复后重做首次初始化。

## 1. 登录服务器，激活原 OFT 环境并建立本轮目录

**目的：**让工具与原 OFT 使用同一个 Python、CUDA、LIBERO 和渲染配置，并把每一轮实验隔离保存。

**在哪执行：**通过自己已有的 SSH 连接登录 A100 服务器，在服务器 Bash 终端执行。先用 `conda env list` 找到原来能运行 OFT 的环境；不要另外猜一个环境名。

🔵 将下面的 `REPLACE_WITH_OFT_ENV` 替换为原 OFT conda 环境名，再执行整段。保留原环境中已有的 `MUJOCO_GL`、`PYOPENGL_PLATFORM` 和 CUDA 配置。

```bash
conda env list || exit 1
conda activate REPLACE_WITH_OFT_ENV || exit 1
set -euo pipefail

export OFT_ROOT=/HUBU-AI096/lgm/ICASSP/peft-in-openvla
export TOOLS_ROOT=/HUBU-AI096/lgm/ICASSP/openvla-trajectory-tools
export DATA_ROOT="$OFT_ROOT/trajectory-data/libero10-order"
export RUN_ROOT="$DATA_ROOT/run-$(date +%Y%m%d-%H%M%S)"

test -f "$OFT_ROOT/experiments/robot/libero/run_libero_eval.py" || exit 1
mkdir -p "$DATA_ROOT"
mkdir "$RUN_ROOT" || exit 1
mkdir -p "$RUN_ROOT"/{logs,plans,preflight,reviews,annotation-inbox,analysis}

{
  printf 'set -euo pipefail\n'
  printf 'export OFT_ROOT=%q\n' "$OFT_ROOT"
  printf 'export TOOLS_ROOT=%q\n' "$TOOLS_ROOT"
  printf 'export DATA_ROOT=%q\n' "$DATA_ROOT"
  printf 'export RUN_ROOT=%q\n' "$RUN_ROOT"
} > "$RUN_ROOT/run-env.sh"

printf '本轮目录：%s\n恢复变量文件：%s\n' "$RUN_ROOT" "$RUN_ROOT/run-env.sh"
which python
python --version
```

**检查：**记下终端打印的完整 `RUN_ROOT` 和 `run-env.sh` 路径。本轮的试采、正式采集、审阅页和分析都放在这里；同一轮后续步骤不要重新运行 `RUN_ROOT=...date...`。`mkdir` 若报告目录已存在，应建立新一轮目录，避免混入旧回合。

打开新 SSH 终端或失败后重新登录时，需要先重新激活**同一个原 OFT 环境**，再恢复变量；`source run-env.sh` 会恢复目录变量与 `set -euo pipefail` 严格模式，不会激活 conda。🔵 分别替换 `REPLACE_WITH_OFT_ENV` 和 `REPLACE_WITH_RUN_ENV_PATH`；后者是刚刚打印的完整 `run-env.sh` 路径。恢复后从失败步骤继续，不重建本轮目录。

```bash
conda activate REPLACE_WITH_OFT_ENV || exit 1
source REPLACE_WITH_RUN_ENV_PATH || exit 1
set -euo pipefail
printf '%s\n' "$OFT_ROOT" "$TOOLS_ROOT" "$RUN_ROOT"
```

**失败时：**若 OFT evaluator 文件不存在，先核对目录，不要继续安装或采集。若 `conda activate` 失败，在原来能够运行 OFT 的终端中继续，并确认 `which python` 指向原环境。新终端未 `source` 时，先恢复变量再运行后续命令。

## 2. 安装或更新独立轨迹工具库

**目的：**安装记录与分析工具，同时使用原环境中已经可用的模型依赖。

**在哪执行：**仍在上一步的服务器终端。以下两种情况只执行适合自己的一种。

第一次安装，确认建议工具目录不存在后：

```bash
if [ -e "$TOOLS_ROOT" ]; then
  echo "工具目录已经存在，请使用下面的更新分支。"
else
  git clone https://github.com/lianggeming14/openvla.git "$TOOLS_ROOT"
fi
```

如果工具库已经存在，先检查，再更新：

```bash
git -C "$TOOLS_ROOT" remote -v
git -C "$TOOLS_ROOT" status --short --branch
```

确认 remote 指向 `lianggeming14/openvla`，当前分支是 `main`，且没有自己未保存的改动，然后执行：

```bash
git -C "$TOOLS_ROOT" pull --ff-only origin main
```

**失败时：**网络失败就先修复服务器 GitHub 访问；目录不是此仓库、工作区有改动或不能快进时先保留现状并处理，不要用 `reset --hard`、强制覆盖或删目录来跳过检查。不要把工具仓库 clone 到原 OFT 仓库内部。

安装并记录环境：

```bash
cd "$TOOLS_ROOT" || exit 1
python -m pip install -e . --no-deps --no-build-isolation
python -m pip install 'matplotlib>=3.7' 'imageio>=2.31' 'imageio-ffmpeg>=0.4.9'
python -m pip check
python -m pip freeze > "$RUN_ROOT/logs/pip-freeze.txt"
git -C "$OFT_ROOT" rev-parse HEAD > "$RUN_ROOT/logs/oft-revision.txt"
git -C "$TOOLS_ROOT" rev-parse HEAD > "$RUN_ROOT/logs/tools-revision.txt"
python -c 'import sys, vla_trajectory; print(sys.executable); print(vla_trajectory.__file__)'
```

`--no-deps --no-build-isolation` 安装工具时复用现有环境；基础依赖 NumPy、Pillow 以及构建所需 setuptools、wheel 应已可用。第二条安装 Matplotlib（轨迹图）、imageio 和 imageio-ffmpeg（真实 MP4 编码），可能下载这些包及其依赖；没有要求升级 torch、CUDA、transformers、LIBERO。安装报缺包或 `pip check` 报冲突时先解决对应问题并保存日志，再继续，不能把依赖错误留到正式采集。

**检查：**Python 输出中的 `vla_trajectory.__file__` 应位于 `$TOOLS_ROOT/vla_trajectory/`。两个 revision 文件、`pip-freeze.txt` 应存在。先确认用户原 OFT 命令在同一环境仍能运行；原命令自身产生的日志或视频不等于本工具的逐步轨迹记录。

## 3. 无模型预检：GPU、接口、实际任务、初态和物体键

### 3.1 运行 doctor 并检查 OFT 源码接口

**目的：**在加载 7B 权重前发现基础环境或 OFT 接口不匹配。**在哪执行：**服务器同一终端，工具库目录。

```bash
set -euo pipefail
python -m vla_trajectory doctor --oft-root "$OFT_ROOT" 2>&1 | tee "$RUN_ROOT/logs/doctor.log"
```

**检查：**命令退出码应为 0，JSON 中 `basic_checks_passed` 为 `true`，GPU 型号/显存、Python 与依赖版本符合当前服务器。doctor 不加载模型，也不验证真实渲染或任务成功；失败先按报告补齐环境，不能直接进入采集。

下面用 AST 检查 evaluator 源码，避免为核对函数签名而导入模型模块：

```bash
python - <<'PY' 2>&1 | tee "$RUN_ROOT/logs/oft-interface.log"
import ast
import os
from pathlib import Path

path = Path(os.environ['OFT_ROOT']) / 'experiments/robot/libero/run_libero_eval.py'
tree = ast.parse(path.read_text(encoding='utf-8'))
functions = {node.name: node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
defined = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
for node in tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        defined.update(alias.asname or alias.name.split('.')[0] for alias in node.names)
for name in ('run_episode', 'run_task', 'get_action', 'process_action', 'log_message'):
    assert name in defined, f'missing evaluator callable/import: {name}'
required = {
    'run_episode': {'cfg', 'env', 'task_description'},
    'run_task': {'cfg', 'task_suite', 'task_id', 'total_episodes', 'total_successes'},
}
for name, expected in required.items():
    node = functions[name]
    actual = {arg.arg for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs}
    assert expected <= actual, f'{name}: missing {sorted(expected - actual)}'
    print(name, sorted(actual))
print('AST interface check passed; no evaluator module or model was loaded.')
PY
```

**检查与失败时：**日志应打印两个函数的参数及 `AST interface check passed`。这是静态接口检查；运行时工具还会再次检查。若自己的 OFT 分支改了这些接口，先对照该分支源码处理兼容性，不应直接修改模型动作语义或绕过检查。

### 3.2 生成两阶段计划，核对实际 LIBERO 状态库与观测

**目的：**确认本地 LIBERO 确有目标任务，找到真实 Task ID，验证计划选择的初态内容不同，并检查 EEF、夹爪与双物体传感器键。以下不加载模型，但会实际创建和关闭一个 LIBERO 离屏仿真环境，需要原服务器渲染配置可用。

```bash
python -m vla_trajectory plan --stage pilot --output "$RUN_ROOT/plans/pilot.json"
python -m vla_trajectory plan --stage formal --output "$RUN_ROOT/plans/formal.json"

cd "$OFT_ROOT" || exit 1
python - <<'PY' 2>&1 | tee "$RUN_ROOT/logs/libero-preflight.log"
import json
import os
from pathlib import Path
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from vla_trajectory.collection import resolve_plan

root = Path(os.environ['RUN_ROOT'])
suite = benchmark.get_benchmark_dict()['libero_10']()
print('Installed libero_10 tasks:')
for i in range(suite.n_tasks):
    print(i, suite.get_task(i).name)

resolved = {}
banks = {}
for stage in ('pilot', 'formal'):
    plan = json.loads((root / 'plans' / f'{stage}.json').read_text(encoding='utf-8'))
    document, bank = resolve_plan(plan, suite)
    resolved[stage], banks[stage] = document, bank
    target = root / 'preflight' / f'{stage}-resolved.json'
    target.write_text(json.dumps(document, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(stage, 'task_id=', document['task_id'], 'state bank size=', len(bank),
          'selected=', document['init_state_indices'], 'bank_sha256=', document['state_bank_sha256'])

document = resolved['pilot']
task = suite.get_task(document['task_id'])
bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
assert bddl.is_file(), bddl
print('BDDL:', bddl)
env = None
try:
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(np.asarray(banks['pilot'][document['init_state_indices'][0]]).copy())
    required = (
        'agentview_image', 'robot0_eye_in_hand_image',
        'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel',
        'alphabet_soup_1_pos', 'alphabet_soup_1_quat',
        'tomato_sauce_1_pos', 'tomato_sauce_1_quat',
    )
    report = {key: {'present': key in obs,
                    'shape': list(np.asarray(obs[key]).shape) if key in obs else None}
              for key in required}
    report['available_keys'] = sorted(obs)
    (root / 'preflight' / 'observation-keys.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    missing = [key for key in required if key not in obs]
    assert not missing, f'missing observation keys: {missing}'
finally:
    if env is not None:
        env.close()
print('Task/state/sensor preflight passed; model weights were not loaded.')
PY
```

**预期文件与检查：**`plans/pilot.json` 有 5 个索引（默认 0–4），`plans/formal.json` 有 20 个索引（默认 0–19）；`preflight/*-resolved.json` 包含真实 Task ID、每个状态的 SHA256、完整状态库指纹、计划行 ID 和对象对。`resolve_plan` 会拒绝任务名不唯一、初态索引越界或所选状态内容重复。`observation-keys.json` 应显示上述键存在，日志末尾应出现通过信息。

目标任务的准确名称为 `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`，对象键前缀为 `alphabet_soup_1`、`tomato_sauce_1`。本工具按准确任务名解析本地 Task ID，不手写一个猜测的编号。

**失败时：**LIBERO 导入、数据路径、BDDL、状态库加载或 EGL/OSMesa 渲染失败时，先修复原 LIBERO 环境并重做本节；不要只看 doctor 通过就继续。若观测键缺失，检查本地 LIBERO 的 object observables 配置和版本，保留此预检输出。不能从扁平 `sim_state` 猽物体位置，也不能把缺失传感器写成正常数值。本节只验证一个初态能创建观测，全部所选初态仍需真实采集验证。

仅在上面预检命令退出码为 0、日志末尾明确出现 `Task/state/sensor preflight passed` 后，才执行下一块回到工具库；预检失败时先停止，不用后续 `cd` 的退出码代替预检结果。

```bash
cd "$TOOLS_ROOT" || exit 1
```

## 4. 启用记录，采集 5 个不同初态

**目的：**检验真实 rollout、逐步记录、视频和人工复核整条链路。**在哪执行：**服务器同一终端，工具库目录。

**关键区别：原来的 `python ...run_libero_eval.py ...` 命令不会自动启用本工具。**本轮必须通过下面的 `vla_trajectory collect` 入口运行。它在同一进程内临时包装 OFT evaluator，退出后恢复原函数；无需手工修改原 OFT 源码。

```bash
cd "$TOOLS_ROOT" || exit 1
if [ -e "$RUN_ROOT/pilot" ]; then
  echo "pilot 已存在；请检查已有数据，重做实验应建立新 RUN_ROOT。"
else
  python -m vla_trajectory collect \
    --oft-root "$OFT_ROOT" \
    --output "$RUN_ROOT/pilot" \
    --plan "$RUN_ROOT/plans/pilot.json" \
    -- \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-10 \
    --task_suite_name libero_10 \
    --initial_states_path DEFAULT \
    --num_trials_per_task 1 \
    --load_in_4bit True \
    2>&1 | tee "$RUN_ROOT/logs/pilot-collect.log"
fi
```

`--oft-root`、`--output`、`--plan` 属于本工具，必须放在单独的 `--` 之前；`--` 之后是原样交给 OFT 的模型参数。保留用户四项参数：checkpoint=`moojink/openvla-7b-oft-finetuned-libero-10`、suite=`libero_10`、`num_trials_per_task=1`、`load_in_4bit=True`。计划模式还显式指定 `initial_states_path=DEFAULT`，使用已经核对并指纹化的官方状态库，不传入另一个自定义状态文件；其余沿用用户现有 OFT 默认设置。

虽然每任务 trial 数仍为 1，计划包装会只选择这个双物体任务，对 5 个初态分别调用原 `run_task` 一次；整个 collect 调用沿用原 eval 的一次模型初始化。这得到 5 个回合，不是重复跑 5 次初态 0，也不是把整个 `libero_10` 十个任务各跑一次。

**记录何时发生：**每次真实 `env.step(action)` 返回后，保存该步动作前 `obs`、实际动作 `action`、动作后 `next_obs`、reward/终止信息，以及可用的 EEF、夹爪、物体位姿、谓词和模拟器状态。模型原始动作、动作块和查询观测按实际执行关联保存。每步先写 NPZ/JSON，再原子更新 manifest 的已提交步数。回合结束的 `finally` 执行收尾（finalize）：更新 `manifest.json`，写 `execution.json`，计划模式再尝试导出完整 MP4 与帧映射。它不是等全部实验结束才一次性保存。

**预期目录：**

```text
RUN_ROOT/
  run-env.sh
  plans/pilot.json
  pilot/
    collection_plan_<hash>.json
    <episode-id>/
      manifest.json
      execution.json
      steps/000000.json
      steps/000000.npz
      task-*_init-*_episode-*.mp4
      task-*_init-*_episode-*.json
```

上述尖括号和星号只是目录结构说明，不能当命令参数使用。实际回合目录名是 episode ID，Task ID/初态在 manifest 和带任务/初态的 MP4 文件名中。**检查：**日志没有权重加载、量化、显存、仿真、录制或编码错误；应有 5 个真实 episode 和一份解析后的 `collection_plan_*.json`。官方失败回合仍是有效实验样本，不能只保留成功回合。

**失败时：**先保留日志与全部已生成数据，按第 9 节处理。命令退出码为 0 也需要下一节逐回合检查；若上游吞掉异常，manifest/execution 仍可能记录错误。重跑不会自动续传或去重，不要反复把同一计划写进同一输出目录来凑数量。

## 5. 逐回合检查，并导出不混淆的离线审阅页

### 5.1 列出回合、校验文件并检查执行计数

**目的：**区分“文件可读”“记录完整”“模型官方成功”三个不同结论。**在哪执行：**服务器同一终端。

先对试采的每个回合运行校验和初始标注生成，并导出独立 HTML：

```bash
python - <<'PY' 2>&1 | tee "$RUN_ROOT/logs/pilot-validate-review.log"
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ['RUN_ROOT'])
episodes = sorted(p.parent for p in (root / 'pilot').glob('*/manifest.json'))
assert len(episodes) == 5, f'expected 5 pilot episodes, found {len(episodes)}'
for episode in episodes:
    for command in ('validate', 'annotate'):
        subprocess.run([sys.executable, '-m', 'vla_trajectory', command, str(episode)], check=True)
    manifest = json.loads((episode / 'manifest.json').read_text(encoding='utf-8'))
    execution = json.loads((episode / 'execution.json').read_text(encoding='utf-8'))
    metadata = manifest['metadata']
    print('EPISODE', episode)
    print('task/state/row:', metadata.get('task_id'), metadata.get('init_state_id'), metadata.get('plan_row_id'))
    print('manifest:', manifest['status'], manifest['outcome'], 'steps=', manifest['num_steps'])
    print('execution:', json.dumps(execution, ensure_ascii=False))
    assert manifest['status'] == 'complete', 'episode did not finalize normally'
    assert execution['step_count_complete'] and execution['recording_complete'], 'incomplete step evidence'
    for phase in ('policy', 'settling'):
        assert execution[f'executed_{phase}_steps'] == execution[f'recorded_{phase}_steps'], phase
    assert execution['video_status'] == 'complete', 'video was not exported successfully'
    html = root / 'reviews' / f'pilot-{episode.name}-review.html'
    subprocess.run([sys.executable, '-m', 'vla_trajectory', 'review', str(episode), '--output', str(html)], check=True)
    print('DOWNLOAD HTML:', html)
print('Five pilot episodes passed file/count checks. Human video/annotation review is still required.')
PY
```

**检查：**`validate` 的 `valid=true` 只说明 manifest/已提交 steps/arrays 可读；仍要核对 `status=complete`、`step_count_complete=true`、`recording_complete=true`，policy 与 settling 的已执行/已记录计数分别相等。`manifest.num_steps` 包含两阶段；分析中的总控制步数只计 policy。还要对照解析计划核对 5 个 `plan_row_id` 和初态指纹没有遗漏或重复，并查看 `execution.observation_fields` 的目标键、维度和缺失标记。

Action Step 从 **1** 开始，只计实际 policy `env.step`，不是模型推理查询次数，也不是动作块数；settling 的 `action_step_id=null`，不计入总控制步数。原始 frame 从 **0** 开始，包含 settling，代表一次 transition；事件对齐该步动作后的 `next_obs`。不要把 MP4 帧序号直接当成 Action Step。

**失败时：**上面脚本会在首个不完整回合停止，日志显示路径。保留该回合，先查对应 `manifest.error`、`execution.video_error` 及采集日志，再按第 9 节处理。不要把字段改成 `true` 或把状态改成 `complete` 来通过检查。

### 5.2 将 HTML 下载到本机，离线逐帧审阅

**目的：**让操作者在本机浏览器完成可追溯的逐次抓放标注。**在哪执行：**文件下载在已有 SSH 客户端的文件面板、SFTP 客户端（例如 WinSCP/Xftp）中操作；打开与标注在本机浏览器中操作。

1. 在服务器文件面板进入日志打印的 `$RUN_ROOT/reviews` 的**实际绝对路径**，把全部 `pilot-<episode-id>-review.html` 下载到本机自己新建的本轮目录。HTML 内嵌图像和数据，可离线打开，无需把整份轨迹搬到本机。
2. 在本机双击其中一个 HTML，或拖入 Chrome/Edge。先核对页面顶部的 episode ID 和录制状态；再展开下方“Episode 元数据与录制状态”，在 metadata 中核对任务与初态和当前要标注的回合相符。
3. 同时把该回合的 MP4 下载并实际播放，确认首观测、末 `next_obs`、settling、画面方向及时间顺序正确。MP4 默认主视角旋转 180°；review 页面保持原始相机数组方向，两者方向表现可能不同。核对同名 sidecar 的帧映射，不凭文件存在就认定视频内容通过。
4. 使用“连续播放”“上一帧/下一帧”“Action Step 跳转”，并勾选“显示动作执行后的观测”确认事件。看不清时对照场景与腕部两路相机。
5. 对**每次尝试**点击“＋ 添加抓放尝试”，页面会自动生成独立 attempt ID，无需手填。对象框填写准确键名 `alphabet_soup_1`（Soup）或 `tomato_sauce_1`（Tomato）；填写尝试起止、抓取操作起点/事件、放置操作起点/事件、子任务结果和依据。所有抓放边界都可先定位画面，再点击对应字段旁的“设为当前帧”。第一次抓 Soup、再抓 Tomato、后来重抓 Soup，要保留三次尝试；抓空、滑落、失败重试不能删掉或合并成固定两个对象槽位。
6. 尝试表默认输入 1-based Action Step；区间标注使用 0-based 原始 frame。操作起点用于单独计算抓取/放置耗时，不能只填事件而把“抓到放的间隔”当成放置耗时。无法确定的对象、事件、边界或结果留空/未知，不补估计数值；`true/false/null` 分别表示人工确认成功/失败/未知。
7. 只把视觉或其他明确证据支持的事件标成抓/放。建议把抓取完成定在首次明确看到物体已被夹持并随夹爪抬起/移动的动作后观测，把释放完成定在首次明确看到物体脱离夹爪、留在目标位置的动作后观测，并在备注记录采用的口径。夹爪闭合或物体已在篮中的目标谓词不能单独证明中间发生抓取；画面遮挡或视频截止而无法确认时留未知。确认该回合全部尝试（包括确实无抓取）后，勾选页面的“已完成抓放复核（包括确认无抓取）”（JSON 字段 `object_reviewed`），填写人工结果与备注。
8. 点击“下载标注 JSON”，保留默认包含 episode ID 的文件名，避免不同回合文件混淆。每个 HTML 都要分别完成；刷新或关闭页面前先下载，页面内存修改不会自动保存。

**检查：**本机已有本轮 5 个审阅 HTML、对应可播放视频与各自下载的标注 JSON。页面完全离线，下载操作**不会写回服务器数据集**，也不替代下一节的上传与导入。若下载被浏览器拦截，先处理浏览器下载提示再继续；若打开了旧 HTML，应重新导出当前 revision 的审阅页。

### 5.3 上传 JSON 并导入对应回合

**目的：**把人工标注真正写入服务器上对应 episode 的 `annotations.json`。**在哪执行：**先用同一 SFTP 文件面板把本机下载的 JSON 上传到本轮 `$RUN_ROOT/annotation-inbox` 的实际目录，然后回到服务器 Bash。

🔵 将 `REPLACE_WITH_EPISODE_DIRECTORY` 替换为上节打印的某个完整回合目录；将 `REPLACE_WITH_UPLOADED_JSON_PATH` 替换为**已经上传到服务器**的该回合 JSON 绝对路径。不是本机的 Downloads 路径。每个回合分别替换和执行。

```bash
export EPISODE_DIR=REPLACE_WITH_EPISODE_DIRECTORY
export ANNOTATION_JSON=REPLACE_WITH_UPLOADED_JSON_PATH
test -f "$EPISODE_DIR/manifest.json" || exit 1
test -f "$ANNOTATION_JSON" || exit 1
python -m vla_trajectory import "$EPISODE_DIR" "$ANNOTATION_JSON"
python -m vla_trajectory summary "$EPISODE_DIR"
python -m vla_trajectory validate "$EPISODE_DIR"
```

**检查：**`import` 输出 `saved=true` 和新 revision；`summary` 与人工填写的尝试、对象和结果一致。官方结果继续保存在 manifest，人工结果单列，不覆盖官方成功/失败。

**失败时：**episode ID 不匹配、证据变化或 `base_revision` 冲突都会拒绝导入。不要手改 revision；用当前 episode 重新 `review` 导出 HTML，再审阅并下载。重复导入同一份旧 JSON 也会发生修订冲突。`recording` 状态回合可预览但不能保存标注，先确认采集进程已经结束并检查异常。

## 6. 试采分析与进入正式阶段的条件

**目的：**用五次真实试采检查容量、吞吐与整条证据链；试采统计单独保存。**在哪执行：**服务器终端。

```bash
python -m vla_trajectory analyze "$RUN_ROOT/pilot" \
  --output "$RUN_ROOT/analysis/pilot" \
  --objects alphabet_soup_1 tomato_sauce_1
du -sh "$RUN_ROOT/pilot"
df -h "$DATA_ROOT"
```

**检查：**`analysis/pilot` 中应有第 8 节列出的三张表、PNG/SVG 和报告；五回合标注均已导入，轨迹/视频/事件可对齐，物体与 EEF 字段无未解决缺失，采集速度与剩余空间可以承受正式阶段。记录试采耗时、数据大小、可播放性和发现的问题。试采允许官方任务失败；进入正式阶段要求记录链路可用，而不是要求五次任务全成功。

**失败时：**若记录、渲染、传感器、编码或审阅链路有问题，修复后用新的独立 `RUN_ROOT` 重做试采，并保留原试采作为故障证据。不要在五次链路未核验时直接开始 20 次正式采集。

## 7. 采集 20 个不同初态，并逐回合重复复核

**目的：**在同一目标任务上取得正式样本。**在哪执行：**通过试采验收的服务器终端；新终端先按第 1 节激活环境并 `source` 本轮 `run-env.sh`。

```bash
cd "$TOOLS_ROOT" || exit 1
if [ -e "$RUN_ROOT/formal" ]; then
  echo "formal 已存在；先检查已有数据，独立重跑应建立新 RUN_ROOT。"
else
  python -m vla_trajectory collect \
    --oft-root "$OFT_ROOT" \
    --output "$RUN_ROOT/formal" \
    --plan "$RUN_ROOT/plans/formal.json" \
    -- \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-10 \
    --task_suite_name libero_10 \
    --initial_states_path DEFAULT \
    --num_trials_per_task 1 \
    --load_in_4bit True \
    2>&1 | tee "$RUN_ROOT/logs/formal-collect.log"
fi
```

默认正式计划使用初态索引 0–19。pilot/formal 可复用其中部分初态索引，但正式阶段自身必须含至少 20 个内容互异初态，不能拿试采五次补足正式数量。状态变化是在同一已知任务上的初态评测，**不能据此声称对未见任务的泛化**。

接着对正式回合做同样的校验、执行完整性检查与独立 HTML 导出：

```bash
python - <<'PY' 2>&1 | tee "$RUN_ROOT/logs/formal-validate-review.log"
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ['RUN_ROOT'])
episodes = sorted(p.parent for p in (root / 'formal').glob('*/manifest.json'))
assert len(episodes) == 20, f'expected 20 formal episodes, found {len(episodes)}'
seen = set()
for episode in episodes:
    for command in ('validate', 'annotate'):
        subprocess.run([sys.executable, '-m', 'vla_trajectory', command, str(episode)], check=True)
    manifest = json.loads((episode / 'manifest.json').read_text(encoding='utf-8'))
    execution = json.loads((episode / 'execution.json').read_text(encoding='utf-8'))
    metadata = manifest['metadata']
    print('EPISODE', episode, 'state=', metadata.get('init_state_id'), 'row=', metadata.get('plan_row_id'))
    print('manifest:', manifest['status'], manifest['outcome'])
    print('execution:', json.dumps(execution, ensure_ascii=False))
    assert manifest['status'] == 'complete'
    assert execution['step_count_complete'] and execution['recording_complete']
    for phase in ('policy', 'settling'):
        assert execution[f'executed_{phase}_steps'] == execution[f'recorded_{phase}_steps']
    assert execution['video_status'] == 'complete'
    identity = metadata['state_sha256']
    assert identity not in seen, 'duplicate formal initial-state content'
    seen.add(identity)
    html = root / 'reviews' / f'formal-{episode.name}-review.html'
    subprocess.run([sys.executable, '-m', 'vla_trajectory', 'review', str(episode), '--output', str(html)], check=True)
    print('DOWNLOAD HTML:', html)
print('Twenty formal episodes passed file/count checks. Human review/import is still required.')
PY
```

**检查与后续：**逐项对照正式 `collection_plan_*.json` 的 20 行，核对 Task ID、初态索引/指纹与行 ID。然后对每个正式回合完整重复第 5.2–5.3 节：下载 HTML/视频到本机、逐次尝试标注、下载 JSON、上传服务器、`import`、`summary`。不能只抽看成功回合；失败、未知和异常必须保留。若需要补采，先保留失败记录和补采原因，明确新的采集批次，最终报告列明实际回合数、互异状态数和纳入规则，不把额外重复回合悄悄并入这 20 次。

## 8. 导出正式三表、轨迹图和报告

**目的：**产生可核对的正式统计与分析材料。**在哪执行：**全部正式人工 JSON 已成功导入后，在服务器终端执行。

```bash
python -m vla_trajectory analyze "$RUN_ROOT/formal" \
  --output "$RUN_ROOT/analysis/formal" \
  --objects alphabet_soup_1 tomato_sauce_1
ls -lh "$RUN_ROOT/analysis/formal"
ls -lh "$RUN_ROOT/analysis/formal/plots"
```

**预期文件：**

| 文件 | 需要核对的内容 |
|---|---|
| `episode_statistics.csv` | 任务、初态、回合、完整抓取序列、官方结果、policy 总步数、EEF 长度及缺失/覆盖情况 |
| `subtask_events.csv` | 每次尝试一行，保留重抓/失败/未知，抓放事件、独立操作起点与耗时、结果和依据 |
| `task_summary.csv` | 正式阶段样本数、不同初态数、成功率分母、对象对顺序比例、平均成功步数 |
| `plots/*_eef.png`、`plots/*_eef.svg` | 每回合等比例 XYZ 路径、坐标随 Action Step 变化、人工抓/放事件标记；缺口不跨接 |
| `analysis_report.md` | 中文实验范围、顺序观察、成功率/步数、未知与缺失、视频覆盖和局限 |

**统计解释与检查：**

- 官方成功率只用已知官方 `success/failure` 作分母，`unknown` 另列；平均成功步数只纳入官方成功且执行计数可信回合。人工成功不能覆盖官方结果。
- A→B/B→A 比例只纳入已经人工完成复核、可比较的样本，明确 A/B 是 Soup/Tomato，列出分母、未知和并列数。完整序列保留 A→B→A 等重抓，不能强行归成两个事件。
- 总步数排除 settling；环境步执行不确定时留空，已确认执行数只作下界。`validate` 可读不等于轨迹完整。
- 抓取/放置操作步数分别从人工起点到对应事件，包含两端；缺边界留空。抓到放的事件间隔另列，不当作纯放置操作耗时。
- EEF 长度逐个 policy transition 计算 `norm(next_xyz-obs_xyz)`，包括第一个 policy 动作；任一应计坐标缺失则完整长度未知，报告已观测长度/覆盖率。人工抓放标记对齐 `next_obs`，图不跨缺口连线。
- 检查报告里的实际回合数、不同状态数、计划行与各表一致；pilot 和 formal 分别分析，失败/未知/异常未被静默删掉。三表、每回合图和报告应与原始证据互相可追溯。

**失败时：**缺 Matplotlib、坐标缺失、人工尝试未复核或视频覆盖不全时，保留报告中的缺失并修复证据链，随后重新分析。`--no-plots` 可以排查表格问题，但不能代替要求中的正式轨迹图。中文标签若显示方框，应在服务器配置可用 CJK 字体后重导；默认英文物体 ID 不受此影响。

真实报告应回答：Soup/Tomato 顺序是否偏向某个方向，不同初态下顺序和轨迹如何变化，并提供样本量、可比较分母、失败/未知及局限。未出现的重抓、并列等情况写“未观察到”，不要制造例子。顺序观察不能直接证明模型主动规划、布局导致顺序改变或路径最优；本轮也不提供未见任务泛化结论。

## 9. 异常、中断和旧数据的处理

**目的：**保持真实证据，避免把不完整运行误报为完成。**在哪执行：**服务器检查日志与原始文件；任何故障处理都先确认对应写入进程已退出。

| 情况 | 如何继续 |
|---|---|
| 录制写入异常、磁盘满、权限不足 | 保留采集日志、manifest、execution 和已提交 steps；修复空间/权限后在新批次补采，不能声称缺失控制步已记录。同步 NPZ/图像会增加耗时，按试采容量估算。 |
| 没有 FFmpeg/codec 或 MP4 导出失败 | 查看 `execution.video_error`，检查 `python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'`。修复编码环境后，可对已有完整 RGB 补导视频；原 execution 的视频错误仍保留为运行证据。 |
| Ctrl+C | 已提交 steps 通常可读，收尾应记录 `interrupted`；如果中断发生在未返回的环境步，`step_count_complete=false`。它不是完整正式样本。要检查中断恢复行为，应另开独立实验目录做专门验收，不在正式 20 次中故意中断。 |
| SIGKILL、断电、进程崩溃 | `finally` 可能未运行，manifest 可停在 `recording`，execution 也可能缺失。先备份并记录人工处置、原状态和实际已提交范围；不把状态改成 `complete`，结果保持未知。 |
| 有旧 rollout 日志或只有旧视频 | 不能事后补出精确的逐步 `obs/action/next_obs`、原始动作块和未保存的物体/EEF 轨迹。用本工具重新运行真实采集，旧资料仅作辅助证据。 |
| 旧标注 JSON 导入冲突 | 重新导出当前 revision 的 HTML，重新核对再下载/导入；不手改 `base_revision`。 |

补导视频的单回合命令如下。🔵 先把 `REPLACE_WITH_EPISODE_DIRECTORY` 替换为实际回合目录：

```bash
export EPISODE_DIR=REPLACE_WITH_EPISODE_DIRECTORY
python -m vla_trajectory video "$EPISODE_DIR"
```

独立 `video` 命令不覆盖已有 MP4/sidecar；若已有失败的输出占用默认名称，可指定一个新名称并保留原证据：

```bash
python -m vla_trajectory video "$EPISODE_DIR" --output "$EPISODE_DIR/recovered-review.mp4"
```

补导后实际播放、检查 sidecar 并重新分析；不要改写原 `execution.video_status` 来掩盖运行失败。视频完整不能补救缺失控制步骤。默认完整视频是首个 `obs` 加每一步 `next_obs`，含 settling 和最终观测；播放 fps 与实际 control frequency 分别记录。

## 10. 收尾交付与完成边界

**目的：**保存可复现证据并界定什么可以宣布完成。**在哪执行：**服务器保存原始实验目录，本机审阅正式报告；代码与正式文档按授权交付仓库。

基础实验完成前应确认：5 次试采链路已核验；同一双物体任务的正式阶段有 20 个内容互异初态；每回合逐步数据、执行完整性、可播放视频/帧映射、人工尝试与导入修订均核对；三表、每回合 PNG/SVG、报告及环境/revision/计划证据齐备。单有软件测试通过、模型进程跑完或几张视频截图，不能宣布此项真实业务验收完成。

把本轮 `RUN_ROOT` 作为实验归档，包含日志、计划、状态指纹、原始数据、视频、人工 JSON 和分析结果；保留故障与补采说明。模型权重、原始轨迹/视频、缓存、凭据与私有原要求不上传公开 GitHub。代码、测试、脚本与正式文档交付 [lianggeming14/openvla](https://github.com/lianggeming14/openvla)；正式实验报告公开前应检查其中不含原始敏感数据。飞书仍需可用连接、认证和明确接收位置，当前不能声称已发送。

接口与任务参考：本地实际安装版本和本轮预检输出为准，可对照 [LIBERO 官方任务清单](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/benchmark/libero_suite_task_map.py)、[LIBERO 官方 BDDL 目录](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/master/libero/libero/bddl_files)、[OpenVLA-OFT 官方说明](https://github.com/moojink/openvla-oft)。要求逐项证据见 [要求与验收矩阵](REQUIREMENTS_TRACEABILITY.md)。
