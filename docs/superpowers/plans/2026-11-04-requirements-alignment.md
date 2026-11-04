# LIBERO-Long 原文要求对齐 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将已发布采集工具补齐至原文要求的抓放标注、固定初态实验、视频、三表、轨迹图和报告链路。

**Architecture:** 沿用不可变 transition 与可逆 OFT 包装。标注、批量/视频、分析三组使用独立文件和固定接口，控制器负责统一集成和发布；原始私有正文不发布。

**Tech Stack:** Python 3.10+、NumPy、Pillow、stdlib CSV；可选 Matplotlib 和视频编码器，不升级 torch/CUDA/OFT 依赖。

**Spec:** [完整设计与数据接口](../specs/2026-11-04-libero-long-requirements-design.md)，绑定私有原文 blob `ff1ac6625ee3ff3917cdddcbe9ded101881c1a62`。

## Global Constraints

- 保持模型 `moojink/openvla-7b-oft-finetuned-libero-10`、`libero_10`、`num_trials_per_task=1`、`load_in_4bit=True`；不训练或修改模型。
- record frame 从 0 开始并包括 settling；Action Step 从 1 开始且只计 policy env.step，不计推理次数。
- 官方成功来自 manifest；人工抓放未知保留 null；重复尝试不去重，自动谓词不冒充抓取。
- pilot 五个不同初态；formal 至少二十个不同初态；实际 A100 评测和代码验证分开。
- 私有正文、凭据、原始轨迹、模型权重不提交公开库；保留用户公开仓库设置。

## Review Focus

- 初始化运动很大、首个 policy 动作或最后一步成功：计数、路径和视频终态保持正确。
- 抓取失败/只看见放置/重抓/同一步并列/旧标注导入：不臆造事件或清空新字段。
- 初态索引不同但状态内容相同、越界、配置冲突：明确失败，不能得到表面上的二十回合。
- 缺 EEF 坐标/落盘失败/unknown outcome/synthetic/重复 episode ID：指标、覆盖和分母诚实。
- UTF-8、视频编码失败、只读 Git 元数据：原始数据仍安全，公开发布逐文件核验 SHA。

---

### Task 1: 结构化人工抓放标注与离线复核

**Files:** `vla_trajectory/annotations.py`、`vla_trajectory/review.py`、`tests/test_annotations.py`、`tests/test_review.py`。

**Interfaces:** 产出设计中的 annotation v2 `object_attempts`、`object_reviewed`；保持 `load_annotations/save_annotations/export_review` 签名，旧文档读入及 import 兼容。尝试 frame 均为原始 transition 索引，UI 显示/输入 1-based Action Step 并映射。

- [ ] 先写失败测试：重复对象三次尝试、抓失败/只放置/未知步、旧 v1 不清空 v2、新版空数组清空、settling 拒绝、自动证据与修订保护。
- [ ] 实现 v2 验证和迁移，扩展原子保存/导入流程。
- [ ] 先写 HTML/实际 JS 下载回传测试，再实现结构化编辑、复核完成标记、动作步映射和连续播放。
- [ ] 运行标注与 review 定向测试，报告失败在先及通过输出，独立审查后进入集成。

### Task 2: 单任务 5/20 固定初态与完整视频

**Files:** 新建 `vla_trajectory/collection.py`、`vla_trajectory/video.py`、`tests/test_collection.py`、`tests/test_video.py`；修改 `vla_trajectory/oft.py`、`tests/test_oft.py`。

**Interfaces:** 产出设计中的 `create_plan/validate_plan`，`instrument(..., collection_plan=None)` 和 `export_episode_video`。每回合身份字段及 `execution.json` 供 Task 3 消费；默认目标任务和对象使用设计中的 exact name。

- [ ] 先写失败测试：目标任务筛选、五/二十个指定初态各一次、模型只加载一次、重复内容/ID、越界和配置冲突。
- [ ] 实现可逆 run_task 包装及解析后的计划、元信息、1-based action_step_id、实际/已记录 env.step 计数；旧无计划调用保持兼容。
- [ ] 先写视频终态/初始化/帧映射/命名/编码失败测试，再实现流式 MP4 导出与原子 sidecar；有真实编码器时做实际编码解码测试。
- [ ] 运行 collection/video/OFT 定向测试，记录编码器可用性与剩余实机边界，独立审查后进入集成。

### Task 3: 三张 CSV、XYZ 图与行为分析报告

**Files:** 新建 `vla_trajectory/analysis.py`、必要时 `vla_trajectory/plots.py`、`tests/test_analysis.py`。

**Interfaces:** 消费 Task 1 标注、Task 2 身份与 execution.json，产出设计中的 `analyze_episode/export_analysis` 以及三张 CSV、图、报告。无 execution.json 的完整旧回合可按 phase 计数；错误回合无法证明总控制步时留空，并另列已记录步数。

- [ ] 先写失败测试：排除大幅 settling 运动、包含首个 policy 动作、推理块多步、缺坐标断线、人工结果不改官方成功。
- [ ] 实现指标与尝试行，保留重复、未知和并列；明示成功率/顺序分母，按任务与阶段分组。
- [ ] 先写 CSV/report/plot 输出测试，再实现中文 CSV、科学图和分析报告；默认拒绝将 synthetic 作为真实实验，检测重复 episode ID。
- [ ] 运行分析定向测试并生成明确标记的 synthetic 样例，独立审查统计定义和输出图。

### Task 4: CLI、说明与交付核验

**Files:** `vla_trajectory/cli.py`、`vla_trajectory/__init__.py`、`pyproject.toml`、`.github/workflows/tests.yml`、`README.md`、`docs/A100_ACCEPTANCE.md`、`docs/IMPLEMENTATION_STATUS.md`、`docs/REQUIREMENTS_TRACEABILITY.md`、`tests/test_cli.py`、新计划模板及发布包。

**Interfaces:** CLI 新增 `plan`、`video`、`analyze`，collect 增加 `--plan`；模型参数仍全部置于 `--` 后原样转发。`plan --stage pilot|formal --output ...` 默认产生目标任务的 5/20 清单；`--state-ids` 可显式指定；`analyze --objects A B --include-synthetic --no-plots` 对应分析接口。

- [ ] 先写实际 CLI 失败测试：计划生成/collect 参数转发/分析 CSV 与标注导入/视频入口，再接入新模块及可选依赖。
- [ ] 更新 README 与 A100 清单：5 次试采、20 初态正式采集、逐回合人工复核、三表/图/报告；移除“原文未读取”的当前状态描述。
- [ ] 同步发布版本 `0.2.0`、包内版本和下载链接，保留旧 wheel 作为历史文件。
- [ ] 写原文逐项验收矩阵，区分代码完成、CPU 验证、A100 待执行和飞书待连接。
- [ ] 全量测试、synthetic 端到端、视频/图的实际验证及独立最终审查；修复重要发现。
- [ ] CI 安装可选 analysis/video 依赖并执行真实 MP4 编码/解码测试；本地无编码器时明确跳过该硬件/依赖测试，不冒称本地已验证编码。
- [ ] 重新打包并通过 GitHub 连接器更新 openvla，保留现有其他文件，逐个远程 SHA 核验并检查 CI；私有正文不上传。

## 执行记录

本地 `.git` 只读且没有可写提交历史，使用 `.work/requirements-alignment/baseline` 和定向 unified diff 作为审查基线；完成后用已授权 GitHub 连接器提交。用户已要求直接执行和上传，按该授权持续推进。
