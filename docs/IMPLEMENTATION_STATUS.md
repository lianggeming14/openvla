# 交付与验证记录 · 2026-11-04

## 当前版本与来源

本轮代码版本 **0.2.0**，已按私有原方案完整核对并实现补齐。原文共七节、171 行、5645 字节，Git blob `ff1ac6625ee3ff3917cdddcbe9ded101881c1a62`，本地私有镜像哈希一致；原文不提交公开库或源码包。公开 [验收矩阵](REQUIREMENTS_TRACEABILITY.md) 仅说明要求、实现和证据，不复制私有正文。

当前新增：v2 结构化人工抓放尝试与离线播放/动作步定位；五次试采/二十初态计划；OFT 单任务固定初态采集、实际控制步计数和传感器字段/缺失报告；含终态的视频及帧映射；三张 CSV、PNG/SVG 和中文报告；CLI `plan/video/analyze` 与 `collect --plan`。

用户参数保留为 `moojink/openvla-7b-oft-finetuned-libero-10`、`libero_10`、`num_trials_per_task=1`、`load_in_4bit=True`。工具不训练或修改模型，仍复用官方推理与任务循环。

## 已取得的 CPU 证据

|范围|实际结果|说明|
|---|---|---|
|此前 0.1.0 全量复验|134 passed，退出码 0|首次发布基线，不代表本轮所有新增功能验收|
|v2 标注与 review 定向测试|首次合并 77 passed；最终路径保护修复后 review 单组 30 passed，退出码 0|真实 JSON 往返、旧版迁移、修订保护、实际 JS/Blob 与动作步映射，执行证据/视频/sidecar 防覆盖|
|计划/OFT/视频定向测试|44 passed、1 skipped，退出码 0|固定状态、模型只加载一次、执行/记录计数、帧序/终帧/错误路径；真实编码项跳过|
|分析与绘图定向测试|27 passed，退出码 0|三表、分母/未知、重试、缺口、PNG/SVG、视频帧映射核验|
|CLI 定向测试|9 passed，退出码 0|新入口、真实分析 CSV/人工导入、子进程参数转发、计划驱动状态选择|
|最终完整集成测试|230 passed、1 skipped，13.11s，退出码 0|唯一 skip 是本机缺编码器的真实 MP4 编码/解码；其他新增与原有测试全部通过|

分组定向结果不相加计算全量；完整集成测试单独执行并取得上表结果。所有本地测试均在 CPU 开发机，未执行真实 A100 模型。独立审查及定向复审已通过，修复了清理异常覆盖主异常、环境释放和复核页覆盖执行证据的问题。测试中曾出现 Windows 文件占用导致的 recorder 原子替换失败，未改 recorder 隐藏问题，未更改代码的重跑通过，执行报告保留该记录。

实际 synthetic 审查样例生成 3 个演示回合、三表及 6 个 PNG/SVG，图已打开检查；仅验证工具行为，不作模型结果。自动视频边界测试的编码输出是明确不可播放的占位内容，不能作为真实 MP4 编码或真实视频交付证据。

CLI 另实际执行 `demo → analyze --include-synthetic`，返回 1 回合、三表/报告路径和 2 个 PNG/SVG 路径，嵌套 Path 已正确序列化为 JSON；没有提供人工抓放时事件表为 0 行，不自动补造事件。两个默认计划模板已通过实际 plan 命令生成并重新校验；正式文档相对链接检查通过。

最终 wheel 已构建并以 `--no-index --no-deps` 安装到独立工作目录；13 个包内 Python 文件与源文件逐字节一致。实际从该 wheel 导入 0.2.0，运行 `plan`（5/20）及 `demo → validate → annotate → review → import → analyze`：得到一条 synthetic 回合、一条人工尝试，24 个动作步，抓取/放置动作步 3/6，三表、报告及 PNG/SVG。该安装验证同样不是模型评测。

## 视频编码与 CI

本地有 Matplotlib、imageio，没有 imageio_ffmpeg 或 PATH 中的 FFmpeg。真实 MP4 编码/解码测试明确 skip，本地未完成实际编码。帧序、终态、旋转、命名/sidecar、编码失败和原子清理以外部编码边界替身验证。

CI 安装 `.[test,analysis,video]` 并运行全量测试，包含真实 MP4 编码/解码；每次提交的实际运行状态见 [GitHub Actions](https://github.com/lianggeming14/openvla/actions)，不能把本地 skip 算作编码通过。分析的视频覆盖检查校验非空 MP4、匹配 sidecar 与逐帧映射，不解码媒体；A100 验收仍需实际播放每条视频。

## GitHub 与发行文件

仓库：[lianggeming14/openvla](https://github.com/lianggeming14/openvla)，保留用户已创建的公开设置。

- **首次 0.1.0 发布已读回核验**：main commit `3be25265c3c9a09ea05b24c06a50d90169b9c74d`，26 个远程文件 Git blob SHA 与本地逐项一致，包含 24 个源码/测试/脚本/文档及 downloads 下两个发行包。
- **本轮 0.2.0 交付范围**：36 个源码/测试/脚本/正式文档与计划模板文件，以及 `downloads/` 中的源码 ZIP 和新 wheel；新增实现不属于上述旧 commit，当前版本以 [main 提交历史](https://github.com/lianggeming14/openvla/commits/main) 为准。
- 新 wheel 为 `openvla_trajectory_tools-0.2.0-py3-none-any.whl`；旧 `0.1.0` wheel 保留为历史文件。源码 ZIP 为 `openvla-source.zip`，保留 `openvla/` 包装目录，包含新模块和 `plans/`。
- 本地 `.git` 元数据只读；远程提交通过已授权的 GitHub 连接器完成。发布流程逐项核对远程 Git blob SHA 与本地内容，分支提交和 CI 结果分别读回确认。

仅上传代码、测试、脚本、正式文档和明确发行包，排除私有原文、权重、原始轨迹、缓存、凭据和整个工作区。

## 仍待实机与渠道验收

真实 A100 的权重加载、渲染、五次试采、二十个不同初态正式评测、逐回合视频/人工复核、真实三表/图/报告和行为结论均未执行。当前缺 SSH 主机与 OFT/数据目录连接；操作见 [A100 清单](A100_ACCEPTANCE.md)。没有实验数据时不填造成功率或顺序偏好。

飞书文档发送仍缺可用工具/认证及明确接收位置，尚未发送。后续 GitHub 要求没有撤销飞书交付；连接恢复后继续。本文不代表已部署后台同步或定时服务。
