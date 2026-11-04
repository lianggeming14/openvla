# 交付与验证记录 · 2026-11-04

## 本地交付

- Python 包：vla_trajectory；逐步原子保存、OFT 运行时接入、LIBERO 目标谓词自动事件、人工复核导出/导入、顺序汇总。
- 离线演示：outputs/demo-review.html（由 synthetic demo 数据生成，不提交数据至 GitHub）。
- 安装包：dist/openvla_trajectory_tools-0.1.0-py3-none-any.whl。
- 源码包：dist/openvla-source.zip（仅从明确的源码、测试和文档目录生成，不含权重/数据/凭据）。
- README 和 docs/A100_ACCEPTANCE.md 提供现有 conda 环境的安装与验收命令。

## 实际执行的验证

- `python -m pytest -p no:cacheprovider`：2026-11-04 复验 134 passed，退出码 0；CPU 环境。包含 recorder 65、annotations 39、review 17、OFT adapter 10、CLI 3。
- `pip wheel --no-deps --no-build-isolation`：构建纯 Python wheel。
- 演示命令 demo → validate → annotate → review → summary 通过；24 步 synthetic 数据，标注 source=synthetic_demo。
- CLI 回归测试覆盖人工 JSON 导入、修订号增加和旧版本导入冲突。
- 用户提供的 LIBERO-10 + 4-bit 启动参数已适配；新增独立子进程测试证明 OFT 模块导入时和 eval 调用时均收到原样参数（不包含记录器的传输选项）。修复了此前导入完成后才切换 argv 的顺序问题；未运行真实 GPU 模型。
- HTML 的数据/图片导出、脚本语法与事件处理通过 Python 和 Node 最小 DOM 测试；未做真实浏览器视觉 QA（浏览器自动化受策略限制）。

## 独立审查及修复

两位独立审查者发现并复现四项问题，均增加了失败在先的回归测试后修复，并完成限定范围复审：

1. manifest 原子替换后立即中断导致内存步数过时：写入/结束/读 manifest 前以磁盘提交状态为准。
2. recording 状态保存标注后新增步骤导致标注失效：只允许终态保存，预览保留，证据快照变化时拒绝导出/保存。
3. OFT 短 deque 丢弃块前部动作但索引仍从零开始：记录实际完整模型输出中的源索引。
4. 首步失败丢失 supplied initial_state：在执行前写入 manifest 元信息。

## 实现选择与限制

依据用户确认的 OpenVLA-OFT + LIBERO 环境开发。尚未读取私有《VLA 模型执行轨迹采集与任务顺序分析》正文，因此未验证其完整字段/标签要求；该方案到手后可能需要调整。

自动标签采用目标谓词变化，人工中间阶段使用自由文本片段；不推断隐藏意图、视觉抓取成功或因果顺序。采用离线 HTML + JSON 导入以适应远程 A100，不依赖服务器。同步 NPZ 压缩/落盘会增加评测耗时，需实机量测吞吐与容量。

现有 OFT/LIBERO 依赖不自动升级；上游原评测入口自身对 checkpoint 配置/建模文件的更新行为保持原样。用户于 2026-11-04 创建了公开的 `lianggeming14/openvla` 仓库，本次交付保留其可见性；上传范围为本工具源码、测试、脚本、正式文档及安装/源码包。

## GitHub 交付

- 仓库：[lianggeming14/openvla](https://github.com/lianggeming14/openvla)，分支 `main`。
- 通过已连接的 GitHub 账号提交源文件，保留项目目录结构；安装包与源码包位于 `downloads/`。
- 目标账号的 GitHub App 安装授权已补齐，并已验证实际 blob 写入；此前的 403 权限问题已解决。
- 本地 `.git` 元数据仍为只读，远程提交使用 GitHub 连接器完成；其他机器可正常克隆此仓库。

## 尚未完成

- A100 上真实权重加载、MuJoCo 渲染和小规模 rollout；用户完整启动命令已经收到，当前仍没有服务器连接。
- 原私有采集方案逐项核对。
- 飞书文档同步：当前会话没有可用的飞书发送工具或认证，尚未发送。
