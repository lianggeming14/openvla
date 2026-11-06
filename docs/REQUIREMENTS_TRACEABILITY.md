# 原方案要求与验收矩阵 · 0.2.0

2026-11-04 已完整读取私有原方案七节、171 行、5645 字节，本地镜像 Git blob 为 `ff1ac6625ee3ff3917cdddcbe9ded101881c1a62`。本文仅按项说明实现及验收状态，不复制私有正文。原文仅本地私有保存，不进入公开提交或发行包。设计见 [原文对齐设计](superpowers/specs/2026-11-04-libero-long-requirements-design.md)。

“代码实现/CPU 验证”不表示 A100 验收通过；synthetic 与编码边界替身也不算真实模型或视频证据。当前全部真实模型实验仍待执行。

|来源范围/要求摘要|实现与正式交付|已取得的验证|A100/最终验收|
|---|---|---|---|
|一：分析真实模型的顺序与轨迹|复用 OFT 官方评测，不训练/修改模型；人工抓放+统计|OFT/analysis/CLI 定向测试|真实观察与结论待数据|
|一：抓取/放置消耗的控制步骤|六个原始 frame 边界，映射 1-based policy 动作步；操作数/尝试数/事件间隔分列|annotations/review/analysis 的边界、settling、未知测试|逐回合对齐真实视频待核|
|一：不同初始布局下的变化|固定状态索引与内容指纹，分回合记录/报告，不推断主动规划|collection 状态范围/重复内容/指纹保护测试|真实二十初态比较待执行|
|二：官方环境/接口与状态知识|保留现有 OFT/LIBERO；doctor、README、A100 清单；明确传感器字段映射|模块签名核对、CLI 参数转发、observation_fields 测试|用户环境/版本/实际字段待核|
|三 Step 1：指定官方模型与任务套件|moojink/openvla-7b-oft-finetuned-libero-10、libero_10，用户 trials=1、4bit=True 不变|独立子进程检验导入/评测参数原样到达|权重加载/渲染/运行待 A100|
|三 Step 1：完整任务视频|video 流式首 obs+所有 next_obs，终帧/settling、命名、sidecar|帧序/终态/旋转/异常 CPU 测试；带编码器 CI 执行真实编码/解码|本地编码 skip；真实每条 A100 MP4 待播放|
|三 Step 2：实际动作与 EEF/夹爪/物体位姿|recorder NPZ/JSON、OFT obs/next_obs/raw/executed action；字段可用性明确|recorder/OFT 保存、初态、错误路径测试|真实传感器覆盖/单位/姿态待核|
|三 Step 2：结果与总 Action Steps|官方 manifest.outcome；execution.json 执行/记录分开，排除 settling/推理次数|成功/失败/异常、落盘后失败、环境步不确定测试|真实计数/完整性逐回合核对|
|三 Step 3：首抓/次抓、抓放步骤|v2 object_attempts、object_reviewed、离线逐帧与动作步输入|真实 JS/下载/import 往返，未知/settling 映射测试|人工逐回合复核待执行|
|三 Step 3：重抓、重试与顺序改变|每次尝试独立 ID，重复对象不去重；失败/null 保留|v2 schema、顺序 A→B→A、并列/未知测试|真实尝试证据待采集|
|三 Step 4：五次试采与二十固定初态|plan pilot=5、formal>=20；准确任务名解析、内容互异、状态库指纹|collection/OFT/CLI 计划、只加载一次、逐状态调用测试|5+20 真实回合未运行|
|四 表 1：回合统计|episode_statistics.csv，任务/初态/回合、抓取序列、官方结果、步数、EEF 长度|analysis CSV/官方与人工结果分离、缺失与计数测试|真实 CSV 待生成核对|
|四 表 2：逐尝试事件统计|subtask_events.csv，每次尝试一行、事件步、操作步、结果与备注|重复/失败/null、操作区间包含两端，CLI 导入后 CSV 测试|真实人工事件待确认|
|四 表 3：任务行为汇总|task_summary.csv，成功率、方向比例、成功平均步数及分母/不同初态|未知/失败分母、任务/阶段/对象/synthetic 分组测试|真实比例待实验；不填示例数值|
|四：逐步原始数据|每步 NPZ+JSON、初态/模拟器状态、图像、元信息；禁用 pickle|原子提交、读取、异常和证据快照测试|真实原始数据待生成；只存私有|
|五 1：运行代码与说明|vla_trajectory、scripts、README、A100 清单、计划模板|完整本地230 passed/1 codec skip、独立审查通过、0.2.0 wheel隔离安装与完整CLI链路验证|远程当前提交及CI结果见仓库；实机仍待验|
|五 2/4：完整轨迹及任务/初态命名视频|episode 目录、execution 完整性、任务/初态/episode MP4 与映射|视频 provenance 与所有 transition 精确匹配测试|真实原始数据/视频未交付|
|五 3：三类统计文件|三张 UTF-8 BOM CSV，正确引用、公式前缀转义|中文/逗号/换行/公式前缀及 CSV 行数测试|真实表待 A100 数据与人工核对|
|五 5：末端 XYZ 与事件可视化|plots PNG+SVG、等比例 XYZ、坐标随动作步变化，抓/放后状态标记|Matplotlib 图测试及 synthetic 图打开检查|真实图待生成，不跨坐标缺口|
|五 6：报告顺序偏好/成功率/步数分布|analysis_report.md，样本/分母/缺失/视频覆盖/初态观察与解释局限|中文报告、synthetic 标识、视频映射测试|真实结论待数据|
|六：基础目标整体|一个双物体任务，二十不同初态、完整轨迹/视频/顺序、三表/图|代码链路 CPU 验证；不能据此宣称实验完成|全部待 A100 真实验收|
|六：扩展多个任务|plan 支持任意准确任务名与显式对象对，按任务分组|自定义任务不继承默认对象测试|多任务实机扩展未验收|
|六：自动抓放与布局影响进阶|当前人工抓放，自动事件只表示目标谓词；有初态比较，不做布局因果研究|自动/人工证据独立性测试|自动抓放识别与因果实验未实现/未验收|
|七：固定顺序偏好与布局变化问题|报告观察值、方向分母、轨迹/初态差异，不推断规划意图或最优路径|统计/报告生成测试，仅 synthetic 演示|真实答案尚不能给出|
|持续交付：GitHub 与飞书|公开仓库保留可见性；仅代码/正式文档/发行包|0.2.0 独立审查、全量/包验证及逐文件远程哈希交付流程，保留旧wheel历史|GitHub结果见main及Actions；飞书缺连接/认证未发送|

## 判定口径

官方成功率以 manifest 的已知成功和失败为分母，未知单列；人工 outcome 不覆盖官方结果。顺序比例以已复核且可可靠比较的回合为分母，未知/并列不强分方向；重复抓取保留。平均成功步数只取官方成功且总执行数可信者。pilot、formal、任务、对象对和 synthetic 分开。

EEF 长度为每个 policy transition 的前后坐标差之和，包含第一个动作；任何应计坐标缺失或执行记录不完整时完整长度未知，只给观测长度与覆盖率。执行步异常无法确认时总步数未知，已执行数作下界。视频覆盖核验要求匹配 episode/任务/初态及完整逐帧 sidecar，不代替媒体解码或人工播放。

## 当前证据与未完成项

分组、完整测试及安装包验证记录见 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)。本轮最终完整本地测试为 230 passed、1 skipped，独立审查与修复复审通过；真实编码本地缺依赖明确 skip。已安装编码器的 CI 运行状态以 [当前提交的 Actions](https://github.com/lianggeming14/openvla/actions) 为准，配置本身不被当作成功结果。

用户已提供 OFT 根目录 `/HUBU-AI096/lgm/ICASSP/peft-in-openvla`，尚无可用的 A100 服务器会话，未产生真实 5+20 回合、原始数据、完整视频、人工标签或真实模型分析结论。按 [A100 详细操作手册](A100_ACCEPTANCE.md) 执行；需要手动替换的值可在 [离线 HTML 手册](A100_RUNBOOK.html) 下载打开后查看蓝色标记。飞书缺可用发送认证与明确接收位置，未发送。不用 synthetic 或先前版本的发布结果代替实机验收。
