# 论文研究交接

本分支 `paper` 从远程 `main` 建立，面向当前替代梯度工作的写作与证据复核。先读 README.zh-CN.md、MATH_RESEARCH_INDEX_ZH.md、results/paper/README.md 和 RUNNING.md。

- 结果以 results/paper/RESULTS_TABLE.json 的预定最终终点为入口，FILE_MANIFEST.json 可追到原始来源及导出证据。历史协议的时间状态不覆盖最终 assessment。
- 区分来源 ANN、直接转换和微调 SNN；把共同微调收益与新梯度相对三角的额外收益分开。不要混用不同 L/T、骨干、划分和指标。
- 当前通用 horizon、历史 T2 闭式、作者 Fast-SNN 前向和 ViStream 有界实现分别记录在 IMPLEMENTATION_VARIANTS.json。不要把旧均匀相位或多等级事件结果改称当前方法。
- 保留负结果和反例。DiMP/ViStream 为本地跟踪任务，不是官方 SOT/MOT 基准；AO 与 HOTA 不横比。标准 R18 的当前匹配官方测试证据缺项已注明，不以 Fast-SNN 的不同结构代替。
- 局部条件均值 Jacobian 的证明不等于全网硬任务梯度或准确率保证。数学索引明确区分已证明结论、假设和猜想。
- 数据与权重不在 Git 中。源码、完整训练环境与历史实验的覆盖范围见 RUNNING.md 及 provenance/code_inventory.json。不得从已有哈希推断已托管 checkpoint。
- 不因写作自动启动新训练。T=15、Transformer 和退役 R50 分类实验不恢复。此分支不包含其他 session 的稀疏研究。
