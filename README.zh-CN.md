# ANN–SNN 转换中的共享相位轨迹梯度

[English](README.md) | **简体中文**

`paper` 分支用于把当前工作交接给论文写作 agent：保留核心方法、任务代码、数学推导和实验结果。从远程 `main` 建立；数据、权重、环境、无关研究和训练大日志不纳入该分支。

训练前向仍采用确定性的 signed 脉冲，只替换反向。把同一三角分布初始相位作用于完整神经元轨迹，计算条件平均输出电荷的 Jacobian，处理发放、复位与后续撤回之间的时间信用。部署只运行一条确定性 SNN 轨迹。L 表示来源 ANN 的量化级数，T 表示 SNN 时间步。

建议按下面的入口阅读：

| 内容 | 入口 |
|---|---|
| 最终终点结果与评估口径 | [结果说明](results/paper/README.md)、[CSV 表格](results/paper/RESULTS_TABLE.csv)、[JSON 表格](results/paper/RESULTS_TABLE.json) |
| 各组实际梯度版本、前向规则与冻结代码哈希 | [实现版本索引](results/paper/IMPLEMENTATION_VARIANTS.json) |
| 动机、证明、假设、反例与中文报告 | [数学研究索引](MATH_RESEARCH_INDEX_ZH.md)、[文献索引](MATH_REFERENCES_ZH.md) |
| 环境、任务运行入口与外部资源 | [RUNNING.md](RUNNING.md) |
| 代码来源和筛选范围 | [代码清单](provenance/code_inventory.json)、[上游说明](THIRD_PARTY.md) |
| 尚未核实的历史证据 | [缺项说明](results/paper/SCOPE_AND_GAPS.json) |

核心算子为 [triangular_density_horizon.py](src/conversion/triangular_density_horizon.py)，可选加速为 [triangular_density_triton.py](src/conversion/triangular_density_triton.py)，匹配三角对照为 [ecl_triangular_surrogate.py](src/conversion/ecl_triangular_surrogate.py)。历史 T=2 闭式实现和 ViStream 有界神经元另行标注，不混称为当前通用实现。

结果区分来源 ANN、直接转换及微调 SNN，使用预定最终 epoch，保留持平和负结果。CIFAR100 使用官方测试集，Tiny-ImageNet 使用官方带标签验证集，CIFAR10-DVS 使用已保存的固定本地划分，Gesture 使用官方 trials。跟踪来自本地 MOT16 留出任务，不是官方跟踪基准。当前方法的标准 R18 匹配证据未核实的历史项目明确列缺项，不拿不同结构的 Fast-SNN R18 代替。

数学结论针对条件平均神经元输出的局部 Jacobian；不能写成硬任务损失的精确梯度、无偏全网任务梯度或正确率保证。材料中保留假设与反例。最终跟踪实验未显示相对三角梯度的额外收益，写作时应如实保留这个边界。

公共依赖通过 `pip install -r requirements.txt` 安装。各任务的额外依赖、上游仓库与命令见 RUNNING.md；CPU 检查命令见英文 README。部分 CUDA 与作者框架集成检查需要另行准备资源。权重仅保留哈希，没有随分支公开托管。

随包代码不等于全部历史环境的完整重放，具体覆盖范围见代码清单与各组结果证据。本分支尚不包含 ICML 论文正文。
