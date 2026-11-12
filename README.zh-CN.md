# 共享相位时序梯度：低时间步 ANN–SNN 转换

[English](README.md) | **简体中文**

**Shared-Phase Temporal Gradients for Low-Step ANN-to-SNN Adaptation**

本项目提供基于 ResNet18 和 ResNet50 的 ANN–SNN 转换与适配代码。核心方法利用共享初始相位下的跨时间响应构造替代梯度，训练时保留硬脉冲前向传播，推理时只执行一条 SNN 轨迹和一次最终分类。

教师 ANN 仅用于带知识蒸馏的训练配置。最终推理不需要教师模型，也不使用多视图或重复完整推理融合。M3 整数事件扩展允许每步输出更高精度的事件，增加了事件容量，因此不能将它视为与二值脉冲等成本的比较。

## 环境安装

需要 Python 3.10+、PyTorch 2.2+：

```bash
pip install -r requirements.txt
```

命令默认使用 CPU。通过 `--device cuda:0` 选择逻辑 GPU，通过 `CUDA_VISIBLE_DEVICES` 选择物理 GPU。以下示例使用物理 GPU 2、3；设置 `CUDA_VISIBLE_DEVICES=2,3` 后，它们分别对应逻辑 GPU 0、1。

原始 SNN 适配实验使用双卡 `DataParallel`，各卡分别计算 BatchNorm 统计量，并保留主卡的运行统计量。使用 `--data-parallel` 启用这一并行方式。

## 项目结构

```text
configs/
  experiments.json       # 训练与评测预设
  cifar100_split.json    # CIFAR-100 固定数据划分
  weights.json           # 权重文件名、SHA256 和已有下载地址
scripts/
  train.py               # ANN 训练、QAT、SNN 适配
  evaluate.py            # 干净数据集与 CIFAR100-C 评测
  benchmark.py           # 推理耗时与事件统计
  plot.py                # 机制图与扰动结果图
src/
  conversion/            # 脉冲算子与梯度方法
  models/                # ResNet 骨干和检查点加载
  third_party/           # 保留的上游 QCFS 实现
  ...                    # 公共数据、训练、评测等模块
tests/                   # 算子、模型加载和训练恢复测试
```

数据、权重、训练日志和生成的图表不进入 Git 仓库。上游实现来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 数据与权重

### 数据准备

使用解压后的 CIFAR-10、CIFAR-100 Python 版本。`--data` 可以指向数据集目录，也可以指向其父目录，例如：

```text
data/
  cifar-10-batches-py/
  cifar-100-python/
  CIFAR-100-C/            # 可选，用于扰动评测
```

程序不自动下载数据。训练流程只读取训练数据，官方测试集通过独立评测命令读取。

CIFAR-100 使用仓库提供的[固定划分](configs/cifar100_split.json)：

| 用途 | 图片数量 | 来源 |
|---|---:|---|
| 训练 `training` | 42,000 | 官方训练集 |
| 校准 `calibration` | 1,000 | 官方训练集 |
| 开发评估 `selection` | 2,000 | 官方训练集 |
| 确认评估 `confirmation` | 5,000 | 官方训练集 |
| 官方测试 | 10,000 | 官方测试集 |

**确认集与官方测试集分开，确认集来自官方训练集。** 上述四个训练集内划分互不重叠。

CIFAR-10 使用全部 50,000 张训练图片，不额外划分内部验证集。所用公开 ANN 来源模型曾使用官方测试集进行模型选择，因此后续转换实验继承了这一测试集使用历史。

CIFAR100-C 需要包含 `labels.npy` 和各扰动类型数组的目录。程序评测论文采用的 15 种扰动、每种 5 个强度，共 75 种情况。

### 权重准备

[weights.json](configs/weights.json) 列出了原始检查点对应的建议文件名和 SHA256。上游 CIFAR-10 ANN 已有固定版本的下载地址。

**本项目训练得到的权重尚未公开托管。** 下方命令以相应权重已经放入本地 `weights/` 为前提；权重清单本身不包含模型文件。

原始实验检查点和公开训练流程生成的检查点均支持推理。最终 SNN 检查点包含所需权重及阈值，评测时不依赖原始 ANN、教师、校准缓存或旧实验目录。加载时需选择对应的实验预设，因为权重张量不能单独表达所有时间步和事件容量设置。

## 快速评测

以 CIFAR-100、ResNet18、M3/T2 为例：

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/evaluate.py c100-m3-t2 \
  --checkpoint weights/c100-m3-t2.pt --data data \
  --device cuda:0 --batch-size 1024 --output runs/c100-m3-t2-test.json
```

更换预设名和检查点即可评测 `c100-m3-t4`、`c10-m3-t2` 等配置。

- 默认评测官方测试集。
- `--split selection` 或 `--split confirmation`：评测 CIFAR-100 的相应训练集内保留划分。
- `--unadapted`：输入 QAT 检查点，直接转换为指定 SNN 后评测，不进行 SNN 适配训练。

CIFAR100-C 评测示例：

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/evaluate.py c100-phase-t2 \
  --checkpoint weights/c100-phase-t2.pt --corruptions data/CIFAR-100-C \
  --device cuda:0 --output runs/phase-corruptions.json
```

## 训练与复现

### 训练设置

[experiments.json](configs/experiments.json) 集中定义各实验预设：

| 阶段 | 训练轮数 | 主要设置 |
|---|---:|---|
| CIFAR-100 ResNet18 ANN 来源训练 | 400 | 普通 ReLU ANN、混合标签 CE |
| QCFS 量化感知训练（QAT） | 200 | 通道阈值可训练 |
| 硬脉冲 SNN 适配 | 160 | 阈值固定，更新网络参数 |

使用固定随机种子 `2026100321`、SGD、前 5 轮线性预热及后续余弦学习率衰减。数据增强采用像素空间补零、随机裁剪、水平翻转，并按相同概率选择 Mixup 或 CutMix。

带 KD 的 SNN 配置使用混合标签 CE，加上 `0.5 × KL(teacher || student)`，蒸馏温度为 1。教师保持冻结及评测模式。

### 主方法训练

从已准备的 QAT 来源模型和普通 ANN 教师开始：

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/train.py c100-m3-t2 \
  --data data --source weights/c100-qat.pt --teacher weights/c100-ann.pt \
  --output runs/c100-m3-t2 --device cuda:0 --data-parallel
```

其中 `--source` 是学生的 QAT 初始化，`--teacher` 是知识蒸馏使用的普通 ANN。两者在训练中的作用不同；最终评测只需要训练得到的 SNN 检查点。

### 来源模型准备

ANN 训练和 QAT 使用独立预设。以下命令可训练 CIFAR-100 ResNet18 ANN：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train.py c100-ann \
  --data data --output runs/c100-ann --device cuda:0
```

复现已报告的 QAT 路线时，使用原始 ANN 检查点及其配套初始阈值：

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/train.py c100-qat \
  --data data --source weights/c100-ann.pt \
  --thresholds weights/c100-qat-thresholds.pt \
  --output runs/c100-qat --device cuda:0 --data-parallel
```

CIFAR-100 ResNet18 的 QAT 预设要求提供初始阈值；阈值文件记录了来源 ANN 的 SHA256。独立重训的 ANN 需要匹配的校准结果，不能直接复用来源哈希不同的阈值文件。

其他路线：

- **CIFAR-10**：从固定版本的公开 `c10-ann.pt` 开始，以 `c10-qat` 准备 QAT 来源，再训练 `c10-phase-t4` 或 `c10-m3-t2`。省略 `--thresholds` 时，`c10-qat` 会执行原配方的 CPU 采样校准。
- **ResNet50**：使用已有 ANN 来源，经 `c100-resnet50-qat` 和 `c100-resnet50-t2` 两个阶段。QAT 支持加载配套阈值或执行采样校准；本项目未提供新的 ResNet50 ANN 从零训练配方。

### 显存调整与断点恢复

可通过 `--batch-size` 适配显存，训练图片和总 epoch 保持原设置。实际 batch、并行方式、更新次数和耗时会记录到输出目录。

新训练要求输出目录为空。对公开训练流程生成的实验，在原命令后添加 `--resume`，即可从输出目录的 `latest.pt` 恢复优化器及随机状态。恢复时可以省略 `--source` 和 `--thresholds`，但 KD 配置仍需提供同一个教师检查点。

历史内部实验检查点支持评测；新的公开训练流程不直接恢复其优化器状态。正式测试应在训练结束后单独执行，不使用官方测试结果选择训练检查点。

## 实验配置与结果

`M` 表示每步有符号事件计数的最大绝对值，`T` 表示推理时间步数。R18、R50 分别表示 ResNet18、ResNet50。

以下为原始已完成实验的官方测试集 Top-1 正确率。公开训练代码重构后已进行一致性检查，尚未重新完成全流程训练。

| 配置名 | 数据集 / 骨干 | M / T | 梯度方法 / 目标 | Top-1（%） |
|---|---|---|---|---:|
| `c100-m3-t2` | CIFAR-100 / R18 | 3 / 2 | 共享相位 / CE+KD | **80.32** |
| `c100-m3-t4` | CIFAR-100 / R18 | 3 / 4 | 共享相位 / CE+KD | 80.46 |
| `c100-phase-t2` | CIFAR-100 / R18 | 1 / 2 | 共享相位 / CE | 77.29 |
| `c100-triangular-t2` | CIFAR-100 / R18 | 1 / 2 | 三角替代梯度 / CE | 76.46 |
| `c100-diagonal-t2` | CIFAR-100 / R18 | 1 / 2 | 对角相位消融 / CE | 73.49 |
| `c100-phase-kd-t2` | CIFAR-100 / R18 | 1 / 2 | 共享相位 / CE+KD | 77.87 |
| `c100-phase-kd-t4` | CIFAR-100 / R18 | 1 / 4 | 共享相位 / CE+KD | 80.18 |
| `c100-triangular-kd-t4` | CIFAR-100 / R18 | 1 / 4 | 三角替代梯度 / CE+KD | 79.34 |
| `c100-resnet50-t2` | CIFAR-100 / R50 | 1 / 2 | 共享相位 / CE | 77.97 |
| `c10-phase-t4` | CIFAR-10 / R18 | 1 / 4 | 共享相位 / CE+KD | 95.95 |
| `c10-m3-t2` | CIFAR-10 / R18 | 3 / 2 | 共享相位 / CE+KD | **96.15** |

M3/T2 与 M1/T4 的比较同时改变了时间步和事件容量。ResNet50 结果用于验证整套方法的迁移效果，不单独归因于梯度改动。

## 成本测量、绘图与测试

推理成本测量：

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/benchmark.py c100-m3-t2 \
  --checkpoint weights/c100-m3-t2.pt --data data --device cuda:0 \
  --output runs/c100-m3-t2-cost.json
```

默认使用 1,000 张训练或校准图片，batch 为 256，预热 3 次、计时 5 次。输入预先加载，预处理和事件统计钩子不计入推理耗时。报告包括非零有符号事件、单位事件计数，以及表示事件幅值所需的比特事件数。

测量对象是 **FP32 稠密 PyTorch 仿真**，不代表神经形态硬件实测能耗或稀疏算子加速。CPU 计时可用于功能检查，不应与论文的 GPU 耗时直接比较。

生成机制图及扰动对比图：

```bash
python scripts/plot.py phase --output figures
python scripts/plot.py corruption --phase runs/phase-corruptions.json \
  --baseline runs/triangular-corruptions.json --output figures/corruptions.pdf
```

其中两个扰动 JSON 均由 `evaluate.py --corruptions` 生成，应使用对应方法和基线的检查点。

运行 CPU 单元测试：

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests
```

测试覆盖脉冲算子、梯度行为、独立检查点加载和训练中断后的恢复一致性。通过各入口的 `--help` 可查看完整参数，例如：

```bash
python scripts/train.py --help
python scripts/evaluate.py --help
```
