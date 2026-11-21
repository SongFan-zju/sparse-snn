# 本地数学资料的引用入口

本文件仅整理已有缓存索引和笔记，无新联网或完备新颖性调查。外部论文全文不随分支打包。缓存版本、下载出处和原文件SHA见 `artifacts/papers/index.json` 与 `artifacts/papers/phase_triangle_analysis_20261116/index_phase_triangle.json`。下列关联是项目已有笔记的研究定位，写作时仍需核对原文具体公式与适用范围。

| 资料 | 原文入口 | 本项目需要区分的问题 |
|---|---|---|
| Efficient ANN-SNN Conversion with Error Compensation Learning | https://arxiv.org/abs/2506.01968 ；缓存v1；另有ICML2025 proceedings版本 https://proceedings.mlr.press/v267/liu25bl.html | 标准 signed ECL前向与论文实验基线；版本不可混用。缓存v1 Table8在PDF第15页，不能把ImageNet表格写成CIFAR100结果 |
| Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks（QCFS） | https://arxiv.org/abs/2303.04347 | 来源ANN量化L、直接转换与后续SNN微调应分开 |
| Julia Gygax & Friedemann Zenke, Elucidating the theoretical underpinnings of surrogate gradient learning in spiking neural networks | https://arxiv.org/abs/2404.14964 ；缓存v3，2024-11-17 | SG、平滑概率模型和随机自动微分的关系；一般多层SG非保守性不是本项目首次发现 |
| Ma et al., Exploiting noise as a resource for computation and learning in spiking neural networks | https://arxiv.org/abs/2305.16044 ；DOI https://doi.org/10.1016/j.patter.2023.100831 | 三角噪声/密度与三角SG的联系已有工作，不能把三角形状本身作为新颖性 |
| Kang et al., A²SG: Adaptive and Asymmetric Surrogate Gradients for Training Deep Spiking Neural Networks | https://proceedings.mlr.press/v306/kang26g.html | 自适应/非对称SG和时间方向一致性是相关工作；本项目应具体比较完整signed轨迹边界积分，而非笼统声称首个时间一致SG |
| Van den Berghe et al., Adaptive Surrogate Gradients for Sequential Reinforcement Learning in Spiking Neural Networks | https://proceedings.neurips.cc/paper_files/paper/2025/file/d94b46ec30adee2bbb134f813fc9dde0-Paper-Conference.pdf | 已有关于SG斜率、序列与预热的研究；不能把简单调梯度幅度当作当前完整轨迹方法的贡献 |

内部主要推导是本仓库对真实实现的研究材料，不冒称外部已发表定理。潜在贡献的准确候选表述是：固定标准signed硬前向，对共享初始三角相位驱动的完整发放/复位/有资格撤回轨迹先积分、再求条件均值VJP；统一计数尾边界递推给出一般T实现，并用明确局部目标解释时间信用修正。其独立新颖性仍需写作agent对最接近文献作公式级核查，短查重不能保证首次提出。
