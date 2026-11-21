# 共享三角相位轨迹梯度：给写作 agent 的研究索引

整理日期：2026-11-21。本目录是已有研究材料的筛选交接，不是新论文正文、完备查新或新增证明。保持原报告与原始数值不变；以下版本说明优先于历史文件里的运行状态和笼统的“跟踪”称谓。最终实验表以干净分支的实验结果索引为准。

## 先读哪些文件

1. `reports/phase_gradient_math_20261119/数学猜想与证明交接报告.txt`：已有14节中文报告，PDF同名；`report_source.json`是文字源。报告第2–8节覆盖定义、证明、误差分解、下降条件和反例，第9节是固定真实轨迹证据。第10节实验状态是11月19日快照，不能作为当前最终结果表。
2. `artifacts/tracking_gradient_math_20261119/unified_count_boundary.md`：主数学入口，统一无上限标准 signed ECL 与历史有上限状态机；直接覆盖当前分类/DVS/DiMP 使用的非零负阈值。
3. 同目录 `task_alignment.md`：简明动机、任务梯度误差分解和可检验的研究猜想。`foundations.md`提供最清楚的两步信用归还、局部下降、混合偏导及反例推导；其中早期泛称“跟踪”的 bounded 部分仅指 ViStream，须按下面的实现映射阅读。
4. 同目录 `real_trace_check/report.md`、`selection.json`、`results.json`：预固定2张训练图、16条末层轨迹的完整证据，保留15条 density 更近、1条三角更近的反例。没有把这个局部比例作为总体概率或准确率归因。
5. 同目录 `tracking_objective.md`：原生 DiMP 模板/搜索、有限步滤波优化与定位监督的数学背景。`credit_cpu.json`、`task_jump_cpu.json`仅为合成小例的旧输出，不是跟踪基准结果；本包不带其上游 DiMP 环境。

`motivation_zh.md`是早期动机草稿；`boundary_background.md`保存 ViStream 负结果及目标不匹配的边界。它们用于追溯，不覆盖统一定理或当前结果。

## 当前数学对象与可说到的结论

保持原来一次确定性硬推理，只有反向更换。每步最多一次正事件或一次有资格的负撤回。L是共同来源 ANN 的量化级数，T是 SNN 时间步；证明里的可选计数上限 K 不是新调参项。固定输入电流 I、阈值、初态及负阈值，把同一个三角相位偏移用于整条事件轨迹，定义条件电荷均值 μ(I)=Eδ[y(I;δ)]。当前反向为 `(Dμ)^T g_hard`，辅助相位不在部署时采样。

动机是完整状态轨迹上的时间信用：初期发放会改变后续复位和撤回资格。普通三角也通过 soft reset 做时间反传；共享相位的区别在于同时计算其他可行状态分支的边界移动。一步斜率与匹配三角相同；多步可产生负的跨时间项，把最终已被撤回的早期事件信用抵消。

| 结论状态 | 已有材料支持的内容 |
|---|---|
| 已有证明推导和CPU复核 | 统一计数尾事件边界递推；尾和得到电荷均值；避开非光滑边界时的精确条件均值 VJP；T=1三角形状匹配 |
| 明确假设下成立 | 固定读出 g，`Q(I)=g^T μ(I)`；若局部梯度非零、Lipschitz且小步留在该光滑邻域，沿负 VJP 下降。这是局部目标结论 |
| 恒等式与充分条件 | 对明确的条件平均任务 Ψ，`R=∇Ψ−Jμ^Tg_hard`、`D=(Jμ−Jtriangle)^Tg_hard`；两者平方误差差为 `||D||²+2<R,D>`，`||D||>2||R||`足以使 density 更近 |
| 有限数值支持 | 预固定16条真实末层轨迹中15条更近、1条不占优；合成 DiMP 小例验证固定 hard adjoint 的条件线性目标 |
| 仍是猜想 | 这些局部校正是端到端收益的主要原因；残差汇合的方向改善；模板/搜索联合信用改善真实定位；局部结果组合成参数级或全网下降 |

不能写成 hard loss 的精确梯度、无偏全网随机损失梯度、全网保守场、全局收敛或正确率保证。通常 `Eℓ(y)`、`ℓ(Ey)`和硬前向损失不同。T=1凸二次损失已存在当前SG与精确条件期望任务梯度符号相反的例子；这也适用于同一步三角。条件数也不保证更好。ties采用实现明确的单侧约定，实数命题按几乎处处表述；不能据连续积分忽略端点而改变部署比较符号。

## 神经元与历史证明的对应

| 应用 | 实际神经元 | 使用哪份证明 |
|---|---|---|
| 当前标准分类、DVS、Tiny、DiMP18 | 无计数上限 standard signed ECL，初态θ/2、count=0、负阈值−0.001；DiMP用 `triangular_density_horizon.py` | `unified_count_boundary.md` 的 K=∞、非零ε情形 |
| Fast-SNN 分类/检测 | 保留作者原生 `>=` 正比较、signed IF及任务读出；具体部署以对应源码/快照为准 | 连续相位积分可使用统一计数定理，不能把标准 ECL 的严格比较覆写到框架中 |
| 历史 ViStream | `bounded_phase.py`；有限累计计数上限K=L、ε=0，可能继承窗口前初态 | `foundations.md` 有界推导，以及统一定理ε=0化简 |

当前 DiMP18 **不是** bounded_phase。旧 Uniform[0,1] 相位和旧 T2 闭式是历史兼容实现：Uniform不能当作当前三角分布；T2和一般T在 ties 上可有不同约定。它们不因此成为本轮新方法或新增实验。

## 必须保留的跟踪结果边界

ViStream 修正后已复核负结果：三角 HOTA 22.019327、新梯度19.682367，差−2.336960pp；来自 `results/vistream_context_grad_20261118/final_assessment.json`。它是本地 MOT16 留出2304帧，不是私有官方测试；不能继续用已修复的旧 detach 解释该轮失败。

**11月21日交接时 DiMP 全量比较已完成，更新历史报告“尚未完成”的状态。** `results/dimp18_tracking_full_20261119/final_assessment.json`：20轮最终 AO 三角36.801262、新梯度36.653223，差−0.148039pp；直接 ECL 37.490892，两微调臂均低于直接转换。229条本地 GT 初始化 SOT 轨迹、32589帧访问/30418评分帧；不是官方 SOT benchmark，也不能与 ViStream 的 MOT HOTA 横比。最终独立训练链路检查未发现新实现错误；保留终点，不用 epoch10 替换它。数学小例不构成跟踪有效的证据。

这些结果的完整JSON由实验结果包保留；本索引只作准确引用，不另建实验总表。Tiny 最新终点以总实验表为准，历史数学报告不包含该轮最终数值。固定单种子 seed=2026100321；不设置多种子继续门槛。

## CPU复现与文件用途

将本包按相对路径合并到代码分支根部后，从仓库根运行以下脚本。需要已安装 PyTorch 的项目 Python 环境及 `src` 的传递导入；不需要训练数据、checkpoint、GPU或完整 DiMP 环境。`MATH_SOURCE_REQUIREMENTS.json`分别列出筛选时本机源码指纹和实际导出代码指纹，实际历史实验仍以各自冻结快照为准。冻结QCFS模块与本机重构版有已核对的差异；已用实际代码包独立重跑三项检查，输出仍与原始证据一致，详见 `MATH_CODE_COMPATIBILITY_ZH.md`。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python artifacts/tracking_gradient_math_20261119/check_unified_count_boundary_cpu.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python artifacts/tracking_gradient_math_20261119/check_positive_mechanism.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python artifacts/tracking_gradient_math_20261119/check_foundations_cpu.py
```

三个脚本分别复核当前非零ε统一定理、T1匹配与R/D误差恒等式、历史bounded局部机制/反例；均只在自身目录重写对应小JSON。已有JSON是原始证据副本；交接副本的额外CPU复核记录在 `MATH_BUNDLE_VERIFICATION.json`。

真实轨迹证据以 `results.json`保存全部16条电流/相位cells/Jacobian/R/D及不占优点，以`selection.json`保存选择约定。包中不带原始图像缓存和权重，不把重新抽取真实轨迹列为纯数学小检查；原抽取脚本仍在原工作仓库 `artifacts/tracking_gradient_math_20261119/real_trace_check/run.py`。当前干净分支可直接检查已保存数组，完整重新抽取需原数据、来源、manifest及冻结snapshot。

本包刻意排除 `reports/.../study/`、`earlier_scope/`、旧zip、图片预览、权重、数据、论文全文，以及其他session的 weight sparsity/early-exit/lowrank材料。来源索引保留原始URL、版本和论文SHA；其中相对缓存路径是资料出处，不意味着相关PDF已打包。原报告原样保留，另有路径规范化的 `report_source.portable.json`；逻辑根和原始/导出SHA的区别见 `MATH_PORTABILITY_ZH.md`。逐文件来源、体积、`original_source_sha256`与`export_sha256`见 `MATH_BUNDLE_MANIFEST.json`。
