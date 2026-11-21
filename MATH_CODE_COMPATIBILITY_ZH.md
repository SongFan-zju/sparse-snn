# 数学小检查与导出代码的兼容性

2026-11-21：已只读比较代码包与本机数学材料筛选时的10个依赖模块。9个逐字节一致；`src/conversion/qcfs_ecl_resnet18.py`存在以下已知差异。

| 指纹对象 | SHA256 |
|---|---|
| 数学包初次筛选时本机当前源码 | `e6a82eb40d3dcba8daf31b14ab7ea4842408cf9daa86866ed71d43d10c12e625` |
| 实际随代码包导出的冻结版本 | `9c9e183eb1071584095572945a976edf5eff2dc6e1e1dcdbe46600a3511b54de` |

本机版本将一次状态更新抽为 `DualThresholdSequence.advance(current, state)`；导出冻结版本直接在 `forward` 内写循环。已核对初始膜θ/2、count=0、signed正比较`>`、unsigned正比较`>=`、负撤回阈值−0.001及资格、输出电荷、reset与count的运算顺序一致。这里的差异涉及接口重构，不是把本机较新接口声称为历史冻结实现。

三项小检查使用的对象不依赖新增的 `advance` 接口：统一计数检查直接调用 horizon 的均值/VJP；机制检查调用独立 `hard_sequence` 与三角SG；有界基础检查调用 `bounded_phase` 与 ViStream 的 `_TriangleEvent`。代码包中这些模块与初次检查的源码一致。`qcfs_ecl_resnet18.py`作为传递依赖被导入，冻结版仍提供所需类。

随后以 `${CODE_BUNDLE_ROOT}` 为工作目录和唯一指定的源码 `PYTHONPATH`，对数学包内三项脚本重新执行；脚本自身根目录没有 `src`，因此实际导入来自代码包。设置 `PYTHONDONTWRITEBYTECODE=1`、CUDA不可见与CPU线程2，没有改动代码包或原始结果。三项均退出0、CUDA未初始化，重写在数学暂存副本内的小JSON均与原始证据逐字节一致。记录见 `MATH_BUNDLE_VERIFICATION.json`。

`MATH_SOURCE_REQUIREMENTS.json`明确分别保存 `local_source_sha256` 与 `exported_code_sha256`。前者是初次筛选时本机来源指纹，后者才是随分支导出的模块指纹；两者不得混写。该兼容检查验证三个已有数学小检查的可复现性，不替代历史训练快照的来源绑定，也不声称两份文件在所有外部调用方式下接口兼容。
