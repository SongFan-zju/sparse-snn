# 边界背景，不作为正向机制主线

主报告已按用户最新要求改为分析“新梯度为什么有效”，见`task_alignment.md`。下列已证事实只限定理论适用范围，不构成本轮继续改造ViStream loss或重训的安排。

1. 局部相位均值Jacobian乘hard-output任务导数，一般不等于相位平均非线性任务loss的梯度。主报告用R明确保留这一项；不能把`E[硬路径普通导数]`当成均值响应的导数，因为离散边界有分布意义上的贡献。
2. ViStream身份CE对某个检测列所有身份score的共同平移不变，梯度列和为零；原生可拒配assignment有cost=.7阈值，故对该平移并不不变。用冻结原生`scipy_lapjv`的2×2 score算例，CE=.055844不变时可从2个匹配变成1个；另一个score矩阵CE更低(.042254)也只有1个匹配。这里未证明这些任意score变化能由真实特征/权重实现，更没有证明它解释全部HOTA下降。
3. 已完成、已复核的ViStream context3epoch中density相对三角HOTA−2.3370、IDF1−3.1018pp。以此作为该任务配置的负结果，不恢复旧cold/folded解释，不用它否认已有分类/DVS正收益。

2×2例子的代码和数值保留于本目录`check_math.py`、`checks.json`；它们是主报告转向前产生的局部边界例，不是新增训练结果。当前机制实验以`mechanism_checks.json`为准。
