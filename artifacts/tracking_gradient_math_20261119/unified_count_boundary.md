# 同时覆盖分类ECL与bounded跟踪神经元的统一计数边界定理

这是 `foundations.md` 的必要覆盖补充，不改变已有方法、不训练模型。主命题同时包含：分类的无计数上限、负阈值−ε（实际ε=.001），以及跟踪的有限计数上限、ε=0。它使条件轨迹Jacobian精确性的理论直接覆盖当前CIFAR100/DVS正收益实验实际使用的分类算子，而不只覆盖ViStream局部模型。

为避免与来源ANN量化级数混淆，下面用 **K** 表示可选的神经元计数上限：ViStream中K=L；当前分类signed ECL没有这个上限，取K=∞。K只是统一证明中的符号，不是给分类方法增加一个超参数。来源ANN的L不应被误当成分类ECL的count cap。

## 命题：带可选计数上限与负阈值的条件边界递推

固定有限T、阈值θ>0、负阈值幅度ε≥0、整数初始计数c₀≥0、归一化膜q₀∈R。若K有限，要求K为正整数且c₀≤K。令η=ε/θ，初始归一化膜为q₀+δ，δ来自固定分布。对每一步，设

\[
S_t=\sum_{s=1}^t I_s/\theta,\qquad
u_t=q_0+\delta+S_t-(C_{t-1}-c_0).
\]

一次正发放或一次有资格负撤回的实数状态机为

\[
e_t=1\{u_t>1,\ C_{t-1}<K\}
-1\{u_t\le-\eta,\ C_{t-1}\ge1\},
\quad C_t=C_{t-1}+e_t,\quad y_t=\theta e_t.
\]

K=∞时第一项的计数上限条件自动成立。严格/非严格比较只改变相位边界点；对连续相位分布其质量为0。数学均值结论因此也覆盖ViStream实际`>=1/<0`及Fast-SNN实际`>=θ/<=−ε`的比较约定。部署hard forward仍必须保留各自原比较，不能据此修改实际推理。

定义 `p_{t,k}=k−c₀−q₀−S_t`。则对每个合法计数级k≥1，存在扩展实数阈值B_{t,k}，使

\[
\{C_t\ge k\}=\{\delta\ge B_{t,k}\}
\]

成立（忽略连续分布下零概率的边界点）。初始化为

\[
B_{0,k}=\begin{cases}-\infty,&c_0\ge k,\\+\infty,&c_0<k,\end{cases}
\quad B_{t,0}=-\infty.
\]

有限K时 `B_{t,K+1}=+∞`；无上限时所有不可达级 `k>c₀+t` 都为+∞。递推为

\[
\boxed{\ B_{t,k}=\min\!\left(
B_{t-1,k+1},\ 
\max(B_{t-1,k},p_{t,k}-\eta),\ 
\max(B_{t-1,k-1},p_{t,k})
\right).\ }
\]

### 证明

固定k，按上一时刻计数分为四类，利用每步事件只能为−1、0、1：

1. **C_{t−1}≥k+1**：即使本步撤回一次也仍≥k，不需要额外膜条件。
2. **C_{t−1}=k**：需要本步不撤回。因为k≥1，撤回资格成立；不撤回等价于 `u_t>−η`，即 `δ>p_{t,k}−η`。本步若正发放同样仍≥k。
3. **C_{t−1}=k−1**：只有正发放能达到k。若K有限，合法k≤K保证k−1<K，有发放资格；膜条件为 `u_t>1`，即 `δ>p_{t,k}`。
4. **C_{t−1}≤k−2**：一次发放也无法达到k。

可将达到≥k的集合写成以下三个半直线集合的并：

\[
\{C_{t-1}\ge k+1\}
\ \cup\ \big(\{C_{t-1}\ge k\}\cap\{\delta\ge p_{t,k}-\eta\}\big)
\ \cup\ \big(\{C_{t-1}\ge k-1\}\cap\{\delta\ge p_{t,k}\}\big).
\]

第二、三项虽写成`≥`计数集合，不会增加错误状态：前一时刻≥k+1的状态已由第一项覆盖；前一时刻=k的状态若满足δ≥p，则必满足δ≥p−η（用到η≥0），因此第三项也没有把应撤回至k−1的状态纳入。由归纳假设，每个计数事件都是相位半直线；交集阈值是max，并集阈值是min，得到所述递推。初始计数确定，初始化显然成立，归纳完成。

由该集合刻画可知 `B_{t,k−1}≤B_{t,k}≤B_{t,k+1}`，这也是零负阈值化简所需的顺序性质。

## 推论一：ε=0时严格化简为bounded递推

令η=0。由 `B_{t−1,k−1}≤B_{t−1,k}`，有

\[
\max(B_{t-1,k-1},p)\le\max(B_{t-1,k},p).
\]

所以三项min中的中间项永远不会小于最后一项，可以消去：

\[
\boxed{\ B_{t,k}=\min\left(B_{t-1,k+1},
\max(B_{t-1,k-1},p_{t,k})\right).\ }
\]

这与 `bounded_phase._vjp` 的`boundary=minimum(bounds[2:],maximum(bounds[:-2],crossing))`完全一致，不是近似。其`crossing`就是p，初始`bounds`由实际c₀决定。

## 推论二：任意固定CDF的均值与局部VJP

令F为δ的固定连续CDF，F(−∞)=0、F(+∞)=1。整数非负计数的tail-sum公式给出

\[
M_t:=\mathbb E[C_t]=\sum_{k=1}^{K_t}[1-F(B_{t,k})],
\quad K_t=\begin{cases}K,&K<\infty\\c_0+t,&K=\infty,\end{cases}
\]
\[
\boxed{\ \mu_t=\mathbb E[y_t]=\theta(M_t-M_{t-1}),\qquad M_0=c_0.\ }
\]

均值公式只需要固定连续F；**连续CDF本身不保证存在连续密度**。写下面的ρ导数公式时，另假设F在活动边界处可微且F′=ρ连续（实际三角分布满足）。固定θ、ε、q₀、c₀，避开活动分支碰撞等非光滑输入，结论对I几乎处处成立。每个有限活动边界来自某个前缀r：

\[
B_{t,k}=a_{t,k}-\sum_{s=1}^r I_s/\theta,
\quad \frac{\partial B_{t,k}}{\partial I_s}
=-\frac1\theta 1\{s\le r\}.
\]

常数a可能含级数、q₀、c₀、η，但它们都是固定条件。设origin(t,k)=r，g_{T+1}=0，则

\[
\boxed{\ (D_I\mu^Tg)_s
=\sum_{t=s}^T(g_t-g_{t+1})
\sum_k\rho(B_{t,k})1\{\operatorname{origin}(t,k)\ge s\}.\ }
\]

证明：`Σ_t g_t μ_t = θ Σ_t(g_t−g_{t+1})M_t − θg₁c₀`。末项为常数；对`1−F(B)`求导，ρ前面的θ与边界的1/θ恰好相消，得到该式。这正是实现保留边界origin、按相邻adjoint差加权，再把贡献分配给相关输入前缀的原因。

不在ties点宣称唯一Jacobian，也不对有限精度硬前向的所有恰好相等情形作精确实数声明。T2闭式实现与一般horizon实现可有不同的tie约定；不改变上述几乎处处结果。

## 与实际分类代码逐项对应

当前 `triangular_density_horizon._partition_chunk` 使用物理初始膜边界A，而上面的B是相对q₀的归一化偏移。分类c₀=0、q₀=.5、K=∞，因此

\[
A_{t,k}=\theta(B_{t,k}+.5),\qquad
P_{t,k}=k\theta-\sum_{s\le t}I_s.
\]

把统一递推乘θ并平移，得到

\[
A_{t,k}=\min\big(A_{t-1,k+1},
\max(A_{t-1,k},P_{t,k}-\epsilon),
\max(A_{t-1,k-1},P_{t,k})\big).
\]

这与真实实现对应如下：

| 统一数学对象 | 分类实际实现 |
|---|---|
| 无上限、c₀=0时只需k=1,…,t | 每tick新增一个level，`levels=arange(1,t+2)`（代码tick为0起点） |
| P_{t,k}=kθ−ΣI | `positive=levels*theta-total` |
| P_{t,k}−ε，实际ε=.001 | `negative=positive-1e-3` |
| max(A_prev,k,P−ε) | `hold_or_retract=maximum(same,negative)` |
| max(A_prev,k−1,P) | `emit=maximum(lower,positive)` |
| 三项min | `candidate=minimum(hold_or_retract,emit)`，再与`higher`取min |
| M_t=Σ[1−F_A(A)] | `charge=theta*(1-triangular_density_cdf(boundaries/theta)).sum(0)` |
| μ_t=θ(M_t−M_prev) | `result[t]=charge-previous_charge` |
| ρ_δ(B)=ρ_A(A/θ) | `(1-abs(boundaries/theta-.5)).clamp_min(0)` |
| (g_t−g_{t+1})与前缀origin | `adjoint`、`origins`、`origins>=input_time`的求和 |

这里`triangular_density_cdf(z)`是模态.5、支撑[−.5,1.5]的归一化初始膜CDF；相对δ的CDF为 `F_δ(b)=F_A(b+.5)`，故分布只做坐标平移，没有偷偷改动ρ。

该定理也逐神经元适用于固定、可广播的θ；每个神经元取其自身标量阈值即可。它是条件于实际输入电流的局部结论，**没有**因此传播跨层相位联合分布，不能升级为全网随机loss的无偏梯度定理。

## 小型CPU验证：直接覆盖当前分类算子

脚本 `check_unified_count_boundary_cpu.py` 使用固定seed=2026100321、CPU float64、2线程。没有GPU、训练、数据集或公共代码修改。仅10个T4固定seed样例、少量初始状态，未做参数网格。

1. 对真实 `triangle_density_partition_mean/vjp`（无上限、ε=.001）与统一递推/精确相位cell积分比较：均值最大差`8.88e−16`，VJP对统一递推autograd最大差`1.33e−15`，VJP对cell积分有限差分最大差`6.30e−10`。
2. 有限cap、不同合法c₀的几个状态中，统一均值与精确cell积分一致；ε=0时三项递推与化简两项递推逐值完全一致。另含一个有限cap且ε=.001的状态，验证一般式不是只对两端特殊情形有效。
3. 原信用归还例子取分类实际ε=.001时，真实代码给

\[
\mu=(.405,-.3596995),\quad
J_\rho=\begin{pmatrix}.9&0\\-.599&.301\end{pmatrix},\quad
J_\rho^T(1,1)=(.301,.301).
\]

仍有负跨时间项和发放/撤回信用配对。取固定线性读出g=(.3,1)、步长η_step=.001：期望读出−.2381995，density更新后−.2383983795（下降），triangle更新后−.2381909515（上升）。因此 `foundations.md` 的正向局部下降机制也确实存在于**当前真实分类算子**，不是仅bounded/ε=0例子的特殊现象。

CPU日志及结构化结果：`unified_count_boundary_checks.log`、`unified_count_boundary_checks.json`，`cuda_initialized=false`。复现：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
 /home/songf/miniforge3/envs/sparse/bin/python \
 artifacts/tracking_gradient_math_20261119/check_unified_count_boundary_cpu.py
```

结论边界保持不变：证明的是辅助完整轨迹的**条件电荷均值Jacobian**，以及固定线性读出下的局部下降保证；不是任意非线性全网loss或正确率/HOTA提升定理。它补齐了数学对象与现有分类正收益实验实际算子之间的对应关系。
