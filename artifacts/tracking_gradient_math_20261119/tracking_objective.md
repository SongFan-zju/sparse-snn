# 用DiMP跟踪目标解释共享相位梯度的有效机制

**主角度：全轨迹条件均值的Jacobian能保留跨时间步的任务信用，尤其减少短T下晚时间步受到的额外衰减。备选角度：检验这种信用保持是否转为模板—搜索之间的目标响应margin改善。** 已有分类/DVS收益是经验出发点；本报告给出能迁移到跟踪的明确数学对象和一个真实DiMP算子的CPU机制例，尚未验证跟踪准确率提升。没有新增GPU、训练或共享环境改动。

## 1. 先把真实跟踪梯度写清

DiMP18不是“两个身份embedding做CE”。令共同骨干为f_θ，支持/模板帧特征Z=f_θ(z)，搜索帧特征X=f_θ(x)。分类支路使用layer3，先经过原生分类特征模块，以下简记仍为Z、X。作者源码执行

\[
w_0=\operatorname{Init}(Z,B_z),\quad
w_{k+1}=U_k(w_k,Z,B_z),\quad
r_k=\operatorname{Conv}(X,w_k).
\]

U_k是解析的DiMP steepest-descent/Gauss–Newton更新，步长也依赖当前特征和滤波器；训练要反传整个有限K步，不把它替换成已经求到最优的ridge回归。令V_k=∂w_k/∂Z，则

\[
V_0=D_Z\operatorname{Init},\qquad
V_{k+1}=(D_wU_k)V_k+D_ZU_k.
\]

**V_0、更新方向与步长的特征依赖都属于模板信用。** 不能只对最终搜索响应反传而detach模板。另一方面，部署时可缓存模板/支持特征，后续在线更新复用已经提取的搜索特征；每个新搜索帧只执行一次骨干，滤波与框细化不要求再跑完整视觉网络。

IoU支路接layer2+layer3：原生`AtomIoUNet.forward()`只取**第一个支持帧**计算调制向量m=A(Z_0,B_0)，然后以该调制和搜索特征预测proposal的IoU：

\[
\hat{u}=Q(X^{(2)},X^{(3)},m,b),\quad
\mathcal L=\sum_{k=0}^{K}\lambda_k\ell_{\rm LBHinge}(r_k,y)
 +\lambda_{\rm IoU}\|\hat u-u_{GT}\|_{\rm mean}^2.
\]

因此第2、3个支持帧没有该IoU模板项是作者设计，不是梯度被意外切断。官方R18设置的最终/初始分类权重各100，中间迭代合计400，IoU权重1；权重数字不能代替实际梯度范数。

若a_k=λ_k∇_{r_k}ℓ、b=λ_IoU∇_{û}MSE，则简化写成

\[
g_Z=\sum_k V_k^T(D_w\operatorname{Conv})^Ta_k
 +(D_ZA)^T(D_mQ)^Tb,
\]
\[
g_X=\sum_k(D_X\operatorname{Conv})^Ta_k+(D_XQ)^Tb,
\qquad
G_\theta=(D_\theta f(z))^Tg_Z+(D_\theta f(x))^Tg_X.
\]

实际接入还须包含分类特征模块、不同支持帧和layer2/3的链式导数。我们SG接在骨干每个独立激活调用对应的signed IF上；U_k、相关/滤波、PrROI与IoU头先保留原生连续运算。R18 `BasicBlock`复用了同一`relu`模块两次，转换时应拆独立神经元状态。训练使用`DiMPActor`/模型训练forward，不能拿带`no_grad()`的部署tracker入口直接求训练梯度。

## 2. 主角度：为什么“整条轨迹”能提供更合理的时间信用

单个神经元在T步内只抽象为一个共享初始相位φ；不同图像和不同神经元不共用这个状态。其硬前向仍为固定φ₀=1/2：

\[
v_t=m_{t-1}+I_t,\quad
e_t=\mathbf1[v_t>\vartheta]
 -\mathbf1[C_{t-1}\ge1]\mathbf1[v_t\le-\epsilon],
\]
\[
q_t=\vartheta e_t,\quad m_t=v_t-q_t,\quad C_t=C_{t-1}+e_t.
\]

固定原有ρ(φ)=max(1−|φ−1/2|,0)，支持[−1/2,3/2]，定义条件均值μ(I)=E_φ[q(I,φ)]。这是用于反向的假想平滑分布，不要求部署随机化。当前算法真实使用

\[
\boxed{G_{\rm density}=J_\mu(I)^Tg_{\rm hard}},\qquad
g_{\rm hard}=\left.\frac{\partial\mathcal L}{\partial q}\right|_{\text{共同硬前向}}.
\]

关键不是“让阶跃有非零梯度”，而是让一个初始相位引起的**发放、复位和有资格撤回的全轨迹变化**共同决定J_μ。普通三角也沿reset递推；区别在于它把各时刻局部阈值斜率连乘，这个乘积通常不等于共享初态下整条轨迹的均值Jacobian。

一个直接可解释的特例是头部消费时间平均电荷f=(1/T)Σ_tq_t。令普通三角的输出斜率β_t=∂q_t/∂v_t，资格门保持硬值，则其BPTT给出

\[
\frac{\widetilde\partial f}{\partial I_s}
=\frac1T\sum_{t=s}^{T}\beta_t\prod_{u=s}^{t-1}(1-\beta_u)
=\frac1T\left[1-\prod_{u=s}^{T}(1-\beta_u)\right].
\]

在0≤β_t≤1的区域，这使较早电流获得更多后续机会，晚步信用容易缩小。它不是所有输入都出错，也不是未考虑reset；它是**局部斜率乘积所引入的特定时间权重**。共享相位J_μ直接用完整轨迹积分决定这些权重；在最终均值电荷局部正好满足μ_f=ΣI_t/T的区域，各步真实条件均值敏感度均为1/T，晚步不应再额外衰减。

### 可以严格主张的局部命题

固定同一次硬跟踪forward算出的g_hard，并在这个局部更新中stop-gradient它。定义

\[
\Psi_{\rm lin}(I;I_0)
=\langle g_{\rm hard}(I_0),\mu(I)\rangle.
\]

在当前实现的条件均值定义及可微区域内，density恰好是∇_IΨ_lin；tri一般是近似。给定相同电流步长范数η，令d=−G/‖G‖，则

\[
D\Psi_{\rm lin}[d_{\rm density}]=-\|G_{\rm density}\|,
\qquad
D\Psi_{\rm lin}[d_{\rm tri}]
=-\|G_{\rm density}\|\cos(G_{\rm density},G_{\rm tri}).
\]

这解释了有效性的一个具体来源：**在同一个真实任务adjoint下，减少时间信用方向失真，而不仅是把梯度整体放大。** 当跟踪头在该局部足够接近线性时，这个任务切平面具有直接意义。命题条件化于给定电流和adjoint；多层SNN中的下游adjoint也可能来自SG，不能据此推出整个网络对应某个精确平滑全局loss。

### 已完成：真实DiMP算子的极小CPU例

`check_credit_cpu.py`使用作者原生2步`DiMPSteepestDescentGN`＋`apply_filter`＋`LBHinge`，分别把两个神经元解码特征放入模板支或搜索支。固定seed=2026100321、T4；两条电流预先固定为

- (0.18,0.42,−0.15,0.37)，硬事件(0,+1,−1,+1)；
- (0.70,−0.55,0.34,0.12)，硬事件(+1,−1,0,+1)。

阈值1，硬解码特征均0.25，独立phase-cell积分均值分别0.205、0.1525。**参考adjoint确实来自共同hard特征，而不是mean特征。** 独立参考通过移动相位cell的CDF权重求导，不使用hard-loss几乎处处为零的差分。等范数电流步长固定0.001，没有扫描输入、步长或种子。

| native DiMP头的局部支路 | tri对∇Ψ_lin相对误差 | tri方向cos | density方向cos/误差 | 相同步长下−ΔΨ_lin，tri / density |
|---|---:|---:|---:|---:|
| 模板支 | 16.011% | 0.989136 | 1.000000 / 0 | 3.36654e−6 / 3.40352e−6 |
| 搜索支 | 15.378% | 0.989922 | 1.000000 / 0 | 8.43392e−6 / 8.51978e−6 |

两者都给下降方向；这个例子中density的等范数下降约多1.10%/1.02%，只支持方向更忠于该局部对象，不声称收益很大。第一条神经元第4tick的tri信用只有density的68%，第1tick约99.95%；这正是上面局部斜率乘积公式的可见结果。

该检查使用官方实现的算子及默认初始化参数，**不是加载官方checkpoint后的完整跟踪头**。它只用合成的2个特征位置和4通道5×5特征图、固定初始滤波器；**未包含分类特征卷积、真实初始化器PrROI、完整CNN、IoU头、数据集或参数更新**。CUDA前后均未初始化。完整结果在`credit_cpu.json`。`F(mean)`另外记录为secondary：它的adjoint通常不同；没有把它的梯度当作当前算法。

## 3. 备选角度：关系敏感的响应margin信用

DiMP的空间响应对模板滤波器与搜索特征有乘性关系。对固定k，r=Xw(Z)，扰动满足

\[
\delta r=X\,\delta w+\delta X\,w+\delta X\,\delta w,
\qquad\delta w\simeq V_k\delta Z.
\]

因此同样大小的特征误差未必同样重要：目标和最强干扰位置之间的差、模板如何改变滤波器、以及搜索特征如何与之匹配，才决定局部定位。不要仅用单支特征MSE解释有效性。

令p为目标区域的最大响应位置，j为区域外最强干扰位置，在二者唯一且暂不切换时，M=r_p−r_j，则

\[
dM=(X_p-X_j)V_k\,dZ+(dX_p-dX_j)w.
\]

这给出第二个可检验正向机制：如果density保留了有助于目标—干扰分离的时间信用，其在**固定真实margin adjoint**下的条件均值VJP也应更接近phase-cell参考；而同一个共享骨干更新的模板、搜索两项应该共同计入。可以先保留原loss，用margin作解释量，不为了分析另加一套训练loss或参数。

IoU支路同理产生另一组空间/通道信用。原生总信用是加权的g_cls+g_IoU；将这两项分别传播只用于检查各自贡献及夹角，不是首轮新增消融训练。需要核验的是合成信用是否仍提高目标margin及框质量；不能仅因某个loss下降便把全部改善归给我们的SG。

这个备选不是说原LBHinge无效。一个简单响应空间例说明为什么还应观察margin：标签(1,0,0)，响应从(0.8,0.79,0.7)变为(0.79,0.791,0.6)，原LBHinge从0.38470下降至0.34326，但margin从+0.010变为−0.001。它只是一个可能的响应扰动，**不是density造成错误的实测案例**。同样，IoU值拟合改善与最终框细化改善也应分别观察。

## 4. 下一步最小验证与清楚的边界

优先只验证主角度：已有DiMP18官方checkpoint与真实CPU骨干加载证据保留；未来建立一个共享L4来源、T4硬前向后，用固定少量template/search对，在最终骨干激活位置取得原生跟踪adjoint。对几个预先固定神经元做同样phase-cell参考，比较tri/density的时间信用、cos和等范数Ψ_lin局部下降；不做梯度形状网格。PrROI的原生GPU前后向一旦可运行，再原样纳入IoU支路；本轮CPU没有以替换ROI或stub冒充该证据。

**预测1（主）：** 在均值累计电荷对晚步电流仍敏感，而局部三角β_T较小时，density更能保留晚步信用；若实际样本不存在这种区域，不能把收益归于此机制。

**预测2（主到任务）：** 当真实跟踪头的hard/mean特征adjoint接近，减少时间Jacobian失真应更容易转为定位loss/响应margin的有利变化。若Jacobian参考更准但真实任务变化不改善，应说明这个局部机制不足以预测该任务收益，不能继续声称已完成迁移解释。

**预测3（备）：** 相比单支特征误差，模板/搜索组合后的margin变化应更能解释哪些配对受益；检查两支贡献，不能预设模板一定比搜索曲率更强。

边界只需保留一条精确区分。令F为真实下游跟踪头，g₀=∇F(q_hard)，g_μ=∇F(μ)。实际算法是J_μᵀg₀，**不是**∇F(μ)=J_μᵀg_μ，也不是∇E[F(q_φ)]。与mean-feature目标的误差可以分解为

\[
G_{\rm density}-\nabla F(\mu)=J_\mu^T(g_0-g_\mu),
\]
\[
G_{\rm tri}-\nabla F(\mu)
=(J_{\rm tri}-J_\mu)^Tg_0+J_\mu^T(g_0-g_\mu).
\]

density去掉的是第一类“时间Jacobian近似”来源，不保证剩余项小，也不保证tri两项不会偶然抵消。另有`task_jump_cpu.json`展示条件期望任务loss与线性读出确实可不同，作为边界证据留存，不用它倒推既有负结果的原因。PrDiMP的位置概率与这里的初始相位分布是不同变量，名字相同不构成适配依据。

最后仍须用相同source、L/T、seed、图像访问及epoch的两臂真实SOT指标判断迁移：完整轨迹success/AO、precision、SR50/75；这些数学小例和局部margin都不替代跟踪准确率，不横比MOT HOTA，也不需要双完整推理融合。

## 源码与复现

所有数学接口优先读取本地锁定官方源码`artifacts/tracking_framework_candidates_20261119/upstream/pytracking-7eb9e74bd3d40e29dbcec444902237da13de247b/`：`ltr/models/tracking/dimpnet.py`、`target_classifier/linear_filter.py`和`optimizer.py`、`ltr/actors/tracking.py`、`ltr/models/loss/target_classification.py`、`ltr/models/bbreg/atom_iou_net.py`、`pytracking/tracker/dimp/dimp.py`。对应[官方固定仓库](https://github.com/visionml/pytracking/tree/7eb9e74bd3d40e29dbcec444902237da13de247b)、[原论文](https://arxiv.org/abs/1904.07220)。SG依据`src/conversion/ecl_triangular_surrogate.py`、`general_phase_adjoint.py`、`triangular_density_horizon.py`。本轮未重新联网/下载。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/songf/miniforge3/envs/sparse/bin/python artifacts/tracking_gradient_math_20261119/check_credit_cpu.py
```
