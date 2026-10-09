## 论文思路：系统建模 → 化简成 ski-rental → 随机化策略 → minimax + 对偶求解 → 落成算法

### A.系统建模
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/048a15199cce4e399f90ed10a995472a.png)

卸载调度器（Unloading Scheduler）是核心，负责决定如何卸载，当端点发来请求推理引擎确认推理所需层是否已在内存中，若在则开始推理，若不在则给加载引擎发消息让其load缺失层，并且通知卸载调度器停止正在进行的卸载，推理结束后再向调度器发送空闲消息，由调度器继续调度如何卸载


一个具有 $|\mathcal{I}|$ 层的LLM，如果模型的第 $i$ 层保留在内存中，它占用 $c_i$ 的内存空间。当新请求到达时，当前推理所需的所有参数必须驻留在内存中。如果第 $i$ 层缺失（即已被卸载），则必须重新加载，产生记为 $d_i$ 的加载延迟。

**内存成本**定义为空闲期间内存占用与占用时长的乘积（*优化目标*）。为了降低内存成本，层应尽早卸载。相反，为了减少后续请求的加载延迟，层必须保留在内存中。
因此，内存成本可以被视为模型空闲期间的一种资源支出，用以换取降低加载延迟的收益。这种权衡需要仔细确定 $(t_1, \dots, t_i, \dots, t_{|\mathcal{I}|})$（*决策变量*），以**联合优化加载延迟和内存成本。**

各层在内存中的保留状态定义模型的**保留状态** $s_i$。当第1层到第 $i$ 层被保留、第 $i+1$ 层到第 $|\mathcal{I}|$ 层被卸载时，模型处于状态 $s_i$。

从状态 $s_i$ 转换到状态 $s_{i-1}$ 意味着第 $i$ 层被卸载。定义 $t_i$ 对应于确定从状态 $s_i$ 到状态 $s_{i-1}$ 的转换时间。

在状态 $s_i$ 下，模型占用内存为 $C_i = \sum_{i'\leq i} c_{i'}$，从状态 $s_i$ 开始服务请求会产生加载延迟 $D_i = \sum_{i'\geq i} d_{i'}$。$C$递增，$D$递减，就是两目标冲突的数学根源。

---

### B. 目标与约束

QoS约束规定加载延迟不得超过 $D_{QoS}$。处于状态 $s_0$ 到 $s_{\bar{i}-1}$ 会违反QoS约束，这意味着为了满足QoS要求，第1层到第 $\bar{i}$ 层**必须始终保留**在内存中。因此，我们有：

$$
\{t_i = +\infty \mid 1 \leq i \leq \bar{i},\ \bar{i} = \{\min_{i\in \mathcal{I}}\{i\} \mid D_i \leq D_{QoS}\}\}. \tag{1a}
$$


剩余的问题涉及确定第 $\bar{i}+1$ 层到第 $|\mathcal{I}|$ 层的卸载时间。 $S_i(t)$ 表示模型在时刻 $t$ 处于状态 $s_i$ 的概率，是一个桥梁变量，把决策$t_i$和目标连起来。

对于在 $\tau$ 时刻到达的下一个请求，在线联合优化问题如下：

$$
\begin{array}{rl}
\min & \sum_{i \in \mathcal{I} \cup \{0\}} \eta D_i S_i(\tau) + \int_0^\tau \sum_{i \in \mathcal{I} \cup \{0\}} C_i S_i(t) \, \mathrm{d}t, \\[1.2ex]
\mathrm{s.t.} & \sum_{i \in \mathcal{I} \cup \{0\}} S_i(t) = 1, \quad t \geq 0, \\[0.6ex]
& S_i(t) = 0, \quad 0 \leq i \leq \bar{i}-1,
\end{array} \tag{2a,2b,2c}
$$

其中 $\eta$ 是分配给加载延迟的权重。优化目标是在满足加载延迟最大允许值的QoS约束的前提下，最小化以下两项的加权和：（1）服务下一个请求时产生的加载延迟；（2）模型空闲期间累积的内存成本。

求解该问题的主要挑战在于未来请求到达时间 $\tau$ 的不确定性，在做出决策时 $\tau$ 是未知的。

---

### C. 化简成 ski-rental（2-斜率滑雪租赁）

**动机**：（2a）里 $\tau$ 未知，是标准的"在线决策 vs 不确定时长"结构，正好对应经典的**滑雪租赁问题**——滑雪时长未知，每天租（类比*内存成本*，持续付费）还是一次性买（类比*加载延迟*，一次性付费）。多种保留状态 $s_i$ ⇒ **多斜率**滑雪租赁。

**关键假设（把多斜率压成 2-斜率）**：加载延迟 ≈ 正比于参数量/内存占用，即

$$
d_i = k\,c_i.
$$

> 依据：加载延迟主要由参数量和磁盘→GPU 带宽（PCIe）决定，表III 实测各类层 $d_i/c_i$ 基本恒定（≈0.32 ms/MB），假设成立。

**为什么能压成 2-斜率**：若一直停在状态 $s_{i_1}$、$s_{i_2}$，到时刻 $t$ 的总成本分别是"内存斜率×t + 一次性延迟"两条直线：

$$
\sum_{i'\leq i_1} c_{i'}\, t + \sum_{i' > i_1} \eta d_{i'}
\qquad\text{和}\qquad
\sum_{i'\leq i_2} c_{i'}\, t + \sum_{i' > i_2} \eta d_{i'}.
$$

令两条线相等求交点：

$$
\sum_{i'\leq i_1} c_{i'}\, t + \sum_{i' > i_1} \eta d_{i'} = \sum_{i'\leq i_2} c_{i'}\, t + \sum_{i' > i_2} \eta d_{i'}. \tag{3a}
$$

代入 $d_i = k c_i$ 化简得 $t = \eta k$——**与选哪一对状态无关**，所有直线交于同一点 $B$：

$$
B = \frac{\eta D_{\bar{i}}}{\,C_{|\mathcal{I}|} - C_{\bar{i}}\,}.
$$

**推论（只剩两个候选状态）**：谁的直线在 $\tau$ 处最低就选谁，

$$
s_{\bar{i}} = \arg\min_{s_i} \{C_i \tau + \eta D_i\}\ \ (\tau > B),
\qquad
s_{|\mathcal{I}|} = \arg\min_{s_i} \{C_i \tau + \eta D_i\}\ \ (\tau \leq B).
$$

即 $\tau>B$（请求来得晚）→ 收缩到最省内存的合法状态 $s_{\bar i}$；$\tau\le B$（来得早）→ 全保留 $s_{|\mathcal{I}|}$。**中间状态永远不会最优**，所以第 $\bar{i}+1$ 到 $|\mathcal{I}|$ 层必然在**同一时刻 $\bar{t}$ 一起卸载**：

$$
\bar{t} = t_{\bar{i}+1} = \dots = t_{|\mathcal{I}|}.
$$

**映射**：状态 $s_{\bar{i}}$ ↔ **买**（一次性花 $B$），状态 $s_{|\mathcal{I}|}$ ↔ **租**（单位租金 1）。原问题 → 求唯一决策变量 $\bar{t}$ 的 2-斜率滑雪租赁问题。

---

### D. 随机化策略 $P(t)$ 与统计信息

**思路转变**：确定性策略在最坏情况下会被对手卡（competitive ratio 上限 $e/(e-1)$），故改用**随机化策略**：把决策 $\bar{t}$ 变成一个概率密度 $P(t)$（在时刻 $t$ 才卸载的概率），$P(t)$ 只在 $0\le t\le B$ 上有定义：

$$
P(t) = p(t) + \alpha\,\delta(t) + \beta\,\delta(t - B), \tag{4a}
$$

- $p(t)$：$0<t<B$ 上的连续密度；
- $\alpha$：$t=0$ 立即卸载（=买/全收缩）的概率质量；
- $\beta$：$t=B$ 才卸载的概率质量；用狄拉克 $\delta$ 表示两个离散点。
- **新的决策变量变成 $p(t),\alpha,\beta$。**

**引入到达间隔统计量**（比只知道 $B$ 更强）：长间隔概率 $q_{B^+}$、短间隔一阶矩 $\mu_{B^-}$、短间隔二阶矩 $\sigma_{B^-}$：

$$
\int_0^B q(\tau)\,\mathrm{d}\tau + q_{B^+} = 1, \tag{5a}
$$
$$
\int_0^B \tau\, q(\tau)\,\mathrm{d}\tau = \mu_{B^-}, \tag{5b}
$$
$$
\int_0^B \tau^2 q(\tau)\,\mathrm{d}\tau = \sigma_{B^-}, \tag{5c}
$$

其中 $q(\tau)$ 是下一个请求到达时间的密度。

**期望成本**（把 $P(t)$ 和到达时间 $\tau$ 组合起来）：
$\tau \le B$ 时

$$
C(P(t),\, \tau \le B) = \int_0^\tau (t + B)P(t)\,\mathrm{d}t + \int_\tau^B \tau\, P(t)\,\mathrm{d}t, \tag{6a}
$$

$\tau > B$ 时

$$
C(P(t),\, \tau > B) = \int_0^B (t + B)P(t)\,\mathrm{d}t. \tag{7a}
$$

对到达分布 $q(\tau)$ 求期望，得**期望在线成本**：

$$
J(P(t), q(\tau)) = \int_0^B C(P(t),\, \tau \le B)\, q(\tau)\,\mathrm{d}\tau + C(P(t),\, \tau > B)\, q_{B^+}. \tag{8a}
$$

---

### E. minimax + 拉格朗日对偶

**博弈设定**：对手挑最坏的 $q(\tau)$ 把成本抬到最高，我们挑 $P(t)$ 把最坏成本压到最低 → **极小极大**：

$$
\begin{array}{rl}
\min\limits_{P(t)} \max\limits_{q(\tau)} & J(P(t), q(\tau)), \\[0.8ex]
\mathrm{s.t.} & (5a),\ (5b),\ (5c), \\[0.4ex]
& \int_0^B p(t)\,\mathrm{d}t = 1 - \alpha - \beta,
\end{array} \tag{9b}
$$

最后一条约束保证 $P(t)$ 是合法概率密度（总质量为 1）。

**对内层 max 做对偶**：给 (5a)(5b)(5c) 配拉格朗日乘子 $\lambda_1,\lambda_2,\lambda_3$，并代入 (4a)：

$$
\begin{array}{l}
L(q(\tau), \lambda_1, \lambda_2, \lambda_3) = q_{B^+}\!\int_0^B (t + B) p(t)\,\mathrm{d}t + \alpha q_{B^+} B \\[0.6ex]
\quad + \int_0^B \big[C(P(t), \tau \le B) - \lambda_1 - \lambda_2 \tau - \lambda_3 \tau^2\big] q(\tau)\,\mathrm{d}\tau \\[0.6ex]
\quad + 2\beta q_{B^+} B + \lambda_1 (1 - q_{B^+}) + \lambda_2 \mu_{B^-} + \lambda_3 \sigma_{B^-},
\end{array} \tag{10a}
$$

对偶函数 $g=\sup_{q(\tau)} L$ 要有界，方括号内必须恒为 0：

$$
C(P(t), \tau \le B) - \lambda_1 - \lambda_2 \tau - \lambda_3 \tau^2 = 0.
$$

于是得到**对偶问题**：

$$
\begin{array}{rl}
\min\limits_{\lambda_1, \lambda_2, \lambda_3} & \Big[ q_{B^+}\!\int_0^B (t + B) p(t)\,\mathrm{d}t + \alpha q_{B^+} B + 2\beta q_{B^+} B + \lambda_1 (1 - q_{B^+}) + \lambda_2 \mu_{B^-} + \lambda_3 \sigma_{B^-} \Big], \\[0.8ex]
\mathrm{s.t.} & C(P(t), \tau \le B) - \lambda_1 - \lambda_2 \tau - \lambda_3 \tau^2 = 0, \\[0.4ex]
& 0 \le \lambda_1,\quad 0 \le \lambda_2,\quad 0 \le \lambda_3.
\end{array} \tag{11b}
$$

**解出 $p(t)$**：把约束式对 $\tau$ 求二阶导（再把 $\tau$ 换成 $t$），得到一阶常微分方程：

$$
\frac{\mathrm{d}p(t)}{\mathrm{d}t} = \frac{1}{B}\big(p(t) + 2\lambda_3\big). \tag{12a}
$$

以 (9b) 为边界条件，解为指数形式：

$$
p(t) = c_0\, e^{t/B} - 2\lambda_3, \tag{13a}
$$
$$
c_0 = \frac{1 - \alpha - \beta + 2\lambda_3 B}{B(e - 1)}. \tag{13b}
$$

**非负性约束**（$p(t)\ge 0$）给出 $\lambda_3$ 上界：

$$
\lambda_3 \le \frac{1 - \alpha - \beta}{2B(e - 2)}. \tag{14a}
$$

**把乘子回代**，$\lambda_1,\lambda_2$ 变成 $\alpha,\beta,\lambda_3$ 的函数：

$$
\lambda_1 = \alpha B, \tag{15a}
$$
$$
\lambda_2 = \beta + \frac{e}{e-1}(1 - \alpha - \beta) + \frac{(4 - 2e)B}{e-1}\,\lambda_3. \tag{15b}
$$

---

### F. 线性规划求 $\alpha,\beta$ → 落成策略

把 (15b) 代回 (11b)、并把 (14a) 作为附加约束，对偶问题等价成关于 $\lambda_2,\lambda_3$ 的**线性规划**：

$$
\begin{array}{rl}
\min\limits_{\lambda_2, \lambda_3} & \dfrac{(1 - \alpha - \beta)e\, q_{B^+} + B}{e - 1} + \alpha B + 2\beta q_{B^+} + B \\[0.6ex]
& + \lambda_2 \mu_{B^-} + \lambda_3 \Big( \dfrac{(3 - e)q_{B^+} + B^2}{e - 1} + \sigma_{B^-} \Big), \\[0.8ex]
\mathrm{s.t.} & \lambda_2 = \beta + \dfrac{e}{e-1}(1 - \alpha - \beta) + \dfrac{(4 - 2e)B}{e-1}\,\lambda_3, \\[0.4ex]
& 0 \le \lambda_2,\quad 0 \le \lambda_3 \le \dfrac{1 - \alpha - \beta}{2B(e - 2)}.
\end{array} \tag{16c}
$$

LP 的最优必在 $\lambda_3$ 可行区间的**两个端点顶点**之一取得：

$$
\lambda_2 = \beta + \frac{e}{e-1}(1 - \alpha - \beta),\quad \lambda_3 = 0, \tag{17a}
$$
$$
\lambda_2 = 1 - \alpha,\quad \lambda_3 = \frac{1 - \alpha - \beta}{2B(e - 2)}. \tag{17b}
$$

**再对 $\alpha,\beta$ 求解**：分别把 (17a)、(17b) 代回目标，得两个关于 $\alpha,\beta$ 的 LP（区别在于 $\lambda_3$ 不同 ⇒ $p(t)$ 形状不同）：

代入 (17a)：

$$
\begin{array}{rl}
\min\limits_{\alpha, \beta} & \Big[ -\dfrac{e}{e-1}(\mu_{B^-} + q_{B^+} + B) + B \Big] \alpha \\[0.6ex]
& + \Big[ -\dfrac{e}{e-1}(\mu_{B^-} + q_{B^+} + B) + \mu_{B^-} + 2q_{B^+} + B \Big] \beta \\[0.6ex]
& + \dfrac{e}{e-1}(\mu_{B^-} + q_{B^+} + B), \\[0.8ex]
\mathrm{s.t.} & \alpha + \beta \le 1,\quad 0 \le \alpha,\quad 0 \le \beta.
\end{array} \tag{18a}
$$

代入 (17b)：

$$
\begin{array}{rl}
\min\limits_{\alpha, \beta} & \Big[ -\dfrac{(2e - 3)q_{B^+} + B}{2(e - 2)} - \dfrac{\sigma_{B^-}}{2B(e - 2)} - \mu_{B^-} + B \Big] \alpha \\[0.6ex]
& + \Big[ -\dfrac{(2e - 3)q_{B^+} + B}{2(e - 2)} - \dfrac{\sigma_{B^-}}{2B(e - 2)} + 2q_{B^+} + B \Big] \beta \\[0.6ex]
& + \dfrac{(2e - 3)q_{B^+} + B}{2(e - 2)} + \dfrac{\sigma_{B^-}}{2B(e - 2)} + \mu_{B^-}, \\[0.8ex]
\mathrm{s.t.} & \alpha + \beta \le 1,\quad 0 \le \alpha,\quad 0 \le \beta.
\end{array} \tag{19a}
$$

**枚举顶点求目标成本**（LP 最优在可行三角形顶点取得）：

对 (18a)：

$$
\alpha=1,\beta=0:\ C_{1,0} = B, \tag{20a}
$$
$$
\alpha=0,\beta=1:\ C_{0,1} = \mu_{B^-} + 2q_{B^+} + B, \tag{20b}
$$
$$
\alpha=0,\beta=0:\ C_{0,0} = \frac{e(\mu_{B^-} + q_{B^+} + B)}{e - 1}. \tag{20c}
$$

对 (19a)：

$$
\alpha=1,\beta=0:\ C'_{1,0} = B, \tag{21a}
$$
$$
\alpha=0,\beta=1:\ C'_{0,1} = \mu_{B^-} + 2q_{B^+} + B, \tag{21b}
$$
$$
\alpha=0,\beta=0:\ C'_{0,0} = \frac{(2e - 3)q_{B^+} + B^2 + \sigma_{B^-}}{2(e - 2)} + \mu_{B^-}. \tag{21c}
$$

> 注：$C_{1,0}=C'_{1,0}$、$C_{0,1}=C'_{0,1}$ 完全相同，两组只有 $(0,0)$ 顶点不同（$C_{0,0}$ vs $C'_{0,0}$），故只需比较 4 个候选：$C_{1,0},\,C_{0,1},\,C_{0,0},\,C'_{0,0}$。

**最终策略**：谁的目标成本最小就用谁对应的 $P(t)$：

$$
P(t) =
\begin{cases}
\delta(t), & C_{1,0} < \min\{C_{0,1}, C_{0,0}, C'_{0,0}\}, \\[0.6ex]
\delta(t - B), & C_{0,1} < \min\{C_{1,0}, C_{0,0}, C'_{0,0}\}, \\[0.6ex]
\dfrac{1}{B(e - 2)}\big(e^{t/B} - 1\big), & C'_{0,0} < \min\{C_{1,0}, C_{0,1}, C_{0,0}\}, \\[0.6ex]
\dfrac{1}{B(e - 1)}\,e^{t/B}, & \text{其他}.
\end{cases} \tag{22a}
$$

后两种连续密度积分得 CDF（用于采样 $\bar t$）：

$$
\hat{P}(t) =
\begin{cases}
\dfrac{e^{t/B} - t/B - 1}{e - 2}, & 0 \le t < B, \\[0.8ex]
1, & B \le t,
\end{cases} \tag{23a}
$$
$$
\hat{P}(t) =
\begin{cases}
\dfrac{e^{t/B} - 1}{e - 1}, & 0 \le t < B, \\[0.8ex]
1, & B \le t.
\end{cases} \tag{23b}
$$

（23a）比（23b）**更倾向多保留**层（更晚卸载）。

**取 $\bar t$**：$\delta(t)$→$\bar t=0$；$\delta(t-B)$→$\bar t=B$；后两种从 $\mathcal{U}[0,1]$ 采一个 $\xi$，取满足 $\xi\le \hat P(t)$ 的最小 $t$（逆变换采样）。

---

### G. 落成算法 LUA（层粒度卸载）

**运行逻辑**：等空闲事件 → 更新统计量 → 按 QoS 定 $\bar i$、算 $B$ → `ComputeStrategy` 出 $\bar t$ → 到点卸载 $\bar i+1\!\sim\!|\mathcal I|$ 层；若中途来请求（繁忙事件）立即中止卸载。动态负载下用**滑动窗口**维护最近到达间隔，让 $q_{B^+},\mu_{B^-},\sigma_{B^-}$ 跟得上负载变化。

```text
算法 1  层粒度卸载算法 (LUA)
输入： c_i, d_i, η, 𝓘, D_QoS

1:  for i ∈ 𝓘 ∪ {0}:
2:      C_i ← Σ_{i'≤i} c_{i'} ,  D_i ← Σ_{i'≥i} d_{i'}

3:  while 未收到空闲事件 do
4:      等待空闲事件
5:  更新 q_{B+}, μ_{B-}, σ_{B-} ；t_idle ← 当前时间
6:  ī ← { min_{i∈𝓘}{i} | D_i ≤ D_QoS } ;  B ← η·D_ī / (C_{|𝓘|} − C_ī)
7:  t̄ ← ComputeStrategy(B, q_{B+}, μ_{B-}, σ_{B-})
8:  { t_i ← +∞ | 1 ≤ i ≤ ī } ,  { t_i ← t̄ | ī+1 ≤ i ≤ |𝓘| }
9:  while 未收到繁忙事件 do
10:     if 当前时间 − t_idle ≥ t̄ then
11:         向卸载引擎发卸载事件，卸载第 (ī+1) ~ |𝓘| 层
12: 跳转到第 3 行

13: function ComputeStrategy(B, q_{B+}, μ_{B-}, σ_{B-})
14:     从 𝒰[0,1] 采样 ξ
15:     按 (20a)(20b)(20c)(21c) 算 C_{1,0}, C_{0,1}, C_{0,0}, C'_{0,0}
16:     if  C_{1,0} < min{C_{0,1}, C_{0,0}, C'_{0,0}} then
17:         t̄ ← 0
18:     else if C_{0,1} < min{C_{1,0}, C_{0,0}, C'_{0,0}} then
19:         t̄ ← B
20:     else if C'_{0,0} < min{C_{1,0}, C_{0,1}, C_{0,0}} then
21:         按 (23a) 算 P̂(t)
22:         t̄ ← min{ t | ξ ≤ P̂(t) }
23:     else
24:         按 (23b) 算 P̂(t)
25:         t̄ ← min{ t | ξ ≤ P̂(t) }
26:     return t̄
```

> 复现要点：第 2 行 $D_i=\sum_{i'\ge i}d_{i'}$ 与状态定义一致即可（关键是 $C$ 递增、$D$ 递减）；第 15 行只需 4 个候选成本；第 22/25 行是对 (23a)/(23b) 的逆变换采样。

---

### H. 时间复杂度

- 第 6 行按 QoS 选 $\bar i$：$O(|\mathcal{I}|)$；更新卸载定时器：$O(|\mathcal{I}|)$。
- `ComputeStrategy`（第 13–25 行）：采样 $O(1)$ + 算 4 个成本 $O(1)$ + 定 $\bar t$ $O(1)$ = $O(1)$。

**整体最坏复杂度 $O(|\mathcal{I}|)$**（与层数线性）。在 NVIDIA Jetson AGX Orin 上实测，单次决策开销约 **1 ms**，开销极低。
